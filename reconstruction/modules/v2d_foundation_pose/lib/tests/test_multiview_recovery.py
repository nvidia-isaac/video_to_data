import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


LIB_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_POSE_LAST = object()


def _set_module(monkeypatch, name: str, module: types.ModuleType):
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_package(monkeypatch, name: str):
    module = types.ModuleType(name)
    module.__path__ = []
    return _set_module(monkeypatch, name, module)


def _install_multiview_tracker_stubs(monkeypatch):
    _install_package(monkeypatch, "v2d")
    _install_package(monkeypatch, "v2d.mesh")
    _install_package(monkeypatch, "v2d.mesh.lib")
    mesh_mod = types.ModuleType("v2d.mesh.lib.mesh")
    mesh_mod.Mesh = object
    _set_module(monkeypatch, "v2d.mesh.lib.mesh", mesh_mod)

    _install_package(monkeypatch, "v2d.mv")
    _install_package(monkeypatch, "v2d.mv.math")
    numpy_fn_mod = types.ModuleType("v2d.mv.math.numpy_fn")
    numpy_fn_mod.xyz_to_uv = lambda *args, **kwargs: None
    numpy_fn_mod.se3_from_rot_trans = lambda rot, trans: np.block([
        [rot, np.asarray(trans).reshape(3, 1)],
        [np.zeros((1, 3)), np.ones((1, 1))],
    ])
    _set_module(monkeypatch, "v2d.mv.math.numpy_fn", numpy_fn_mod)

    scipy_mod = _install_package(monkeypatch, "scipy")
    spatial_mod = _install_package(monkeypatch, "scipy.spatial")
    transform_mod = types.ModuleType("scipy.spatial.transform")

    class DummyRotation:
        @staticmethod
        def from_matrix(matrix):
            return DummyRotation()

        def mean(self, weights=None):
            return self

        def as_matrix(self):
            return np.eye(3)

    transform_mod.Rotation = DummyRotation
    _set_module(monkeypatch, "scipy.spatial.transform", transform_mod)
    scipy_mod.spatial = spatial_mod
    spatial_mod.transform = transform_mod

    estimater_mod = types.ModuleType("estimater")
    estimater_mod.FoundationPose = object
    _set_module(monkeypatch, "estimater", estimater_mod)

    _install_package(monkeypatch, "learning")
    _install_package(monkeypatch, "learning.training")
    score_mod = types.ModuleType("learning.training.predict_score")
    score_mod.ScorePredictor = object
    _set_module(monkeypatch, "learning.training.predict_score", score_mod)
    refine_mod = types.ModuleType("learning.training.predict_pose_refine")
    refine_mod.PoseRefinePredictor = object
    _set_module(monkeypatch, "learning.training.predict_pose_refine", refine_mod)

    utils_mod = types.ModuleType("Utils")
    utils_mod.set_seed = lambda seed: None
    _set_module(monkeypatch, "Utils", utils_mod)

    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(empty_cache=lambda: None)
    torch_mod.float = "float"
    torch_mod.as_tensor = lambda value, **kwargs: np.asarray(value)
    _set_module(monkeypatch, "torch", torch_mod)

    trimesh_mod = types.ModuleType("trimesh")
    trimesh_mod.bounds = types.SimpleNamespace(
        oriented_bounds=lambda mesh: (np.eye(4), np.ones(3)),
    )
    _set_module(monkeypatch, "trimesh", trimesh_mod)

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.convexHull = lambda pts: pts
    cv2_mod.fillConvexPoly = lambda img, pts, color: img
    _set_module(monkeypatch, "cv2", cv2_mod)

    nvdiffrast_mod = _install_package(monkeypatch, "nvdiffrast")
    nvdiffrast_torch_mod = types.ModuleType("nvdiffrast.torch")
    nvdiffrast_torch_mod.RasterizeCudaContext = object
    _set_module(monkeypatch, "nvdiffrast.torch", nvdiffrast_torch_mod)
    nvdiffrast_mod.torch = nvdiffrast_torch_mod


def _import_multiview_tracker(monkeypatch):
    _install_multiview_tracker_stubs(monkeypatch)
    name = "v2d.foundation_pose.lib.multiview_tracker"
    _install_package(monkeypatch, "v2d.foundation_pose")
    _install_package(monkeypatch, "v2d.foundation_pose.lib")
    backends_mod = types.ModuleType("v2d.foundation_pose.lib.backends")
    backends_mod.DEFAULT_BACKEND = "nvidia_tensorrt"
    backends_mod.create_predictor_backend = lambda *args, **kwargs: None
    _set_module(monkeypatch, "v2d.foundation_pose.lib.backends", backends_mod)
    spec = importlib.util.spec_from_file_location(name, LIB_DIR / "multiview_tracker.py")
    module = importlib.util.module_from_spec(spec)
    _set_module(monkeypatch, name, module)
    spec.loader.exec_module(module)
    return module


class FakeEstimator:
    def __init__(self, track_pose=None, register_pose=None):
        self.track_pose = np.eye(4) if track_pose is None else track_pose
        self.register_pose = np.eye(4) if register_pose is None else register_pose
        self.register_calls = 0
        self.track_calls = 0

    def track_one(self, **kwargs):
        self.track_calls += 1
        return self.track_pose

    def register(self, **kwargs):
        self.register_calls += 1
        return self.register_pose


def _make_tracker(module, visibility_sequence, estimators=None, pose_last=_DEFAULT_POSE_LAST):
    tracker = module.MultiViewTracker.__new__(module.MultiViewTracker)
    tracker._estimators = estimators or [FakeEstimator(), FakeEstimator()]
    tracker.mesh = types.SimpleNamespace(vertices=np.zeros((1, 3)))
    tracker.pose_last = np.eye(4) if pose_last is _DEFAULT_POSE_LAST else pose_last
    tracker.depth_direction_trust = 0.5
    tracker.visible_ratio_cutoff_high = 0.3
    tracker.visible_ratio_cutoff_low = 0.3
    tracker.precision_high = 1.0
    tracker.precision_low = 1.0
    tracker.symmetry_group = None
    tracker.full_rotational_symmetry = False
    tracker.recovery_enabled = True
    tracker.recovery_refine_iter = 5
    tracker.recovery_min_views = 1
    tracker.recovery_min_valid_depth_pixels = 2
    tracker.recovery_visible_ratio_cutoff = 0.3
    tracker.recovery_attempt_stride = 1
    tracker.under_supported_recovery_enabled = True
    tracker.mask_pose_iou_cutoff = 0.05
    tracker.mask_explained_ratio_cutoff = 0.10
    tracker.register_debug_path = None
    tracker._lost_frame_count = 0
    tracker._under_supported_frame_count = 0
    tracker.last_status = {}
    tracker._visibility_sequence = list(visibility_sequence)
    tracker._compute_visibility = lambda *args, **kwargs: tracker._visibility_sequence.pop(0)
    tracker._weighted_se3_mean_from_poses = (
        lambda poses, Ts, visible, select: np.asarray(poses)[np.where(select)[0][0]]
    )

    def sync_to_avg(pose, Ts):
        tracker.pose_last = np.asarray(pose).copy()

    tracker._sync_to_avg = sync_to_avg
    return tracker


def _frame_inputs():
    rgbs = [np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)]
    depths = [np.ones((2, 2), dtype=float), np.ones((2, 2), dtype=float)]
    masks = [np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)]
    Ks = [np.eye(3), np.eye(3)]
    Ts = [np.eye(4), np.eye(4)]
    return rgbs, depths, masks, Ks, Ts


def test_multiview_tracker_shares_one_predictor_backend(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    scorer = object()
    refiner = object()
    runtime_info = {"backend": "nvidia_tensorrt"}
    factory_calls = []
    estimator_kwargs = []

    def create_backend(backend, weights_dir):
        factory_calls.append((backend, weights_dir))
        return types.SimpleNamespace(
            scorer=scorer,
            refiner=refiner,
            runtime_info=runtime_info,
        )

    class RecordingFoundationPose:
        def __init__(self, **kwargs):
            estimator_kwargs.append(kwargs)

    trimesh_mesh = types.SimpleNamespace(
        vertices=np.zeros((8, 3)),
        vertex_normals=np.zeros((8, 3)),
    )
    mesh = types.SimpleNamespace(to_trimesh=lambda: trimesh_mesh)
    monkeypatch.setattr(module, "create_predictor_backend", create_backend)
    monkeypatch.setattr(module, "FoundationPose", RecordingFoundationPose)

    tracker = module.MultiViewTracker(
        mesh=mesh,
        weights_dir="/weights",
        num_cameras=4,
        backend="nvidia_tensorrt",
    )

    assert factory_calls == [("nvidia_tensorrt", "/weights")]
    assert tracker.runtime_info is runtime_info
    assert len(estimator_kwargs) == 4
    assert all(kwargs["scorer"] is scorer for kwargs in estimator_kwargs)
    assert all(kwargs["refiner"] is refiner for kwargs in estimator_kwargs)


def _pose(x=0.0, yaw_deg=0.0):
    pose = np.eye(4)
    pose[0, 3] = x
    theta = np.deg2rad(yaw_deg)
    pose[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return pose


def _yaw_deg(pose):
    return float(np.degrees(np.arctan2(pose[1, 0], pose[0, 0])))


def _tracked_status(select_count=2, num_cameras=2, visible=0.8):
    select_idx = [i < select_count for i in range(num_cameras)]
    visible_ratios = [visible if selected else 0.0 for selected in select_idx]
    return {
        "status": "tracked",
        "pose_valid": True,
        "select_idx": select_idx,
        "visible_ratios": visible_ratios,
    }


def test_full_rotational_registration_uses_one_reference_rotation(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    tracker = _make_tracker(module, [], pose_last=None)
    tracker.full_rotational_symmetry = True
    raw = [_pose(x=0.1, yaw_deg=25.0), _pose(x=0.2, yaw_deg=-80.0)]

    avg, world, ref_idx, snaps = tracker._fuse_registered_poses(
        raw, [np.eye(4), np.eye(4)], np.array([0.4, 0.9]), np.array([True, True]),
    )

    assert ref_idx == 1
    assert _yaw_deg(avg) == pytest.approx(-80.0)
    assert [_yaw_deg(pose) for pose in world] == pytest.approx([-80.0, -80.0])
    assert snaps == [[(0, 0.0), (0, 0.0)]]


def test_bounded_canonicalization_matches_exhaustive_selection(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    poses = [_pose(x=0.1, yaw_deg=77.0), _pose(x=0.2, yaw_deg=-143.0)]
    group = [_pose(yaw_deg=value) for value in range(0, 360, 30)]
    reference = _pose(yaw_deg=-14.0)

    actual, indices = module._canonicalize_poses_with_indices(
        poses, group, reference,
    )

    for row, pose in enumerate(poses):
        candidates = [pose @ symmetry for symmetry in group]
        distances = []
        for candidate in candidates:
            relative = reference[:3, :3].T @ candidate[:3, :3]
            distances.append(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        expected_index = int(np.argmin(distances))
        assert indices[row][0] == expected_index
        assert indices[row][1] == pytest.approx(distances[expected_index])
        np.testing.assert_allclose(actual[row], candidates[expected_index])


def test_full_rotational_tracking_holds_rotation_and_updates_translation(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    estimators = [
        FakeEstimator(track_pose=_pose(x=0.4, yaw_deg=120.0)),
        FakeEstimator(track_pose=_pose(x=0.6, yaw_deg=-45.0)),
    ]
    tracker = _make_tracker(
        module, [np.array([0.8, 0.7])], estimators=estimators,
        pose_last=_pose(x=0.1, yaw_deg=31.0),
    )
    tracker.full_rotational_symmetry = True
    rgbs, depths, masks, Ks, Ts = _frame_inputs()

    avg, world, _, _ = tracker._track_registered_frame(
        rgbs, depths, masks, Ks, Ts, frame_index=1,
    )

    assert _yaw_deg(avg) == pytest.approx(31.0)
    assert avg[0, 3] == pytest.approx(0.4)
    assert [_yaw_deg(pose) for pose in world] == pytest.approx([31.0, 31.0])


def test_full_rotational_single_view_tracks_translation_without_reregistering(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    under_supported_frames = 403
    estimators = [
        FakeEstimator(track_pose=_pose(x=0.4, yaw_deg=120.0)),
        FakeEstimator(track_pose=_pose(x=0.6, yaw_deg=-45.0)),
    ]
    tracker = _make_tracker(
        module, [np.array([0.8, 0.0])] * under_supported_frames,
        estimators=estimators,
        pose_last=_pose(x=0.1, yaw_deg=31.0),
    )
    tracker.full_rotational_symmetry = True
    tracker.recovery_min_views = 2

    for frame_index in range(281, 281 + under_supported_frames):
        result = tracker.track(*_frame_inputs(), frame_index=frame_index)

    assert result.status["status"] == "tracked"
    assert result.status["pose_valid"] is True
    assert result.status["under_supported"] is True
    assert result.status["recovery_attempted"] is False
    assert result.status["recovery_mode"] == (
        "full_rotational_observable_translation"
    )
    assert result.select_idx.tolist() == [True, False]
    assert _yaw_deg(result.avg_pose) == pytest.approx(31.0)
    assert result.avg_pose[0, 3] == pytest.approx(0.4)
    assert [est.register_calls for est in estimators] == [0, 0]
    assert [est.track_calls for est in estimators] == [
        under_supported_frames, under_supported_frames,
    ]


def test_full_rotational_recovery_keeps_prior_rotation(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    tracker = _make_tracker(module, [], pose_last=_pose(x=0.1, yaw_deg=44.0))
    tracker.full_rotational_symmetry = True
    recovered = [_pose(x=0.7, yaw_deg=-170.0), _pose(x=0.8, yaw_deg=5.0)]

    avg, world, _, _ = tracker._fuse_registered_poses(
        recovered, [np.eye(4), np.eye(4)],
        np.array([0.9, 0.8]), np.array([True, True]),
    )

    assert _yaw_deg(avg) == pytest.approx(44.0)
    assert avg[0, 3] == pytest.approx(0.7)
    assert [_yaw_deg(pose) for pose in world] == pytest.approx([44.0, 44.0])


def test_first_frame_track_registers_object(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    registered_pose = np.eye(4)
    registered_pose[0, 3] = 1.5
    estimators = [FakeEstimator(register_pose=registered_pose), FakeEstimator()]
    tracker = _make_tracker(
        module,
        [np.array([0.8, 0.0])],
        estimators=estimators,
        pose_last=None,
    )

    result = tracker.track(*_frame_inputs(), frame_index=0, register_iteration=7)

    assert np.allclose(result.avg_pose, registered_pose)
    assert result.visible_ratios.tolist() == [0.8, 0.0]
    assert result.select_idx.tolist() == [True, False]
    assert [est.register_calls for est in estimators] == [1, 1]
    assert [est.track_calls for est in estimators] == [0, 0]
    assert result.status["status"] == "registered"
    assert result.status["pose_valid"] is True
    assert result.status["frame_index"] == 0


def test_seed_pose_resets_loss_counters(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    tracker = _make_tracker(module, [])
    seed = np.eye(4)
    seed[2, 3] = 0.25
    tracker._lost_frame_count = 4
    tracker._under_supported_frame_count = 3

    tracker.seed_pose(seed, [np.eye(4), np.eye(4)])

    assert np.allclose(tracker.pose_last, seed)
    assert tracker._lost_frame_count == 0
    assert tracker._under_supported_frame_count == 0


def test_weak_frame_zero_registration_is_marked_under_supported(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    tracker = _make_tracker(
        module,
        [np.array([0.8, 0.0])],
        pose_last=None,
    )
    tracker.recovery_min_views = 2

    result = tracker.track(*_frame_inputs(), frame_index=0)

    assert result.status["status"] == "registered"
    assert result.status["pose_valid"] is True
    assert result.status["under_supported"] is True
    assert result.status["under_supported_initialization"] is True


def test_valid_tracking_sets_pose_valid(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    tracker = _make_tracker(module, [np.array([0.7, 0.0])])

    result = tracker.track(*_frame_inputs(), frame_index=1)

    assert np.allclose(result.avg_pose, np.eye(4))
    assert result.visible_ratios.tolist() == [0.7, 0.0]
    assert result.select_idx.tolist() == [True, False]
    assert result.status["status"] == "tracked"
    assert result.status["pose_valid"] is True
    assert result.status["frame_index"] == 1


def test_all_view_loss_without_candidate_holds_last_pose(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    estimators = [FakeEstimator(), FakeEstimator()]
    tracker = _make_tracker(module, [np.array([0.0, 0.0])], estimators=estimators)
    held_pose = np.eye(4)
    held_pose[0, 3] = 4.0
    tracker.pose_last = held_pose

    rgbs, depths, masks, Ks, Ts = _frame_inputs()
    depths = [np.zeros((2, 2), dtype=float), np.zeros((2, 2), dtype=float)]
    masks = [np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)]
    result = tracker.track(
        rgbs, depths, masks, Ks, Ts, frame_index=2,
    )

    assert np.allclose(result.avg_pose, held_pose)
    assert all(np.allclose(p, held_pose) for p in result.world_poses)
    assert result.select_idx.tolist() == [False, False]
    assert [est.register_calls for est in estimators] == [0, 0]
    assert result.status["status"] == "held"
    assert result.status["pose_valid"] is False
    assert result.status["recovery_failure_reason"] == "no_candidate_views"
    assert result.status["frame_index"] == 2


def test_all_view_loss_recovers_from_valid_mask_and_depth(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    recovered_pose = np.eye(4)
    recovered_pose[0, 3] = 2.0
    estimators = [FakeEstimator(register_pose=recovered_pose), FakeEstimator()]
    tracker = _make_tracker(
        module,
        [np.array([0.0, 0.0]), np.array([0.8, 0.0])],
        estimators=estimators,
    )

    result = tracker.track(*_frame_inputs(), frame_index=3)

    assert np.allclose(result.avg_pose, recovered_pose)
    assert result.visible_ratios.tolist() == [0.8, 0.0]
    assert result.select_idx.tolist() == [True, False]
    assert [est.register_calls for est in estimators] == [1, 1]
    assert result.status["status"] == "recovered"
    assert result.status["pose_valid"] is True
    assert result.status["frame_index"] == 3
    assert np.allclose(tracker.pose_last, recovered_pose)


def test_all_view_loss_rejects_low_visibility_recovery(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    recovered_pose = np.eye(4)
    recovered_pose[0, 3] = 2.0
    estimators = [FakeEstimator(register_pose=recovered_pose), FakeEstimator()]
    tracker = _make_tracker(
        module,
        [np.array([0.0, 0.0]), np.array([0.2, 0.0])],
        estimators=estimators,
    )
    held_pose = np.eye(4)
    held_pose[0, 3] = 4.0
    tracker.pose_last = held_pose

    result = tracker.track(*_frame_inputs(), frame_index=4)

    assert np.allclose(result.avg_pose, held_pose)
    assert all(np.allclose(p, held_pose) for p in result.world_poses)
    assert result.select_idx.tolist() == [False, False]
    assert [est.register_calls for est in estimators] == [1, 1]
    assert result.status["status"] == "held"
    assert result.status["pose_valid"] is False
    assert result.status["recovery_failure_reason"] == "not_enough_recovered_views"
    assert result.status["frame_index"] == 4
    assert np.allclose(tracker.pose_last, held_pose)


def test_under_supported_candidates_require_mask_evidence_and_low_overlap(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    tracker = _make_tracker(module, [])
    world_poses = [np.eye(4), np.eye(4)]
    depths = [np.ones((2, 2), dtype=float), np.ones((2, 2), dtype=float)]
    masks = [np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=bool)]
    Ks = [np.eye(3), np.eye(3)]
    Ts = [np.eye(4), np.eye(4)]
    select_idx = np.array([True, False])
    tracker._mesh_silhouette_mask = lambda *args, **kwargs: np.zeros((2, 2), dtype=bool)

    valid_pixels, pose_iou, explained, candidate_idx = (
        tracker._under_supported_recovery_candidates(
            world_poses, depths, masks, Ks, Ts, select_idx,
        )
    )

    assert valid_pixels.tolist() == [4, 4]
    assert pose_iou.tolist() == [0.0, 0.0]
    assert explained.tolist() == [0.0, 0.0]
    assert candidate_idx.tolist() == [False, True]

    depths[1] = np.array([[1.0, 0.0], [0.0, 0.0]])
    valid_pixels, _, _, candidate_idx = tracker._under_supported_recovery_candidates(
        world_poses, depths, masks, Ks, Ts, select_idx,
    )
    assert valid_pixels.tolist() == [4, 1]
    assert candidate_idx.tolist() == [False, False]

    depths[1] = np.ones((2, 2), dtype=float)
    tracker._mesh_silhouette_mask = lambda *args, **kwargs: np.ones((2, 2), dtype=bool)
    _, pose_iou, explained, candidate_idx = tracker._under_supported_recovery_candidates(
        world_poses, depths, masks, Ks, Ts, select_idx,
    )
    assert pose_iou.tolist() == [1.0, 1.0]
    assert explained.tolist() == [1.0, 1.0]
    assert candidate_idx.tolist() == [False, False]


def test_under_supported_recovers_from_non_selected_mask_evidence(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    recovered_pose = np.eye(4)
    recovered_pose[0, 3] = 2.0
    estimators = [FakeEstimator(), FakeEstimator(register_pose=recovered_pose)]
    tracker = _make_tracker(
        module,
        [np.array([0.8, 0.0]), np.array([0.8, 0.8])],
        estimators=estimators,
    )
    tracker.recovery_min_views = 2
    tracker._mesh_silhouette_mask = lambda *args, **kwargs: np.zeros((2, 2), dtype=bool)

    result = tracker.track(*_frame_inputs(), frame_index=5)

    assert result.select_idx.tolist() == [True, True]
    assert [est.register_calls for est in estimators] == [0, 1]
    assert result.status["status"] == "partially_recovered"
    assert result.status["pose_valid"] is True
    assert result.status["under_supported"] is True
    assert result.status["recovery_mode"] == "under_supported_mask_evidence"
    assert result.status["under_supported_recovery_candidate_idx"] == [False, True]
    assert result.status["under_supported_recovery_select_idx"] == [True, True]


def test_under_supported_failed_recovery_holds_last_pose(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    recovered_pose = np.eye(4)
    recovered_pose[0, 3] = 2.0
    estimators = [FakeEstimator(), FakeEstimator(register_pose=recovered_pose)]
    tracker = _make_tracker(
        module,
        [np.array([0.8, 0.0]), np.array([0.8, 0.0])],
        estimators=estimators,
    )
    tracker.recovery_min_views = 2
    held_pose = np.eye(4)
    held_pose[0, 3] = 4.0
    tracker.pose_last = held_pose
    tracker._mesh_silhouette_mask = lambda *args, **kwargs: np.zeros((2, 2), dtype=bool)

    result = tracker.track(*_frame_inputs(), frame_index=6)

    assert np.allclose(result.avg_pose, held_pose)
    assert all(np.allclose(p, held_pose) for p in result.world_poses)
    assert result.select_idx.tolist() == [True, False]
    assert [est.register_calls for est in estimators] == [0, 1]
    assert result.status["status"] == "held"
    assert result.status["pose_valid"] is False
    assert result.status["under_supported"] is True
    assert result.status["recovery_failure_reason"] == "not_enough_recovered_views"
    assert np.allclose(tracker.pose_last, held_pose)


def test_under_supported_without_mask_evidence_preserves_one_view_tracking(monkeypatch):
    module = _import_multiview_tracker(monkeypatch)
    estimators = [FakeEstimator(), FakeEstimator()]
    tracker = _make_tracker(
        module,
        [np.array([0.8, 0.0])],
        estimators=estimators,
    )
    tracker.recovery_min_views = 2
    rgbs, depths, masks, Ks, Ts = _frame_inputs()
    depths[1] = np.zeros((2, 2), dtype=float)
    masks[1] = np.zeros((2, 2), dtype=bool)
    tracker._mesh_silhouette_mask = lambda *args, **kwargs: np.zeros((2, 2), dtype=bool)

    result = tracker.track(rgbs, depths, masks, Ks, Ts, frame_index=7)

    assert result.status["status"] == "tracked"
    assert result.status["pose_valid"] is True
    assert result.status["under_supported"] is True
    assert result.status["under_supported_recovery_candidate_idx"] == [False, False]
    assert [est.register_calls for est in estimators] == [0, 0]


def _install_mv_videos_stubs(monkeypatch):
    _install_package(monkeypatch, "v2d")
    _install_package(monkeypatch, "v2d.common")
    datatypes_mod = types.ModuleType("v2d.common.datatypes")
    datatypes_mod.DepthImage = object
    datatypes_mod.Mask = object
    _set_module(monkeypatch, "v2d.common.datatypes", datatypes_mod)

    video_mod = types.ModuleType("v2d.common.video")
    video_mod.FrameSource = object
    video_mod.get_video_writer = lambda *args, **kwargs: None
    _set_module(monkeypatch, "v2d.common.video", video_mod)

    _install_package(monkeypatch, "v2d.mesh")
    _install_package(monkeypatch, "v2d.mesh.lib")
    mesh_mod = types.ModuleType("v2d.mesh.lib.mesh")
    mesh_mod.Mesh = object
    _set_module(monkeypatch, "v2d.mesh.lib.mesh", mesh_mod)

    _install_package(monkeypatch, "v2d.mv")
    _install_package(monkeypatch, "v2d.mv.math")
    numpy_fn_mod = types.ModuleType("v2d.mv.math.numpy_fn")
    numpy_fn_mod.pose_two_euro_filter = lambda poses: poses
    _set_module(monkeypatch, "v2d.mv.math.numpy_fn", numpy_fn_mod)
    rig_mod = types.ModuleType("v2d.mv.rig")
    rig_mod.RigConfig = object
    _set_module(monkeypatch, "v2d.mv.rig", rig_mod)

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.INTER_AREA = 0
    cv2_mod.INTER_LINEAR = 1
    cv2_mod.INTER_NEAREST = 2
    cv2_mod.FONT_HERSHEY_SIMPLEX = 0
    cv2_mod.LINE_AA = 16
    cv2_mod.resize = lambda arr, size, interpolation=None: arr
    cv2_mod.rectangle = lambda img, pt1, pt2, color, thickness: img
    cv2_mod.getTextSize = lambda text, font, scale, thickness: ((len(text) * 10, 20), 0)
    cv2_mod.putText = lambda img, *args, **kwargs: img
    _set_module(monkeypatch, "cv2", cv2_mod)

    imageio_mod = _install_package(monkeypatch, "imageio")
    imageio_v3_mod = types.ModuleType("imageio.v3")
    imageio_v3_mod.imwrite = lambda *args, **kwargs: None
    _set_module(monkeypatch, "imageio.v3", imageio_v3_mod)
    imageio_mod.v3 = imageio_v3_mod

    omegaconf_mod = types.ModuleType("omegaconf")
    omegaconf_mod.OmegaConf = object
    _set_module(monkeypatch, "omegaconf", omegaconf_mod)
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda iterable, **kwargs: iterable
    _set_module(monkeypatch, "tqdm", tqdm_mod)
    trimesh_mod = types.ModuleType("trimesh")
    trimesh_mod.bounds = types.SimpleNamespace(
        oriented_bounds=lambda mesh: (np.eye(4), np.ones(3)),
    )
    _set_module(monkeypatch, "trimesh", trimesh_mod)

    _install_package(monkeypatch, "v2d.foundation_pose")
    _install_package(monkeypatch, "v2d.foundation_pose.lib")
    backends_mod = types.ModuleType("v2d.foundation_pose.lib.backends")
    backends_mod.DEFAULT_BACKEND = "nvidia_tensorrt"
    backends_mod.SUPPORTED_BACKENDS = ("nvidia_tensorrt", "nvlabs_pytorch")
    _set_module(monkeypatch, "v2d.foundation_pose.lib.backends", backends_mod)
    fp_utils_mod = types.ModuleType("v2d.foundation_pose.lib.fp_utils")
    fp_utils_mod.draw_posed_3d_box = lambda *args, **kwargs: kwargs["img"]
    fp_utils_mod.draw_xyz_axis = lambda img, *args, **kwargs: img
    _set_module(monkeypatch, "v2d.foundation_pose.lib.fp_utils", fp_utils_mod)
    tracker_mod = types.ModuleType("v2d.foundation_pose.lib.multiview_tracker")
    tracker_mod.MultiViewTracker = object
    tracker_mod.RegistrationFailure = RuntimeError
    _set_module(monkeypatch, "v2d.foundation_pose.lib.multiview_tracker", tracker_mod)
    symmetry_mod = types.ModuleType("v2d.foundation_pose.lib.symmetry")
    symmetry_mod.DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG = 10.0
    symmetry_mod.identity_symmetry_spec = lambda step=10.0: types.SimpleNamespace(
        transforms=np.eye(4)[None], full_rotational=False,
        candidate_count=1, mode="none", provenance=lambda: {},
    )
    symmetry_mod.load_symmetry_spec = lambda path, continuous_step_deg=10.0: (
        symmetry_mod.identity_symmetry_spec(continuous_step_deg)
    )
    _set_module(monkeypatch, "v2d.foundation_pose.lib.symmetry", symmetry_mod)


def _import_mv_videos_to_poses(monkeypatch):
    _install_mv_videos_stubs(monkeypatch)
    name = "v2d.foundation_pose.lib.mv_videos_to_poses"
    spec = importlib.util.spec_from_file_location(name, LIB_DIR / "mv_videos_to_poses.py")
    module = importlib.util.module_from_spec(spec)
    _set_module(monkeypatch, name, module)
    spec.loader.exec_module(module)
    return module


def test_grouped_config_parsing(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    cfg = types.SimpleNamespace(
        recovery={
            "enabled": False,
            "refine_iter": 9,
            "min_views": 3,
            "min_valid_depth_pixels": 42,
            "visible_ratio_cutoff": 0.4,
            "attempt_stride": 2,
            "under_supported": {
                "enabled": False,
                "mask_pose_iou_cutoff": 0.02,
                "mask_explained_ratio_cutoff": 0.08,
            },
        },
        repair={
            "enabled": True,
            "arbitration": {
                "max_step_translation_m": 0.2,
                "max_step_rotation_deg": 35,
                "visible_ratio_tolerance": 0.05,
            },
            "recovery_trigger": {
                "enabled": False,
                "max_window": 40,
                "anchor_stable_frames": 6,
            },
            "snap_trigger": {
                "enabled": True,
                "max_span": 180,
                "anchor_stable_frames": 9,
                "outlier_window": 11,
                "rotation_mad_scale": 4.0,
                "translation_mad_scale": 6.0,
                "min_rotation_deg": 12,
                "max_translation_m": 0.09,
                "max_burst_frames": 2,
            },
        },
    )

    recovery = module._recovery_config_from_cfg(cfg)
    repair = module._repair_config_from_cfg(cfg, recovery.min_views, track_refine_iter=4)

    assert recovery.enabled is False
    assert recovery.refine_iter == 9
    assert recovery.min_views == 3
    assert recovery.under_supported_enabled is False
    assert recovery.mask_pose_iou_cutoff == 0.02
    assert repair.recovery_trigger_enabled is False
    assert repair.recovery_trigger_max_window == 40
    assert repair.snap_trigger_max_span == 180
    assert repair.snap_trigger_max_burst_frames == 2
    assert repair.recovery_min_views == 3
    assert repair.track_refine_iter == 4


def test_pose_tracking_metadata_preserves_v1_and_adds_global_diagnostics(
    monkeypatch,
):
    module = _import_mv_videos_to_poses(monkeypatch)
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    recovery = module.RecoveryConfig(enabled=True)
    repair = module.RepairConfig(enabled=False)

    metadata = module._build_pose_tracking_metadata(
        frame_start=10,
        frame_end_exclusive=13,
        filtered_poses=poses,
        pose_valid_mask=np.array([True, False, True]),
        select_mask=np.array(
            [[True, False], [False, False], [True, True]],
        ),
        tracking_status=[
            {"status": "registered"},
            {"status": "held"},
            {"status": "recovered"},
        ],
        cam_names=["front", "back"],
        registration_camera_names=["front"],
        highest_visibility_registration_camera="front",
        registration_visible_ratios={"front": 0.8, "back": 0.1},
        recovery_config=recovery,
        repair_config=repair,
        smooth_across_recovery=False,
    )

    assert metadata["schema_version"] == 1
    assert metadata["source_frame_start"] == 10
    assert metadata["source_frame_end_exclusive"] == 13
    assert metadata["pose_count"] == 3
    assert metadata["registration_camera_names"] == ["front"]
    assert metadata["tracking_camera_frame_counts"] == {
        "front": 2,
        "back": 1,
    }
    assert metadata["pose_valid_count"] == 2
    assert metadata["camera_supported_frame_count"] == 2
    assert metadata["tracking_policy"] == {
        "recovery_enabled": True,
        "repair_enabled": False,
        "smooth_across_recovery": False,
    }
    assert metadata["tracking_status_counts"] == {
        "held": 1,
        "recovered": 1,
        "registered": 1,
    }
    assert metadata["artifacts"]["poses"] == "poses.npy"
    assert "repair_status" not in metadata["artifacts"]


def test_pose_tracking_metadata_rejects_mismatched_source_interval(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    with pytest.raises(RuntimeError, match="pose count"):
        module._build_pose_tracking_metadata(
            frame_start=10,
            frame_end_exclusive=14,
            filtered_poses=np.repeat(np.eye(4)[None], 3, axis=0),
            pose_valid_mask=np.ones(3, dtype=bool),
            select_mask=np.ones((3, 1), dtype=bool),
            tracking_status=[{"status": "tracked"}] * 3,
            cam_names=["front"],
            registration_camera_names=["front"],
            highest_visibility_registration_camera="front",
            registration_visible_ratios={"front": 0.8},
            recovery_config=module.RecoveryConfig(),
            repair_config=module.RepairConfig(),
            smooth_across_recovery=False,
        )


def test_filter_valid_pose_segments_keeps_invalid_placeholders(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    calls = []

    def fake_filter(segment):
        calls.append(segment[:, 0, 3].tolist())
        out = segment.copy()
        out[:, 0, 3] += 100.0
        return out

    module.pose_two_euro_filter = fake_filter
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, 0, 3] = np.arange(5)
    valid = np.array([True, True, False, True, True])

    filtered = module._filter_valid_pose_segments(
        poses, valid, smooth_across_recovery=False,
    )

    assert calls == [[0.0, 1.0], [3.0, 4.0]]
    assert filtered[2, 0, 3] == 2.0
    assert filtered[[0, 1, 3, 4], 0, 3].tolist() == [100.0, 101.0, 103.0, 104.0]


def test_filter_valid_pose_segments_can_smooth_across_recovery(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    calls = []

    def fake_filter(segment):
        calls.append(segment[:, 0, 3].tolist())
        out = segment.copy()
        out[:, 0, 3] += 100.0
        return out

    module.pose_two_euro_filter = fake_filter
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, 0, 3] = np.arange(5)
    valid = np.array([True, True, False, True, True])

    filtered = module._filter_valid_pose_segments(
        poses, valid, smooth_across_recovery=True,
    )

    assert calls == [[0.0, 1.0, 3.0, 4.0]]
    assert filtered[2, 0, 3] == 2.0
    assert filtered[[0, 1, 3, 4], 0, 3].tolist() == [100.0, 101.0, 103.0, 104.0]


def test_front_masked_depth_stats_ignore_invalid_depths(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    depth = np.array([
        [1.0, 2.0, np.nan],
        [0.0, 0.0005, 4.0],
    ])
    mask = np.array([
        [True, True, True],
        [True, True, False],
    ])

    stats = module._masked_depth_stats(depth, mask)

    assert stats == {"mean_m": 1.5, "valid_pixels": 2}


def test_backward_repair_start_detection(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    statuses = [
        {"status": "registered", "pose_valid": True, "under_supported_initialization": True},
        {"status": "tracked", "pose_valid": True},
        {"status": "held", "pose_valid": False},
        {"status": "recovered", "pose_valid": True},
        {"status": "partially_recovered", "pose_valid": True},
        {"status": "tracked", "pose_valid": True, "recovery_attempted": True},
        {"status": "tracked", "pose_valid": False},
    ]

    assert module._backward_repair_start_indices(statuses) == [0, 2, 3, 4, 5, 6]


def test_backward_anchor_uses_last_frame_of_first_stable_run(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    statuses = [
        {"status": "recovered", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
        {"status": "tracked", "pose_valid": True, "under_supported": True, "select_idx": [True, False], "visible_ratios": [0.8, 0.0]},
        {"status": "tracked", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
        {"status": "tracked", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
        {"status": "tracked", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
        {"status": "tracked", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
    ]
    poses = np.array([_pose(x=i * 0.01) for i in range(len(statuses))])

    anchor = module._find_backward_repair_anchor(
        statuses,
        poses,
        start_idx=0,
        max_window=10,
        min_future_stable_frames=3,
        recovery_min_views=2,
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
    )

    assert anchor == 4


def test_backward_anchor_rejects_unsettled_pose_jumps(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    statuses = [
        {"status": "recovered", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
        {"status": "tracked", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
        {"status": "tracked", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
        {"status": "tracked", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
    ]
    poses = np.array([_pose(), _pose(x=0.0), _pose(x=0.3), _pose(x=0.31)])

    anchor = module._find_backward_repair_anchor(
        statuses,
        poses,
        start_idx=0,
        max_window=10,
        min_future_stable_frames=3,
        recovery_min_views=2,
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
    )

    assert anchor is None


def test_backward_acceptance_requires_valid_continuous_equal_support(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    forward_status = {
        "select_idx": [True, False],
        "visible_ratios": [0.4, 0.0],
    }
    result = types.SimpleNamespace(
        status={"pose_valid": True},
        avg_pose=_pose(x=0.04),
        select_idx=np.array([True, True]),
        visible_ratios=np.array([0.4, 0.4]),
    )

    accepted, reason, delta = module._accept_backward_result(
        result,
        forward_status,
        next_future_pose=_pose(x=0.05),
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
    )

    assert accepted is True
    assert reason is None
    assert delta["translation_m"] < 0.02


def test_backward_acceptance_ignores_selected_count_but_rejects_jump_and_visible_deficit(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    forward_status = {
        "select_idx": [True, True],
        "visible_ratios": [0.4, 0.4],
    }
    lower_support = types.SimpleNamespace(
        status={"pose_valid": True},
        avg_pose=_pose(x=0.04),
        select_idx=np.array([True, False]),
        visible_ratios=np.array([0.8, 0.0]),
    )
    visible_deficit = types.SimpleNamespace(
        status={"pose_valid": True},
        avg_pose=_pose(x=0.04),
        select_idx=np.array([True, False]),
        visible_ratios=np.array([0.4, 0.0]),
    )
    jump = types.SimpleNamespace(
        status={"pose_valid": True},
        avg_pose=_pose(x=1.0),
        select_idx=np.array([True, True]),
        visible_ratios=np.array([0.4, 0.4]),
    )

    accepted, reason, _ = module._accept_backward_result(
        lower_support,
        forward_status,
        next_future_pose=_pose(x=0.05),
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
    )
    assert accepted is True
    assert reason is None

    accepted, reason, _ = module._accept_backward_result(
        visible_deficit,
        forward_status,
        next_future_pose=_pose(x=0.05),
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
    )
    assert accepted is False
    assert reason == "lower_visible_ratio_sum"

    accepted, reason, _ = module._accept_backward_result(
        jump,
        forward_status,
        next_future_pose=_pose(x=0.05),
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
    )
    assert accepted is False
    assert reason == "translation_jump"


def test_backward_repair_no_anchor_leaves_forward_outputs_unchanged(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    poses = np.array([_pose(x=0.0), _pose(x=0.5)])
    valid = np.array([False, True])
    statuses = [
        {"status": "held", "pose_valid": False, "select_idx": [False], "visible_ratios": [0.0]},
        {"status": "tracked", "pose_valid": True, "under_supported": True, "select_idx": [True], "visible_ratios": [0.8]},
    ]
    select_mask = np.array([[False], [True]])
    visible = np.array([[0.0], [0.8]])
    tracker = types.SimpleNamespace(
        recovery_enabled=True,
        under_supported_recovery_enabled=True,
    )

    (
        out_poses,
        out_valid,
        out_statuses,
        out_select,
        out_visible,
        records,
        candidate_poses,
    ) = module._run_backward_repair(
        tracker=tracker,
        poses=poses.copy(),
        pose_valid_mask=valid.copy(),
        tracking_status=[dict(status) for status in statuses],
        select_mask=select_mask.copy(),
        visible_ratios_history=visible.copy(),
        frame_context=module.TrackingFrameContext(
            frame_sources=[],
            depth_sources=[],
            mask_sources=[],
            cam_names=["front"],
            Ks=[],
            Ts=[],
            scale_target_size=None,
        ),
        repair_config=module.RepairConfig(
            recovery_min_views=1,
            recovery_trigger_max_window=5,
            recovery_trigger_anchor_stable_frames=2,
            snap_trigger_enabled=False,
        ),
    )

    assert np.allclose(out_poses, poses)
    assert out_valid.tolist() == valid.tolist()
    assert out_statuses == statuses
    assert np.array_equal(out_select, select_mask)
    assert np.allclose(out_visible, visible)
    assert np.isnan(candidate_poses).all()
    assert records == [{
        "trigger_type": "recovery_trigger",
        "trigger_frame": 0,
        "start_floor": 0,
        "start_floor_status": "held",
        "anchor_min_views": 1,
        "anchor_frame": None,
        "accepted_frames": [],
        "kept_existing_frames": [],
        "rejected_frames": [],
        "failure_reason": "no_stable_future_anchor",
    }]


def test_candidate_first_repair_newer_anchor_owns_overlap(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    poses = np.array([_pose(x=i * 0.01) for i in range(8)])
    valid = np.ones(len(poses), dtype=bool)
    statuses = [_tracked_status(select_count=1, num_cameras=1) for _ in poses]
    select_mask = np.ones((len(poses), 1), dtype=bool)
    visible = np.ones((len(poses), 1), dtype=float)
    anchors = [
        {
            "trigger_type": "recovery_trigger",
            "trigger_frame": 0,
            "start_floor": 0,
            "anchor_frame": 3,
            "anchor_min_views": 1,
        },
        {
            "trigger_type": "snap_trigger",
            "trigger_frame": 4,
            "start_floor": 1,
            "anchor_frame": 6,
            "anchor_min_views": 1,
        },
    ]

    class FakeRepairTracker:
        recovery_enabled = True
        under_supported_recovery_enabled = True

        def seed_pose(self, pose, Ts):
            self.anchor_frame = int(round(float(np.asarray(pose)[0, 3]) * 100))

        def track(self, *args, frame_index, **kwargs):
            return types.SimpleNamespace(
                avg_pose=_pose(x=frame_index * 0.01, yaw_deg=self.anchor_frame),
                status={
                    "pose_valid": True,
                    "select_idx": [True],
                    "visible_ratios": [1.0],
                },
                select_idx=np.array([True]),
                visible_ratios=np.array([1.0]),
            )

    module._detect_repair_anchors = lambda **kwargs: sorted(
        anchors,
        key=lambda a: (a["anchor_frame"], a["trigger_frame"]),
        reverse=True,
    )
    loaded_source_frames = []

    def fake_load_tracking_inputs_at(**kwargs):
        loaded_source_frames.append(kwargs["frame_index"])
        return [None], [None], [None]

    module._load_tracking_inputs_at = fake_load_tracking_inputs_at

    (
        out_poses,
        out_valid,
        out_statuses,
        _,
        _,
        records,
        candidate_poses,
    ) = module._run_backward_repair(
        tracker=FakeRepairTracker(),
        poses=poses.copy(),
        pose_valid_mask=valid.copy(),
        tracking_status=[dict(status) for status in statuses],
        select_mask=select_mask.copy(),
        visible_ratios_history=visible.copy(),
        frame_context=module.TrackingFrameContext(
            frame_sources=[],
            depth_sources=[],
            mask_sources=[],
            cam_names=["front"],
            Ks=[np.eye(3)],
            Ts=[np.eye(4)],
            scale_target_size=None,
            source_frame_start=100,
        ),
        repair_config=module.RepairConfig(recovery_min_views=1),
    )

    assert out_valid.all()
    assert records[0]["anchor_frame"] == 6
    assert records[0]["accepted_frames"] == [1, 2, 3, 4, 5]
    assert records[1]["anchor_frame"] == 3
    assert records[1]["accepted_frames"] == [0]
    assert [item["frame"] for item in records[1]["kept_existing_frames"]] == [2, 1]
    assert out_statuses[1]["status"] == "repair_replaced"
    assert out_statuses[1]["repair_anchor_frame"] == 6
    assert out_statuses[2]["repair_anchor_frame"] == 6
    assert out_statuses[0]["repair_anchor_frame"] == 3
    assert round(_yaw_deg(out_poses[1]), 1) == 6.0
    assert round(_yaw_deg(out_poses[0]), 1) == 3.0
    assert loaded_source_frames == [105, 104, 103, 102, 101, 102, 101, 100]
    assert np.isfinite(candidate_poses[[0, 1, 2, 3, 4, 5]]).all()
    assert np.isnan(candidate_poses[[6, 7]]).all()


def test_recovery_triggered_span_ignores_large_translation_jump(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    poses = np.array([
        _pose(x=0.0),
        _pose(x=1.0, yaw_deg=80),
        _pose(x=1.01, yaw_deg=81),
        _pose(x=1.02, yaw_deg=82),
    ])
    statuses = [
        {"status": "recovered", "pose_valid": True, "select_idx": [True, True], "visible_ratios": [0.8, 0.8]},
        _tracked_status(select_count=2),
        _tracked_status(select_count=2),
        _tracked_status(select_count=2),
    ]

    anchors = module._detect_repair_anchors(
        statuses=statuses,
        poses=poses,
        num_cameras=2,
        repair_config=module.RepairConfig(
            recovery_min_views=2,
            recovery_trigger_max_window=5,
            recovery_trigger_anchor_stable_frames=2,
            max_step_translation_m=0.15,
            max_step_rotation_deg=45,
            snap_trigger_enabled=False,
        ),
    )

    assert len(anchors) == 1
    assert anchors[0]["trigger_type"] == "recovery_trigger"
    assert anchors[0]["start_floor"] == 0
    assert anchors[0]["anchor_frame"] == 2


def test_repair_anchors_are_sorted_newest_first(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    poses = np.array([_pose(x=i * 0.01) for i in range(20)])
    statuses = [_tracked_status(select_count=4, num_cameras=4) for _ in poses]

    module._backward_repair_start_indices = lambda statuses: [1]
    module._find_backward_repair_anchor = lambda *args, **kwargs: 5
    module._detect_rotation_snap_bursts = lambda *args, **kwargs: [{
        "snap_start_frame": 7,
        "snap_end_frame": 8,
        "snap_step_frames": [8],
        "snap_rotation_deg": 30.0,
        "snap_translation_m": 0.0,
    }]
    module._snap_repair_start_index = lambda *args, **kwargs: 2
    module._find_snap_repair_anchor = lambda *args, **kwargs: (12, 4)

    anchors = module._detect_repair_anchors(
        statuses=statuses,
        poses=poses,
        num_cameras=4,
        repair_config=module.RepairConfig(recovery_min_views=2),
    )

    assert [(a["trigger_type"], a["anchor_frame"]) for a in anchors] == [
        ("snap_trigger", 12),
        ("recovery_trigger", 5),
    ]


def test_rotation_snap_detector_finds_single_and_burst_outliers(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    yaws = [0, 1, 2, 3, 23, 24, 25, 26, 42, 59, 77, 78, 79]
    poses = np.array([_pose(yaw_deg=yaw) for yaw in yaws])
    statuses = [_tracked_status(select_count=3, num_cameras=4) for _ in yaws]

    bursts = module._detect_rotation_snap_bursts(
        poses=poses,
        statuses=statuses,
        outlier_window=4,
        rotation_mad_scale=3.0,
        translation_mad_scale=5.0,
        min_rotation_deg=10.0,
        max_translation_m=0.08,
        max_burst_frames=3,
    )

    assert [(b["snap_start_frame"], b["snap_end_frame"]) for b in bursts] == [(3, 4), (7, 10)]
    assert round(bursts[0]["snap_rotation_deg"], 1) == 20.0
    assert round(bursts[1]["snap_rotation_deg"], 1) == 51.0


def test_rotation_snap_detector_rejects_translation_outlier(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    poses = np.array([
        _pose(x=0.0, yaw_deg=0),
        _pose(x=0.01, yaw_deg=1),
        _pose(x=0.02, yaw_deg=2),
        _pose(x=0.20, yaw_deg=25),
        _pose(x=0.21, yaw_deg=26),
    ])
    statuses = [_tracked_status(select_count=2) for _ in range(len(poses))]

    bursts = module._detect_rotation_snap_bursts(
        poses=poses,
        statuses=statuses,
        outlier_window=3,
        rotation_mad_scale=3.0,
        translation_mad_scale=3.0,
        min_rotation_deg=10.0,
        max_translation_m=0.08,
        max_burst_frames=3,
    )

    assert bursts == []


def test_rotation_snap_detector_rejects_continuous_rotation(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    poses = np.array([_pose(yaw_deg=i * 12) for i in range(8)])
    statuses = [_tracked_status(select_count=2) for _ in range(len(poses))]

    bursts = module._detect_rotation_snap_bursts(
        poses=poses,
        statuses=statuses,
        outlier_window=3,
        rotation_mad_scale=3.0,
        translation_mad_scale=5.0,
        min_rotation_deg=10.0,
        max_translation_m=0.08,
        max_burst_frames=3,
    )

    assert bursts == []


def test_snap_span_start_uses_prior_suspicious_event_or_bounded_lookback(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    statuses = [_tracked_status(select_count=2) for _ in range(12)]
    statuses[3] = {
        "status": "partially_recovered",
        "pose_valid": True,
        "select_idx": [True, True],
        "visible_ratios": [0.8, 0.8],
    }

    assert module._snap_repair_start_index(statuses, snap_start_frame=9, max_span=10) == 3
    statuses[3] = _tracked_status(select_count=2)
    assert module._snap_repair_start_index(statuses, snap_start_frame=9, max_span=4) == 5


def test_snap_repair_anchor_prefers_all_cameras_then_falls_back(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    poses = np.array([_pose(x=i * 0.01) for i in range(12)])
    statuses = [_tracked_status(select_count=2, num_cameras=4) for _ in range(12)]
    for i in range(5, 9):
        statuses[i] = _tracked_status(select_count=3, num_cameras=4)
    for i in range(9, 12):
        statuses[i] = _tracked_status(select_count=4, num_cameras=4)

    anchor, views = module._find_snap_repair_anchor(
        statuses=statuses,
        poses=poses,
        search_after_idx=3,
        max_window=10,
        stable_frames=3,
        num_cameras=4,
        recovery_min_views=2,
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
    )
    assert (anchor, views) == (11, 4)

    anchor, views = module._find_snap_repair_anchor(
        statuses=statuses,
        poses=poses,
        search_after_idx=3,
        max_window=5,
        stable_frames=3,
        num_cameras=4,
        recovery_min_views=2,
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
    )
    assert (anchor, views) == (7, 3)


def test_backward_acceptance_allows_visible_ratio_tolerance(monkeypatch):
    module = _import_mv_videos_to_poses(monkeypatch)
    forward_status = {
        "select_idx": [True, True],
        "visible_ratios": [0.5, 0.5],
    }
    result = types.SimpleNamespace(
        status={"pose_valid": True},
        avg_pose=_pose(x=0.01),
        select_idx=np.array([True, True]),
        visible_ratios=np.array([0.48, 0.44]),
    )

    accepted, reason, _ = module._accept_backward_result(
        result,
        forward_status,
        next_future_pose=_pose(x=0.02),
        max_step_translation_m=0.15,
        max_step_rotation_deg=45,
        visible_ratio_tolerance=0.10,
    )

    assert accepted is True
    assert reason is None
