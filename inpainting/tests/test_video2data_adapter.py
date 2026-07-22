from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inpainting.adapters import video2data
from inpainting.adapters.video2data import (
    ExistingSharpaBackend,
    UpstreamOutputError,
    arrays_from_bundle,
    convert,
    default_sequence_id,
    load_result_bundle,
    load_wilor_json_bundle,
    mirror_right_space_params_for_native_left,
    preflight,
    trajectory_from_tracking,
)
from inpainting.contracts import (
    VideoGeometry,
    validate_robot_trajectory_file,
    validate_tracking_file,
)


class FakeMano:
    identity = "fake-mano"

    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray | str]] = []

    def forward(
        self,
        *,
        side: str,
        betas: np.ndarray,
        global_orient: np.ndarray,
        finger_pose: np.ndarray,
    ) -> dict[str, np.ndarray]:
        self.calls.append(
            {
                "side": side,
                "betas": betas.copy(),
                "global_orient": global_orient.copy(),
                "finger_pose": finger_pose.copy(),
            }
        )
        count = len(global_orient)
        joints = np.zeros((count, 21, 3), dtype=np.float32)
        joints[:, :, 1] = np.arange(21, dtype=np.float32)
        vertices = np.zeros((count, 778, 3), dtype=np.float32)
        vertices[:, :, 0] = 1.0
        quaternions = np.zeros((count, 21, 4), dtype=np.float32)
        quaternions[:, :, 0] = 1.0
        return {
            "joints": joints,
            "joints_wxyz": quaternions,
            "vertices": vertices,
        }


class FakeSharpa:
    identity = "fake-sharpa"

    def __init__(self) -> None:
        self.valid: dict[str, np.ndarray] | None = None

    def retarget(
        self,
        *,
        joints: dict[str, np.ndarray],
        joints_wxyz: dict[str, np.ndarray],
        valid: dict[str, np.ndarray],
        mano_to_robot_scale: float,
    ) -> dict[str, dict[str, np.ndarray]]:
        assert mano_to_robot_scale == pytest.approx(1.2)
        self.valid = {side: values.copy() for side, values in valid.items()}
        result = {}
        for side in ("left", "right"):
            frame_count = len(valid[side])
            # Deliberately return finite invalid rows: the adapter must mask
            # them instead of treating solver continuity as an observation.
            result[side] = {
                "wrist_position": np.nan_to_num(joints[side][:, 0]).astype(np.float32),
                "wrist_wxyz": np.tile(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    (frame_count, 1),
                ),
                "finger_joints": np.ones((frame_count, 2), dtype=np.float32),
                "finger_joint_names": np.asarray(["joint_a", "joint_b"]),
            }
        return result


def _bundle_arrays(frame_count: int = 3) -> dict[str, np.ndarray]:
    camera_to_world = np.tile(np.eye(4, dtype=np.float32), (frame_count, 1, 1))
    arrays: dict[str, np.ndarray] = {
        "camera_to_world_transform": camera_to_world,
        "camera_is_valid": np.ones(frame_count, dtype=np.bool_),
    }
    for side in ("left", "right"):
        arrays.update(
            {
                f"hand_{side}_betas": np.zeros(10, dtype=np.float32),
                f"hand_{side}_wrist_orient_in_camera": np.zeros(
                    (frame_count, 3), dtype=np.float32
                ),
                f"hand_{side}_wrist_trans_in_camera": np.zeros(
                    (frame_count, 3), dtype=np.float32
                ),
                f"hand_{side}_finger_pose": np.zeros(
                    (frame_count, 15, 3), dtype=np.float32
                ),
                f"hand_{side}_scale": np.ones(frame_count, dtype=np.float32),
                f"hand_{side}_is_valid": np.ones(frame_count, dtype=np.bool_),
            }
        )
    return arrays


def _write_bundle(
    root: Path,
    arrays: dict[str, np.ndarray] | None = None,
    *,
    hand_source: str = "wilor",
) -> Path:
    result_dir = root / "result"
    result_dir.mkdir(parents=True)
    arrays = arrays or _bundle_arrays()
    np.savez(result_dir / "result.npz", **arrays)
    frame_count = int(arrays["camera_to_world_transform"].shape[0])
    manifest = {
        "schema_version": 1,
        "n_frames": frame_count,
        "result_npz": "result.npz",
        "camera_pose_convention": "camera_to_world",
        "sources": {
            "pipeline": "ego_wilor",
            "hand_pose_source": hand_source,
            "camera_to_world_dir": "synthetic-camera-poses",
        },
        "keys": sorted(arrays),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest))
    return result_dir


def _wilor_record(
    *,
    is_right: bool,
    width: int = 100,
    height: int = 50,
    betas: float = 0.0,
) -> dict:
    return {
        "is_right": is_right,
        "score": 1.0,
        "bbox": {"x0": 10.0, "y0": 5.0, "x1": 30.0, "y1": 35.0},
        "mano": {
            "betas": [betas] * 10,
            "global_orient": [1.0, 2.0, 3.0],
            "hand_pose": [0.0] * 45,
        },
        "camera": {
            "pred_cam_t_full": [1.0, 2.0, 10.0],
            "scaled_focal_length": 50.0,
        },
        "image_size": [width, height],
    }


def _write_raw_wilor_camera(root: Path) -> tuple[Path, Path, Path]:
    json_dir = root / "wilor"
    json_dir.mkdir(parents=True)
    (json_dir / "000000.json").write_text(
        json.dumps([_wilor_record(is_right=True, betas=0.0)])
    )
    (json_dir / "000001.json").write_text(
        json.dumps([_wilor_record(is_right=False, betas=2.0)])
    )
    intrinsic = root / "egocentric_intrinsic.txt"
    np.savetxt(intrinsic, np.array([[100.0, 0.0, 40.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]]))
    extrinsic = root / "egocentric_frame_extrinsic.npy"
    np.save(extrinsic, np.tile(np.eye(4), (2, 1, 1)))
    return json_dir, intrinsic, extrinsic


def test_left_parameter_conversion_is_non_mutating() -> None:
    orient = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    fingers = np.ones((1, 15, 3), dtype=np.float32)
    corrected_orient, corrected_fingers = mirror_right_space_params_for_native_left(
        orient, fingers
    )
    np.testing.assert_array_equal(corrected_orient, [[1.0, -2.0, -3.0]])
    np.testing.assert_array_equal(corrected_fingers[..., 0], 1.0)
    np.testing.assert_array_equal(corrected_fingers[..., 1:], -1.0)
    np.testing.assert_array_equal(orient, [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(fingers, 1.0)


def test_conversion_applies_scale_world_pose_and_authoritative_validity(
    tmp_path: Path,
) -> None:
    arrays = _bundle_arrays()
    arrays["camera_to_world_transform"][:, :3, 3] = [0.0, 5.0, 0.0]
    arrays["hand_left_wrist_orient_in_camera"][0] = [1.0, 2.0, 3.0]
    arrays["hand_left_finger_pose"][0, 0] = [4.0, 5.0, 6.0]
    arrays["hand_left_wrist_trans_in_camera"][0] = [10.0, 0.0, 0.0]
    arrays["hand_left_scale"][0] = 2.0
    arrays["hand_left_is_valid"][:] = [True, False, True]
    arrays["camera_is_valid"][:] = [True, True, False]
    arrays["hand_right_is_valid"][:] = False
    result_dir = _write_bundle(tmp_path, arrays)
    bundle = load_result_bundle(result_dir, expected_frames=3)
    mano = FakeMano()
    sharpa = FakeSharpa()

    tracking, trajectory = arrays_from_bundle(
        bundle, mano_backend=mano, sharpa_backend=sharpa
    )

    np.testing.assert_array_equal(tracking["left_valid"], [True, False, False])
    np.testing.assert_array_equal(tracking["right_valid"], [False, False, False])
    # Fake MANO wrist x=0, posed-vertex centroid x=1: scaling by 2 gives -1.
    # Then cam_t x=10 and world translation y=5.
    np.testing.assert_allclose(tracking["left_wrist_position"][0], [9.0, 5.0, 0.0])
    assert np.isnan(tracking["left_joints_3d"][1:]).all()
    assert np.isnan(trajectory["left_wrist_position"][1:]).all()
    assert np.isnan(trajectory["left_finger_joints"][1:]).all()
    assert np.isnan(trajectory["right_finger_joints"]).all()
    left_call = next(call for call in mano.calls if call["side"] == "left")
    np.testing.assert_array_equal(left_call["global_orient"], [[1.0, -2.0, -3.0]])
    np.testing.assert_array_equal(left_call["finger_pose"][0, 0], [4.0, -5.0, -6.0])
    assert sharpa.valid is not None
    np.testing.assert_array_equal(sharpa.valid["left"], [True, False, False])


def test_camera_rotation_is_applied_to_positions_and_joint_quaternions(
    tmp_path: Path,
) -> None:
    arrays = _bundle_arrays(frame_count=1)
    arrays["hand_right_is_valid"][:] = False
    arrays["camera_to_world_transform"][0, :3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    result_dir = _write_bundle(tmp_path, arrays)
    tracking, _ = arrays_from_bundle(
        load_result_bundle(result_dir),
        mano_backend=FakeMano(),
        sharpa_backend=FakeSharpa(),
    )
    np.testing.assert_allclose(
        tracking["left_wrist_wxyz"][0],
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1e-6,
    )


def test_bundle_rejects_frame_mismatch_and_empty_learned_output(tmp_path: Path) -> None:
    result_dir = _write_bundle(tmp_path)
    with pytest.raises(UpstreamOutputError, match="source video has 4"):
        load_result_bundle(result_dir, expected_frames=4)

    arrays = _bundle_arrays()
    arrays["hand_left_is_valid"][:] = False
    arrays["hand_right_is_valid"][:] = False
    result_dir = _write_bundle(tmp_path / "empty", arrays)
    with pytest.raises(UpstreamOutputError, match="no valid WiLoR/HaMeR"):
        load_result_bundle(result_dir)


def test_preflight_reports_exact_missing_bundle_and_licensed_mano_paths(
    tmp_path: Path,
) -> None:
    source_video = tmp_path / "source.mp4"
    report = preflight(
        result_dir=tmp_path / "outputs" / "result",
        source_video=source_video,
        output_dir=tmp_path / "condition" / "tracking",
        mano_model_dir=tmp_path / "mano_v1_2",
        robot_assets_dir=tmp_path / "robot_assets",
    )
    assert report["state"] == "blocked"
    codes = {item["code"] for item in report["blockers"]}
    assert "missing_reconstruction_bundle" in codes
    assert "missing_source_video" in codes
    assert "missing_licensed_mano_model" in codes
    assert report["expected_upstream_files"] == [
        str((tmp_path / "outputs" / "result" / "result.npz").resolve()),
        str((tmp_path / "outputs" / "result" / "manifest.json").resolve()),
    ]
    assert report["mano_files"] == [
        str((tmp_path / "mano_v1_2" / "models" / "MANO_LEFT.pkl").resolve()),
        str((tmp_path / "mano_v1_2" / "models" / "MANO_RIGHT.pkl").resolve()),
    ]


def test_raw_wilor_uses_side_identity_real_intrinsics_and_taco_world(
    tmp_path: Path,
) -> None:
    json_dir, intrinsic, extrinsic = _write_raw_wilor_camera(tmp_path)
    bundle = load_wilor_json_bundle(
        json_dir,
        geometry=VideoGeometry(frame_count=2, width=100, height=50, fps=24.0),
        taco_intrinsic=intrinsic,
        taco_extrinsic=extrinsic,
    )
    assert bundle.input_mode == "raw_wilor_taco_camera"
    np.testing.assert_array_equal(bundle.arrays["hand_right_is_valid"], [True, False])
    np.testing.assert_array_equal(bundle.arrays["hand_left_is_valid"], [False, True])
    # Virtual centroid pixel=(55,35); real z=20, then unproject through TACO K.
    np.testing.assert_allclose(
        bundle.arrays["hand_right_wrist_trans_in_camera"][0],
        [3.0, 1.5, 20.0],
    )
    # The focal correction scales depth, not full xyz. Reprojection under the
    # real (anisotropic, off-center) K must retain WiLoR's virtual centroid.
    virtual_uv = np.array(
        [50.0 * 1.0 / 10.0 + 100.0 / 2.0, 50.0 * 2.0 / 10.0 + 50.0 / 2.0]
    )
    x_real, y_real, z_real = bundle.arrays[
        "hand_right_wrist_trans_in_camera"
    ][0]
    real_uv = np.array(
        [100.0 * x_real / z_real + 40.0, 200.0 * y_real / z_real + 20.0]
    )
    np.testing.assert_allclose(real_uv, virtual_uv, atol=1e-6)
    np.testing.assert_array_equal(bundle.arrays["hand_right_betas"], 0.0)
    np.testing.assert_array_equal(bundle.arrays["hand_left_betas"], 2.0)
    tracking, trajectory = arrays_from_bundle(
        bundle,
        mano_backend=FakeMano(),
        sharpa_backend=FakeSharpa(),
    )
    np.testing.assert_array_equal(tracking["right_valid"], [True, False])
    np.testing.assert_array_equal(tracking["left_valid"], [False, True])
    np.testing.assert_allclose(tracking["right_wrist_position"][0], [3.0, 1.5, 20.0])
    assert np.isnan(trajectory["left_wrist_position"][0]).all()


def test_raw_wilor_rejects_missing_frames_and_ambiguous_same_side(tmp_path: Path) -> None:
    json_dir, intrinsic, extrinsic = _write_raw_wilor_camera(tmp_path)
    (json_dir / "000001.json").unlink()
    with pytest.raises(UpstreamOutputError, match="not exactly frame-aligned"):
        load_wilor_json_bundle(
            json_dir,
            geometry=VideoGeometry(frame_count=2, width=100, height=50, fps=30.0),
            taco_intrinsic=intrinsic,
            taco_extrinsic=extrinsic,
        )

    (json_dir / "000001.json").write_text(
        json.dumps(
            [_wilor_record(is_right=True), _wilor_record(is_right=True)]
        )
    )
    with pytest.raises(UpstreamOutputError, match="stable identity is ambiguous"):
        load_wilor_json_bundle(
            json_dir,
            geometry=VideoGeometry(frame_count=2, width=100, height=50, fps=30.0),
            taco_intrinsic=intrinsic,
            taco_extrinsic=extrinsic,
        )


def test_raw_wilor_preflight_can_reach_ready_without_running_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_dir, intrinsic, extrinsic = _write_raw_wilor_camera(tmp_path)
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"placeholder")
    mano_root = tmp_path / "mano"
    (mano_root / "models").mkdir(parents=True)
    for side in ("LEFT", "RIGHT"):
        (mano_root / "models" / f"MANO_{side}.pkl").write_bytes(b"licensed-test")
    assets = tmp_path / "assets"
    (assets / "xmls" / "sharpawave").mkdir(parents=True)
    for side in ("left", "right"):
        mesh_dir = assets / "meshes" / "sharpa_wave" / side
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "mesh.STL").write_bytes(b"solid test\nendsolid test\n")
        (assets / "xmls" / "sharpawave" / f"{side}_sharpawave.xml").write_text(
            f'<mujoco><compiler meshdir="../../meshes/sharpa_wave/{side}"/>'
            '<asset><mesh name="mesh" file="mesh.STL"/></asset></mujoco>'
        )
    monkeypatch.setattr(
        video2data,
        "probe_video",
        lambda _: VideoGeometry(frame_count=2, width=100, height=50, fps=29.97),
    )
    monkeypatch.setattr(video2data, "_module_available", lambda _: True)
    report = preflight(
        wilor_json_dir=json_dir,
        source_video=source_video,
        taco_intrinsic=intrinsic,
        taco_extrinsic=extrinsic,
        output_dir=tmp_path / "output",
        mano_model_dir=mano_root,
        robot_assets_dir=assets,
    )
    assert report["state"] == "ready"
    assert report["input_mode"] == "raw_wilor_taco_camera"
    assert report["video"]["fps"] == pytest.approx(29.97)
    assert report["bundle"]["valid_frames"] == {"left": 1, "right": 1}


def test_convert_with_synthetic_backends_writes_strict_contracts_and_true_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_dir = _write_bundle(tmp_path, _bundle_arrays(frame_count=2), hand_source="hamer")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"synthetic test placeholder")
    monkeypatch.setattr(
        video2data,
        "probe_video",
        lambda _: VideoGeometry(frame_count=2, width=1920, height=1080, fps=29.97),
    )
    output_dir = tmp_path / "condition" / "tracking"
    metadata = convert(
        result_dir=result_dir,
        source_video=source_video,
        output_dir=output_dir,
        sequence_id="synthetic",
        mano_backend=FakeMano(),
        sharpa_backend=FakeSharpa(),
    )

    assert metadata["state"] == "complete"
    assert metadata["video"]["fps"] == pytest.approx(29.97)
    assert metadata["hand_pose_source"] == "hamer"
    validate_tracking_file(output_dir / "tracking.npz", expected_frames=2)
    validate_robot_trajectory_file(output_dir / "robot_trajectory.npz", expected_frames=2)
    written_metadata = json.loads((output_dir / "adapter_metadata.json").read_text())
    assert written_metadata["outputs"]["tracking"]["sha256"]
    assert not list(output_dir.glob("*.partial*"))


def test_packaged_identity_camera_fallback_requires_static_authorization(
    tmp_path: Path,
) -> None:
    result_dir = _write_bundle(tmp_path)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"]["camera_to_world_dir"] = None
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(UpstreamOutputError, match="no camera_to_world_dir"):
        load_result_bundle(result_dir)
    bundle = load_result_bundle(result_dir, allow_static_camera=True)
    assert bundle.frame_count == 3


def test_default_sequence_id_skips_generic_stage_directories(tmp_path: Path) -> None:
    assert default_sequence_id(tmp_path / "clip-a" / "outputs" / "result") == "clip-a"
    assert (
        default_sequence_id(tmp_path / "clip-b" / "v2d" / "tracking" / "wilor_raw")
        == "clip-b"
    )


def test_raw_taco_world_to_camera_is_inverted_for_position_and_orientation(
    tmp_path: Path,
) -> None:
    json_dir, intrinsic, extrinsic = _write_raw_wilor_camera(tmp_path)
    world_to_camera = np.tile(np.eye(4), (2, 1, 1))
    world_to_camera[0, :3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    world_to_camera[0, :3, 3] = [1.0, 2.0, 3.0]
    np.save(extrinsic, world_to_camera)
    bundle = load_wilor_json_bundle(
        json_dir,
        geometry=VideoGeometry(frame_count=2, width=100, height=50, fps=24.0),
        taco_intrinsic=intrinsic,
        taco_extrinsic=extrinsic,
    )
    tracking, _ = arrays_from_bundle(
        bundle, mano_backend=FakeMano(), sharpa_backend=FakeSharpa()
    )
    camera_point = np.array([3.0, 1.5, 20.0])
    expected_world = world_to_camera[0, :3, :3].T @ (
        camera_point - world_to_camera[0, :3, 3]
    )
    np.testing.assert_allclose(
        tracking["right_wrist_position"][0], expected_world, atol=1e-6
    )
    np.testing.assert_allclose(
        tracking["right_wrist_wxyz"][0],
        [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)],
        atol=1e-6,
    )


def test_camera_and_local_joint_rotations_compose_in_world_order(tmp_path: Path) -> None:
    class RotatedMano(FakeMano):
        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            rotation_x = np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
            )
            quaternion = video2data._rotation_matrices_to_wxyz(rotation_x)
            result["joints_wxyz"][:] = quaternion
            return result

    arrays = _bundle_arrays(frame_count=1)
    arrays["hand_right_is_valid"][:] = False
    rotation_z = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    arrays["camera_to_world_transform"][0, :3, :3] = rotation_z
    tracking, _ = arrays_from_bundle(
        load_result_bundle(_write_bundle(tmp_path, arrays)),
        mano_backend=RotatedMano(),
        sharpa_backend=FakeSharpa(),
    )
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    expected = video2data._rotation_matrices_to_wxyz(rotation_z @ rotation_x)
    np.testing.assert_allclose(tracking["left_joints_wxyz"][0, 0], expected, atol=1e-6)


def test_trajectory_preserves_tracking_coordinate_frame(tmp_path: Path) -> None:
    bundle = load_result_bundle(_write_bundle(tmp_path, _bundle_arrays(frame_count=1)))
    tracking, _ = arrays_from_bundle(
        bundle, mano_backend=FakeMano(), sharpa_backend=FakeSharpa()
    )
    tracking["coordinate_frame"] = np.asarray("camera")
    trajectory = trajectory_from_tracking(tracking, sharpa_backend=FakeSharpa())
    assert str(trajectory["coordinate_frame"].item()) == "camera"


def test_sharpa_rejects_large_residual_and_resets_temporal_seed() -> None:
    class FakeTorch:
        float32 = np.float32

        @staticmethod
        def as_tensor(value, **_):
            return np.asarray(value, dtype=np.float32)

    class FakeRobot:
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    class FakeKinematics:
        robot = FakeRobot()
        robot_finger_joint_names = {0: "a", 1: "b"}

        def __init__(self):
            self.seeds = []
            self.calls = 0

        def compute(self, _joints, _quats, *, source_to_robot_scale, qpos):
            assert source_to_robot_scale == pytest.approx(1.2)
            self.seeds.append(qpos.copy())
            solved = qpos.copy()
            solved[0] += 10.0
            errors = ([0.001], [0.1], [0.001])[self.calls]
            self.calls += 1
            return {
                "q": solved,
                "frame_task_errors": errors,
                "num_optimization_iterations": 3,
            }

    backend = object.__new__(ExistingSharpaBackend)
    backend._torch = FakeTorch()
    backend._device = "cpu"
    backend._wrist_pose = lambda position, quaternion: (
        np.asarray(position),
        np.asarray(quaternion)[[1, 2, 3, 0]],
    )
    kinematics = FakeKinematics()
    backend._kinematics = {"left": kinematics}
    backend.max_frame_task_error_m = 0.05
    backend.diagnostics = {}
    joints = np.zeros((3, 21, 3), dtype=np.float32)
    joints[:, 0, 0] = [1.0, 2.0, 3.0]
    quaternions = np.zeros((3, 21, 4), dtype=np.float32)
    quaternions[..., 0] = 1.0
    result = backend._retarget_side(
        "left", joints, quaternions, np.ones(3, dtype=np.bool_), 1.2
    )
    np.testing.assert_array_equal(result["valid"], [True, False, True])
    assert kinematics.seeds[0][0] == pytest.approx(1.0)
    assert kinematics.seeds[1][0] == pytest.approx(11.0)
    assert kinematics.seeds[2][0] == pytest.approx(3.0)
    assert backend.diagnostics["left"]["rejected_frames"] == 1


def test_overwrite_failure_cannot_leave_stale_complete_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_dir = _write_bundle(tmp_path, _bundle_arrays(frame_count=2))
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"synthetic")
    monkeypatch.setattr(
        video2data,
        "probe_video",
        lambda _: VideoGeometry(frame_count=2, width=100, height=50, fps=30.0),
    )
    output_dir = tmp_path / "output"
    convert(
        result_dir=result_dir,
        source_video=source_video,
        output_dir=output_dir,
        mano_backend=FakeMano(),
        sharpa_backend=FakeSharpa(),
    )
    original_replace = Path.replace

    def fail_second_artifact(self: Path, target: Path):
        if self.name.startswith(".robot_trajectory.npz") and self.name.endswith(
            ".partial.npz"
        ):
            raise OSError("injected second-artifact commit failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_second_artifact)
    with pytest.raises(OSError, match="injected"):
        convert(
            result_dir=result_dir,
            source_video=source_video,
            output_dir=output_dir,
            overwrite=True,
            mano_backend=FakeMano(),
            sharpa_backend=FakeSharpa(),
        )
    assert not (output_dir / "adapter_metadata.json").exists()
    assert not list(output_dir.glob("*.partial*"))


def test_preflight_blocks_when_camera_intersection_has_no_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = _bundle_arrays(frame_count=2)
    arrays["camera_is_valid"][:] = False
    result_dir = _write_bundle(tmp_path, arrays)
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"synthetic")
    mano_root = tmp_path / "mano"
    (mano_root / "models").mkdir(parents=True)
    for side in ("LEFT", "RIGHT"):
        (mano_root / "models" / f"MANO_{side}.pkl").write_bytes(b"test")
    monkeypatch.setattr(
        video2data,
        "probe_video",
        lambda _: VideoGeometry(frame_count=2, width=100, height=50, fps=30.0),
    )
    monkeypatch.setattr(video2data, "_module_available", lambda _: True)
    monkeypatch.setattr(video2data, "sharpa_asset_blockers", lambda _: [])
    report = preflight(
        result_dir=result_dir,
        source_video=source_video,
        output_dir=tmp_path / "output",
        mano_model_dir=mano_root,
        robot_assets_dir=tmp_path / "assets",
    )
    assert report["state"] == "blocked"
    assert "no_effective_hand_observations" in {
        item["code"] for item in report["blockers"]
    }
    assert report["bundle"]["effective_valid_frames"] == {"left": 0, "right": 0}
