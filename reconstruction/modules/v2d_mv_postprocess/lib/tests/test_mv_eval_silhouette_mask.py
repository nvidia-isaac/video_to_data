import sys
from pathlib import Path

import numpy as np
import pytest

LIB_DIR = Path(__file__).resolve().parents[1]
DOCKER_DIR = LIB_DIR.parent / "docker"
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(DOCKER_DIR))

import mv_eval_silhouette_mask as sm
import run_mv_eval_silhouette_mask_human as human_wrapper
import run_mv_eval_silhouette_mask_object as object_wrapper


class FakeFrameSource:
    n_frames = 2
    image_size = (8, 8)
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    instances = []

    def __init__(self, path):
        self.path = path
        self.closed = False
        FakeFrameSource.instances.append(self)

    @classmethod
    def from_path(cls, path):
        return cls(path)

    def __getitem__(self, idx):
        return self.mask.copy()

    def close(self):
        self.closed = True


class FakeRenderer:
    instances = []

    def __init__(self, image_size):
        self.image_size = image_size
        self.add_calls = []
        self.pose_updates = []
        self.render_depth_meshes = []
        FakeRenderer.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def add_persistent_mesh(self, mesh, pose=None):
        self.add_calls.append((mesh, pose.copy()))
        return 11

    def set_persistent_mesh_pose(self, handle, pose):
        self.pose_updates.append((handle, pose.copy()))

    def render_depth(self, meshes, K, T):
        self.render_depth_meshes.append(list(meshes))
        return np.ones(self.image_size[::-1], dtype=np.float32)


def _patch_runtime(monkeypatch):
    FakeFrameSource.instances = []
    FakeRenderer.instances = []
    monkeypatch.setattr(sm, "FrameSource", FakeFrameSource)
    monkeypatch.setattr(sm, "Renderer", FakeRenderer)


def _verts():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _faces():
    return np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )


def test_bbox_containment_helpers():
    assert sm.sam2_bbox_in_render_bbox_ratio((2, 2, 8, 8), (2, 2, 8, 8)) == 1.0
    assert sm.sam2_bbox_in_render_bbox_ratio((4, 4, 8, 8), (2, 2, 10, 10)) == 1.0
    assert sm.sam2_bbox_in_render_bbox_ratio((2, 2, 8, 8), (5, 2, 8, 8)) == 0.5
    assert sm.sam2_bbox_in_render_bbox_ratio((2, 2, 8, 8), None) == 0.0
    assert sm.pad_bbox((2, 2, 8, 8), 8, (10, 10)) == (0, 0, 10, 10)
    assert sm.pad_bbox(
        (100, 100, 300, 200), 8, (500, 500), 0.10,
    ) == (80, 90, 320, 210)
    assert sm.pad_bbox(
        (100, 100, 120, 120), 8, (500, 500), 0.10,
    ) == (92, 92, 128, 128)
    assert sm.pad_bbox(None, 8, (10, 10)) is None


def test_bbox_component_filter_ignores_one_and_two_pixel_specks():
    sam2 = np.zeros((12, 12), dtype=np.uint8)
    sam2[3:6, 3:6] = 255
    sam2[0, 11] = 255
    sam2[11, 0:2] = 255
    rendered = np.zeros((12, 12), dtype=bool)
    rendered[3:6, 3:6] = True

    result = sm.compute_silhouette_mask_frame_metrics(
        sam2,
        rendered,
        min_mask_pixels=10,
        min_bbox_component_pixels=3,
    )

    assert result["sam2_bbox"] == (0, 0, 12, 12)
    assert result["filtered_sam2_bbox"] == (3, 3, 6, 6)
    assert result["removed_sam2_bbox_component_count"] == 2
    assert result["removed_sam2_bbox_component_pixels"] == 3
    assert result["filtered_sam2_bbox_in_padded_render_bbox_ratio"] == 1.0


def test_bbox_component_filter_retains_multiple_legitimate_components():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1, 1:4] = 255
    mask[9:12, 9] = 255

    filtered, stats = sm.filter_mask_components_for_bbox(mask, 3)

    assert np.array_equal(filtered, mask > 0)
    assert sm.mask_to_bbox(filtered) == (1, 1, 10, 12)
    assert stats["filtered_sam2_bbox_component_count"] == 2
    assert stats["removed_sam2_bbox_component_count"] == 0


def test_bbox_component_filter_removes_tiny_component_relative_to_largest():
    mask = np.zeros((140, 140), dtype=np.uint8)
    mask[10:110, 10:110] = 255
    mask[130, 130:137] = 255

    filtered, stats = sm.filter_mask_components_for_bbox(
        mask,
        3,
        min_component_fraction_of_largest=0.001,
    )

    assert sm.mask_to_bbox(filtered) == (10, 10, 110, 110)
    assert stats["largest_sam2_bbox_component_pixels"] == 10_000
    assert stats["effective_min_sam2_bbox_component_pixels"] == 10
    assert stats["removed_sam2_bbox_component_count"] == 1
    assert stats["removed_sam2_bbox_component_pixels"] == 7


def test_bbox_component_filter_keeps_substantial_relative_component():
    mask = np.zeros((140, 140), dtype=np.uint8)
    mask[10:110, 10:110] = 255
    mask[125:130, 130:134] = 255

    filtered, stats = sm.filter_mask_components_for_bbox(
        mask,
        3,
        min_component_fraction_of_largest=0.001,
    )

    assert np.array_equal(filtered, mask > 0)
    assert stats["effective_min_sam2_bbox_component_pixels"] == 10
    assert stats["filtered_sam2_bbox_component_count"] == 2


def test_bbox_component_filter_uses_eight_connectivity():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = 255

    filtered, stats = sm.filter_mask_components_for_bbox(mask, 3)

    assert np.array_equal(filtered, mask > 0)
    assert stats["sam2_bbox_component_count"] == 1
    assert stats["filtered_sam2_bbox_component_count"] == 1


def test_substantial_disconnected_component_remains_a_bbox_failure():
    sam2 = np.zeros((20, 20), dtype=np.uint8)
    sam2[5:10, 5:10] = 255
    sam2[15:18, 15:18] = 255
    rendered = np.zeros((20, 20), dtype=bool)
    rendered[5:10, 5:10] = True

    result = sm.compute_silhouette_mask_frame_metrics(
        sam2,
        rendered,
        min_bbox_component_pixels=3,
    )

    assert result["filtered_sam2_bbox"] == (5, 5, 18, 18)
    assert result["filtered_sam2_bbox_in_padded_render_bbox_ratio"] < 0.8


def test_bbox_filter_does_not_change_residual_metrics_or_source_mask():
    sam2 = np.zeros((12, 12), dtype=np.uint8)
    sam2[3:7, 3:7] = 255
    sam2[0, 11] = 255
    original = sam2.copy()
    rendered = np.zeros((12, 12), dtype=bool)
    rendered[3:7, 3:7] = True

    unfiltered = sm.compute_silhouette_mask_frame_metrics(
        sam2, rendered, min_bbox_component_pixels=1
    )
    filtered = sm.compute_silhouette_mask_frame_metrics(
        sam2, rendered, min_bbox_component_pixels=3
    )

    assert np.array_equal(sam2, original)
    for key in (
        "unexplained_sam2_pixels",
        "unexplained_sam2_eroded_pixels",
        "unexplained_sam2_ratio",
        "over_render_pixels",
        "over_render_eroded_pixels",
        "over_render_ratio",
    ):
        assert filtered[key] == unfiltered[key]


def test_all_bbox_components_filtered_is_bad_but_eligible():
    sam2 = np.zeros((15, 15), dtype=np.uint8)
    for y, x in ((1, 1), (1, 6), (1, 11), (6, 1), (6, 6)):
        sam2[y, x : x + 2] = 255
    rendered = np.ones((15, 15), dtype=bool)

    result = sm.compute_silhouette_mask_frame_metrics(
        sam2,
        rendered,
        min_mask_pixels=10,
        min_bbox_component_pixels=3,
    )

    assert result["skipped"] is False
    assert result["filtered_sam2_bbox"] is None
    assert result["filtered_sam2_bbox_in_padded_render_bbox_ratio"] == 0.0
    assert result["removed_sam2_bbox_component_pixels"] == 10


def test_missing_render_bbox_is_a_containment_failure():
    sam2 = np.zeros((12, 12), dtype=np.uint8)
    sam2[3:7, 3:7] = 255

    result = sm.compute_silhouette_mask_frame_metrics(
        sam2,
        np.zeros((12, 12), dtype=bool),
        min_bbox_component_pixels=3,
        render_bbox_padding_pixels=8,
    )

    assert result["padded_render_bbox"] is None
    assert result["filtered_sam2_bbox_in_padded_render_bbox_ratio"] == 0.0


def test_debug_canvas_uses_rgb_colors_and_compares_filtered_to_padded_bbox():
    sam2 = np.zeros((40, 40), dtype=np.uint8)
    sam2[10:20, 10:20] = 255
    sam2[1, 38] = 255
    rendered = np.zeros((40, 40), dtype=bool)
    rendered[11:19, 11:19] = True
    metric = sm.compute_silhouette_mask_frame_metrics(
        sam2,
        rendered,
        min_bbox_component_pixels=3,
        render_bbox_padding_pixels=8,
    )

    canvas = sm._debug_canvas(sam2 > 0, rendered, metric)

    top_left = canvas[:20, :20]
    comparison = canvas[20:, :20]
    assert np.any(np.all(top_left == (0, 255, 255), axis=2))
    assert not np.any(np.all(top_left == (255, 255, 0), axis=2))
    assert np.any(np.all(comparison == (255, 0, 255), axis=2))
    assert np.any(np.all(comparison == (255, 165, 0), axis=2))
    assert np.any(np.all(comparison == (0, 220, 0), axis=2))


def test_frame_metric_perfect_match_has_zero_unexplained_ratio():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    result = sm.compute_silhouette_mask_frame_metrics(mask, mask > 127)

    assert result["skipped"] is False
    assert result["unexplained_sam2_ratio"] == 0.0
    assert result["over_render_ratio"] == 0.0
    assert result["sam2_bbox"] == (2, 2, 8, 8)
    assert result["render_bbox"] == (2, 2, 8, 8)
    assert result["filtered_sam2_bbox"] == (2, 2, 8, 8)
    assert result["padded_render_bbox"] == (2, 2, 8, 8)
    assert result["bbox_intersection_pixels"] == 36
    assert result["sam2_bbox_pixels"] == 36
    assert result["render_bbox_pixels"] == 36
    assert result["sam2_bbox_in_render_bbox_ratio"] == 1.0
    assert result["filtered_sam2_bbox_in_padded_render_bbox_ratio"] == 1.0

    binary_mask = (mask > 127).astype(np.uint8)
    binary_result = sm.compute_silhouette_mask_frame_metrics(binary_mask, binary_mask)
    assert binary_result["skipped"] is False
    assert binary_result["unexplained_sam2_ratio"] == 0.0


def test_frame_metric_shifted_render_has_unexplained_residual():
    sam2 = np.zeros((12, 12), dtype=np.uint8)
    sam2[2:10, 2:10] = 255
    rendered = np.zeros((12, 12), dtype=bool)
    rendered[2:10, 5:12] = True

    result = sm.compute_silhouette_mask_frame_metrics(sam2, rendered)

    assert result["unexplained_sam2_eroded_pixels"] > 0
    assert result["unexplained_sam2_ratio"] > 0


def test_frame_metric_erodes_one_pixel_sam2_residual_strip():
    sam2 = np.zeros((10, 10), dtype=np.uint8)
    sam2[2:8, 2:8] = 255
    rendered = np.zeros((10, 10), dtype=bool)
    rendered[2:8, 3:8] = True

    result = sm.compute_silhouette_mask_frame_metrics(sam2, rendered)

    assert result["unexplained_sam2_pixels"] == 6
    assert result["unexplained_sam2_eroded_pixels"] == 0
    assert result["unexplained_sam2_ratio"] == 0.0


def test_frame_metric_skips_tiny_sam2_mask():
    sam2 = np.zeros((5, 5), dtype=np.uint8)
    sam2[1, 1] = 255
    rendered = np.ones((5, 5), dtype=bool)

    result = sm.compute_silhouette_mask_frame_metrics(
        sam2,
        rendered,
        min_mask_pixels=10,
    )

    assert result["skipped"] is True
    assert result["reason"] == "sam2_mask_pixels<10"
    assert result["sam2_bbox_in_render_bbox_ratio"] is None
    assert result["filtered_sam2_bbox_in_padded_render_bbox_ratio"] is None


def test_frame_metric_reports_over_render_separately():
    sam2 = np.zeros((12, 12), dtype=np.uint8)
    sam2[4:8, 4:8] = 255
    rendered = np.ones((12, 12), dtype=bool)

    result = sm.compute_silhouette_mask_frame_metrics(sam2, rendered)

    assert result["unexplained_sam2_ratio"] == 0.0
    assert result["over_render_eroded_pixels"] > 0
    assert result["over_render_ratio"] > 0


def test_summary_includes_bbox_containment_stats():
    sam2 = np.zeros((10, 10), dtype=np.uint8)
    sam2[2:8, 2:8] = 255
    matched = sam2 > 127
    partial = np.zeros((10, 10), dtype=bool)
    partial[2:8, 5:8] = True

    summary = sm._summarize_frame_metrics(
        [
            sm.compute_silhouette_mask_frame_metrics(sam2, matched),
            sm.compute_silhouette_mask_frame_metrics(sam2, partial),
        ]
    )

    assert summary["mean_sam2_bbox_in_render_bbox_ratio"] == 0.75
    assert summary["median_sam2_bbox_in_render_bbox_ratio"] == 0.75
    assert summary["total_sam2_bbox_in_render_bbox_ratio"] == 0.75
    assert summary["total_sam2_bbox_pixels"] == 72
    assert summary["total_bbox_intersection_pixels"] == 54


def _alignment_metric(frame_idx, containment=None, unexplained_ratio=0.0):
    if containment is None:
        return {
            "frame_idx": frame_idx,
            "skipped": True,
            "reason": "sam2_mask_pixels<10",
            "sam2_mask_pixels": 9,
        }
    return {
        "frame_idx": frame_idx,
        "skipped": False,
        "sam2_mask_pixels": 10,
        "unexplained_sam2_ratio": unexplained_ratio,
        "sam2_bbox_in_render_bbox_ratio": containment,
        "filtered_sam2_bbox_in_padded_render_bbox_ratio": containment,
    }


def _alignment_result(cam_name, containments):
    frame_metrics = [
        _alignment_metric(frame_idx, containment)
        for frame_idx, containment in enumerate(containments)
    ]
    return sm.SilhouetteMaskCameraResult(
        cam_name=cam_name,
        metrics={},
        frame_metrics=frame_metrics,
        timings={},
    )


def test_object_alignment_minimum_pixel_boundary():
    sam2 = np.zeros((10, 10), dtype=np.uint8)
    sam2.reshape(-1)[:10] = 255
    rendered = sam2 > 127

    eligible = sm.compute_silhouette_mask_frame_metrics(
        sam2,
        rendered,
        min_mask_pixels=10,
    )
    sam2.reshape(-1)[9] = 0
    skipped = sm.compute_silhouette_mask_frame_metrics(
        sam2,
        rendered,
        min_mask_pixels=10,
    )

    assert eligible["skipped"] is False
    assert skipped["skipped"] is True


def test_object_alignment_summary_retains_bad_run_diagnostics():
    summary = sm._summarize_object_alignment_camera(
        [
            _alignment_metric(0, 0.80, unexplained_ratio=1.0),
            *[
                _alignment_metric(frame_idx, 0.7999, unexplained_ratio=0.0)
                for frame_idx in range(1, 6)
            ],
        ],
        min_bbox_containment=0.80,
    )

    assert summary["eligible_frames"] == 6
    assert summary["bad_frames"] == 5
    assert summary["longest_bad_run"] == 5
    assert summary["bad_spans"] == [
        {
            "start_frame": 1,
            "end_frame": 5,
            "length": 5,
            "min_filtered_sam2_bbox_in_padded_render_bbox_ratio": 0.7999,
            "max_unexplained_sam2_ratio": 0.0,
        }
    ]
    assert summary[
        "min_filtered_sam2_bbox_in_padded_render_bbox_ratio"
    ] == 0.7999
    assert summary["max_unexplained_sam2_ratio"] == 1.0


def test_object_alignment_good_and_ineligible_frames_break_bad_runs():
    summary = sm._summarize_object_alignment_camera(
        [
            *[_alignment_metric(frame_idx, 0.7) for frame_idx in range(4)],
            _alignment_metric(4, 0.8),
            *[_alignment_metric(frame_idx, 0.7) for frame_idx in range(5, 9)],
            _alignment_metric(9, None),
            *[_alignment_metric(frame_idx, 0.7) for frame_idx in range(10, 14)],
        ],
        min_bbox_containment=0.80,
    )

    assert summary["bad_frames"] == 12
    assert summary["longest_bad_run"] == 4
    assert [span["length"] for span in summary["bad_spans"]] == [4, 4, 4]


def test_object_alignment_gate_fails_any_camera_above_bad_fraction():
    gate = sm._build_object_alignment_quality_gate(
        [
            _alignment_result("front", [0.9] * 18 + [0.7, 0.7]),
            _alignment_result("side", [0.9] * 20),
        ],
        min_bbox_containment=0.80,
        min_segment_pixels=10,
        min_bbox_component_pixels=3,
        render_bbox_padding_pixels=8,
        max_bad_frame_fraction=0.05,
    )

    assert gate["status"] == "FAIL"
    assert gate["eligible_camera_frames"] == 40
    assert gate["bad_camera_frames"] == 2
    assert gate["failing_cameras"] == ["front"]
    assert gate["metric"] == "filtered_sam2_bbox_in_padded_render_bbox_ratio"
    assert gate["diagnostic_metrics"] == [
        "sam2_bbox_in_render_bbox_ratio",
        "unexplained_sam2_ratio",
        "over_render_ratio",
    ]
    assert gate["per_camera"]["front"]["min_sam2_bbox_in_render_bbox_ratio"] == 0.7
    assert gate["per_camera"]["front"]["max_unexplained_sam2_ratio"] == 0.0
    assert gate["per_camera"]["front"]["longest_bad_run"] == 2
    assert gate["thresholds"] == {
        "min_object_silhouette_bbox_containment": 0.8,
        "min_object_segment_pixels": 10,
        "min_object_bbox_component_pixels": 3,
        "min_object_bbox_component_fraction_of_largest": 0.0,
        "object_silhouette_render_bbox_padding_pixels": 8,
        "object_silhouette_render_bbox_padding_fraction": 0.0,
        "max_object_silhouette_bad_frame_fraction": 0.05,
    }


def test_object_alignment_exact_five_percent_passes_and_distributed_failures_fail():
    exact_boundary = sm._build_object_alignment_quality_gate(
        [_alignment_result("front", [0.7] + [0.9] * 19)],
        min_bbox_containment=0.8,
        min_segment_pixels=10,
        min_bbox_component_pixels=3,
        render_bbox_padding_pixels=8,
        max_bad_frame_fraction=0.05,
    )
    distributed = sm._build_object_alignment_quality_gate(
        [_alignment_result("front", [0.7, *([0.9] * 9), 0.7, *([0.9] * 9)])],
        min_bbox_containment=0.8,
        min_segment_pixels=10,
        min_bbox_component_pixels=3,
        render_bbox_padding_pixels=8,
        max_bad_frame_fraction=0.05,
    )

    assert exact_boundary["status"] == "PASS"
    assert exact_boundary["per_camera"]["front"]["longest_bad_run"] == 1
    assert distributed["status"] == "FAIL"
    assert distributed["per_camera"]["front"]["longest_bad_run"] == 1


def test_long_bad_run_below_fraction_does_not_fail_independently():
    gate = sm._build_object_alignment_quality_gate(
        [_alignment_result("front", [0.7] * 4 + [0.9] * 96)],
        min_bbox_containment=0.8,
        min_segment_pixels=10,
        min_bbox_component_pixels=3,
        render_bbox_padding_pixels=8,
        max_bad_frame_fraction=0.05,
    )

    assert gate["status"] == "PASS"
    assert gate["per_camera"]["front"]["longest_bad_run"] == 4


def test_object_alignment_gate_is_inconclusive_without_eligible_frames():
    gate = sm._build_object_alignment_quality_gate(
        [_alignment_result("front", [None, None])],
        min_bbox_containment=0.80,
        min_segment_pixels=10,
        min_bbox_component_pixels=3,
        render_bbox_padding_pixels=8,
        max_bad_frame_fraction=0.05,
    )

    assert gate["status"] == "INCONCLUSIVE"
    assert gate["eligible_camera_frames"] == 0
    assert gate["failing_cameras"] == []


def test_object_metrics_output_keeps_gate_and_per_frame_diagnostics(tmp_path):
    frame_metrics = [_alignment_metric(0, None)]
    result = sm.SilhouetteMaskCameraResult(
        cam_name="front",
        metrics=sm._summarize_frame_metrics(frame_metrics),
        frame_metrics=frame_metrics,
        timings={},
    )
    gate = sm._build_object_alignment_quality_gate(
        [result],
        min_bbox_containment=0.80,
        min_segment_pixels=10,
        min_bbox_component_pixels=3,
        render_bbox_padding_pixels=8,
        max_bad_frame_fraction=0.05,
    )

    metrics = sm._write_metrics(
        cam_names=["front"],
        results=[result],
        output_path=tmp_path / "silhouette_mask_metrics.json",
        debug=0,
        vis_dir=None,
        tile_shape=(1, 1),
        tile_image_size=None,
        object_quality_gate=gate,
    )

    assert metrics["quality_gate"]["status"] == "INCONCLUSIVE"
    assert metrics["frame_metrics"] == {"front": frame_metrics}


def test_rigid_object_path_uses_persistent_mesh_and_empty_dynamic_render(monkeypatch):
    _patch_runtime(monkeypatch)
    poses = np.stack([np.eye(4), np.eye(4)])
    job = sm.RigidSilhouetteMaskCameraJob(
        cam_name="front",
        cam_intrinsics=np.eye(3),
        cam_extrinsics=np.eye(4),
        mask_dir=Path("mask"),
        canonical_verts=_verts(),
        faces=_faces(),
        poses=poses,
        eval_image_size=None,
        erosion_kernel=3,
        erosion_iterations=1,
        min_mask_pixels=10,
        min_bbox_component_pixels=3,
        min_bbox_component_fraction_of_largest=0.001,
        render_bbox_padding_pixels=8,
        debug=0,
        vis_dir=None,
    )

    result = sm._eval_rigid_silhouette_mask_camera(job)

    renderer = FakeRenderer.instances[0]
    assert len(renderer.add_calls) == 1
    assert [handle for handle, _ in renderer.pose_updates] == [11, 11]
    assert renderer.render_depth_meshes == [[], []]
    assert result.metrics["frames_evaluated"] == 2
    assert result.metrics["median_unexplained_sam2_ratio"] == 0.0


def test_human_path_builds_per_frame_dynamic_mesh(monkeypatch):
    _patch_runtime(monkeypatch)
    mesh_verts = np.stack([_verts(), _verts()])
    job = sm.SilhouetteMaskCameraJob(
        cam_name="front",
        cam_intrinsics=np.eye(3),
        cam_extrinsics=np.eye(4),
        mask_dir=Path("mask"),
        faces=_faces(),
        mesh_verts=mesh_verts,
        eval_image_size=None,
        erosion_kernel=3,
        erosion_iterations=1,
        min_mask_pixels=10,
        debug=0,
        vis_dir=None,
    )

    result = sm._eval_silhouette_mask_camera(job)

    renderer = FakeRenderer.instances[0]
    assert renderer.add_calls == []
    assert [len(meshes) for meshes in renderer.render_depth_meshes] == [1, 1]
    assert result.metrics["frames_evaluated"] == 2


def test_frame_count_mismatch_raises_clear_error(monkeypatch):
    _patch_runtime(monkeypatch)
    FakeFrameSource.n_frames = 1
    job = sm.SilhouetteMaskCameraJob(
        cam_name="front",
        cam_intrinsics=np.eye(3),
        cam_extrinsics=np.eye(4),
        mask_dir=Path("mask"),
        faces=_faces(),
        mesh_verts=np.stack([_verts(), _verts()]),
        eval_image_size=None,
        erosion_kernel=3,
        erosion_iterations=1,
        min_mask_pixels=10,
        debug=0,
        vis_dir=None,
    )

    with pytest.raises(ValueError, match="frame count mismatch"):
        sm._eval_silhouette_mask_camera(job)

    FakeFrameSource.n_frames = 2


def test_docker_object_wrapper_passes_expected_inputs(monkeypatch):
    calls = []
    monkeypatch.setattr(object_wrapper, "run_in_container", lambda **kwargs: calls.append(kwargs))

    object_wrapper.run_mv_eval_silhouette_mask_object(
        camera_params_path="/cam",
        object_mesh_path="/mesh.glb",
        object_pose_dir="/poses",
        output_dir="/out",
        mask_dir="/masks",
        config_path="/config.yaml",
        dev=True,
    )

    call = calls[0]
    assert call["module"] == "v2d.mv.postprocess.lib.mv_eval_silhouette_mask_object"
    assert call["gpus"] is True
    assert call["inputs"] == {
        "camera_params_path": "/cam",
        "object_mesh_path": "/mesh.glb",
        "object_pose_dir": "/poses",
        "config_path": "/config.yaml",
        "mask_dir": "/masks",
    }
    assert "depth_dir" not in call["inputs"]


def test_docker_human_wrapper_passes_expected_inputs(monkeypatch):
    calls = []
    monkeypatch.setattr(human_wrapper, "run_in_container", lambda **kwargs: calls.append(kwargs))

    human_wrapper.run_mv_eval_silhouette_mask_human(
        camera_params_path="/cam",
        human_pose_dir="/human",
        output_dir="/out",
        mask_dir="/masks",
        config_path="/config.yaml",
        dev=True,
    )

    call = calls[0]
    assert call["module"] == "v2d.mv.postprocess.lib.mv_eval_silhouette_mask_human"
    assert call["gpus"] is True
    assert call["inputs"] == {
        "camera_params_path": "/cam",
        "human_pose_dir": "/human",
        "config_path": "/config.yaml",
        "mask_dir": "/masks",
    }
    assert "depth_dir" not in call["inputs"]
