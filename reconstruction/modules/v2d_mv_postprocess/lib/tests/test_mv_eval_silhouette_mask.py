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


def test_frame_metric_perfect_match_has_zero_unexplained_ratio():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    result = sm.compute_silhouette_mask_frame_metrics(mask, mask > 127)

    assert result["skipped"] is False
    assert result["unexplained_sam2_ratio"] == 0.0
    assert result["over_render_ratio"] == 0.0
    assert result["sam2_bbox"] == (2, 2, 8, 8)
    assert result["render_bbox"] == (2, 2, 8, 8)
    assert result["bbox_intersection_pixels"] == 36
    assert result["sam2_bbox_pixels"] == 36
    assert result["render_bbox_pixels"] == 36
    assert result["sam2_bbox_in_render_bbox_ratio"] == 1.0

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
