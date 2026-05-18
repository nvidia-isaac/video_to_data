import sys
from pathlib import Path

import numpy as np
import pytest

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

try:
    import mv_render_hoi_overlay
except ImportError as exc:
    pytest.skip(f"mv_render_hoi_overlay dependencies unavailable: {exc}", allow_module_level=True)


class FakeSource:
    n_frames = 2
    image_size = (64, 64)

    def __init__(self):
        self.closed = False

    def iter_frames(self):
        yield np.zeros((64, 64, 3), dtype=np.uint8)
        yield np.ones((64, 64, 3), dtype=np.uint8) * 20

    def close(self):
        self.closed = True


class FakeFrameSource:
    source = FakeSource()

    @classmethod
    def from_path(cls, path):
        return cls.source


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write_frame(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


class FakeRenderer:
    instances = []

    def __init__(self, image_size):
        self.image_size = image_size
        self.added_meshes = []
        self.pose_updates = []
        self.render_meshes = []
        FakeRenderer.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def add_persistent_mesh(self, mesh, pose=None):
        self.added_meshes.append((mesh, pose.copy()))
        return 1

    def set_persistent_mesh_pose(self, handle, pose):
        self.pose_updates.append((handle, pose.copy()))

    def render_overlay(self, meshes, K, T, image):
        self.render_meshes.append(meshes)
        return np.zeros_like(image, dtype=np.float32)


class ObjectMeshThatMustNotBeCopied:
    def copy(self):
        raise AssertionError("object mesh should be persistent, not copied per frame")


class FakeCfg:
    cameras = [0, 1]
    rgb_path_template = "{cam_name}.h5"
    output_dir = "/tmp/out"
    object_mesh_path = "/tmp/object.glb"
    object_pose_path = "/tmp/poses.npy"
    mhr_mesh_mv_path = "/tmp/mhr_mesh_mv.pt"

    def get(self, key, default=None):
        return default


class FakeCamParam:
    K = np.eye(3)
    T = np.eye(4)


class FakeCam:
    def __init__(self, name):
        self.name = name
        self.param = FakeCamParam()


class FakeRig:
    def get_camera(self, cam_id):
        return FakeCam(f"cam_{cam_id}")


def test_render_hoi_overlay_uses_persistent_object_mesh(monkeypatch, tmp_path):
    writer = FakeWriter()
    FakeFrameSource.source = FakeSource()
    FakeRenderer.instances = []
    monkeypatch.setattr(mv_render_hoi_overlay, "FrameSource", FakeFrameSource)
    monkeypatch.setattr(mv_render_hoi_overlay, "Renderer", FakeRenderer)
    monkeypatch.setattr(mv_render_hoi_overlay, "get_video_writer", lambda *args, **kwargs: writer)

    object_poses = np.stack([np.eye(4), np.eye(4)])
    object_poses[1, 0, 3] = 3.0

    mv_render_hoi_overlay.render_hoi_overlay(
        rgb_path=tmp_path / "rgb.h5",
        output_path=tmp_path / "front_hoi_overlay.mp4",
        object_mesh=ObjectMeshThatMustNotBeCopied(),
        object_poses=object_poses,
        human_vertices=np.zeros((2, 4, 3), dtype=np.float32),
        human_faces=np.array([[0, 1, 2]], dtype=np.int64),
        cam_intrinsics=np.eye(3),
        cam_extrinsics=np.eye(4),
    )

    renderer = FakeRenderer.instances[0]
    assert len(renderer.added_meshes) == 1
    assert np.array_equal(renderer.added_meshes[0][1], object_poses[0])
    assert [handle for handle, _ in renderer.pose_updates] == [1, 1]
    assert np.array_equal(renderer.pose_updates[0][1], object_poses[0])
    assert np.array_equal(renderer.pose_updates[1][1], object_poses[1])
    assert [len(meshes) for meshes in renderer.render_meshes] == [1, 1]
    assert len(writer.frames) == 2
    assert writer.closed
    assert FakeFrameSource.source.closed


def test_render_hoi_overlay_prints_sparse_progress_without_tqdm(monkeypatch, tmp_path, capsys):
    writer = FakeWriter()
    FakeFrameSource.source = FakeSource()
    FakeRenderer.instances = []
    monkeypatch.setattr(mv_render_hoi_overlay, "FrameSource", FakeFrameSource)
    monkeypatch.setattr(mv_render_hoi_overlay, "Renderer", FakeRenderer)
    monkeypatch.setattr(mv_render_hoi_overlay, "get_video_writer", lambda *args, **kwargs: writer)

    object_poses = np.stack([np.eye(4), np.eye(4)])

    mv_render_hoi_overlay.render_hoi_overlay(
        rgb_path=tmp_path / "rgb.h5",
        output_path=tmp_path / "front_hoi_overlay.mp4",
        object_mesh=ObjectMeshThatMustNotBeCopied(),
        object_poses=object_poses,
        human_vertices=np.zeros((2, 4, 3), dtype=np.float32),
        human_faces=np.array([[0, 1, 2]], dtype=np.int64),
        cam_intrinsics=np.eye(3),
        cam_extrinsics=np.eye(4),
        show_progress=False,
        progress_interval=0.5,
    )

    output = capsys.readouterr().out
    assert "Rendering HOI overlay front_hoi_overlay: 1/2 (50%), " in output
    assert "Rendering HOI overlay front_hoi_overlay: 2/2 (100%), " in output
    assert "it/s" in output


def test_camera_jobs_can_disable_progress_for_parallel_workers():
    jobs = mv_render_hoi_overlay._build_camera_render_jobs(
        FakeCfg(),
        FakeRig(),
        show_progress=False,
        progress_interval=0.25,
    )

    assert [job.cam_name for job in jobs] == ["cam_0", "cam_1"]
    assert [job.show_progress for job in jobs] == [False, False]
    assert [job.progress_interval for job in jobs] == [0.25, 0.25]


def test_parallel_camera_jobs_preserve_order(monkeypatch, tmp_path):
    submitted = []
    max_workers_seen = []

    def fake_worker(job):
        submitted.append(job.cam_name)
        return mv_render_hoi_overlay.CameraRenderResult(
            cam_name=job.cam_name,
            output_path=job.output_path,
            timings={},
        )

    class FakeExecutor:
        def __init__(self, max_workers):
            max_workers_seen.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def map(self, fn, jobs, chunksize=1):
            assert chunksize == 1
            return [fn(job) for job in jobs]

    monkeypatch.setattr(mv_render_hoi_overlay, "_render_camera_overlay_worker", fake_worker)
    monkeypatch.setattr(mv_render_hoi_overlay, "ProcessPoolExecutor", FakeExecutor)

    jobs = [
        mv_render_hoi_overlay.CameraRenderJob(
            cam_name=f"cam_{idx}",
            rgb_path=tmp_path / f"cam_{idx}.h5",
            output_path=tmp_path / f"cam_{idx}_hoi_overlay.mp4",
            object_mesh_path=tmp_path / "object.glb",
            object_pose_path=tmp_path / "poses.npy",
            mhr_mesh_mv_path=tmp_path / "mhr_mesh_mv.pt",
            cam_intrinsics=np.eye(3),
            cam_extrinsics=np.eye(4),
        )
        for idx in range(4)
    ]

    results = mv_render_hoi_overlay._run_camera_render_jobs(jobs, render_workers=4)

    assert max_workers_seen == [4]
    assert submitted == ["cam_0", "cam_1", "cam_2", "cam_3"]
    assert [result.cam_name for result in results] == submitted
