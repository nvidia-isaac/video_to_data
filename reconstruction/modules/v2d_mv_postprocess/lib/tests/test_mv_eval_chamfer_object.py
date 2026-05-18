import sys
from pathlib import Path

import numpy as np

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

import mv_eval_chamfer


class FakeDepthImage:
    def __init__(self, depth):
        self.depth = depth.astype(np.float32)

    @staticmethod
    def from_array(arr):
        return FakeDepthImage(arr)


class FakeFrameSource:
    n_frames = 2

    def __init__(self, kind):
        self.kind = kind
        self.closed = False

    @classmethod
    def from_path(cls, path):
        return cls("mask" if "mask" in str(path) else "depth")

    def __getitem__(self, idx):
        if self.kind == "mask":
            return np.full((4, 4), 255, dtype=np.uint8)
        return np.ones((4, 4), dtype=np.float32)

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
        return 7

    def set_persistent_mesh_pose(self, handle, pose):
        self.pose_updates.append((handle, pose.copy()))

    def render_depth(self, meshes, K, T):
        self.render_depth_meshes.append(list(meshes))
        return np.ones(self.image_size[::-1], dtype=np.float32)


class FakeExecutor:
    max_workers_seen = []

    def __init__(self, max_workers):
        FakeExecutor.max_workers_seen.append(max_workers)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def map(self, fn, jobs, chunksize=1):
        assert chunksize == 1
        return [fn(job) for job in jobs]


def _patch_metric_dependencies(monkeypatch):
    FakeRenderer.instances = []
    monkeypatch.setattr(mv_eval_chamfer, "DepthImage", FakeDepthImage)
    monkeypatch.setattr(mv_eval_chamfer, "FrameSource", FakeFrameSource)
    monkeypatch.setattr(mv_eval_chamfer, "Renderer", FakeRenderer)
    monkeypatch.setattr(
        mv_eval_chamfer,
        "depth_to_xyz",
        lambda depth, K, T, mask=None: np.array(
            [[float(i), 0.0, 0.0] for i in range(10)],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        mv_eval_chamfer,
        "visible_vertices",
        lambda verts, mesh_zbuf, K, T: np.ones(verts.shape[0], dtype=bool),
    )
    monkeypatch.setattr(
        mv_eval_chamfer,
        "xyz_to_uv",
        lambda verts, K, T, image_size: (
            np.zeros((verts.shape[0], 2), dtype=np.int64),
            np.ones(verts.shape[0], dtype=bool),
        ),
    )


def _canonical_verts():
    return np.array([[float(i), 0.0, 0.0] for i in range(10)], dtype=np.float32)


def _poses():
    poses = np.stack([np.eye(4), np.eye(4)])
    poses[1, 0, 3] = 1.0
    return poses


def _job(show_progress=True):
    return mv_eval_chamfer.RigidChamferCameraJob(
        cam_name="front",
        cam_intrinsics=np.eye(3),
        cam_extrinsics=np.eye(4),
        depth_dir=Path("depth"),
        mask_dir=Path("mask"),
        canonical_verts=_canonical_verts(),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        poses=_poses(),
        eval_image_size=None,
        anomaly_median_mm=1e9,
        anomaly_outlier_pct=1e9,
        debug=0,
        vis_dir=None,
        show_progress=show_progress,
        progress_interval=0.5,
    )


def _generic_job(show_progress=True):
    canonical_hom = np.concatenate(
        [_canonical_verts(), np.ones((_canonical_verts().shape[0], 1))],
        axis=1,
    )
    mesh_verts = np.stack([
        (canonical_hom @ pose.T)[:, :3]
        for pose in _poses()
    ])
    return mv_eval_chamfer.ChamferCameraJob(
        cam_name="front",
        cam_intrinsics=np.eye(3),
        cam_extrinsics=np.eye(4),
        depth_dir=Path("depth"),
        mask_dir=Path("mask"),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        mesh_verts=mesh_verts,
        eval_image_size=None,
        anomaly_median_mm=1e9,
        anomaly_outlier_pct=1e9,
        debug=0,
        vis_dir=None,
        show_progress=show_progress,
        progress_interval=0.5,
    )


def test_rigid_chamfer_uses_persistent_mesh_and_empty_dynamic_depth_render(monkeypatch):
    _patch_metric_dependencies(monkeypatch)

    result = mv_eval_chamfer._eval_rigid_chamfer_camera(_job())

    renderer = FakeRenderer.instances[0]
    assert len(renderer.add_calls) == 1
    assert np.array_equal(renderer.add_calls[0][1], _poses()[0])
    assert [handle for handle, _ in renderer.pose_updates] == [7, 7]
    assert np.array_equal(renderer.pose_updates[1][1], _poses()[1])
    assert renderer.render_depth_meshes == [[], []]
    assert result.metrics is not None
    assert len(result.metrics["per_frame_mm"]) == 2


def test_generic_chamfer_uses_per_frame_dynamic_mesh_render(monkeypatch):
    _patch_metric_dependencies(monkeypatch)

    result = mv_eval_chamfer._eval_chamfer_camera(_generic_job())

    renderer = FakeRenderer.instances[0]
    assert renderer.add_calls == []
    assert renderer.pose_updates == []
    assert [len(meshes) for meshes in renderer.render_depth_meshes] == [1, 1]
    assert result.metrics is not None
    assert len(result.metrics["per_frame_mm"]) == 2


def test_rigid_chamfer_sparse_progress_without_tqdm(monkeypatch, capsys):
    _patch_metric_dependencies(monkeypatch)

    mv_eval_chamfer._eval_rigid_chamfer_camera(_job(show_progress=False))

    output = capsys.readouterr().out
    assert "Chamfer front: 1/2 (50%), " in output
    assert "Chamfer front: 2/2 (100%), " in output
    assert "it/s" in output


def test_generic_chamfer_sparse_progress_without_tqdm(monkeypatch, capsys):
    _patch_metric_dependencies(monkeypatch)

    mv_eval_chamfer._eval_chamfer_camera(_generic_job(show_progress=False))

    output = capsys.readouterr().out
    assert "Chamfer front: 1/2 (50%), " in output
    assert "Chamfer front: 2/2 (100%), " in output
    assert "it/s" in output


def test_rigid_chamfer_worker_dispatch_preserves_order(monkeypatch):
    FakeExecutor.max_workers_seen = []
    seen = []

    def fake_eval(job):
        seen.append(job.cam_name)
        return mv_eval_chamfer.RigidChamferCameraResult(
            cam_name=job.cam_name,
            metrics=None,
            frame_dists=[],
            timings={},
        )

    monkeypatch.setattr(mv_eval_chamfer, "_eval_rigid_chamfer_camera", fake_eval)
    monkeypatch.setattr(mv_eval_chamfer, "ProcessPoolExecutor", FakeExecutor)
    jobs = [
        mv_eval_chamfer.RigidChamferCameraJob(
            cam_name=name,
            cam_intrinsics=np.eye(3),
            cam_extrinsics=np.eye(4),
            depth_dir=Path("depth"),
            mask_dir=Path("mask"),
            canonical_verts=_canonical_verts(),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            poses=_poses(),
            eval_image_size=None,
            anomaly_median_mm=30.0,
            anomaly_outlier_pct=10.0,
            debug=0,
            vis_dir=None,
        )
        for name in ["front", "back", "left", "right"]
    ]

    results = mv_eval_chamfer._run_rigid_chamfer_camera_jobs(jobs, camera_workers=4)

    assert FakeExecutor.max_workers_seen == [4]
    assert seen == ["front", "back", "left", "right"]
    assert [result.cam_name for result in results] == seen


def test_generic_chamfer_worker_dispatch_preserves_order(monkeypatch):
    FakeExecutor.max_workers_seen = []
    seen = []

    def fake_eval(job):
        seen.append(job.cam_name)
        return mv_eval_chamfer.ChamferCameraResult(
            cam_name=job.cam_name,
            metrics=None,
            frame_dists=[],
            timings={},
        )

    monkeypatch.setattr(mv_eval_chamfer, "_eval_chamfer_camera", fake_eval)
    monkeypatch.setattr(mv_eval_chamfer, "ProcessPoolExecutor", FakeExecutor)
    jobs = [
        mv_eval_chamfer.ChamferCameraJob(
            cam_name=name,
            cam_intrinsics=np.eye(3),
            cam_extrinsics=np.eye(4),
            depth_dir=Path("depth"),
            mask_dir=Path("mask"),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
            mesh_verts=np.stack([_canonical_verts(), _canonical_verts()]),
            eval_image_size=None,
            anomaly_median_mm=30.0,
            anomaly_outlier_pct=10.0,
            debug=0,
            vis_dir=None,
        )
        for name in ["front", "back", "left", "right"]
    ]

    results = mv_eval_chamfer._run_chamfer_camera_jobs(jobs, camera_workers=4)

    assert FakeExecutor.max_workers_seen == [4]
    assert seen == ["front", "back", "left", "right"]
    assert [result.cam_name for result in results] == seen


def test_rigid_object_metrics_match_generic_chamfer_path(monkeypatch, tmp_path):
    _patch_metric_dependencies(monkeypatch)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    canonical_verts = _canonical_verts()
    poses = _poses()
    canonical_hom = np.concatenate(
        [canonical_verts, np.ones((canonical_verts.shape[0], 1))],
        axis=1,
    )
    mesh_verts = np.stack([
        (canonical_hom @ pose.T)[:, :3]
        for pose in poses
    ])

    generic = mv_eval_chamfer.mv_eval_chamfer(
        cam_names=["front"],
        cam_intrinsics=[np.eye(3)],
        cam_extrinsics=[np.eye(4)],
        depth_dirs=[Path("depth")],
        mask_dirs=[Path("mask")],
        faces=faces,
        mesh_verts=mesh_verts,
        output_path=tmp_path / "generic.json",
    )
    rigid = mv_eval_chamfer.mv_eval_chamfer_rigid_object(
        cam_names=["front"],
        cam_intrinsics=[np.eye(3)],
        cam_extrinsics=[np.eye(4)],
        depth_dirs=[Path("depth")],
        mask_dirs=[Path("mask")],
        canonical_verts=canonical_verts,
        faces=faces,
        poses=poses,
        output_path=tmp_path / "rigid.json",
    )

    assert rigid == generic


def test_generic_chamfer_parent_preserves_tiling_camera_order(monkeypatch, tmp_path):
    captured = {}

    def fake_run(jobs, camera_workers):
        captured["camera_workers"] = camera_workers
        return [
            mv_eval_chamfer.ChamferCameraResult(
                cam_name=job.cam_name,
                metrics={"mean_mm": 1.0, "median_mm": 1.0, "per_frame_mm": [1.0]},
                frame_dists=[0.001],
                timings={},
            )
            for job in jobs
        ]

    def fake_tile(**kwargs):
        captured["cam_names"] = kwargs["cam_names"]

    monkeypatch.setattr(mv_eval_chamfer, "_run_chamfer_camera_jobs", fake_run)
    monkeypatch.setattr(mv_eval_chamfer, "_tile_chamfer_videos", fake_tile)

    mv_eval_chamfer.mv_eval_chamfer(
        cam_names=["front", "back", "left"],
        cam_intrinsics=[np.eye(3)] * 3,
        cam_extrinsics=[np.eye(4)] * 3,
        depth_dirs=[Path("depth")] * 3,
        mask_dirs=[Path("mask")] * 3,
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        mesh_verts=np.stack([_canonical_verts(), _canonical_verts()]),
        output_path=tmp_path / "metrics.json",
        debug=1,
        vis_dir=tmp_path / "vis",
        camera_workers=3,
    )

    assert captured["camera_workers"] == 3
    assert captured["cam_names"] == ["front", "back", "left"]


def test_rigid_object_parent_preserves_tiling_camera_order(monkeypatch, tmp_path):
    captured = {}

    def fake_run(jobs, camera_workers):
        return [
            mv_eval_chamfer.RigidChamferCameraResult(
                cam_name=job.cam_name,
                metrics={"mean_mm": 1.0, "median_mm": 1.0, "per_frame_mm": [1.0]},
                frame_dists=[0.001],
                timings={},
            )
            for job in jobs
        ]

    def fake_tile(**kwargs):
        captured["cam_names"] = kwargs["cam_names"]

    monkeypatch.setattr(mv_eval_chamfer, "_run_rigid_chamfer_camera_jobs", fake_run)
    monkeypatch.setattr(mv_eval_chamfer, "_tile_chamfer_videos", fake_tile)

    mv_eval_chamfer.mv_eval_chamfer_rigid_object(
        cam_names=["front", "back", "left"],
        cam_intrinsics=[np.eye(3)] * 3,
        cam_extrinsics=[np.eye(4)] * 3,
        depth_dirs=[Path("depth")] * 3,
        mask_dirs=[Path("mask")] * 3,
        canonical_verts=_canonical_verts(),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        poses=_poses(),
        output_path=tmp_path / "metrics.json",
        debug=1,
        vis_dir=tmp_path / "vis",
        camera_workers=3,
    )

    assert captured["cam_names"] == ["front", "back", "left"]
