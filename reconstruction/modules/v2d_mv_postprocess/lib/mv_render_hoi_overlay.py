from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
import time

import cv2
import numpy as np
import pyglet
pyglet.options['headless'] = True
import torch
import trimesh
from tqdm import tqdm

from v2d.mv.rig import RigConfig
from v2d.common.video import FrameSource, get_video_writer, tile_videos
from v2d.mv.vis.renderer import Renderer

HUMAN_MESH_COLOR = np.array([102, 230, 179], dtype=np.uint8)  # light green
_MISSING = object()


@dataclass(frozen=True)
class CameraRenderJob:
    cam_name: str
    rgb_path: Path
    output_path: Path
    object_mesh_path: Path | None
    object_pose_path: Path | None
    mhr_mesh_mv_path: Path | None
    cam_intrinsics: np.ndarray
    cam_extrinsics: np.ndarray
    profile: bool = False
    show_progress: bool = True
    progress_interval: float = 0.1


@dataclass(frozen=True)
class CameraRenderResult:
    cam_name: str
    output_path: Path
    timings: dict[str, float]


def _log_profile(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def _format_timing(timings: dict[str, float]) -> str:
    return ", ".join(f"{name}={seconds:.2f}s" for name, seconds in timings.items())


def _cfg_get(cfg, key: str, default=None):
    value = _MISSING
    if hasattr(cfg, "get"):
        value = cfg.get(key, _MISSING)
    if value is _MISSING and hasattr(cfg, key):
        value = getattr(cfg, key)
    return default if value is _MISSING else value


def _optional_path(value) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return Path(text)


def _validate_overlay_asset_paths(
    object_mesh_path: Path | None,
    object_pose_path: Path | None,
    mhr_mesh_mv_path: Path | None,
) -> None:
    if (object_mesh_path is None) != (object_pose_path is None):
        raise ValueError(
            "Object overlay requires both object_mesh_path and "
            "object_pose_path/object_pose_dir."
        )
    if object_mesh_path is None and mhr_mesh_mv_path is None:
        raise ValueError(
            "HOI overlay requires at least one mesh source: provide object "
            "assets, human assets, or both."
        )


def _resolve_overlay_asset_paths(cfg) -> tuple[Path | None, Path | None, Path | None]:
    object_mesh_path = _optional_path(_cfg_get(cfg, "object_mesh_path"))
    object_pose_path = _optional_path(_cfg_get(cfg, "object_pose_path"))
    object_pose_dir = _optional_path(_cfg_get(cfg, "object_pose_dir"))
    if object_pose_path is None and object_pose_dir is not None:
        object_pose_path = object_pose_dir / "poses.npy"

    mhr_mesh_mv_path = _optional_path(_cfg_get(cfg, "mhr_mesh_mv_path"))
    human_pose_dir = _optional_path(_cfg_get(cfg, "human_pose_dir"))
    if mhr_mesh_mv_path is None and human_pose_dir is not None:
        mhr_mesh_mv_path = human_pose_dir / "mhr_mesh_mv.pt"

    _validate_overlay_asset_paths(
        object_mesh_path=object_mesh_path,
        object_pose_path=object_pose_path,
        mhr_mesh_mv_path=mhr_mesh_mv_path,
    )
    return object_mesh_path, object_pose_path, mhr_mesh_mv_path


def _validate_overlay_assets(
    object_mesh: trimesh.Trimesh | None,
    object_poses: np.ndarray | None,
    human_vertices: np.ndarray | None,
    human_faces: np.ndarray | None,
) -> None:
    if (object_mesh is None) != (object_poses is None):
        raise ValueError("Object overlay requires both object_mesh and object_poses.")
    if (human_vertices is None) != (human_faces is None):
        raise ValueError("Human overlay requires both human_vertices and human_faces.")
    if object_mesh is None and human_vertices is None:
        raise ValueError(
            "HOI overlay requires at least one mesh source: provide object "
            "assets, human assets, or both."
        )


def _load_overlay_assets(
    object_mesh_path: Path | None,
    object_pose_path: Path | None,
    mhr_mesh_mv_path: Path | None,
):
    _validate_overlay_asset_paths(
        object_mesh_path=object_mesh_path,
        object_pose_path=object_pose_path,
        mhr_mesh_mv_path=mhr_mesh_mv_path,
    )

    object_mesh = None
    object_poses = None
    if object_mesh_path is not None:
        object_mesh = trimesh.load(object_mesh_path, process=False, force='mesh')
        object_poses = np.load(object_pose_path)

    human_vertices = None
    human_faces = None
    if mhr_mesh_mv_path is not None:
        mhr_mesh = torch.load(mhr_mesh_mv_path, weights_only=False, map_location="cpu")
        human_vertices = mhr_mesh["pred_vertices"].cpu().numpy()
        human_faces = mhr_mesh["faces"].cpu().numpy()

    _validate_overlay_assets(
        object_mesh=object_mesh,
        object_poses=object_poses,
        human_vertices=human_vertices,
        human_faces=human_faces,
    )

    return object_mesh, object_poses, human_vertices, human_faces


def render_hoi_overlay(
    rgb_path: Path,
    output_path: Path,
    object_mesh: trimesh.Trimesh | None = None,
    object_poses: np.ndarray | None = None,
    human_vertices: np.ndarray | None = None,
    human_faces: np.ndarray | None = None,
    cam_intrinsics: np.ndarray | None = None,
    cam_extrinsics: np.ndarray | None = None,
    profile: bool = False,
    show_progress: bool = True,
    progress_interval: float = 0.1,
) -> dict[str, float]:
    """Render object + human mesh overlay onto video frames.

    Args:
        rgb_path: Path to RGB frames (image dir, .h5, or video file).
        output_path: Output video path.
        object_mesh: Optional object mesh in its canonical frame.
        object_poses: Optional (N, 4, 4) per-frame object-to-world poses.
        human_vertices: Optional (N, V, 3) human vertices in world frame.
        human_faces: Optional (F, 3) human mesh face indices.
        cam_intrinsics: (3, 3) camera intrinsic matrix.
        cam_extrinsics: (4, 4) T_world_from_camera matrix.
    """
    _validate_overlay_assets(
        object_mesh=object_mesh,
        object_poses=object_poses,
        human_vertices=human_vertices,
        human_faces=human_faces,
    )
    if cam_intrinsics is None or cam_extrinsics is None:
        raise ValueError("cam_intrinsics and cam_extrinsics are required.")
    total_start = time.perf_counter()
    timings = {
        "setup": 0.0,
        "pose": 0.0,
        "human_mesh": 0.0,
        "render": 0.0,
        "annotate": 0.0,
        "write": 0.0,
    }

    source = FrameSource.from_path(rgb_path)
    frame_limits = [source.n_frames]
    if object_poses is not None:
        frame_limits.append(len(object_poses))
    if human_vertices is not None:
        frame_limits.append(len(human_vertices))
    n_frames = min(frame_limits)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    human_colors = None
    if human_vertices is not None:
        human_colors = np.tile(HUMAN_MESH_COLOR, (human_vertices.shape[1], 1))

    try:
        writer = get_video_writer(output_path, fps=30, crf=23)

        setup_start = time.perf_counter()
        with Renderer(image_size=source.image_size) as renderer:
            object_handle = None
            if object_mesh is not None and n_frames > 0:
                object_handle = renderer.add_persistent_mesh(
                    object_mesh,
                    pose=object_poses[0],
                )
            timings["setup"] += time.perf_counter() - setup_start

            progress = tqdm(
                islice(source.iter_frames(), n_frames),
                total=n_frames,
                desc=f"Rendering HOI overlay {output_path.stem}",
                disable=not show_progress,
            )
            frame_loop_start = time.perf_counter()
            progress_interval = min(max(progress_interval, 0.0), 1.0)
            next_progress_fraction = progress_interval
            for i, image in enumerate(progress):
                if object_handle is not None:
                    pose_start = time.perf_counter()
                    renderer.set_persistent_mesh_pose(object_handle, object_poses[i])
                    timings["pose"] += time.perf_counter() - pose_start

                meshes = []
                if human_vertices is not None:
                    mesh_start = time.perf_counter()
                    human_mesh_i = trimesh.Trimesh(
                        vertices=human_vertices[i],
                        faces=human_faces,
                        vertex_colors=human_colors,
                        process=False,
                    )
                    meshes.append(human_mesh_i)
                    timings["human_mesh"] += time.perf_counter() - mesh_start

                render_start = time.perf_counter()
                rendered_image = renderer.render_overlay(
                    meshes=meshes,
                    K=cam_intrinsics,
                    T=cam_extrinsics,
                    image=image,
                ) * 255.0
                timings["render"] += time.perf_counter() - render_start

                annotate_start = time.perf_counter()
                frame_text = f"Frame {i}"
                (tw, th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(rendered_image, frame_text, (rendered_image.shape[1] - tw - 10, th + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                timings["annotate"] += time.perf_counter() - annotate_start

                write_start = time.perf_counter()
                writer.write_frame(rendered_image.astype(np.uint8))
                timings["write"] += time.perf_counter() - write_start

                if not show_progress and progress_interval > 0 and n_frames > 0:
                    completed = i + 1
                    fraction = completed / n_frames
                    if fraction >= next_progress_fraction:
                        elapsed = max(time.perf_counter() - frame_loop_start, 1e-9)
                        print(
                            f"Rendering HOI overlay {output_path.stem}: "
                            f"{completed}/{n_frames} "
                            f"({fraction:.0%}), "
                            f"{completed / elapsed:.2f} it/s"
                        )
                        next_progress_fraction = (
                            int(fraction / progress_interval) + 1
                        ) * progress_interval

        write_start = time.perf_counter()
        writer.close()
        writer = None
        timings["write"] += time.perf_counter() - write_start
    finally:
        if writer is not None:
            writer.close()
        source.close()

    timings["total"] = time.perf_counter() - total_start
    _log_profile(
        profile,
        f"Profile {output_path.name}: frames={n_frames}, {_format_timing(timings)}",
    )
    print(f"Saved HOI overlay video to {output_path}")
    return timings


def _render_camera_overlay_worker(job: CameraRenderJob) -> CameraRenderResult:
    load_start = time.perf_counter()
    object_mesh, object_poses, human_vertices, human_faces = _load_overlay_assets(
        object_mesh_path=job.object_mesh_path,
        object_pose_path=job.object_pose_path,
        mhr_mesh_mv_path=job.mhr_mesh_mv_path,
    )
    asset_load = time.perf_counter() - load_start
    _log_profile(
        job.profile,
        f"Profile {job.cam_name}: asset_load={asset_load:.2f}s",
    )
    timings = render_hoi_overlay(
        rgb_path=job.rgb_path,
        output_path=job.output_path,
        object_mesh=object_mesh,
        object_poses=object_poses,
        human_vertices=human_vertices,
        human_faces=human_faces,
        cam_intrinsics=job.cam_intrinsics,
        cam_extrinsics=job.cam_extrinsics,
        profile=job.profile,
        show_progress=job.show_progress,
        progress_interval=job.progress_interval,
    )
    timings = {"asset_load": asset_load, **timings}
    return CameraRenderResult(
        cam_name=job.cam_name,
        output_path=job.output_path,
        timings=timings,
    )


def _run_camera_render_jobs(
    jobs: list[CameraRenderJob],
    render_workers: int,
) -> list[CameraRenderResult]:
    if render_workers <= 1:
        return [_render_camera_overlay_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=render_workers) as executor:
        return list(executor.map(_render_camera_overlay_worker, jobs, chunksize=1))


def _build_camera_render_jobs(
    cfg,
    rig: RigConfig,
    show_progress: bool = True,
    progress_interval: float = 0.1,
) -> list[CameraRenderJob]:
    jobs: list[CameraRenderJob] = []
    profile = bool(cfg.get("profile", False))
    object_mesh_path, object_pose_path, mhr_mesh_mv_path = _resolve_overlay_asset_paths(cfg)
    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        jobs.append(
            CameraRenderJob(
                cam_name=cam.name,
                rgb_path=Path(cfg.rgb_path_template.format(cam_name=cam.name)),
                output_path=Path(cfg.output_dir) / f"{cam.name}_hoi_overlay.mp4",
                object_mesh_path=object_mesh_path,
                object_pose_path=object_pose_path,
                mhr_mesh_mv_path=mhr_mesh_mv_path,
                cam_intrinsics=cam.param.K,
                cam_extrinsics=cam.param.T,
                profile=profile,
                show_progress=show_progress,
                progress_interval=progress_interval,
            )
        )
    return jobs


def render_hoi_overlay_from_config(cfg):
    total_start = time.perf_counter()
    profile = bool(cfg.get("profile", False))
    render_workers = max(1, int(cfg.get("render_workers", 1)))
    progress_interval = float(cfg.get("progress_interval", 0.1))

    rig_start = time.perf_counter()
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)
    _log_profile(
        profile,
        f"Profile render_hoi_overlay: rig_load={time.perf_counter() - rig_start:.2f}s",
    )

    overlay_paths: list[Path] = []
    cam_names: list[str] = []
    jobs = _build_camera_render_jobs(
        cfg,
        rig,
        show_progress=render_workers == 1,
        progress_interval=progress_interval,
    )

    if render_workers == 1:
        load_start = time.perf_counter()
        object_mesh_path, object_pose_path, mhr_mesh_mv_path = _resolve_overlay_asset_paths(cfg)
        object_mesh, object_poses, human_vertices, human_faces = _load_overlay_assets(
            object_mesh_path=object_mesh_path,
            object_pose_path=object_pose_path,
            mhr_mesh_mv_path=mhr_mesh_mv_path,
        )
        _log_profile(
            profile,
            f"Profile render_hoi_overlay: asset_load={time.perf_counter() - load_start:.2f}s",
        )
        for job in jobs:
            print(f"Rendering HOI overlay for camera {job.cam_name}...")
            render_hoi_overlay(
                rgb_path=job.rgb_path,
                output_path=job.output_path,
                object_mesh=object_mesh,
                object_poses=object_poses,
                human_vertices=human_vertices,
                human_faces=human_faces,
                cam_intrinsics=job.cam_intrinsics,
                cam_extrinsics=job.cam_extrinsics,
                profile=profile,
                show_progress=job.show_progress,
                progress_interval=job.progress_interval,
            )
            overlay_paths.append(job.output_path)
            cam_names.append(job.cam_name)
    else:
        print(f"Rendering HOI overlays with {render_workers} worker(s)...")
        results = _run_camera_render_jobs(jobs, render_workers=render_workers)
        overlay_paths = [result.output_path for result in results]
        cam_names = [result.cam_name for result in results]

    tile_shape = tuple(cfg.get("tile_shape", [2, 2]))
    tile_image_size_cfg = cfg.get("tile_image_size", None)
    tile_image_size = tuple(tile_image_size_cfg) if tile_image_size_cfg is not None else None
    tiled_path = Path(cfg.output_dir) / "tiled_hoi_overlay.mp4"
    print(f"Tiling {len(overlay_paths)} overlays into {tiled_path}...")
    tile_start = time.perf_counter()
    tile_sources = [FrameSource.from_path(p) for p in overlay_paths]
    try:
        tile_videos(
            sources=tile_sources,
            output_path=tiled_path,
            tile_shape=tile_shape,
            output_image_size=tile_image_size,
            video_names=cam_names,
        )
    finally:
        for source in tile_sources:
            source.close()
    _log_profile(
        profile,
        f"Profile render_hoi_overlay: tiling={time.perf_counter() - tile_start:.2f}s",
    )
    _log_profile(
        profile,
        f"Profile render_hoi_overlay: total={time.perf_counter() - total_start:.2f}s",
    )
    print(f"Saved tiled HOI overlay to {tiled_path}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Render HOI overlay for all cameras from config")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--rgb_dir", type=str, required=True)
    parser.add_argument("--object_mesh_path", type=str)
    parser.add_argument("--object_pose_dir", type=str)
    parser.add_argument("--human_pose_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_render_hoi_overlay.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides = {k: v for k, v in vars(args).items() if k != "config_path" and v is not None}
    cfg = OmegaConf.merge(cfg, overrides)
    render_hoi_overlay_from_config(cfg)
