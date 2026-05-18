from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import trimesh

from v2d.common.datatypes import DepthImage
from v2d.common.video import FrameSource, get_video_writer, tile_videos
from v2d.mv.math.numpy_fn import depth_to_xyz, visible_vertices, xyz_to_uv
from v2d.mv.vis.renderer import Renderer

VERTEX_RADIUS = 2


@dataclass(frozen=True)
class RigidChamferCameraJob:
    cam_name: str
    cam_intrinsics: np.ndarray
    cam_extrinsics: np.ndarray
    depth_dir: Path
    mask_dir: Path
    canonical_verts: np.ndarray
    faces: np.ndarray
    poses: np.ndarray
    eval_image_size: tuple[int, int] | None
    anomaly_median_mm: float
    anomaly_outlier_pct: float
    debug: int
    vis_dir: Path | None
    profile: bool = False
    show_progress: bool = True
    progress_interval: float = 0.1


@dataclass(frozen=True)
class RigidChamferCameraResult:
    cam_name: str
    metrics: dict | None
    frame_dists: list[float]
    timings: dict[str, float]


@dataclass(frozen=True)
class ChamferCameraJob:
    cam_name: str
    cam_intrinsics: np.ndarray
    cam_extrinsics: np.ndarray
    depth_dir: Path
    mask_dir: Path
    faces: np.ndarray
    mesh_verts: np.ndarray
    eval_image_size: tuple[int, int] | None
    anomaly_median_mm: float
    anomaly_outlier_pct: float
    debug: int
    vis_dir: Path | None
    profile: bool = False
    show_progress: bool = True
    progress_interval: float = 0.1


@dataclass(frozen=True)
class ChamferCameraResult:
    cam_name: str
    metrics: dict | None
    frame_dists: list[float]
    timings: dict[str, float]


def _format_timing(timings: dict[str, float]) -> str:
    return ", ".join(f"{name}={seconds:.2f}s" for name, seconds in timings.items())


def _transform_vertices(canonical_hom: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (canonical_hom @ pose.T)[:, :3]


def _maybe_log_sparse_progress(
    *,
    prefix: str,
    completed: int,
    total: int,
    start_time: float,
    next_fraction: float,
    interval: float,
) -> float:
    if total <= 0 or interval <= 0:
        return next_fraction

    fraction = completed / total
    if fraction < next_fraction:
        return next_fraction

    elapsed = max(time.perf_counter() - start_time, 1e-9)
    print(
        f"{prefix}: {completed}/{total} ({fraction:.0%}), "
        f"{completed / elapsed:.2f} it/s"
    )
    return (int(fraction / interval) + 1) * interval


def _summarize_camera_metrics(
    cam_name: str,
    cam_dists: list[float],
    print_summary: bool = True,
) -> dict | None:
    if not cam_dists:
        return None

    arr = np.array(cam_dists)
    metrics = {
        "mean_mm": float(arr.mean() * 1000),
        "median_mm": float(np.median(arr) * 1000),
        "per_frame_mm": [float(v * 1000) for v in cam_dists],
    }
    if print_summary:
        print(f"  {cam_name}: mean={metrics['mean_mm']:.1f}mm  "
              f"median={metrics['median_mm']:.1f}mm  "
              f"({len(cam_dists)} frames)")
    return metrics


def _tile_chamfer_videos(
    *,
    cam_names: list[str],
    debug: int,
    vis_dir: Path | None,
    tile_shape: tuple[int, int],
    tile_image_size: tuple[int, int] | None,
) -> None:
    if debug <= 0 or vis_dir is None or len(cam_names) <= 1:
        return

    vis_paths = [
        vis_dir / f"{name}.mp4"
        for name in cam_names
        if (vis_dir / f"{name}.mp4").exists()
    ]
    if len(vis_paths) <= 1:
        return

    tiled_path = vis_dir / "tiled_chamfer.mp4"
    print(f"Tiling {len(vis_paths)} chamfer videos into {tiled_path}...")
    tile_sources = [FrameSource.from_path(p) for p in vis_paths]
    try:
        tile_videos(
            sources=tile_sources,
            output_path=tiled_path,
            tile_shape=tile_shape,
            output_image_size=tile_image_size,
            video_names=[p.stem for p in vis_paths],
        )
    except Exception as e:
        print(f"WARNING: tile_videos failed: {e}. Skipping tiled viz.")
    finally:
        for source in tile_sources:
            source.close()


def _draw_colorbar(
    canvas: np.ndarray,
    vmax_mm: float,
    bar_width: int = 20,
    bar_height: int = 150,
    margin: int = 10,
) -> None:
    """Draw a vertical jet colorbar in the bottom-right corner (in-place)."""
    H, W = canvas.shape[:2]
    x1 = W - margin - bar_width
    y1 = H - margin - bar_height
    x2 = x1 + bar_width
    y2 = y1 + bar_height

    for row in range(bar_height):
        normed = 1.0 - row / (bar_height - 1)
        color_rgb = (np.array(cm.jet(normed)[:3]) * 255).astype(np.uint8)
        color_bgr = color_rgb[::-1].tolist()
        cv2.line(canvas, (x1, y1 + row), (x2, y1 + row), color_bgr, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    cv2.putText(canvas, f"{vmax_mm:.0f}mm", (x1 - 50, y1 + 10), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(canvas, "0mm", (x1 - 32, y2), font, font_scale, (255, 255, 255), thickness)


def _render_vertex_heatmap(
    image_shape: tuple[int, int],
    uv: np.ndarray,
    dists: np.ndarray,
    frame_idx: int,
    vmax_mm: float = 100.0,
    anomaly_msg: str | None = None,
) -> np.ndarray:
    """Render per-vertex distances as colored circles on an image.

    Args:
        image_shape: (H, W) canvas size.
        uv: (N, 2) pixel coordinates of projected vertices.
        dists: (N,) distances in meters.
        frame_idx: Frame number for display.
        vmax_mm: Colormap saturation value in mm.
        anomaly_msg: If set, displayed as a warning on the image.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    H, W = image_shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    dists_mm = np.clip(dists * 1000, 0, vmax_mm)
    normed = dists_mm / vmax_mm
    colors = (cm.jet(normed)[:, :3] * 255).astype(np.uint8)

    for pt, color in zip(uv.astype(int), colors):
        u, v = pt
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(canvas, (u, v), VERTEX_RADIUS, color[::-1].tolist(), -1)

    _draw_colorbar(canvas, vmax_mm)

    label = f"Frame {frame_idx}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(canvas, label, (W - tw - 10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if anomaly_msg:
        cv2.putText(canvas, anomaly_msg, (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return canvas


def _eval_rigid_chamfer_camera(job: RigidChamferCameraJob) -> RigidChamferCameraResult:
    from scipy.spatial import cKDTree

    timings = {
        "setup": 0.0,
        "read": 0.0,
        "resize": 0.0,
        "pointcloud": 0.0,
        "pose": 0.0,
        "render_depth": 0.0,
        "visibility": 0.0,
        "kdtree": 0.0,
        "debug_write": 0.0,
    }
    total_start = time.perf_counter()

    depth_source = FrameSource.from_path(job.depth_dir)
    mask_source = FrameSource.from_path(job.mask_dir)
    writer = None
    cam_dists: list[float] = []

    try:
        n_frames = job.poses.shape[0]
        if depth_source.n_frames != n_frames or mask_source.n_frames != n_frames:
            raise ValueError(
                f"camera {job.cam_name}: frame count mismatch "
                f"(depth={depth_source.n_frames}, mask={mask_source.n_frames}, expected={n_frames})"
            )

        setup_start = time.perf_counter()
        first_depth = DepthImage.from_array(depth_source[0]).depth
        H_orig, W_orig = first_depth.shape[:2]

        if job.eval_image_size is not None:
            W_eval, H_eval = job.eval_image_size
            sx, sy = W_eval / W_orig, H_eval / H_orig
            K_eval = job.cam_intrinsics.copy()
            K_eval[0, :] *= sx
            K_eval[1, :] *= sy
        else:
            W_eval, H_eval = W_orig, H_orig
            K_eval = job.cam_intrinsics

        if job.debug > 0 and job.vis_dir:
            job.vis_dir.mkdir(parents=True, exist_ok=True)
            writer = get_video_writer(job.vis_dir / f"{job.cam_name}.mp4", fps=30, crf=23)

        canonical_hom = np.concatenate(
            [job.canonical_verts, np.ones((job.canonical_verts.shape[0], 1))],
            axis=1,
        )
        canonical_mesh = trimesh.Trimesh(
            vertices=job.canonical_verts,
            faces=job.faces,
            process=False,
        )
        timings["setup"] += time.perf_counter() - setup_start

        def _write_placeholder(frame_idx: int, reason: str) -> None:
            if writer is None:
                return
            debug_start = time.perf_counter()
            canvas = np.zeros((H_eval, W_eval, 3), dtype=np.uint8)
            label = f"Frame {frame_idx}: {reason}"
            cv2.putText(canvas, label, (10, H_eval - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write_frame(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            timings["debug_write"] += time.perf_counter() - debug_start

        with Renderer(image_size=(W_eval, H_eval)) as renderer:
            object_handle = None
            if n_frames > 0:
                object_handle = renderer.add_persistent_mesh(
                    canonical_mesh,
                    pose=job.poses[0],
                )

            progress = tqdm(
                range(n_frames),
                desc=f"Chamfer {job.cam_name}",
                disable=not job.show_progress,
            )
            progress_start = time.perf_counter()
            progress_interval = min(max(job.progress_interval, 0.0), 1.0)
            next_progress_fraction = progress_interval

            def _log_frame_progress(frame_idx: int) -> None:
                nonlocal next_progress_fraction
                if job.show_progress:
                    return
                next_progress_fraction = _maybe_log_sparse_progress(
                    prefix=f"Chamfer {job.cam_name}",
                    completed=frame_idx + 1,
                    total=n_frames,
                    start_time=progress_start,
                    next_fraction=next_progress_fraction,
                    interval=progress_interval,
                )

            for i in progress:
                read_start = time.perf_counter()
                depth = DepthImage.from_array(depth_source[i]).depth
                mask_raw = mask_source[i].astype(np.float32) / 255.0
                timings["read"] += time.perf_counter() - read_start

                resize_start = time.perf_counter()
                if job.eval_image_size is not None:
                    depth = cv2.resize(depth, (W_eval, H_eval), interpolation=cv2.INTER_LINEAR)
                    mask_raw = cv2.resize(mask_raw, (W_eval, H_eval), interpolation=cv2.INTER_NEAREST)
                elif mask_raw.shape[:2] != depth.shape[:2]:
                    mask_raw = cv2.resize(
                        mask_raw, (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask = (mask_raw > 0.5) & (depth > 0.001)
                timings["resize"] += time.perf_counter() - resize_start

                pointcloud_start = time.perf_counter()
                pts_world = depth_to_xyz(depth, K_eval, job.cam_extrinsics, mask=mask)
                timings["pointcloud"] += time.perf_counter() - pointcloud_start
                if pts_world.shape[0] < 10:
                    _write_placeholder(i, "no valid depth points")
                    _log_frame_progress(i)
                    continue

                pose_start = time.perf_counter()
                pose = job.poses[i]
                if object_handle is not None:
                    renderer.set_persistent_mesh_pose(object_handle, pose)
                verts_np = _transform_vertices(canonical_hom, pose)
                timings["pose"] += time.perf_counter() - pose_start

                render_start = time.perf_counter()
                mesh_zbuf = renderer.render_depth([], K_eval, job.cam_extrinsics)
                timings["render_depth"] += time.perf_counter() - render_start

                visibility_start = time.perf_counter()
                vis = visible_vertices(verts_np, mesh_zbuf, K_eval, job.cam_extrinsics)

                H_m, W_m = mask.shape[:2]
                uv_int, in_bounds = xyz_to_uv(verts_np, K_eval, job.cam_extrinsics, image_size=(W_m, H_m))
                mask_vis = np.zeros(verts_np.shape[0], dtype=bool)
                ib_idx = np.where(in_bounds)[0]
                mask_vis[ib_idx] = mask[uv_int[ib_idx, 1], uv_int[ib_idx, 0]]
                vis = vis & mask_vis
                vis_verts = verts_np[vis]
                timings["visibility"] += time.perf_counter() - visibility_start

                if vis_verts.shape[0] < 10:
                    _write_placeholder(i, "no visible vertices")
                    _log_frame_progress(i)
                    continue

                kdtree_start = time.perf_counter()
                tree = cKDTree(pts_world)
                per_vert_dists, _ = tree.query(vis_verts, k=1)
                mean_dist = float(per_vert_dists.mean())
                cam_dists.append(mean_dist)

                dists_mm = per_vert_dists * 1000
                median_mm = float(np.median(dists_mm))
                q1, q3 = np.percentile(dists_mm, [25, 75])
                iqr = q3 - q1
                outlier_thresh = q3 + 3.0 * iqr
                n_outliers = int(np.sum(dists_mm > outlier_thresh))
                outlier_pct = n_outliers / len(dists_mm) * 100
                timings["kdtree"] += time.perf_counter() - kdtree_start

                anomaly_msg = None
                if median_mm > job.anomaly_median_mm or outlier_pct > job.anomaly_outlier_pct:
                    anomaly_msg = (
                        f"ANOMALY: median={median_mm:.1f}mm "
                        f"outliers={n_outliers}/{len(dists_mm)} ({outlier_pct:.0f}%)"
                    )
                    message = f"  {job.cam_name} frame {i}: {anomaly_msg}"
                    if job.show_progress:
                        tqdm.write(message)
                    else:
                        print(message)

                if writer is not None:
                    debug_start = time.perf_counter()
                    heatmap = _render_vertex_heatmap(
                        depth.shape[:2], uv_int[vis], per_vert_dists,
                        frame_idx=i, anomaly_msg=anomaly_msg,
                    )
                    writer.write_frame(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
                    timings["debug_write"] += time.perf_counter() - debug_start

                _log_frame_progress(i)

        if writer is not None:
            debug_start = time.perf_counter()
            writer.close()
            writer = None
            timings["debug_write"] += time.perf_counter() - debug_start
    finally:
        if writer is not None:
            writer.close()
        depth_source.close()
        mask_source.close()

    timings["total"] = time.perf_counter() - total_start
    if job.profile:
        print(
            f"Profile chamfer object {job.cam_name}: "
            f"frames={job.poses.shape[0]}, {_format_timing(timings)}"
        )

    return RigidChamferCameraResult(
        cam_name=job.cam_name,
        metrics=_summarize_camera_metrics(
            job.cam_name,
            cam_dists,
            print_summary=False,
        ),
        frame_dists=cam_dists,
        timings=timings,
    )


def _eval_chamfer_camera(job: ChamferCameraJob) -> ChamferCameraResult:
    from scipy.spatial import cKDTree

    timings = {
        "setup": 0.0,
        "read": 0.0,
        "resize": 0.0,
        "pointcloud": 0.0,
        "mesh_build": 0.0,
        "render_depth": 0.0,
        "visibility": 0.0,
        "kdtree": 0.0,
        "debug_write": 0.0,
    }
    total_start = time.perf_counter()

    depth_source = FrameSource.from_path(job.depth_dir)
    mask_source = FrameSource.from_path(job.mask_dir)
    writer = None
    cam_dists: list[float] = []

    try:
        n_frames = job.mesh_verts.shape[0]
        if depth_source.n_frames != n_frames or mask_source.n_frames != n_frames:
            raise ValueError(
                f"camera {job.cam_name}: frame count mismatch "
                f"(depth={depth_source.n_frames}, mask={mask_source.n_frames}, expected={n_frames})"
            )

        setup_start = time.perf_counter()
        first_depth = DepthImage.from_array(depth_source[0]).depth
        H_orig, W_orig = first_depth.shape[:2]

        if job.eval_image_size is not None:
            W_eval, H_eval = job.eval_image_size
            sx, sy = W_eval / W_orig, H_eval / H_orig
            K_eval = job.cam_intrinsics.copy()
            K_eval[0, :] *= sx
            K_eval[1, :] *= sy
        else:
            W_eval, H_eval = W_orig, H_orig
            K_eval = job.cam_intrinsics

        if job.debug > 0 and job.vis_dir:
            job.vis_dir.mkdir(parents=True, exist_ok=True)
            writer = get_video_writer(job.vis_dir / f"{job.cam_name}.mp4", fps=30, crf=23)
        timings["setup"] += time.perf_counter() - setup_start

        def _write_placeholder(frame_idx: int, reason: str) -> None:
            if writer is None:
                return
            debug_start = time.perf_counter()
            canvas = np.zeros((H_eval, W_eval, 3), dtype=np.uint8)
            label = f"Frame {frame_idx}: {reason}"
            cv2.putText(canvas, label, (10, H_eval - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write_frame(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            timings["debug_write"] += time.perf_counter() - debug_start

        with Renderer(image_size=(W_eval, H_eval)) as renderer:
            progress = tqdm(
                range(n_frames),
                desc=f"Chamfer {job.cam_name}",
                disable=not job.show_progress,
            )
            progress_start = time.perf_counter()
            progress_interval = min(max(job.progress_interval, 0.0), 1.0)
            next_progress_fraction = progress_interval

            def _log_frame_progress(frame_idx: int) -> None:
                nonlocal next_progress_fraction
                if job.show_progress:
                    return
                next_progress_fraction = _maybe_log_sparse_progress(
                    prefix=f"Chamfer {job.cam_name}",
                    completed=frame_idx + 1,
                    total=n_frames,
                    start_time=progress_start,
                    next_fraction=next_progress_fraction,
                    interval=progress_interval,
                )

            for i in progress:
                read_start = time.perf_counter()
                depth = DepthImage.from_array(depth_source[i]).depth
                mask_raw = mask_source[i].astype(np.float32) / 255.0
                timings["read"] += time.perf_counter() - read_start

                resize_start = time.perf_counter()
                if job.eval_image_size is not None:
                    depth = cv2.resize(depth, (W_eval, H_eval), interpolation=cv2.INTER_LINEAR)
                    mask_raw = cv2.resize(mask_raw, (W_eval, H_eval), interpolation=cv2.INTER_NEAREST)
                elif mask_raw.shape[:2] != depth.shape[:2]:
                    mask_raw = cv2.resize(
                        mask_raw, (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask = (mask_raw > 0.5) & (depth > 0.001)
                timings["resize"] += time.perf_counter() - resize_start

                pointcloud_start = time.perf_counter()
                pts_world = depth_to_xyz(depth, K_eval, job.cam_extrinsics, mask=mask)
                timings["pointcloud"] += time.perf_counter() - pointcloud_start
                if pts_world.shape[0] < 10:
                    _write_placeholder(i, "no valid depth points")
                    _log_frame_progress(i)
                    continue

                mesh_start = time.perf_counter()
                verts_np = job.mesh_verts[i]
                frame_mesh = trimesh.Trimesh(vertices=verts_np, faces=job.faces, process=False)
                timings["mesh_build"] += time.perf_counter() - mesh_start

                render_start = time.perf_counter()
                mesh_zbuf = renderer.render_depth([frame_mesh], K_eval, job.cam_extrinsics)
                timings["render_depth"] += time.perf_counter() - render_start

                visibility_start = time.perf_counter()
                vis = visible_vertices(verts_np, mesh_zbuf, K_eval, job.cam_extrinsics)

                H_m, W_m = mask.shape[:2]
                uv_int, in_bounds = xyz_to_uv(verts_np, K_eval, job.cam_extrinsics, image_size=(W_m, H_m))
                mask_vis = np.zeros(verts_np.shape[0], dtype=bool)
                ib_idx = np.where(in_bounds)[0]
                mask_vis[ib_idx] = mask[uv_int[ib_idx, 1], uv_int[ib_idx, 0]]
                vis = vis & mask_vis
                vis_verts = verts_np[vis]
                timings["visibility"] += time.perf_counter() - visibility_start

                if vis_verts.shape[0] < 10:
                    _write_placeholder(i, "no visible vertices")
                    _log_frame_progress(i)
                    continue

                kdtree_start = time.perf_counter()
                tree = cKDTree(pts_world)
                per_vert_dists, _ = tree.query(vis_verts, k=1)
                mean_dist = float(per_vert_dists.mean())
                cam_dists.append(mean_dist)

                dists_mm = per_vert_dists * 1000
                median_mm = float(np.median(dists_mm))
                q1, q3 = np.percentile(dists_mm, [25, 75])
                iqr = q3 - q1
                outlier_thresh = q3 + 3.0 * iqr
                n_outliers = int(np.sum(dists_mm > outlier_thresh))
                outlier_pct = n_outliers / len(dists_mm) * 100
                timings["kdtree"] += time.perf_counter() - kdtree_start

                anomaly_msg = None
                if median_mm > job.anomaly_median_mm or outlier_pct > job.anomaly_outlier_pct:
                    anomaly_msg = (
                        f"ANOMALY: median={median_mm:.1f}mm "
                        f"outliers={n_outliers}/{len(dists_mm)} ({outlier_pct:.0f}%)"
                    )
                    message = f"  {job.cam_name} frame {i}: {anomaly_msg}"
                    if job.show_progress:
                        tqdm.write(message)
                    else:
                        print(message)

                if writer is not None:
                    debug_start = time.perf_counter()
                    heatmap = _render_vertex_heatmap(
                        depth.shape[:2], uv_int[vis], per_vert_dists,
                        frame_idx=i, anomaly_msg=anomaly_msg,
                    )
                    writer.write_frame(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
                    timings["debug_write"] += time.perf_counter() - debug_start

                _log_frame_progress(i)

        if writer is not None:
            debug_start = time.perf_counter()
            writer.close()
            writer = None
            timings["debug_write"] += time.perf_counter() - debug_start
    finally:
        if writer is not None:
            writer.close()
        depth_source.close()
        mask_source.close()

    timings["total"] = time.perf_counter() - total_start
    if job.profile:
        print(
            f"Profile chamfer {job.cam_name}: "
            f"frames={job.mesh_verts.shape[0]}, {_format_timing(timings)}"
        )

    return ChamferCameraResult(
        cam_name=job.cam_name,
        metrics=_summarize_camera_metrics(
            job.cam_name,
            cam_dists,
            print_summary=False,
        ),
        frame_dists=cam_dists,
        timings=timings,
    )


def _run_chamfer_camera_jobs(
    jobs: list[ChamferCameraJob],
    camera_workers: int,
) -> list[ChamferCameraResult]:
    if camera_workers <= 1:
        return [_eval_chamfer_camera(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=camera_workers) as executor:
        return list(executor.map(_eval_chamfer_camera, jobs, chunksize=1))


def _run_rigid_chamfer_camera_jobs(
    jobs: list[RigidChamferCameraJob],
    camera_workers: int,
) -> list[RigidChamferCameraResult]:
    if camera_workers <= 1:
        return [_eval_rigid_chamfer_camera(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=camera_workers) as executor:
        return list(executor.map(_eval_rigid_chamfer_camera, jobs, chunksize=1))


def mv_eval_chamfer_rigid_object(
    cam_names: list[str],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    depth_dirs: list[Path],
    mask_dirs: list[Path],
    canonical_verts: np.ndarray,
    faces: np.ndarray,
    poses: np.ndarray,
    output_path: Path,
    eval_image_size: tuple[int, int] | None = None,
    anomaly_median_mm: float = 30.0,
    anomaly_outlier_pct: float = 10.0,
    debug: int = 0,
    vis_dir: Path | None = None,
    tile_shape: tuple[int, int] = (2, 2),
    tile_image_size: tuple[int, int] | None = None,
    camera_workers: int = 1,
    profile: bool = False,
    progress_interval: float = 0.1,
) -> dict:
    """Compute object chamfer using a persistent rigid object mesh renderer."""
    camera_workers = max(1, int(camera_workers))
    show_progress = camera_workers == 1

    jobs = [
        RigidChamferCameraJob(
            cam_name=cam_name,
            cam_intrinsics=cam_intrinsics[idx],
            cam_extrinsics=cam_extrinsics[idx],
            depth_dir=depth_dirs[idx],
            mask_dir=mask_dirs[idx],
            canonical_verts=canonical_verts,
            faces=faces,
            poses=poses,
            eval_image_size=eval_image_size,
            anomaly_median_mm=anomaly_median_mm,
            anomaly_outlier_pct=anomaly_outlier_pct,
            debug=debug,
            vis_dir=vis_dir,
            profile=profile,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
        for idx, cam_name in enumerate(cam_names)
    ]

    results = _run_rigid_chamfer_camera_jobs(jobs, camera_workers=camera_workers)

    per_camera: dict[str, dict] = {}
    all_frame_dists: list[float] = []
    for result in results:
        if result.metrics:
            per_camera[result.cam_name] = result.metrics
            print(f"  {result.cam_name}: mean={result.metrics['mean_mm']:.1f}mm  "
                  f"median={result.metrics['median_mm']:.1f}mm  "
                  f"({len(result.frame_dists)} frames)")
        all_frame_dists.extend(result.frame_dists)

    combined = {}
    if all_frame_dists:
        arr = np.array(all_frame_dists)
        combined = {
            "mean_mm": float(arr.mean() * 1000),
            "median_mm": float(np.median(arr) * 1000),
        }
        print(f"\n  Combined: mean={combined['mean_mm']:.1f}mm  "
              f"median={combined['median_mm']:.1f}mm  "
              f"({len(all_frame_dists)} total frames)")

    metrics = {"per_camera": per_camera, "combined": combined}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")

    _tile_chamfer_videos(
        cam_names=cam_names,
        debug=debug,
        vis_dir=vis_dir,
        tile_shape=tile_shape,
        tile_image_size=tile_image_size,
    )

    return metrics


def mv_eval_chamfer(
    cam_names: list[str],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    depth_dirs: list[Path],
    mask_dirs: list[Path],
    faces: np.ndarray,
    mesh_verts: np.ndarray,
    output_path: Path,
    eval_image_size: tuple[int, int] | None = None,
    anomaly_median_mm: float = 30.0,
    anomaly_outlier_pct: float = 10.0,
    debug: int = 0,
    vis_dir: Path | None = None,
    tile_shape: tuple[int, int] = (2, 2),
    tile_image_size: tuple[int, int] | None = None,
    camera_workers: int = 1,
    profile: bool = False,
    progress_interval: float = 0.1,
) -> dict:
    """Compute per-camera distance from visible mesh vertices to depth cloud.

    Vertex visibility is determined by rasterizing the mesh z-buffer via
    pyrender, then checking the mask for occlusion.

    Args:
        cam_names: List of camera names.
        cam_intrinsics: List of (3, 3) intrinsic matrices (at depth resolution).
        cam_extrinsics: List of (4, 4) camera-to-world extrinsic matrices.
        depth_dirs: List of directories with per-frame depth PNGs.
        mask_dirs: List of directories with per-frame mask PNGs.
        faces: (F, 3) mesh face indices (constant topology).
        mesh_verts: (N, V, 3) mesh vertices in world frame.
        output_path: Where to save the JSON metrics.
        eval_image_size: (W, H) to resize depth/mask for evaluation.
        anomaly_median_mm: Threshold for anomaly detection.
        anomaly_outlier_pct: Threshold for anomaly detection.
        debug: If > 0, save per-camera heatmap videos to vis_dir.
        vis_dir: Directory for heatmap videos (used when debug > 0).
        camera_workers: Number of camera processes to run in parallel.
        profile: If true, print per-camera timing breakdowns.
        progress_interval: Fractional progress interval for multi-worker logs.

    Returns:
        Metrics dict (also saved as JSON).
    """
    camera_workers = max(1, int(camera_workers))
    show_progress = camera_workers == 1
    jobs = [
        ChamferCameraJob(
            cam_name=cam_name,
            cam_intrinsics=cam_intrinsics[idx],
            cam_extrinsics=cam_extrinsics[idx],
            depth_dir=depth_dirs[idx],
            mask_dir=mask_dirs[idx],
            faces=faces,
            mesh_verts=mesh_verts,
            eval_image_size=eval_image_size,
            anomaly_median_mm=anomaly_median_mm,
            anomaly_outlier_pct=anomaly_outlier_pct,
            debug=debug,
            vis_dir=vis_dir,
            profile=profile,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
        for idx, cam_name in enumerate(cam_names)
    ]

    results = _run_chamfer_camera_jobs(jobs, camera_workers=camera_workers)

    per_camera: dict[str, dict] = {}
    all_frame_dists: list[float] = []
    for result in results:
        if result.metrics:
            per_camera[result.cam_name] = result.metrics
            print(f"  {result.cam_name}: mean={result.metrics['mean_mm']:.1f}mm  "
                  f"median={result.metrics['median_mm']:.1f}mm  "
                  f"({len(result.frame_dists)} frames)")
        all_frame_dists.extend(result.frame_dists)

    combined = {}
    if all_frame_dists:
        arr = np.array(all_frame_dists)
        combined = {
            "mean_mm": float(arr.mean() * 1000),
            "median_mm": float(np.median(arr) * 1000),
        }
        print(f"\n  Combined: mean={combined['mean_mm']:.1f}mm  "
              f"median={combined['median_mm']:.1f}mm  "
              f"({len(all_frame_dists)} total frames)")

    metrics = {"per_camera": per_camera, "combined": combined}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")

    _tile_chamfer_videos(
        cam_names=cam_names,
        debug=debug,
        vis_dir=vis_dir,
        tile_shape=tile_shape,
        tile_image_size=tile_image_size,
    )

    return metrics
