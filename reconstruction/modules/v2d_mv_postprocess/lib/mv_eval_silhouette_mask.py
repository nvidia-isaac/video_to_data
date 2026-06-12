from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm
import trimesh

from v2d.common.video import FrameSource, get_video_writer, tile_videos
from v2d.mv.vis.renderer import Renderer


@dataclass(frozen=True)
class RigidSilhouetteMaskCameraJob:
    cam_name: str
    cam_intrinsics: np.ndarray
    cam_extrinsics: np.ndarray
    mask_dir: Path
    canonical_verts: np.ndarray
    faces: np.ndarray
    poses: np.ndarray
    eval_image_size: tuple[int, int] | None
    erosion_kernel: int
    erosion_iterations: int
    min_mask_pixels: int
    debug: int
    vis_dir: Path | None
    profile: bool = False
    show_progress: bool = True
    progress_interval: float = 0.1


@dataclass(frozen=True)
class SilhouetteMaskCameraJob:
    cam_name: str
    cam_intrinsics: np.ndarray
    cam_extrinsics: np.ndarray
    mask_dir: Path
    faces: np.ndarray
    mesh_verts: np.ndarray
    eval_image_size: tuple[int, int] | None
    erosion_kernel: int
    erosion_iterations: int
    min_mask_pixels: int
    debug: int
    vis_dir: Path | None
    profile: bool = False
    show_progress: bool = True
    progress_interval: float = 0.1


@dataclass(frozen=True)
class SilhouetteMaskCameraResult:
    cam_name: str
    metrics: dict
    frame_metrics: list[dict]
    timings: dict[str, float]


def _format_timing(timings: dict[str, float]) -> str:
    return ", ".join(f"{name}={seconds:.2f}s" for name, seconds in timings.items())


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
    if completed == total or fraction + 1e-12 >= next_fraction:
        elapsed = time.perf_counter() - start_time
        pct = 100.0 * fraction
        print(f"{prefix}: {completed}/{total} ({pct:.0f}%), elapsed={elapsed:.1f}s")
        while next_fraction <= fraction + 1e-12:
            next_fraction += interval
    return next_fraction


def _mask_to_bool(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.bool_:
        return mask
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    if float(np.nanmax(mask)) <= 1.0:
        return mask > 0.5
    return mask > 127


def _erode_mask(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if kernel_size <= 1 or iterations <= 0:
        return mask
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=int(iterations))
    return eroded.astype(bool)


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return a tight half-open (x0, y0, x1, y1) bbox around foreground."""
    mask = mask.astype(bool)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y0 = int(np.argmax(rows))
    y1 = int(mask.shape[0] - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(mask.shape[1] - np.argmax(cols[::-1]))
    return x0, y0, x1, y1


def bbox_area(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_intersection_area(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> int:
    if a is None or b is None:
        return 0
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    return max(0, ix1 - ix0) * max(0, iy1 - iy0)


def sam2_bbox_in_render_bbox_ratio(
    sam2_bbox: tuple[int, int, int, int] | None,
    render_bbox: tuple[int, int, int, int] | None,
) -> float | None:
    sam2_area = bbox_area(sam2_bbox)
    if sam2_area <= 0:
        return None
    return float(bbox_intersection_area(sam2_bbox, render_bbox) / sam2_area)


def compute_silhouette_mask_frame_metrics(
    sam2_mask: np.ndarray,
    render_mask: np.ndarray,
    *,
    frame_idx: int = 0,
    erosion_kernel: int = 3,
    erosion_iterations: int = 1,
    min_mask_pixels: int = 10,
) -> dict:
    """Compute asymmetric SAM2-vs-render silhouette residuals for one frame."""
    sam2 = _mask_to_bool(sam2_mask)
    rendered = render_mask.astype(bool)
    if sam2.shape != rendered.shape:
        raise ValueError(
            f"mask shape mismatch: sam2={sam2.shape}, render={rendered.shape}"
        )

    mask_pixels = int(sam2.sum())
    render_pixels = int(rendered.sum())
    sam2_bbox = mask_to_bbox(sam2)
    render_bbox = mask_to_bbox(rendered)
    sam2_bbox_pixels = bbox_area(sam2_bbox)
    render_bbox_pixels = bbox_area(render_bbox)
    bbox_intersection_pixels = bbox_intersection_area(sam2_bbox, render_bbox)
    bbox_containment = sam2_bbox_in_render_bbox_ratio(sam2_bbox, render_bbox)
    if mask_pixels < min_mask_pixels:
        return {
            "frame_idx": int(frame_idx),
            "skipped": True,
            "reason": f"sam2_mask_pixels<{min_mask_pixels}",
            "sam2_mask_pixels": mask_pixels,
            "render_mask_pixels": render_pixels,
            "sam2_bbox": sam2_bbox,
            "render_bbox": render_bbox,
            "bbox_intersection_pixels": bbox_intersection_pixels,
            "sam2_bbox_pixels": sam2_bbox_pixels,
            "render_bbox_pixels": render_bbox_pixels,
            "sam2_bbox_in_render_bbox_ratio": None,
        }

    unexplained = sam2 & ~rendered
    unexplained_eroded = _erode_mask(
        unexplained,
        erosion_kernel,
        erosion_iterations,
    )
    over_render = rendered & ~sam2
    over_render_eroded = _erode_mask(
        over_render,
        erosion_kernel,
        erosion_iterations,
    )

    unexplained_pixels = int(unexplained.sum())
    unexplained_eroded_pixels = int(unexplained_eroded.sum())
    over_render_pixels = int(over_render.sum())
    over_render_eroded_pixels = int(over_render_eroded.sum())

    return {
        "frame_idx": int(frame_idx),
        "skipped": False,
        "sam2_mask_pixels": mask_pixels,
        "render_mask_pixels": render_pixels,
        "unexplained_sam2_pixels": unexplained_pixels,
        "unexplained_sam2_eroded_pixels": unexplained_eroded_pixels,
        "unexplained_sam2_ratio": float(unexplained_eroded_pixels / mask_pixels),
        "over_render_pixels": over_render_pixels,
        "over_render_eroded_pixels": over_render_eroded_pixels,
        "over_render_ratio": float(
            over_render_eroded_pixels / render_pixels if render_pixels > 0 else 0.0
        ),
        "sam2_bbox": sam2_bbox,
        "render_bbox": render_bbox,
        "bbox_intersection_pixels": bbox_intersection_pixels,
        "sam2_bbox_pixels": sam2_bbox_pixels,
        "render_bbox_pixels": render_bbox_pixels,
        "sam2_bbox_in_render_bbox_ratio": bbox_containment,
    }


def _summarize_frame_metrics(frame_metrics: list[dict]) -> dict:
    valid = [m for m in frame_metrics if not m.get("skipped", False)]
    skipped = [m for m in frame_metrics if m.get("skipped", False)]
    summary: dict = {
        "frames_total": len(frame_metrics),
        "frames_evaluated": len(valid),
        "frames_skipped": len(skipped),
        "skipped_reasons": {},
    }
    for metric in skipped:
        reason = metric.get("reason", "unknown")
        summary["skipped_reasons"][reason] = summary["skipped_reasons"].get(reason, 0) + 1

    if not valid:
        summary.update(
            {
                "mean_unexplained_sam2_ratio": None,
                "median_unexplained_sam2_ratio": None,
                "total_unexplained_sam2_ratio": None,
                "mean_over_render_ratio": None,
                "median_over_render_ratio": None,
                "total_over_render_ratio": None,
                "mean_sam2_bbox_in_render_bbox_ratio": None,
                "median_sam2_bbox_in_render_bbox_ratio": None,
                "total_sam2_bbox_in_render_bbox_ratio": None,
                "total_sam2_bbox_pixels": 0,
                "total_bbox_intersection_pixels": 0,
            }
        )
        return summary

    unexplained_ratios = np.array(
        [m["unexplained_sam2_ratio"] for m in valid],
        dtype=float,
    )
    over_ratios = np.array([m["over_render_ratio"] for m in valid], dtype=float)
    bbox_ratios = np.array(
        [m["sam2_bbox_in_render_bbox_ratio"] for m in valid],
        dtype=float,
    )
    total_mask_pixels = sum(int(m["sam2_mask_pixels"]) for m in valid)
    total_render_pixels = sum(int(m["render_mask_pixels"]) for m in valid)
    total_unexplained_pixels = sum(int(m["unexplained_sam2_eroded_pixels"]) for m in valid)
    total_over_pixels = sum(int(m["over_render_eroded_pixels"]) for m in valid)
    total_sam2_bbox_pixels = sum(int(m["sam2_bbox_pixels"]) for m in valid)
    total_bbox_intersection_pixels = sum(int(m["bbox_intersection_pixels"]) for m in valid)

    summary.update(
        {
            "mean_unexplained_sam2_ratio": float(unexplained_ratios.mean()),
            "median_unexplained_sam2_ratio": float(np.median(unexplained_ratios)),
            "total_unexplained_sam2_ratio": float(total_unexplained_pixels / total_mask_pixels),
            "mean_over_render_ratio": float(over_ratios.mean()),
            "median_over_render_ratio": float(np.median(over_ratios)),
            "total_over_render_ratio": float(
                total_over_pixels / total_render_pixels if total_render_pixels > 0 else 0.0
            ),
            "mean_sam2_bbox_in_render_bbox_ratio": float(bbox_ratios.mean()),
            "median_sam2_bbox_in_render_bbox_ratio": float(np.median(bbox_ratios)),
            "total_sam2_bbox_in_render_bbox_ratio": float(
                total_bbox_intersection_pixels / total_sam2_bbox_pixels
            ),
            "total_sam2_mask_pixels": int(total_mask_pixels),
            "total_render_mask_pixels": int(total_render_pixels),
            "total_unexplained_sam2_eroded_pixels": int(total_unexplained_pixels),
            "total_over_render_eroded_pixels": int(total_over_pixels),
            "total_sam2_bbox_pixels": int(total_sam2_bbox_pixels),
            "total_bbox_intersection_pixels": int(total_bbox_intersection_pixels),
        }
    )
    return summary


def _prepare_eval_geometry(
    mask_source: FrameSource,
    cam_intrinsics: np.ndarray,
    eval_image_size: tuple[int, int] | None,
) -> tuple[int, int, np.ndarray]:
    W_orig, H_orig = mask_source.image_size
    K_eval = cam_intrinsics.copy()
    if eval_image_size is None:
        return W_orig, H_orig, K_eval

    W_eval, H_eval = eval_image_size
    sx, sy = W_eval / W_orig, H_eval / H_orig
    K_eval[0, :] *= sx
    K_eval[1, :] *= sy
    return W_eval, H_eval, K_eval


def _read_eval_mask(
    mask_source: FrameSource,
    frame_idx: int,
    image_size: tuple[int, int],
) -> np.ndarray:
    mask = _mask_to_bool(mask_source[frame_idx])
    W_eval, H_eval = image_size
    if mask.shape[:2] != (H_eval, W_eval):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (W_eval, H_eval),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def _debug_canvas(
    sam2_mask: np.ndarray,
    render_mask: np.ndarray,
    metric: dict,
) -> np.ndarray:
    unexplained = sam2_mask & ~render_mask
    over_render = render_mask & ~sam2_mask

    H, W = sam2_mask.shape
    panel_size = (max(1, W // 2), max(1, H // 2))

    def _scale_bbox(
        bbox: tuple[int, int, int, int] | list[int] | None,
        src_size: tuple[int, int],
        dst_size: tuple[int, int],
    ) -> tuple[int, int, int, int] | None:
        if bbox is None:
            return None
        src_w, src_h = src_size
        dst_w, dst_h = dst_size
        sx = dst_w / src_w
        sy = dst_h / src_h
        x0, y0, x1, y1 = bbox
        return (
            max(0, min(dst_w - 1, int(np.floor(x0 * sx)))),
            max(0, min(dst_h - 1, int(np.floor(y0 * sy)))),
            max(0, min(dst_w - 1, int(np.ceil(x1 * sx)) - 1)),
            max(0, min(dst_h - 1, int(np.ceil(y1 * sy)) - 1)),
        )

    def _panel(
        mask: np.ndarray,
        color: tuple[int, int, int],
        label: str,
        bbox: tuple[int, int, int, int] | list[int] | None = None,
        bbox_color: tuple[int, int, int] = (255, 255, 0),
        extra_bbox: tuple[int, int, int, int] | list[int] | None = None,
        extra_bbox_color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        mask = cv2.resize(
            mask.astype(np.uint8),
            panel_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
        panel[mask] = color
        scaled_bbox = _scale_bbox(bbox, (W, H), panel_size)
        if scaled_bbox is not None:
            x0, y0, x1, y1 = scaled_bbox
            cv2.rectangle(panel, (x0, y0), (x1, y1), bbox_color, 2)
        scaled_extra_bbox = _scale_bbox(extra_bbox, (W, H), panel_size)
        if scaled_extra_bbox is not None:
            x0, y0, x1, y1 = scaled_extra_bbox
            cv2.rectangle(panel, (x0, y0), (x1, y1), extra_bbox_color, 2)
        cv2.putText(
            panel,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return panel

    top = np.concatenate(
        [
            _panel(
                sam2_mask,
                (255, 255, 255),
                "SAM2 + boxes",
                metric.get("sam2_bbox"),
                extra_bbox=metric.get("render_bbox"),
            ),
            _panel(
                render_mask,
                (0, 220, 0),
                "Render",
                metric.get("render_bbox"),
                bbox_color=(0, 255, 0),
            ),
        ],
        axis=1,
    )
    bottom = np.concatenate(
        [
            _panel(unexplained, (255, 60, 60), "SAM2 - render"),
            _panel(over_render, (80, 160, 255), "Render - SAM2"),
        ],
        axis=1,
    )
    canvas = np.concatenate([top, bottom], axis=0)
    if metric.get("skipped", False):
        text = f"Frame {metric['frame_idx']}: {metric.get('reason', 'skipped')}"
    else:
        text = (
            f"Frame {metric['frame_idx']}: "
            f"unexpl={metric['unexplained_sam2_ratio']:.3f} "
            f"over={metric['over_render_ratio']:.3f} "
            f"bbox={metric['sam2_bbox_in_render_bbox_ratio']:.3f}"
        )
    cv2.putText(
        canvas,
        text,
        (10, canvas.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return canvas


def _tile_silhouette_videos(
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

    tiled_path = vis_dir / "tiled_silhouette_mask.mp4"
    print(f"Tiling {len(vis_paths)} silhouette videos into {tiled_path}...")
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


def _eval_rigid_silhouette_mask_camera(
    job: RigidSilhouetteMaskCameraJob,
) -> SilhouetteMaskCameraResult:
    timings = {"setup": 0.0, "read": 0.0, "render_depth": 0.0, "metric": 0.0, "debug_write": 0.0}
    total_start = time.perf_counter()
    mask_source = FrameSource.from_path(job.mask_dir)
    writer = None
    frame_metrics: list[dict] = []

    try:
        n_frames = job.poses.shape[0]
        if mask_source.n_frames != n_frames:
            raise ValueError(
                f"camera {job.cam_name}: frame count mismatch "
                f"(mask={mask_source.n_frames}, expected={n_frames})"
            )

        setup_start = time.perf_counter()
        W_eval, H_eval, K_eval = _prepare_eval_geometry(
            mask_source,
            job.cam_intrinsics,
            job.eval_image_size,
        )
        canonical_mesh = trimesh.Trimesh(
            vertices=job.canonical_verts,
            faces=job.faces,
            process=False,
        )
        if job.debug > 0 and job.vis_dir:
            job.vis_dir.mkdir(parents=True, exist_ok=True)
            writer = get_video_writer(job.vis_dir / f"{job.cam_name}.mp4", fps=30, crf=23)
        timings["setup"] += time.perf_counter() - setup_start

        with Renderer(image_size=(W_eval, H_eval)) as renderer:
            object_handle = None
            if n_frames > 0:
                object_handle = renderer.add_persistent_mesh(
                    canonical_mesh,
                    pose=job.poses[0],
                )

            progress = tqdm(
                range(n_frames),
                desc=f"Silhouette {job.cam_name}",
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
                    prefix=f"Silhouette {job.cam_name}",
                    completed=frame_idx + 1,
                    total=n_frames,
                    start_time=progress_start,
                    next_fraction=next_progress_fraction,
                    interval=progress_interval,
                )

            for i in progress:
                read_start = time.perf_counter()
                sam2_mask = _read_eval_mask(mask_source, i, (W_eval, H_eval))
                timings["read"] += time.perf_counter() - read_start

                if object_handle is not None:
                    renderer.set_persistent_mesh_pose(object_handle, job.poses[i])

                render_start = time.perf_counter()
                render_mask = renderer.render_depth([], K_eval, job.cam_extrinsics) > 0
                timings["render_depth"] += time.perf_counter() - render_start

                metric_start = time.perf_counter()
                metric = compute_silhouette_mask_frame_metrics(
                    sam2_mask,
                    render_mask,
                    frame_idx=i,
                    erosion_kernel=job.erosion_kernel,
                    erosion_iterations=job.erosion_iterations,
                    min_mask_pixels=job.min_mask_pixels,
                )
                frame_metrics.append(metric)
                timings["metric"] += time.perf_counter() - metric_start

                if writer is not None:
                    debug_start = time.perf_counter()
                    writer.write_frame(_debug_canvas(sam2_mask, render_mask, metric))
                    timings["debug_write"] += time.perf_counter() - debug_start

                _log_frame_progress(i)
    finally:
        if writer is not None:
            writer.close()
        mask_source.close()

    timings["total"] = time.perf_counter() - total_start
    if job.profile:
        print(
            f"Profile silhouette object {job.cam_name}: "
            f"frames={job.poses.shape[0]}, {_format_timing(timings)}"
        )

    return SilhouetteMaskCameraResult(
        cam_name=job.cam_name,
        metrics=_summarize_frame_metrics(frame_metrics),
        frame_metrics=frame_metrics,
        timings=timings,
    )


def _eval_silhouette_mask_camera(job: SilhouetteMaskCameraJob) -> SilhouetteMaskCameraResult:
    timings = {"setup": 0.0, "read": 0.0, "mesh_build": 0.0, "render_depth": 0.0, "metric": 0.0, "debug_write": 0.0}
    total_start = time.perf_counter()
    mask_source = FrameSource.from_path(job.mask_dir)
    writer = None
    frame_metrics: list[dict] = []

    try:
        n_frames = job.mesh_verts.shape[0]
        if mask_source.n_frames != n_frames:
            raise ValueError(
                f"camera {job.cam_name}: frame count mismatch "
                f"(mask={mask_source.n_frames}, expected={n_frames})"
            )

        setup_start = time.perf_counter()
        W_eval, H_eval, K_eval = _prepare_eval_geometry(
            mask_source,
            job.cam_intrinsics,
            job.eval_image_size,
        )
        if job.debug > 0 and job.vis_dir:
            job.vis_dir.mkdir(parents=True, exist_ok=True)
            writer = get_video_writer(job.vis_dir / f"{job.cam_name}.mp4", fps=30, crf=23)
        timings["setup"] += time.perf_counter() - setup_start

        with Renderer(image_size=(W_eval, H_eval)) as renderer:
            progress = tqdm(
                range(n_frames),
                desc=f"Silhouette {job.cam_name}",
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
                    prefix=f"Silhouette {job.cam_name}",
                    completed=frame_idx + 1,
                    total=n_frames,
                    start_time=progress_start,
                    next_fraction=next_progress_fraction,
                    interval=progress_interval,
                )

            for i in progress:
                read_start = time.perf_counter()
                sam2_mask = _read_eval_mask(mask_source, i, (W_eval, H_eval))
                timings["read"] += time.perf_counter() - read_start

                mesh_start = time.perf_counter()
                frame_mesh = trimesh.Trimesh(
                    vertices=job.mesh_verts[i],
                    faces=job.faces,
                    process=False,
                )
                timings["mesh_build"] += time.perf_counter() - mesh_start

                render_start = time.perf_counter()
                render_mask = renderer.render_depth([frame_mesh], K_eval, job.cam_extrinsics) > 0
                timings["render_depth"] += time.perf_counter() - render_start

                metric_start = time.perf_counter()
                metric = compute_silhouette_mask_frame_metrics(
                    sam2_mask,
                    render_mask,
                    frame_idx=i,
                    erosion_kernel=job.erosion_kernel,
                    erosion_iterations=job.erosion_iterations,
                    min_mask_pixels=job.min_mask_pixels,
                )
                frame_metrics.append(metric)
                timings["metric"] += time.perf_counter() - metric_start

                if writer is not None:
                    debug_start = time.perf_counter()
                    writer.write_frame(_debug_canvas(sam2_mask, render_mask, metric))
                    timings["debug_write"] += time.perf_counter() - debug_start

                _log_frame_progress(i)
    finally:
        if writer is not None:
            writer.close()
        mask_source.close()

    timings["total"] = time.perf_counter() - total_start
    if job.profile:
        print(
            f"Profile silhouette {job.cam_name}: "
            f"frames={job.mesh_verts.shape[0]}, {_format_timing(timings)}"
        )

    return SilhouetteMaskCameraResult(
        cam_name=job.cam_name,
        metrics=_summarize_frame_metrics(frame_metrics),
        frame_metrics=frame_metrics,
        timings=timings,
    )


def _run_rigid_silhouette_mask_camera_jobs(
    jobs: list[RigidSilhouetteMaskCameraJob],
    camera_workers: int,
) -> list[SilhouetteMaskCameraResult]:
    if camera_workers <= 1:
        return [_eval_rigid_silhouette_mask_camera(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=camera_workers) as executor:
        return list(executor.map(_eval_rigid_silhouette_mask_camera, jobs, chunksize=1))


def _run_silhouette_mask_camera_jobs(
    jobs: list[SilhouetteMaskCameraJob],
    camera_workers: int,
) -> list[SilhouetteMaskCameraResult]:
    if camera_workers <= 1:
        return [_eval_silhouette_mask_camera(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=camera_workers) as executor:
        return list(executor.map(_eval_silhouette_mask_camera, jobs, chunksize=1))


def _write_metrics(
    *,
    cam_names: list[str],
    results: list[SilhouetteMaskCameraResult],
    output_path: Path,
    debug: int,
    vis_dir: Path | None,
    tile_shape: tuple[int, int],
    tile_image_size: tuple[int, int] | None,
) -> dict:
    per_camera: dict[str, dict] = {}
    all_frames: list[dict] = []
    for result in results:
        per_camera[result.cam_name] = result.metrics
        all_frames.extend(result.frame_metrics)
        primary = result.metrics.get("median_unexplained_sam2_ratio")
        diagnostic = result.metrics.get("median_over_render_ratio")
        bbox_containment = result.metrics.get("median_sam2_bbox_in_render_bbox_ratio")
        if primary is None:
            print(
                f"  {result.cam_name}: no valid frames "
                f"({result.metrics['frames_skipped']} skipped)"
            )
        else:
            print(
                f"  {result.cam_name}: unexplained_median={primary:.4f} "
                f"over_render_median={diagnostic:.4f} "
                f"bbox_containment_median={bbox_containment:.4f} "
                f"({result.metrics['frames_evaluated']} frames)"
            )

    metrics = {
        "metric": "unexplained_sam2_ratio",
        "diagnostic_metric": "over_render_ratio",
        "combined": _summarize_frame_metrics(all_frames),
        "per_camera": per_camera,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {output_path}")

    _tile_silhouette_videos(
        cam_names=cam_names,
        debug=debug,
        vis_dir=vis_dir,
        tile_shape=tile_shape,
        tile_image_size=tile_image_size,
    )
    return metrics


def mv_eval_silhouette_mask_rigid_object(
    cam_names: list[str],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    mask_dirs: list[Path],
    canonical_verts: np.ndarray,
    faces: np.ndarray,
    poses: np.ndarray,
    output_path: Path,
    eval_image_size: tuple[int, int] | None = None,
    erosion_kernel: int = 3,
    erosion_iterations: int = 1,
    min_mask_pixels: int = 10,
    debug: int = 0,
    vis_dir: Path | None = None,
    tile_shape: tuple[int, int] = (2, 2),
    tile_image_size: tuple[int, int] | None = None,
    camera_workers: int = 1,
    profile: bool = False,
    progress_interval: float = 0.1,
) -> dict:
    camera_workers = max(1, int(camera_workers))
    show_progress = camera_workers == 1
    jobs = [
        RigidSilhouetteMaskCameraJob(
            cam_name=cam_name,
            cam_intrinsics=cam_intrinsics[idx],
            cam_extrinsics=cam_extrinsics[idx],
            mask_dir=mask_dirs[idx],
            canonical_verts=canonical_verts,
            faces=faces,
            poses=poses,
            eval_image_size=eval_image_size,
            erosion_kernel=erosion_kernel,
            erosion_iterations=erosion_iterations,
            min_mask_pixels=min_mask_pixels,
            debug=debug,
            vis_dir=vis_dir,
            profile=profile,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
        for idx, cam_name in enumerate(cam_names)
    ]
    results = _run_rigid_silhouette_mask_camera_jobs(jobs, camera_workers=camera_workers)
    return _write_metrics(
        cam_names=cam_names,
        results=results,
        output_path=output_path,
        debug=debug,
        vis_dir=vis_dir,
        tile_shape=tile_shape,
        tile_image_size=tile_image_size,
    )


def mv_eval_silhouette_mask(
    cam_names: list[str],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    mask_dirs: list[Path],
    faces: np.ndarray,
    mesh_verts: np.ndarray,
    output_path: Path,
    eval_image_size: tuple[int, int] | None = None,
    erosion_kernel: int = 3,
    erosion_iterations: int = 1,
    min_mask_pixels: int = 10,
    debug: int = 0,
    vis_dir: Path | None = None,
    tile_shape: tuple[int, int] = (2, 2),
    tile_image_size: tuple[int, int] | None = None,
    camera_workers: int = 1,
    profile: bool = False,
    progress_interval: float = 0.1,
) -> dict:
    camera_workers = max(1, int(camera_workers))
    show_progress = camera_workers == 1
    jobs = [
        SilhouetteMaskCameraJob(
            cam_name=cam_name,
            cam_intrinsics=cam_intrinsics[idx],
            cam_extrinsics=cam_extrinsics[idx],
            mask_dir=mask_dirs[idx],
            faces=faces,
            mesh_verts=mesh_verts,
            eval_image_size=eval_image_size,
            erosion_kernel=erosion_kernel,
            erosion_iterations=erosion_iterations,
            min_mask_pixels=min_mask_pixels,
            debug=debug,
            vis_dir=vis_dir,
            profile=profile,
            show_progress=show_progress,
            progress_interval=progress_interval,
        )
        for idx, cam_name in enumerate(cam_names)
    ]
    results = _run_silhouette_mask_camera_jobs(jobs, camera_workers=camera_workers)
    return _write_metrics(
        cam_names=cam_names,
        results=results,
        output_path=output_path,
        debug=debug,
        vis_dir=vis_dir,
        tile_shape=tile_shape,
        tile_image_size=tile_image_size,
    )
