# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

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

    Returns:
        Metrics dict (also saved as JSON).
    """
    from scipy.spatial import cKDTree

    n_frames = mesh_verts.shape[0]

    per_camera_sources: list[tuple[FrameSource, FrameSource]] = []
    for cam_idx, cam_name in enumerate(cam_names):
        depth_source = FrameSource.from_path(depth_dirs[cam_idx])
        mask_source = FrameSource.from_path(mask_dirs[cam_idx])
        if depth_source.n_frames != n_frames or mask_source.n_frames != n_frames:
            raise ValueError(
                f"camera {cam_name}: frame count mismatch "
                f"(depth={depth_source.n_frames}, mask={mask_source.n_frames}, expected={n_frames})"
            )
        per_camera_sources.append((depth_source, mask_source))

    per_camera: dict[str, dict] = {}
    all_frame_dists: list[float] = []

    for cam_idx, cam_name in enumerate(cam_names):
        K = cam_intrinsics[cam_idx]
        T = cam_extrinsics[cam_idx]
        depth_source, mask_source = per_camera_sources[cam_idx]

        first_depth = DepthImage.from_array(depth_source[0]).depth
        H_orig, W_orig = first_depth.shape[:2]

        if eval_image_size is not None:
            W_eval, H_eval = eval_image_size
            sx, sy = W_eval / W_orig, H_eval / H_orig
            K_eval = K.copy()
            K_eval[0, :] *= sx
            K_eval[1, :] *= sy
        else:
            W_eval, H_eval = W_orig, H_orig
            K_eval = K

        cam_dists: list[float] = []
        writer = None
        if debug > 0 and vis_dir:
            vis_dir.mkdir(parents=True, exist_ok=True)
            writer = get_video_writer(vis_dir / f"{cam_name}.mp4", fps=30, crf=23)

        def _write_placeholder(frame_idx: int, reason: str) -> None:
            if writer is None:
                return
            canvas = np.zeros((H_eval, W_eval, 3), dtype=np.uint8)
            label = f"Frame {frame_idx}: {reason}"
            cv2.putText(canvas, label, (10, H_eval - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write_frame(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        with Renderer(image_size=(W_eval, H_eval)) as renderer:
            for i in tqdm(range(n_frames), desc=f"Chamfer {cam_name}"):
                depth = DepthImage.from_array(depth_source[i]).depth
                mask_raw = mask_source[i].astype(np.float32) / 255.0

                if eval_image_size is not None:
                    depth = cv2.resize(depth, (W_eval, H_eval), interpolation=cv2.INTER_LINEAR)
                    mask_raw = cv2.resize(mask_raw, (W_eval, H_eval), interpolation=cv2.INTER_NEAREST)
                elif mask_raw.shape[:2] != depth.shape[:2]:
                    mask_raw = cv2.resize(
                        mask_raw, (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask = (mask_raw > 0.5) & (depth > 0.001)

                pts_world = depth_to_xyz(depth, K_eval, T, mask=mask)
                if pts_world.shape[0] < 10:
                    _write_placeholder(i, "no valid depth points")
                    continue

                verts_np = mesh_verts[i]
                frame_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces, process=False)
                mesh_zbuf = renderer.render_depth([frame_mesh], K_eval, T)

                vis = visible_vertices(verts_np, mesh_zbuf, K_eval, T)

                H_m, W_m = mask.shape[:2]
                uv_int, in_bounds = xyz_to_uv(verts_np, K_eval, T, image_size=(W_m, H_m))
                mask_vis = np.zeros(verts_np.shape[0], dtype=bool)
                ib_idx = np.where(in_bounds)[0]
                mask_vis[ib_idx] = mask[uv_int[ib_idx, 1], uv_int[ib_idx, 0]]
                vis = vis & mask_vis

                vis_verts = verts_np[vis]

                if vis_verts.shape[0] < 10:
                    _write_placeholder(i, "no visible vertices")
                    continue

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

                anomaly_msg = None
                if median_mm > anomaly_median_mm or outlier_pct > anomaly_outlier_pct:
                    anomaly_msg = (
                        f"ANOMALY: median={median_mm:.1f}mm "
                        f"outliers={n_outliers}/{len(dists_mm)} ({outlier_pct:.0f}%)"
                    )
                    tqdm.write(f"  {cam_name} frame {i}: {anomaly_msg}")

                if writer is not None:
                    heatmap = _render_vertex_heatmap(
                        depth.shape[:2], uv_int[vis], per_vert_dists,
                        frame_idx=i, anomaly_msg=anomaly_msg,
                    )
                    writer.write_frame(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))

        if writer is not None:
            writer.close()

        if cam_dists:
            arr = np.array(cam_dists)
            per_camera[cam_name] = {
                "mean_mm": float(arr.mean() * 1000),
                "median_mm": float(np.median(arr) * 1000),
                "per_frame_mm": [float(v * 1000) for v in cam_dists],
            }
            all_frame_dists.extend(cam_dists)
            print(f"  {cam_name}: mean={per_camera[cam_name]['mean_mm']:.1f}mm  "
                  f"median={per_camera[cam_name]['median_mm']:.1f}mm  "
                  f"({len(cam_dists)} frames)")

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

    if debug > 0 and vis_dir and len(cam_names) > 1:
        vis_paths = [vis_dir / f"{name}.mp4" for name in cam_names if (vis_dir / f"{name}.mp4").exists()]
        if len(vis_paths) > 1:
            tiled_path = vis_dir / "tiled_chamfer.mp4"
            print(f"Tiling {len(vis_paths)} chamfer videos into {tiled_path}...")
            try:
                tile_videos(
                    sources=[FrameSource.from_path(p) for p in vis_paths],
                    output_path=tiled_path,
                    tile_shape=tile_shape,
                    output_image_size=tile_image_size,
                    video_names=[p.stem for p in vis_paths],
                )
            except Exception as e:
                print(f"WARNING: tile_videos failed: {e}. Skipping tiled viz.")

    return metrics
