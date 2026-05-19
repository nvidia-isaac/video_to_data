"""Shared utilities for multiview depth fusion: backproject, merge, downsample, clean."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh

from v2d.common.datatypes import DepthImage
from v2d.common.video import FrameSource
from v2d.mv.math.numpy_fn import depth_to_xyz


def fuse_multiview_depth(
    frame_indices: list[int],
    depth_dirs: list[Path],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    cam_resolutions: list[np.ndarray],
    rgb_dirs: list[Path] | None = None,
    mask_dirs: list[Path] | None = None,
    max_depth: float = 5.0,
    image_scale: float = 0.5,
    voxel_size: float | None = 0.005,
    statistical_outlier_neighbors: int = 20,
    statistical_outlier_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Backproject and merge depth from multiple cameras for selected frames.

    Args:
        frame_indices: Which frames to include (0-based indices into sorted
            depth file listing).
        depth_dirs: Per-camera directories containing depth PNGs.
        cam_intrinsics: Per-camera (3,3) intrinsic matrices (from RigConfig,
            at full resolution).
        cam_extrinsics: Per-camera (4,4) camera-to-world transforms.
        cam_resolutions: Per-camera (2,) arrays [width, height] at full
            resolution (from RigConfig). Used together with *image_scale* to
            compute the target processing size.
        rgb_dirs: Per-camera directories of RGB images (for coloring).
        mask_dirs: Per-camera directories of mask PNGs; masked pixels are excluded.
        max_depth: Discard points beyond this distance (meters).
        image_scale: Scale factor applied to cam_resolutions to determine the
            target processing size. Depth, images, and masks are resized to
            this target; intrinsics are scaled accordingly.
        voxel_size: Voxel size for downsampling (meters). None to skip.
        statistical_outlier_neighbors: Neighbors for statistical outlier removal.
        statistical_outlier_std: Std-ratio for statistical outlier removal.

    Returns:
        (points (P,3) float64, colors (P,3) uint8 or None).
    """
    n_cams = len(depth_dirs)
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    has_color = rgb_dirs is not None

    depth_sources = [FrameSource.from_path(d) for d in depth_dirs]
    image_sources = [FrameSource.from_path(d) for d in rgb_dirs] if has_color else [None] * n_cams
    mask_sources = [FrameSource.from_path(d) for d in mask_dirs] if mask_dirs is not None else [None] * n_cams

    for fi in frame_indices:
        for cam_idx in range(n_cams):
            ds = depth_sources[cam_idx]
            if fi >= ds.n_frames:
                continue

            depth = DepthImage.from_array(ds[fi]).depth

            K = cam_intrinsics[cam_idx].copy()
            T = cam_extrinsics[cam_idx]

            res = cam_resolutions[cam_idx]
            target_W = int(round(res[0] * image_scale))
            target_H = int(round(res[1] * image_scale))

            K[0, :] *= image_scale
            K[1, :] *= image_scale

            if depth.shape[:2] != (target_H, target_W):
                depth = cv2.resize(depth, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

            image = None
            if has_color and image_sources[cam_idx] is not None:
                isrc = image_sources[cam_idx]
                if fi < isrc.n_frames:
                    image = isrc[fi]
                    if image.shape[:2] != (target_H, target_W):
                        image = cv2.resize(image, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

            mask_bool = None
            if mask_sources[cam_idx] is not None:
                msrc = mask_sources[cam_idx]
                if fi < msrc.n_frames:
                    mask_raw = msrc[fi].astype(np.float32) / 255.0
                    if mask_raw.shape[:2] != (target_H, target_W):
                        mask_raw = cv2.resize(
                            mask_raw, (target_W, target_H),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_bool = mask_raw > 0.5

            valid = (depth > 0) & (depth < max_depth)
            if mask_bool is not None:
                valid = valid & mask_bool

            points = depth_to_xyz(depth, K, T, mask=valid)
            all_points.append(points)

            if image is not None:
                colors = image[valid]
                all_colors.append(colors)

    if not all_points:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, np.zeros((0, 3), dtype=np.uint8) if has_color else None

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0) if all_colors else None

    if voxel_size is not None or statistical_outlier_neighbors > 0:
        merged_points, merged_colors = _clean_pointcloud(
            merged_points,
            merged_colors,
            voxel_size=voxel_size,
            nb_neighbors=statistical_outlier_neighbors,
            std_ratio=statistical_outlier_std,
        )

    return merged_points, merged_colors


def _clean_pointcloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    voxel_size: float | None = 0.005,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Downsample and remove outliers using Open3D."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    if nb_neighbors > 0:
        pcd, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

    out_points = np.asarray(pcd.points)
    out_colors = None
    if colors is not None:
        out_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    return out_points, out_colors


def save_colored_ply(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    """Write a PLY file with XYZ + optional RGB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is not None:
        rgba = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])
        cloud = trimesh.PointCloud(vertices=points, colors=rgba)
    else:
        cloud = trimesh.PointCloud(vertices=points)
    cloud.export(str(path))
