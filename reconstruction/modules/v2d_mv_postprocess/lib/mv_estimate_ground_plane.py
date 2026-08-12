# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Estimate the ground plane from multiview depth + MHR foot keypoints."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from v2d.mv.rig import RigConfig

from v2d.mv.postprocess.lib.fuse_depth_pointcloud import (
    fuse_multiview_depth,
    save_colored_ply,
)

logger = logging.getLogger(__name__)

MHR_FOOT_INDICES = list(range(15, 21))


def _ransac_plane(
    points: np.ndarray,
    threshold: float = 0.02,
    n_iterations: int = 1000,
    max_normal_angle: float = 15.0,
    up_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """3-point RANSAC plane fit with near-horizontal constraint.

    Args:
        points: (N, 3) candidate floor points.
        threshold: Inlier distance threshold (meters).
        n_iterations: Number of RANSAC iterations.
        max_normal_angle: Maximum angle (degrees) between fitted normal and vertical.
        up_hint: (3,) approximate up direction. If None, uses [0, 0, 1].

    Returns:
        (plane_coeffs [a,b,c,d], inlier_mask (N,) bool).
    """
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])
    up_hint = up_hint / np.linalg.norm(up_hint)
    cos_limit = np.cos(np.radians(max_normal_angle))

    N = len(points)
    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    best_plane = np.array([0.0, 0.0, 1.0, 0.0])

    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len

        if abs(np.dot(normal, up_hint)) < cos_limit:
            continue

        d = -np.dot(normal, p0)
        dists = np.abs(points @ normal + d)
        inliers = dists < threshold
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_plane = np.append(normal, d)

    if best_count >= 3:
        inlier_pts = points[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        normal = Vt[-1]
        normal /= np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        best_plane = np.append(normal, d)
        best_inliers = np.abs(points @ normal + d) < threshold

    return best_plane, best_inliers


def estimate_ground_plane(
    depth_dirs: list[Path],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    cam_resolutions: list[np.ndarray],
    rig: RigConfig,
    output_path: Path,
    mhr_params_path: Path | None = None,
    mask_dirs: list[Path] | None = None,
    rgb_dirs: list[Path] | None = None,
    n_sample_frames: int = 10,
    height_band: float = 0.3,
    ransac_threshold: float = 0.02,
    ransac_iterations: int = 1000,
    max_normal_angle: float = 15.0,
    max_depth: float = 5.0,
    image_scale: float = 0.5,
) -> dict:
    """Estimate the ground plane from fused depth + MHR foot prior.

    Returns:
        Dict with ``plane`` [a,b,c,d], ``n_inliers``, ``foot_plane_dist_stats``.
    """
    from v2d.common.video import FrameSource

    n_total = FrameSource.from_path(depth_dirs[0]).n_frames
    if n_total == 0:
        raise ValueError(f"No depth frames found in {depth_dirs[0]}")

    sample_indices = np.linspace(0, n_total - 1, min(n_sample_frames, n_total))
    sample_indices = np.unique(sample_indices.astype(int)).tolist()

    logger.info("Fusing depth for %d frames across %d cameras …",
                len(sample_indices), len(depth_dirs))
    points, _ = fuse_multiview_depth(
        frame_indices=sample_indices,
        depth_dirs=depth_dirs,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        cam_resolutions=cam_resolutions,
        mask_dirs=mask_dirs,
        max_depth=max_depth,
        image_scale=image_scale,
        voxel_size=0.01,
        statistical_outlier_neighbors=20,
        statistical_outlier_std=2.0,
    )
    logger.info("Fused point cloud: %d points", len(points))

    ref_cam = rig.get_camera_by_name("front_stereo_camera_left")
    ref_cam_pos = ref_cam.param.T[:3, 3]

    foot_kps_world: np.ndarray | None = None
    if mhr_params_path is not None and mhr_params_path.exists():
        mhr_params = torch.load(str(mhr_params_path), weights_only=False, map_location="cpu")
        kp3d = mhr_params["pred_keypoints_3d"].cpu().numpy()  # (N, K, 3)
        foot_kps_world = kp3d[:, MHR_FOOT_INDICES, :].reshape(-1, 3)
        logger.info("Loaded %d foot keypoints from MHR", len(foot_kps_world))

    up_hint = _estimate_up_from_camera(ref_cam.param.T)
    logger.info("Up direction from camera: [%.4f, %.4f, %.4f]",
                up_hint[0], up_hint[1], up_hint[2])

    if foot_kps_world is not None and len(foot_kps_world) > 0:
        foot_height = np.median(foot_kps_world @ up_hint)
        logger.info("Foot height (median along up): %.4f m", foot_height)
        heights = points @ up_hint
        band_mask = np.abs(heights - foot_height) < height_band
        candidate_points = points[band_mask]
        logger.info("Floor candidates after foot-height filter: %d / %d",
                     len(candidate_points), len(points))
    else:
        logger.warning("No MHR foot keypoints — using histogram fallback for floor band")
        heights = points @ up_hint
        hist, bin_edges = np.histogram(heights, bins=200)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        lowest_peak_idx = np.argmax(hist[:len(hist) // 3])
        foot_height = bin_centers[lowest_peak_idx]
        band_mask = np.abs(heights - foot_height) < height_band
        candidate_points = points[band_mask]
        logger.info("Histogram floor height: %.4f m, candidates: %d",
                     foot_height, len(candidate_points))

    if len(candidate_points) < 10:
        logger.warning("Too few floor candidates (%d), using all points", len(candidate_points))
        candidate_points = points

    plane, inlier_mask = _ransac_plane(
        candidate_points,
        threshold=ransac_threshold,
        n_iterations=ransac_iterations,
        max_normal_angle=max_normal_angle,
        up_hint=up_hint,
    )

    normal = plane[:3]
    if np.dot(normal, ref_cam_pos) + plane[3] < 0:
        plane = -plane

    foot_dist_stats = {}
    if foot_kps_world is not None and len(foot_kps_world) > 0:
        signed_dists = foot_kps_world @ plane[:3] + plane[3]
        foot_dist_stats = {
            "mean": float(np.mean(signed_dists)),
            "std": float(np.std(signed_dists)),
            "min": float(np.min(signed_dists)),
            "max": float(np.max(signed_dists)),
            "median": float(np.median(signed_dists)),
        }
        if abs(foot_dist_stats["median"]) > 0.1:
            logger.warning(
                "Median foot-to-plane distance is %.3f m — plane may be inaccurate",
                foot_dist_stats["median"],
            )

    result = {
        "plane": plane.tolist(),
        "n_inliers": int(inlier_mask.sum()),
        "n_candidates": len(candidate_points),
        "n_total_points": len(points),
        "foot_plane_dist_stats": foot_dist_stats,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Ground plane saved to %s", output_path)
    logger.info("  plane: [%.6f, %.6f, %.6f, %.6f]", *plane)
    logger.info("  inliers: %d / %d candidates", inlier_mask.sum(), len(candidate_points))

    _export_debug_ply(
        output_path.parent / "ground_plane_debug.ply",
        plane=plane,
        depth_dirs=depth_dirs,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        cam_resolutions=cam_resolutions,
        rgb_dirs=rgb_dirs,
        max_depth=max_depth,
        image_scale=image_scale,
    )

    return result


def _export_debug_ply(
    ply_path: Path,
    plane: np.ndarray,
    depth_dirs: list[Path],
    cam_intrinsics: list[np.ndarray],
    cam_extrinsics: list[np.ndarray],
    cam_resolutions: list[np.ndarray],
    rgb_dirs: list[Path] | None,
    max_depth: float,
    image_scale: float,
) -> None:
    """Save a PLY with frame-0 colored point cloud + plane grid overlay."""
    frame0_pts, frame0_colors = fuse_multiview_depth(
        frame_indices=[0],
        depth_dirs=depth_dirs,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        cam_resolutions=cam_resolutions,
        rgb_dirs=rgb_dirs,
        max_depth=max_depth,
        image_scale=image_scale,
        voxel_size=0.01,
        statistical_outlier_neighbors=20,
        statistical_outlier_std=2.0,
    )

    normal = plane[:3] / np.linalg.norm(plane[:3])
    d = plane[3] / np.linalg.norm(plane[:3])
    center = -d * normal

    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    grid_size = 4.0
    grid_res = 200
    half = grid_size / 2.0
    lin = np.linspace(-half, half, grid_res)
    uu, vv = np.meshgrid(lin, lin)
    plane_pts = (
        center[None, None, :]
        + uu[:, :, None] * u[None, None, :]
        + vv[:, :, None] * v[None, None, :]
    ).reshape(-1, 3)
    plane_colors = np.full((len(plane_pts), 3), 255, dtype=np.uint8)
    plane_colors[:, 1] = 0
    plane_colors[:, 2] = 0

    all_pts = np.concatenate([frame0_pts, plane_pts], axis=0)
    if frame0_colors is not None:
        all_colors = np.concatenate([frame0_colors, plane_colors], axis=0)
    else:
        scene_colors = np.full((len(frame0_pts), 3), 180, dtype=np.uint8)
        all_colors = np.concatenate([scene_colors, plane_colors], axis=0)

    save_colored_ply(ply_path, all_pts, all_colors)
    logger.info("Debug PLY saved to %s (%d scene + %d plane points)",
                ply_path, len(frame0_pts), len(plane_pts))


def _estimate_up_from_camera(ref_cam_T: np.ndarray) -> np.ndarray:
    """Derive "up" from the reference camera's orientation.

    In OpenCV convention the camera Y-axis points down, so "up" in world
    frame is the negated second column of the camera-to-world rotation.
    """
    up = -ref_cam_T[:3, 1]
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return up / norm


def estimate_ground_plane_from_config(cfg) -> dict:
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)

    cam_intrinsics: list[np.ndarray] = []
    cam_extrinsics: list[np.ndarray] = []
    cam_resolutions: list[np.ndarray] = []
    depth_dirs: list[Path] = []
    rgb_dirs: list[Path] | None = None
    mask_dirs: list[Path] | None = None
    if cfg.get("rgb_dir"):
        rgb_dirs = []
    if cfg.get("mask_dir"):
        mask_dirs = []

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        cam_intrinsics.append(cam.param.K)
        cam_extrinsics.append(cam.param.T)
        cam_resolutions.append(cam.param.resolution)
        depth_dirs.append(Path(cfg.depth_path_template.format(cam_name=cam.name)))
        if rgb_dirs is not None:
            rgb_dirs.append(Path(cfg.rgb_path_template.format(cam_name=cam.name)))
        if mask_dirs is not None:
            mask_dirs.append(
                Path(cfg.mask_path_template.format(
                    cam_name=cam.name, mask_obj_id=cfg.mask_obj_id
                ))
            )

    mhr_params_path = Path(cfg.mhr_params_mv_path) if cfg.get("mhr_params_mv_path") else None

    return estimate_ground_plane(
        depth_dirs=depth_dirs,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        cam_resolutions=cam_resolutions,
        rig=rig,
        output_path=Path(cfg.output_path),
        mhr_params_path=mhr_params_path,
        mask_dirs=mask_dirs,
        rgb_dirs=rgb_dirs,
        n_sample_frames=cfg.n_sample_frames,
        height_band=cfg.height_band,
        ransac_threshold=cfg.ransac_threshold,
        ransac_iterations=cfg.ransac_iterations,
        max_normal_angle=cfg.max_normal_angle,
        max_depth=cfg.max_depth,
        image_scale=cfg.image_scale,
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Estimate ground plane from multiview depth")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--human_pose_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rgb_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_estimate_ground_plane.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides = {
        "camera_params_path": args.camera_params_path,
        "depth_dir": args.depth_dir,
        "human_pose_dir": args.human_pose_dir,
        "output_dir": args.output_dir,
    }
    if args.rgb_dir is not None:
        overrides["rgb_dir"] = args.rgb_dir
    if args.mask_dir is not None:
        overrides["mask_dir"] = args.mask_dir
    cfg = OmegaConf.merge(cfg, overrides)
    estimate_ground_plane_from_config(cfg)
