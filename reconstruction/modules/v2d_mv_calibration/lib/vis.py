# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Calibration visualization utilities using Rerun."""

import logging
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from v2d.mv.rig import CameraParam
from v2d.mv.math.numpy_fn import distort_polynomial, reproject, se3_inv


logger = logging.getLogger(__name__)


def _frustum_lines(
    img_size: tuple[int, int],
    frustum_scale: float = 0.1,
    frustum_depth: float = 0.2,
    rot: np.ndarray = np.eye(3),
    trans: np.ndarray = np.zeros(3),
) -> list[list[np.ndarray]]:
    """Generate camera frustum line segments for visualization."""
    w, h = img_size[0] / img_size[0], img_size[1] / img_size[0]
    points = np.array([
        [0, 0, 0],
        [-w * frustum_scale, -h * frustum_scale, frustum_depth],
        [w * frustum_scale, -h * frustum_scale, frustum_depth],
        [w * frustum_scale, h * frustum_scale, frustum_depth],
        [-w * frustum_scale, h * frustum_scale, frustum_depth],
    ])
    points = points @ rot.T + trans
    return [
        [points[0], points[1]], [points[0], points[2]],
        [points[0], points[3]], [points[0], points[4]],
        [points[1], points[2]], [points[2], points[3]],
        [points[3], points[4]], [points[4], points[1]],
    ]


def _axes_lines(axes_scale: float = 0.1):
    """Generate RGB coordinate axes line segments."""
    pts = [[0, 0, 0], [axes_scale, 0, 0], [0, axes_scale, 0], [0, 0, axes_scale]]
    lines = [[pts[0], pts[1]], [pts[0], pts[2]], [pts[0], pts[3]]]
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    return lines, colors


def visualize_camera_and_target_poses(
    output_file: Path,
    camera_params_seq: list[list[CameraParam]],
    target_poses_seq: list[np.ndarray],
    frustum_colors: np.ndarray,
):
    """Visualize camera frustums and target poses over optimization iterations.

    Args:
        output_file: Path to write .rrd file.
        camera_params_seq: Per-iteration list of camera params.
        target_poses_seq: Per-iteration array of target poses.
        frustum_colors: (T, N, 3) RGB colors for each camera at each iteration.
    """
    import rerun as rr

    rr.init("camera_and_target_poses")
    rr.save(str(output_file))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    for t, (camera_params, target_poses) in tqdm(
        enumerate(zip(camera_params_seq, target_poses_seq)),
        total=len(camera_params_seq),
        desc="Visualizing",
    ):
        rr.set_time("frame_id", sequence=t)
        for j, param in enumerate(camera_params):
            T = param.T
            rot, trans = T[:3, :3], T[:3, 3]
            rr.log(f"world/cam_{j}", rr.Transform3D(translation=trans, mat3x3=rot))
            color = frustum_colors[t, j].tolist()
            fl = _frustum_lines(tuple(param.resolution))
            rr.log(f"world/cam_{j}/frustum", rr.LineStrips3D(fl, colors=color))

        for j, target_pose in enumerate(target_poses):
            al, ac = _axes_lines()
            rot, trans = target_pose[:3, :3], target_pose[:3, 3]
            rr.log(f"world/target_{j}", rr.Transform3D(translation=trans, mat3x3=rot))
            rr.log(f"world/target_{j}/axes", rr.LineStrips3D(al, colors=ac))

    logger.info(f"Visualization saved to {output_file}")


def visualize_reprojected_points(
    output_dir: Path,
    frame_source,
    target_xyz: np.ndarray,
    per_cam_features: list[np.ndarray | None],
    est_camera_param: CameraParam,
    opt_camera_param: CameraParam,
    est_target_poses: np.ndarray,
    opt_target_poses: np.ndarray,
    frame_indices: list[int] | None = None,
    radius: int = 1,
    shift: int = 4,
):
    """Draw reprojected points on images for visual inspection.

    Args:
        output_dir: Directory to write annotated images.
        frame_source: FrameSource for reading images.
        target_xyz: (P, 3) target 3D points.
        per_cam_features: Per-correspondence detected features (or None).
        est_camera_param: Estimated (pre-BA) camera params.
        opt_camera_param: Optimized (post-BA) camera params.
        est_target_poses: (N, 4, 4) estimated target poses.
        opt_target_poses: (N, 4, 4) optimized target poses.
        frame_indices: Original frame indices for each correspondence
            entry. When provided, only those frames are visualized and
            per_cam_features/poses are indexed by correspondence order.
            When None, falls back to sequential indexing (legacy behavior).
    """
    K, D = est_camera_param.K, est_camera_param.D
    distort_fn = partial(distort_polynomial, coeffs=D) if len(D) > 0 else None
    est_T_cam_world = se3_inv(est_camera_param.T)
    opt_T_cam_world = se3_inv(opt_camera_param.T)
    shift_factor = 1 << shift
    r = radius * shift_factor

    n_corr = len(est_target_poses)
    if frame_indices is None:
        frame_indices = list(range(n_corr))

    output_dir.mkdir(parents=True, exist_ok=True)

    for corr_idx, img_idx in tqdm(
        enumerate(frame_indices), total=len(frame_indices),
        desc="Drawing reprojections",
    ):
        if img_idx >= frame_source.n_frames:
            logger.warning("Frame index %d out of range (%d frames), skipping", img_idx, frame_source.n_frames)
            continue

        img = frame_source[img_idx]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        uv_est = reproject(
            target_xyz, K, est_T_cam_world @ est_target_poses[corr_idx], distort_fn,
        )
        uv_opt = reproject(
            target_xyz, K, opt_T_cam_world @ opt_target_poses[corr_idx], distort_fn,
        )

        cv2.putText(img, "Features", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, "Estimated", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(img, "Optimized", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        feat = per_cam_features[corr_idx] if corr_idx < len(per_cam_features) else None
        if feat is not None:
            for pt in feat:
                p = (pt * shift_factor + 0.5).astype(int)
                cv2.circle(img, p.tolist(), r, (0, 255, 255), -1, shift=shift)
        for pt in uv_est:
            p = (pt * shift_factor + 0.5).astype(int)
            cv2.circle(img, p.tolist(), r, (255, 0, 255), -1, shift=shift)
        for pt in uv_opt:
            p = (pt * shift_factor + 0.5).astype(int)
            cv2.circle(img, p.tolist(), r, (0, 255, 0), -1, shift=shift)

        cv2.imwrite(str(output_dir / f"reproj_{img_idx:06d}.png"), img)

    logger.info(f"Reprojection visualizations saved to {output_dir}")
