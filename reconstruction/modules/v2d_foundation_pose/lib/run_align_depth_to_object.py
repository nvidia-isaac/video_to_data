# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Align raw monocular depth to an object mesh via FP-guided affine grid search.

Searches over (scale, shift) in D_aligned = scale * D_raw + shift and returns
the corrected depth image that best fits the mesh at the reference frame.
"""
import argparse
import logging
import os

import cv2

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask
from v2d.common.datatypes import Image as V2dImage
from v2d.mesh.lib.mesh import Mesh
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_align_depth_to_object(
    mesh_path: str,
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    weights_dir: str,
    output_depth_path: str,
    scale_lo: float = 0.5,
    scale_hi: float = 2.0,
    shift_lo: float = -0.5,
    shift_hi: float = 0.5,
    n_scale_samples: int = 7,
    n_shift_samples: int = 5,
    n_levels: int = 3,
    iou_weight: float = 1.0,
    depth_weight: float = 1.0,
    registration_iterations: int = 5,
) -> DepthImage:
    """Align raw monocular depth to object mesh, saving the corrected depth PNG.

    Args:
        mesh_path:               Object mesh file.
        rgb_path:                Reference frame RGB image (PNG).
        depth_path:              Raw monocular depth PNG (uint16 inverse-depth encoding).
        mask_path:               Object segmentation mask PNG.
        intrinsics_path:         Camera intrinsics JSON.
        weights_dir:             FoundationPose weights directory.
        output_depth_path:       Output path for the corrected depth PNG.
        scale_lo:                Lower bound of scale search range. Default 0.5.
        scale_hi:                Upper bound of scale search range. Default 2.0.
        shift_lo:                Lower bound of shift search range (metres). Default -0.5.
        shift_hi:                Upper bound of shift search range (metres). Default 0.5.
        n_scale_samples:         Scale candidates per level. Default 7.
        n_shift_samples:         Shift candidates per level. Default 5.
        n_levels:                Refinement levels. Default 3.
        iou_weight:              Weight for mask IoU score. Default 1.0.
        depth_weight:            Weight for depth MARE score. Default 1.0.
        registration_iterations: FP register() iterations per candidate. Default 5.

    Returns:
        Corrected DepthImage (also saved to output_depth_path).
    """
    mesh = Mesh.load(mesh_path)
    tracker = FoundationPoseTracker(mesh, weights_dir)

    intrinsics = CameraIntrinsics.load(intrinsics_path)
    depth_raw = DepthImage.load(depth_path)
    mask = Mask.load(mask_path)

    frame = cv2.imread(rgb_path)
    if frame is None:
        raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
    rgb = V2dImage(data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    corrected = tracker.align_depth_to_object(
        rgb, depth_raw, mask, intrinsics,
        scale_lo=scale_lo, scale_hi=scale_hi,
        shift_lo=shift_lo, shift_hi=shift_hi,
        n_scale_samples=n_scale_samples, n_shift_samples=n_shift_samples,
        n_levels=n_levels, iou_weight=iou_weight, depth_weight=depth_weight,
        registration_iterations=registration_iterations,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_depth_path)), exist_ok=True)
    corrected.save(output_depth_path)
    logger.info(f"Saved aligned depth to {output_depth_path}")
    return corrected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align depth to object mesh via FP affine grid search")
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--rgb_path", required=True)
    parser.add_argument("--depth_path", required=True)
    parser.add_argument("--mask_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--output_depth_path", required=True)
    parser.add_argument("--scale_lo", type=float, default=0.5)
    parser.add_argument("--scale_hi", type=float, default=2.0)
    parser.add_argument("--shift_lo", type=float, default=-0.5)
    parser.add_argument("--shift_hi", type=float, default=0.5)
    parser.add_argument("--n_scale_samples", type=int, default=7)
    parser.add_argument("--n_shift_samples", type=int, default=5)
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--iou_weight", type=float, default=1.0)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--registration_iterations", type=int, default=5)
    args = parser.parse_args()
    run_align_depth_to_object(
        args.mesh_path, args.rgb_path, args.depth_path, args.mask_path,
        args.intrinsics_path, args.weights_dir, args.output_depth_path,
        scale_lo=args.scale_lo, scale_hi=args.scale_hi,
        shift_lo=args.shift_lo, shift_hi=args.shift_hi,
        n_scale_samples=args.n_scale_samples, n_shift_samples=args.n_shift_samples,
        n_levels=args.n_levels, iou_weight=args.iou_weight, depth_weight=args.depth_weight,
        registration_iterations=args.registration_iterations,
    )
