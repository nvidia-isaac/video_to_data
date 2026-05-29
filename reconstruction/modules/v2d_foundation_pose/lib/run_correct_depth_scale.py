# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
FP-guided per-frame depth scale correction.

After a first FoundationPose tracking pass, the mesh can be rendered at each
estimated pose to obtain a reliable depth map for the object region. Comparing
this rendered depth against the MoGe depth yields a per-frame scalar correction:

    scale_k = median( rendered_depth[valid] / moge_depth[valid] )

where valid = object mask & rendered_depth > 0 & moge_depth > 0.

This corrects the global scale error that monocular metric depth models exhibit
at close range (where training-distribution coverage is sparse).

The per-frame scales are smoothed with a temporal median filter before being
applied, avoiding the introduction of frame-to-frame jitter. Frames with
insufficient overlap (< min_valid_pixels) are interpolated from neighbours.
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
from scipy.ndimage import median_filter

_FP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FoundationPose")
if _FP_DIR not in sys.path:
    sys.path.insert(0, _FP_DIR)

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
from v2d.foundation_pose.lib.fp_utils import nvdiffrast_render, make_mesh_tensors
from v2d.mesh.lib.mesh import Mesh

import nvdiffrast.torch as dr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_correct_depth_scale(
    poses_dir: str,
    mesh_path: str,
    depth_folder: str,
    intrinsics_path: str,
    output_folder: str,
    masks_folder: str = None,
    smoothing_window: int = 11,
    min_valid_pixels: int = 50,
    batch_size: int = 32,
) -> None:
    """
    Correct per-frame MoGe depth scale using FoundationPose rendered depth.

    For each frame, renders the mesh at the FP-estimated pose and computes:
        scale_k = median( rendered_depth / moge_depth )  over valid object pixels

    Scales are temporally smoothed then applied to the full depth map.
    Frames with too few valid pixels fall back to their neighbours' scale.

    Args:
        poses_dir:          Directory of per-frame Transform3d JSON files (FP output).
        mesh_path:          Mesh used during tracking.
        depth_folder:       Directory of MoGe depth PNGs to correct.
        intrinsics_path:    Camera intrinsics JSON (stable recommended).
        output_folder:      Destination for corrected depth PNGs.
        masks_folder:       Optional SAM2 mask folder. When provided, scale is
                            computed only at object pixels. When None, all pixels
                            where both depths are positive are used.
        smoothing_window:   Temporal median filter window (frames). Default 11.
        min_valid_pixels:   Minimum valid pixels to compute a reliable scale.
                            Frames below this threshold are interpolated. Default 50.
        batch_size:         Poses to render per GPU call. Default 32.
    """
    intrinsics = CameraIntrinsics.load(intrinsics_path)
    K  = intrinsics.to_matrix()
    H, W = intrinsics.height, intrinsics.width

    trimesh_mesh  = Mesh.load(mesh_path).to_trimesh()
    mesh_tensors  = make_mesh_tensors(trimesh_mesh)

    pose_files    = sorted(f for f in os.listdir(poses_dir) if f.endswith('.json'))
    frame_indices = [int(os.path.splitext(f)[0]) for f in pose_files]
    N = len(pose_files)
    logger.info(f"Computing depth scale corrections for {N} frames")

    glctx = dr.RasterizeCudaContext()

    # ---- Render all poses in batches, compute per-frame raw scales ----
    raw_scales = np.full(N, np.nan)

    for chunk_start in range(0, N, batch_size):
        chunk_files   = pose_files[chunk_start : chunk_start + batch_size]
        chunk_indices = frame_indices[chunk_start : chunk_start + batch_size]

        ob_in_cams = torch.stack([
            torch.as_tensor(
                Transform3d.load(os.path.join(poses_dir, f)).to_matrix(),
                device='cuda', dtype=torch.float,
            )
            for f in chunk_files
        ])

        with torch.no_grad():
            _, rendered_depths, _ = nvdiffrast_render(
                K=K, H=H, W=W,
                ob_in_cams=ob_in_cams,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                get_normal=False,
            )

        rendered_np = rendered_depths.cpu().numpy()  # (B, H, W)

        for i, (frame_idx, rendered) in enumerate(zip(chunk_indices, rendered_np)):
            depth_path = os.path.join(depth_folder, f"{frame_idx:06d}.png")
            if not os.path.exists(depth_path):
                continue
            moge = DepthImage.load(depth_path).depth  # (H, W) float32 in metres

            rendered_valid = rendered > 0.001
            moge_valid     = moge > 0.001

            if masks_folder is not None:
                mask_path = os.path.join(masks_folder, f"{frame_idx:06d}.png")
                if os.path.exists(mask_path):
                    obj_mask = Mask.load(mask_path).mask.astype(bool)
                else:
                    obj_mask = np.ones((H, W), dtype=bool)
            else:
                obj_mask = np.ones((H, W), dtype=bool)

            valid = obj_mask & rendered_valid & moge_valid
            if valid.sum() < min_valid_pixels:
                logger.debug(f"Frame {frame_idx}: only {valid.sum()} valid pixels — will interpolate")
                continue

            scale = float(np.median(rendered[valid] / moge[valid]))
            raw_scales[chunk_start + i] = scale
            logger.debug(f"Frame {frame_idx}: raw scale={scale:.4f}")

    valid_mask = ~np.isnan(raw_scales)
    logger.info(
        f"Valid scale estimates: {valid_mask.sum()}/{N}  "
        f"mean={np.nanmean(raw_scales):.4f}  "
        f"std={np.nanstd(raw_scales):.4f}"
    )

    # ---- Interpolate missing frames ----
    if not valid_mask.all():
        indices = np.arange(N)
        raw_scales = np.interp(indices, indices[valid_mask], raw_scales[valid_mask])

    # ---- Temporal smoothing in log-space ----
    log_scales    = np.log(np.clip(raw_scales, 1e-6, None))
    log_smoothed  = median_filter(log_scales, size=smoothing_window, mode='reflect')
    smooth_scales = np.exp(log_smoothed)

    logger.info(
        f"Smoothed scales — mean={smooth_scales.mean():.4f}  "
        f"min={smooth_scales.min():.4f}  max={smooth_scales.max():.4f}"
    )

    # ---- Apply corrections ----
    os.makedirs(output_folder, exist_ok=True)
    for i, frame_idx in enumerate(frame_indices):
        depth_path = os.path.join(depth_folder, f"{frame_idx:06d}.png")
        if not os.path.exists(depth_path):
            continue
        d = DepthImage.load(depth_path)
        corrected = DepthImage(depth=(d.depth * smooth_scales[i]).astype(np.float32))
        corrected.save(os.path.join(output_folder, f"{frame_idx:06d}.png"))

    logger.info(f"Corrected depth written to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP-guided per-frame depth scale correction")
    parser.add_argument("--poses_dir",         required=True)
    parser.add_argument("--mesh_path",         required=True)
    parser.add_argument("--depth_folder",      required=True)
    parser.add_argument("--intrinsics_path",   required=True)
    parser.add_argument("--output_folder",     required=True)
    parser.add_argument("--masks_folder",      default=None)
    parser.add_argument("--smoothing_window",  type=int,   default=11)
    parser.add_argument("--min_valid_pixels",  type=int,   default=50)
    parser.add_argument("--batch_size",        type=int,   default=32)
    args = parser.parse_args()
    run_correct_depth_scale(
        args.poses_dir,
        args.mesh_path,
        args.depth_folder,
        args.intrinsics_path,
        args.output_folder,
        masks_folder=args.masks_folder,
        smoothing_window=args.smoothing_window,
        min_valid_pixels=args.min_valid_pixels,
        batch_size=args.batch_size,
    )
