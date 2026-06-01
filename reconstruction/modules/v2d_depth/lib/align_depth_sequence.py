# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Temporal scale alignment for monocular depth sequences.

Monocular depth models (MoGe, UniDepth, etc.) estimate each frame independently,
causing per-frame scale drift that makes depth estimates temporally inconsistent.

This module corrects that by:
  1. Extracting sparse feature correspondences between each frame and a reference
     frame, restricted to background pixels (outside the object mask).
  2. Computing the per-frame depth scale factor as the median ratio of reference
     depth to current depth at matched pixel locations.
  3. Smoothing the raw per-frame scale signal in log-space with a median filter
     to remove outlier frames without blurring spatial depth structure.
  4. Applying the smoothed per-frame scale to each depth map.

Only the scalar scale is corrected (not shift), which is appropriate for metric
depth models. For affine-ambiguous models, extend to fit scale + shift.
"""
import logging
import os

import cv2
import numpy as np

from v2d.common.datatypes import DepthImage, Mask

logger = logging.getLogger(__name__)

# Minimum number of inlier matches required to trust a frame's scale estimate.
_MIN_MATCHES = 10


def _load_frame_gray(frames_folder: str, idx: int) -> np.ndarray | None:
    path = os.path.join(frames_folder, f"{idx:06d}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _estimate_frame_scale(
    ref_gray: np.ndarray,
    ref_depth: np.ndarray,
    ref_bg_mask: np.ndarray,
    frame_gray: np.ndarray,
    frame_depth: np.ndarray,
    frame_bg_mask: np.ndarray,
) -> float | None:
    """Estimate the scale factor for one frame relative to the reference.

    Matches SIFT features in background regions, then computes the median ratio
    ref_depth[pt] / frame_depth[pt] over valid matched points. Returns None if
    there are insufficient matches.
    """
    sift = cv2.SIFT_create()

    # Mask keypoint detection to background regions only
    ref_mask_u8 = (ref_bg_mask > 0).astype(np.uint8) * 255
    frame_mask_u8 = (frame_bg_mask > 0).astype(np.uint8) * 255

    kp_ref, desc_ref = sift.detectAndCompute(ref_gray, ref_mask_u8)
    kp_frame, desc_frame = sift.detectAndCompute(frame_gray, frame_mask_u8)

    if desc_ref is None or desc_frame is None or len(kp_ref) < 4 or len(kp_frame) < 4:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(desc_ref, desc_frame, k=2)

    # Lowe's ratio test
    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    if len(good) < _MIN_MATCHES:
        return None

    H, W = ref_depth.shape
    ratios = []
    for m in good:
        x_ref, y_ref = kp_ref[m.queryIdx].pt
        x_fr, y_fr = kp_frame[m.trainIdx].pt
        xi_ref, yi_ref = int(round(x_ref)), int(round(y_ref))
        xi_fr, yi_fr = int(round(x_fr)), int(round(y_fr))

        if not (0 <= xi_ref < W and 0 <= yi_ref < H):
            continue
        if not (0 <= xi_fr < W and 0 <= yi_fr < H):
            continue

        d_ref = ref_depth[yi_ref, xi_ref]
        d_fr = frame_depth[yi_fr, xi_fr]

        if d_ref > 0.001 and d_fr > 0.001:
            ratios.append(d_ref / d_fr)

    if len(ratios) < _MIN_MATCHES:
        return None

    return float(np.median(ratios))


def _smooth_scales_log(
    raw_scales: dict[int, float],
    all_indices: list[int],
    window: int,
) -> dict[int, float]:
    """Smooth a sparse dict of scale estimates in log-space using a median filter.

    Frames with no estimate are filled by interpolation from neighbours before
    smoothing. The smoothed result is returned for all indices.
    """
    indices = sorted(all_indices)
    n = len(indices)
    idx_to_pos = {idx: i for i, idx in enumerate(indices)}

    # Build raw log-scale array; NaN for frames without an estimate
    log_raw = np.full(n, np.nan)
    for idx, s in raw_scales.items():
        if idx in idx_to_pos and s is not None and s > 0:
            log_raw[idx_to_pos[idx]] = np.log(s)

    # Linear interpolation to fill NaNs
    valid = np.where(~np.isnan(log_raw))[0]
    if len(valid) == 0:
        return {idx: 1.0 for idx in indices}
    log_filled = np.interp(np.arange(n), valid, log_raw[valid])

    # Median filter in log-space
    half = window // 2
    log_smoothed = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        log_smoothed[i] = np.median(log_filled[lo:hi])

    return {indices[i]: float(np.exp(log_smoothed[i])) for i in range(n)}


def align_depth_sequence(
    depth_folder: str,
    frames_folder: str,
    masks_folder: str,
    output_folder: str,
    reference_frame: int = 0,
    smoothing_window: int = 11,
) -> None:
    """Align a monocular depth sequence to a reference frame via per-frame scale correction.

    For each frame:
      - Matches SIFT features in background regions (outside the object mask)
        between that frame and the reference frame.
      - Estimates the depth scale factor as median(d_ref / d_frame) at matches.
      - Smooths all per-frame scales in log-space with a median filter.
      - Writes scale-corrected depth PNGs to output_folder.

    Args:
        depth_folder:     Folder of uint16 depth PNGs ({000000,...}.png).
        frames_folder:    Folder of RGB PNGs at the same frame indices.
        masks_folder:     Folder of object mask PNGs (non-zero = object, background = 0).
        output_folder:    Destination for aligned depth PNGs.
        reference_frame:  Frame index whose scale is treated as ground truth.
        smoothing_window: Median filter window size for log-scale smoothing. Must be odd.
    """
    # Discover available frames from depth folder
    depth_files = sorted(f for f in os.listdir(depth_folder) if f.endswith(".png"))
    if not depth_files:
        raise RuntimeError(f"No depth PNGs found in {depth_folder}")

    all_indices = [int(os.path.splitext(f)[0]) for f in depth_files]
    os.makedirs(output_folder, exist_ok=True)

    # Load reference data
    ref_depth_img = DepthImage.load(os.path.join(depth_folder, f"{reference_frame:06d}.png"))
    ref_depth = ref_depth_img.depth
    ref_gray = _load_frame_gray(frames_folder, reference_frame)

    ref_mask_path = os.path.join(masks_folder, f"{reference_frame:06d}.png")
    if os.path.exists(ref_mask_path):
        ref_bg_mask = ~Mask.load(ref_mask_path).mask.astype(bool)
    else:
        ref_bg_mask = np.ones(ref_depth.shape, dtype=bool)

    if ref_gray is None:
        raise RuntimeError(f"Failed to load reference frame image: {reference_frame:06d}.png")

    # Estimate per-frame raw scale factors
    raw_scales: dict[int, float] = {reference_frame: 1.0}

    for idx in all_indices:
        if idx == reference_frame:
            continue

        frame_gray = _load_frame_gray(frames_folder, idx)
        frame_depth_path = os.path.join(depth_folder, f"{idx:06d}.png")
        frame_depth = DepthImage.load(frame_depth_path).depth

        mask_path = os.path.join(masks_folder, f"{idx:06d}.png")
        if os.path.exists(mask_path):
            frame_bg_mask = ~Mask.load(mask_path).mask.astype(bool)
        else:
            frame_bg_mask = np.ones(frame_depth.shape, dtype=bool)

        if frame_gray is None:
            logger.warning(f"Frame {idx}: missing RGB, skipping scale estimation")
            continue

        scale = _estimate_frame_scale(
            ref_gray, ref_depth, ref_bg_mask,
            frame_gray, frame_depth, frame_bg_mask,
        )
        if scale is None:
            logger.warning(f"Frame {idx}: insufficient matches for scale estimation")
        else:
            logger.debug(f"Frame {idx}: raw scale={scale:.4f}")
            raw_scales[idx] = scale

    logger.info(f"Scale estimated for {len(raw_scales)}/{len(all_indices)} frames")

    # Smooth in log-space
    smoothed_scales = _smooth_scales_log(raw_scales, all_indices, smoothing_window)

    # Apply corrected scales and write output depth
    for idx in all_indices:
        src_path = os.path.join(depth_folder, f"{idx:06d}.png")
        dst_path = os.path.join(output_folder, f"{idx:06d}.png")

        scale = smoothed_scales.get(idx, 1.0)
        if abs(scale - 1.0) < 1e-6:
            # No correction needed — copy as-is
            import shutil
            shutil.copy2(src_path, dst_path)
        else:
            depth_img = DepthImage.load(src_path)
            corrected = DepthImage(depth=depth_img.depth * scale)
            corrected.save(dst_path)

        logger.debug(f"Frame {idx}: smoothed scale={scale:.4f}")

    logger.info(f"Aligned depth written to {output_folder}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Align monocular depth sequence to reference frame")
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--frames_folder", required=True)
    parser.add_argument("--masks_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--reference_frame", type=int, default=0)
    parser.add_argument("--smoothing_window", type=int, default=11)

    args = parser.parse_args()
    align_depth_sequence(
        args.depth_folder,
        args.frames_folder,
        args.masks_folder,
        args.output_folder,
        reference_frame=args.reference_frame,
        smoothing_window=args.smoothing_window,
    )
