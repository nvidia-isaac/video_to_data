# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np


def compute_scale_robust_log(pred, target, mask, max_iterations=30, huber_delta=1.345, tolerance=1e-12):
    """Fit a positive scale from valid pixelwise depth ratios using a Huber location in log space."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError(f"Depth and mask shape mismatch: pred={pred.shape}, target={target.shape}, mask={mask.shape}")
    valid = mask & np.isfinite(pred) & np.isfinite(target) & (pred > 0.0) & (target > 0.0)
    if not np.any(valid):
        raise ValueError("Huber log-ratio scale requires at least one finite positive depth pair")
    log_ratio = np.log(target[valid]) - np.log(pred[valid])
    center = float(np.median(log_ratio))
    for _ in range(int(max_iterations)):
        sigma = max(1.4826 * float(np.median(np.abs(log_ratio - center))), 1e-8)
        cutoff = float(huber_delta) * sigma
        residual = np.abs(log_ratio - center)
        weights = np.minimum(1.0, cutoff / np.maximum(residual, 1e-12))
        updated = float(np.sum(weights * log_ratio) / np.sum(weights))
        if abs(updated - center) < float(tolerance):
            center = updated
            break
        center = updated
    scale = float(np.exp(center))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Huber log-ratio scale is invalid: {scale}")
    return scale
