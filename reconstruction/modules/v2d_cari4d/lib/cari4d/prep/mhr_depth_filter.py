from __future__ import annotations

import numpy as np


def filter_depth_like_smpl(depth_m: np.ndarray, *, device: str = "cuda") -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must have shape (H, W), got {depth_m.shape}")
    import Utils

    filtered = Utils.erode_depth(depth_m, radius=2, device=device)
    filtered = Utils.bilateral_filter_depth(filtered, radius=2, device=device)
    return np.asarray(filtered, dtype=np.float32)


def filtered_depth_points_like_smpl(depth_m: np.ndarray, K: np.ndarray, human_mask: np.ndarray, *, max_points: int = 12000) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    human_mask = np.asarray(human_mask, dtype=bool)
    if depth_m.shape != human_mask.shape:
        raise ValueError(f"depth and mask shapes must match, got {depth_m.shape} and {human_mask.shape}")
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    valid = human_mask & np.isfinite(depth_m) & (depth_m >= 0.001) & (depth_m < 100.0)
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3) or K[0, 0] == 0 or K[1, 1] == 0:
        raise ValueError(f"K must be a valid 3x3 intrinsic matrix, got {K}")
    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    x = (u.astype(np.float32) - K[0, 2]) * z / K[0, 0]
    y = (v.astype(np.float32) - K[1, 2]) * z / K[1, 1]
    points = np.stack((x, y, z), axis=-1).astype(np.float32)
    if len(points) <= max_points:
        return np.asarray(points, dtype=np.float32)
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return np.asarray(points[indices], dtype=np.float32)
