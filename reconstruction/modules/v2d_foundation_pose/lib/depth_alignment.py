"""Temporal depth alignment via ICP on scene point clouds.

Aligns raw monocular depth for each frame to a metric reference depth by
unprojecting both into point clouds, then alternating between:
  1. ICP step  — estimates the rigid camera-motion transform.
  2. Affine step — solves for (scale, shift) linearly given the rigid T.

The foreground mask (object + hands) is excluded from fitting so that
moving objects don't pollute the background-based alignment.
"""
import logging

import numpy as np
from scipy.spatial import KDTree

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask

logger = logging.getLogger(__name__)


def align_depth_to_reference_depth(
    depth_raw: DepthImage,
    depth_ref: DepthImage,
    intrinsics: CameraIntrinsics,
    fg_mask: Mask = None,
    fg_mask_ref: Mask = None,
    n_iterations: int = 3,
    outlier_trim_ratio: float = 0.2,
    max_points: int = 20000,
) -> DepthImage:
    """Align raw monocular depth to a metric reference depth via ICP + affine solve.

    Finds (scale, shift) such that scale * depth_raw + shift best matches depth_ref
    on the static background, accounting for small camera motion via ICP.

    Each iteration:
      1. ICP: find rigid T that aligns the current affine-corrected source cloud
         to the reference cloud (handles camera motion).
      2. Affine solve: with T fixed, solve for (scale, shift) via linear least
         squares using the ICP inlier correspondences.

    Args:
        depth_raw:          Raw monocular depth for the current frame.
        depth_ref:          Metric reference depth (output of align_depth_to_object).
        intrinsics:         Camera intrinsics (assumed same for both frames).
        fg_mask:            Foreground mask for current frame (object + hands).
                            Excluded from fitting. None = use all valid pixels.
        fg_mask_ref:        Foreground mask for reference frame. If None, fg_mask
                            is reused as a proxy.
        n_iterations:       Alternating ICP/affine iterations. Default 3.
        outlier_trim_ratio: Fraction of worst-fitting correspondences to discard
                            each iteration. Default 0.2.
        max_points:         Max background points to use (random subsample).
                            Default 20000.

    Returns:
        Corrected DepthImage: clip(scale * depth_raw + shift, 0).
    """
    K = intrinsics.to_matrix()
    H, W = intrinsics.height, intrinsics.width

    bg_mask = _background_mask(depth_raw.depth, fg_mask, H, W)
    bg_mask_ref = _background_mask(
        depth_ref.depth,
        fg_mask_ref if fg_mask_ref is not None else fg_mask,
        H, W,
    )

    PC_ref, _ = _unproject(depth_ref.depth, K, bg_mask_ref, max_points)
    PC_raw, (us, vs) = _unproject(depth_raw.depth, K, bg_mask, max_points)
    dirs = _ray_directions(us, vs, K)  # (N, 3) — unit ray directions per pixel

    if len(PC_ref) < 10 or len(PC_raw) < 10:
        logger.warning("Insufficient background points for depth alignment — returning raw depth")
        return depth_raw

    T = np.eye(4, dtype=np.float64)
    s, b = 1.0, 0.0

    for it in range(n_iterations):
        # Apply current affine to raw points
        PC_t = s * PC_raw + b * dirs  # (N, 3)

        # Apply current rigid transform
        R, t_vec = T[:3, :3], T[:3, 3]
        PC_t_aligned = (R @ PC_t.T).T + t_vec  # (N, 3)

        # Nearest-neighbour matching to reference cloud
        tree = KDTree(PC_ref)
        dists, nn_idx = tree.query(PC_t_aligned, k=1)

        # Trim outliers (hands, object, noise)
        thresh = np.quantile(dists, 1.0 - outlier_trim_ratio)
        inliers = dists <= thresh
        if inliers.sum() < 10:
            logger.warning(f"Iter {it}: too few inliers ({inliers.sum()}) — stopping early")
            break

        src_in = PC_t[inliers]            # affine-corrected source (pre-T), inliers only
        tgt_in = PC_ref[nn_idx[inliers]]  # matched reference points

        # ICP step: solve rigid T from affine-corrected source to matched target
        T = _solve_rigid(src_in, tgt_in)
        R, t_vec = T[:3, :3], T[:3, 3]

        # Affine step: solve (s, b) with T fixed.
        # After applying T: R @ (s * P_raw_i + b * dir_i) + t = P_ref_i
        # => s * (R @ P_raw_i) + b * (R @ dir_i) = P_ref_i - t  (linear in s, b)
        rp = (R @ PC_raw[inliers].T).T   # (M, 3)
        rd = (R @ dirs[inliers].T).T     # (M, 3)
        rhs = tgt_in - t_vec             # (M, 3)

        A = np.stack([rp.ravel(), rd.ravel()], axis=1)  # (3M, 2)
        y = rhs.ravel()                                  # (3M,)
        result, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        s, b = float(result[0]), float(result[1])

        logger.debug(f"Iter {it}: scale={s:.4f}  shift={b:.4f}  inliers={inliers.sum()}")

    logger.info(f"Depth alignment: scale={s:.4f}  shift={b:.4f}")
    corrected = np.clip(s * depth_raw.depth + b, 0.0, None).astype(np.float32)
    return DepthImage(depth=corrected)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _background_mask(depth: np.ndarray, fg_mask: Mask, H: int, W: int) -> np.ndarray:
    valid = depth > 0.001
    if fg_mask is not None:
        valid = valid & ~fg_mask.mask.astype(bool)
    return valid


def _unproject(
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Unproject masked depth pixels to 3D points.

    Returns:
        pts:      (N, 3) float64 point cloud.
        (us, vs): pixel coordinates of the returned points (after subsampling).
    """
    H, W = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    us = u_grid[mask].astype(np.float64)
    vs = v_grid[mask].astype(np.float64)
    d = depth[mask].astype(np.float64)

    if len(d) > max_points:
        idx = np.random.choice(len(d), max_points, replace=False)
        us, vs, d = us[idx], vs[idx], d[idx]

    X = (us - K[0, 2]) * d / K[0, 0]
    Y = (vs - K[1, 2]) * d / K[1, 1]
    Z = d
    return np.stack([X, Y, Z], axis=1), (us, vs)


def _ray_directions(us: np.ndarray, vs: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Normalised ray direction vectors for each pixel (homogeneous, z=1)."""
    dx = (us - K[0, 2]) / K[0, 0]
    dy = (vs - K[1, 2]) / K[1, 1]
    return np.stack([dx, dy, np.ones_like(dx)], axis=1)


def _solve_rigid(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Solve for rigid T (4x4) such that R @ src + t ≈ tgt (SVD method)."""
    c_src = src.mean(0)
    c_tgt = tgt.mean(0)
    H = (src - c_src).T @ (tgt - c_tgt)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = c_tgt - R @ c_src
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
