"""
ICP utilities for aligning NLF SMPL predictions to depth point clouds.
Based on cari4d_internal/tools/icp_utils.py
"""
import open3d as o3d
import numpy as np
import torch
from typing import Optional


def translation_only_icp_torch(src: o3d.geometry.PointCloud, 
                               tgt: o3d.geometry.PointCloud, 
                               R_fixed: np.ndarray = np.eye(3), 
                               voxel_size: float = 0.01, 
                               max_iter: int = 30, 
                               tol: float = 0.001, 
                               max_iters: Optional[list] = None) -> np.ndarray:
    """
    Translation-only ICP using PyTorch3D for fast CUDA-accelerated nearest neighbor search.
    Falls back to Open3D CPU implementation if PyTorch3D GPU is not available.
    Optimizes only z-axis translation.
    
    Args:
        src: Source point cloud (Open3D)
        tgt: Target point cloud (Open3D)
        R_fixed: Fixed rotation matrix (default: identity)
        voxel_size: Voxel size for downsampling
        max_iter: Maximum iterations per level
        tol: Convergence tolerance
        max_iters: List of max iterations for each pyramid level (default: [25, 10, 5])
    
    Returns:
        4x4 transformation matrix
    """
    if max_iters is None:
        max_iters = [25, 10, 5]
    
    # Try PyTorch3D GPU version first
    try:
        from pytorch3d.ops import knn_points
        
        # Work on copies
        src_work = o3d.geometry.PointCloud(src)
        tgt_work = o3d.geometry.PointCloud(tgt)

        # Apply (fixed) rotation once
        src_work.rotate(R_fixed, center=(0, 0, 0))

        total_t = np.zeros(3)
        voxel_radius = [voxel_size * 8, voxel_size * 4, voxel_size]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for it, radius in zip(max_iters, voxel_radius):
            # Keep Open3D voxel downsampling
            src_down = src_work.voxel_down_sample(voxel_size=voxel_size)
            tgt_down = tgt_work.voxel_down_sample(voxel_size=voxel_size)

            # Convert to torch tensors
            src_pts = torch.from_numpy(np.asarray(src_down.points)).float().to(device)  # (N,3)
            tgt_pts = torch.from_numpy(np.asarray(tgt_down.points)).float().to(device)  # (M,3)
            if src_pts.numel() == 0 or tgt_pts.numel() == 0:
                break

            thr2 = float((radius * 2) ** 2)
            total_t_it = np.zeros(3)

            for _ in range(it):
                try:
                    # PyTorch3D batched knn (k=1)
                    # shapes: dists (1,N,1), idx (1,N,1)
                    dists, idx, _ = knn_points(src_pts.unsqueeze(0), tgt_pts.unsqueeze(0), K=1)
                    d2 = dists[0, :, 0]  # (N,)
                    nn_idx = idx[0, :, 0].long()  # (N,)

                    # inlier mask by squared distance
                    mask = d2 < thr2
                    if mask.sum().item() == 0:
                        break

                    # gather matched target points
                    tgt_matched = tgt_pts[nn_idx]  # (N,3)

                    # residuals (tgt - src) for inliers only
                    res = tgt_matched[mask] - src_pts[mask]  # (K,3)

                    # mean translation update; constrain to z only
                    delta_t = res.mean(dim=0)
                    delta_t[:2] = 0.0

                    if torch.norm(delta_t).item() < tol:
                        break

                    # apply update on the working src points tensor
                    src_pts = src_pts + delta_t

                    # accumulate totals for outputs and to advance full-res cloud
                    delta_np = delta_t.detach().cpu().numpy()
                    total_t = total_t + delta_np
                    total_t_it += delta_np
                except RuntimeError as e:
                    if "GPU" in str(e) or "CUDA" in str(e):
                        # PyTorch3D GPU not available, fall back to Open3D
                        raise RuntimeError("PyTorch3D GPU not available") from e
                    raise

            # advance the high-res source cloud for the next pyramid level
            src_work.translate(total_t_it)

        # Build final 4x4
        T = np.eye(4)
        T[:3, :3] = R_fixed
        T[:3, 3] = total_t
        return T
    
    except (ImportError, RuntimeError) as e:
        # Fallback to Open3D CPU implementation
        if "GPU" in str(e) or "CUDA" in str(e) or "Not compiled" in str(e) or isinstance(e, ImportError):
            import warnings
            warnings.warn(f"PyTorch3D GPU not available ({e}), falling back to Open3D CPU implementation. This will be slower.")
            return translation_only_icp_open3d(src, tgt, R_fixed, voxel_size, max_iters)
        raise


def translation_only_icp_open3d(src: o3d.geometry.PointCloud,
                                tgt: o3d.geometry.PointCloud,
                                R_fixed: np.ndarray = np.eye(3),
                                voxel_size: float = 0.01,
                                max_iters: Optional[list] = None) -> np.ndarray:
    """
    Translation-only ICP using Open3D (CPU implementation).
    Optimizes only z-axis translation.
    """
    if max_iters is None:
        max_iters = [25, 10, 5]
    
    # Work on copies
    src_work = o3d.geometry.PointCloud(src)
    tgt_work = o3d.geometry.PointCloud(tgt)

    # Apply (fixed) rotation once
    src_work.rotate(R_fixed, center=(0, 0, 0))

    total_t = np.zeros(3)
    voxel_radius = [voxel_size * 8, voxel_size * 4, voxel_size]

    for it, radius in zip(max_iters, voxel_radius):
        src_down = src_work.voxel_down_sample(voxel_size=voxel_size)
        tgt_down = tgt_work.voxel_down_sample(voxel_size=voxel_size)

        if len(src_down.points) == 0 or len(tgt_down.points) == 0:
            break

        # Build KDTree for target
        tgt_tree = o3d.geometry.KDTreeFlann(tgt_down)
        total_t_it = np.zeros(3)

        for _ in range(it):
            src_pts = np.asarray(src_down.points)
            matches = []
            distances = []

            # Find nearest neighbors
            for pt in src_pts:
                [_, idx, dist] = tgt_tree.search_knn_vector_3d(pt, 1)
                if len(idx) > 0:
                    matches.append(np.asarray(tgt_down.points)[idx[0]])
                    distances.append(dist[0])

            if len(matches) == 0:
                break

            matches = np.array(matches)
            distances = np.array(distances)
            thr2 = (radius * 2) ** 2

            # Inlier mask
            mask = distances < thr2
            if mask.sum() == 0:
                break

            # Residuals for inliers only
            res = matches[mask] - src_pts[mask]

            # Mean translation update; constrain to z only
            delta_t = res.mean(axis=0)
            delta_t[:2] = 0.0

            if np.linalg.norm(delta_t) < 0.001:
                break

            # Apply update to downsampled cloud
            src_down.translate(delta_t)
            total_t = total_t + delta_t
            total_t_it += delta_t

        # Advance the high-res source cloud
        src_work.translate(total_t_it)

    # Build final 4x4
    T = np.eye(4)
    T[:3, :3] = R_fixed
    T[:3, 3] = total_t
    return T

