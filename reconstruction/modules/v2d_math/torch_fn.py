from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


#### Projective Geometry ####

def reproject(
    xyz: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    distort_fn: Callable | None = None,
    eps: float = 1e-4,
):
    """
    Args:
        xyz: (P, 3) tensor of points in world coordinates [x, y, z]
        K: (3, 3) tensor of camera intrinsic matrix
        T: (4, 4) tensor of camera extrinsic matrix (world to camera)
    Returns:
        reprojection: (P, 2) tensor of reprojection in 2D image coordinates [u, v]
        depth_mask: (P,) tensor of depth mask
    """
    rot = T[:3, :3]  # (3, 3)
    trans = T[:3, 3]  # (3,)
    xyz_cam = xyz @ rot.T + trans
    z = xyz_cam[:, 2:3]
    z_safe = torch.where(z.abs() < eps, torch.where(z >= 0, eps, -eps), z)
    xy = xyz_cam[:, :2] / z_safe  # (P, 2)
    if distort_fn is not None:
        xy = distort_fn(xy)
    return xy @ K[:2, :2].T + K[:2, 2], xyz_cam[:, 2] > eps  # (P, 2), (P,)


def reproject_multiview(
    xyz: torch.Tensor,
    K: torch.Tensor,
    T: torch.Tensor,
    distort_fn: Callable | None = None,
    eps: float = 1e-4,
):
    """
    Args:
        xyz: (N, P, 3) tensor of points in world coordinates [x, y, z]
        K: (C, 3, 3) tensor of camera intrinsic matrices
        T: (C, 4, 4) tensor of camera extrinsic matrices (world to camera)
    Returns:
        reprojections: (C, N, P, 2) tensor of reprojections in 2D image coordinates [u, v]
        depth_mask: (C, N, P) tensor of depth mask
    """
    rot = T[:, :3, :3]  # (C, 3, 3)
    trans = T[:, :3, 3].reshape(-1, 1, 1, 3)  # (C, 1, 1, 3)
    xyz_cam = torch.einsum("cab,npb->cnpa", rot, xyz) + trans  # (C, N, P, 3)

    z = xyz_cam[..., 2:3]
    z_safe = torch.where(z.abs() < eps, torch.where(z >= 0, eps, -eps), z)
    xy = xyz_cam[..., :2] / z_safe  # (C, N, P, 2)
    C, N, P, _ = xy.shape
    xy = xy.reshape(C, N * P, -1)
    if distort_fn is not None:
        xy = distort_fn(xy)
    K_f = K[:, :2, :2].transpose(-1, -2)  # (C, 2, 2)
    K_p = K[:, :2, 2].reshape(-1, 1, 2)  # (C, 1, 2)
    uv = xy @ K_f + K_p  # (C, N * P, 2)
    return uv.reshape(C, N, P, 2), xyz_cam[..., 2] > eps


def distort_polynomial(
    xy: torch.Tensor,
    coeffs: torch.Tensor,
):
    """
    Args:
        xy: (C, N, 2) tensor of points from C cameras in 2D image coordinates [x, y]
        coeffs: (C, 8) tensor of distortion coefficients
    Returns:
        uv_distorted: (C, N, 2) tensor of distorted points in 2D image coordinates [x, y]
    """
    x = xy[..., 0]
    y = xy[..., 1]
    r2 = x**2 + y**2  # (C, N)
    k1, k2, p1, p2, k3, k4, k5, k6 = coeffs.split(1, dim=-1)  # (C, 1)
    kr = (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2)
    x_distorted = x * kr + p1 * (2.0 * x * y) + p2 * (r2 + 2.0 * x**2)
    y_distorted = y * kr + p1 * (r2 + 2.0 * y**2) + p2 * (2.0 * x * y)
    return torch.stack([x_distorted, y_distorted], dim=-1)  # (C, N, 2)


#### Manifold Geometry ####

def so3_exp_map(aa: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the exponential map of SO(3) using Rodrigues' formula.
    Maps axis-angle vectors to rotation matrices.
    
    Args:
        aa: (..., 3) axis-angle rotation vectors
        eps: epsilon for numerical stability near zero
    Returns:
        (..., 3, 3) rotation matrices
    """
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)  # (..., 1)
    theta_clamped = theta.clamp(min=eps)
    
    k = aa / theta_clamped  # unit axis (..., 3)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]

    # Skew-symmetric matrix [k]_x
    zero = torch.zeros_like(kx)
    K = torch.stack([
        zero,   -kz,    ky,
        kz,     zero,  -kx,
        -ky,      kx,    zero
    ], dim=-1).reshape(aa.shape[:-1] + (3, 3))
    
    I = torch.eye(3, device=aa.device, dtype=aa.dtype)
    I = I.expand(aa.shape[:-1] + (3, 3))
    
    sin_t = torch.sin(theta)[..., None]  # (..., 1, 1)
    cos_t = torch.cos(theta)[..., None]  # (..., 1, 1)
    
    # Rodrigues' formula: R = I + sin(t)K + (1-cos(t))K^2
    # For small angles, this is stable enough with float32 if eps is handled
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def so3_log_map(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the logarithmic map of SO(3).
    Maps rotation matrices to axis-angle vectors.
    
    Args:
        R: (..., 3, 3) rotation matrices
        eps: epsilon for numerical stability
    Returns:
        (..., 3) axis-angle rotation vectors
    """
    # Clamp trace to valid range for acos
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    theta = torch.acos(cos_theta)
    theta2 = theta * theta

    # Small-angle mask
    small = theta2 < (eps * eps)

    # Skew-symmetric part
    #   S = (R - R^T)/2 ~ [omega_hat]
    S = 0.5 * (R - R.transpose(-1, -2))  # (..., 3, 3)

    # For small theta, omega ≈ vee(S)
    omega_small = torch.stack(
        [S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]],
        dim=-1
    )  # (..., 3)

    # For general case, use theta/(2 sin theta) * (R - R^T)
    sin_theta = torch.sin(theta)
    # Avoid division by zero
    scale = theta / (sin_theta + eps)  # (...,)

    omega_big = scale.unsqueeze(-1) * torch.stack(
        [S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]],
        dim=-1
    )  # (..., 3)

    omega = torch.where(small.unsqueeze(-1), omega_small, omega_big)  # (..., 3)
    return omega


def se3_exp_map(pose_vec: torch.Tensor):
    """
    Args:
        pose_vec: (..., 6) tensor [rx, ry, rz, tx, ty, tz]
    Returns:
        pose_mat: (..., 4, 4) homogeneous transform
    """
    rot_vec = pose_vec[..., 0:3]
    trans_vec = pose_vec[..., 3:6]

    R = so3_exp_map(rot_vec)

    T = torch.zeros(pose_vec.shape[:-1] + (4, 4), device=pose_vec.device, dtype=pose_vec.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3]  = trans_vec
    T[..., 3, 3]   = 1.0
    return T


def se3_log_map(pose_mat: torch.Tensor) -> torch.Tensor:
    """
    Convert an SE(3) matrix to exponential coordinates.
    Args:
        pose_mat: (..., 4, 4) tensor
    Returns:
        pose_vec: (..., 6) tensor [rx, ry, rz, tx, ty, tz]
    """
    assert pose_mat.shape[-2:] == (4, 4)

    # Extract rotation R and translation t
    R = pose_mat[..., :3, :3]    # (..., 3, 3)
    t = pose_mat[..., :3, 3]     # (..., 3)

    omega = so3_log_map(R)

    # --- Translation part: just return t directly to match se3_exp_map ---
    xi = torch.cat([omega, t], dim=-1)  # (..., 6)
    return xi


def se3_inv(pose_mat: torch.Tensor) -> torch.Tensor:
    """
    Invert an SE(3) matrix.
    Args:
        pose_mat: (..., 4, 4) tensor
    Returns:
        inv_pose: (..., 4, 4) tensor
    """
    R = pose_mat[..., :3, :3]
    t = pose_mat[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.einsum("...ij,...j->...i", R_inv, t)
    
    inv_pose = torch.zeros_like(pose_mat)
    inv_pose[..., :3, :3] = R_inv
    inv_pose[..., :3, 3] = t_inv
    inv_pose[..., 3, 3] = 1.0
    return inv_pose


def so3_relative_angle(rot1: torch.Tensor, rot2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the relative angle between two rotation matrices.
    Arguments:
        rot1: (..., 3, 3) rotation matrices
        rot2: (..., 3, 3) rotation matrices
    Returns:
        angle: (..., ) relative angles
    """
    cos = ((rot1 @ rot2.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2.0
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def se3_split_distance(pose1: torch.Tensor, pose2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the distance of SE(3) matrices by separately computing the distance of rotation and translation.
    Arguments:
        pose1: (..., 4, 4), N=batch dimension
        pose2: (..., 4, 4), N=batch dimension
    Returns:
        rot_dist: (..., )
        trans_dist: (..., )
    """
    rot1 = pose1[..., :3, :3]
    rot2 = pose2[..., :3, :3]
    trans1 = pose1[..., :3, 3]
    trans2 = pose2[..., :3, 3]
    return so3_relative_angle(rot1, rot2), torch.linalg.norm(trans1 - trans2, dim=-1)


def se3_pose_mean_pairwise_distance(
    poses: torch.Tensor,
    dist_weight: float = 10.0,
) -> torch.Tensor:
    """
    Compute the mean pairwise distance of  a set of SE(3) matrices.
    Arguments:
        poses: (D, 4, 4), D=compare dimension
        dist_weight: float, weight of translation distance
    Returns:
        dist: (D,) tensor of mean pairwise distances
    """
    D = poses.shape[0]
    repeated_poses = torch.repeat_interleave(poses, D, dim=0)  # (D*D, 4, 4)
    tiled_poses = torch.tile(poses, (D, 1, 1))  # (D*D, 4, 4)
    rot_dist, trans_dist = se3_split_distance(repeated_poses, tiled_poses)  # (D*D,), (D*D,)
    rot_dist = rot_dist.reshape(D, D).mean(dim=-1)
    trans_dist = trans_dist.reshape(D, D).mean(dim=-1)
    dist = rot_dist + dist_weight * trans_dist
    return dist


def se3_pose_select_arg(
    poses: torch.Tensor,
    dist_weight: float = 10.0,
    top_n: int = 2,
) -> torch.Tensor:
    """
    Select the best pose from the given poses by the minimum pairwise distance.
    Arguments:
        poses: (D, 4, 4), D=compare dimension
        dist_weight: float, weight of translation distance
        top_n: int, number of best poses to average
    Returns:
        best_idx: (top_n,) tensor of indices
    """
    dist = se3_pose_mean_pairwise_distance(poses, dist_weight)
    best_idx = torch.argsort(dist)[:top_n]
    return best_idx


def se3_pose_inliers(
    poses: torch.Tensor,
    dist_weight: float = 10.0,
    z_score_threshold: float = 2.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Compute the outlier of SE(3) matrices.
    Arguments:
        poses:(D, 4, 4), D=compare dimension
        dist_weight: float, weight of translation distance
    Returns:
        inliers_mask: (D,) tensor of inliers mask
    """
    dist = se3_pose_mean_pairwise_distance(poses, dist_weight)  # (D,)
    mean = torch.mean(dist)
    std = torch.std(dist)
    z_score = (dist - mean) / (std + eps)
    inliers_mask = z_score < z_score_threshold  # (D,)
    return inliers_mask


def rotation_6d_to_matrix(x6d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Args:
        x6d: (..., 6) tensor of 6D rotation representations
        eps: float, epsilon for numerical stability
    Returns:
        (..., 3, 3) tensor of rotation matrices
    """

    a1, a2 = x6d[..., :3], x6d[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    Args:
        R: (..., 3, 3) tensor of rotation matrices
    Returns:
        (..., 6) tensor of 6D rotation representations
    """
    batch_dim = R.shape[:-2]
    return R[..., :2, :].clone().reshape(batch_dim + (6,))


#### Loss Functions ####

def l2_distance(
    x: torch.Tensor,
    y: torch.Tensor,
):
    """
    Computes L2 distance: distance = sum((x - y)^2)
    Args:
        x: (..., D) tensor
        y: (..., D) tensor
    Returns:
        distance: (..., ) tensor of L2 distances
    """
    return torch.sum(torch.square(x - y), dim=-1)


def geman_mcclure_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float,
):
    """
    Computes Geman-McClure distance: rho(r) = r^2 / (c^2 + r^2)
    Args:
        x: (..., D) tensor
        y: (..., D) tensor
        c: float, scale for Geman-McClure distance
    Returns:
        distance: (..., ) tensor of Geman-McClure distances
    """
    r_sq = torch.sum(torch.square(x - y), dim=-1)
    c_sq = c ** 2
    loss = r_sq / (c_sq + r_sq)
    return loss


def recursive_to(x: any, target: torch.device):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        if target == "numpy":
            return x.numpy()
        else:
            return x.to(target)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x
