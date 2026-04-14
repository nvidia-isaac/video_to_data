"""
Thin wrapper around the gsplat rasterization function.

gsplat uses OpenCV coordinate convention:
  - Camera looks along +Z
  - Y axis points down
  - X axis points right

This matches the depth-space coordinate system established in initialization.py.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from v2d.common.datatypes import CameraIntrinsics
from v2d.gsplat.lib.scene import GaussianScene, FeatureGaussians


@dataclass
class RenderResult:
    rgb: torch.Tensor          # (H, W, 3) in [0, 1]
    depth: torch.Tensor        # (H, W) in depth-space units
    alpha: torch.Tensor        # (H, W) in [0, 1]


def build_viewmat(device: str = 'cuda') -> torch.Tensor:
    """
    Build world-to-camera matrix for a static camera at the depth-space origin.
    Camera at origin looking along +Z → viewmat = identity.
    Returns (1, 4, 4).
    """
    return torch.eye(4, device=device).unsqueeze(0)


def build_K(intrinsics: CameraIntrinsics, device: str = 'cuda') -> torch.Tensor:
    """Build (1, 3, 3) camera intrinsics matrix."""
    K = torch.tensor([
        [intrinsics.fx, 0.0,           intrinsics.cx],
        [0.0,           intrinsics.fy, intrinsics.cy],
        [0.0,           0.0,           1.0],
    ], dtype=torch.float32, device=device)
    return K.unsqueeze(0)  # (1, 3, 3)


def render(
    scene: GaussianScene,
    world_positions: torch.Tensor,        # (N, 3) — may differ from scene.positions
    viewmat: torch.Tensor,                # (1, 4, 4) world-to-camera
    K: torch.Tensor,                      # (1, 3, 3)
    H: int,
    W: int,
    near: float = 0.01,
    far: float = 100.0,
    sh_degree: int = 3,
) -> RenderResult:
    """
    Rasterize the scene given world-space positions.

    world_positions may be different from scene.positions when deformation has
    been applied (body LBS, object SE(3) transforms).
    """
    from gsplat import rasterization

    renders, alphas, _ = rasterization(
        means=world_positions,
        quats=scene.rotations,          # (N, 4) normalised
        scales=scene.scales,            # (N, 3)
        opacities=scene.opacities,      # (N,)
        colors=scene.sh_features,       # (N, 16, 3) for SH degree 3
        viewmats=viewmat,               # (1, 4, 4)
        Ks=K,                           # (1, 3, 3)
        width=W,
        height=H,
        near_plane=near,
        far_plane=far,
        sh_degree=sh_degree,
        render_mode='RGB+D',
        backgrounds=torch.zeros(1, 3, device=world_positions.device),
        packed=False,
    )
    # renders: (1, H, W, 4)  alphas: (1, H, W, 1)
    rgb = renders[0, :, :, :3]   # (H, W, 3)
    depth = renders[0, :, :, 3]  # (H, W)
    alpha = alphas[0, :, :, 0]   # (H, W)

    return RenderResult(rgb=rgb, depth=depth, alpha=alpha)


def render_semantic(
    scene: GaussianScene,
    world_positions: torch.Tensor,  # (N, 3)
    entity_class_map: torch.Tensor, # (N,) int — compact class index per Gaussian
    n_classes: int,
    viewmat: torch.Tensor,          # (1, 4, 4)
    K: torch.Tensor,                # (1, 3, 3)
    H: int,
    W: int,
) -> torch.Tensor:                  # (H, W, n_classes) soft class probabilities
    """
    Render per-Gaussian class labels through the full depth-ordered scene.

    Each Gaussian carries a one-hot class vector derived from its entity_id.
    Alpha compositing propagates these through depth ordering, so occluded
    Gaussians naturally contribute less to the output class probabilities.

    Returns (H, W, n_classes) — each pixel is a soft probability vector that
    sums to approximately the total scene alpha at that pixel (not normalised
    to 1, analogous to how rendered RGB is not divided by alpha).
    """
    from gsplat import rasterization

    # One-hot class vectors: (N, 1, n_classes) — gsplat requires 3D (N, K, C) even for sh_degree=0
    class_colors = torch.zeros(
        scene.num_gaussians, n_classes, device=world_positions.device, dtype=torch.float32
    )
    class_colors.scatter_(1, entity_class_map.unsqueeze(1), 1.0)
    class_colors = class_colors.unsqueeze(1)  # (N, 1, n_classes)

    renders, _, _ = rasterization(
        means=world_positions,
        quats=scene.rotations,
        scales=scene.scales,
        opacities=scene.opacities,
        colors=class_colors,          # (N, 1, n_classes) — sh_degree=0, 1 SH band
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        near_plane=0.01,
        far_plane=100.0,
        sh_degree=0,                  # constant per-Gaussian, no SH
        render_mode='RGB',
        backgrounds=torch.zeros(1, n_classes, device=world_positions.device),
        packed=False,
    )
    # renders: (1, H, W, n_classes)
    return renders[0]  # (H, W, n_classes)


def project_and_sample_features(
    feat_gaussians: FeatureGaussians,
    world_positions: torch.Tensor,   # (M, 3) — may differ from feat_gaussians.positions
    K: torch.Tensor,                 # (1, 3, 3) scaled to feature-map resolution
    target_features: torch.Tensor,  # (h, w, D) pre-extracted feature map
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project feature Gaussians to 2D and sample target features at those locations.

    Gradients flow through:
      - world_positions → pose (R, t) and feat_gaussians._positions
      - feat_gaussians._features → cosine similarity

    Returns:
      gauss_feats:  (M', D) — feature vectors of valid (in-frame) Gaussians
      target_feats: (M', D) — bilinearly sampled target at each projected location
    Both tensors have the same M' ≤ M rows; M' can be 0.
    """
    import torch.nn.functional as F_func

    h, w, D = target_features.shape
    k = K[0]  # (3, 3)

    # Perspective projection
    proj = world_positions @ k.T   # (M, 3)
    z = proj[:, 2]
    u = proj[:, 0] / z.clamp(min=1e-4)  # (M,)  pixel x
    v = proj[:, 1] / z.clamp(min=1e-4)  # (M,)  pixel y

    # Keep only in-frame points in front of camera
    valid = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    if not valid.any():
        dummy = world_positions.sum() * 0.0
        return (feat_gaussians.features[:0] + dummy,
                target_features.reshape(-1, D)[:0].detach() + dummy)

    # Normalise to [-1, 1] for grid_sample (u→x, v→y)
    u_n = u / (w - 1) * 2.0 - 1.0  # (M,)
    v_n = v / (h - 1) * 2.0 - 1.0  # (M,)
    grid = torch.stack([u_n, v_n], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, M, 2)

    # Bilinear sample: (1, D, h, w) → (1, D, 1, M) → (M, D)
    target_4d = target_features.permute(2, 0, 1).unsqueeze(0)  # (1, D, h, w)
    sampled = F_func.grid_sample(target_4d, grid, mode='bilinear',
                                 align_corners=True, padding_mode='border')
    sampled = sampled.squeeze(0).squeeze(1).permute(1, 0)  # (M, D)

    return feat_gaussians.features[valid], sampled[valid].detach()


def render_entity_silhouette(
    scene: GaussianScene,
    entity_mask: torch.Tensor,      # (N,) bool — True for the entity we want
    world_positions: torch.Tensor,  # (N, 3)
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Render alpha channel for a specific entity (e.g., body or one object).
    Returns (H, W) in [0, 1].
    """
    from gsplat import rasterization

    # Zero out opacities for all other entities
    opacities_masked = scene.opacities.clone()
    opacities_masked[~entity_mask] = 0.0

    # Simple constant-color render is enough for silhouette
    colors = torch.ones(scene.num_gaussians, 3, device=world_positions.device)

    _, alphas, _ = rasterization(
        means=world_positions,
        quats=scene.rotations,
        scales=scene.scales,
        opacities=opacities_masked,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        near_plane=0.01,
        render_mode='RGB',
        packed=False,
    )
    return alphas[0, :, :, 0]  # (H, W)
