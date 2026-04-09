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
from v2d.gsplat.lib.scene import GaussianScene


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
