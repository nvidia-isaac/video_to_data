"""
Loss functions for the gsplat optimization.

All losses return scalar tensors. Weights are applied in optimization.py.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def loss_rgb(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 photometric loss. rendered/target: (H, W, 3) or (B, H, W, 3)."""
    return F.l1_loss(rendered, target)


def loss_ssim(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    SSIM loss (1 - SSIM). Uses pytorch-msssim.
    rendered/target: (H, W, 3) → internally converted to (1, 3, H, W).
    """
    try:
        from pytorch_msssim import ssim
        # Rearrange to (B, C, H, W)
        if rendered.ndim == 3:
            r = rendered.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            t = target.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
        else:
            r = rendered.permute(0, 3, 1, 2).clamp(0, 1)
            t = target.permute(0, 3, 1, 2).clamp(0, 1)
        return 1.0 - ssim(r, t, data_range=1.0, size_average=True)
    except ImportError:
        # Fallback: simple gradient-based proxy
        return _simple_ssim_approx(rendered, target)


def _simple_ssim_approx(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lightweight SSIM-like loss without pytorch_msssim dependency."""
    diff = rendered - target
    return (diff ** 2).mean() + diff.abs().mean()


def loss_depth(
    rendered_depth: torch.Tensor,       # (H, W) or (B, H, W)
    target_depth: torch.Tensor,         # same shape, depth in world units
    valid_mask: Optional[torch.Tensor] = None,  # bool, same shape
) -> torch.Tensor:
    """
    L1 depth regularisation loss.
    rendered_depth and target_depth must be in the same coordinate system (depth-space).
    """
    if valid_mask is not None:
        diff = (rendered_depth[valid_mask] - target_depth[valid_mask]).abs()
    else:
        valid = (target_depth > 0) & (rendered_depth > 0)
        diff = (rendered_depth[valid] - target_depth[valid]).abs()
    if diff.numel() == 0:
        return rendered_depth.sum() * 0.0
    return diff.mean()


def loss_mask(
    rendered_alpha: torch.Tensor,   # (H, W) in [0, 1]
    target_mask: torch.Tensor,      # (H, W) bool or float in {0, 1}
) -> torch.Tensor:
    """
    Symmetric BCE silhouette loss between rendered alpha and SAM2 mask.
    Pushes alpha UP where mask=1 and DOWN where mask=0.
    Use for the full scene combined mask (background needs both directions).
    """
    alpha = rendered_alpha.clamp(1e-6, 1 - 1e-6)
    mask = target_mask.float()
    return F.binary_cross_entropy(alpha, mask)


def loss_mask_asymmetric(
    rendered_alpha: torch.Tensor,   # (H, W) in [0, 1]
    target_mask: torch.Tensor,      # (H, W) bool or float in {0, 1}
    outside_weight: float = 0.1,
) -> torch.Tensor:
    """
    Asymmetric silhouette loss for physically-grounded entities (body, objects).

    - Full BCE weight where mask=1: strongly pulls opacity up so the entity is
      visible where SAM2 confirms it.
    - Reduced weight (outside_weight) where mask=0: gentle boundary signal that
      prevents Gaussians from spreading freely, without aggressively suppressing
      opacity on occluded/thin parts that correctly project outside the 2D mask
      (e.g. a hand partially behind the torso).

    Use symmetric loss_mask for the full-scene combined mask (background needs
    equal pressure in both directions).
    """
    alpha = rendered_alpha.clamp(1e-6, 1 - 1e-6)
    mask = target_mask.float()
    per_pixel = F.binary_cross_entropy(alpha, mask, reduction='none')
    w = torch.where(mask > 0.5,
                    torch.ones_like(mask),
                    torch.full_like(mask, outside_weight))
    return (per_pixel * w).mean()


def loss_anchor(
    canonical_positions: torch.Tensor,  # (N, 3) current canonical positions
    target_positions: torch.Tensor,     # (N, 3) fixed reference (mesh / SMPL T-pose)
) -> torch.Tensor:
    """
    Soft anchor: pull canonical Gaussian positions back toward their mesh reference.

    Prevents monocular ambiguity from letting Gaussians drift to positions that
    explain the limited observed viewpoints but don't correspond to the true 3D
    surface.  Weight controls the tradeoff between trusting the mesh shape vs.
    allowing refinement from video (clothing, mesh inaccuracies, etc.).
    """
    return F.mse_loss(canonical_positions, target_positions)


def loss_temporal_smooth(
    pose_tensor: torch.Tensor,  # (T, D) — any per-frame pose vector
) -> torch.Tensor:
    """Penalise large per-frame differences (first-order temporal smoothness)."""
    if pose_tensor.shape[0] < 2:
        return pose_tensor.sum() * 0.0
    diff = pose_tensor[1:] - pose_tensor[:-1]
    return (diff ** 2).mean()


def loss_skinning_sparsity(skinning_weights: torch.Tensor) -> torch.Tensor:
    """L1 sparsity on skinning weights — encourages few bones per Gaussian."""
    return skinning_weights.abs().mean()


def loss_rigid(
    canonical_positions: torch.Tensor,  # (N, 3) canonical object positions
    world_positions: torch.Tensor,       # (N, 3) world positions after SE(3)
    R: torch.Tensor,                     # (3, 3) rotation
    t: torch.Tensor,                     # (3,) translation
) -> torch.Tensor:
    """
    Penalise deviation from a perfectly rigid transform.
    Expected world_pos = canonical_pos @ R.T + t.
    """
    expected = canonical_positions @ R.T + t.unsqueeze(0)
    return F.mse_loss(world_positions, expected.detach())


def compute_total_loss(
    rendered_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    rendered_depth: Optional[torch.Tensor],
    target_depth: Optional[torch.Tensor],
    rendered_alpha: Optional[torch.Tensor],
    target_mask: Optional[torch.Tensor],
    body_pose_params=None,
    scene=None,
    weights: Optional[dict] = None,
) -> Tuple['torch.Tensor', dict]:
    """
    Compute weighted total loss and return per-term breakdown.
    weights: override default loss weights (keys: rgb, ssim, depth, mask, smooth, skinning)
    """
    w = {
        'rgb': 1.0,
        'ssim': 0.2,
        'depth': 0.1,
        'mask': 0.5,
        'smooth': 0.01,
        'skinning': 0.01,
    }
    if weights is not None:
        w.update(weights)

    terms = {}

    terms['rgb'] = loss_rgb(rendered_rgb, target_rgb)
    terms['ssim'] = loss_ssim(rendered_rgb, target_rgb)

    if rendered_depth is not None and target_depth is not None:
        terms['depth'] = loss_depth(rendered_depth, target_depth)
    else:
        terms['depth'] = torch.tensor(0.0, device=rendered_rgb.device)

    if rendered_alpha is not None and target_mask is not None:
        terms['mask'] = loss_mask(rendered_alpha, target_mask)
    else:
        terms['mask'] = torch.tensor(0.0, device=rendered_rgb.device)

    if body_pose_params is not None:
        go = body_pose_params.global_orient  # (T, 3)
        bp = body_pose_params.body_pose      # (T, J*3)
        tr = body_pose_params.transl         # (T, 3)
        terms['smooth'] = (
            loss_temporal_smooth(go)
            + loss_temporal_smooth(bp)
            + loss_temporal_smooth(tr)
        ) / 3.0
    else:
        terms['smooth'] = torch.tensor(0.0, device=rendered_rgb.device)

    if scene is not None and scene.skinning_weights is not None:
        terms['skinning'] = loss_skinning_sparsity(scene.skinning_weights)
    else:
        terms['skinning'] = torch.tensor(0.0, device=rendered_rgb.device)

    total = sum(w[k] * v for k, v in terms.items())
    return total, {k: v.item() for k, v in terms.items()}
