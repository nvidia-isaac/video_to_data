"""Losses for joint hand+object Gaussian refinement.

Photometric:    L1 between rendered RGB and image RGB, masked to the union
                of hand+object SAM2 masks (background pixels are not
                supervised — Gaussians don't have to explain them).

Silhouette:     BCE-ish L1 between per-set rendered alpha and the
                corresponding SAM2 mask. Object alpha vs. object mask, each
                hand alpha vs. its own SAM2 hand mask.

Depth:          Huber on rendered depth vs. MoGe depth, masked to the union
                mask AND to pixels where MoGe is finite. Low default weight
                (0.05) since MoGe depth is temporally jittery.

Smoothness:     Penalize first-difference of per-frame pose params (axis-angle
                + translation for object; global_orient + hand_pose + cam_t
                for hand).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def photometric_loss(
    rendered_rgb: torch.Tensor,    # (H, W, 3)
    target_rgb: torch.Tensor,      # (H, W, 3)
    union_mask: torch.Tensor,      # (H, W) in {0, 1}
    mask_background_to_black: bool = False,
) -> torch.Tensor:
    """Per-foreground-pixel L1 between rendered and target RGB.

    With ``mask_background_to_black=False`` (default): only foreground (mask)
    pixels contribute — safer when the SAM2 masks are noisy, since wrong
    mask pixels don't get a hard "match black" target imposed.

    With ``mask_background_to_black=True``: target's background is zeroed to
    match the renderer's black BG and the L1 is summed over the whole image
    (then normalized by foreground pixel count). Penalizes Gaussians that
    leak outside the mask, but makes mask errors more costly.

    In both cases the normalization is ``sum / union_mask.sum()`` so per-
    foreground-pixel gradient magnitude is image-size-invariant.
    """
    n = union_mask.sum().clamp_min(1.0)
    if mask_background_to_black:
        target = target_rgb * union_mask.unsqueeze(-1)
        diff = (rendered_rgb - target).abs().sum(dim=-1)    # (H, W)
        return diff.sum() / n
    diff = (rendered_rgb - target_rgb).abs().sum(dim=-1)    # (H, W)
    return (diff * union_mask).sum() / n


def delta_p_regularizer(delta_p: torch.Tensor) -> torch.Tensor:
    """Sum-squared L2 on per-Gaussian position offsets in canonical frame.

    Pulls Gaussians back toward their mesh-vertex anchor; without this the
    optimizer is free to slide them anywhere photometrically convenient,
    which manifests as Gaussians "scattering" off the mesh surface.

    Sum reduction (not mean) avoids the 1/N gradient dilution: with .mean()
    over N=1000 Gaussians, each Δp_i gets ~1000× less gradient, so a
    weight of 100 effectively acts like 0.1 — too weak to compete with
    photometric pulls of ~0.1-1.0 magnitude.
    """
    return (delta_p * delta_p).sum()


def silhouette_loss(
    rendered_class_map: torch.Tensor,    # (H, W, K) per-pixel class probabilities
    target_class_map: torch.Tensor,      # (H, W, K) one-hot per-pixel class targets
    class_weights: torch.Tensor | None = None,   # (K,) optional per-class weight
) -> torch.Tensor:
    """Per-pixel L1 between a composited class-label render and the SAM2
    one-hot class targets, with optional per-class weighting.

    Class-label rendering: each Gaussian carries a K-dim one-hot label
    (object=class 0, hand_i=class i+1). All Gaussians render together with
    proper depth ordering, so the gsplat rasterizer's alpha compositing
    naturally accounts for occlusion: at a pixel where the hand is in
    front of the object, the rendered class probability is dominated by
    "hand" — exactly what SAM2 labels there.

    ``class_weights`` (shape (K,)) lets the caller weight different mask
    sources differently, e.g. trust the SAM2 object mask more than the
    hand masks when the latter are flickery.
    """
    diff = (rendered_class_map - target_class_map).abs()       # (H, W, K)
    if class_weights is not None:
        diff = diff * class_weights                             # broadcast (K,) → (H, W, K)
    return diff.mean()


def depth_loss(
    rendered_depth: torch.Tensor,  # (H, W)
    target_depth: torch.Tensor,    # (H, W)
    union_mask: torch.Tensor,      # (H, W) in {0, 1}
    huber_delta: float = 0.05,
) -> torch.Tensor:
    """Huber on depth difference, masked to union ∩ valid-target.

    ``target_depth`` may contain +inf (sky / no MoGe estimate) — those pixels
    are excluded from the loss.
    """
    valid = union_mask * torch.isfinite(target_depth).float()
    diff = (rendered_depth - target_depth) * valid
    abs_diff = diff.abs()
    quad = 0.5 * diff * diff
    lin  = huber_delta * (abs_diff - 0.5 * huber_delta)
    huber = torch.where(abs_diff <= huber_delta, quad, lin)
    n = valid.sum().clamp_min(1.0)
    return huber.sum() / n


def temporal_smoothness(param: torch.Tensor) -> torch.Tensor:
    """Sum-squared first-difference of a (T, D) parameter tensor.

    Sum reduction (not mean) keeps per-frame-pair gradient magnitude
    independent of sequence length; with mean, a 500-frame clip dilutes
    each pair's gradient by ~1/(3*(T-1)) ≈ 1/1500, which makes weights
    in the 0.01-0.1 range effectively negligible.

    Use ``rotation_smoothness`` instead for axis-angle parameters — naive
    L2 smoothness on axis-angle is broken near the |θ|=π boundary because
    of the ±2π·n/‖n‖ representational ambiguity.
    """
    if param.shape[0] < 2:
        return param.new_zeros(())
    d = param[1:] - param[:-1]
    return (d * d).sum()


def rotation_smoothness(axis_angle: torch.Tensor) -> torch.Tensor:
    """Sum-squared first-difference of axis-angle rotations, measured in
    quaternion space with double-cover sign alignment.

    Why not L2 directly on axis-angle: the same rotation has two
    representations across the |θ|=π boundary (``aa`` and
    ``aa - 2π·aa/‖aa‖``). Per-frame regressors like FoundationPose / HaMeR
    sometimes pick different representations for nearly-identical
    rotations, so direct L2 smoothness sees a phantom huge difference and
    pushes the optimizer to discontinuously flip — visible as sporadic
    orientation flips in refined poses.

    Converting to unit quaternions and sign-aligning consecutive pairs
    (q ≡ -q) makes wrap invisible to the loss. The squared-difference is
    then proportional to ``1 - cos(θ_rel/2)``, a smooth function of the
    relative rotation angle.

    Accepts (T, 3) for a single rotation per frame or (T, ..., 3) for
    multiple (e.g. (T, 15, 3) for the 15-joint MANO ``hand_pose``).
    """
    if axis_angle.shape[0] < 2:
        return axis_angle.new_zeros(())

    angle = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half = angle * 0.5
    w = torch.cos(half)
    xyz = axis_angle / angle * torch.sin(half)
    q = torch.cat([w, xyz], dim=-1)                             # (T, ..., 4)

    dot = (q[1:] * q[:-1]).sum(dim=-1, keepdim=True)
    q_aligned = torch.where(dot < 0, -q[1:], q[1:])
    d = q_aligned - q[:-1]
    return (d * d).sum()


def balanced_photometric_loss(
    rendered_rgb: torch.Tensor,             # (H, W, 3)
    target_rgb: torch.Tensor,               # (H, W, 3)
    obj_mask: torch.Tensor,                 # (H, W) in {0, 1}
    hand_masks: list[torch.Tensor],         # each (H, W) in {0, 1}
    include_background: bool = True,
) -> torch.Tensor:
    """Per-pixel L1 weighted so each *entity* contributes equally,
    regardless of how many pixels it covers.

    For each entity i (object, hand_0, hand_1, ..., optionally background),
    every pixel in that entity's mask gets per-pixel weight 1 / N_i. The
    integral of the weight over the entity's mask is therefore 1 — so a
    1%-coverage hand contributes the same total loss mass as a 10%-coverage
    object or an 89%-coverage background. Without this rebalancing, a
    full-image L1 is dominated by whatever covers the most pixels.

    Background, when included, is the complement of the union of all
    foreground masks.
    """
    diff = (rendered_rgb - target_rgb).abs().sum(dim=-1)        # (H, W)
    weight_map = torch.zeros_like(diff)
    weight_map = weight_map + obj_mask / obj_mask.sum().clamp_min(1.0)

    union = obj_mask.clone()
    for hm in hand_masks:
        weight_map = weight_map + hm / hm.sum().clamp_min(1.0)
        union = torch.maximum(union, hm)

    if include_background:
        bg_mask = 1.0 - union
        weight_map = weight_map + bg_mask / bg_mask.sum().clamp_min(1.0)

    return (diff * weight_map).sum()


def beta_prior_loss(beta: torch.Tensor, beta_init: torch.Tensor) -> torch.Tensor:
    """Sum-squared deviation of MANO shape from its initialization.

    ``beta`` and ``beta_init`` are (10,). Sum (not mean) so each individual
    component is penalized at full weight — keeps the prior tight regardless
    of the 10-dim normalization.
    """
    d = beta - beta_init
    return (d * d).sum()
