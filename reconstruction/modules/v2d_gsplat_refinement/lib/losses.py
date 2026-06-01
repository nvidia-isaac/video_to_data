# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


# ---------------------------------------------------------------------------
# SSIM (Gaussian-window, masked). 11x11 σ=1.5 window, matching the 3DGS /
# original SSIM paper. Cached per (window_size, channels, device, dtype) so
# the window tensor is built once.
# ---------------------------------------------------------------------------

_SSIM_WINDOW_CACHE: dict = {}


def _ssim_window(window_size: int, channels: int, device, dtype) -> torch.Tensor:
    key = (window_size, channels, device, dtype)
    win = _SSIM_WINDOW_CACHE.get(key)
    if win is not None:
        return win
    sigma = 1.5
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]                                  # (K, K)
    win = w2d.expand(channels, 1, window_size, window_size).contiguous()
    _SSIM_WINDOW_CACHE[key] = win
    return win


def _ssim_map(
    x: torch.Tensor,           # (1, C, H, W)
    y: torch.Tensor,           # (1, C, H, W)
    data_range: float,
    window_size: int = 11,
) -> torch.Tensor:
    """Per-pixel SSIM map, (1, C, H, W). Padding=window_size//2."""
    C = x.shape[1]
    pad = window_size // 2
    win = _ssim_window(window_size, C, x.device, x.dtype)
    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x2
    sig_y2 = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y2
    sig_xy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    return num / den


def photometric_ssim_loss(
    rendered_rgb: torch.Tensor,    # (H, W, 3)
    target_rgb: torch.Tensor,      # (H, W, 3)
    union_mask: torch.Tensor,      # (H, W) in {0, 1}
    window_size: int = 11,
) -> torch.Tensor:
    """1 − SSIM between rendered and target RGB, masked to ``union_mask``.

    Standard 11x11 Gaussian-window SSIM (3DGS default). The per-pixel SSIM
    map is computed over the full image and then masked — SSIM windows
    straddling the mask boundary see some invalid content, but for typical
    fg coverage this is a small artifact and matches the 3DGS reference
    implementation. Data range fixed to 1.0 (RGB in [0, 1]).
    """
    x = rendered_rgb.permute(2, 0, 1).unsqueeze(0)                 # (1, 3, H, W)
    y = target_rgb  .permute(2, 0, 1).unsqueeze(0)                 # (1, 3, H, W)
    smap = _ssim_map(x, y, data_range=1.0, window_size=window_size)
    per_pixel = smap.mean(dim=1).squeeze(0)                        # (H, W)
    n = union_mask.sum().clamp_min(1.0)
    return 1.0 - (per_pixel * union_mask).sum() / n


def depth_ssim_loss(
    rendered_depth: torch.Tensor,  # (H, W)
    target_depth: torch.Tensor,    # (H, W), may contain +inf
    union_mask: torch.Tensor,      # (H, W) in {0, 1}
    window_size: int = 11,
    eps: float = 1e-3,
) -> torch.Tensor:
    """1 − SSIM on log-depth, masked and percentile-normalized.

    Depth is unbounded, so we (a) work in log space — same invariance to
    multiplicative scaling as ``depth_gradient_loss`` — and (b) rescale to
    roughly [0, 1] using the 5th/95th percentiles of the target's valid
    region (detached, so the scaling itself doesn't get optimized). This
    keeps SSIM's C1 / C2 constants (calibrated for data_range=1) in the
    right regime regardless of scene scale.

    +inf target pixels are excluded via the mask and replaced with ``eps``
    inside log() so gradients stay finite.
    """
    valid = union_mask * torch.isfinite(target_depth).float()
    if float(valid.sum()) < 10.0:
        return rendered_depth.sum() * 0.0

    dp = torch.log(rendered_depth.clamp_min(eps))
    dt = torch.log(target_depth.nan_to_num(posinf=eps).clamp_min(eps))

    with torch.no_grad():
        dt_v = dt[valid > 0.5]
        lo = torch.quantile(dt_v, 0.05)
        hi = torch.quantile(dt_v, 0.95)
        rng = (hi - lo).clamp_min(1e-3)
    dp_n = ((dp - lo) / rng).clamp(0.0, 1.0)
    dt_n = ((dt - lo) / rng).clamp(0.0, 1.0)

    x = dp_n.unsqueeze(0).unsqueeze(0)                             # (1, 1, H, W)
    y = dt_n.unsqueeze(0).unsqueeze(0)
    smap = _ssim_map(x, y, data_range=1.0, window_size=window_size)
    per_pixel = smap.squeeze(0).squeeze(0)                         # (H, W)
    n = valid.sum().clamp_min(1.0)
    return 1.0 - (per_pixel * valid).sum() / n


def photometric_loss(
    rendered_rgb: torch.Tensor,    # (H, W, 3)
    target_rgb: torch.Tensor,      # (H, W, 3)
    union_mask: torch.Tensor,      # (H, W) in {0, 1}
    mask_background_to_black: bool = False,
    use_l2: bool = False,
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
        d = rendered_rgb - target
    else:
        d = rendered_rgb - target_rgb
    if use_l2:
        per_pixel = (d * d).sum(dim=-1)                     # (H, W)
    else:
        per_pixel = d.abs().sum(dim=-1)
    if mask_background_to_black:
        return per_pixel.sum() / n
    return (per_pixel * union_mask).sum() / n


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


def depth_ordering_loss(
    depth_fg:   torch.Tensor,    # (H, W) E[d] from foreground-only render
    depth_bg:   torch.Tensor,    # (H, W) E[d] from background-only render
    fg_mask:    torch.Tensor,    # (H, W) {0,1} foreground pixels
    margin:     float = 0.0,     # require depth_fg ≤ depth_bg - margin
) -> torch.Tensor:
    """Penetration penalty: foreground Gaussians should be in front of the
    background at foreground-mask pixels.

    Per-pixel loss = max(0, depth_fg - depth_bg + margin), masked to the
    foreground mask and averaged. Fires only when the optimizer pushes
    hand or object Gaussians *behind* the background — exactly the
    penetration mode for that pair. Background pixels (where fg_mask=0)
    contribute zero. Units: [depth].

    Differentiable through both depth maps and the mask is constant
    (gradient flows back into the Gaussian centers).
    """
    diff = (depth_fg - depth_bg + float(margin)).clamp(min=0.0)         # (H, W)
    n = fg_mask.sum().clamp_min(1.0)
    return (diff * fg_mask).sum() / n


def depth_variance_loss(
    depth_var: torch.Tensor,      # (H, W) per-pixel E[d²] − E[d]², ≥ 0
    mask: torch.Tensor | None = None,    # (H, W) {0,1} or None for all pixels
) -> torch.Tensor:
    """Mean per-pixel alpha-weighted depth variance — depth-variance proxy
    for Mip-NeRF 360 distortion loss.

    Penalizes the spread of compositing weights along the view ray: when a
    pixel sums contributions from Gaussians at different depths, the
    variance grows. Minimizing it concentrates per-pixel mass at one
    depth, killing "floater" Gaussians and producing a thin-shell surface.

    Units are [depth²]. If your scene scale is ~0.1 m, variance scales
    are ~0.01, so a weight of ~1.0 gives a loss term of similar magnitude
    to other regularizers; tune from there.
    """
    if mask is None:
        return depth_var.mean()
    n = mask.sum().clamp_min(1.0)
    return (depth_var * mask).sum() / n


def opacity_binary_loss(opacities: torch.Tensor) -> torch.Tensor:
    """Sum of α(1 − α) — pushes each Gaussian's opacity toward 0 or 1.

    The product is zero at α = 0 and α = 1, peaks at 0.25 when α = 0.5, and
    is smooth everywhere (gradient pushes away from 0.5 toward the nearer
    extreme). Sum reduction matches ``delta_p_regularizer``: per-Gaussian
    pressure doesn't get diluted by 1/N, so weights here are comparable to
    the other sequence-wide regularizers.

    Pairs well with face-anchored Gaussians: low-opacity (invisible)
    Gaussians "drop out" rather than stay as semi-transparent fog, which
    tightens the rendered silhouette and lets the optimizer commit to which
    faces actually contribute to the surface.
    """
    return (opacities * (1.0 - opacities)).sum()


def face_delta_p_regularizer(
    delta_p_local: torch.Tensor,    # (F, 3) in face-local (T, B, N) coords
    w_tangent:        float = 1.0,   # penalty on tangent + bitangent slide
    w_normal_outward: float = 100.0, # penalty on Δp_N > 0 (leaks outside the mesh)
    w_normal_inward:  float = 0.0,   # penalty on Δp_N < 0 (sinks into the volume)
) -> torch.Tensor:
    """Asymmetric sum-squared regularizer on face-anchored Gaussian offsets.

    Used in place of ``delta_p_regularizer`` for ``FaceGaussians`` /
    ``HandFaceGaussians``, whose ``_delta_p`` is in face-local (T, B, N)
    coordinates rather than canonical world coordinates.

    The three components have distinct meanings:
      - tangent + bitangent (axes 0, 1): Δp slides the Gaussian across the
        face, parallel to the surface. Light penalty — we want Gaussians to
        slide freely to wherever the photometric signal is best.
      - normal outward (axis 2, positive): Δp pushes the Gaussian out
        through the surface, away from the mesh interior. Heavy penalty —
        this is the "stay inside the volume" constraint.
      - normal inward (axis 2, negative): Δp sinks the Gaussian into the
        volume. Default zero (free) — combined with the outward penalty,
        this lets Gaussians populate the mesh interior under photometric
        pull while never leaking outside.

    Sum reduction matches ``delta_p_regularizer`` so weight magnitudes are
    comparable to the legacy regularizer.
    """
    tb_sq    = (delta_p_local[:, :2] ** 2).sum()
    n        = delta_p_local[:, 2]
    n_out_sq = n.clamp(min=0.0).pow(2).sum()
    n_in_sq  = n.clamp(max=0.0).pow(2).sum()
    return (
        w_tangent        * tb_sq
        + w_normal_outward * n_out_sq
        + w_normal_inward  * n_in_sq
    )


def silhouette_loss(
    rendered_class_map: torch.Tensor,    # (H, W, K) per-pixel class probabilities
    target_class_map: torch.Tensor,      # (H, W, K) one-hot per-pixel class targets
    class_weights: torch.Tensor | None = None,   # (K,) optional per-class weight
    use_l2: bool = False,
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
    d = rendered_class_map - target_class_map                   # (H, W, K)
    diff = (d * d) if use_l2 else d.abs()
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


def depth_gradient_loss(
    rendered_depth: torch.Tensor,  # (H, W)
    target_depth: torch.Tensor,    # (H, W)
    union_mask: torch.Tensor,      # (H, W) in {0, 1}
    log_space: bool = True,
    eps: float = 1e-3,
) -> torch.Tensor:
    """L1 between per-pixel depth gradients of ``rendered`` and ``target``.

    Shape-only depth supervision: ``rendered`` only has to match the *slope*
    of ``target``, not its absolute level. With ``log_space=True`` (default)
    we differentiate ``log(d)`` instead of ``d`` itself, which makes the loss
    invariant to a global multiplicative scaling of either map. That's the
    right invariance for monocular depth like MoGe, whose dominant errors are
    scale/offset rather than local shape.

    Notation: subscript x = horizontal finite difference, y = vertical.
    ``union_mask`` and a finite-target check gate both endpoints of each
    difference so we never compare across an invalid pixel.
    """
    valid = union_mask * torch.isfinite(target_depth).float()
    if log_space:
        d_r = torch.log(rendered_depth.clamp_min(eps))
        d_t = torch.log(target_depth.clamp_min(eps))
    else:
        d_r = rendered_depth
        d_t = target_depth

    # Horizontal differences (between cols c and c+1).
    gx_r = d_r[:, 1:] - d_r[:, :-1]
    gx_t = d_t[:, 1:] - d_t[:, :-1]
    mx   = valid[:, 1:] * valid[:, :-1]
    lx   = (gx_r - gx_t).abs() * mx

    # Vertical differences (between rows r and r+1).
    gy_r = d_r[1:, :] - d_r[:-1, :]
    gy_t = d_t[1:, :] - d_t[:-1, :]
    my   = valid[1:, :] * valid[:-1, :]
    ly   = (gy_r - gy_t).abs() * my

    n = (mx.sum() + my.sum()).clamp_min(1.0)
    return (lx.sum() + ly.sum()) / n


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


def quat_smoothness(q: torch.Tensor) -> torch.Tensor:
    """Sum-squared first-difference of unit quaternions with sign alignment.

    Same semantics as ``rotation_smoothness`` but takes quats directly
    (T, ..., 4) — useful when the quat has been composed in quat space
    (e.g. world-frame pose = q_bg⁻¹ ⊗ q_cam) and converting to axis-angle
    would hit the wrap singularity at θ=π.

    Sign alignment uses the standard q ≡ −q double-cover: flip the sign
    of any consecutive quat whose dot product with the previous is < 0.
    """
    if q.shape[0] < 2:
        return q.new_zeros(())
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dot = (q[1:] * q[:-1]).sum(dim=-1, keepdim=True)
    q_aligned = torch.where(dot < 0, -q[1:], q[1:])
    d = q_aligned - q[:-1]
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
    use_l2: bool = False,
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
    d = rendered_rgb - target_rgb
    if use_l2:
        diff = (d * d).sum(dim=-1)                              # (H, W)
    else:
        diff = d.abs().sum(dim=-1)                              # (H, W)
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


def pose_init_prior_loss(
    axis_angle: torch.Tensor,             # (T, 3) current
    translation: torch.Tensor,            # (T, 3) current
    axis_angle_init: torch.Tensor,        # (T, 3) frozen reference (FP/HaMeR)
    translation_init: torch.Tensor,       # (T, 3) frozen reference
    per_frame_weights: torch.Tensor,      # (T,) e.g. confidence
) -> torch.Tensor:
    """Per-frame weighted squared distance from an initial pose.

    Rotation distance is measured in quaternion space with double-cover
    sign alignment so axis-angle representation wraps don't appear as
    huge phantom drift. Translation uses plain squared L2.

    Used to anchor high-confidence frames near their input poses while
    leaving low-confidence frames (low ``per_frame_weights``) free to
    move under photometric pull.
    """
    # axis-angle → unit quaternion.
    a = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.cat([
        torch.cos(a * 0.5),
        axis_angle / a * torch.sin(a * 0.5),
    ], dim=-1)
    a0 = axis_angle_init.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q0 = torch.cat([
        torch.cos(a0 * 0.5),
        axis_angle_init / a0 * torch.sin(a0 * 0.5),
    ], dim=-1)
    dot = (q * q0).sum(dim=-1, keepdim=True)
    q0_aligned = torch.where(dot < 0, -q0, q0)
    rot_diff_sq = ((q - q0_aligned) ** 2).sum(dim=-1)               # (T,)

    trans_diff_sq = ((translation - translation_init) ** 2).sum(dim=-1)
    return (per_frame_weights * (rot_diff_sq + trans_diff_sq)).sum()


def beta_prior_loss(beta: torch.Tensor, beta_init: torch.Tensor) -> torch.Tensor:
    """Sum-squared deviation of MANO shape from its initialization.

    ``beta`` and ``beta_init`` are (10,). Sum (not mean) so each individual
    component is penalized at full weight — keeps the prior tight regardless
    of the 10-dim normalization.
    """
    d = beta - beta_init
    return (d * d).sum()


def intrinsics_prior_loss(
    fx: torch.Tensor, fy: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
    fx_init: torch.Tensor, fy_init: torch.Tensor,
    cx_init: torch.Tensor, cy_init: torch.Tensor,
) -> torch.Tensor:
    """Sum-squared deviation of intrinsics from their calibrated init.

    Anchors fx/fy/cx/cy to the input JSON values. Without this, fx is
    degenerate with global scene scale on no-depth runs and will wander.
    Tight default weight (~1e3) keeps K close to init while still letting
    it absorb small calibration drift.
    """
    return ((fx - fx_init) ** 2
          + (fy - fy_init) ** 2
          + (cx - cx_init) ** 2
          + (cy - cy_init) ** 2)


def hand_scale_prior_loss(
    scale: torch.Tensor, scale_init: torch.Tensor,
) -> torch.Tensor:
    """Squared deviation of per-track hand_scale from its initialization.

    Both are scalar tensors. The init comes from align_hands' n_pixels-
    weighted median (a strong estimator); the prior keeps the learned value
    from drifting if photometric signal is weak (e.g. small hand silhouettes).
    """
    d = scale - scale_init
    return d * d


# ---------------------------------------------------------------------------
# SuGaR-style surface-alignment regularizers.
#
# Pure 3D losses on the Gaussian field — no depth-image supervision is used.
# Two pieces, following Guédon & Lepetit 2023:
#   * scale anisotropy:    push each Gaussian toward a flat disk (one scale
#                          much smaller than the others).
#   * density regularizer: at sampled probes drawn along each Gaussian's
#                          normal at ±s_min, compare the local mixture
#                          density to the ideal thin-shell density exp(-½).
#
# Notes on scope: SuGaR was originally formulated for an unconstrained set
# of free Gaussians. In our pipeline hand + object Gaussians are anchored
# to mesh / MANO vertices and don't strictly need these losses (the anchor
# spring already pins them to a surface). The bg Gaussians are anchored
# to a noisy depth-unprojected point cloud and are the primary target for
# SuGaR regularization.
# ---------------------------------------------------------------------------


def scale_anisotropy_loss(scales: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian flatness penalty — rewards *disk* shape only.

    ``scales`` is ``(N, 3)`` (already exponentiated, not log). Sorted ascending:
    ``s_min ≤ s_med ≤ s_max``. Loss is::

        L = mean(s_min / s_med) + mean(1 − s_med / s_max)

    The first term encourages a single very thin axis (``s_min`` ≪ the
    others); the second pins the remaining two axes to be similar. Both are
    near zero only when ``(s_min, s_med, s_max) ≈ (ε, S, S)`` — a flat disk.
    Needle / spike shapes ``(ε, ε, S)`` keep the first term at ~1 and are
    properly penalized (which the older ``min/max``-only form did not do).
    """
    if scales.shape[0] == 0:
        return scales.sum() * 0.0
    s_sorted, _ = torch.sort(scales, dim=-1)                       # ascending
    s_min = s_sorted[..., 0]
    s_med = s_sorted[..., 1]
    s_max = s_sorted[..., 2]
    return ((s_min / s_med.clamp_min(1e-8)).mean()
            + (1.0 - s_med / s_max.clamp_min(1e-8)).mean())


def sugar_sdf_losses(
    means: torch.Tensor,         # (N, 3) Gaussian centers, world frame
    quats: torch.Tensor,         # (N, 4) (w, x, y, z) canonical rotation
    scales: torch.Tensor,        # (N, 3) per-axis sigmas (post-exp)
    opacities: torch.Tensor,     # (N,)
    depth: torch.Tensor,         # (H, W) MoGe depth in camera frame
    K: torch.Tensor,             # (3, 3) intrinsics
    R_w2c: torch.Tensor,         # (3, 3) world→camera rotation for this frame
    t_w2c: torch.Tensor,         # (3,)   world→camera translation
    union_mask: torch.Tensor,    # (H, W) {0,1} valid pixels (e.g. bg)
    n_samples: int = 1000,
    n_neighbors: int = 8,
    compute_normal: bool = True,
    samples_on_surface_weight: float = 0.2,
    sampling_scale_factor: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SuGaR-paper density + normal regularization (Eqs. 5-10, Guédon &
    Lepetit 2024).

    Operates on a single camera frame at a time so the caller can pick which
    frame's depth to anchor against per step. World frame = whatever frame
    the Gaussians live in (gsplat's "world", typically the ref-frame cam).

    Returns ``(sdf_loss, normal_loss)``. Both are scalars; ``normal_loss``
    is a zero tensor (still differentiable w.r.t. ``means``) when
    ``compute_normal=False``.

    ``sdf_loss`` combines two terms (matching SuGaR's coarse-density trainer):
      * SDF-match: ``|f̂(p) − f(p)|`` — depth-implied SDF should match the
        Gaussian-mixture SDF. This is the Eq. 9 term.
      * Samples-on-surface: ``|f̂(p)| / σ`` (weighted by
        ``samples_on_surface_weight``, default 0.2 from the paper). This
        directly pulls probe points — and therefore their anchor
        Gaussians — toward the depth surface. Without it the SDF-match
        term can be satisfied by Gaussians that drift away from the
        surface together with a matching density profile.

    Theory in two lines:
      * ``f̂(p)`` = signed distance from probe ``p`` to the depth-implied
        surface along the camera's z axis, i.e. ``z_depth(u,v) − p_cam.z``.
      * ``f(p)`` = ``sign(f̂) · sg* · √(−2 log d(p))`` where ``d`` is the
        K-NN Gaussian-mixture density at ``p`` and ``sg*`` is the smallest
        scale of the closest Gaussian. (Eq. 7-8.)
      * Sample probes from each anchor Gaussian's distribution ``N(μ, Σ)``
        with spread ``sampling_scale_factor·s`` (Eq. 9; SuGaR uses 1.5 by
        default). Density losses then encourage ``f`` and ``f̂`` to agree.
      * Normal-consistency (Eq. 10): the *direction* of ``∇d`` (which is
        parallel to ``∇f`` up to a scalar) should match the closest
        Gaussian's smallest-scale axis ``ng*``. Computed analytically.
    """
    from .gaussians import quat_to_rotmat

    N = means.shape[0]
    if N == 0 or n_samples <= 0:
        zero = means.sum() * 0.0
        return zero, zero

    # ---- 1. Pick anchors and sample probes from N(μ, Σ) ----
    n_probes = min(int(n_samples), N)
    probe_idx = torch.randperm(N, device=means.device)[:n_probes]
    R_anchors = quat_to_rotmat(quats[probe_idx])    # (P, 3, 3)
    s_anchors = scales[probe_idx].clamp_min(1e-6)   # (P, 3)
    z = torch.randn(n_probes, 3, device=means.device)
    # p = μ + R · diag(sampling_scale_factor · s) · z. SuGaR uses 1.5·s
    # (sdf_sampling_scale_factor) — wider than the raw Gaussian so probes
    # cover the off-surface region where the SDF has informative gradient.
    local_offset = sampling_scale_factor * s_anchors * z
    p = means[probe_idx] + torch.einsum("pij,pj->pi", R_anchors, local_offset)

    # ---- 2. f̂(p): depth-implied SDF along camera z ----
    p_cam = p @ R_w2c.T + t_w2c[None, :]                          # (P, 3)
    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    z_c = p_cam[:, 2].clamp_min(1e-3)
    u = fx * p_cam[:, 0] / z_c + cx
    v = fy * p_cam[:, 1] / z_c + cy

    H, W = int(depth.shape[-2]), int(depth.shape[-1])
    iu = u.round().long().clamp(0, W - 1)
    iv = v.round().long().clamp(0, H - 1)

    z_depth = depth[iv, iu]                                       # (P,)
    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    valid = (
        in_frame
        & (union_mask[iv, iu] > 0.5)
        & torch.isfinite(z_depth)
        & (z_depth > 0)
        & (p_cam[:, 2] > 0)
    )
    f_hat = z_depth - p_cam[:, 2]                                 # (P,)

    # ---- 3. f(p): SDF from Gaussian-mixture density ----
    k = int(min(n_neighbors, N))
    # Chunked KNN: (P, N) cdist OOMs at P,N ~ 50k+. Same chunking pattern
    # as density_regularizer_local — cap each chunk at ~1e8 float entries.
    chunk = max(1, int(1e8 // max(N, 1)))
    knn_idx_chunks: list[torch.Tensor] = []
    for s in range(0, n_probes, chunk):
        e = min(s + chunk, n_probes)
        d_chunk = torch.cdist(p[s:e], means)
        _, ki = d_chunk.topk(k, dim=-1, largest=False)
        knn_idx_chunks.append(ki)
    knn_idx = torch.cat(knn_idx_chunks, dim=0)                    # (P, k)

    nbr_means     = means    [knn_idx]                            # (P, k, 3)
    nbr_quats     = quats    [knn_idx]                            # (P, k, 4)
    nbr_scales    = scales   [knn_idx].clamp_min(1e-6)            # (P, k, 3)
    nbr_opacities = opacities[knn_idx]                            # (P, k)
    nbr_R = quat_to_rotmat(nbr_quats.reshape(-1, 4)).reshape(n_probes, k, 3, 3)

    delta       = p[:, None, :] - nbr_means                       # (P, k, 3)
    delta_local = torch.einsum("pkji,pkj->pki", nbr_R, delta)     # R^T · delta
    mahal_sq    = (delta_local / nbr_scales).pow(2).sum(dim=-1)   # (P, k)
    contrib     = nbr_opacities * torch.exp(-0.5 * mahal_sq)      # (P, k)
    d           = contrib.sum(dim=-1).clamp(1e-8, 1.0 - 1e-8)     # (P,)

    closest = knn_idx[:, 0]                                       # (P,)
    s_closest_min = scales[closest].min(dim=-1).values            # (P,)
    f_mag = s_closest_min * torch.sqrt(-2.0 * torch.log(d))       # (P,)
    f = torch.sign(f_hat.detach()) * f_mag                        # (P,)

    valid_f = valid.to(f.dtype)
    n_valid = valid.sum().clamp_min(1).to(f.dtype)
    sdf_match = ((f_hat - f).abs() * valid_f).sum() / n_valid

    # Samples-on-surface (SuGaR coarse-density Eq. ~9 + samples_on_surface
    # term in their trainer): penalize the signed depth distance directly,
    # normalized by its (detached) scale so the loss is scale-invariant.
    # This is the term that actually pulls Gaussian centers onto the
    # depth-implied surface — without it the SDF-match loss can be
    # satisfied by Gaussians sitting off-surface with a matching density
    # profile around them.
    f_hat_valid = f_hat[valid]
    if f_hat_valid.numel() > 1:
        sigma = f_hat_valid.detach().abs().mean().clamp_min(1e-6)
    else:
        sigma = torch.ones((), device=f_hat.device, dtype=f_hat.dtype)
    samples_on_surface = (f_hat.abs() * valid_f).sum() / n_valid / sigma
    sdf_loss = sdf_match + samples_on_surface_weight * samples_on_surface

    # ---- 4. Normal consistency (Eq. 10) ----
    # ∇d analytically: ∂d/∂p = -Σ_k contrib_k · Σ_k⁻¹ · (p − μ_k)
    #               = -Σ_k contrib_k · R_k · diag(1/s²) · R_k^T · (p − μ_k)
    #               = -Σ_k contrib_k · R_k · (delta_local_k / s_k²)
    # ∇f is parallel to ∇d up to a positive scalar; use ∇d direction.
    if not compute_normal:
        zero = means.sum() * 0.0
        return sdf_loss, zero

    local_inv_s2 = delta_local / nbr_scales.pow(2)                # (P, k, 3)
    grad_world_per_k = torch.einsum("pkij,pkj->pki", nbr_R, local_inv_s2)
    grad_d = -(contrib[..., None] * grad_world_per_k).sum(dim=1)  # (P, 3)
    grad_norm = grad_d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    grad_unit = grad_d / grad_norm                                # (P, 3)

    # Closest Gaussian's smallest-scale axis (normal direction).
    R_closest = quat_to_rotmat(quats[closest])                    # (P, 3, 3)
    s_min_axis = scales[closest].argmin(dim=-1)                   # (P,)
    n_closest = torch.gather(
        R_closest, dim=2,
        index=s_min_axis[:, None, None].expand(-1, 3, 1),
    ).squeeze(-1)                                                 # (P, 3)
    n_closest = n_closest / n_closest.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Sign-invariant alignment: 1 - <grad_unit, n>² is zero when parallel
    # OR anti-parallel. Avoids the canonical-sign ambiguity of normals.
    dot = (grad_unit * n_closest).sum(dim=-1)                     # (P,)
    normal_loss = (1.0 - dot.pow(2)).mean()

    return sdf_loss, normal_loss


def density_regularizer_local(
    means: torch.Tensor,         # (N, 3) Gaussian centers (world frame)
    quats: torch.Tensor,         # (N, 4) (w, x, y, z) world-frame canonical rotation
    scales: torch.Tensor,        # (N, 3) per-axis sigmas (post-exp)
    opacities: torch.Tensor,     # (N,)
    n_neighbors: int = 8,
    subsample_frac: float = 0.2,
) -> torch.Tensor:
    """SuGaR-style density loss on the Gaussian field.

    Per-step subsample: pick ``ceil(subsample_frac * N)`` Gaussians at random
    as probe anchors. For each anchor:
      * find its smallest-scale axis (the disk normal) by reading the column
        of its rotation matrix that corresponds to the min(scales) index;
      * sample 2 probe points at the center ± ``s_min`` along that normal;
      * compute the K-nearest-neighbor mixture density at each probe,
        using anisotropic Mahalanobis distance ``Σᵢ (R^T (p−μ))ᵢ² / sᵢ²``;
      * compare to the ideal thin-shell density ``α · exp(-½)``.

    Returns ``mean |f_actual − f_ideal|``. Loss is bounded; magnitude depends
    on opacity scale (typically O(0.1)–O(1)). Wire as ``w_density_bg``.

    No supervision from depth images. The Gaussian field is its own teacher.
    """
    from .gaussians import quat_to_rotmat

    N = means.shape[0]
    if N == 0 or subsample_frac <= 0:
        return means.sum() * 0.0

    n_probes = max(1, int(N * subsample_frac))
    probe_idx = torch.randperm(N, device=means.device)[:n_probes]

    p_centers = means    [probe_idx]                   # (P, 3)
    s_probe   = scales   [probe_idx]                   # (P, 3)
    q_probe   = quats    [probe_idx]                   # (P, 4)
    a_probe   = opacities[probe_idx]                   # (P,)

    s_min, s_min_idx = s_probe.min(dim=-1)             # (P,), (P,)
    R = quat_to_rotmat(q_probe)                         # (P, 3, 3)
    # The k-th column of R is the world direction of the local k-th axis.
    normal = torch.gather(
        R, dim=2,
        index=s_min_idx[:, None, None].expand(-1, 3, 1),
    ).squeeze(-1)                                      # (P, 3)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    probe_pos = torch.cat([
        p_centers + normal * s_min[:, None],
        p_centers - normal * s_min[:, None],
    ], dim=0)                                          # (2P, 3)

    # Cap the number of neighbors at N (small scenes).
    k = int(min(n_neighbors, N))

    # KNN: compute pairwise distances probes vs all means, take top-k.
    # Memory: (2P) * N floats. Chunked when 2P*N exceeds ~1e8 entries.
    chunk = max(1, int(1e8 // max(N, 1)))
    knn_idx_chunks: list[torch.Tensor] = []
    for s in range(0, probe_pos.shape[0], chunk):
        e = min(s + chunk, probe_pos.shape[0])
        d = torch.cdist(probe_pos[s:e], means)
        _, ki = d.topk(k, dim=-1, largest=False)
        knn_idx_chunks.append(ki)
    knn_idx = torch.cat(knn_idx_chunks, dim=0)         # (2P, k)

    nbr_means     = means     [knn_idx]                # (2P, k, 3)
    nbr_quats     = quats     [knn_idx]                # (2P, k, 4)
    nbr_scales    = scales    [knn_idx]                # (2P, k, 3)
    nbr_opacities = opacities [knn_idx]                # (2P, k)

    nbr_R = quat_to_rotmat(nbr_quats.reshape(-1, 4)).reshape(
        probe_pos.shape[0], k, 3, 3,
    )                                                  # (2P, k, 3, 3)
    delta = probe_pos[:, None, :] - nbr_means          # (2P, k, 3)
    # Local-frame delta: R^T @ (p − μ).
    delta_local = torch.einsum("pkji,pkj->pki", nbr_R, delta)
    mahal_sq = (delta_local / nbr_scales.clamp_min(1e-8)).pow(2).sum(dim=-1)
    contrib = nbr_opacities * torch.exp(-0.5 * mahal_sq)
    f_actual = contrib.sum(dim=-1)                     # (2P,)

    # Ideal thin-shell density at the ±s_min probes: opacity · exp(-½).
    f_ideal = a_probe.repeat(2) * float(torch.exp(torch.tensor(-0.5)))
    return (f_actual - f_ideal).abs().mean()
