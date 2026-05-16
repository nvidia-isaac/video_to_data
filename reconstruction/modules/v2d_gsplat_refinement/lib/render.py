"""gsplat render wrapper for the refinement loop.

Renders Gaussian sets into the camera frame. We build everything already in
camera coords (object pose / MANO cam_t are applied by the pose fields), so
the view matrix passed to gsplat is identity and only the intrinsics matter.

Three convenience entry points:

  render_rgb_depth(frame, K, H, W) -> (RGB, depth, alpha)
      Single combined render of one ``GaussianFrame`` (typically the
      concatenation of all sets) for the photometric + depth loss.

  render_alpha(frame, K, H, W)     -> alpha
      Cheaper render returning only the alpha channel; used per-set for the
      silhouette loss.
"""
from __future__ import annotations

import torch

from .gaussians import GaussianFrame


# Lazy import to keep import-time light when only running tests on CPU
# without gsplat installed (e.g. unit tests of io.py).
def _rasterization():
    from gsplat.rendering import rasterization
    return rasterization


_BG_BLACK = (0.0, 0.0, 0.0)


def _identity_viewmat(device: torch.device) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)   # (1, 4, 4)


def render_rgb_depth(
    frame: GaussianFrame,
    K: torch.Tensor,        # (3, 3)
    width: int,
    height: int,
    extra_features: torch.Tensor | None = None,    # (N, F) extra per-Gaussian channels
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    compute_depth_variance: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Composite render of a Gaussian set in camera frame.

    If ``extra_features`` is provided, those per-Gaussian channels are
    composited alongside RGB in a single pass (gsplat rasterizes any
    feature dimension), and the corresponding (H, W, F) feature map is
    returned. This is how the silhouette loss avoids needing N+1 separate
    render passes: pack one-hot class labels into ``extra_features`` and
    the same alpha compositing that produces RGB also gives a per-pixel
    "front-most class" map that handles occlusion correctly.

    When ``compute_depth_variance=True``, an additional channel carrying
    per-Gaussian ``d²`` (cam-frame z squared) is composited and the
    function returns a per-pixel depth-variance map ``Var[d] = E[d²] − E[d]²``.
    This is the depth-variance approximation to Mip-NeRF 360's distortion
    loss (gsplat's mainline 3DGS rasterizer doesn't expose distloss; only
    the 2DGS path does). The variance is alpha-normalized to match
    ``depth = E[d]``, so it's a true variance with units of [depth²].

    Returns:
        rgb:        (H, W, 3)
        depth:      (H, W)              -- expected depth E[d] (alpha-weighted, normalized)
        alpha:      (H, W)              -- accumulated opacity
        features:   (H, W, F) or None   -- composited extra channels
        depth_var:  (H, W) or None      -- E[d²] − E[d]², ≥ 0 (None when not requested)
    """
    rasterization = _rasterization()
    device = frame.means.device
    viewmat = _identity_viewmat(device)
    Ks = K.unsqueeze(0)

    # Pack channels: [user-extra | d² (optional)].
    extra_parts: list[torch.Tensor] = []
    if extra_features is not None:
        extra_parts.append(extra_features)
    F_user = extra_features.shape[-1] if extra_features is not None else 0
    if compute_depth_variance:
        # Means live in cam frame here (viewmat is identity at this layer),
        # so cam-z = means[:, 2]. Detach the centers? No — we *want* the
        # gradient to push means toward smaller per-pixel variance.
        d2 = (frame.means[:, 2] ** 2).unsqueeze(-1)                  # (N, 1)
        extra_parts.append(d2)

    if extra_parts:
        extra_all = torch.cat(extra_parts, dim=-1)                   # (N, F_user + (1?))
        colors = torch.cat([frame.colors, extra_all], dim=-1)        # (N, 3 + F)
        F_extra = extra_all.shape[-1]
    else:
        colors = frame.colors
        F_extra = 0

    bg = torch.zeros(3 + F_extra, device=device, dtype=torch.float32)

    out, alphas, _ = rasterization(
        means      = frame.means,
        quats      = frame.quats,
        scales     = frame.scales,
        opacities  = frame.opacities,
        colors     = colors,
        viewmats   = viewmat,
        Ks         = Ks,
        width      = width,
        height     = height,
        near_plane = near_plane,
        far_plane  = far_plane,
        render_mode= "RGB+ED",       # ED = expected depth (alpha-weighted, normalized)
        backgrounds= bg.unsqueeze(0),
    )
    # out: (1, H, W, 3 + F_extra + 1) — last channel is depth.
    rgb     = out[0, ..., :3]
    depth   = out[0, ..., -1]                                        # E[d], alpha-normalized
    alpha   = alphas[0, ..., 0]

    feats: torch.Tensor | None = None
    if F_user > 0:
        feats = out[0, ..., 3:3 + F_user]

    depth_var: torch.Tensor | None = None
    if compute_depth_variance:
        # Composited d² is NOT alpha-normalized by gsplat (extra features go
        # through the standard sum_i T_i α_i x_i composite). Divide by alpha
        # to get true E[d²], then variance = E[d²] − E[d]². Clamp alpha to
        # avoid division by zero on empty pixels; clamp variance to ≥ 0 for
        # float-precision safety (Jensen's inequality guarantees ≥ 0 in
        # exact arithmetic).
        d2_raw = out[0, ..., 3 + F_user]                             # (H, W)
        e_d2 = d2_raw / alpha.clamp_min(1e-6)
        depth_var = (e_d2 - depth * depth).clamp_min(0.0)

    return rgb, depth, alpha, feats, depth_var


def render_alpha(
    frame: GaussianFrame,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> torch.Tensor:
    """Render only the accumulated opacity. (H, W)."""
    rasterization = _rasterization()
    device = frame.means.device
    viewmat = _identity_viewmat(device)
    Ks = K.unsqueeze(0)
    # Use a 1-channel "color" of constant 1 so RGB is just alpha; we then
    # take the alpha channel directly. Cheaper than dragging a 3-channel
    # color through.
    ones = torch.ones_like(frame.colors[..., :1])
    _, alphas, _ = rasterization(
        means      = frame.means,
        quats      = frame.quats,
        scales     = frame.scales,
        opacities  = frame.opacities,
        colors     = ones,
        viewmats   = viewmat,
        Ks         = Ks,
        width      = width,
        height     = height,
        near_plane = near_plane,
        far_plane  = far_plane,
        render_mode= "RGB",
    )
    return alphas[0, ..., 0]
