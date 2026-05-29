# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Background Gaussian set + per-frame rigid background→camera pose.

Models the static scene behind the foreground hand+object so the
photometric loss can be applied to the *full* image rather than just the
mask region. Initialized from a reference frame's MoGe depth: every pixel
outside the foreground union mask and with finite depth becomes one
Gaussian in 3D, anchored in the world frame defined by that reference
frame's camera frame.

Background pose is parameterized as world→camera (axis_angle, translation)
per frame, mirroring ``ObjectPoseField``. The reference frame's pose is
identity by construction; all other frames are also initialized identity
and the optimizer discovers camera motion via photometric drift.

Caveat: identity-init is only a good basin for sequences with small
camera motion. For freely-moving (egocentric) cameras you'd want to seed
from an external VO/SfM solution before this kicks in.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gaussians import (
    GaussianFrame,
    _logit,
    axis_angle_to_quat,
    quat_mul,
    quat_to_rotmat,
    rotmat_to_quat,
)


# ---------------------------------------------------------------------------
# Background Gaussians: same parameter set as ObjectGaussians, but anchored
# to a depth-unprojected world-frame point cloud rather than a mesh.
# ---------------------------------------------------------------------------

class BackgroundGaussians(nn.Module):
    def __init__(
        self,
        anchor_positions: torch.Tensor,   # (N, 3) in world (= ref-frame camera) frame
        init_color: torch.Tensor,         # (N, 3) RGB in [0, 1]
        init_scale: torch.Tensor,         # (N,)   per-point initial Gaussian size
        init_opacity: float = 0.9,
    ) -> None:
        super().__init__()
        N = anchor_positions.shape[0]
        device = anchor_positions.device

        self.register_buffer("anchor", anchor_positions.contiguous())
        self._delta_p       = nn.Parameter(torch.zeros(N, 3, device=device))
        self._quat_canon    = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)
        )
        # Per-point init scale (depth-aware: farther points get larger Gaussians
        # so 1-pixel image coverage holds at any depth).
        log_scale = init_scale.clamp_min(1e-6).log().unsqueeze(-1).expand(-1, 3).contiguous()
        self._log_scale     = nn.Parameter(log_scale.to(device).clone())
        self._opacity_logit = nn.Parameter(
            torch.full((N,), float(_logit(init_opacity)), device=device)
        )
        self._color = nn.Parameter(init_color.contiguous().clone())

    def num_gaussians(self) -> int:
        return self.anchor.shape[0]

    def forward(
        self,
        R_bg: torch.Tensor,    # (3, 3) world→camera rotation
        t_bg: torch.Tensor,    # (3,)   world→camera translation
    ) -> GaussianFrame:
        p_world = self.anchor + self._delta_p
        means = p_world @ R_bg.T + t_bg
        q_bg = rotmat_to_quat(R_bg).expand_as(self._quat_canon)
        quats = quat_mul(q_bg, self._quat_canon)
        return GaussianFrame(
            means     = means,
            quats     = quats,
            scales    = self._log_scale.exp(),
            opacities = torch.sigmoid(self._opacity_logit),
            colors    = self._color,
        )


# ---------------------------------------------------------------------------
# Background pose field: per-frame rigid world→camera transform.
# ---------------------------------------------------------------------------

class BackgroundPoseField(nn.Module):
    def __init__(self, n_frames: int, device, ref_frame_t: int = 0) -> None:
        super().__init__()
        # Identity init for every frame. The reference frame's pose is
        # *exactly* identity by construction; other frames start there and
        # the optimizer picks up camera motion via photometric drift.
        self.axis_angle  = nn.Parameter(torch.zeros(n_frames, 3, device=device))
        self.translation = nn.Parameter(torch.zeros(n_frames, 3, device=device))
        self.ref_frame_t = int(ref_frame_t)

    def num_frames(self) -> int:
        return self.axis_angle.shape[0]

    def forward(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        q = axis_angle_to_quat(self.axis_angle[t])
        R = quat_to_rotmat(q)
        return R, self.translation[t]

    def batched_forward(
        self, t_idx: torch.Tensor       # (B,) long tensor of positional indices
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aa = self.axis_angle[t_idx]                        # (B, 3)
        q  = axis_angle_to_quat(aa)                         # (B, 4)
        R  = quat_to_rotmat(q)                              # (B, 3, 3)
        return R, self.translation[t_idx]                   # (B, 3, 3), (B, 3)


# ---------------------------------------------------------------------------
# Initialization from a reference frame's depth + RGB.
# ---------------------------------------------------------------------------

def _unproject_frame_to_world(
    rgb: torch.Tensor,            # (H, W, 3) in [0, 1]
    depth: torch.Tensor,          # (H, W) metric depth, possibly +inf for invalid
    union_mask: torch.Tensor,     # (H, W) in {0, 1} — foreground (object + hands)
    K: torch.Tensor,              # (3, 3) intrinsics
    T_w2c: torch.Tensor | None = None,   # (4, 4); identity == "world is this cam"
    max_points: int | None = None,
    scale_factor: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unproject background pixels of a single frame into the gsplat world.

    Pixels outside ``union_mask`` with finite positive depth are unprojected
    through K to camera-frame 3D points, then transformed to world via
    ``T_w2c^{-1}`` (== ``T_c2w``). Returns ``(anchors_world, colors,
    init_scales)``. ``init_scales = z_in_camera * scale_factor / fx`` so each
    Gaussian covers ~one image pixel projection at the depth where it lives.

    Identity ``T_w2c`` means "world == this frame's camera frame" and the
    output is in that frame.
    """
    H, W, _ = rgb.shape
    device = rgb.device
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    valid = (union_mask < 0.5) & torch.isfinite(depth) & (depth > 0.0)
    valid_idx = torch.nonzero(valid, as_tuple=False)            # (M, 2) [v, u]
    if valid_idx.shape[0] == 0:
        return (torch.zeros(0, 3, device=device),
                torch.zeros(0, 3, device=device),
                torch.zeros(0,    device=device))

    if max_points is not None and valid_idx.shape[0] > max_points:
        perm = torch.randperm(valid_idx.shape[0], device=device)[:max_points]
        valid_idx = valid_idx[perm]

    v = valid_idx[:, 0].to(torch.float32)
    u = valid_idx[:, 1].to(torch.float32)
    z = depth[valid_idx[:, 0], valid_idx[:, 1]]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts_cam = torch.stack([x, y, z], dim=-1)                    # (N, 3)
    if T_w2c is not None:
        T_c2w = torch.linalg.inv(T_w2c)
        pts_h = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=-1)
        pts_world = (pts_h @ T_c2w.T)[:, :3]
    else:
        pts_world = pts_cam

    colors  = rgb[valid_idx[:, 0], valid_idx[:, 1]]             # (N, 3)
    init_scales = (z / fx) * scale_factor                       # (N,)
    return pts_world, colors, init_scales


def _voxel_dedup(
    points: torch.Tensor,        # (N, 3) world coords
    colors: torch.Tensor,        # (N, 3)
    scales: torch.Tensor,        # (N,)
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One point per occupied voxel. Keeps the first point that lands in each
    voxel. Cheap and stable; quality matches voxel size, not the per-voxel
    aggregation method.
    """
    if points.shape[0] == 0 or voxel_size <= 0:
        return points, colors, scales
    keys = torch.floor(points / voxel_size).to(torch.int64)
    keys_np = keys.detach().cpu().numpy()
    import numpy as np
    _, first = np.unique(keys_np, axis=0, return_index=True)
    first_t = torch.from_numpy(first).to(points.device)
    return points[first_t], colors[first_t], scales[first_t]


def init_background_multiframe(
    rgbs:        list[torch.Tensor],   # each (H, W, 3) in [0, 1]
    depths:      list[torch.Tensor],   # each (H, W) metric depth (+inf allowed)
    union_masks: list[torch.Tensor],   # each (H, W) in {0, 1}
    T_w2c_list:  list[torch.Tensor],   # each (4, 4) world→camera; gsplat world
    K:           torch.Tensor,         # (3, 3) shared intrinsics
    voxel_size:  float = 0.005,
    max_points:  int   = 50000,
    per_frame_max_points: int = 200000,
    scale_factor: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse per-frame depth unprojections into a single gsplat-world cloud.

    Each frame contributes its background pixels (outside its own foreground
    mask) lifted to world via the supplied world→camera pose. Voxel dedup
    removes near-duplicate points where multiple frames see the same surface;
    a final random subsample caps the total Gaussian count.
    """
    if not rgbs:
        raise RuntimeError("init_background_multiframe: no frames provided.")
    parts_p: list[torch.Tensor] = []
    parts_c: list[torch.Tensor] = []
    parts_s: list[torch.Tensor] = []
    for rgb, depth, mask, T_w2c in zip(rgbs, depths, union_masks, T_w2c_list):
        p, c, s = _unproject_frame_to_world(
            rgb=rgb, depth=depth, union_mask=mask, K=K,
            T_w2c=T_w2c, max_points=per_frame_max_points,
            scale_factor=scale_factor,
        )
        if p.shape[0] > 0:
            parts_p.append(p); parts_c.append(c); parts_s.append(s)
    if not parts_p:
        raise RuntimeError(
            "init_background_multiframe: every frame's background was empty "
            "(foreground masks cover everything or depth is all invalid)."
        )
    points = torch.cat(parts_p, dim=0)
    colors = torch.cat(parts_c, dim=0)
    scales = torch.cat(parts_s, dim=0)

    points, colors, scales = _voxel_dedup(points, colors, scales, voxel_size)

    if points.shape[0] > max_points:
        perm = torch.randperm(points.shape[0], device=points.device)[:max_points]
        points = points[perm]; colors = colors[perm]; scales = scales[perm]

    return points, colors, scales


def init_background_from_depth(
    rgb: torch.Tensor,            # (H, W, 3) in [0, 1]
    depth: torch.Tensor,          # (H, W) metric depth, possibly +inf for invalid
    union_mask: torch.Tensor,     # (H, W) in {0, 1} — foreground (object + hands)
    K: torch.Tensor,              # (3, 3) intrinsics
    max_points: int = 50000,
    scale_factor: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unproject every background pixel to a 3D point in the camera frame.

    Returns:
        anchors:     (N, 3) point cloud in camera-frame == world-frame
        colors:      (N, 3) RGB in [0, 1]
        init_scales: (N,)   per-point Gaussian scale ≈ z * pixel_pitch * factor

    Filters out foreground pixels and pixels with invalid depth, then
    optionally subsamples to ``max_points`` for tractable rasterization
    (a 1080p frame can yield ~2M background pixels — way more than gsplat
    needs to cover the scene at typical Gaussian sizes).
    """
    H, W, _ = rgb.shape
    device = rgb.device
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    valid = (union_mask < 0.5) & torch.isfinite(depth) & (depth > 0.0)
    valid_idx = torch.nonzero(valid, as_tuple=False)            # (M, 2) [v, u]
    if valid_idx.shape[0] == 0:
        raise RuntimeError(
            "Background init: no valid pixels in reference frame "
            "(every pixel is either foreground or has invalid MoGe depth)."
        )

    if valid_idx.shape[0] > max_points:
        perm = torch.randperm(valid_idx.shape[0], device=device)[:max_points]
        valid_idx = valid_idx[perm]

    v = valid_idx[:, 0].to(torch.float32)
    u = valid_idx[:, 1].to(torch.float32)
    z = depth[valid_idx[:, 0], valid_idx[:, 1]]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    anchors = torch.stack([x, y, z], dim=-1)                    # (N, 3)
    colors  = rgb[valid_idx[:, 0], valid_idx[:, 1]]             # (N, 3)
    # Per-point init scale: 1-pixel projected size at depth z is ~ z / fx.
    # Multiply by scale_factor (~1-2) so neighboring Gaussians overlap
    # enough to fill the surface without gaps.
    init_scales = (z / fx) * scale_factor                       # (N,)

    return anchors, colors, init_scales
