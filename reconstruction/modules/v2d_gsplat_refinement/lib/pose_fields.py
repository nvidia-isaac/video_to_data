# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-frame learnable pose state.

Two modules:

  ObjectPoseField -- (T, 6) of [axis-angle, translation] per frame; ``scale``
                     is held as a buffer (carried through to output but not
                     optimized in the simplest setup, since post-FoundationPose
                     scales are [1, 1, 1] and we add no global scale param).

  HandPoseField   -- (T, 3) global_orient axis-angle, (T, 45) hand_pose
                     axis-angle, (T, 3) cam_t. Owns a manotorch ``ManoLayer``
                     and exposes ``posed_verts_camera(t)`` so the renderer
                     can pull world-frame MANO vertices each step. ``betas``
                     is a buffer in this minimal version (per-identity but
                     not optimized).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gaussians import axis_angle_to_quat, quat_to_rotmat
from .io import HandPoseTrack, ObjectPoseTrack


# ---------------------------------------------------------------------------
# Object pose field
# ---------------------------------------------------------------------------

def _quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) (w, x, y, z) → (..., 3) axis-angle (canonical, |angle| ≤ π).

    Canonicalize first: the double-cover ``q ≡ −q`` represents the same
    rotation, but ``2·acos(w)`` only returns the short-way form when
    ``w ≥ 0``. Flip the sign of any quat with negative w so the resulting
    axis-angle magnitude stays in [0, π] rather than the "long way around"
    [π, 2π] band.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    # Canonical hemisphere: w ≥ 0.
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., 0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt((1.0 - w * w).clamp_min(1e-12))
    axis = q[..., 1:] / s.unsqueeze(-1).clamp_min(1e-12)
    # When angle is ~0, axis is undefined; safe to keep aa = 0 in that case.
    aa = axis * angle.unsqueeze(-1)
    aa = torch.where(angle.unsqueeze(-1) < 1e-6,
                     torch.zeros_like(aa), aa)
    return aa


class ObjectPoseField(nn.Module):
    """Per-frame object→camera pose, parameterized as axis-angle + translation."""

    def __init__(self, track: ObjectPoseTrack) -> None:
        super().__init__()
        # Convert (T, 4) quats to axis-angle so we have a minimal,
        # singularity-free parameterization for small updates.
        aa = _quat_to_axis_angle(track.rotations)
        self.axis_angle  = nn.Parameter(aa.contiguous())             # (T, 3)
        self.translation = nn.Parameter(track.translations.clone())  # (T, 3)
        self.register_buffer("scale", track.scales.clone())          # (T, 3)
        self.frame_indices = list(track.frame_indices)

    def num_frames(self) -> int:
        return self.axis_angle.shape[0]

    def forward(self, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (R (3,3), t (3,)) for frame index ``t`` (positional, not frame_idx)."""
        q = axis_angle_to_quat(self.axis_angle[t])         # (4,)
        R = quat_to_rotmat(q)                               # (3, 3)
        return R, self.translation[t]

    def batched_forward(
        self, t_idx: torch.Tensor       # (B,) long tensor of positional indices
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched version of forward — returns ((B, 3, 3), (B, 3))."""
        aa = self.axis_angle[t_idx]                        # (B, 3)
        q  = axis_angle_to_quat(aa)                         # (B, 4)
        R  = quat_to_rotmat(q)                              # (B, 3, 3)
        return R, self.translation[t_idx]

    def export_track(self) -> ObjectPoseTrack:
        """Return the optimized state as an ObjectPoseTrack (for I/O)."""
        q = axis_angle_to_quat(self.axis_angle)             # (T, 4)
        return ObjectPoseTrack(
            rotations     = q.detach(),
            translations  = self.translation.detach(),
            scales        = self.scale.detach(),
            frame_indices = list(self.frame_indices),
        )


# ---------------------------------------------------------------------------
# Hand pose field (wraps manotorch.ManoLayer)
# ---------------------------------------------------------------------------

class IntrinsicsField(nn.Module):
    """Learnable camera intrinsics (fx, fy, cx, cy).

    Stores each component as a scalar ``nn.Parameter`` and exposes a
    ``K()`` method that assembles the (3, 3) intrinsics matrix on demand
    so downstream renderers see a fresh tensor with current values.

    Per-component ``register_buffer`` copies of the init values anchor the
    L2 prior — without it, the focal length can drift along the
    focal/depth degeneracy (doubling fx is photometrically equivalent to
    halving every scene-z, when depth supervision is absent).

    Each parameter independently controls its ``requires_grad`` via the
    ``learn_focal`` / ``learn_principal_point`` constructor flags, so the
    optimizer can refine principal point alone (safer) while keeping
    focal length pinned to the calibrated input.
    """

    def __init__(
        self,
        K: torch.Tensor,                         # (3, 3) input intrinsics
        learn_focal: bool = False,
        learn_principal_point: bool = False,
    ) -> None:
        super().__init__()
        device = K.device
        # Parameters: split per-component so each is independently learnable.
        self.fx = nn.Parameter(
            K[0, 0].detach().clone().to(device, dtype=torch.float32),
            requires_grad=bool(learn_focal),
        )
        self.fy = nn.Parameter(
            K[1, 1].detach().clone().to(device, dtype=torch.float32),
            requires_grad=bool(learn_focal),
        )
        self.cx = nn.Parameter(
            K[0, 2].detach().clone().to(device, dtype=torch.float32),
            requires_grad=bool(learn_principal_point),
        )
        self.cy = nn.Parameter(
            K[1, 2].detach().clone().to(device, dtype=torch.float32),
            requires_grad=bool(learn_principal_point),
        )
        # Init buffers — fixed reference for the prior loss.
        self.register_buffer("fx_init", self.fx.detach().clone())
        self.register_buffer("fy_init", self.fy.detach().clone())
        self.register_buffer("cx_init", self.cx.detach().clone())
        self.register_buffer("cy_init", self.cy.detach().clone())

    def K(self) -> torch.Tensor:
        """Assemble the (3, 3) intrinsics tensor from the four scalars.

        Differentiable w.r.t. fx / fy / cx / cy. Constructed fresh each
        call so downstream consumers always see current values.
        """
        zero = self.fx.new_zeros(())
        one  = self.fx.new_ones(())
        return torch.stack([
            torch.stack([self.fx,  zero,    self.cx]),
            torch.stack([zero,    self.fy,  self.cy]),
            torch.stack([zero,    zero,    one]),
        ], dim=0)

    def has_learnable(self) -> bool:
        return any(p.requires_grad for p in (self.fx, self.fy, self.cx, self.cy))


class HandPoseField(nn.Module):
    """Per-frame MANO global_orient + hand_pose + cam_t for a single hand,
    plus a single shared per-identity ``betas`` (10,).

    ``betas`` is initialized from the mean of HaMeR's per-frame predictions
    and optimized with a tight Gaussian prior pulling back to that init
    (``betas_init`` is a buffer holding the prior mean). Per-frame β is
    anatomically incorrect; one shared vector keeps hand size consistent
    across the sequence while still allowing it to adapt globally.

    Left-hand mirroring follows ``align_hands.py``: negate vertex x after
    the layer call.
    """

    def __init__(
        self,
        track: HandPoseTrack,
        mano_assets_root: str,
        device: str | torch.device = "cuda",
        learn_hand_scale: bool = False,
    ) -> None:
        super().__init__()
        # Lazy import so non-hand pipelines don't pay for manotorch.
        from manotorch.manolayer import ManoLayer

        self.global_orient = nn.Parameter(track.global_orient.clone())    # (T, 3)
        self.hand_pose     = nn.Parameter(track.hand_pose.clone())         # (T, 45)
        self.cam_t         = nn.Parameter(track.cam_t.clone())             # (T, 3)

        # Shared β: mean of HaMeR's per-frame estimates → (10,). Optimized
        # softly via beta_prior_loss(self.betas, self.betas_init).
        beta_init = track.betas.mean(dim=0).clone()                        # (10,)
        self.betas = nn.Parameter(beta_init.clone())                       # (10,)
        self.register_buffer("betas_init", beta_init.clone())              # (10,)

        self.is_right      = bool(track.is_right)
        self.frame_indices = list(track.frame_indices)

        # Per-track multiplicative depth correction from align_hands. The
        # MANO mesh is rescaled around its centroid (in MANO/rest frame,
        # before adding cam_t) so the projected silhouette matches the
        # image. cam_t carries the additive (dz) shift; together they
        # correct both modes of depth mismatch.
        #
        # Stored as a Parameter so refine can optimize it when
        # ``learn_hand_scale=True``; otherwise ``requires_grad=False`` and
        # the parameter is excluded from optimizer param groups (functions
        # as a fixed buffer). ``hand_scale_init`` is kept as a buffer to
        # anchor the soft prior pulling the learned value back to the
        # align_hands estimate.
        scale_init = float(getattr(track, "hand_scale", 1.0) or 1.0)
        self.hand_scale = nn.Parameter(
            torch.tensor(scale_init, device=device, dtype=torch.float32),
            requires_grad=bool(learn_hand_scale),
        )
        self.register_buffer(
            "hand_scale_init",
            torch.tensor(scale_init, device=device, dtype=torch.float32),
        )

        # ManoLayer always produces right-hand vertices in its own frame; we
        # mirror x for left hands at output (matches v2d_hamer/align_hands.py).
        self.mano = ManoLayer(
            rot_mode         = "axisang",
            use_pca          = False,
            side             = "right",
            center_idx       = None,
            mano_assets_root = mano_assets_root,
        ).to(device)

        # Per-vertex LBS rotation requires subtracting the rest-pose joint
        # rotation from each frame's joint rotation: R_def_j = R_j @ R_j_rest^T.
        # Cache the rest-pose rotation inverses (a fixed buffer) so we can
        # compute deformation rotations on the fly.
        with torch.no_grad():
            zero_pose  = torch.zeros(1, 48, device=device)
            zero_betas = torch.zeros(1, 10, device=device)
            out_rest = self.mano(zero_pose, zero_betas)
        if hasattr(out_rest, "transforms_abs"):
            R_rest = out_rest.transforms_abs[0, :, :3, :3].detach()       # (J, 3, 3)
        else:
            # Older manotorch builds expose this as a positional output;
            # fall back to identity (rest joint orientations are identity at
            # zero pose for MANO's kinematic tree, so this is correct).
            R_rest = torch.eye(3, device=device).unsqueeze(0).repeat(16, 1, 1)
        # Inverse of a rotation is its transpose.
        self.register_buffer("_R_joints_rest_inv", R_rest.transpose(-1, -2).contiguous())

        # Skinning weights: shape (1, N, J) or (N, J) depending on manotorch
        # version. Normalize to (N, J) and stash as a non-learnable buffer.
        W = self.mano.th_weights
        if W.ndim == 3:
            W = W.squeeze(0)
        self.register_buffer("_skin_weights", W.detach().clone())

    def num_frames(self) -> int:
        return self.global_orient.shape[0]

    def num_verts(self) -> int:
        return int(self.mano.th_v_template.shape[1]) if hasattr(self.mano, "th_v_template") else 778

    def posed_verts_and_rotmats_camera(
        self, t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return MANO vertices and per-vertex LBS rotation matrices in
        camera frame for frame ``t`` (positional index, not frame_idx).

        Pipeline:
          1. ManoLayer forward → posed verts (N, 3) in MANO local frame and
             absolute joint transforms (J, 4, 4), with global_orient applied
             via the root joint.
          2. Per-joint deformation rotation: R_def_j = R_j @ R_j_rest^T.
          3. Per-vertex rotation: R_v = sum_j W[v, j] * R_def_j   (LBS).
          4. Mirror x for left hand: verts[:, 0] *= -1, and conjugate the
             rotation by M = diag(-1, 1, 1) → R' = M R M.
          5. Translate verts by cam_t (rotation unaffected).
        """
        pose_aa = torch.cat([self.global_orient[t], self.hand_pose[t]], dim=-1)  # (48,)
        out = self.mano(pose_aa.unsqueeze(0), self.betas.unsqueeze(0))
        verts = out.verts[0]                                                     # (N, 3)

        # Per-joint deformation rotation. transforms_abs is (B, J, 4, 4).
        if hasattr(out, "transforms_abs"):
            R_joints = out.transforms_abs[0, :, :3, :3]                          # (J, 3, 3)
        else:
            # Should not occur with current manotorch — fall back to identity.
            R_joints = torch.eye(3, device=verts.device).unsqueeze(0).expand(16, -1, -1)
        R_def = R_joints @ self._R_joints_rest_inv                               # (J, 3, 3)

        # LBS skinning: R_v = sum_j W[v, j] * R_def[j].
        R_per_vert = torch.einsum("vj,jik->vik", self._skin_weights, R_def)      # (N, 3, 3)

        if not self.is_right:
            mirror = verts.new_tensor([-1.0, 1.0, 1.0])
            verts = verts * mirror
            M = torch.diag(mirror)
            R_per_vert = M @ R_per_vert @ M

        # Multiplicative depth correction (hand_scale) — applied around the
        # mesh centroid in MANO frame, before cam_t. Rotations untouched
        # (uniform isotropic scale). Skipped only when scale is exactly 1
        # AND not learnable (avoids killing gradient flow when learnable
        # happens to start at 1.0).
        if self.hand_scale.requires_grad or float(self.hand_scale) != 1.0:
            c = verts.mean(dim=0, keepdim=True)                                  # (1, 3)
            verts = (verts - c) * self.hand_scale + c

        verts = verts + self.cam_t[t]                                            # (N, 3)
        return verts, R_per_vert

    def batched_posed_verts_and_rotmats_camera(
        self, t_idx: torch.Tensor       # (B,) long tensor of positional indices
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched version of ``posed_verts_and_rotmats_camera``.

        Returns:
            verts:      (B, N_verts, 3) MANO vertices in camera frame.
            R_per_vert: (B, N_verts, 3, 3) per-vertex LBS deformation
                        rotation in camera frame.

        Single MANO forward over the whole batch (replaces B per-frame
        calls); LBS skinning is one big einsum. The biggest speedup of
        this whole change comes from this method — at batch=1 ManoLayer
        is mostly kernel-launch overhead.
        """
        B = int(t_idx.shape[0])
        pose_aa_b = torch.cat(
            [self.global_orient[t_idx], self.hand_pose[t_idx]], dim=-1
        )                                                                    # (B, 48)
        betas_b = self.betas.unsqueeze(0).expand(B, -1)                      # (B, 10)
        out = self.mano(pose_aa_b, betas_b)
        verts = out.verts                                                    # (B, N, 3)

        if hasattr(out, "transforms_abs"):
            R_joints = out.transforms_abs[:, :, :3, :3]                      # (B, J, 3, 3)
        else:
            R_joints = torch.eye(3, device=verts.device).expand(B, 16, 3, 3).contiguous()
        # Per-joint deformation rotation: R_def = R_current @ R_rest^T.
        R_def = R_joints @ self._R_joints_rest_inv.unsqueeze(0)              # (B, J, 3, 3)

        # Per-vertex LBS skinning: R_v = sum_j W[v, j] * R_def[b, j].
        R_per_vert = torch.einsum(
            "vj,bjik->bvik", self._skin_weights, R_def
        )                                                                    # (B, N, 3, 3)

        if not self.is_right:
            mirror = verts.new_tensor([-1.0, 1.0, 1.0])
            verts = verts * mirror                                           # broadcast over (B, N, 3)
            M = torch.diag(mirror)
            # M @ R @ M for every (b, v); contract via broadcasting.
            R_per_vert = M @ R_per_vert @ M

        # Multiplicative depth correction (hand_scale), per-batch-elt-centred.
        # Skip only when fixed-at-1.0 (preserves gradient flow when learnable).
        if self.hand_scale.requires_grad or float(self.hand_scale) != 1.0:
            c = verts.mean(dim=1, keepdim=True)                              # (B, 1, 3)
            verts = (verts - c) * self.hand_scale + c

        verts = verts + self.cam_t[t_idx].unsqueeze(1)                       # (B, 1, 3) → (B, N, 3)
        return verts, R_per_vert

    def wrist_pose_camera(
        self, t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-frame wrapper around ``batched_wrist_pose_camera``."""
        t_idx = torch.as_tensor([int(t)], device=self.cam_t.device, dtype=torch.long)
        R, tt = self.batched_wrist_pose_camera(t_idx)
        return R[0], tt[0]

    def batched_wrist_pose_camera(
        self, t_idx: torch.Tensor       # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-frame wrist (joint 0) cam-frame pose.

        Returns ((B, 3, 3), (B, 3)) — rotation and translation of the wrist
        joint in camera frame. Includes left/right mirror conjugation and
        the centroid-based hand_scale + cam_t offsets used by the vertex
        path, so the wrist transform is consistent with the rendered hand.

        The wrist orientation is MANO joint 0's absolute rotation (which
        equals the root rotation, ``global_orient``, up to mirror). The
        wrist position is MANO joint 0's absolute translation after the
        same centroid-scale + cam_t transforms applied to vertices.

        Used by ``WristAttachedGaussians`` to attach rigid arm Gaussians
        to the hand without distorting MANO.
        """
        B = int(t_idx.shape[0])
        pose_aa_b = torch.cat(
            [self.global_orient[t_idx], self.hand_pose[t_idx]], dim=-1
        )                                                                    # (B, 48)
        betas_b = self.betas.unsqueeze(0).expand(B, -1)                      # (B, 10)
        out = self.mano(pose_aa_b, betas_b)
        verts = out.verts                                                    # (B, N, 3)

        if hasattr(out, "transforms_abs"):
            R_wrist = out.transforms_abs[:, 0, :3, :3]                       # (B, 3, 3)
            t_wrist = out.transforms_abs[:, 0, :3, 3]                        # (B, 3)
        else:
            R_wrist = torch.eye(3, device=verts.device).expand(B, 3, 3).contiguous()
            # Fallback: joint 0 ≈ origin in MANO local frame.
            t_wrist = torch.zeros(B, 3, device=verts.device)

        if not self.is_right:
            mirror = verts.new_tensor([-1.0, 1.0, 1.0])
            verts   = verts * mirror
            t_wrist = t_wrist * mirror
            M = torch.diag(mirror)
            R_wrist = M @ R_wrist @ M

        # Apply centroid-based hand_scale to the wrist translation (same
        # transform applied to verts). hand_scale is uniform → no effect on
        # rotation.
        if self.hand_scale.requires_grad or float(self.hand_scale) != 1.0:
            c = verts.mean(dim=1)                                            # (B, 3)
            t_wrist = (t_wrist - c) * self.hand_scale + c

        t_wrist = t_wrist + self.cam_t[t_idx]                                # (B, 3)
        return R_wrist, t_wrist

    def export_track(self, raw_records: list[dict]) -> HandPoseTrack:
        """Return the optimized state as a HandPoseTrack for I/O.

        ``self.betas`` is shared (10,); broadcast to (T, 10) so the on-disk
        per-frame schema is preserved.
        """
        T = self.global_orient.shape[0]
        betas_per_frame = self.betas.detach().unsqueeze(0).expand(T, -1).contiguous()
        return HandPoseTrack(
            global_orient = self.global_orient.detach(),
            hand_pose     = self.hand_pose.detach(),
            betas         = betas_per_frame,
            cam_t         = self.cam_t.detach(),
            is_right      = self.is_right,
            frame_indices = list(self.frame_indices),
            raw_records   = raw_records,
            hand_scale    = float(self.hand_scale.detach().cpu().item()),
        )
