# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Post-process modules that correct drift in retargeted whole-body data.

The retarget loop in ``scripts/retarget/nvhuman_to_g1.py`` anchors the robot to
the ground on frame 0 but has no mechanism to track drift over the rest of the
sequence. The reconstruction's camera-to-world mapping can drift several
centimeters over long sequences, causing the robot to appear to sink or float
and causing objects to misalign with their rest surfaces.

This module provides a two-pass correction that runs **after** the existing
retarget loop. It consumes the loop's outputs (packaged in
:class:`FirstPassResult`) and produces two corrections:

1. A per-frame Z-offset for the robot (:func:`compute_plane_alignment_offsets`)
   that drags the lowest contact point (e.g. foot sole) to a reference plane
   every frame. The plane abstraction (:class:`ReferencePlane`) is
   horizontal-only for now (``z = constant``) but the call shape is
   intentionally future-proof: a ``contact_xyz`` array of any contact points
   the caller decides are relevant, and a plane object that knows its own
   signed distance.
2. A per-frame corrected object trajectory (:func:`correct_object_trajectory`)
   that follows two simple rules:

   - **Rule 1** (interaction): when the robot is interacting with the object,
     preserve the reconstructed hand-to-object relative pose by applying the
     same robot Z-delta to the object.
   - **Rule 2** (no interaction): the object is stationary in world frame,
     anchored to the adjacent contact segment so that release / pickup are
     seamless (no pop). Specifically, a no-interaction segment is held at:
       * the **last frame of the preceding contact** (release pose) when one
         exists, else
       * the **first frame of the following contact** (upcoming pickup pose)
         when one exists, else
       * the segment median (degenerate case: sequence has no contact at
         all, so there is no physical "placed" pose to honor).
     This mirrors what physically happens: the object is set down at
     ``P_release`` and stays there until the next pickup.

Interaction detection is its own pluggable module
(:func:`compute_interaction_mask`) so future strategies (beyond hand-contact)
plug in without touching object correction.

All functions are pure NumPy and do **not** require Pinocchio or Isaac. They
can be unit-tested against a synthetic :class:`FirstPassResult` without
rerunning IK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FirstPassResult:
    """Outputs from the retarget loop (pass 1).

    All arrays are post-pass-1 (i.e. the loop's frame-0 ``ground_z_offset`` is
    already applied), so the ``ankle_frame_xyz`` Z values line up with
    ``ee_pose_w`` Z values. All positions are in meters, all orientations are
    wxyz quaternions unless suffixed with ``axis_angle``.
    """

    fps: float
    # Robot world-frame trajectories (T, ...).
    robot_root_position: np.ndarray  # (T, 3)
    robot_root_wxyz: np.ndarray  # (T, 4)
    robot_joint_positions: np.ndarray  # (T, J)
    ee_pose_w: np.ndarray  # (T, E, 7) -- [x,y,z,qw,qx,qy,qz]
    # Object trajectories (T, ...).
    object_root_position: np.ndarray  # (T, 3)
    object_root_axis_angle: np.ndarray  # (T, 3)
    object_body_position: np.ndarray  # (T, B, 3)
    object_body_wxyz: np.ndarray  # (T, B, 4)
    # Per-side per-frame hand-contact mask (0.0/1.0).
    hand_contact_active_per_frame: np.ndarray  # (T, S)
    # Source-raw translations shifted by pass-1 ground offset.
    nvhuman_head_translation: np.ndarray  # (T, 3)
    nvhuman_root_translation: np.ndarray  # (T, 3)
    # NEW: cached ankle XYZ per frame (post pass-1 ground offset, pre-clamp).
    # Consistent with ee_pose_w's Z convention.
    ankle_frame_xyz: np.ndarray  # (T, 2, 3) -- index 0 = left, 1 = right


@dataclass(frozen=True)
class ReferencePlane:
    """Plane defined by a unit normal and a signed offset.

    Plane equation: ``normal . p + offset = 0`` (``p`` in robot world frame).
    For a horizontal plane at ``z = z0``, ``normal = (0, 0, 1)`` and
    ``offset = -z0`` so the legacy ``ReferencePlane(z=0.0)`` becomes
    ``ReferencePlane.horizontal(z=0.0)``.

    The plane is oriented so ``normal[2] > 0`` in robot frame; a transformed
    plane whose vertical component is below ``1e-6`` raises because a
    near-vertical "ground" plane is unusable for foot anchoring.

    Attributes:
        normal: Unit-normal triple. Stored as a tuple so the dataclass stays
            hashable; normalized at construction.
        offset: Signed offset such that ``n . p + offset = 0`` lies on the
            plane.
    """

    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    offset: float = 0.0

    def __post_init__(self) -> None:
        """Normalize ``normal`` (preserving direction) and validate vertical."""
        n = np.asarray(self.normal, dtype=np.float64)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            raise ValueError(f"ReferencePlane normal has zero length: {self.normal}")
        n = n / norm
        if abs(float(n[2])) < 1e-6:
            raise ValueError(
                "ReferencePlane normal is near-horizontal "
                f"(normal_z={float(n[2]):+.3e}); cannot ground feet on a vertical "
                "plane. Reorient the plane before constructing it."
            )
        if float(n[2]) < 0.0:
            n = -n
            object.__setattr__(self, "offset", -float(self.offset))
        object.__setattr__(self, "normal", (float(n[0]), float(n[1]), float(n[2])))

    @classmethod
    def horizontal(cls, z: float = 0.0) -> "ReferencePlane":
        """Return a horizontal plane at world Z = ``z`` (normal = +Z)."""
        return cls(normal=(0.0, 0.0, 1.0), offset=-float(z))

    @property
    def normal_z(self) -> float:
        """Vertical component of the (already-oriented) unit normal."""
        return float(self.normal[2])

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance ``n . p + offset`` for points ``(..., 3)``.

        Positive = above the plane along the unit normal (must-stay-above
        convention).
        """
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 0 or arr.shape[-1] != 3:
            raise ValueError(f"points last axis must be 3; got shape {arr.shape}")
        n = np.asarray(self.normal, dtype=np.float64)
        return np.einsum("...i,i->...", arr, n) + float(self.offset)

    def vertical_offset_to_plane(self, points: np.ndarray) -> np.ndarray:
        """Signed Z shift that drives each point's signed-distance to zero.

        For a horizontal plane this is just ``-signed_distance``. For a
        tilted plane with vertical component ``n_z``, vertical shift
        ``dz`` reduces signed distance by ``n_z * dz``, so the shift that
        zeros the distance is ``dz = -signed_distance / n_z``.
        """
        return -self.signed_distance(points) / self.normal_z


@dataclass
class PlaneAlignmentConfig:
    """Configuration for :func:`compute_plane_alignment_offsets`.

    The algorithm is deliberately simple: per frame, offset = negative of the
    minimum signed distance across contact points, then smooth with a
    rolling median (robust to brief swing-phase foot lifts and single-frame
    reconstruction outliers). An optional Savitzky-Golay pass after the
    median can produce gentler trajectory transitions.

    Attributes:
        median_window: Rolling-median window size (frames). Should be wider
            than the longest expected "both feet off the ground" gap and
            narrower than the timescale of real drift. Default ``11`` frames
            covers typical walking swing phases at 30 fps (~0.37 s).
        savgol_window: Optional Savitzky-Golay post-smoothing window length
            (odd frames). Set ``0`` to disable. Defaults to disabled because
            the rolling median alone is already a strong attenuator of
            single-frame noise, and extra smoothing lags real drift onsets.
        savgol_polyorder: Savitzky-Golay polynomial order. Only used when
            ``savgol_window > 0``.
    """

    median_window: int = 11
    savgol_window: int = 0
    savgol_polyorder: int = 3


@dataclass
class InteractionMaskConfig:
    """Configuration for :func:`compute_interaction_mask`.

    Attributes:
        strategy: Detection strategy.

            - ``hand_contact`` (default): OR the per-side
              ``hand_contact_active_per_frame`` signal from pass 1. This
              is the CORRECT-BY-DEFAULT choice because:

              1. ``hand_contact_active_per_frame`` now runs against the
                 object mesh at its raw (ungrounded) reconstruction Z, so
                 finger/palm contact fires accurately on true grips (e.g.
                 on ``skinny_wood_chair`` it reports ~1200 any-side frames
                 of real grip, vs. the noise-corrupted velocity signal).
              2. Rule 2 in ``correct_object_trajectory`` anchors the
                 object to the adjacent contact pose during no-contact
                 runs, which is what a physically-stationary placed /
                 pre-pickup object should look like. That anchoring only
                 triggers for no-contact frames, so an overly-eager
                 interaction mask (e.g. the old velocity default) forces
                 the object to track its noisy raw trajectory everywhere
                 and the "rest" segments visibly jitter.

            - ``hand_contact_or_velocity``: OR of hand_contact with a
              simple object-motion detector. A frame is interaction if
              its object root XY speed exceeds ``v_interact_mps``.
              Intended for sequences where the object is carried by
              non-hand body parts (e.g. a box hugged to the torso) so the
              palm/fingertip contact detector does not fire. NOT the
              default because the velocity signal is easily tripped by
              reconstruction noise during rest, which eliminates all
              no-contact frames and disables Rule 2 (the anchoring that
              keeps placed objects still).

        v_interact_mps: Object-root XY speed threshold (meters per second)
            used only by ``hand_contact_or_velocity``. Defaults to 5 cm/s.
        pad_frames: Number of frames to dilate the velocity-based mask on
            both sides so brief slow moments inside a carry don't chop
            the segment into tiny pieces.
    """

    strategy: Literal["hand_contact", "hand_contact_or_velocity"] = "hand_contact"
    v_interact_mps: float = 0.05
    pad_frames: int = 3


@dataclass
class ObjectCorrectionConfig:
    """Configuration for :func:`correct_object_trajectory`.

    Attributes:
        min_seg_frames: Segments shorter than this are absorbed into the
            adjacent segment to suppress single-frame contact dropouts.
        boundary_blend_frames: Number of frames over which to blend the
            correction across a segment boundary. Symmetric cosine S-curve
            on position; slerp on orientation.
    """

    min_seg_frames: int = 3
    boundary_blend_frames: int = 6


# ---------------------------------------------------------------------------
# Module 2: robot plane alignment
# ---------------------------------------------------------------------------


def compute_plane_alignment_offsets(
    contact_xyz: np.ndarray,
    plane: ReferencePlane,
    cfg: PlaneAlignmentConfig,
) -> np.ndarray:
    """Per-frame offset (along +Z) that puts the lowest contact on the plane.

    This is the "drag lowest contact to the plane" strategy, which replaces
    the prior stance-gated contact-aware drift estimator. The new approach
    is more robust to time-varying ground drift (the chair-sequence regime)
    because it does not depend on the cascade
    stance-detect -> NaN -> interp -> savgol -> pin-frame-0 that was
    silently losing amplitude when the real ground level moved mid-sequence.

    Algorithm:

    1. Compute signed distance to the plane for every contact point per
       frame.
    2. Take the minimum per frame (the foot closest to / deepest below the
       plane). The offset that would place that point exactly on the plane
       is ``-min(signed_distance)``.
    3. Apply a rolling median to absorb short excursions where every
       contact is briefly above the plane (e.g. a single swing-phase frame
       where both feet happen to be off the ground in reconstruction noise)
       or single-frame reconstruction spikes.
    4. Optionally apply Savitzky-Golay smoothing on top for gentler
       transitions (disabled by default).

    Args:
        contact_xyz: ``(T, K, 3)`` world-frame contact points per frame
            (e.g. per-side sole XYZ, or palm XYZ, or any point set the
            caller decides is the relevant grounding contact).
        plane: Reference plane. Horizontal-only for now.
        cfg: Smoothing parameters.

    Returns:
        ``(T,)`` offsets. For a horizontal plane, callers apply as
        ``position[..., 2] += offsets[...]`` to any world-frame Z
        coordinate they want grounded to the plane.
    """
    arr = np.asarray(contact_xyz, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("contact_xyz must have shape (T, K, 3); " f"got {arr.shape}")
    T, K, _ = arr.shape
    if T == 0:
        return np.zeros(0, dtype=np.float64)
    if K == 0:
        return np.zeros(T, dtype=np.float64)

    # Per-frame raw offset: vertical shift that drives the lowest contact
    # point's signed distance to zero (``dz = -signed_distance / normal_z``;
    # for a horizontal plane this collapses to the legacy ``-signed_distance``).
    dz = plane.vertical_offset_to_plane(arr)  # (T, K)
    raw = dz.max(axis=1)  # (T,)  -- lift by the largest correction needed

    # Rolling median smoothing.
    win = max(1, int(cfg.median_window))
    if win > 1 and T >= 3:
        if win % 2 == 0:
            win += 1
        # medfilt requires kernel_size <= T and odd; clamp safely.
        win = min(win, T if T % 2 == 1 else T - 1)
        if win >= 3:
            smoothed = medfilt(raw, kernel_size=win)
        else:
            smoothed = raw.copy()
    else:
        smoothed = raw.copy()

    # Optional Savgol smoothing (off by default).
    sw = int(cfg.savgol_window)
    if sw > 0 and T > sw:
        if sw % 2 == 0:
            sw += 1
        poly = max(1, min(int(cfg.savgol_polyorder), sw - 1, 5))
        smoothed = savgol_filter(smoothed, sw, poly, mode="nearest")

    return smoothed.astype(np.float64)


# ---------------------------------------------------------------------------
# Module 3: interaction mask
# ---------------------------------------------------------------------------


def compute_interaction_mask(
    first_pass: FirstPassResult,
    cfg: InteractionMaskConfig,
) -> np.ndarray:
    """Per-frame boolean mask for robot-object interaction.

    See :class:`InteractionMaskConfig` for the available strategies. The
    returned mask is consumed by :func:`correct_object_trajectory`: true
    frames get Rule 1 (object follows the robot), false frames get Rule 2
    (object is stationary, anchored to the adjacent contact).

    Args:
        first_pass: Pass-1 outputs. Uses
            ``hand_contact_active_per_frame`` and, for the velocity-based
            strategy, ``object_root_position`` and ``fps``.
        cfg: Strategy selection.

    Returns:
        ``(T,)`` boolean array where ``True`` means the robot is interacting
        with the object on that frame.
    """
    hca = np.asarray(first_pass.hand_contact_active_per_frame, dtype=np.float64)
    if hca.ndim == 0:
        T = 0
    elif hca.ndim == 1:
        T = hca.shape[0]
        hca = hca[:, None]
    else:
        T = hca.shape[0]

    if T == 0:
        return np.zeros(0, dtype=bool)

    # Hand-contact component: OR across sides.
    if hca.shape[1] == 0:
        hand_mask = np.zeros(T, dtype=bool)
    else:
        hand_mask = np.any(hca > 0.5, axis=1)

    if cfg.strategy == "hand_contact":
        return hand_mask
    if cfg.strategy != "hand_contact_or_velocity":
        raise NotImplementedError(
            f"Unknown interaction-mask strategy: {cfg.strategy!r}"
        )

    # Velocity-based component: frames where the object root is moving
    # faster than the threshold (in XY plane — vertical motion alone is a
    # weaker signal and can be caused by gravity/settling).
    obj_pos = np.asarray(first_pass.object_root_position, dtype=np.float64)
    if obj_pos.ndim != 2 or obj_pos.shape[-1] != 3 or obj_pos.shape[0] != T:
        return hand_mask
    fps = float(first_pass.fps)
    if fps <= 0 or T < 2:
        return hand_mask
    dt = 1.0 / fps
    # Finite-difference speed; pad the first frame with 0.
    diff = np.diff(obj_pos[:, :2], axis=0)
    speed = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.linalg.norm(diff, axis=-1) / dt]
    )  # (T,)
    vel_mask = speed > float(cfg.v_interact_mps)

    # Dilate the velocity mask by `pad_frames` on each side so brief slow
    # moments within a carry don't fragment the interaction segment.
    pad = max(0, int(cfg.pad_frames))
    if pad > 0 and vel_mask.any():
        kernel = np.ones(2 * pad + 1, dtype=bool)
        padded = np.convolve(
            vel_mask.astype(np.int32), kernel.astype(np.int32), mode="same"
        )
        vel_mask = padded > 0

    return hand_mask | vel_mask


# ---------------------------------------------------------------------------
# Module 4: object trajectory correction
# ---------------------------------------------------------------------------


def correct_object_trajectory(
    first_pass: FirstPassResult,
    interaction_mask: np.ndarray,
    robot_delta_z: np.ndarray,
    cfg: ObjectCorrectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Corrected object trajectory from rule-1 / rule-2 / boundary blend.

    Args:
        first_pass: Pass-1 outputs. Uses object arrays and their shapes.
        interaction_mask: ``(T,)`` bool from :func:`compute_interaction_mask`.
        robot_delta_z: ``(T,)`` from :func:`compute_plane_alignment_offsets`.
        cfg: Segmentation / blending parameters.

    Returns:
        Tuple of ``(root_position, root_axis_angle, body_position,
        body_wxyz)`` as arrays with the same shape as their ``first_pass``
        counterparts.
    """
    raw_root_pos = np.asarray(first_pass.object_root_position, dtype=np.float64)
    raw_root_aa = np.asarray(first_pass.object_root_axis_angle, dtype=np.float64)
    raw_body_pos = np.asarray(first_pass.object_body_position, dtype=np.float64)
    raw_body_wxyz = np.asarray(first_pass.object_body_wxyz, dtype=np.float64)
    mask = np.asarray(interaction_mask, dtype=bool)
    delta_z = np.asarray(robot_delta_z, dtype=np.float64)

    T = raw_root_pos.shape[0]
    if mask.shape[0] != T or delta_z.shape[0] != T:
        raise ValueError(
            f"Length mismatch: T={T}, mask={mask.shape[0]}, "
            f"delta_z={delta_z.shape[0]}"
        )

    corr_root_pos = raw_root_pos.copy()
    corr_root_aa = raw_root_aa.copy()
    corr_body_pos = raw_body_pos.copy()
    corr_body_wxyz = raw_body_wxyz.copy()

    if T == 0:
        return corr_root_pos, corr_root_aa, corr_body_pos, corr_body_wxyz

    # Derive raw root quats from axis-angle once.
    raw_root_quat = R.from_rotvec(raw_root_aa).as_quat(scalar_first=True)  # (T, 4)

    # Enumerate and merge short segments.
    segments = _enumerate_segments(mask)
    segments = _absorb_short_segments(segments, cfg.min_seg_frames)

    # Pre-compute a constant anchor pose for each no-contact segment. The
    # anchor is the adjacent contact segment's corrected boundary pose
    # (preceding contact's last frame preferred; following contact's first
    # frame as the fallback). If no contact exists anywhere in the sequence
    # we degrade to the segment median (there is no physical "placed" pose
    # to honor in that case).
    no_contact_anchors: dict[
        int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for idx, seg in enumerate(segments):
        if not seg[2]:
            no_contact_anchors[idx] = _compute_no_contact_anchor(
                seg_idx=idx,
                segments=segments,
                raw_root_pos=raw_root_pos,
                raw_root_quat=raw_root_quat,
                raw_body_pos=raw_body_pos,
                raw_body_wxyz=raw_body_wxyz,
                delta_z=delta_z,
            )

    # Phase A: per-segment native correction (no blending).
    for idx, seg in enumerate(segments):
        _write_segment(
            seg,
            no_contact_anchor=no_contact_anchors.get(idx),
            raw_root_pos=raw_root_pos,
            raw_root_quat=raw_root_quat,
            raw_body_pos=raw_body_pos,
            raw_body_wxyz=raw_body_wxyz,
            delta_z=delta_z,
            corr_root_pos=corr_root_pos,
            corr_root_aa=corr_root_aa,
            corr_body_pos=corr_body_pos,
            corr_body_wxyz=corr_body_wxyz,
        )

    # Phase B: boundary blends.
    blend = max(0, int(cfg.boundary_blend_frames))
    if blend > 0 and len(segments) > 1:
        for i in range(len(segments) - 1):
            _blend_boundary(
                left_seg=segments[i],
                right_seg=segments[i + 1],
                left_anchor=no_contact_anchors.get(i),
                right_anchor=no_contact_anchors.get(i + 1),
                blend_frames=blend,
                raw_root_pos=raw_root_pos,
                raw_root_quat=raw_root_quat,
                raw_body_pos=raw_body_pos,
                raw_body_wxyz=raw_body_wxyz,
                delta_z=delta_z,
                corr_root_pos=corr_root_pos,
                corr_root_aa=corr_root_aa,
                corr_body_pos=corr_body_pos,
                corr_body_wxyz=corr_body_wxyz,
            )

    return corr_root_pos, corr_root_aa, corr_body_pos, corr_body_wxyz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quaternion_mean(quats_wxyz: np.ndarray) -> np.ndarray:
    """Markley quaternion mean of ``(N, 4)`` wxyz quats. Returns ``(4,)``.

    Handles the quaternion double-cover via the largest-eigenvalue eigenvector
    of the (4x4) outer-product-sum matrix, so sign flips across ``q`` and
    ``-q`` do not bias the mean.
    """
    q = np.asarray(quats_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected (N, 4); got {q.shape}")
    if q.shape[0] == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # Normalize rows before accumulating; guards against degenerate inputs.
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    qn = q / norms
    M = qn.T @ qn / q.shape[0]
    eigvals, eigvecs = np.linalg.eigh(M)
    mean = eigvecs[:, np.argmax(eigvals)]
    # Canonical sign: scalar component non-negative.
    if mean[0] < 0:
        mean = -mean
    # Renormalize (eigh vectors are already unit-norm, but keep for safety).
    mean = mean / max(np.linalg.norm(mean), 1e-12)
    return mean


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp between two wxyz quats at blend ``t`` in ``[0, 1]``. Returns ``(4,)``."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    # Ensure same hemisphere.
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        out = (1.0 - t) * q0 + t * q1
        return out / max(np.linalg.norm(out), 1e-12)
    theta = np.arccos(dot)
    s = np.sin(theta)
    out = (np.sin((1.0 - t) * theta) * q0 + np.sin(t * theta) * q1) / s
    return out / max(np.linalg.norm(out), 1e-12)


def _enumerate_segments(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Return list of ``(start, end, in_contact)`` covering all frames."""
    T = mask.shape[0]
    if T == 0:
        return []
    segments: list[tuple[int, int, bool]] = []
    cur = bool(mask[0])
    start = 0
    for t in range(1, T):
        v = bool(mask[t])
        if v != cur:
            segments.append((start, t, cur))
            start = t
            cur = v
    segments.append((start, T, cur))
    return segments


def _absorb_short_segments(
    segments: list[tuple[int, int, bool]], min_seg_frames: int
) -> list[tuple[int, int, bool]]:
    """Greedy: merge any segment shorter than ``min_seg_frames`` with its neighbor."""
    if not segments or min_seg_frames <= 1:
        return list(segments)
    merged: list[tuple[int, int, bool]] = []
    for seg in segments:
        length = seg[1] - seg[0]
        if merged and length < min_seg_frames:
            ps, _, pv = merged[-1]
            merged[-1] = (ps, seg[1], pv)
        else:
            merged.append(seg)
    # Also handle a leading short segment: if it exists and we have a successor,
    # merge forward instead of leaving it orphaned.
    if len(merged) >= 2 and (merged[0][1] - merged[0][0]) < min_seg_frames:
        _, end0, _ = merged[0]
        s1, e1, v1 = merged[1]
        merged[0:2] = [(merged[0][0], e1, v1)]
    return merged


def _contact_pose_at(
    t: int,
    raw_root_pos: np.ndarray,
    raw_root_quat: np.ndarray,
    raw_body_pos: np.ndarray,
    raw_body_wxyz: np.ndarray,
    delta_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rule-1 pose at frame ``t``: raw pose with only the robot Z-delta added."""
    root_pos = raw_root_pos[t].copy()
    root_pos[2] += delta_z[t]
    root_quat = raw_root_quat[t].copy()
    body_pos = raw_body_pos[t].copy()
    body_pos[..., 2] += delta_z[t]
    body_wxyz = raw_body_wxyz[t].copy()
    return root_pos, root_quat, body_pos, body_wxyz


def _compute_no_contact_anchor(
    *,
    seg_idx: int,
    segments: list[tuple[int, int, bool]],
    raw_root_pos: np.ndarray,
    raw_root_quat: np.ndarray,
    raw_body_pos: np.ndarray,
    raw_body_wxyz: np.ndarray,
    delta_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the constant anchor pose for a no-contact segment.

    Priority:

    1. **Preceding-contact anchor** (seamless release): the corrected pose
       on the last frame of the nearest preceding contact segment. This is
       the physically "placed" pose.
    2. **Following-contact anchor** (seamless pickup): the corrected pose
       on the first frame of the nearest following contact segment. Used
       only when there is no preceding contact at all in the sequence so
       far.
    3. **Segment-median fallback**: used only when the whole sequence has
       no contact (there is no physical anchor to honor). Degenerate case.
    """
    start, end, in_contact = segments[seg_idx]
    assert not in_contact

    # 1. Preceding contact segment (closest, searching backward).
    for i in range(seg_idx - 1, -1, -1):
        prev_start, prev_end, prev_in = segments[i]
        if prev_in:
            return _contact_pose_at(
                prev_end - 1,
                raw_root_pos,
                raw_root_quat,
                raw_body_pos,
                raw_body_wxyz,
                delta_z,
            )

    # 2. Following contact segment (closest, searching forward).
    for i in range(seg_idx + 1, len(segments)):
        next_start, next_end, next_in = segments[i]
        if next_in:
            return _contact_pose_at(
                next_start,
                raw_root_pos,
                raw_root_quat,
                raw_body_pos,
                raw_body_wxyz,
                delta_z,
            )

    # 3. Degenerate fallback: no contact anywhere in the sequence.
    seg_slice_root_pos = raw_root_pos[start:end]
    seg_slice_root_quat = raw_root_quat[start:end]
    seg_slice_body_pos = raw_body_pos[start:end]
    seg_slice_body_wxyz = raw_body_wxyz[start:end]
    root_pos = np.median(seg_slice_root_pos, axis=0)
    root_quat = _quaternion_mean(seg_slice_root_quat)
    body_pos = np.median(seg_slice_body_pos, axis=0)
    B = seg_slice_body_wxyz.shape[1] if seg_slice_body_wxyz.ndim >= 2 else 0
    body_wxyz = np.empty((B, 4), dtype=np.float64)
    for b in range(B):
        body_wxyz[b] = _quaternion_mean(seg_slice_body_wxyz[:, b])
    return root_pos, root_quat, body_pos, body_wxyz


def _segment_pose_at(
    t: int,
    seg: tuple[int, int, bool],
    no_contact_anchor: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    raw_root_pos: np.ndarray,
    raw_root_quat: np.ndarray,
    raw_body_pos: np.ndarray,
    raw_body_wxyz: np.ndarray,
    delta_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``seg``'s native correction at time ``t``.

    Returns ``(root_pos, root_quat, body_pos, body_wxyz)``.

    ``t`` may lie outside ``[seg.start, seg.end)`` for boundary-blending
    purposes. No-contact (rule-2) poses are constants so they extend trivially.
    Contact (rule-1) poses are clamped to the segment's nearest boundary frame
    when ``t`` is outside the segment: a contact segment's "extrapolation"
    outside its own range is constant at its boundary value, which avoids
    pulling blended values toward potentially unrelated ``raw[t]`` samples
    from a neighboring reconstruction regime.

    ``no_contact_anchor`` must be supplied for non-contact segments (computed
    by :func:`_compute_no_contact_anchor`) and is ignored for contact segments.
    """
    start, end, in_contact = seg
    if in_contact:
        t_eff = max(start, min(end - 1, t))
        root_pos, root_quat, body_pos, body_wxyz = _contact_pose_at(
            t_eff, raw_root_pos, raw_root_quat, raw_body_pos, raw_body_wxyz, delta_z
        )
    else:
        if no_contact_anchor is None:
            raise ValueError("no_contact_anchor required for non-contact segment")
        # Fresh copies so callers cannot mutate the cached anchor.
        root_pos, root_quat, body_pos, body_wxyz = no_contact_anchor
        root_pos = root_pos.copy()
        root_quat = root_quat.copy()
        body_pos = body_pos.copy()
        body_wxyz = body_wxyz.copy()
    return root_pos, root_quat, body_pos, body_wxyz


def _write_segment(
    seg: tuple[int, int, bool],
    *,
    no_contact_anchor: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    raw_root_pos: np.ndarray,
    raw_root_quat: np.ndarray,
    raw_body_pos: np.ndarray,
    raw_body_wxyz: np.ndarray,
    delta_z: np.ndarray,
    corr_root_pos: np.ndarray,
    corr_root_aa: np.ndarray,
    corr_body_pos: np.ndarray,
    corr_body_wxyz: np.ndarray,
) -> None:
    """Write ``seg``'s native correction into the corrected arrays."""
    start, end, in_contact = seg
    if in_contact:
        corr_root_pos[start:end] = raw_root_pos[start:end]
        corr_root_pos[start:end, 2] += delta_z[start:end]
        # orientation unchanged (rule 1)
        corr_root_aa[start:end] = R.from_quat(
            raw_root_quat[start:end], scalar_first=True
        ).as_rotvec()
        if corr_body_pos.ndim == 3:
            corr_body_pos[start:end] = raw_body_pos[start:end]
            corr_body_pos[start:end, :, 2] += delta_z[start:end, None]
        if corr_body_wxyz.ndim == 3:
            corr_body_wxyz[start:end] = raw_body_wxyz[start:end]
        return

    # Rule 2: broadcast the pre-computed anchor pose across the segment.
    if no_contact_anchor is None:
        raise ValueError("no_contact_anchor required for non-contact segment")
    root_pos, root_quat, body_pos, body_wxyz = no_contact_anchor
    corr_root_pos[start:end] = root_pos[None, :]
    corr_root_aa[start:end] = R.from_quat(root_quat, scalar_first=True).as_rotvec()[
        None, :
    ]
    if corr_body_pos.ndim == 3:
        corr_body_pos[start:end] = body_pos[None, :, :]
    if corr_body_wxyz.ndim == 3:
        corr_body_wxyz[start:end] = body_wxyz[None, :, :]


def _blend_boundary(
    *,
    left_seg: tuple[int, int, bool],
    right_seg: tuple[int, int, bool],
    left_anchor: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    right_anchor: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    blend_frames: int,
    raw_root_pos: np.ndarray,
    raw_root_quat: np.ndarray,
    raw_body_pos: np.ndarray,
    raw_body_wxyz: np.ndarray,
    delta_z: np.ndarray,
    corr_root_pos: np.ndarray,
    corr_root_aa: np.ndarray,
    corr_body_pos: np.ndarray,
    corr_body_wxyz: np.ndarray,
) -> None:
    """Cosine S-curve blend across a ``blend_frames``-wide boundary window.

    Overwrites the window centered at the boundary with a blend of the two
    adjoining segments' native corrections.

    With the updated Rule 2 (no-contact anchored to adjacent contact), the
    blend is often a no-op at contact->no-contact boundaries because both
    sides produce the same pose at the boundary frame. It still handles the
    no-contact->contact (pickup) transition, where the anchored pose may
    differ slightly from the first contact frame's raw + delta, and any
    degenerate no-contact-only sequence.
    """
    t_b = right_seg[0]
    half = max(1, blend_frames // 2)
    blend_lo = max(t_b - half, left_seg[0])
    blend_hi = min(t_b + half, right_seg[1])
    n = blend_hi - blend_lo
    if n <= 1:
        return

    # Cosine ramp: 0 at blend_lo, 1 at blend_hi - 1.
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))

    for idx, t in enumerate(range(blend_lo, blend_hi)):
        w = float(ramp[idx])
        l_pos, l_quat, l_bpos, l_bwxyz = _segment_pose_at(
            t,
            left_seg,
            left_anchor,
            raw_root_pos,
            raw_root_quat,
            raw_body_pos,
            raw_body_wxyz,
            delta_z,
        )
        r_pos, r_quat, r_bpos, r_bwxyz = _segment_pose_at(
            t,
            right_seg,
            right_anchor,
            raw_root_pos,
            raw_root_quat,
            raw_body_pos,
            raw_body_wxyz,
            delta_z,
        )
        corr_root_pos[t] = (1.0 - w) * l_pos + w * r_pos
        corr_root_aa[t] = R.from_quat(
            _slerp_wxyz(l_quat, r_quat, w), scalar_first=True
        ).as_rotvec()
        if corr_body_pos.ndim == 3:
            corr_body_pos[t] = (1.0 - w) * l_bpos + w * r_bpos
        if corr_body_wxyz.ndim == 3:
            B = corr_body_wxyz.shape[1]
            for b in range(B):
                corr_body_wxyz[t, b] = _slerp_wxyz(l_bwxyz[b], r_bwxyz[b], w)


# ---------------------------------------------------------------------------
# Module 5: ground_plane.json loader (reconstruction-side static plane)
# ---------------------------------------------------------------------------


def _transform_plane_under_rigid(
    plane: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Transform a plane ``(a, b, c, d)`` under ``x_new = R x_old + t``.

    Plane equation transforms as ``n_new = R @ n_old`` and
    ``d_new = d_old - n_new^T @ t`` so points satisfying ``n_old.p_old + d = 0``
    in the source frame still satisfy ``n_new.p_new + d_new = 0`` after the
    rigid mapping.
    """
    n_old = np.asarray(plane[:3], dtype=np.float64)
    d_old = float(plane[3])
    R_arr = np.asarray(rotation, dtype=np.float64)
    t_arr = np.asarray(translation, dtype=np.float64).reshape(3)
    n_new = R_arr @ n_old
    d_new = d_old - float(n_new @ t_arr)
    return np.concatenate([n_new, [d_new]])


def load_ground_plane_robot_frame(
    path: Path | str,
    *,
    cv_to_source: np.ndarray,
    first_frame_anchor: np.ndarray,
    source_to_robot: np.ndarray,
) -> ReferencePlane | None:
    """Load ``ground_plane.json`` and transform it into the robot frame.

    The reconstruction stores the plane in OpenCV camera coordinates
    (X=right, Y=down, Z=forward). The retargeter then composes three
    transforms before the robot world frame is reached:

    1. ``cv_to_source`` — homogeneous (4x4) flip from CV to the source
       skeleton's world frame (see ``_convert_object_poses_cv_to_soma``).
    2. ``first_frame_anchor`` — homogeneous (4x4) anchor that places the
       source skeleton's frame-0 pelvis at the origin (see
       ``SOMA._first_frame_transform``).
    3. ``source_to_robot`` — 3x3 rotation that swaps source axes into
       robot axes (``config.r_world`` in the new SOMA pipeline).

    The plane is transformed under each rigid mapping and finally returned
    as a :class:`ReferencePlane` in robot frame, or ``None`` when the JSON
    file is absent so the caller can fall back to the legacy heuristics.

    Args:
        path: Path to ``ground_plane.json``.
        cv_to_source: ``(4, 4)`` homogeneous transform from CV to source frame.
        first_frame_anchor: ``(4, 4)`` homogeneous transform that anchors the
            source skeleton on its frame-0 pelvis.
        source_to_robot: ``(3, 3)`` rotation from source world to robot world.

    Returns:
        :class:`ReferencePlane` in robot frame, or ``None`` if the JSON does
        not exist on disk. Raises :class:`ValueError` for malformed payloads.
    """
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "plane" not in payload or len(payload["plane"]) != 4:
        raise ValueError(f"{path}: expected a 4-element 'plane' list (ax+by+cz+d=0)")
    plane = np.asarray(payload["plane"], dtype=np.float64)

    # 1. CV -> source frame (rigid).
    cv_to_source_arr = np.asarray(cv_to_source, dtype=np.float64)
    if cv_to_source_arr.shape != (4, 4):
        raise ValueError(f"cv_to_source must be (4, 4); got {cv_to_source_arr.shape}")
    plane = _transform_plane_under_rigid(
        plane, cv_to_source_arr[:3, :3], cv_to_source_arr[:3, 3]
    )

    # 2. First-frame anchor (rigid).
    anchor_arr = np.asarray(first_frame_anchor, dtype=np.float64)
    if anchor_arr.shape != (4, 4):
        raise ValueError(f"first_frame_anchor must be (4, 4); got {anchor_arr.shape}")
    plane = _transform_plane_under_rigid(plane, anchor_arr[:3, :3], anchor_arr[:3, 3])

    # 3. Source -> robot rotation (no translation).
    src_to_robot_arr = np.asarray(source_to_robot, dtype=np.float64)
    if src_to_robot_arr.shape != (3, 3):
        raise ValueError(
            f"source_to_robot must be (3, 3); got {src_to_robot_arr.shape}"
        )
    plane = _transform_plane_under_rigid(
        plane, src_to_robot_arr, np.zeros(3, dtype=np.float64)
    )

    n = plane[:3]
    d = float(plane[3])
    return ReferencePlane(normal=(float(n[0]), float(n[1]), float(n[2])), offset=d)
