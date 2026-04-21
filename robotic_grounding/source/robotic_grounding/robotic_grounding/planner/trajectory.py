"""Trajectory building for planner warmup and reference playback.

Builds a 4-part trajectory: hold nominal → interpolate → hold start → reference.
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def build_interp_trajectory(
    nominal_ee: dict,
    ref_left_pos: np.ndarray,
    ref_left_quat: np.ndarray,
    ref_right_pos: np.ndarray,
    ref_right_quat: np.ndarray,
    fps: float,
    hold_start_s: float = 5.0,
    interp_s: float = 5.0,
    hold_end_s: float = 5.0,
    n_ref: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a warmup + reference trajectory for the planner.

    Segments:
        1. Hold nominal pose for hold_start_s seconds
        2. SLERP/linear interpolate from nominal to reference[0] over interp_s
        3. Hold reference[0] for hold_end_s seconds
        4. Append reference data (first n_ref frames, or all if n_ref=-1)

    Args:
        nominal_ee: Dict with left_pos, left_quat, right_pos, right_quat (wxyz).
        ref_left_pos: (T, 3) reference left wrist positions.
        ref_left_quat: (T, 4) reference left wrist quaternions (wxyz).
        ref_right_pos: (T, 3) reference right wrist positions.
        ref_right_quat: (T, 4) reference right wrist quaternions (wxyz).
        fps: Output frame rate.
        hold_start_s: Duration of nominal hold.
        interp_s: Duration of interpolation.
        hold_end_s: Duration of start-pose hold.
        n_ref: Number of reference frames to include (-1 = all).

    Returns:
        (left_pos, left_quat, right_pos, right_quat, segment_info)
        Each is a concatenated (N_total, ...) array.
        segment_info: dict with segment lengths and ref_start index.
    """
    n_hold_start = int(hold_start_s * fps)
    n_interp = int(interp_s * fps)
    n_hold_end = int(hold_end_s * fps)
    if n_ref < 0:
        n_ref = len(ref_left_pos)
    else:
        n_ref = min(n_ref, len(ref_left_pos))

    # Nominal pose (single frame)
    nom_lp = np.array(nominal_ee["left_pos"], dtype=np.float32)
    nom_lq = np.array(nominal_ee["left_quat"], dtype=np.float32)
    nom_rp = np.array(nominal_ee["right_pos"], dtype=np.float32)
    nom_rq = np.array(nominal_ee["right_quat"], dtype=np.float32)

    # Reference start (frame 0)
    tgt_lp = ref_left_pos[0].astype(np.float32)
    tgt_lq = ref_left_quat[0].astype(np.float32)
    tgt_rp = ref_right_pos[0].astype(np.float32)
    tgt_rq = ref_right_quat[0].astype(np.float32)

    parts_lp, parts_lq, parts_rp, parts_rq = [], [], [], []

    # Segment 1: hold nominal
    if n_hold_start > 0:
        parts_lp.append(np.tile(nom_lp, (n_hold_start, 1)))
        parts_lq.append(np.tile(nom_lq, (n_hold_start, 1)))
        parts_rp.append(np.tile(nom_rp, (n_hold_start, 1)))
        parts_rq.append(np.tile(nom_rq, (n_hold_start, 1)))

    # Segment 2: interpolate nominal → reference[0]
    if n_interp > 0:
        alpha = np.linspace(0.0, 1.0, n_interp, dtype=np.float32)

        # Linear position interpolation
        parts_lp.append(nom_lp[None] + alpha[:, None] * (tgt_lp - nom_lp)[None])
        parts_rp.append(nom_rp[None] + alpha[:, None] * (tgt_rp - nom_rp)[None])

        # SLERP quaternion interpolation
        parts_lq.append(_slerp_array(nom_lq, tgt_lq, alpha))
        parts_rq.append(_slerp_array(nom_rq, tgt_rq, alpha))

    # Segment 3: hold reference start
    if n_hold_end > 0:
        parts_lp.append(np.tile(tgt_lp, (n_hold_end, 1)))
        parts_lq.append(np.tile(tgt_lq, (n_hold_end, 1)))
        parts_rp.append(np.tile(tgt_rp, (n_hold_end, 1)))
        parts_rq.append(np.tile(tgt_rq, (n_hold_end, 1)))

    # Segment 4: reference data
    ref_start = n_hold_start + n_interp + n_hold_end
    parts_lp.append(ref_left_pos[:n_ref].astype(np.float32))
    parts_lq.append(ref_left_quat[:n_ref].astype(np.float32))
    parts_rp.append(ref_right_pos[:n_ref].astype(np.float32))
    parts_rq.append(ref_right_quat[:n_ref].astype(np.float32))

    seg_info = {
        "n_hold_start": n_hold_start,
        "n_interp": n_interp,
        "n_hold_end": n_hold_end,
        "n_ref": n_ref,
        "ref_start": ref_start,
    }

    return (
        np.concatenate(parts_lp),
        np.concatenate(parts_lq),
        np.concatenate(parts_rp),
        np.concatenate(parts_rq),
        seg_info,
    )


def _slerp_array(
    q0_wxyz: np.ndarray,
    q1_wxyz: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """SLERP between two wxyz quaternions at multiple alpha values.

    Args:
        q0_wxyz: (4,) start quaternion, wxyz.
        q1_wxyz: (4,) end quaternion, wxyz.
        alpha: (N,) interpolation weights in [0, 1].

    Returns:
        (N, 4) interpolated quaternions, wxyz.
    """
    q0_xyzw = q0_wxyz[[1, 2, 3, 0]]
    q1_xyzw = q1_wxyz[[1, 2, 3, 0]]
    key_rots = Rotation.from_quat([q0_xyzw, q1_xyzw])
    slerp = Slerp([0.0, 1.0], key_rots)
    interp_rots = slerp(alpha)
    out_xyzw = interp_rots.as_quat()
    return out_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
