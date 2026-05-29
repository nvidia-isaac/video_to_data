# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Gap-fill per-track aligned hand records by interpolating between real frames.

Consumes the output of hamer's ``align_hands`` (real-only, post-alignment):
  aligned_dir/<track_id>/<frame:06d>.json
with fields ``mano``, ``cam_t``, ``intrinsics``, ``image_size``, ``diagnostics``.

Why post-align (and not pre-align)?
  Interpolated frames are guesses — their MANO mesh may not match the image
  silhouette there. Aligning a guess against real depth gives the wrong cam_t.
  By running alignment first (on real detections only) and interpolating the
  *aligned* records, we keep alignment outputs trustworthy and still hand
  downstream consumers (overlays, gsplat refinement) a gap-free trajectory.

Inputs:
  aligned_dir/<track_id>/<frame:06d>.json   aligned real-only records
  masks_dir/<track_id>/<frame:06d>.png      SAM2 propagated track masks
                                            (gates which missing frames
                                            deserve interpolation: only ones
                                            where the hand is visible)

Output:
  output_dir/<track_id>/<frame:06d>.json    same schema as input plus:
    "interpolated": bool                    False on real frames, True on filled
    "interpolation": {"prev","next","weight"}    only on filled frames

Algorithm per track:
  1. Discover SAM2 mask frames (the candidate domain).
  2. Real frames pass through with ``interpolated: false``. When
     ``--betas fixed``, betas are overwritten with the median per track so
     real + interpolated frames share the same hand shape.
  3. For each missing frame f with a SAM2 mask, find nearest real
     neighbours. If the bracket gap exceeds --max_gap_frames, skip.
  4. SLERP global_orient + each of 15 hand_pose joints (axis-angle ↔ quat).
  5. Linear interp cam_t. ``intrinsics`` + ``image_size`` are taken from a
     neighbour (they're sequence-constant after MoGe stabilisation).
  6. ``diagnostics`` is filled with marker values
     (dz=0, n_pixels=0, scale=1.0, cam_t_pre_dz=cam_t, scaled_focal=NaN)
     so the schema stays uniform; downstream code should key on
     ``interpolated`` to discriminate.
  7. No extrapolation outside [first_real, last_real].

Usage:
    python -m v2d.wilor.lib.tracks_interpolate \\
        --aligned_dir /data/wilor_aligned \\
        --masks_dir   /data/masks \\
        --output_dir  /data/wilor_aligned_filled \\
        --betas fixed --max_gap_frames 15
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Optional

import numpy as np


def _axisangle_to_quat(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle → (..., 4) quaternion (w, x, y, z)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    half = theta * 0.5
    sin_half_over_theta = np.where(small, 0.5, np.sin(half) / np.where(small, 1.0, theta))
    w = np.cos(half)
    xyz = aa * sin_half_over_theta
    return np.concatenate([w, xyz], axis=-1)


def _quat_to_axisangle(q: np.ndarray) -> np.ndarray:
    """(..., 4) quaternion (w, x, y, z) → (..., 3) axis-angle."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w = np.clip(q[..., 0:1], -1.0, 1.0)
    xyz = q[..., 1:4]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    theta = 2.0 * np.arctan2(sin_half, w)
    axis = np.where(sin_half < 1e-8, np.zeros_like(xyz), xyz / np.where(sin_half < 1e-8, 1.0, sin_half))
    return axis * theta


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP on (..., 4) quaternions; t scalar in [0, 1]."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    eps = 1e-6
    close = dot > 1.0 - eps
    omega = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_omega = np.sin(omega)
    a = np.where(close, 1.0 - t, np.sin((1.0 - t) * omega) / np.where(close, 1.0, sin_omega))
    b = np.where(close,       t, np.sin(       t  * omega) / np.where(close, 1.0, sin_omega))
    return a * q0 + b * q1


def _slerp_axisangle(aa0: np.ndarray, aa1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two axis-angle vectors (or batches)."""
    q0 = _axisangle_to_quat(np.asarray(aa0))
    q1 = _axisangle_to_quat(np.asarray(aa1))
    q  = _slerp(q0, q1, t)
    return _quat_to_axisangle(q)


def _interp_aligned_record(
    prev: dict, nxt: dict, t: float, betas_override: Optional[list],
) -> dict:
    """Blend two real aligned records into a filled record at fraction t in [0, 1].

    Caller adds track_id / frame_idx / interpolated / interpolation.
    """
    g0 = np.array(prev["mano"]["global_orient"], dtype=np.float64)
    g1 = np.array(nxt ["mano"]["global_orient"], dtype=np.float64)
    h0 = np.array(prev["mano"]["hand_pose"],     dtype=np.float64).reshape(-1, 3)
    h1 = np.array(nxt ["mano"]["hand_pose"],     dtype=np.float64).reshape(-1, 3)
    global_orient = _slerp_axisangle(g0, g1, t).tolist()
    hand_pose     = _slerp_axisangle(h0, h1, t).reshape(-1).tolist()

    if betas_override is not None:
        betas = list(betas_override)
    else:
        b0 = np.array(prev["mano"]["betas"], dtype=np.float64)
        b1 = np.array(nxt ["mano"]["betas"], dtype=np.float64)
        betas = ((1 - t) * b0 + t * b1).tolist()

    c0 = np.array(prev["cam_t"], dtype=np.float64)
    c1 = np.array(nxt ["cam_t"], dtype=np.float64)
    cam_t = ((1 - t) * c0 + t * c1).tolist()

    # intrinsics + image_size are sequence-constant; pick either neighbour.
    # hand_scale is per-track (constant across frames within a track), so we
    # just forward it unchanged when present.
    out = {
        "is_right":   prev["is_right"],
        "image_size": prev.get("image_size") or nxt.get("image_size"),
        "intrinsics": prev["intrinsics"],
        "mano": {
            "betas":         betas,
            "global_orient": global_orient,
            "hand_pose":     hand_pose,
        },
        "cam_t":      cam_t,
        # Marker diagnostics so the schema stays uniform with align_hands
        # output. Downstream code should look at `interpolated` to know.
        "diagnostics": {
            "dz":           0.0,
            "n_pixels":     0,
            "scale":        1.0,
            "cam_t_pre_dz": cam_t,
            "scaled_focal": math.nan,
        },
    }
    if "hand_scale" in prev:
        out["hand_scale"] = float(prev["hand_scale"])
    elif "hand_scale" in nxt:
        out["hand_scale"] = float(nxt["hand_scale"])
    return out


def _mask_frames(masks_track_dir: str) -> set[int]:
    out: set[int] = set()
    for p in glob.glob(os.path.join(masks_track_dir, "*.png")):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            out.add(int(stem))
        except ValueError:
            continue
    return out


def _real_frames(track_dir: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(glob.glob(os.path.join(track_dir, "*.json"))):
        with open(p) as f:
            rec = json.load(f)
        out[int(rec["frame_idx"])] = rec
    return out


def _track_median_betas(real: dict[int, dict]) -> list[float]:
    arr = np.array([r["mano"]["betas"] for r in real.values()], dtype=np.float64)
    return np.median(arr, axis=0).tolist()


def _wrist_u(rec: dict) -> float | None:
    """Project the wrist's cam_t through the record's intrinsics → image u (px).

    Used as a position proxy for the egocentric handedness tiebreak: a wrist
    in the left half of the image is more likely a left hand and vice versa.
    Returns None when intrinsics or cam_t is missing or malformed.
    """
    intr  = rec.get("intrinsics")
    cam_t = rec.get("cam_t")
    if intr is None or cam_t is None:
        return None
    try:
        z = max(float(cam_t[2]), 1e-3)
        return float(intr["fx"]) * float(cam_t[0]) / z + float(intr["cx"])
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def _aggregate_handedness(
    track_real_records: dict[str, dict[int, dict]],
) -> dict[str, dict]:
    """Per-track canonical ``is_right`` from a uniform majority vote across
    real records, with a position-based tiebreak when multiple tracks vote
    the same handedness.

    Returns a dict keyed by track id (string) with structure::

        {tid: {"is_right":     bool,
               "votes_right":  int,
               "votes_left":   int,
               "n_frames":     int,
               "mean_wrist_u": float | None,
               "tiebroke":     bool}}

    Tiebreak (only fires when 2+ tracks share canonical handedness):
      * "right" cluster: the track with the HIGHEST mean wrist_u keeps right;
        all others in the cluster flip to left.
      * "left" cluster: the track with the LOWEST mean wrist_u keeps left;
        all others flip to right.
    This is the standard egocentric convention (right hand on the right side
    of the image).
    """
    per_track: dict[str, dict] = {}
    for tid, real in track_real_records.items():
        if not real:
            per_track[tid] = {
                "is_right": True, "votes_right": 0, "votes_left": 0,
                "n_frames": 0, "mean_wrist_u": None, "tiebroke": False,
            }
            continue
        vr = sum(1 for r in real.values() if bool(r.get("is_right")))
        vl = len(real) - vr
        us = [u for u in (_wrist_u(r) for r in real.values()) if u is not None]
        mean_u = float(np.mean(us)) if us else None
        per_track[tid] = {
            "is_right":     vr >= vl,           # ties → right (arbitrary, tiebreak handles it)
            "votes_right":  int(vr),
            "votes_left":   int(vl),
            "n_frames":     int(len(real)),
            "mean_wrist_u": mean_u,
            "tiebroke":     False,
        }

    # Detect clusters (multiple tracks with the same canonical handedness) and
    # flip the "least-justifiable" claims based on mean wrist u.
    for hd in (True, False):
        tids = [t for t, d in per_track.items() if d["is_right"] is hd]
        if len(tids) <= 1:
            continue
        side = "right" if hd else "left"
        tids_with_u = [t for t in tids if per_track[t]["mean_wrist_u"] is not None]
        if not tids_with_u:
            print(f"  WARNING: {len(tids)} tracks {tids} all vote {side}-handed; "
                  f"cannot tiebreak (no projected wrist u available).")
            continue
        if hd:
            keeper = max(tids_with_u, key=lambda t: per_track[t]["mean_wrist_u"])
        else:
            keeper = min(tids_with_u, key=lambda t: per_track[t]["mean_wrist_u"])
        flipped = [t for t in tids if t != keeper]
        if flipped:
            print(f"  WARNING: {len(tids)} tracks {tids} all vote {side}-handed; "
                  f"tiebreaking by mean wrist u — track {keeper} keeps {side}, "
                  f"flipping {flipped}.")
        for t in flipped:
            per_track[t]["is_right"] = not hd
            per_track[t]["tiebroke"] = True
    return per_track


def tracks_interpolate(
    aligned_dir: str,
    masks_dir: str,
    output_dir: str,
    betas: str = "fixed",
    max_gap_frames: int = 100000,
    extrapolate: bool = True,
) -> None:
    """Fill per-track aligned records to cover every SAM2-masked frame.

    ``extrapolate=True`` (default) clamps to the nearest real record for
    frames outside ``[first_real, last_real]``; ``False`` restricts the
    output to that bracket. ``max_gap_frames`` caps interior gap size
    (default effectively unlimited).
    """
    if betas not in ("fixed", "interp"):
        raise ValueError(f"--betas must be 'fixed' or 'interp', got {betas!r}")

    track_dirs = sorted(
        d for d in glob.glob(os.path.join(aligned_dir, "*"))
        if os.path.isdir(d)
    )
    if not track_dirs:
        raise FileNotFoundError(f"No track subdirs in {aligned_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # First pass: load real records per track + compute per-track canonical
    # handedness from the entire sequence (per-frame WiLoR classifications
    # vote; tracks_from_wilor_masks already filtered to plausible hand
    # detections per frame). This corrects ref-frame misclassifications that
    # would otherwise stick in hand_tracks.json + propagate to refinement.
    track_real: dict[str, dict[int, dict]] = {}
    for track_dir in track_dirs:
        tid = os.path.basename(track_dir)
        track_real[tid] = _real_frames(track_dir)

    handedness = _aggregate_handedness(track_real)
    for tid, d in handedness.items():
        flag = "TIEBROKE" if d["tiebroke"] else ""
        print(f"  track {tid}: votes R={d['votes_right']} L={d['votes_left']} "
              f"→ is_right={d['is_right']}  {flag}".rstrip())

    summary: list[tuple[str, int, int, int, int]] = []

    for track_dir in track_dirs:
        tid = os.path.basename(track_dir)
        out_track = os.path.join(output_dir, tid)
        os.makedirs(out_track, exist_ok=True)

        real = track_real[tid]
        canonical_is_right = bool(handedness[tid]["is_right"])
        if not real:
            print(f"  track {tid}: no real aligned records — nothing to do")
            summary.append((tid, 0, 0, 0))
            continue
        sorted_real_idx = sorted(real.keys())
        first_real, last_real = sorted_real_idx[0], sorted_real_idx[-1]

        mask_track_dir = os.path.join(masks_dir, tid)
        candidate_frames = _mask_frames(mask_track_dir) if os.path.isdir(mask_track_dir) else set(real.keys())
        candidate_frames |= set(real.keys())
        if not extrapolate:
            candidate_frames = {f for f in candidate_frames
                                if first_real <= f <= last_real}

        betas_override = _track_median_betas(real) if betas == "fixed" else None

        n_real = 0
        n_filled = 0
        n_extrap = 0
        n_gap_too_large = 0

        for f in sorted(candidate_frames):
            out_path = os.path.join(out_track, f"{f:06d}.json")
            if f in real:
                rec = dict(real[f])
                if betas_override is not None:
                    rec["mano"] = {**rec["mano"], "betas": list(betas_override)}
                rec["interpolated"] = False
                # Ensure track_id is present (align_hands writes it but
                # be defensive in case of upstream variation).
                rec.setdefault("track_id", int(tid))
                rec.setdefault("frame_idx", f)
                # Canonicalize handedness to the per-track majority. Per-frame
                # WiLoR classifications can disagree (and often the ref frame
                # is the outlier); we honor the sequence-level vote here so
                # downstream consumers see a single is_right per track.
                rec["is_right"] = canonical_is_right
                with open(out_path, "w") as fh:
                    json.dump(rec, fh, indent=2)
                n_real += 1
                continue

            prev_real = max((r for r in sorted_real_idx if r < f), default=None)
            next_real = min((r for r in sorted_real_idx if r > f), default=None)
            if prev_real is None and next_real is None:
                continue
            if prev_real is None or next_real is None:
                # Extrapolation: clamp to the nearest real record. cam_t and
                # MANO pose are held constant — better than nothing for
                # downstream consumers that need every-frame coverage.
                nearest = prev_real if prev_real is not None else next_real
                rec = dict(real[nearest])
                if betas_override is not None:
                    rec["mano"] = {**rec["mano"], "betas": list(betas_override)}
                rec["track_id"]     = int(tid)
                rec["frame_idx"]    = f
                rec["is_right"]     = canonical_is_right
                rec["interpolated"] = True
                rec["interpolation"] = {"prev": int(nearest), "next": int(nearest),
                                        "weight": 0.0}
                with open(out_path, "w") as fh:
                    json.dump(rec, fh, indent=2)
                n_extrap += 1
                continue
            gap = next_real - prev_real
            if gap > max_gap_frames:
                n_gap_too_large += 1
                continue
            t = (f - prev_real) / float(gap)
            rec = _interp_aligned_record(real[prev_real], real[next_real], t, betas_override)
            rec["track_id"]    = int(tid)
            rec["frame_idx"]   = f
            rec["is_right"]    = canonical_is_right
            rec["interpolated"] = True
            rec["interpolation"] = {"prev": prev_real, "next": next_real,
                                    "weight": float(t)}
            with open(out_path, "w") as fh:
                json.dump(rec, fh, indent=2)
            n_filled += 1

        summary.append((tid, n_real, n_filled, n_extrap, n_gap_too_large))
        print(f"  track {tid}: real={n_real}, filled={n_filled}, "
              f"extrapolated={n_extrap}, skipped (gap>{max_gap_frames})={n_gap_too_large}")

    print()
    print("Summary (track_id, real, filled, extrapolated, skipped_gap):")
    for tid, nr, nf, nx, ng in summary:
        print(f"  {tid}: {nr:>5} real  +{nf:>5} filled  +{nx:>5} extrap   ({ng} skipped)")

    handedness_path = os.path.join(output_dir, "handedness.json")
    with open(handedness_path, "w") as f:
        json.dump(handedness, f, indent=2)
    print(f"\nSaved per-track canonical handedness → {handedness_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned_dir",    required=True)
    parser.add_argument("--masks_dir",      required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--betas",          default="fixed", choices=("fixed", "interp"),
                        help="Shape (betas) policy: 'fixed' = median per track, "
                             "'interp' = per-frame linear (default: fixed).")
    parser.add_argument("--max_gap_frames", type=int, default=100000,
                        help="Skip filling when the bracket of real "
                             "detections is wider than this many frames. "
                             "Default effectively unlimited.")
    parser.add_argument("--no_extrapolate", action="store_true",
                        help="Disable clamp-extrapolation outside "
                             "[first_real, last_real]. With this set, frames "
                             "before/after the bracket are not filled.")
    args = parser.parse_args()
    tracks_interpolate(
        aligned_dir    = args.aligned_dir,
        masks_dir      = args.masks_dir,
        output_dir     = args.output_dir,
        betas          = args.betas,
        max_gap_frames = args.max_gap_frames,
        extrapolate    = not args.no_extrapolate,
    )


if __name__ == "__main__":
    main()
