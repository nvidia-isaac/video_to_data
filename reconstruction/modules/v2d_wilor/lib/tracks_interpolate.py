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
    return {
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


def tracks_interpolate(
    aligned_dir: str,
    masks_dir: str,
    output_dir: str,
    betas: str = "fixed",
    max_gap_frames: int = 15,
) -> None:
    if betas not in ("fixed", "interp"):
        raise ValueError(f"--betas must be 'fixed' or 'interp', got {betas!r}")

    track_dirs = sorted(
        d for d in glob.glob(os.path.join(aligned_dir, "*"))
        if os.path.isdir(d)
    )
    if not track_dirs:
        raise FileNotFoundError(f"No track subdirs in {aligned_dir}")

    os.makedirs(output_dir, exist_ok=True)

    summary: list[tuple[str, int, int, int]] = []

    for track_dir in track_dirs:
        tid = os.path.basename(track_dir)
        out_track = os.path.join(output_dir, tid)
        os.makedirs(out_track, exist_ok=True)

        real = _real_frames(track_dir)
        if not real:
            print(f"  track {tid}: no real aligned records — nothing to do")
            summary.append((tid, 0, 0, 0))
            continue
        sorted_real_idx = sorted(real.keys())
        first_real, last_real = sorted_real_idx[0], sorted_real_idx[-1]

        mask_track_dir = os.path.join(masks_dir, tid)
        candidate_frames = _mask_frames(mask_track_dir) if os.path.isdir(mask_track_dir) else set(real.keys())
        candidate_frames |= set(real.keys())
        candidate_frames = {f for f in candidate_frames if first_real <= f <= last_real}

        betas_override = _track_median_betas(real) if betas == "fixed" else None

        n_real = 0
        n_filled = 0
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
                with open(out_path, "w") as fh:
                    json.dump(rec, fh, indent=2)
                n_real += 1
                continue

            prev_real = max((r for r in sorted_real_idx if r < f), default=None)
            next_real = min((r for r in sorted_real_idx if r > f), default=None)
            if prev_real is None or next_real is None:
                continue
            gap = next_real - prev_real
            if gap > max_gap_frames:
                n_gap_too_large += 1
                continue
            t = (f - prev_real) / float(gap)
            rec = _interp_aligned_record(real[prev_real], real[next_real], t, betas_override)
            rec["track_id"]    = int(tid)
            rec["frame_idx"]   = f
            rec["interpolated"] = True
            rec["interpolation"] = {"prev": prev_real, "next": next_real,
                                    "weight": float(t)}
            with open(out_path, "w") as fh:
                json.dump(rec, fh, indent=2)
            n_filled += 1

        summary.append((tid, n_real, n_filled, n_gap_too_large))
        print(f"  track {tid}: real={n_real}, filled={n_filled}, "
              f"skipped (gap>{max_gap_frames})={n_gap_too_large}")

    print()
    print("Summary (track_id, real, filled, skipped_gap):")
    for tid, nr, nf, ng in summary:
        print(f"  {tid}: {nr:>5} real  +{nf:>5} filled   ({ng} skipped)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned_dir",    required=True)
    parser.add_argument("--masks_dir",      required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--betas",          default="fixed", choices=("fixed", "interp"),
                        help="Shape (betas) policy: 'fixed' = median per track, "
                             "'interp' = per-frame linear (default: fixed).")
    parser.add_argument("--max_gap_frames", type=int, default=15,
                        help="Skip filling when the bracket of real "
                             "detections is wider than this many frames.")
    args = parser.parse_args()
    tracks_interpolate(
        aligned_dir    = args.aligned_dir,
        masks_dir      = args.masks_dir,
        output_dir     = args.output_dir,
        betas          = args.betas,
        max_gap_frames = args.max_gap_frames,
    )


if __name__ == "__main__":
    main()
