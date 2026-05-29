# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Gap-fill per-track *pre-align* wilor records by interpolating between real frames.

Consumes the camera-frame wilor schema written by ``tracks_from_wilor_masks``:
  wilor_dir/<track_id>/<frame:06d>.json
with fields:
  is_right, score, bbox, mano: {betas, global_orient, hand_pose},
  camera: {pred_cam_t_full, scaled_focal_length}, image_size,
  match_iou, track_id, frame_idx.

Why pre-align?
  We want a wilor record for every frame the hand is visible (SAM2 has a
  mask) so that the silhouette-intersect mask refinement step can produce
  refined masks for *every* such frame. Alignment itself still consumes
  the real-only wilor records (interpolated guesses shouldn't be aligned
  against real depth — that's the post-align interpolator's job).

Inputs:
  wilor_dir/<track_id>/<frame:06d>.json   per-track wilor records (real only)
  masks_dir/<track_id>/<frame:06d>.png    SAM2 propagated track masks
                                          (gates which missing frames
                                          deserve interpolation)

Output:
  output_dir/<track_id>/<frame:06d>.json  same schema as input plus
    "interpolated": bool                    False on real, True on filled
    "interpolation": {"prev","next","weight"}     only on filled frames

Algorithm per track:
  1. Load all real records; aggregate canonical handedness (whole-sequence
     uniform majority + position tiebreak).
  2. Discover SAM2 mask frames in [first_real, last_real]; these are the
     candidate frames.
  3. Real frames pass through with ``interpolated: false`` and the
     canonical ``is_right`` applied. With ``--betas fixed`` (default),
     per-frame ``betas`` is overwritten by the median per track.
  4. For each missing candidate frame, find bracketing real neighbours.
     Skip if the bracket gap exceeds ``--max_gap_frames``. Otherwise:
       * SLERP global_orient + each of 15 hand_pose joints,
       * linearly interpolate ``camera.pred_cam_t_full``,
       * linearly interpolate ``bbox``,
       * carry ``camera.scaled_focal_length`` from a neighbour
         (sequence-constant in WiLoR), and
       * mark ``match_iou=0.0`` (no real match available at this frame).
  5. No extrapolation outside [first_real, last_real].

Usage:
    python -m v2d.wilor.lib.wilor_tracks_interpolate \\
        --wilor_dir  /data/wilor \\
        --masks_dir  /data/masks \\
        --output_dir /data/wilor_filled \\
        --betas fixed --max_gap_frames 15
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional

import numpy as np

# Shared helpers live in the aligned-schema interpolator; reuse them to keep
# rotation math + handedness aggregation in one place.
from v2d.wilor.lib.tracks_interpolate import (
    _aggregate_handedness,
    _mask_frames,
    _real_frames,
    _slerp_axisangle,
    _track_median_betas,
)


def _interp_wilor_record(
    prev: dict, nxt: dict, t: float, betas_override: Optional[list],
) -> dict:
    """Blend two real wilor records at fraction ``t`` ∈ [0, 1].

    Returns a record in the wilor (pre-align) schema; caller adds
    ``track_id`` / ``frame_idx`` / ``interpolated`` / ``interpolation``.
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

    c0 = np.array(prev["camera"]["pred_cam_t_full"], dtype=np.float64)
    c1 = np.array(nxt ["camera"]["pred_cam_t_full"], dtype=np.float64)
    pred_cam_t_full = ((1 - t) * c0 + t * c1).tolist()
    f0 = float(prev["camera"]["scaled_focal_length"])
    f1 = float(nxt ["camera"]["scaled_focal_length"])
    scaled_focal = (1 - t) * f0 + t * f1

    bb0 = prev.get("bbox") or {}
    bb1 = nxt .get("bbox") or {}
    if bb0 and bb1:
        bbox = {k: (1 - t) * float(bb0[k]) + t * float(bb1[k])
                for k in ("x0", "y0", "x1", "y1")}
    else:
        bbox = bb0 or bb1

    out: dict = {
        "is_right":   prev["is_right"],
        "score":      0.0,
        "bbox":       bbox,
        "mano": {
            "betas":         betas,
            "global_orient": global_orient,
            "hand_pose":     hand_pose,
        },
        "camera": {
            "pred_cam_t_full":     pred_cam_t_full,
            "scaled_focal_length": float(scaled_focal),
        },
        "image_size": prev.get("image_size") or nxt.get("image_size"),
        "match_iou":  0.0,
    }
    return out


def wilor_tracks_interpolate(
    wilor_dir: str,
    masks_dir: str,
    output_dir: str,
    betas: str = "fixed",
    max_gap_frames: int = 100000,
    extrapolate: bool = True,
) -> None:
    """Fill gaps in per-track wilor records (pre-align schema).

    ``extrapolate=True`` (default) clamps to the nearest real record for
    frames outside ``[first_real, last_real]``; ``False`` restricts the
    output to that bracket. ``max_gap_frames`` caps interior gap size
    (default effectively unlimited).
    """
    if betas not in ("fixed", "interp"):
        raise ValueError(f"--betas must be 'fixed' or 'interp', got {betas!r}")

    track_dirs = sorted(
        d for d in glob.glob(os.path.join(wilor_dir, "*"))
        if os.path.isdir(d)
    )
    if not track_dirs:
        raise FileNotFoundError(f"No track subdirs in {wilor_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Whole-sequence canonical handedness (uniform vote + position tiebreak),
    # mirroring the post-align interpolator. Per-frame WiLoR classifications
    # are noisy; canonicalizing here keeps the filled wilor records
    # consistent with what align_hands + tracks_interpolate later canonicalize
    # on the aligned side.
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
            print(f"  track {tid}: no real wilor records — nothing to do")
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
                rec.setdefault("track_id",  int(tid))
                rec.setdefault("frame_idx", f)
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
                # Clamp-extrapolate from the nearest real record.
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
            rec = _interp_wilor_record(real[prev_real], real[next_real], t, betas_override)
            rec["track_id"]     = int(tid)
            rec["frame_idx"]    = f
            rec["is_right"]     = canonical_is_right
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
    parser.add_argument("--wilor_dir",      required=True)
    parser.add_argument("--masks_dir",      required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--betas",          default="fixed", choices=("fixed", "interp"))
    parser.add_argument("--max_gap_frames", type=int, default=100000)
    parser.add_argument("--no_extrapolate", action="store_true")
    args = parser.parse_args()
    wilor_tracks_interpolate(
        wilor_dir      = args.wilor_dir,
        masks_dir      = args.masks_dir,
        output_dir     = args.output_dir,
        betas          = args.betas,
        max_gap_frames = args.max_gap_frames,
        extrapolate    = not args.no_extrapolate,
    )


if __name__ == "__main__":
    main()
