# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Per-track per-frame WiLoR reconstruction driven by SAM2 hand masks.

Mirrors ``v2d.hamer.lib.masks_to_hands`` so the two backends are A/B-swappable
behind the same call sites. The mask gives us a centroid and a tight bbox; we
derive a square bbox via ``mask_side * bbox_expansion``, then call WiLoR's
external-bbox API on that single crop with ``is_right`` fixed by the track.

Output layout:
    <output_dir>/<track_id>/<frame_id:06d>.json

JSON schema (per file, single detection per track per frame): same fields as
``image_to_hands.py`` minus the list wrapper, plus ``track_id`` and ``frame_idx``.

Frames where the mask is below ``mask_min_pixels`` (or absent) get no output.

Usage:
    python -m v2d.wilor.lib.masks_to_hands \\
        --frames_dir  /data/frames \\
        --masks_dir   /data/masks \\
        --tracks_path /data/hand_tracks.json \\
        --output_dir  /data/wilor \\
        --weights_dir /data/weights/wilor
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from v2d.common.datatypes import BoundingBox
from v2d.wilor.lib._wilor import get_pipeline, run_wilor_on_bboxes


def _mask_centroid_and_size(mask: np.ndarray) -> tuple[float, float, float, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0, 0.0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    side = float(max(x1 - x0, y1 - y0))
    return cx, cy, side, int(xs.size)


def masks_to_hands(
    frames_dir: str,
    masks_dir: str,
    tracks_path: str,
    output_dir: str,
    weights_dir: str,
    bbox_expansion: float = 1.7,
    mask_min_pixels: int = 256,
) -> None:
    with open(tracks_path) as f:
        tracks = json.load(f)["tracks"]
    hand_tracks = [t for t in tracks if t.get("role", "hand") == "hand"]
    if not hand_tracks:
        raise RuntimeError(f"No hand tracks in {tracks_path}")

    print(f"Loading WiLoR from cache at {weights_dir}...")
    get_pipeline(weights_dir)

    for track in hand_tracks:
        oid       = int(track["object_id"])
        is_right  = bool(track["is_right"])
        track_dir = os.path.join(output_dir, str(oid))
        os.makedirs(track_dir, exist_ok=True)

        mask_files = sorted(glob.glob(os.path.join(masks_dir, str(oid), "*.png")))
        side_str = "right" if is_right else "left"
        print(f"\nTrack {oid} ({side_str}): {len(mask_files)} mask frames")

        n_written = n_skipped = 0
        for mask_path in tqdm(mask_files, desc=f"  track {oid}", ncols=80, unit="frame"):
            frame_idx = int(os.path.splitext(os.path.basename(mask_path))[0])
            frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
                if not os.path.exists(frame_path):
                    continue
            out_path = os.path.join(track_dir, f"{frame_idx:06d}.json")
            if os.path.exists(out_path):
                continue

            mask = np.asarray(Image.open(mask_path)) > 0
            cx, cy, mask_side, npx = _mask_centroid_and_size(mask)
            if npx < mask_min_pixels:
                n_skipped += 1
                continue

            image = np.asarray(Image.open(frame_path).convert("RGB"))
            H, W = image.shape[:2]
            size = mask_side * bbox_expansion
            x0 = float(np.clip(cx - size / 2, 0.0, W))
            y0 = float(np.clip(cy - size / 2, 0.0, H))
            x1 = float(np.clip(cx + size / 2, 0.0, W))
            y1 = float(np.clip(cy + size / 2, 0.0, H))
            bbox = BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

            recs = run_wilor_on_bboxes(image, [bbox], [is_right],
                                       weights_dir=weights_dir)
            if not recs:
                n_skipped += 1
                continue
            rec = recs[0]
            rec["track_id"]  = oid
            rec["frame_idx"] = frame_idx
            with open(out_path, "w") as f:
                json.dump(rec, f, indent=2)
            n_written += 1

        print(f"  track {oid}: wrote {n_written}, skipped {n_skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir",  required=True)
    parser.add_argument("--masks_dir",   required=True)
    parser.add_argument("--tracks_path", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--bbox_expansion", type=float, default=1.7)
    parser.add_argument("--mask_min_pixels", type=int, default=256)
    args = parser.parse_args()
    masks_to_hands(
        frames_dir      = args.frames_dir,
        masks_dir       = args.masks_dir,
        tracks_path     = args.tracks_path,
        output_dir      = args.output_dir,
        weights_dir     = args.weights_dir,
        bbox_expansion  = args.bbox_expansion,
        mask_min_pixels = args.mask_min_pixels,
    )


if __name__ == "__main__":
    main()
