"""Per-track per-frame HaMeR regression driven by SAM2 hand masks.

For each (track, frame) pair, this module:
  1. Loads the SAM2 mask, computes its centroid and tight bbox.
  2. Builds a square crop centred at the centroid, sized as
     ``max(mask_bbox_side) * bbox_expansion``  → forgiving to partial masks.
  3. Runs HaMeR on the crop.
  4. Saves a JSON record with MANO params + camera fields.

Output layout:
    <output_dir>/<track_id>/<frame_id:06d>.json

JSON schema (per file, single detection per track per frame):
    {
      "track_id":   int,
      "is_right":   bool,
      "frame_idx":  int,
      "image_size": [W, H],
      "crop":       {"cx": ..., "cy": ..., "size": ...},
      "mano":       {
         "betas":         [10 floats],
         "global_orient": [3 floats],     # axis-angle, converted from rotmat
         "hand_pose":     [45 floats]     # axis-angle, 15×3, converted
      },
      "camera":     {
         "pred_cam_t_full":     [tx, ty, tz],
         "scaled_focal_length": float
      }
    }

Frames where the mask is below ``mask_min_pixels`` (or absent) get no
output file — downstream code should treat missing as "hand not visible."

Usage:
    python -m v2d.hamer.lib.masks_to_hands \\
        --frames_dir /data/frames \\
        --masks_dir  /data/masks \\
        --tracks_path /data/hand_tracks.json \\
        --output_dir /data/hamer \\
        --weights_dir /data/weights/hamer
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

from v2d.hamer.lib._hamer import get_model, run_hamer, rotmat_to_axis_angle


def _mask_centroid_and_size(mask: np.ndarray) -> tuple[float, float, float, int]:
    """Return (cx, cy, side, n_pixels) of the mask's tight bbox."""
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

    print(f"Loading HaMeR from {weights_dir}...")
    model, cfg = get_model(weights_dir)

    for track in hand_tracks:
        oid       = int(track["object_id"])
        is_right  = bool(track["is_right"])
        track_dir = os.path.join(output_dir, str(oid))
        os.makedirs(track_dir, exist_ok=True)

        mask_files = sorted(glob.glob(os.path.join(masks_dir, str(oid), "*.png")))
        side = "right" if is_right else "left"
        print(f"\nTrack {oid} ({side}): {len(mask_files)} mask frames")

        n_written = n_skipped = 0
        for mask_path in tqdm(mask_files, desc=f"  track {oid}", ncols=80, unit="frame"):
            frame_idx = int(os.path.splitext(os.path.basename(mask_path))[0])
            frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.png")
            if not os.path.exists(frame_path):
                # try .jpg
                frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
                if not os.path.exists(frame_path):
                    continue
            out_path = os.path.join(track_dir, f"{frame_idx:06d}.json")
            if os.path.exists(out_path):
                continue   # skip already-done frames

            mask = np.asarray(Image.open(mask_path)) > 0
            cx, cy, mask_side, npx = _mask_centroid_and_size(mask)
            if npx < mask_min_pixels:
                n_skipped += 1
                continue

            image = np.asarray(Image.open(frame_path).convert("RGB"))
            H, W = image.shape[:2]
            size = mask_side * bbox_expansion
            # Clamp center so the crop stays a real bbox (HaMeR's crop helper
            # handles out-of-image padding internally, but keeping center sane
            # avoids silly edge cases).
            cx = float(np.clip(cx, 0.0, W - 1))
            cy = float(np.clip(cy, 0.0, H - 1))

            out = run_hamer(model, cfg, image, cx, cy, size, is_right=is_right)

            record = {
                "track_id":   oid,
                "is_right":   is_right,
                "frame_idx":  frame_idx,
                "image_size": [int(W), int(H)],
                "crop":       {"cx": cx, "cy": cy, "size": float(size)},
                "mano": {
                    "betas":         out["betas"].tolist(),
                    "global_orient": rotmat_to_axis_angle(out["global_orient"]).reshape(-1).tolist(),
                    "hand_pose":     rotmat_to_axis_angle(out["hand_pose"]).reshape(-1).tolist(),
                },
                "camera": {
                    "pred_cam_t_full":     out["pred_cam_t_full"].tolist(),
                    "scaled_focal_length": out["scaled_focal_length"],
                },
            }
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            n_written += 1

        print(f"  track {oid}: wrote {n_written}, skipped {n_skipped} (mask too small)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir",  required=True)
    parser.add_argument("--masks_dir",   required=True,
                        help="Root masks dir; contains <track_id>/*.png subdirs.")
    parser.add_argument("--tracks_path", required=True,
                        help="hand_tracks.json (object_id → is_right mapping).")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--weights_dir", required=True,
                        help="Dir containing _DATA/hamer_ckpts/.")
    parser.add_argument("--bbox_expansion", type=float, default=1.7)
    parser.add_argument("--mask_min_pixels", type=int, default=256)
    args = parser.parse_args()
    masks_to_hands(
        frames_dir       = args.frames_dir,
        masks_dir        = args.masks_dir,
        tracks_path      = args.tracks_path,
        output_dir       = args.output_dir,
        weights_dir      = args.weights_dir,
        bbox_expansion   = args.bbox_expansion,
        mask_min_pixels  = args.mask_min_pixels,
    )


if __name__ == "__main__":
    main()
