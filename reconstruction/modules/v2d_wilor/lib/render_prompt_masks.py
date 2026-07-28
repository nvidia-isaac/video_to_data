# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render reference-frame WiLoR MANO silhouettes as SAM2 mask prompts.

Inputs:
  wilor_json                                WiLoR detections for one frame
  image_path                                source frame, used for output size
  mano_assets_root                          dir containing models/MANO_RIGHT.pkl

Outputs:
  output_dir/<object_id>/<frame:06d>.png    binary mask prompts for SAM2
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
from PIL import Image

from v2d.wilor.lib.tracks_from_wilor_masks import _mano_layer, _silhouette


def render_wilor_prompt_masks(
    wilor_json: str,
    image_path: str,
    output_dir: str,
    mano_assets_root: str,
    frame_index: int = 0,
    first_object_id: int = 1,
    min_mask_pixels: int = 1,
) -> list[dict]:
    with open(wilor_json) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(f"No WiLoR detections in {wilor_json}")

    W, H = Image.open(image_path).size
    mano = _mano_layer(mano_assets_root)
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    os.makedirs(output_dir, exist_ok=True)

    written: list[dict] = []
    try:
        for i, det in enumerate(detections):
            object_id = int(first_object_id) + i
            mask = _silhouette(det, mano, renderer, W, H)
            n_pixels = int(mask.sum())
            if n_pixels < int(min_mask_pixels):
                raise RuntimeError(
                    f"Rendered WiLoR mask for object_id={object_id} has "
                    f"{n_pixels} pixels (< {min_mask_pixels})"
                )

            track_dir = os.path.join(output_dir, str(object_id))
            os.makedirs(track_dir, exist_ok=True)
            out_path = os.path.join(track_dir, f"{frame_index:06d}.png")
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(out_path)
            written.append({
                "object_id": object_id,
                "path": out_path,
                "pixels": n_pixels,
            })
    finally:
        renderer.delete()

    for rec in written:
        print(
            f"  object_id={rec['object_id']}: wrote {rec['pixels']} px "
            f"mask -> {rec['path']}"
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wilor_json", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--first_object_id", type=int, default=1)
    parser.add_argument("--min_mask_pixels", type=int, default=1)
    args = parser.parse_args()
    render_wilor_prompt_masks(
        wilor_json=args.wilor_json,
        image_path=args.image_path,
        output_dir=args.output_dir,
        mano_assets_root=args.mano_assets_root,
        frame_index=args.frame_index,
        first_object_id=args.first_object_id,
        min_mask_pixels=args.min_mask_pixels,
    )


if __name__ == "__main__":
    main()
