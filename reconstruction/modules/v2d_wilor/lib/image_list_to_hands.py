"""WiLoR over a folder of images: per-image JSON list in <output_dir>/<stem>.json.

If ``--bboxes_dir`` is supplied, each image's corresponding ``<stem>.json``
under that directory is used as external bbox input. Images without a
matching bbox JSON are skipped.

Output schema per file: see ``image_to_hands.py``.

Usage:
    python -m v2d.wilor.lib.image_list_to_hands \\
        --images_dir  /data/frames \\
        --output_dir  /data/wilor \\
        --weights_dir /data/weights/wilor
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from v2d.common.datatypes import BoundingBox
from v2d.wilor.lib._wilor import (
    get_pipeline, run_wilor_detect, run_wilor_on_bboxes,
)
from v2d.wilor.lib.image_to_hands import _load_external_bboxes


_IMG_EXTS = (".png", ".jpg", ".jpeg")


def image_list_to_hands(
    images_dir: str,
    output_dir: str,
    weights_dir: str,
    bboxes_dir: Optional[str] = None,
) -> None:
    files = sorted(
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if p.lower().endswith(_IMG_EXTS)
    )
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    os.makedirs(output_dir, exist_ok=True)
    # Warm the pipeline once before the loop so the progress bar starts clean.
    get_pipeline(weights_dir)

    for path in tqdm(files, desc="wilor", ncols=80, unit="img"):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{stem}.json")
        if os.path.exists(out_path):
            continue

        image = np.asarray(Image.open(path).convert("RGB"))
        if bboxes_dir is None:
            records = run_wilor_detect(image, weights_dir=weights_dir)
        else:
            bb_path = os.path.join(bboxes_dir, f"{stem}.json")
            if not os.path.exists(bb_path):
                continue
            bboxes, is_right = _load_external_bboxes(bb_path)
            if not bboxes:
                records = []
            else:
                records = run_wilor_on_bboxes(image, bboxes, is_right,
                                              weights_dir=weights_dir)

        records.sort(key=lambda d: d["score"], reverse=True)
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_dir",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--bboxes_dir", default=None,
                        help="Optional dir of <stem>.json bboxes; skip WiLoR's detector when set.")
    args = parser.parse_args()
    image_list_to_hands(
        images_dir  = args.images_dir,
        output_dir  = args.output_dir,
        weights_dir = args.weights_dir,
        bboxes_dir  = args.bboxes_dir,
    )


if __name__ == "__main__":
    main()
