# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Single-image WiLoR: write a JSON list of {bbox, mano, camera} detections.

Detector mode:
    --bboxes_path UNSET  → use WiLoR's internal detector.
    --bboxes_path SET    → consume an external bbox JSON (mediapipe-style),
                           run reconstruction only on those boxes.

JSON schema (list, possibly empty):
    [{
       "is_right": bool,
       "score":    float,
       "bbox":     {"x0":..,"y0":..,"x1":..,"y1":..},
       "mano":     {"betas":[10], "global_orient":[3], "hand_pose":[45]},
       "camera":   {"pred_cam_t_full":[3], "scaled_focal_length": float},
       "image_size": [W, H]
    }]

Usage:
    python -m v2d.wilor.lib.image_to_hands \\
        --image_path  /data/frame.jpg \\
        --output_path /data/hands.json \\
        --weights_dir /data/weights/wilor
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np
from PIL import Image

from v2d.common.datatypes import BoundingBox
from v2d.wilor.lib._wilor import run_wilor_detect, run_wilor_on_bboxes


def _load_external_bboxes(path: str) -> tuple[List[BoundingBox], List[Optional[bool]]]:
    with open(path) as f:
        data = json.load(f)
    bboxes, is_right = [], []
    for d in data:
        b = d["bbox"]
        bboxes.append(BoundingBox(x0=b["x0"], y0=b["y0"], x1=b["x1"], y1=b["y1"]))
        is_right.append(d.get("is_right"))
    return bboxes, is_right


def image_to_hands(
    image_path: str,
    output_path: str,
    weights_dir: str,
    bboxes_path: Optional[str] = None,
) -> List[dict]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = np.asarray(Image.open(image_path).convert("RGB"))

    if bboxes_path is None:
        records = run_wilor_detect(image, weights_dir=weights_dir)
    else:
        bboxes, is_right = _load_external_bboxes(bboxes_path)
        records = run_wilor_on_bboxes(image, bboxes, is_right, weights_dir=weights_dir)

    records.sort(key=lambda d: d["score"], reverse=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Detected {len(records)} hand(s). Saved to: {output_path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_path",  required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--bboxes_path", default=None,
                        help="Optional external bbox JSON; if set, skip WiLoR's detector.")
    args = parser.parse_args()
    image_to_hands(
        image_path  = args.image_path,
        output_path = args.output_path,
        weights_dir = args.weights_dir,
        bboxes_path = args.bboxes_path,
    )


if __name__ == "__main__":
    main()
