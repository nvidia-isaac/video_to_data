"""Check that SAM2 object masks are contained within labeled bboxes.

For each camera, derives the tight bounding box of the SAM2 mask (object 0,
first frame) and checks what fraction of it falls inside the labeled bbox
(expanded by ``--bbox_padding`` of its size for wiggle room).  The metric is
``containment = intersection_area / mask_bbox_area``.  A value of 1.0 means
the mask bbox is fully inside the padded label bbox.  This tolerates occlusion
(mask smaller than label) and objects with negative space.

Exits non-zero if the average containment across cameras is below
``--min_containment``, which causes the OSMO task to FAIL and blocks
downstream tasks (e.g. foundation_pose).

Writes ``check_object_mask.json`` with per-camera containment values.

Usage (inside container):
    python -m v2d.mv.postprocess.lib.check_object_mask \
        --mask_dir /data/sam2_object_masks \
        --labeled_bbox_dir /data/preprocess/labeled_bboxes \
        --output_dir /data/check_object_mask \
        --min_containment 0.8 \
        --bbox_padding 0.1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

from v2d.common.video import FrameSource


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) tight bbox around foreground, or None if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y0, y1 = int(np.argmax(rows)), int(mask.shape[0] - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(mask.shape[1] - np.argmax(cols[::-1]))
    return x0, y0, x1, y1


def _pad_bbox(
    bbox: tuple[float, float, float, float], padding: float,
) -> tuple[float, float, float, float]:
    """Expand bbox by *padding* fraction of its width/height on each side."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    dx, dy = w * padding, h * padding
    return (x0 - dx, y0 - dy, x1 + dx, y1 + dy)


def _containment(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """Fraction of *inner* bbox area that overlaps with *outer* bbox."""
    ix0 = max(inner[0], outer[0])
    iy0 = max(inner[1], outer[1])
    ix1 = min(inner[2], outer[2])
    iy1 = min(inner[3], outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / inner_area if inner_area > 0 else 0.0


def check_object_mask(
    mask_dir: str,
    labeled_bbox_dir: str,
    output_dir: str,
    min_containment: float = 0.8,
    bbox_padding: float = 0.1,
) -> dict:
    bbox_jsons = sorted(glob.glob(os.path.join(labeled_bbox_dir, "*.json")))

    per_camera: dict[str, float] = {}
    for bbox_path in bbox_jsons:
        cam_name = os.path.splitext(os.path.basename(bbox_path))[0]

        with open(bbox_path) as f:
            bbox_data: dict = json.load(f)
        if not bbox_data:
            continue
        first_frame_key = sorted(bbox_data.keys())[0]
        detections = bbox_data[first_frame_key]
        if not detections:
            continue
        bbox = detections[0]["box"]

        # Try HDF5 first, then PNG directory
        obj_mask_h5 = os.path.join(mask_dir, cam_name, "0.h5")
        obj_mask_dir = os.path.join(mask_dir, cam_name, "0")
        if os.path.isfile(obj_mask_h5):
            mask_source_path = obj_mask_h5
        elif os.path.isdir(obj_mask_dir):
            mask_source_path = obj_mask_dir
        else:
            print(f"  {cam_name}: no mask data at {obj_mask_dir}, skipping")
            continue

        try:
            mask_source = FrameSource.from_path(mask_source_path)
        except FileNotFoundError:
            print(f"  {cam_name}: no mask frames found, skipping")
            continue

        sam_mask = mask_source[0] > 127
        mask_bbox = _mask_to_bbox(sam_mask)
        if mask_bbox is None:
            print(f"  {cam_name}: empty SAM2 mask, skipping")
            continue

        label_box = (bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
        padded_label = _pad_bbox(label_box, bbox_padding)
        cont = _containment(mask_bbox, padded_label)
        per_camera[cam_name] = round(cont, 4)
        print(f"  {cam_name}: containment = {cont:.4f}")

    if not per_camera:
        avg_cont = 0.0
        print("WARNING: no valid camera comparisons")
    else:
        avg_cont = sum(per_camera.values()) / len(per_camera)

    status = "FAIL" if avg_cont < min_containment else "PASS"
    reason = (
        f"avg containment {avg_cont:.3f} < threshold {min_containment}"
        if status == "FAIL"
        else ""
    )

    decision = {
        "status": status,
        "reason": reason,
        "avg_containment": round(avg_cont, 4),
        "per_camera_containment": per_camera,
        "threshold": min_containment,
        "bbox_padding": bbox_padding,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "check_object_mask.json")
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2)

    print(f"Mask QC: {status}  (avg containment {avg_cont:.3f}, threshold {min_containment})")
    if status == "FAIL":
        sys.exit(1)

    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gate: SAM2 object mask containment within labeled bbox"
    )
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="SAM2 object mask output directory")
    parser.add_argument("--labeled_bbox_dir", type=str, required=True,
                        help="Directory with {cam_name}.json labeled bboxes")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_containment", type=float, default=0.8,
                        help="Minimum avg containment ratio (0-1)")
    parser.add_argument("--bbox_padding", type=float, default=0.1,
                        help="Fractional padding applied to labeled bbox")
    args = parser.parse_args()

    check_object_mask(
        mask_dir=args.mask_dir,
        labeled_bbox_dir=args.labeled_bbox_dir,
        output_dir=args.output_dir,
        min_containment=args.min_containment,
        bbox_padding=args.bbox_padding,
    )
