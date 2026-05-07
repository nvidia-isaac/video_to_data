"""MediaPipe HandLandmarker (Tasks API): bbox + handedness from a single image.

Uses the newer ``mediapipe.tasks.vision.HandLandmarker`` model, which is more
robust than the legacy ``mediapipe.solutions.hands`` on close-up / non-frontal
hands. The model file ``hand_landmarker.task`` is baked into the docker image
under ``/opt/mediapipe_tasks/``.

Usage:
    python -m v2d.hand_detector.lib.image_to_hand_bboxes \\
        --image_path /data/frame.jpg \\
        --output_path /data/hands/000000.json
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from v2d.common.datatypes import BoundingBox

_DEFAULT_TASK_PATH = "/opt/mediapipe_tasks/hand_landmarker.task"

_DETECTOR = None


def _get_detector(task_path: str, max_num_hands: int, min_detection_confidence: float):
    """Return a singleton HandLandmarker, recreating it on config change."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker, HandLandmarkerOptions, RunningMode,
    )

    global _DETECTOR
    key = (task_path, max_num_hands, min_detection_confidence)
    if _DETECTOR is None or _DETECTOR[0] != key:
        if _DETECTOR is not None:
            _DETECTOR[1].close()
        if not os.path.exists(task_path):
            raise FileNotFoundError(
                f"HandLandmarker task file not found at {task_path}. "
                "Rebuild the v2d_hand_detector image (it downloads the file "
                "at build time)."
            )
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        _DETECTOR = (key, HandLandmarker.create_from_options(options))
    return _DETECTOR[1]


def _bbox_from_landmarks(landmarks, w: int, h: int, pad_ratio: float) -> BoundingBox:
    """Compute a padded axis-aligned bbox from a list of normalized landmarks."""
    xs = np.array([lm.x for lm in landmarks], dtype=np.float64) * w
    ys = np.array([lm.y for lm in landmarks], dtype=np.float64) * h
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    bw, bh = x1 - x0, y1 - y0
    pad = pad_ratio * max(bw, bh)
    return BoundingBox(
        x0=max(0.0,        x0 - pad),
        y0=max(0.0,        y0 - pad),
        x1=min(float(w),   x1 + pad),
        y1=min(float(h),   y1 + pad),
    )


def image_to_hand_bboxes(
    image_path: str,
    output_path: str,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    pad_ratio: float = 0.15,
    selfie: bool = False,
    task_path: str = _DEFAULT_TASK_PATH,
) -> list[dict]:
    """Detect hands in a single image and write JSON list of detections.

    Args:
        image_path:               Input image (PNG/JPG).
        output_path:              Output JSON path.
        max_num_hands:            Maximum hands to detect (default: 2).
        min_detection_confidence: MediaPipe detection threshold (default: 0.5).
        pad_ratio:                Padding around landmark bbox, as a fraction of
                                  the longer side. Hand crops for downstream
                                  reconstructors typically want ~10–20 % pad.
        selfie:                   True if the camera is front-facing (mirrored).
                                  MediaPipe's handedness assumes selfie input;
                                  for non-mirrored footage we flip the label so
                                  ``is_right`` reflects actual hand identity in
                                  the camera frame. Default: False (rear/ego).

    Output JSON schema (list of detections, possibly empty):
        [
          {
            "is_right":   bool,
            "score":      float,                    # MediaPipe handedness conf
            "bbox":       {"x0": ..., "y0": ..., "x1": ..., "y1": ...}
          }
        ]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    import mediapipe as mp

    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]

    detector = _get_detector(task_path, max_num_hands, min_detection_confidence)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = detector.detect(mp_image)

    detections: list[dict] = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        cat = handedness[0]   # one Category per hand
        mp_is_right = (cat.category_name == "Right")
        # MediaPipe's label assumes selfie/mirrored input; flip for ego/rear.
        is_right = mp_is_right if selfie else (not mp_is_right)
        box = _bbox_from_landmarks(landmarks, w, h, pad_ratio)
        detections.append({
            "is_right": bool(is_right),
            "score":    float(cat.score),
            "bbox":     box.to_dict(),
        })

    detections.sort(key=lambda d: d["score"], reverse=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(detections, f, indent=2)

    print(f"Detected {len(detections)} hand(s). Saved to: {output_path}")
    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_path",  required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--pad_ratio", type=float, default=0.15)
    parser.add_argument("--selfie", action="store_true",
                        help="Input is from a selfie/front-facing camera "
                             "(MediaPipe's native handedness convention).")
    parser.add_argument("--task_path", default=_DEFAULT_TASK_PATH,
                        help=f"Path to hand_landmarker.task (default: {_DEFAULT_TASK_PATH})")
    args = parser.parse_args()
    image_to_hand_bboxes(
        image_path               = args.image_path,
        output_path              = args.output_path,
        max_num_hands            = args.max_num_hands,
        min_detection_confidence = args.min_detection_confidence,
        pad_ratio                = args.pad_ratio,
        selfie                   = args.selfie,
        task_path                = args.task_path,
    )


if __name__ == "__main__":
    main()
