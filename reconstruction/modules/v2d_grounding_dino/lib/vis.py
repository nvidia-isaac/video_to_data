# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visualization utilities for Grounding DINO bounding box outputs.

Draws detected bounding boxes on images/video frames and saves debug images.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

# BGR colors cycled per label for visual variety
_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 165, 0),  # orange
]


def _color_for(label: str) -> tuple:
    return _COLORS[hash(label) % len(_COLORS)]


def _draw_detections(image_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image. Returns annotated copy."""
    out = image_bgr.copy()
    for det in detections:
        box = det['box']
        x0, y0, x1, y1 = int(box['x0']), int(box['y0']), int(box['x1']), int(box['y1'])
        label = det['label']
        conf = det['confidence']
        color = _color_for(label)

        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness=2)

        text = f"{label} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty = max(y0 - 6, th + baseline)
        cv2.rectangle(out, (x0, ty - th - baseline), (x0 + tw, ty + baseline), color, -1)
        cv2.putText(out, text, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return out


def visualize_image_bboxes(image_path: str, detections: list[dict], debug_dir: str):
    """Draw detections on a single image and save to debug_dir.

    Args:
        image_path: Path to the source image.
        detections: List of detection dicts from image_to_object_bboxes.
        debug_dir: Directory to save the annotated image.
    """
    os.makedirs(debug_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    annotated = _draw_detections(image, detections)
    stem = Path(image_path).stem
    out_path = os.path.join(debug_dir, f"{stem}.jpg")
    cv2.imwrite(out_path, annotated)
    return out_path


def visualize_image_list_bboxes(rgb_path: str, results: dict, debug_dir: str):
    """Draw detections on each image and save to debug_dir.

    Args:
        rgb_path: Path to source frames (image dir, .h5, or video file).
        results: Dict mapping image stem → list of detection dicts
                 (output of image_list_to_object_bboxes).
        debug_dir: Directory to save annotated images.
    """
    from v2d.common.video import FrameSource

    os.makedirs(debug_dir, exist_ok=True)
    source = FrameSource.from_path(rgb_path)
    stem_to_idx = {s: i for i, s in enumerate(source.stems)}

    written = 0
    for stem, detections in results.items():
        if stem not in stem_to_idx:
            continue
        image = source[stem_to_idx[stem]]
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated = _draw_detections(image_bgr, detections)
        out_path = os.path.join(debug_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, annotated)
        written += 1
    print(f"  [vis] saved {written} annotated images to: {debug_dir}")


def visualize_video_bboxes(video_path: str, results: dict, debug_dir: str):
    """Draw detections on each frame of a video and save annotated frames to debug_dir.

    Args:
        video_path: Path to the source video file.
        results: Dict mapping frame index (str) → list of detection dicts
                 (output of video_to_object_bboxes).
        debug_dir: Directory to save annotated frame images.
    """
    os.makedirs(debug_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    written = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        detections = results.get(str(frame_idx), [])
        annotated = _draw_detections(frame_bgr, detections)
        out_path = os.path.join(debug_dir, f"{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, annotated)
        frame_idx += 1
        written += 1

    cap.release()
    print(f"  [vis] saved {written} annotated frames to: {debug_dir}")
