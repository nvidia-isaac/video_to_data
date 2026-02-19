"""Grounding DINO: detect objects in every frame of a video using a text prompt.

Takes a video and a text prompt, runs Grounding DINO inference on each frame,
and writes a single JSON file keyed by frame number.

Usage:
    python video_to_object_bboxes.py \
        --video_path /data/video.mp4 \
        --output_path /data/bboxes.json \
        --prompt "robot arm"

Output JSON format:
    {
      "0": [
        {"label": "robot arm", "confidence": 0.87,
         "box": {"x0": 120.5, "y0": 80.2, "x1": 340.8, "y1": 280.4}},
        ...
      ],
      "1": [...],
      ...
    }
    Keys are frame indices (0-based). Boxes are in absolute pixel coordinates,
    sorted by confidence descending.
"""

import argparse
import json
import os
import subprocess

import cv2
import groundingdino
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_convert

from modules.common.datatypes import BoundingBox

_CHECKPOINT_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
_model = None


def _get_model(model_dir: str):
    global _model
    if _model is None:
        config_path = os.path.join(
            groundingdino.__path__[0], 'config', 'GroundingDINO_SwinT_OGC.py'
        )
        model_path = os.path.join(model_dir, 'groundingdino_swint_ogc.pth')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
        if not os.path.exists(model_path):
            print(f"Checkpoint not found at {model_path}. Downloading...")
            os.makedirs(model_dir, exist_ok=True)
            subprocess.run(
                ["wget", "-q", "--show-progress", _CHECKPOINT_URL, "-O", model_path],
                check=True,
            )
        _model = load_model(config_path, model_path)
    return _model


_transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _frame_to_gdino_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR OpenCV frame to a normalized RGB tensor for Grounding DINO.

    Applies the same resize+normalize as groundingdino.util.inference.load_image:
    shortest side → 800px, longest side ≤ 1333px.
    """
    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor, _ = _transform(pil_image, None)
    return tensor


def _detect_frame(model, frame_bgr: np.ndarray, prompt: str,
                  box_threshold: float, text_threshold: float) -> list[dict]:
    h, w = frame_bgr.shape[:2]
    image_tensor = _frame_to_gdino_tensor(frame_bgr)

    boxes_cxcywh, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    boxes_xyxy = box_convert(boxes=boxes_cxcywh, in_fmt='cxcywh', out_fmt='xyxy')
    boxes_xyxy = boxes_xyxy * boxes_xyxy.new_tensor([w, h, w, h])

    detections = []
    for box, logit, phrase in zip(boxes_xyxy.tolist(), logits.tolist(), phrases):
        x0, y0, x1, y1 = box
        detections.append({
            'label': phrase,
            'confidence': round(logit, 4),
            'box': BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1).to_dict(),
        })

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections


def video_to_object_bboxes(
    video_path: str,
    output_path: str,
    prompt: str,
    model_dir: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> dict:
    """Run Grounding DINO on every frame of a video and write a single JSON.

    Args:
        video_path: Path to the input video file.
        output_path: Path to write the combined output JSON file.
        prompt: Text prompt describing objects to detect (e.g. "robot arm").
        model_dir: Directory containing groundingdino_swint_ogc.pth.
        box_threshold: Minimum box confidence score (default 0.35).
        text_threshold: Minimum text confidence score (default 0.25).

    Returns:
        Dict mapping frame index (str) → list of detection dicts, also written
        to output_path.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    model = _get_model(model_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = {}
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        detections = _detect_frame(model, frame_bgr, prompt, box_threshold, text_threshold)
        results[str(frame_idx)] = detections

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames...")

    cap.release()

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Done. processed={frame_idx} frames. Saved to: {output_path}")
    return results


def _debug_dir(output_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(output_path)), 'debug')


def main():
    default_model_dir = os.environ.get(
        'MODEL_DIR',
        os.path.join(os.environ.get('DATA_DIR', '/data'), 'grounding_dino', 'models'),
    )

    parser = argparse.ArgumentParser(
        description="Grounding DINO: detect objects in every frame of a video"
    )
    parser.add_argument('--video_path', required=True, help='Path to input video file')
    parser.add_argument('--output_path', required=True,
                        help='Output JSON file path (keyed by frame index)')
    parser.add_argument('--prompt', required=True,
                        help='Text prompt for object detection (e.g. "robot arm")')
    parser.add_argument('--box_threshold', type=float, default=0.35,
                        help='Minimum box confidence score (default: 0.35)')
    parser.add_argument('--text_threshold', type=float, default=0.25,
                        help='Minimum text confidence score (default: 0.25)')
    parser.add_argument('--model_dir', default=default_model_dir,
                        help=f'Directory with groundingdino_swint_ogc.pth '
                             f'(default: $MODEL_DIR or {default_model_dir})')
    parser.add_argument('--vis', action='store_true',
                        help='Save annotated debug frames alongside output')

    args = parser.parse_args()

    results = video_to_object_bboxes(
        video_path=args.video_path,
        output_path=args.output_path,
        prompt=args.prompt,
        model_dir=args.model_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if args.vis:
        from modules.grounding_dino.vis import visualize_video_bboxes
        visualize_video_bboxes(args.video_path, results, _debug_dir(args.output_path))


if __name__ == '__main__':
    main()
