"""Grounding DINO: detect objects in every frame of a video using a text prompt.

Usage:
    python -m v2d.grounding_dino.lib.video_to_object_bboxes \
        --video_path /data/video.mp4 \
        --output_path /data/bboxes.json \
        --prompt "robot arm" \
        --model_dir /data/models
"""

import argparse
import json
import os
import subprocess

import cv2
import groundingdino
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict
import numpy as np
from PIL import Image
import torch
from torchvision.ops import box_convert

from v2d.datatypes import BoundingBox

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
    """Convert a BGR OpenCV frame to a normalized RGB tensor for Grounding DINO."""
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
    """Run Grounding DINO on every frame of a video and write a single JSON."""
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


def main():
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
    parser.add_argument('--model_dir', required=True,
                        help='Directory with groundingdino_swint_ogc.pth')
    parser.add_argument('--debug_output', type=str, default=None,
                        help='Directory to save annotated debug frames')

    args = parser.parse_args()

    results = video_to_object_bboxes(
        video_path=args.video_path,
        output_path=args.output_path,
        prompt=args.prompt,
        model_dir=args.model_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if args.debug_output:
        from v2d.grounding_dino.lib.vis import visualize_video_bboxes
        visualize_video_bboxes(args.video_path, results, args.debug_output)


if __name__ == '__main__':
    main()
