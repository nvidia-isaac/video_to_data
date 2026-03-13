"""Grounding DINO: detect objects in a single image using a text prompt.

Usage:
    python -m v2d.grounding_dino.lib.image_to_object_bboxes \
        --image_path /data/frame.jpg \
        --output_path /data/bboxes/frame.json \
        --prompt "robot arm" \
        --model_dir /data/models
"""

import argparse
import json
import os
import subprocess

import groundingdino
from groundingdino.util.inference import load_image, load_model, predict
from torchvision.ops import box_convert

from v2d.common.datatypes import BoundingBox

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


def image_to_bboxes(
    image_path: str,
    output_path: str,
    prompt: str,
    model_dir: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> list[dict]:
    """Run Grounding DINO on a single image and write detections to JSON."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = _get_model(model_dir)
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape

    boxes_cxcywh, logits, phrases = predict(
        model=model,
        image=image,
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

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=2)

    print(f"Detected {len(detections)} object(s). Saved to: {output_path}")
    return detections


def main():
    parser = argparse.ArgumentParser(
        description="Grounding DINO: detect objects in an image using a text prompt"
    )
    parser.add_argument('--image_path', required=True, help='Path to input image')
    parser.add_argument('--output_path', required=True, help='Path to output JSON file')
    parser.add_argument('--prompt', required=True,
                        help='Text prompt for object detection (e.g. "robot arm")')
    parser.add_argument('--box_threshold', type=float, default=0.35,
                        help='Minimum box confidence score (default: 0.35)')
    parser.add_argument('--text_threshold', type=float, default=0.25,
                        help='Minimum text confidence score (default: 0.25)')
    parser.add_argument('--model_dir', required=True,
                        help='Directory with groundingdino_swint_ogc.pth')
    parser.add_argument('--debug_output', type=str, default=None,
                        help='Directory to save annotated debug images')

    args = parser.parse_args()

    detections = image_to_bboxes(
        image_path=args.image_path,
        output_path=args.output_path,
        prompt=args.prompt,
        model_dir=args.model_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if args.debug_output:
        from v2d.grounding_dino.lib.vis import visualize_image_bboxes
        visualize_image_bboxes(args.image_path, detections, args.debug_output)


if __name__ == '__main__':
    main()
