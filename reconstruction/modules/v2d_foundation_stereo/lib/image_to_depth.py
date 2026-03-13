"""Foundation Stereo: process a single stereo image pair to a depth map.

Usage:
    python -m v2d.foundation_stereo.lib.image_to_depth \
        --left_image_path /data/left/frame.jpg \
        --right_image_path /data/right/frame.jpg \
        --depth_path /data/depth/frame.png \
        --intrinsics_path /data/intrinsics/frame.json \
        --calibration_file /data/calibration.json \
        --model_dir /data/models
"""

import argparse
import json
import os

import cv2

from v2d.datatypes import CameraIntrinsics, DepthImage
from v2d.foundation_stereo.lib.export_engine import ensure_engine
from v2d.foundation_stereo.lib.trt_inference import FoundationStereoInference, disparity_to_depth

_inference: FoundationStereoInference | None = None


def _get_inference(model_dir: str) -> FoundationStereoInference:
    global _inference
    if _inference is None:
        engine_path = ensure_engine(model_dir)
        _inference = FoundationStereoInference(engine_path)
    return _inference


def _load_calibration(args) -> dict:
    if args.calibration_file:
        with open(args.calibration_file) as f:
            return json.load(f)
    return {
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy,
        'baseline': args.baseline,
    }


def image_to_depth(
    left_image_path: str,
    right_image_path: str,
    depth_path: str,
    intrinsics_path: str,
    calibration: dict,
    model_dir: str,
):
    left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
    if left_image is None:
        raise FileNotFoundError(f"Cannot read left image: {left_image_path}")
    if right_image is None:
        raise FileNotFoundError(f"Cannot read right image: {right_image_path}")

    inference = _get_inference(model_dir)
    disparity_px, _ = inference.infer(left_image, right_image)

    fx = float(calibration['fx'])
    fy = float(calibration['fy'])
    cx = float(calibration['cx'])
    cy = float(calibration['cy'])
    baseline_m = float(calibration['baseline'])

    depth_m = disparity_to_depth(disparity_px, fx, baseline_m)

    os.makedirs(os.path.dirname(depth_path) or '.', exist_ok=True)
    DepthImage(depth=depth_m).to_pil_image().save(depth_path)

    os.makedirs(os.path.dirname(intrinsics_path) or '.', exist_ok=True)
    h, w = left_image.shape[:2]
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics.to_dict(), f, indent=2)

    print(f"Saved depth: {depth_path}")
    print(f"Saved intrinsics: {intrinsics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Foundation Stereo: single stereo pair to depth"
    )
    parser.add_argument('--left_image_path', required=True)
    parser.add_argument('--right_image_path', required=True)
    parser.add_argument('--depth_path', required=True, help='Output depth PNG path')
    parser.add_argument('--intrinsics_path', required=True, help='Output intrinsics JSON path')

    cal_group = parser.add_mutually_exclusive_group(required=True)
    cal_group.add_argument('--calibration_file',
                           help='JSON file with keys: fx, fy, cx, cy, baseline')
    cal_group.add_argument('--fx', type=float)

    parser.add_argument('--fy', type=float)
    parser.add_argument('--cx', type=float)
    parser.add_argument('--cy', type=float)
    parser.add_argument('--baseline', type=float)

    parser.add_argument('--model_dir', required=True,
                        help='Directory containing ONNX/engine files')

    args = parser.parse_args()

    if args.fx is not None:
        for attr in ('fy', 'cx', 'cy', 'baseline'):
            if getattr(args, attr) is None:
                parser.error(f"--{attr} is required when using individual calibration args")

    image_to_depth(
        left_image_path=args.left_image_path,
        right_image_path=args.right_image_path,
        depth_path=args.depth_path,
        intrinsics_path=args.intrinsics_path,
        calibration=_load_calibration(args),
        model_dir=args.model_dir,
    )


if __name__ == '__main__':
    main()
