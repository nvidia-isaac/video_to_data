"""Foundation Stereo: process a synchronized stereo image directory pair to depth maps.

Takes two directories of images with matching filenames (left_dir and right_dir),
runs Foundation Stereo TensorRT inference on each pair, and writes:
  - {depth_folder}/{stem}.png   -- 16-bit inverse-depth PNG (DepthImage format)
  - {intrinsics_folder}/{stem}.json -- CameraIntrinsics JSON

Camera calibration must be supplied either as a JSON file or as individual args.

Usage:
    python -m v2d.foundation_stereo.lib.image_list_to_depth \
        --left_dir /data/left \
        --right_dir /data/right \
        --depth_folder /data/depth \
        --intrinsics_folder /data/intrinsics \
        --calibration_file /data/calibration.json \
        --model_dir /data/models

    # process a subset of frames (for parallel workers):
    python -m v2d.foundation_stereo.lib.image_list_to_depth ... --start_idx 0 --end_idx 505
    python -m v2d.foundation_stereo.lib.image_list_to_depth ... --start_idx 505 --end_idx 1010
"""

import argparse
import json
import os
from pathlib import Path

import cv2

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.foundation_stereo.lib.export_engine import ensure_engine
from v2d.foundation_stereo.lib.trt_inference import (
    FoundationStereoInference,
    disparity_to_depth,
)

_inference: FoundationStereoInference | None = None


def _get_inference(model_dir: str) -> FoundationStereoInference:
    global _inference
    if _inference is None:
        engine_path = ensure_engine(model_dir)
        _inference = FoundationStereoInference(engine_path)
    return _inference


def _load_calibration(args) -> dict:
    """Return calibration dict with keys: fx, fy, cx, cy, baseline."""
    if args.calibration_file:
        with open(args.calibration_file) as f:
            cal = json.load(f)
        required = {'fx', 'fy', 'cx', 'cy', 'baseline'}
        missing = required - cal.keys()
        if missing:
            raise ValueError(f"calibration_file is missing keys: {missing}")
        return cal

    for attr in ('fx', 'fy', 'cx', 'cy', 'baseline'):
        if getattr(args, attr) is None:
            raise ValueError(
                f"--{attr} is required when --calibration_file is not provided."
            )
    return {
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy,
        'baseline': args.baseline,
    }


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _find_matching(right_dir: str, stem: str) -> str | None:
    """Find a file in right_dir with the same stem (any image extension)."""
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(right_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def image_list_to_depth(
    left_dir: str,
    right_dir: str,
    depth_folder: str,
    intrinsics_folder: str,
    calibration: dict,
    model_dir: str,
    start_idx: int = 0,
    end_idx: int | None = None,
):
    """Process image pairs from left_dir / right_dir and write outputs.

    start_idx / end_idx slice the sorted file list, enabling parallel workers
    to process non-overlapping frame ranges concurrently.
    """
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(intrinsics_folder, exist_ok=True)

    fx = float(calibration['fx'])
    fy = float(calibration['fy'])
    cx = float(calibration['cx'])
    cy = float(calibration['cy'])
    baseline = float(calibration['baseline'])

    inference = _get_inference(model_dir)

    left_files = sorted(
        p for p in Path(left_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    left_files = left_files[start_idx:end_idx]

    if not left_files:
        raise FileNotFoundError(f"No image files found in left_dir: {left_dir}")

    processed = 0
    skipped = 0

    for left_path in left_files:
        stem = left_path.stem

        out_depth = os.path.join(depth_folder, f"{stem}.png")
        if os.path.exists(out_depth):
            skipped += 1
            continue

        right_path = _find_matching(right_dir, stem)
        if right_path is None:
            print(f"  [skip] no matching right image for: {left_path.name}")
            skipped += 1
            continue

        left_image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_image = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if left_image is None:
            print(f"  [skip] failed to read: {left_path}")
            skipped += 1
            continue
        if right_image is None:
            print(f"  [skip] failed to read: {right_path}")
            skipped += 1
            continue

        disparity_px, _ = inference.infer(left_image, right_image)
        depth_m = disparity_to_depth(disparity_px, fx, baseline)

        depth_img = DepthImage(depth=depth_m)
        depth_img.to_pil_image().save(os.path.join(depth_folder, f"{stem}.png"))

        h, w = left_image.shape[:2]
        intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
        intrinsics.save(os.path.join(intrinsics_folder, f"{stem}.json"))

        processed += 1
        if processed % 10 == 0:
            print(f"  processed {processed} pairs...")

    print(f"Done. processed={processed}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Foundation Stereo: image list to depth maps"
    )
    parser.add_argument('--left_dir', required=True,
                        help='Directory of left camera images')
    parser.add_argument('--right_dir', required=True,
                        help='Directory of right camera images (same filenames as left_dir)')
    parser.add_argument('--depth_folder', required=True,
                        help='Output directory for depth PNG files')
    parser.add_argument('--intrinsics_folder', required=True,
                        help='Output directory for camera intrinsics JSON files')

    cal_group = parser.add_mutually_exclusive_group(required=True)
    cal_group.add_argument('--calibration_file',
                           help='JSON file with keys: fx, fy, cx, cy, baseline')
    cal_group.add_argument('--fx', type=float, help='Focal length x (pixels)')

    parser.add_argument('--fy', type=float, help='Focal length y (pixels)')
    parser.add_argument('--cx', type=float, help='Principal point x (pixels)')
    parser.add_argument('--cy', type=float, help='Principal point y (pixels)')
    parser.add_argument('--baseline', type=float, help='Stereo baseline (meters)')

    parser.add_argument('--model_dir', required=True,
                        help='Directory containing ONNX/engine files')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Index of first frame to process, inclusive (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Index of last frame to process, exclusive (default: all)')

    args = parser.parse_args()

    if args.fx is not None:
        for attr in ('fy', 'cx', 'cy', 'baseline'):
            if getattr(args, attr) is None:
                parser.error(f"--{attr} is required when using individual calibration args")

    calibration = _load_calibration(args)

    image_list_to_depth(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        depth_folder=args.depth_folder,
        intrinsics_folder=args.intrinsics_folder,
        calibration=calibration,
        model_dir=args.model_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )


if __name__ == '__main__':
    main()
