# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
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
import numpy as np

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.common.video import FrameSource, FrameWriter
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


def image_list_to_depth(
    left_dir: str,
    right_dir: str,
    depth_folder: str,
    intrinsics_folder: str,
    calibration: dict,
    model_dir: str,
    start_idx: int = 0,
    end_idx: int | None = None,
    scale: float = 1.0,
):
    """Process image pairs from left_dir / right_dir and write outputs.

    start_idx / end_idx slice the sorted file list, enabling parallel workers
    to process non-overlapping frame ranges concurrently.

    scale: if != 1.0, images are resized before inference and outputs are at
    the scaled resolution. The calibration dict should already reflect the
    scaled intrinsics.

    ``depth_folder`` is auto-detected: if it ends in ``.h5``, depth maps are
    packed into an HDF5 file (uint16 inverse-depth).  Otherwise they are
    written as individual PNGs in a directory.
    """
    os.makedirs(intrinsics_folder, exist_ok=True)

    fx = float(calibration['fx'])
    fy = float(calibration['fy'])
    cx = float(calibration['cx'])
    cy = float(calibration['cy'])
    baseline = float(calibration['baseline'])

    inference = _get_inference(model_dir)

    left_source = FrameSource.from_path(left_dir)
    right_source = FrameSource.from_path(right_dir)

    if left_source.n_frames != right_source.n_frames:
        print(f"  [warn] frame count mismatch: left={left_source.n_frames}, "
              f"right={right_source.n_frames}")

    right_stem_to_idx = {s: i for i, s in enumerate(right_source.stems)}
    left_stems = left_source.stems[start_idx:end_idx]

    is_png_output = Path(depth_folder).suffix.lower() not in (".h5", ".hdf5")
    depth_writer = FrameWriter.from_path(depth_folder)
    processed = 0
    skipped = 0

    try:
        for left_idx, stem in enumerate(left_stems, start=start_idx):
            if is_png_output:
                out_path = os.path.join(depth_folder, f"{stem}.png")
                if os.path.exists(out_path):
                    skipped += 1
                    continue

            if stem not in right_stem_to_idx:
                print(f"  [skip] no matching right frame for: {stem}")
                skipped += 1
                continue

            right_idx = right_stem_to_idx[stem]

            try:
                left_image = left_source[left_idx]
                right_image = right_source[right_idx]
            except Exception as e:
                print(f"  [skip] failed to read frame {stem}: {e}")
                skipped += 1
                continue

            left_bgr = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
            right_bgr = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)

            if scale != 1.0:
                h, w = left_bgr.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                left_bgr = cv2.resize(left_bgr, (new_w, new_h), interpolation=interp)
                right_bgr = cv2.resize(right_bgr, (new_w, new_h), interpolation=interp)

            disparity_px, _ = inference.infer(left_bgr, right_bgr)
            depth_m = disparity_to_depth(disparity_px, fx, baseline)

            depth_img = DepthImage(depth=depth_m)
            depth_uint16 = np.array(depth_img.to_pil_image(), dtype=np.uint16)
            depth_writer.write_frame(depth_uint16, stem=stem)

            h, w = left_bgr.shape[:2]
            intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
            intrinsics.save(os.path.join(intrinsics_folder, f"{stem}.json"))

            processed += 1
            if processed % 10 == 0:
                print(f"  processed {processed} pairs...")
    finally:
        depth_writer.close()

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
