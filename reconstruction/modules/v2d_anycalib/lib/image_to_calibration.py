"""AnyCalib: estimate intrinsics + distortion from a single image, optionally
write an undistorted copy of the input.

Usage:
    python -m v2d.anycalib.lib.image_to_calibration \
        --image_path /data/in.jpg \
        --intrinsics_path /data/intrinsics.json \
        --distortion_path /data/distortion.json \
        --weights_path /data/weights \
        [--undistorted_image_path /data/out.jpg \
         --undistorted_intrinsics_path /data/undistorted_intrinsics.json]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from v2d.anycalib.lib._anycalib import predict_calibration
from v2d.anycalib.lib._undistort import undistort_image


def image_to_calibration(
    image_path: str,
    intrinsics_path: str,
    distortion_path: str,
    weights_path: str,
    cam_id: str = "kb:4",
    model_id: str = "anycalib_gen",
    undistorted_image_path: str | None = None,
    undistorted_intrinsics_path: str | None = None,
    balance: float = 0.0,
) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    intrinsics, distortion = predict_calibration(
        image, weights_path=weights_path, cam_id=cam_id, model_id=model_id,
    )

    os.makedirs(os.path.dirname(os.path.abspath(intrinsics_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(distortion_path)), exist_ok=True)
    intrinsics.save(intrinsics_path)
    distortion.save(distortion_path)
    print(f"Saved intrinsics → {intrinsics_path}")
    print(f"Saved distortion ({distortion.model}) → {distortion_path}")

    if undistorted_image_path is not None:
        rectified, new_intrinsics = undistort_image(image, intrinsics, distortion, balance=balance)
        os.makedirs(os.path.dirname(os.path.abspath(undistorted_image_path)), exist_ok=True)
        Image.fromarray(rectified).save(undistorted_image_path)
        print(f"Saved undistorted image → {undistorted_image_path}")
        if undistorted_intrinsics_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(undistorted_intrinsics_path)), exist_ok=True)
            new_intrinsics.save(undistorted_intrinsics_path)
            print(f"Saved undistorted intrinsics → {undistorted_intrinsics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate camera calibration from a single image")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--distortion_path", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--cam_id", type=str, default="kb:4",
                        help="AnyCalib camera model id (default: kb:4 = 4-param fisheye)")
    parser.add_argument("--model_id", type=str, default="anycalib_gen")
    parser.add_argument("--undistorted_image_path", type=str, default=None)
    parser.add_argument("--undistorted_intrinsics_path", type=str, default=None)
    parser.add_argument("--balance", type=float, default=0.0,
                        help="Fisheye undistort FoV/crop balance: 0 = crop, 1 = full FoV with borders")
    args = parser.parse_args()
    image_to_calibration(
        image_path=args.image_path,
        intrinsics_path=args.intrinsics_path,
        distortion_path=args.distortion_path,
        weights_path=args.weights_path,
        cam_id=args.cam_id,
        model_id=args.model_id,
        undistorted_image_path=args.undistorted_image_path,
        undistorted_intrinsics_path=args.undistorted_intrinsics_path,
        balance=args.balance,
    )
