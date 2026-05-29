# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""AnyCalib: estimate intrinsics + distortion from a folder of images, then
optionally write undistorted copies of every image to an output folder.

The folder is treated as one camera — per-image AnyCalib estimates are
aggregated (median) into a single calibration.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from v2d.anycalib.lib._anycalib import (
    aggregate_calibrations,
    predict_calibration,
)
from v2d.anycalib.lib._undistort import build_undistort_maps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _list_images(folder: str) -> list[Path]:
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No image files found in {folder}")
    return files


def _select_sample_indices(total: int, num_samples: int) -> list[int]:
    if num_samples >= total:
        return list(range(total))
    return np.linspace(0, total - 1, num_samples, dtype=int).tolist()


def image_folder_to_calibration(
    image_folder: str,
    intrinsics_path: str,
    distortion_path: str,
    weights_path: str,
    cam_id: str = "kb:4",
    model_id: str = "anycalib_gen",
    num_samples: int = 16,
    undistorted_folder: str | None = None,
    undistorted_intrinsics_path: str | None = None,
    balance: float = 0.0,
) -> None:
    files = _list_images(image_folder)
    sample_indices = _select_sample_indices(len(files), num_samples)

    calibrations = []
    for k, idx in enumerate(sample_indices, start=1):
        image = np.asarray(Image.open(files[idx]).convert("RGB"))
        calibrations.append(predict_calibration(
            image, weights_path=weights_path, cam_id=cam_id, model_id=model_id,
        ))
        print(f"  sampled {files[idx].name} ({k}/{len(sample_indices)})")

    intrinsics, distortion = aggregate_calibrations(calibrations)

    os.makedirs(os.path.dirname(os.path.abspath(intrinsics_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(distortion_path)), exist_ok=True)
    intrinsics.save(intrinsics_path)
    distortion.save(distortion_path)
    print(f"Saved intrinsics → {intrinsics_path}")
    print(f"Saved distortion ({distortion.model}) → {distortion_path}")

    if undistorted_folder is not None:
        map1, map2, new_intrinsics = build_undistort_maps(intrinsics, distortion, balance=balance)
        os.makedirs(undistorted_folder, exist_ok=True)
        for path in files:
            bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                print(f"  skipping unreadable {path.name}")
                continue
            rectified = cv2.remap(bgr, map1, map2, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(os.path.join(undistorted_folder, path.name), rectified)
        print(f"Saved {len(files)} undistorted images → {undistorted_folder}")
        if undistorted_intrinsics_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(undistorted_intrinsics_path)), exist_ok=True)
            new_intrinsics.save(undistorted_intrinsics_path)
            print(f"Saved undistorted intrinsics → {undistorted_intrinsics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate camera calibration from a folder of images")
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--distortion_path", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--cam_id", type=str, default="kb:4")
    parser.add_argument("--model_id", type=str, default="anycalib_gen")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--undistorted_folder", type=str, default=None)
    parser.add_argument("--undistorted_intrinsics_path", type=str, default=None)
    parser.add_argument("--balance", type=float, default=0.0)
    args = parser.parse_args()
    image_folder_to_calibration(
        image_folder=args.image_folder,
        intrinsics_path=args.intrinsics_path,
        distortion_path=args.distortion_path,
        weights_path=args.weights_path,
        cam_id=args.cam_id,
        model_id=args.model_id,
        num_samples=args.num_samples,
        undistorted_folder=args.undistorted_folder,
        undistorted_intrinsics_path=args.undistorted_intrinsics_path,
        balance=args.balance,
    )
