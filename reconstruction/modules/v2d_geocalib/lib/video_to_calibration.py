# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GeoCalib: estimate intrinsics + camera-frame gravity from a video.

The video path is sampled uniformly, GeoCalib is run on each sampled RGB frame,
and the aggregate intrinsics/distortion/gravity are written as JSON files. The
full per-sample records are also written so downstream bundle alignment can
combine frame-specific gravity estimates with a camera trajectory.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from v2d.geocalib.lib._geocalib import aggregate_calibrations, predict_calibration


def _sample_frame_indices(total: int, num_samples: int) -> list[int]:
    if total <= 0:
        raise ValueError("Empty video")
    if num_samples >= total:
        return list(range(total))
    return np.linspace(0, total - 1, num_samples, dtype=int).tolist()


def _frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t


def video_to_calibration(
    video_path: str,
    intrinsics_path: str,
    distortion_path: str,
    gravity_path: str,
    calibration_path: str,
    weights_path: str | None = None,
    weights: str = "pinhole",
    camera_model: str = "pinhole",
    num_samples: int = 8,
    device: str | None = None,
) -> None:
    cap = cv2.VideoCapture(video_path)
    records: list[dict] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = _sample_frame_indices(total, num_samples)
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            intr, dist, extra = predict_calibration(
                _frame_to_tensor(frame),
                weights_path=weights_path,
                weights=weights,
                camera_model=camera_model,
                device=device,
            )
            records.append({
                "frame_index": int(idx),
                "intrinsics": intr.to_dict(),
                "distortion": dist.to_dict(),
                **extra,
            })
            print(f"  sampled frame {idx} ({len(records)}/{len(sample_indices)})")
    finally:
        cap.release()

    if not records:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    intrinsics, distortion, aggregate = aggregate_calibrations(records)
    os.makedirs(os.path.dirname(os.path.abspath(intrinsics_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(distortion_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(gravity_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(calibration_path)), exist_ok=True)

    intrinsics.save(intrinsics_path)
    distortion.save(distortion_path)

    gravity_record = {
        "model": "GeoCalib",
        "weights": weights,
        "camera_model": camera_model,
        "vector_camera": aggregate["gravity"]["vector_camera"],
        "num_samples": aggregate["num_samples"],
    }
    with open(gravity_path, "w") as f:
        json.dump(gravity_record, f, indent=2)

    with open(calibration_path, "w") as f:
        json.dump({
            "model": "GeoCalib",
            "weights": weights,
            "camera_model": camera_model,
            "intrinsics": intrinsics.to_dict(),
            "distortion": distortion.to_dict(),
            "gravity": gravity_record,
            "samples": records,
        }, f, indent=2)

    print(f"Saved GeoCalib intrinsics → {intrinsics_path}")
    print(f"Saved GeoCalib distortion → {distortion_path}")
    print(f"Saved GeoCalib gravity    → {gravity_path}")
    print(f"Saved GeoCalib samples    → {calibration_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate GeoCalib calibration from a video")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--distortion_path", required=True)
    parser.add_argument("--gravity_path", required=True)
    parser.add_argument("--calibration_path", required=True)
    parser.add_argument("--weights_path", default=None)
    parser.add_argument("--weights", choices=("pinhole", "distorted"), default="pinhole")
    parser.add_argument("--camera_model", default="pinhole",
                        choices=("pinhole", "simple_radial", "radial", "simple_divisional"))
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    video_to_calibration(
        video_path=args.video_path,
        intrinsics_path=args.intrinsics_path,
        distortion_path=args.distortion_path,
        gravity_path=args.gravity_path,
        calibration_path=args.calibration_path,
        weights_path=args.weights_path,
        weights=args.weights,
        camera_model=args.camera_model,
        num_samples=args.num_samples,
        device=args.device,
    )

