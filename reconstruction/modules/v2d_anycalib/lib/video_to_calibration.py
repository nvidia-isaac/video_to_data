# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""AnyCalib: estimate intrinsics + distortion from a video, optionally write an
undistorted copy of the video.

Strategy: sample up to ``num_samples`` frames evenly across the video, run
AnyCalib on each, take the per-parameter median for robustness. If an
undistorted output video is requested, every frame is remapped through the
aggregated calibration and re-encoded with ffmpeg.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import cv2
import numpy as np

from v2d.anycalib.lib._anycalib import (
    aggregate_calibrations,
    predict_calibration,
)
from v2d.anycalib.lib._undistort import build_undistort_maps


def _sample_frame_indices(total: int, num_samples: int) -> list[int]:
    if total <= 0:
        raise ValueError("Empty video")
    if num_samples >= total:
        return list(range(total))
    return np.linspace(0, total - 1, num_samples, dtype=int).tolist()


def _video_meta(cap: cv2.VideoCapture) -> tuple[int, int, int, float]:
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    return width, height, total, fps


def _estimate_calibration(
    video_path: str,
    weights_path: str,
    cam_id: str,
    model_id: str,
    num_samples: int,
):
    cap = cv2.VideoCapture(video_path)
    try:
        _, _, total, _ = _video_meta(cap)
        sample_indices = _sample_frame_indices(total, num_samples)
        calibrations = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            calibrations.append(predict_calibration(
                rgb, weights_path=weights_path, cam_id=cam_id, model_id=model_id,
            ))
            print(f"  sampled frame {idx} ({len(calibrations)}/{len(sample_indices)})")
    finally:
        cap.release()

    if not calibrations:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return aggregate_calibrations(calibrations)


def _remap_video_to_ffmpeg(
    video_path: str,
    output_path: str,
    map1: np.ndarray,
    map2: np.ndarray,
    fps: float,
    width: int,
    height: int,
    crf: int,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            output_path,
        ],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(video_path)
    try:
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rectified = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            proc.stdin.write(rectified.tobytes())
            n += 1
            if n % 100 == 0:
                print(f"  remapped {n} frames...")
        print(f"  remapped {n} frames total")
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with return code {proc.returncode}")


def video_to_calibration(
    video_path: str,
    intrinsics_path: str,
    distortion_path: str,
    weights_path: str,
    cam_id: str = "kb:4",
    model_id: str = "anycalib_gen",
    num_samples: int = 16,
    undistorted_video_path: str | None = None,
    undistorted_intrinsics_path: str | None = None,
    balance: float = 0.0,
    crf: int = 17,
) -> None:
    intrinsics, distortion = _estimate_calibration(
        video_path, weights_path, cam_id, model_id, num_samples,
    )

    os.makedirs(os.path.dirname(os.path.abspath(intrinsics_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(distortion_path)), exist_ok=True)
    intrinsics.save(intrinsics_path)
    distortion.save(distortion_path)
    print(f"Saved intrinsics → {intrinsics_path}")
    print(f"Saved distortion ({distortion.model}) → {distortion_path}")

    if undistorted_video_path is not None:
        map1, map2, new_intrinsics = build_undistort_maps(intrinsics, distortion, balance=balance)
        cap = cv2.VideoCapture(video_path)
        width, height, _, fps = _video_meta(cap)
        cap.release()
        _remap_video_to_ffmpeg(
            video_path, undistorted_video_path, map1, map2,
            fps=fps, width=width, height=height, crf=crf,
        )
        print(f"Saved undistorted video → {undistorted_video_path}")
        if undistorted_intrinsics_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(undistorted_intrinsics_path)), exist_ok=True)
            new_intrinsics.save(undistorted_intrinsics_path)
            print(f"Saved undistorted intrinsics → {undistorted_intrinsics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate camera calibration from a video")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--distortion_path", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--cam_id", type=str, default="kb:4")
    parser.add_argument("--model_id", type=str, default="anycalib_gen")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Number of frames to sample for the calibration estimate")
    parser.add_argument("--undistorted_video_path", type=str, default=None)
    parser.add_argument("--undistorted_intrinsics_path", type=str, default=None)
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--crf", type=int, default=17, help="x264 CRF for the output video")
    args = parser.parse_args()
    video_to_calibration(
        video_path=args.video_path,
        intrinsics_path=args.intrinsics_path,
        distortion_path=args.distortion_path,
        weights_path=args.weights_path,
        cam_id=args.cam_id,
        model_id=args.model_id,
        num_samples=args.num_samples,
        undistorted_video_path=args.undistorted_video_path,
        undistorted_intrinsics_path=args.undistorted_intrinsics_path,
        balance=args.balance,
        crf=args.crf,
    )
