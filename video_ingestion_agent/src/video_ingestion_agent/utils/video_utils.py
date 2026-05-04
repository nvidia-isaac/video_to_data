# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared video utility functions.

Consolidates duplicate video processing logic that was previously spread
across vllm_model.py, api_model.py, segmenter.py, video_utils.py, and
action_segmenter.py.
"""

import base64
import logging
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def get_video_info(video_path: str | Path) -> dict:
    """
    Get video metadata using OpenCV.

    Args:
        video_path: Path to video file.

    Returns:
        Dict with keys: frame_count, fps, duration, width, height.

    Raises:
        RuntimeError: If video cannot be opened or has invalid FPS.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

        duration = frame_count / fps

        return {
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
            "width": width,
            "height": height,
        }
    finally:
        cap.release()


def extract_frames_base64(
    video_path: str,
    fps: int = 4,
    jpeg_quality: int = 85,
) -> list[str]:
    """
    Extract frames from video and encode as base64 JPEG strings.

    Samples frames at the given FPS from the video and returns each
    frame as a base64-encoded JPEG string.

    Args:
        video_path: Path to video file.
        fps: Desired sampling rate in frames per second.
        jpeg_quality: JPEG encoding quality (0-100).

    Returns:
        List of base64-encoded JPEG strings.

    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame indices based on desired FPS
    num_to_extract = max(1, min(int(duration * fps), total_frames))

    if num_to_extract >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_to_extract) for i in range(num_to_extract)]

    frames_base64 = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frames_base64.append(base64.b64encode(buffer).decode("utf-8"))

    cap.release()

    logger.info(
        f"Extracted {len(frames_base64)} frames "
        f"(video: {duration:.1f}s @ {video_fps:.1f}fps, "
        f"sampling @ {fps}fps)"
    )

    return frames_base64


def extract_clip_ffmpeg(
    video_path: str | Path,
    start_t: float,
    end_t: float,
    output_path: str | Path,
    codec: str = "libx264",
    preset: str = "veryfast",
    crf: int = 18,
    target_fps: float | None = None,
) -> bool:
    """
    Extract a video clip using ffmpeg with timestamp-based seeking.

    When *target_fps* is provided the output is re-encoded at that frame
    rate (``-vf fps=<target_fps>``).  This creates a "proxy" file where
    every frame is meaningful for the VLM, and the file-level FPS metadata
    matches the logical sampling rate so that vLLM (with ``num_frames=-1``)
    reads exactly the right number of frames.

    Args:
        video_path: Path to source video.
        start_t: Start time in seconds.
        end_t: End time in seconds.
        output_path: Path for output clip.
        codec: Video codec to use.
        preset: Encoding preset (ultrafast, veryfast, fast, etc.).
        crf: Constant rate factor for quality (lower = better).
        target_fps: If set, re-encode the output at this frame rate.

    Returns:
        True if successful, False otherwise.
    """
    duration = end_t - start_t
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_t),
        "-i",
        str(video_path),
        "-t",
        str(duration),
    ]

    # Downsample to target FPS when requested (creates a clean proxy file)
    if target_fps is not None and target_fps > 0:
        cmd.extend(["-vf", f"fps={target_fps}"])

    cmd.extend(
        [
            "-c:v",
            codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-an",
            "-avoid_negative_ts",
            "make_zero",
            "-loglevel",
            "error",
            str(output_path),
        ]
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timeout for {output_path}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found - please install ffmpeg")
        return False
