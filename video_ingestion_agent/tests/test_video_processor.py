# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Tests for VideoProcessor.extract_frames sampling."""

import itertools

import cv2
import numpy as np
import pytest

from video_ingestion_agent.utils.video_processor import VideoProcessor


def _write_video(path, fps: float, n_frames: int) -> str:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v videos")
    for i in range(n_frames):
        writer.write(np.full((32, 32, 3), i % 256, dtype=np.uint8))
    writer.release()
    return str(path)


def _frame_indices(video_path: str, fps: float) -> list[int]:
    frames = VideoProcessor(video_path).extract_frames(fps=fps)
    # Bound the generator so a sampler that never advances fails instead of hanging.
    return [f.metadata["frame_index"] for f in itertools.islice(frames, 1000)]


def test_extract_frames_integer_ratio(tmp_path):
    video = _write_video(tmp_path / "v.mp4", fps=30, n_frames=90)

    assert _frame_indices(video, fps=10.0) == list(range(0, 90, 3))


def test_extract_frames_fps_above_source_fps(tmp_path):
    video = _write_video(tmp_path / "v.mp4", fps=2, n_frames=6)

    assert _frame_indices(video, fps=4.0) == [0, 1, 2, 3, 4, 5]


def test_extract_frames_non_integer_ratio(tmp_path):
    video = _write_video(tmp_path / "v.mp4", fps=25, n_frames=100)

    indices = _frame_indices(video, fps=10.0)

    assert len(indices) == 40
    timestamps = np.array(indices) / 25.0
    np.testing.assert_allclose(timestamps, np.arange(40) * 0.1, atol=0.5 / 25)
