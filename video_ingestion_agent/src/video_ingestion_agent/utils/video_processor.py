# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Video processing utilities for frame and audio extraction."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata about a video file."""

    path: str
    duration: float  # seconds
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class Frame:
    """Represents a single video frame."""

    frame_id: str
    timestamp: float  # seconds
    image: np.ndarray  # RGB image array
    metadata: dict


class VideoProcessor:
    """
    Process video files for entity graph construction.

    Handles:
    - Frame extraction at specified FPS
    - Audio track extraction
    - Video metadata retrieval

    Example:
        processor = VideoProcessor("demo.mp4")
        metadata = processor.get_metadata()

        for frame in processor.extract_frames(fps=1.0):
            # Process frame
            pass
    """

    def __init__(self, video_path: str):
        """
        Initialize video processor.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        logger.info(f"Opened video: {self.video_path}")

    def __del__(self):
        """Release video capture on cleanup."""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

    def get_metadata(self) -> VideoMetadata:
        """
        Extract video metadata.

        Returns:
            VideoMetadata with duration, fps, resolution, etc.
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = frame_count / fps if fps > 0 else 0

        metadata = VideoMetadata(
            path=str(self.video_path),
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
        )

        logger.info(
            f"Video metadata: {duration:.1f}s, {fps:.1f} FPS, "
            f"{width}x{height}, {frame_count} frames"
        )

        return metadata

    def extract_frames(
        self, fps: float = 1.0, start_time: float | None = None, end_time: float | None = None
    ) -> Iterator[Frame]:
        """
        Extract frames from video at specified FPS.

        Args:
            fps: Target frames per second (e.g., 1.0 = 1 frame/sec)
            start_time: Start time in seconds (None = from beginning)
            end_time: End time in seconds (None = until end)

        Yields:
            Frame objects with image data and metadata
        """
        metadata = self.get_metadata()

        # Calculate frame interval
        source_fps = metadata.fps
        frame_interval = int(source_fps / fps) if fps > 0 else 1

        # Set start position
        if start_time is not None:
            start_frame = int(start_time * source_fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            start_frame = 0

        # Calculate end frame
        if end_time is not None:
            end_frame = int(end_time * source_fps)
        else:
            end_frame = metadata.frame_count

        frame_idx = start_frame
        extracted_count = 0

        logger.info(
            f"Extracting frames: {fps} FPS, "
            f"interval={frame_interval}, "
            f"range=[{start_time or 0:.1f}s, {end_time or metadata.duration:.1f}s]"
        )

        while frame_idx < end_frame:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Read frame
            ret, frame_bgr = self.cap.read()

            if not ret:
                logger.warning(f"Failed to read frame at index {frame_idx}")
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Calculate timestamp
            timestamp = frame_idx / source_fps

            # Create Frame object
            frame = Frame(
                frame_id=f"frame_{frame_idx:08d}",
                timestamp=timestamp,
                image=frame_rgb,
                metadata={
                    "frame_index": frame_idx,
                    "video_path": str(self.video_path),
                    "width": frame_rgb.shape[1],
                    "height": frame_rgb.shape[0],
                },
            )

            yield frame

            frame_idx += frame_interval
            extracted_count += 1

        logger.info(f"Extracted {extracted_count} frames")

    def get_frame_at_time(self, timestamp: float) -> Frame | None:
        """
        Extract a single frame at specific timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            Frame object or None if extraction fails
        """
        metadata = self.get_metadata()

        if timestamp < 0 or timestamp > metadata.duration:
            logger.warning(f"Timestamp {timestamp}s out of range [0, {metadata.duration}s]")
            return None

        # Set position
        frame_idx = int(timestamp * metadata.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read frame
        ret, frame_bgr = self.cap.read()

        if not ret:
            logger.warning(f"Failed to read frame at timestamp {timestamp}s")
            return None

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        return Frame(
            frame_id=f"frame_{frame_idx:08d}",
            timestamp=timestamp,
            image=frame_rgb,
            metadata={
                "frame_index": frame_idx,
                "video_path": str(self.video_path),
                "width": frame_rgb.shape[1],
                "height": frame_rgb.shape[0],
            },
        )
