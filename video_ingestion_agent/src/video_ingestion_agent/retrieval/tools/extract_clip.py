# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool for extracting video clips with multi-video support."""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video_ingestion_agent.retrieval.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ClipInfo:
    """Information about an extracted clip."""

    output_path: str
    start_time: float
    end_time: float
    duration: float
    source_video: str = ""
    video_id: int | None = None

    def __str__(self) -> str:
        video_name = Path(self.source_video).name if self.source_video else ""
        return (
            f"Clip saved: {self.output_path} "
            f"[{self.start_time:.1f}s - {self.end_time:.1f}s] "
            f"(duration: {self.duration:.1f}s)" + (f" from {video_name}" if video_name else "")
        )


class ExtractClipTool(BaseTool):
    """
    Extract video clips based on timestamps.

    Uses ffmpeg to extract precise video segments.
    Supports multi-video databases with video registry pattern.
    """

    def __init__(
        self,
        video_path: str | None = None,
        video_paths: dict[int, str] | None = None,
        output_dir: str = "outputs/clips",
    ):
        """
        Initialize clip extractor.

        Args:
            video_path: Single video path (backward compat)
            video_paths: Dict mapping video_id to video path (for multi-video)
            output_dir: Directory for output clips
        """
        self.video_registry: dict[int, str] = video_paths or {}
        self.default_video_path = video_path

        # Backward compat: if old-style single video_path passed
        if video_path and not video_paths:
            self.video_registry[1] = video_path

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._clip_counter = 0

    def register_video(self, video_id: int, video_path: str):
        """Register a video in the registry."""
        self.video_registry[video_id] = video_path

    def _get_video_path(self, video_id: int | None = None, video_path: str | None = None) -> str:
        """
        Get video path with fallback logic.

        Priority:
        1. Explicit video_path parameter
        2. video_id lookup in registry
        3. default_video_path
        4. First video in registry

        Raises:
            ValueError: If no video path available
        """
        if video_path:
            return video_path

        if video_id is not None and video_id in self.video_registry:
            return self.video_registry[video_id]

        if self.default_video_path:
            return self.default_video_path

        if self.video_registry:
            # Return first registered video as fallback
            first_id = min(self.video_registry.keys())
            return self.video_registry[first_id]

        raise ValueError("No video path available. Specify video_path, video_id, or set default.")

    def _parse_time(self, value) -> float:
        """Parse time value, handling strings like '36.1', '36.1s', '36.1sss'."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove any 's' suffix (handles "36.1s", "36.1sss", etc.)
            cleaned = value.rstrip("s").strip()
            return float(cleaned)
        raise ValueError(f"Cannot parse time: {value}")

    @property
    def name(self) -> str:
        return "extract_clip"

    @property
    def description(self) -> str:
        return (
            "Extract a video clip from a source video. "
            "Specify start and end times in seconds. "
            "For multi-video databases, specify video_path or video_id. "
            "Returns the path to the extracted clip file."
        )

    @property
    def parameters(self) -> dict[str, dict[str, Any]]:
        return {
            "start_time": {
                "type": "number",
                "description": "Start time in seconds",
                "required": True,
            },
            "end_time": {"type": "number", "description": "End time in seconds", "required": True},
            "output_name": {
                "type": "string",
                "description": "Optional output filename (without extension)",
                "required": False,
            },
            "padding": {
                "type": "number",
                "description": "Extra seconds to add before/after (default: 0.5)",
                "required": False,
            },
            "video_id": {
                "type": "number",
                "description": "Video ID to extract from (for multi-video databases)",
                "required": False,
            },
            "video_path": {
                "type": "string",
                "description": "Explicit video path (overrides video_id)",
                "required": False,
            },
        }

    def execute(self, **kwargs) -> ToolResult:
        """Extract video clip."""
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")

        if start_time is None or end_time is None:
            return ToolResult(
                success=False, data=None, error="start_time and end_time are required"
            )

        # Convert to float (LLM may return strings like "36.1" or "36.1s")
        try:
            start_time = self._parse_time(start_time)
            end_time = self._parse_time(end_time)
        except (ValueError, TypeError) as e:
            return ToolResult(success=False, data=None, error=f"Invalid timestamps: {e}")

        if start_time >= end_time:
            return ToolResult(
                success=False, data=None, error="start_time must be less than end_time"
            )

        # Get video path
        try:
            video_id = kwargs.get("video_id")
            video_path_param = kwargs.get("video_path")
            video_path = self._get_video_path(video_id=video_id, video_path=video_path_param)
        except ValueError as e:
            return ToolResult(success=False, data=None, error=str(e))

        # Verify video exists
        if not Path(video_path).exists():
            return ToolResult(success=False, data=None, error=f"Video not found: {video_path}")

        try:
            # Add optional padding
            padding = kwargs.get("padding", 0.5)
            padded_start = max(0, start_time - padding)
            padded_end = end_time + padding

            # Generate output filename
            output_name = kwargs.get("output_name")
            if not output_name:
                self._clip_counter += 1
                video_name = Path(video_path).stem
                output_name = f"{video_name}_clip_{self._clip_counter:03d}_{padded_start:.1f}s-{padded_end:.1f}s"

            output_path = self.output_dir / f"{output_name}.mp4"

            # Extract using ffmpeg
            success = self._extract_with_ffmpeg(
                video_path, padded_start, padded_end, str(output_path)
            )

            if not success:
                return ToolResult(success=False, data=None, error="ffmpeg extraction failed")

            clip_info = ClipInfo(
                output_path=str(output_path),
                start_time=padded_start,
                end_time=padded_end,
                duration=padded_end - padded_start,
                source_video=video_path,
                video_id=video_id,
            )

            return ToolResult(success=True, data=clip_info)

        except Exception as e:
            logger.error(f"Clip extraction failed: {e}")
            return ToolResult(success=False, data=None, error=str(e))

    def _extract_with_ffmpeg(
        self, video_path: str, start_time: float, end_time: float, output_path: str
    ) -> bool:
        """
        Extract clip using ffmpeg.

        Args:
            video_path: Source video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path

        Returns:
            True if successful
        """
        duration = end_time - start_time

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss",
            str(start_time),  # Start time (before input for fast seek)
            "-i",
            video_path,
            "-t",
            str(duration),  # Duration
            "-c:v",
            "libx264",  # Re-encode video
            "-c:a",
            "aac",  # Re-encode audio
            "-avoid_negative_ts",
            "make_zero",
            "-loglevel",
            "error",
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                return False

            # Verify output exists
            if not os.path.exists(output_path):
                logger.error("Output file not created")
                return False

            logger.info(f"Extracted clip: {output_path}")
            return True

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out")
            return False
        except Exception as e:
            logger.error(f"ffmpeg failed: {e}")
            return False
