# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Video utilities for clip extraction.

Provides temporary clip extraction for verification/refinement.
Clips are written to a temp directory and cleaned up after use --
only video_path + absolute timestamps are persisted.

The core ``extract_clip_ffmpeg`` function lives in ``utils.video_utils``
and is re-exported here for backward compatibility.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from video_ingestion_agent.ingestion.state import ClipContext
from video_ingestion_agent.utils.video_utils import extract_clip_ffmpeg  # noqa: F401 -- re-export

logger = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def extract_temp_clips(
    clips: list[ClipContext],
    base_video_dir: Path | None = None,
    target_fps: float | None = None,
) -> tuple[Path, dict[str, Path]]:
    """
    Extract clips to a temporary directory for verification/refinement.

    The returned temp_dir must be cleaned up by the caller after use
    (see cleanup_temp_clips).

    Args:
        clips: List of ClipContext objects with absolute timestamps
        base_video_dir: Optional base directory for resolving relative video paths
        target_fps: If set, re-encode clips at this frame rate (FPS proxy).

    Returns:
        Tuple of (temp_dir, clip_path_map) where clip_path_map maps
        clip_id to the temporary .mp4 file path.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="video_ingestion_clips_"))
    clip_path_map: dict[str, Path] = {}

    logger.info(f"Extracting {len(clips)} clips to temp dir: {temp_dir}")

    for i, clip in enumerate(clips, 1):
        # Resolve video path -- only prepend base_video_dir if the path
        # doesn't already exist (avoids doubling when clip.video_path
        # already contains the full relative path from CWD).
        video_path = Path(clip.video_path)
        if not video_path.exists() and not video_path.is_absolute() and base_video_dir is not None:
            video_path = base_video_dir / video_path

        if not video_path.exists():
            logger.error(f"Source video not found: {video_path}")
            continue

        output_path = temp_dir / f"{clip.clip_id}.mp4"

        logger.info(
            f"[{i}/{len(clips)}] Extracting {clip.clip_id}: "
            f"[{clip.start_t:.1f}s - {clip.end_t:.1f}s]"
        )

        success = extract_clip_ffmpeg(
            video_path=video_path,
            start_t=clip.start_t,
            end_t=clip.end_t,
            output_path=output_path,
            target_fps=target_fps,
        )

        if success:
            clip_path_map[clip.clip_id] = output_path
        else:
            logger.warning(f"Failed to extract clip: {clip.clip_id}")

    logger.info(f"Extracted {len(clip_path_map)}/{len(clips)} clips to {temp_dir}")
    return temp_dir, clip_path_map


def cleanup_temp_clips(temp_dir: Path) -> None:
    """
    Remove the temporary clip directory and all its contents.

    Args:
        temp_dir: Path to the temp directory created by extract_temp_clips
    """
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp clips: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp dir {temp_dir}: {e}")
    else:
        logger.debug(f"Temp dir already removed: {temp_dir}")


def extract_video_chunk(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    target_fps: float | None = None,
) -> bool:
    """
    Extract a video chunk using ffmpeg (for chunked segmentation).

    This is a convenience wrapper around ``extract_clip_ffmpeg`` that
    accepts string paths for backward compatibility.

    Args:
        video_path: Path to source video
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Path to save extracted chunk
        target_fps: If set, re-encode the chunk at this frame rate.

    Returns:
        True if successful, False otherwise
    """
    return extract_clip_ffmpeg(
        video_path=video_path,
        start_t=start_time,
        end_t=end_time,
        output_path=output_path,
        target_fps=target_fps,
    )
