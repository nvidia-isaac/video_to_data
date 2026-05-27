# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Hybrid video segmenter.

Combines chunk-based video processing (ffmpeg extraction, chunk_size /
chunk_overlap) with conversation-style prompt handling and JSON parsing.

Uses the shared ModelManager for all model interactions.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from video_ingestion_agent.ingestion.config import PipelineConfig
from video_ingestion_agent.ingestion.segmentation.dedup import ClipDeduplicator
from video_ingestion_agent.ingestion.segmentation.prompts import (
    SEGMENTATION_SYSTEM_PROMPT,
    SEGMENTATION_USER_PROMPT,
)
from video_ingestion_agent.ingestion.segmentation.video_utils import extract_video_chunk
from video_ingestion_agent.ingestion.state import ClipContext
from video_ingestion_agent.models.model_manager import BaseModel as ModelBase
from video_ingestion_agent.models.model_manager import get_model_manager
from video_ingestion_agent.utils.parsing import parse_llm_json as _parse_llm_json
from video_ingestion_agent.utils.parsing import parse_timestamp as _parse_timestamp
from video_ingestion_agent.utils.video_utils import get_video_info as _get_video_info

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> list[dict]:
    """
    Extract JSON array from model response text.

    Handles responses with markdown code blocks or raw JSON.
    Delegates to ``common.parsing.parse_llm_json``.

    Args:
        text: Model response text

    Returns:
        List of clip dictionaries

    Raises:
        ValueError: If no valid JSON found
    """
    return _parse_llm_json(text, expect_array=True)


def parse_timestamp(timestamp) -> float:
    """
    Parse timestamp in various formats to seconds.

    Delegates to ``common.parsing.parse_timestamp``.

    Args:
        timestamp: Timestamp in any supported format

    Returns:
        Time in seconds
    """
    return _parse_timestamp(timestamp)


def get_video_info(video_path: Path) -> tuple[int, float, float]:
    """
    Extract video metadata using OpenCV.

    Args:
        video_path: Path to the video file

    Returns:
        Tuple of (total_frames, fps, duration_seconds)
    """
    info = _get_video_info(video_path)
    return info["frame_count"], info["fps"], info["duration"]


class HybridSegmenter:
    """
    Hybrid video segmenter combining chunk-based processing with VLM prompts.

    - Chunks the video using ffmpeg
    - Uses conversation-style prompts for segmentation
    - Produces ClipContext objects with absolute timestamps
    - Uses ModelManager for all model interactions
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the hybrid segmenter.

        Args:
            config: Unified pipeline configuration
        """
        self.config = config
        self.seg_config = config.segmentation
        self.model_config = config.models

        # Fall back to built-in defaults when the config leaves prompts empty
        self._system_prompt = self.seg_config.system_prompt.strip() or SEGMENTATION_SYSTEM_PROMPT
        self._user_prompt = self.seg_config.user_prompt.strip() or SEGMENTATION_USER_PROMPT

        self._model: ModelBase | None = None

    def _get_model(self) -> ModelBase:
        """Get VLM model from ModelManager (lazy loaded, cached)."""
        if self._model is None:
            manager = get_model_manager()
            # Only pass vllm_url when using the vLLM backend; for "api" backend
            # pass None so APIModel uses its own default endpoint.
            api_url = (
                self.model_config.vllm_url if self.model_config.vlm_backend == "vllm" else None
            )
            self._model = manager.get_model(
                model_name=self.model_config.vlm_model,
                backend=self.model_config.vlm_backend,
                device=self.model_config.device,
                fps=self.model_config.vlm_fps,
                api_key=self.model_config.api_key,
                api_url=api_url,
                use_local_media=self.model_config.vllm_local_media,
            )
        return self._model

    def segment_video(self, video_path: str | Path) -> list[ClipContext]:
        """
        Segment an entire video into action clips.

        Processes the video in overlapping chunks, segments each chunk via VLM,
        and converts all timestamps to absolute seconds.

        Args:
            video_path: Path to the video file

        Returns:
            List of ClipContext objects with absolute timestamps
        """
        video_path = Path(video_path)
        logger.info(f"Segmenting video: {video_path}")

        # Get video metadata
        total_frames, fps, duration = get_video_info(video_path)
        logger.info(f"  Duration: {duration:.1f}s, {total_frames} frames, {fps:.1f} fps")

        chunk_size = self.seg_config.chunk_size
        chunk_overlap = self.seg_config.chunk_overlap

        logger.info(f"  Chunk size: {chunk_size}s, overlap: {chunk_overlap}s")

        all_clips: list[ClipContext] = []
        chunk_start = 0.0
        global_clip_idx = 0

        while chunk_start < duration:
            chunk_end = min(chunk_start + chunk_size, duration)

            # Segment this chunk
            chunk_clips = self._segment_chunk(
                video_path=str(video_path),
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                video_duration=duration,
                fps=fps,
                global_clip_idx=global_clip_idx,
            )

            all_clips.extend(chunk_clips)
            global_clip_idx += len(chunk_clips)

            # Move to next chunk
            chunk_start += chunk_size - chunk_overlap

        logger.info(f"Total segments before dedup: {len(all_clips)}")

        method = self.seg_config.dedup_method
        deduplicator = ClipDeduplicator(
            overlap_threshold=self.seg_config.dedup_overlap_threshold,
            method=method,
            model=self._get_model() if method == "llm" else None,
        )
        all_clips = deduplicator.run(all_clips)

        # Sanitize: drop clips with negative or ultra-short durations
        min_dur = self.seg_config.min_clip_s
        sanitized: list[ClipContext] = []
        for clip in all_clips:
            dur = clip.end_t - clip.start_t
            if dur <= 0:
                logger.warning(
                    f"Dropping {clip.clip_id}: negative duration "
                    f"[{clip.start_t:.2f}s, {clip.end_t:.2f}s]"
                )
            elif dur < min_dur:
                logger.warning(
                    f"Dropping {clip.clip_id}: duration {dur:.2f}s < min_clip_s ({min_dur}s)"
                )
            else:
                sanitized.append(clip)

        if len(sanitized) < len(all_clips):
            logger.info(
                f"Duration sanitization: {len(all_clips)} -> {len(sanitized)} "
                f"(removed {len(all_clips) - len(sanitized)} invalid clips)"
            )
        all_clips = sanitized

        logger.info(f"Total segments after dedup + sanitize: {len(all_clips)}")
        return all_clips

    def _segment_chunk(
        self,
        video_path: str,
        chunk_start: float,
        chunk_end: float,
        video_duration: float,
        fps: float,
        global_clip_idx: int,
    ) -> list[ClipContext]:
        """
        Segment a single video chunk.

        Args:
            video_path: Path to the source video
            chunk_start: Start of the chunk in seconds
            chunk_end: End of the chunk in seconds
            video_duration: Total video duration in seconds
            fps: Video FPS
            global_clip_idx: Running clip index counter

        Returns:
            List of ClipContext for this chunk (with absolute timestamps)
        """
        logger.info(f"Segmenting chunk [{chunk_start:.1f}s - {chunk_end:.1f}s]")

        # Extract chunk to temp file when it doesn't cover the full video.
        # Without this, the VLM sees the entire video instead of just the
        # chunk window, causing it to produce unparseable responses.
        chunk_duration = chunk_end - chunk_start
        use_temp_chunk = chunk_duration < (video_duration - 0.5)
        temp_video_path = None

        try:
            if use_temp_chunk:
                # Include video basename + PID in temp filename to avoid
                # collisions when multiple workers process in parallel.
                video_stem = Path(video_path).stem
                pid = os.getpid()
                temp_dir = tempfile.gettempdir()
                temp_video_path = os.path.join(
                    temp_dir,
                    f"chunk_{video_stem}_{pid}_{chunk_start:.0f}_{chunk_end:.0f}.mp4",
                )

                logger.info("  Extracting chunk to temp file...")
                if not extract_video_chunk(
                    video_path,
                    chunk_start,
                    chunk_end,
                    temp_video_path,
                    target_fps=self.model_config.vlm_fps,
                ):
                    logger.warning("  Failed to extract chunk, using full video")
                    use_temp_chunk = False
                elif not os.path.isfile(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                    logger.warning("  Temp chunk file is missing or empty, using full video")
                    use_temp_chunk = False

            # Choose which video to analyze (ensure str for transformers compatibility)
            analysis_video = str(temp_video_path if use_temp_chunk else video_path)
            logger.info(f"  Analyzing: {analysis_video} (exists={os.path.isfile(analysis_video)})")

            # Get VLM model
            model = self._get_model()

            # Generate segmentation using VLM
            response = model.generate_from_video(
                video_path=analysis_video,
                prompt=self._user_prompt,
                system_prompt=self._system_prompt,
                max_new_tokens=4096,
                temperature=0.0,
            )

            logger.debug(f"  Raw VLM response:\n{response}")

            # Parse response (timestamps are relative to chunk start at 0)
            clips = self._parse_response(
                response=response,
                video_path=video_path,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                fps=fps,
                global_clip_idx=global_clip_idx,
            )

            logger.info(f"  Found {len(clips)} action segments")
            for clip in clips:
                logger.info(
                    f"    {clip.clip_id}: [{clip.start_t:.1f}s-{clip.end_t:.1f}s] "
                    f"{clip.action} {clip.object}"
                )

            return clips

        except Exception as e:
            logger.error(f"Segmentation failed for chunk: {e}")
            import traceback

            logger.error(traceback.format_exc())

            # Fallback: return entire chunk as single segment
            clip_id = f"{Path(video_path).stem}_clip_{global_clip_idx + 1:04d}"
            return [
                ClipContext(
                    clip_id=clip_id,
                    video_path=video_path,
                    start_t=chunk_start,
                    end_t=chunk_end,
                    object="unknown",
                    action="manipulation",
                    description="Video segment (segmentation failed)",
                    metadata={"fps": fps, "confidence": 0.3},
                )
            ]

        finally:
            # Clean up temp chunk
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except Exception as e:
                    logger.warning(f"  Failed to clean up temp file: {e}")

    def _parse_response(
        self,
        response: str,
        video_path: str,
        chunk_start: float,
        chunk_end: float,
        fps: float,
        global_clip_idx: int,
    ) -> list[ClipContext]:
        """
        Parse VLM response and convert to ClipContext objects with absolute timestamps.

        Args:
            response: Raw VLM response text
            video_path: Path to source video
            chunk_start: Start time of the chunk
            chunk_end: End time of the chunk
            fps: Video FPS
            global_clip_idx: Running clip index counter

        Returns:
            List of ClipContext with absolute timestamps
        """
        try:
            clips_data = parse_json_response(response)
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse VLM response: {e}")
            logger.debug(f"Response: {response}")
            return []

        clips = []
        for clip_data in clips_data:
            try:
                # Parse timestamps (relative to chunk)
                start_t_rel = parse_timestamp(clip_data["start_time"])
                end_t_rel = parse_timestamp(clip_data["end_time"])

                # Skip invalid
                if end_t_rel <= start_t_rel:
                    continue

                clip_duration = end_t_rel - start_t_rel

                # Skip too short
                if clip_duration < self.seg_config.min_clip_s:
                    logger.debug(f"Segment too short ({clip_duration:.1f}s), skipping")
                    continue

                # Cap too long
                if clip_duration > self.seg_config.max_clip_s:
                    logger.warning(f"Segment too long ({clip_duration:.1f}s), capping")
                    end_t_rel = start_t_rel + self.seg_config.max_clip_s

                # Convert to absolute timestamps
                start_t_abs = chunk_start + start_t_rel
                end_t_abs = chunk_start + end_t_rel

                # Clamp to chunk boundaries
                start_t_abs = max(chunk_start, start_t_abs)
                end_t_abs = min(chunk_end, end_t_abs)

                # Generate clip ID
                clip_idx = global_clip_idx + len(clips) + 1
                clip_id = f"{Path(video_path).stem}_clip_{clip_idx:04d}"

                clip = ClipContext(
                    clip_id=clip_id,
                    video_path=video_path,
                    start_t=start_t_abs,
                    end_t=end_t_abs,
                    object=clip_data.get("object", ""),
                    action=clip_data.get("action", ""),
                    description=clip_data.get("description", ""),
                    metadata={
                        "fps": fps,
                        "clip_index": clip_idx,
                        "annotation_source": "vlm_segmentation",
                        "confidence": 1.0,
                    },
                )
                clips.append(clip)

            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed clip entry: {e}")
                continue

        return clips
