# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Core data types for the unified ingestion + entity graph pipeline.

ClipContext uses absolute timestamps (start_t, end_t in seconds) with
top-level object, action, description fields. No persistent clip files
are stored -- only video_path + timestamps for retrieval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from pydantic import BaseModel, Field

from video_ingestion_agent.ingestion.config import PipelineConfig


@dataclass
class ActionSegment:
    """A temporally-bounded action segment identified by VLM."""

    segment_id: int
    start_t: float  # seconds
    end_t: float  # seconds
    object_name: str
    action: str
    description: str
    confidence: float = 1.0


class ClipContext(BaseModel):
    """
    Represents a video clip with absolute timestamps and annotations.

    Designed for retrieval: stores video_path + timestamps, not clip files.
    Temporary .mp4 clips are extracted only for verification/refinement
    and cleaned up immediately after.
    """

    clip_id: str = Field(description="Unique identifier for the clip")
    video_path: str = Field(description="Path to the source video file")
    start_t: float = Field(description="Absolute start time in seconds")
    end_t: float = Field(description="Absolute end time in seconds")

    # Top-level annotation fields (promoted from metadata)
    object: str = Field(default="", description="Object being manipulated")
    action: str = Field(default="", description="Action performed on the object")
    description: str = Field(default="", description="Detailed description of the action")

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (fps, source_video, etc.)"
    )

    @property
    def duration(self) -> float:
        """Duration of the clip in seconds."""
        return self.end_t - self.start_t


class VerificationResult(BaseModel):
    """Result from verifying a clip against quality criteria."""

    clip_id: str = Field(description="Associated clip ID")
    is_valid: bool = Field(description="Whether the clip is correctly segmented and annotated")
    verification_score: float = Field(description="Verification confidence score (0-1)")
    violations: list[str] = Field(default_factory=list, description="List of issues found")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Detailed critic feedback")


class PipelineState(TypedDict):
    """State that flows through the unified pipeline graph."""

    # Input
    video_path: str
    run_dir: Path
    config: PipelineConfig

    # Segmentation
    clips: list[ClipContext]
    segmentation_complete: bool

    # Temporary clip extraction
    temp_dir: Path | None
    clip_path_map: dict[str, Path]

    # Verification
    verifications: list[VerificationResult]
    verification_complete: bool

    # Refinement tracking
    iteration: int
    refinement_needed: bool
    refined_clip_ids: list[str]

    # Entity graph outputs
    entities: list
    relationships: list
    frames: list
    embeddings: list
    linked_entities: list
    linked_relationships: list
    db_paths: dict[str, str]

    # Metadata
    status: str
    error: str | None


def clips_to_action_segments(
    clips: list[ClipContext],
) -> list[ActionSegment]:
    """
    Convert ClipContext objects to ActionSegment dataclasses for entity graph.

    Bridges the Pydantic ClipContext (ingestion) to the dataclass ActionSegment
    used by the database writer.

    Args:
        clips: List of ClipContext objects from ingestion pipeline

    Returns:
        List of ActionSegment dataclasses
    """
    segments = []
    for i, clip in enumerate(clips):
        segment = ActionSegment(
            segment_id=i + 1,
            start_t=clip.start_t,
            end_t=clip.end_t,
            object_name=clip.object,
            action=clip.action,
            description=clip.description,
            confidence=clip.metadata.get("confidence", 1.0),
        )
        segments.append(segment)

    return segments
