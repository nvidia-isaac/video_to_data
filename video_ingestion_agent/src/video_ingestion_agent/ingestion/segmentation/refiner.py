# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
High-level refinement orchestration.

Operates on absolute-timestamp ClipContext objects and a clip_path_map of
temporary clip files.
"""

import logging
from pathlib import Path

from video_ingestion_agent.ingestion.config import PipelineConfig
from video_ingestion_agent.ingestion.segmentation.strategies import (
    ReannotateStrategy,
    RefinementStrategy,
)
from video_ingestion_agent.ingestion.state import ClipContext, VerificationResult

logger = logging.getLogger(__name__)


def refine_clips(
    clips: list[ClipContext],
    verifications: list[VerificationResult],
    clip_path_map: dict[str, Path],
    config: PipelineConfig,
    iteration: int,
    strategy: str = "reannotate",
) -> tuple[list[ClipContext], list[str], dict[str, str]]:
    """
    Refine invalid clips using the specified strategy.

    Only clips marked invalid by the critic are re-annotated.
    Valid clips are preserved as-is.

    Args:
        clips: List of all clips
        verifications: List of verification results
        clip_path_map: Mapping of clip_id to temporary .mp4 file paths
        config: Pipeline configuration
        iteration: Current refinement iteration number
        strategy: Refinement strategy name (default: "reannotate")

    Returns:
        Tuple of:
        - updated_clips: List of clips (invalid replaced with refined)
        - refined_clip_ids: List of clip IDs that were actually refined
        - responses: Dict mapping clip_id to raw model response
    """
    # Find invalid clips
    invalid_verifications = [v for v in verifications if not v.is_valid]
    invalid_clip_ids = {v.clip_id for v in invalid_verifications}

    if not invalid_verifications:
        logger.info("No invalid clips to refine")
        return clips, [], {}

    logger.info(f"Refining {len(invalid_verifications)} invalid clips using '{strategy}' strategy")

    # Initialize strategy
    strategy_instance = _get_strategy(strategy, config)

    # Refine each invalid clip
    refinement_results: dict[str, ClipContext] = {}
    responses: dict[str, str] = {}

    for verification in invalid_verifications:
        # Find the original clip
        original_clip = next((c for c in clips if c.clip_id == verification.clip_id), None)

        if not original_clip:
            logger.warning(f"Could not find clip {verification.clip_id} for refinement")
            continue

        # Get clip video path
        clip_video_path = clip_path_map.get(verification.clip_id)
        if clip_video_path is None or not clip_video_path.exists():
            logger.warning(f"Clip video not found for {verification.clip_id}, keeping original")
            refinement_results[original_clip.clip_id] = original_clip
            continue

        # Refine the clip
        refined_clip, raw_response = strategy_instance.refine_clip(
            original_clip,
            verification,
            clip_video_path,
            iteration,
        )

        if refined_clip:
            refinement_results[original_clip.clip_id] = refined_clip
            if raw_response:
                responses[original_clip.clip_id] = raw_response
        else:
            # Keep original if refinement failed
            logger.warning(f"Refinement failed for {original_clip.clip_id}, keeping original")
            refinement_results[original_clip.clip_id] = original_clip

    # Build updated clips list
    updated_clips = []
    refined_clip_ids = []

    for clip in clips:
        if clip.clip_id in invalid_clip_ids:
            refined = refinement_results.get(clip.clip_id, clip)
            updated_clips.append(refined)

            if refined.metadata.get("refined", False):
                refined_clip_ids.append(clip.clip_id)
        else:
            updated_clips.append(clip)

    logger.info(f"Successfully refined {len(refined_clip_ids)}/{len(invalid_clip_ids)} clips")

    return updated_clips, refined_clip_ids, responses


def _get_strategy(strategy_name: str, config: PipelineConfig) -> RefinementStrategy:
    """
    Get a refinement strategy instance by name.

    Args:
        strategy_name: Name of the strategy
        config: Pipeline configuration

    Returns:
        RefinementStrategy instance
    """
    strategies = {
        "reannotate": ReannotateStrategy,
    }

    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown refinement strategy: '{strategy_name}'. Available strategies: {available}"
        )

    return strategy_class(config)
