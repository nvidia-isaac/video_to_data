# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LangGraph nodes for the segmentation / verification / refinement loop.

Nodes:
  segmentation_node      -- chunk + segment the video
  extract_temp_node      -- extract temp .mp4 clips for critic
  verification_node      -- VLM critic pass
  refinement_node        -- refine invalid clips (reannotate annotations)
  cleanup_temp_node      -- persist final clips + temp file cleanup
  should_refine          -- conditional routing: refine or cleanup
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from video_ingestion_agent.ingestion.io import write_models_jsonl
from video_ingestion_agent.ingestion.segmentation.critic import Critic
from video_ingestion_agent.ingestion.segmentation.refiner import refine_clips
from video_ingestion_agent.ingestion.segmentation.segmenter import HybridSegmenter
from video_ingestion_agent.ingestion.segmentation.video_utils import (
    cleanup_temp_clips,
    extract_temp_clips,
)
from video_ingestion_agent.ingestion.state import ClipContext, PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segmentation_node(state: PipelineState) -> dict[str, Any]:
    """Node: Segment video into action clips using hybrid segmenter."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: SEGMENTATION")
    logger.info("=" * 60)

    config = state["config"]
    video_path = state["video_path"]

    segmenter = HybridSegmenter(config)
    clips = segmenter.segment_video(video_path)

    # Save initial clips
    run_dir = state["run_dir"]
    output_path = run_dir / "clips_stage1.jsonl"
    write_models_jsonl(clips, output_path)
    logger.info(f"Saved {len(clips)} clips to {output_path}")

    return {
        "clips": clips,
        "segmentation_complete": True,
        "status": "segmented",
    }


# ---------------------------------------------------------------------------
# Temp clip extraction
# ---------------------------------------------------------------------------


def extract_temp_node(state: PipelineState) -> dict[str, Any]:
    """Node: Extract temporary clip .mp4 files for verification/refinement."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: EXTRACT TEMP CLIPS")
    logger.info("=" * 60)

    clips = state["clips"]
    video_path = state["video_path"]
    base_video_dir = Path(video_path).parent

    config = state["config"]
    temp_dir, clip_path_map = extract_temp_clips(
        clips=clips,
        base_video_dir=base_video_dir,
        target_fps=config.models.vlm_fps,
    )

    logger.info(f"Extracted {len(clip_path_map)} temp clips to {temp_dir}")

    return {
        "temp_dir": temp_dir,
        "clip_path_map": clip_path_map,
        "status": "temp_extracted",
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verification_node(state: PipelineState) -> dict[str, Any]:
    """
    Node: Verify clips using VLM critic.

    On iteration 0: verify all clips.
    On iteration 1+: only verify refined clips, preserve previous valid results.
    """
    logger.info("=" * 60)
    logger.info("LangGraph Node: VERIFICATION")
    logger.info("=" * 60)

    clips = state["clips"]
    clip_path_map = state["clip_path_map"]
    config = state["config"]
    iteration = state["iteration"]
    refined_clip_ids = state.get("refined_clip_ids", [])
    previous_verifications = state.get("verifications", [])

    # Determine which clips to verify
    if iteration == 0:
        clips_to_verify = clips
        logger.info(f"Initial verification: verifying all {len(clips)} clips")
    else:
        clips_to_verify = [c for c in clips if c.clip_id in refined_clip_ids]
        logger.info(f"Iteration {iteration}: verifying {len(clips_to_verify)} refined clips")

    if clips_to_verify:
        critic = Critic(config)
        results_with_responses = critic.verify_clips_batch(clips_to_verify, clip_path_map)
        new_verifications = [r for r, _ in results_with_responses]

        # Save critic responses
        suffix = f"_iter{iteration}" if iteration > 0 else ""
        critic_dir = state["run_dir"] / f"critic_responses{suffix}"
        critic_dir.mkdir(exist_ok=True)

        for (_result, response), clip in zip(results_with_responses, clips_to_verify, strict=False):
            if response:
                response_file = critic_dir / f"{clip.clip_id}_critic.txt"
                with open(response_file, "w") as f:
                    f.write(response)
    else:
        new_verifications = []

    # Merge results
    if iteration == 0:
        all_verifications = new_verifications
    else:
        current_clip_ids = {c.clip_id for c in clips}
        verified_clip_ids = {v.clip_id for v in new_verifications}
        preserved = [
            v
            for v in previous_verifications
            if v.clip_id in current_clip_ids and v.clip_id not in verified_clip_ids
        ]
        all_verifications = preserved + new_verifications
        logger.info(
            f"Merged: {len(preserved)} preserved + {len(new_verifications)} new "
            f"= {len(all_verifications)} total"
        )

    # Save verification results
    suffix = f"_iter{iteration}" if iteration > 0 else ""
    verification_output = state["run_dir"] / f"verification_results{suffix}.jsonl"
    write_models_jsonl(all_verifications, verification_output)

    # Check if refinement needed
    valid_count = sum(1 for v in all_verifications if v.is_valid)
    invalid_count = len(all_verifications) - valid_count
    max_iterations = config.verification.max_iterations

    logger.info(
        f"Verification: {valid_count}/{len(all_verifications)} valid ({invalid_count} invalid)"
    )

    needs_refinement = invalid_count > 0 and iteration < max_iterations

    return {
        "verifications": all_verifications,
        "verification_complete": True,
        "refinement_needed": needs_refinement,
        "status": "verified",
    }


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


def refinement_node(state: PipelineState) -> dict[str, Any]:
    """Node: Refine invalid clips using reannotation strategy."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: REFINEMENT")
    logger.info("=" * 60)

    clips = state["clips"]
    verifications = state["verifications"]
    clip_path_map = state["clip_path_map"]
    config = state["config"]
    iteration = state["iteration"]
    next_iteration = iteration + 1

    # Run reannotate pass on all invalid clips
    updated_clips, refined_clip_ids, responses_ann = refine_clips(
        clips=clips,
        verifications=verifications,
        clip_path_map=clip_path_map,
        config=config,
        iteration=next_iteration,
        strategy="reannotate",
    )
    # Save refinement responses
    if responses_ann:
        responses_dir = state["run_dir"] / "refinement_responses"
        responses_dir.mkdir(exist_ok=True)
        for clip_id, response in responses_ann.items():
            response_file = responses_dir / f"{clip_id}_iter{next_iteration}.txt"
            with open(response_file, "w") as f:
                f.write(response)

    # ---- Sanitize: remove clips with invalid or ultra-short durations ----
    min_dur = config.segmentation.min_clip_s
    sanitized: list[ClipContext] = []
    for clip in updated_clips:
        dur = clip.end_t - clip.start_t
        if clip.end_t <= clip.start_t:
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

    if len(sanitized) < len(updated_clips):
        logger.info(
            f"Sanitization: {len(updated_clips)} -> {len(sanitized)} clips "
            f"(removed {len(updated_clips) - len(sanitized)} invalid)"
        )
    updated_clips = sanitized

    # Save updated clips
    output_path = state["run_dir"] / f"clips_stage1_iter{next_iteration}.jsonl"
    write_models_jsonl(updated_clips, output_path)

    return {
        "clips": updated_clips,
        "clip_path_map": clip_path_map,
        "iteration": next_iteration,
        "refined_clip_ids": refined_clip_ids,
        "status": "refined",
    }


# ---------------------------------------------------------------------------
# Cleanup + post-refinement
# ---------------------------------------------------------------------------


def cleanup_temp_node(state: PipelineState) -> dict[str, Any]:
    """Node: Clean up temporary clip files after verification/refinement loop."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: CLEANUP TEMP CLIPS")
    logger.info("=" * 60)

    clips = state["clips"]

    # ---- Persist final clips ----
    run_dir = state["run_dir"]
    final_clips_path = run_dir / "clips_final.jsonl"
    write_models_jsonl(clips, final_clips_path)
    logger.info(f"Saved {len(clips)} final clips to {final_clips_path}")

    # ---- Clean up temp directory ----
    temp_dir = state.get("temp_dir")
    if temp_dir:
        cleanup_temp_clips(Path(temp_dir) if isinstance(temp_dir, str) else temp_dir)

    return {
        "clips": clips,
        "temp_dir": None,
        "clip_path_map": {},
        "status": "temp_cleaned",
    }


# ---------------------------------------------------------------------------
# Conditional Routing
# ---------------------------------------------------------------------------


def should_refine(state: PipelineState) -> Literal["refine", "cleanup"]:
    """Conditional: decide whether to refine or move to cleanup."""
    if state["refinement_needed"]:
        logger.info("Routing to refinement node")
        return "refine"
    else:
        logger.info("All clips valid or max iterations reached, moving to cleanup")
        return "cleanup"
