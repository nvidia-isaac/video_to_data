# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end LangGraph pipeline: ingestion + entity graph.

Single workflow:
  segment -> extract_temp_clips -> verify -> (refine loop) ->
  cleanup_temp_clips -> entity_extract -> frame_embeddings ->
  entity_link -> db_write -> report

Classification is removed. Temporary clip .mp4 files are used only for
the verification/refinement loop and cleaned up immediately after.
"""

import logging
from pathlib import Path

from langgraph.graph import END, StateGraph

from video_ingestion_agent.ingestion.config import PipelineConfig
from video_ingestion_agent.ingestion.entity_graph_nodes import (
    database_write_node,
    entity_extraction_node,
    entity_linking_node,
    frame_embedding_node,
    reporting_node,
)
from video_ingestion_agent.ingestion.segmentation_nodes import (
    cleanup_temp_node,
    extract_temp_node,
    refinement_node,
    segmentation_node,
    should_refine,
    verification_node,
)
from video_ingestion_agent.ingestion.state import PipelineState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Graph Builder
# ---------------------------------------------------------------------------


def create_pipeline_graph(
    enable_verification: bool = True,
    enable_refinement: bool = True,
    enable_entity_graph: bool = True,
    enable_reporting: bool = True,
) -> StateGraph:
    """
    Create the end-to-end LangGraph pipeline.

    Pipeline flow:
      segment -> [extract_temp -> verify -> (refine loop) -> cleanup] ->
      [entity_extract -> frame_embed -> entity_link -> db_write] -> report

    Args:
        enable_verification: Include verify/refine loop
        enable_refinement: Enable refinement within verify loop
        enable_entity_graph: Include entity graph building nodes
        enable_reporting: Include HTML report generation

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(PipelineState)

    # Always: segmentation
    workflow.add_node("segment", segmentation_node)
    workflow.set_entry_point("segment")

    # cleanup_temp always runs: it persists clips_final.jsonl and removes
    # any temporary clip files (safe even when temp_dir is None).
    workflow.add_node("cleanup_temp", cleanup_temp_node)

    if enable_verification:
        workflow.add_node("extract_temp", extract_temp_node)
        workflow.add_node("verify", verification_node)

        workflow.add_edge("segment", "extract_temp")
        workflow.add_edge("extract_temp", "verify")

        if enable_refinement:
            workflow.add_node("refine", refinement_node)

            workflow.add_conditional_edges(
                "verify",
                should_refine,
                {
                    "refine": "refine",
                    "cleanup": "cleanup_temp",
                },
            )
            workflow.add_edge("refine", "verify")
        else:
            workflow.add_edge("verify", "cleanup_temp")
    else:
        workflow.add_edge("segment", "cleanup_temp")

    # After cleanup, continue to entity graph or reporting or end
    next_after_cleanup = _next_stage_name(enable_entity_graph, enable_reporting)
    if next_after_cleanup:
        workflow.add_edge("cleanup_temp", next_after_cleanup)
    else:
        workflow.add_edge("cleanup_temp", END)

    # Entity graph nodes
    if enable_entity_graph:
        workflow.add_node("entity_extract", entity_extraction_node)
        workflow.add_node("frame_embed", frame_embedding_node)
        workflow.add_node("entity_link", entity_linking_node)
        workflow.add_node("db_write", database_write_node)

        workflow.add_edge("entity_extract", "frame_embed")
        workflow.add_edge("frame_embed", "entity_link")
        workflow.add_edge("entity_link", "db_write")

        if enable_reporting:
            workflow.add_edge("db_write", "report")
        else:
            workflow.add_edge("db_write", END)

    # Reporting
    if enable_reporting:
        workflow.add_node("report", reporting_node)
        workflow.add_edge("report", END)

    return workflow.compile()


def _next_stage_name(enable_entity_graph: bool, enable_reporting: bool) -> str | None:
    """Determine the next stage after cleanup or segment."""
    if enable_entity_graph:
        return "entity_extract"
    elif enable_reporting:
        return "report"
    return None


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------


def run_pipeline(
    video_path: str,
    run_dir: str | Path,
    config: PipelineConfig,
    graph_db_path: str | Path | None = None,
    vector_db_path: str | Path | None = None,
) -> PipelineState:
    """
    Run the complete end-to-end pipeline.

    Args:
        video_path: Path to input video file
        run_dir: Output directory for this run
        config: Parsed PipelineConfig object
        graph_db_path: Optional explicit path for shared graph DB
                       (used by batch ingestion for multi-shard writes)
        vector_db_path: Optional explicit path for shared vector DB

    Returns:
        Final pipeline state
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if graph_db_path is not None or vector_db_path is not None:
        db_overrides: dict[str, str] = {}
        if graph_db_path is not None:
            db_overrides["graph_db_path"] = str(graph_db_path)
        if vector_db_path is not None:
            db_overrides["vector_db_path"] = str(vector_db_path)
        config = config.model_copy(
            update={"database": config.database.model_copy(update=db_overrides)}
        )

    app = create_pipeline_graph(
        enable_verification=config.enable_verification,
        enable_refinement=config.enable_refinement,
        enable_entity_graph=config.enable_entity_graph,
        enable_reporting=config.enable_reporting,
    )

    initial_state: PipelineState = {
        "video_path": video_path,
        "run_dir": run_dir,
        "config": config,
        "clips": [],
        "segmentation_complete": False,
        "temp_dir": None,
        "clip_path_map": {},
        "verifications": [],
        "verification_complete": False,
        "iteration": 0,
        "refinement_needed": False,
        "refined_clip_ids": [],
        "entities": [],
        "relationships": [],
        "frames": [],
        "embeddings": [],
        "linked_entities": [],
        "linked_relationships": [],
        "db_paths": {},
        "status": "initialized",
        "error": None,
    }

    # Run
    logger.info("=" * 60)
    logger.info("Starting Unified Pipeline (Ingestion + Entity Graph)")
    logger.info(f"Video: {video_path}")
    logger.info(f"Verification: {'enabled' if config.enable_verification else 'disabled'}")
    logger.info(f"Refinement: {'enabled' if config.enable_refinement else 'disabled'}")
    logger.info(f"Entity Graph: {'enabled' if config.enable_entity_graph else 'disabled'}")
    logger.info(f"Reporting: {'enabled' if config.enable_reporting else 'disabled'}")
    logger.info("=" * 60)

    final_state = app.invoke(initial_state)

    # Summary
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info(f"Final status: {final_state['status']}")
    logger.info(f"Total clips: {len(final_state['clips'])}")
    logger.info(f"Iterations: {final_state['iteration']}")
    if final_state.get("verifications"):
        valid = sum(1 for v in final_state["verifications"] if v.is_valid)
        total = len(final_state["verifications"])
        logger.info(f"Validation: {valid}/{total} valid")
    if final_state.get("db_paths"):
        for db_name, db_path in final_state["db_paths"].items():
            logger.info(f"{db_name}: {db_path}")
    logger.info("=" * 60)

    return final_state
