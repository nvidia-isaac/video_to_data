# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LangGraph nodes for entity graph building and reporting.

Nodes:
  entity_extraction_node  -- extract entities/relationships from clip descriptions
  frame_embedding_node    -- extract frame embeddings within clip boundaries
  entity_linking_node     -- link and merge duplicate entities across clips
  database_write_node     -- write entity graph + embeddings to databases
  reporting_node          -- generate HTML report

Each node receives the full PipelineState and returns a partial state dict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from video_ingestion_agent.ingestion.entity_graph.database_writer import DatabaseWriter
from video_ingestion_agent.ingestion.entity_graph.entity_linker import EntityLinker
from video_ingestion_agent.ingestion.entity_graph.extractors import EntityExtractor, VisualExtractor
from video_ingestion_agent.ingestion.state import (
    ClipContext,
    PipelineState,
    clips_to_action_segments,
)
from video_ingestion_agent.utils.vector_database import VectorDatabase
from video_ingestion_agent.utils.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


def _valid_clips(state: PipelineState) -> list[ClipContext]:
    """Return only clips that passed verification.

    If verification was not run (no verifications in state), all clips
    are returned unchanged.
    """
    clips = state["clips"]
    verifications = state.get("verifications", [])
    if not verifications:
        return clips
    valid_ids = {v.clip_id for v in verifications if v.is_valid}
    filtered = [c for c in clips if c.clip_id in valid_ids]
    if len(filtered) < len(clips):
        logger.info(
            f"Filtered to {len(filtered)}/{len(clips)} valid clips "
            f"(excluded {len(clips) - len(filtered)} invalid)"
        )
    return filtered


# ---------------------------------------------------------------------------
# Entity Extraction
# ---------------------------------------------------------------------------


def entity_extraction_node(state: PipelineState) -> dict[str, Any]:
    """Node: Extract entities and relationships from clip descriptions."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: ENTITY EXTRACTION")
    logger.info("=" * 60)

    clips = _valid_clips(state)
    config = state["config"]

    llm_model = config.models.llm_model or config.models.vlm_model
    # Preserve legacy behavior: when llm_backend is omitted, inherit vlm_backend.
    llm_backend = (
        config.models.llm_backend
        if "llm_backend" in config.models.model_fields_set
        else config.models.vlm_backend
    )

    extractor = EntityExtractor(
        llm_model=llm_model,
        device=config.models.device,
        backend=llm_backend,
        api_key=config.models.api_key,
        api_url=config.models.vllm_url,
        save_responses=config.logging.save_responses,
        response_dir=config.logging.response_dir,
    )

    all_entities = []
    all_relationships = []

    for i, clip in enumerate(clips):
        logger.info(
            f"Processing clip {i + 1}/{len(clips)}: "
            f"[{clip.start_t:.1f}s-{clip.end_t:.1f}s] {clip.action}"
        )

        description = clip.description or f"{clip.action} {clip.object}"
        entities, relationships = extractor.extract_from_caption(description, caption_idx=i)

        for entity in entities:
            entity.first_seen = clip.start_t
            entity.last_seen = clip.end_t

        for rel in relationships:
            rel.start_t = clip.start_t
            rel.end_t = clip.end_t
            rel.supporting_evidence = description

        all_entities.extend(entities)
        all_relationships.extend(relationships)

    logger.info(f"Extracted {len(all_entities)} entities, {len(all_relationships)} relationships")

    return {
        "entities": all_entities,
        "relationships": all_relationships,
        "status": "entities_extracted",
    }


# ---------------------------------------------------------------------------
# Frame Embeddings
# ---------------------------------------------------------------------------


def frame_embedding_node(state: PipelineState) -> dict[str, Any]:
    """Node: Extract frame embeddings within clip boundaries."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: FRAME EMBEDDINGS")
    logger.info("=" * 60)

    video_path = state["video_path"]
    clips = _valid_clips(state)
    config = state["config"]

    visual_extractor = VisualExtractor(
        vlm_model=config.models.vlm_model,
        embedding_model=config.models.embedding_model,
        device=config.models.device,
        chunk_size=config.segmentation.chunk_size,
        chunk_overlap=config.segmentation.chunk_overlap,
        vlm_backend=config.models.vlm_backend,
        api_key=config.models.api_key,
    )

    fps = config.processing.fps
    processor = VideoProcessor(str(video_path))
    all_frames = list(processor.extract_frames(fps=fps))
    logger.info(f"Extracted {len(all_frames)} total frames at {fps} fps")

    # Filter to frames within clip boundaries, tagging each with its segment/clip_id
    def find_clip(timestamp: float) -> ClipContext | None:
        for clip in clips:
            if clip.start_t <= timestamp <= clip.end_t:
                return clip
        return None

    frames = []
    for f in all_frames:
        clip = find_clip(f.timestamp)
        if clip is not None:
            f.metadata = {**(f.metadata or {}), "segment_id": clip.clip_id}
            frames.append(f)
    logger.info(f"Filtered to {len(frames)} frames within clip boundaries")

    batch_size = config.models.embedding_batch_size
    if frames:
        embeddings = visual_extractor.extract_embeddings(frames, batch_size=batch_size)
        logger.info(f"Extracted {len(embeddings)} embeddings")
    else:
        embeddings = []
        logger.warning("No frames within clip boundaries - skipping embeddings")

    return {
        "frames": frames,
        "embeddings": embeddings,
        "status": "embeddings_extracted",
    }


# ---------------------------------------------------------------------------
# Entity Linking
# ---------------------------------------------------------------------------


def entity_linking_node(state: PipelineState) -> dict[str, Any]:
    """Node: Link and merge duplicate entities across clips."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: ENTITY LINKING")
    logger.info("=" * 60)

    entities = state["entities"]
    relationships = state["relationships"]
    config = state["config"]

    max_time_gap = config.entity_extraction.max_time_gap

    linker = EntityLinker(max_time_gap=max_time_gap)
    linked_entities = linker.link_entities(entities)
    linked_relationships = linker.link_relationships(relationships)

    logger.info(f"Linked entities: {len(entities)} -> {len(linked_entities)}")

    return {
        "linked_entities": linked_entities,
        "linked_relationships": linked_relationships,
        "status": "entities_linked",
    }


# ---------------------------------------------------------------------------
# Database Write
# ---------------------------------------------------------------------------


def database_write_node(state: PipelineState) -> dict[str, Any]:
    """Node: Write entity graph and embeddings to databases."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: DATABASE WRITE")
    logger.info("=" * 60)

    video_path = state["video_path"]
    clips = _valid_clips(state)
    entities = state["linked_entities"]
    relationships = state["linked_relationships"]
    frames = state["frames"]
    embeddings = state["embeddings"]
    config = state["config"]

    db = config.database
    output_dir = Path(db.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_dim = db.embedding_dim

    graph_db_path = Path(db.graph_db_path) if db.graph_db_path else output_dir / "graph.db"
    vector_db_path = Path(db.vector_db_path) if db.vector_db_path else output_dir / "vector.db"
    graph_db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Get video metadata
    processor = VideoProcessor(str(video_path))
    metadata = processor.get_metadata()

    # Write to graph database
    db_writer = DatabaseWriter(str(graph_db_path))
    video_id = db_writer.write_video_metadata(metadata)
    logger.info(f"Video registered with ID: {video_id}")

    db_writer.write_entities(entities, video_id=video_id)
    db_writer.write_relationships(relationships, video_id=video_id)

    # Convert clips to action segments for DB
    action_segments = clips_to_action_segments(clips)
    db_writer.write_action_segments(action_segments, video_id=video_id)

    db_writer.close()

    # Write to vector database
    vector_db = VectorDatabase(str(vector_db_path), embedding_dim=embedding_dim)

    video_id_str = Path(video_path).stem
    vector_db.add_video(
        video_id=video_id_str,
        path=str(video_path),
        duration=metadata.duration,
        fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
    )

    frame_data = []
    for frame, (frame_id, embedding) in zip(frames, embeddings, strict=False):
        segment_id = (frame.metadata or {}).get("segment_id")
        frame_data.append(
            (frame_id, video_id_str, frame.timestamp, embedding, frame.metadata, segment_id)
        )

    if frame_data:
        vector_db.add_frames_batch(frame_data)
    logger.info(f"Stored {len(frame_data)} frame embeddings")

    logger.info(f"Graph DB: {graph_db_path}")
    logger.info(f"Vector DB: {vector_db_path}")

    return {
        "db_paths": {"graph_db": str(graph_db_path), "vector_db": str(vector_db_path)},
        "status": "db_written",
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def reporting_node(state: PipelineState) -> dict[str, Any]:
    """Node: Generate HTML report for pipeline results."""
    logger.info("=" * 60)
    logger.info("LangGraph Node: REPORTING")
    logger.info("=" * 60)

    from .report import generate_html_report

    clips = state["clips"]
    run_dir = state["run_dir"]
    verifications = state.get("verifications", [])
    config = state["config"]

    config_summary = {
        "verification_enabled": len(verifications) > 0,
        "refinement_iterations": state.get("iteration", 0),
        "entity_graph_enabled": config.enable_entity_graph,
        "chunk_size": config.segmentation.chunk_size,
        "chunk_overlap": config.segmentation.chunk_overlap,
    }

    report_path = generate_html_report(
        clips=clips,
        run_dir=run_dir,
        verifications=verifications if verifications else None,
        config_summary=config_summary,
    )

    logger.info(f"HTML report generated: {report_path}")

    return {"status": "report_generated"}
