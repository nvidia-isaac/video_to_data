#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Query video using LangGraph-based agentic retrieval.

Supports multi-video databases - clips can be extracted from any video
in the database.

Usage:
    python scripts/run_retrieval.py "Find clips where someone picks up a cup" \
        -d outputs/my_video/ \
        -c configs/retrieval.yaml

    # Override video path (if different from what's in database)
    python scripts/run_retrieval.py "..." -d outputs/my_video/ --video /new/path.mp4

Examples:
    # Find specific action
    python scripts/run_retrieval.py "Find all clips where someone picks up an object" -d outputs/my_video/

    # Find by object
    python scripts/run_retrieval.py "Show me interactions with the red cup" -d outputs/my_video/

    # Time-bounded query
    python scripts/run_retrieval.py "What happens in the first 30 seconds?" -d outputs/my_video/
"""

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_ingestion_agent.retrieval import RetrievalAgent, RetrievalConfig, load_retrieval_config
from video_ingestion_agent.retrieval.tools import ExtractClipTool, SearchFramesTool, SearchGraphTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_all_video_paths_from_db(db_path: str) -> dict[int, str]:
    """Read all video paths from database's video_metadata table.

    Returns:
        Dict mapping video_id to video_path
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT id, video_path FROM video_metadata ORDER BY id")
    result = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return result


def get_video_path_from_db(db_path: str) -> str | None:
    """Read first video path from database (backward compat)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT video_path FROM video_metadata LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row:
        return row[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Query video using LangGraph-based agentic retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("query", help="Natural language query")

    parser.add_argument(
        "-d",
        "--database",
        required=True,
        help="Path to database directory containing graph.db and vector.db",
    )

    parser.add_argument("--video", help="Path to source video file (default: read from database)")

    parser.add_argument("-c", "--config", help="Path to agent configuration YAML")

    parser.add_argument(
        "--output-dir",
        default="outputs/clips",
        help="Directory for extracted clips (default: outputs/clips)",
    )

    parser.add_argument(
        "--max-sub-tasks",
        type=int,
        default=5,
        help="Maximum sub-tasks to decompose into (default: 5)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed agent working memory"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging of LLM inputs/outputs (or set VIDEO_INGESTION_AGENT_DEBUG=1)",
    )

    parser.add_argument(
        "--debug-dir",
        default="debug_logs",
        help="Directory for debug logs (default: debug_logs)",
    )

    args = parser.parse_args()

    # Validate database path (directory containing graph.db and vector.db)
    db_dir = Path(args.database)
    if not db_dir.exists():
        logger.error(f"Database directory not found: {db_dir}")
        sys.exit(1)

    # Support both directory and direct file path for backward compatibility
    if db_dir.is_file():
        # User passed a file path directly - derive paths from it
        graph_db_path = db_dir
        vector_db_path = db_dir.parent / f"{db_dir.stem}_vector.db"
    else:
        # Directory mode: expect graph.db and vector.db inside
        graph_db_path = db_dir / "graph.db"
        vector_db_path = db_dir / "vector.db"

    if not graph_db_path.exists():
        logger.error(f"Entity graph database not found: {graph_db_path}")
        sys.exit(1)

    if not vector_db_path.exists():
        logger.warning(f"Vector database not found: {vector_db_path}")
        logger.warning("Semantic frame search will be disabled")
        vector_db_path = None
    else:
        logger.info(f"Database directory: {db_dir}")
        logger.info(f"  Entity graph: {graph_db_path}")
        logger.info(f"  Vector DB:    {vector_db_path}")

    # Get all videos from database (multi-video support)
    video_paths = get_all_video_paths_from_db(str(graph_db_path))

    if not video_paths:
        logger.error("No videos found in database")
        sys.exit(1)

    logger.info(f"Found {len(video_paths)} video(s) in database:")
    missing_videos = []
    for vid, vpath in video_paths.items():
        exists = Path(vpath).exists()
        status = "OK" if exists else "MISSING"
        logger.info(f"  [{vid}] {vpath} ({status})")
        if not exists:
            missing_videos.append(vpath)

    # If user specified a video, use that as default/override
    default_video_path = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            sys.exit(1)
        default_video_path = str(video_path)
        logger.info(f"Using override video: {video_path}")
    elif missing_videos:
        logger.warning(f"Some videos are missing: {missing_videos}")
        logger.warning("Clips from missing videos cannot be extracted")

    # Load config
    if args.config:
        config = load_retrieval_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        config = RetrievalConfig()
        logger.info("No config file specified, using defaults")

    # Apply CLI overrides
    if args.max_sub_tasks != config.agent.max_sub_tasks:
        config = config.model_copy(
            update={"agent": config.agent.model_copy(update={"max_sub_tasks": args.max_sub_tasks})}
        )

    logger.info("Model configuration:")
    logger.info(f"  LLM: {config.models.llm_model} (backend: {config.models.llm_backend})")
    logger.info(f"  Embedding: {config.models.embedding_model}")
    logger.info(f"  Device: {config.models.device}")

    logger.info("Initializing tools...")

    # Initialize tools as dict (for LangGraph agent)
    tools = {}

    # Graph search tool
    graph_tool = SearchGraphTool(str(graph_db_path))
    tools["search_graph"] = graph_tool
    logger.info(f"  ✓ search_graph: {graph_db_path}")

    # Frame search tool (if vector db exists)
    if vector_db_path:
        try:
            frames_tool = SearchFramesTool(
                vector_db_path=str(vector_db_path),
                embedding_model=config.models.embedding_model,
                device=config.models.device,
            )
            tools["search_frames"] = frames_tool
            logger.info(f"  ✓ search_frames: {vector_db_path}")
        except Exception as e:
            logger.warning(f"  ✗ search_frames failed: {e}")

    # Clip extraction tool with multi-video support
    clip_tool = ExtractClipTool(
        video_paths=video_paths,
        video_path=default_video_path,  # Override if specified
        output_dir=args.output_dir,
    )
    tools["extract_clip"] = clip_tool
    logger.info(f"  ✓ extract_clip: {len(video_paths)} video(s) registered")

    # Initialize LangGraph agent
    logger.info("Initializing LangGraph agent...")

    agent = RetrievalAgent(
        config=config,
        tools=tools,
        debug=args.debug,
        debug_dir=args.debug_dir,
    )

    # Run query - use first video as default for backward compat
    first_video_path = default_video_path or list(video_paths.values())[0]
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Query: {args.query}")
    logger.info("=" * 60)

    result = agent.run(args.query, video_path=first_video_path)

    # Display results
    if args.verbose and result.get("working_memory"):
        print("\n" + "=" * 60)
        print("WORKING MEMORY:")
        print("=" * 60)
        for i, mem in enumerate(result["working_memory"], 1):
            print(f"\n--- Memory {i} ---")
            print(mem[:500] + "..." if len(mem) > 500 else mem)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result.get("answer", "No answer generated"))

    if result.get("clips_extracted"):
        print("\n" + "=" * 60)
        print("EXTRACTED CLIPS:")
        print("=" * 60)
        for clip in result["clips_extracted"]:
            print(f"  - {clip}")

    if result.get("sub_tasks"):
        print(f"\n(Processed {len(result['sub_tasks'])} sub-tasks)")

    if not result.get("success", False):
        print(f"\n⚠ Agent completed with error: {result.get('error')}")
        sys.exit(1)

    print("\n✅ Query completed successfully!")


if __name__ == "__main__":
    main()
