#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run the unified ingestion + entity graph pipeline.

Single script that processes a video through:
  1. Action segmentation (chunked VLM)
  2. Verification & refinement (temp clip extraction, VLM critic)
  3. Entity graph building (entity extraction, embeddings, linking, DB write)
  4. HTML report generation
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_ingestion_agent.ingestion import load_config, run_pipeline

# Logging is configured after config is loaded so that the YAML
# logging.level setting is respected.  We set a basic fallback here
# in case early errors occur before config is parsed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the unified pipeline."""
    parser = argparse.ArgumentParser(
        description="Run unified video ingestion + entity graph pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (segmentation + verification + entity graph + report)
  python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml

  # Segmentation + verification only (no entity graph)
  python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml --no-entity-graph

  # Segmentation only (no verification, no entity graph)
  python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml --no-verify --no-entity-graph

  # Custom output directory
  python scripts/run_ingestion.py video.mp4 -c configs/ingestion.yaml -o runs/my_experiment
        """,
    )

    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to pipeline configuration YAML file"
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Output directory (default: runs/<timestamp>)"
    )
    parser.add_argument("--run-name", default=None, help="Name for this run (default: timestamp)")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification/refinement")
    parser.add_argument(
        "--no-refine", action="store_true", help="Disable refinement (verify only, no loop)"
    )
    parser.add_argument("--no-entity-graph", action="store_true", help="Skip entity graph building")
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max refinement iterations",
    )

    args = parser.parse_args()

    # Validate inputs
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    logger.info(f"Loading configuration from {config_path}")
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Apply logging level from config
    log_level = getattr(config, "logging", None)
    if log_level and hasattr(log_level, "level"):
        level_str = log_level.level.upper()
    else:
        level_str = "INFO"
    numeric_level = getattr(logging, level_str, logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    logger.info(f"Log level set to: {level_str}")

    # Apply CLI overrides
    if args.no_verify:
        config.enable_verification = False
        config.enable_refinement = False
    if args.no_refine:
        config.enable_refinement = False
    if args.no_entity_graph:
        config.enable_entity_graph = False
    if args.no_report:
        config.enable_reporting = False
    if args.max_iterations is not None:
        config.verification.max_iterations = args.max_iterations

    # Setup run directory
    if args.output:
        run_dir = Path(args.output)
    else:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(config.paths.runs_dir) / run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    # Also add a file log handler
    log_file = run_dir / "pipeline.log"
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Run directory: {run_dir}")

    # Print configuration summary
    logger.info("=" * 60)
    logger.info("Pipeline Configuration:")
    logger.info(f"  Video: {video_path}")
    logger.info(f"  VLM Model: {config.models.vlm_model} ({config.models.vlm_backend})")
    logger.info(f"  Chunk Size: {config.segmentation.chunk_size}s")
    logger.info(f"  Chunk Overlap: {config.segmentation.chunk_overlap}s")
    logger.info(f"  Verification: {'enabled' if config.enable_verification else 'disabled'}")
    logger.info(f"  Refinement: {'enabled' if config.enable_refinement else 'disabled'}")
    if config.enable_refinement:
        logger.info(f"  Max Iterations: {config.verification.max_iterations}")
    logger.info(f"  Entity Graph: {'enabled' if config.enable_entity_graph else 'disabled'}")
    logger.info(f"  Reporting: {'enabled' if config.enable_reporting else 'disabled'}")
    logger.info("=" * 60)

    # Run pipeline
    try:
        final_state = run_pipeline(
            video_path=str(video_path),
            run_dir=run_dir,
            config=config,
        )

        # Print summary
        logger.info("=" * 60)
        logger.info("Pipeline Summary:")
        logger.info(f"  Video: {video_path}")
        logger.info(f"  Total clips: {len(final_state['clips'])}")
        logger.info(f"  Iterations: {final_state['iteration']}")
        logger.info(f"  Final status: {final_state['status']}")

        if final_state.get("verifications"):
            valid = sum(1 for v in final_state["verifications"] if v.is_valid)
            total = len(final_state["verifications"])
            logger.info(f"  Verified: {valid}/{total} valid ({valid / total:.1%})")

        if final_state.get("db_paths"):
            for name, path in final_state["db_paths"].items():
                logger.info(f"  {name}: {path}")

        if config.enable_reporting:
            report_path = run_dir / "report.html"
            if report_path.exists():
                logger.info(f"  Report: file://{report_path.absolute()}")

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
