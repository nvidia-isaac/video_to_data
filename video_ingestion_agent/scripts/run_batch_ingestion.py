#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Batch video ingestion: discover videos, shard across GPUs, run full pipeline.

Designed for large-scale ingestion (1000+ videos) on a multi-GPU cluster.
A single main process discovers all videos, probes their durations, and
distributes them across N parallel worker sub-processes using duration-aware
LPT (Longest Processing Time first) sharding for near-optimal load balance.

All workers write into a shared graph.db + vector.db (SQLite WAL mode
handles concurrency).

Usage:
  # Single-machine, sequential
  python scripts/run_batch_ingestion.py \\
      --input-dir /path/to/videos \\
      -c configs/ingestion.yaml \\
      --output-dir runs/batch_ingest

  # Multi-GPU parallel (recommended)
  python scripts/run_batch_ingestion.py \\
      --input-dir /path/to/videos \\
      -c configs/ingestion.yaml \\
      --output-dir runs/batch_ingest \\
      --num-shards 8 --resume
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from video_ingestion_agent.ingestion import PipelineConfig, load_config
from video_ingestion_agent.utils.sharding import (
    discover_videos,
    filter_processed_videos,
    process_videos,
    run_parallel_workers,
    shard_videos_lpt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resume Support
# ---------------------------------------------------------------------------


def get_processed_videos(graph_db_path: Path) -> set[str]:
    """
    Query video_metadata table for already-processed video paths.

    Returns an empty set if the DB does not exist yet.
    """
    if not graph_db_path.exists():
        return set()
    try:
        conn = sqlite3.connect(str(graph_db_path))
        cursor = conn.execute("SELECT video_path FROM video_metadata")
        paths = {row[0] for row in cursor.fetchall()}
        conn.close()
        return paths
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch video ingestion into entity graph DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory containing video files (searched recursively)",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Shared output directory (graph.db + vector.db written here)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of parallel workers. Uses duration-aware LPT sharding "
        "for near-optimal load balance. Default: 1 (sequential).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos already present in graph.db",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Process at most N videos (useful for testing)",
    )

    # Internal arguments used by worker sub-processes (not for end-user use)
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--video-paths", nargs="+", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Validate
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared DB paths (all workers write here)
    graph_db_path = output_dir / "graph.db"
    vector_db_path = output_dir / "vector.db"

    # ---- Determine video list ----
    if args.worker_mode and args.video_paths:
        # Worker sub-process: use the explicitly provided video list
        my_videos = [Path(p) for p in args.video_paths]
        worker_id = args.worker_id
    else:
        # Main process: discover all videos
        my_videos = discover_videos(input_dir)
        worker_id = 0
        logger.info(f"Discovered {len(my_videos)} videos under {input_dir}")

    # ---- Resume ----
    if args.resume:
        already_done = get_processed_videos(graph_db_path)
        my_videos = filter_processed_videos(my_videos, already_done)

    # ---- Optional cap ----
    if args.max_videos is not None:
        my_videos = my_videos[: args.max_videos]
        logger.info(f"Capped to {len(my_videos)} videos (--max-videos)")

    if not my_videos:
        logger.info("Nothing to process. Exiting.")
        return

    # ---- Logging per worker ----
    shard_log = output_dir / f"worker_{worker_id}.log"
    file_handler = logging.FileHandler(str(shard_log))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # ================================================================
    # Parallel mode (main process only, not worker sub-processes)
    # ================================================================
    if not args.worker_mode and args.num_shards > 1 and len(my_videos) > 1:
        logger.info("=" * 60)
        logger.info("Batch Ingestion (PARALLEL)")
        logger.info(f"  Videos: {len(my_videos)}")
        logger.info(f"  Workers: {args.num_shards}")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Output: {output_dir}")
        logger.info("=" * 60)

        # Save metadata
        meta = {
            "start_time": datetime.now().isoformat(),
            "config_path": str(config_path),
            "input_dir": str(input_dir),
            "total_videos": len(my_videos),
            "num_shards": args.num_shards,
            "mode": "parallel",
        }
        meta_path = output_dir / "ingestion_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        shards = shard_videos_lpt(my_videos, args.num_shards)

        base_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input-dir",
            str(args.input_dir),
            "-c",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--worker-mode",
        ]
        if args.resume:
            base_cmd.append("--resume")
        if args.max_videos is not None:
            base_cmd.extend(["--max-videos", str(args.max_videos)])

        def _shard_args(wid: int, shard: list[Path]) -> list[str]:
            return ["--worker-id", str(wid), "--video-paths"] + [str(v) for v in shard]

        def _env_for_worker(wid: int) -> dict[str, str]:
            """Assign one GPU per worker to avoid VRAM contention (OOM)."""
            env = os.environ.copy()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible:
                devices = [d.strip() for d in visible.split(",") if d.strip()]
            else:
                try:
                    import torch

                    devices = [str(i) for i in range(torch.cuda.device_count())]
                except Exception:
                    devices = ["0"]
            if devices:
                env["CUDA_VISIBLE_DEVICES"] = devices[wid % len(devices)]
            return env

        start_time = time.time()
        run_parallel_workers(
            shards=shards,
            base_cmd=base_cmd,
            shard_to_args=_shard_args,
            output_dir=output_dir,
            env_for_worker=_env_for_worker,
        )
        total_elapsed = time.time() - start_time
        logger.info(f"\nAll workers finished in {total_elapsed:.1f}s")
        return

    # ================================================================
    # Sequential mode (single worker or worker sub-process)
    # ================================================================
    logger.info(f"Loading config from {config_path}")
    config: PipelineConfig = load_config(config_path)

    t_start_all = time.time()
    results = process_videos(
        videos=my_videos,
        output_dir=output_dir,
        config=config,
        graph_db_path=graph_db_path,
        vector_db_path=vector_db_path,
        worker_id=worker_id,
        resume=args.resume,
    )

    # ---- Summary ----
    elapsed_all = time.time() - t_start_all
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")

    logger.info("\n" + "=" * 60)
    logger.info(f"Worker {worker_id} complete!")
    logger.info(f"  Total videos: {len(my_videos)}")
    logger.info(f"  Success: {success}")
    logger.info(f"  Failed:  {failed}")
    logger.info(f"  Elapsed: {elapsed_all:.0f}s ({elapsed_all / 60:.1f}m)")
    logger.info(f"  Graph DB: {graph_db_path}")
    logger.info(f"  Vector DB: {vector_db_path}")
    logger.info("=" * 60)

    # Write final summary JSON
    summary_path = output_dir / f"summary_worker_{worker_id}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "worker_id": worker_id,
                "total_videos": len(my_videos),
                "success": success,
                "failed": failed,
                "elapsed_s": round(elapsed_all, 1),
                "timestamp": datetime.now().isoformat(),
                "results": results,
            },
            f,
            indent=2,
        )

    if failed > 0:
        logger.warning(f"{failed} video(s) failed. See worker log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
