#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HOT3D benchmark runner: ingestion (via run_benchmark.py) + HOT3D evaluation.

Usage:
    python scripts/run_benchmark_hot3d.py \\
        -c configs/benchmark_hot3d.yaml \\
        --ground-truth data/benchmark/hot3d/ground_truth_clips.jsonl

    # Limit to specific videos for a smoke test
    python scripts/run_benchmark_hot3d.py \\
        -c configs/benchmark_hot3d.yaml \\
        --ground-truth data/benchmark/hot3d/ground_truth_clips.jsonl \\
        --video-ids P0001_4bf4e21a P0002_36766e32

    # Only run evaluation against an existing predictions file
    python scripts/run_benchmark_hot3d.py \\
        --skip-ingestion \\
        -c configs/benchmark_hot3d.yaml \\
        --ground-truth data/benchmark/hot3d/ground_truth_clips.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from video_ingestion_agent.ingestion import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _aggregate_predictions(benchmark_dir: Path) -> Path:
    """Concatenate per-video clips_final.jsonl files into all_predictions.jsonl.

    Tags each row with the video_id derived from the run subdirectory name.
    """
    out = benchmark_dir / "all_predictions.jsonl"
    n = 0
    with out.open("w") as out_fh:
        for run_dir in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
            clips = run_dir / "clips_final.jsonl"
            if not clips.exists():
                continue
            video_id = run_dir.name
            with clips.open() as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    row = json.loads(raw)
                    row.setdefault("video_id", video_id)
                    out_fh.write(json.dumps(row) + "\n")
                    n += 1
    logger.info("Aggregated %d predictions → %s", n, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="HOT3D ground-truth JSONL produced by build_v2p_ground_truth.py.",
    )
    parser.add_argument("--video-ids", nargs="*", default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override config.paths.runs_dir.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip the agent run; just aggregate + evaluate.",
    )
    parser.add_argument(
        "--no-object-similarity",
        action="store_true",
        help="Skip the sentence-transformer object-similarity metric in eval.",
    )
    parser.add_argument(
        "--tiou-thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    benchmark_dir = args.output_dir or Path(config.paths.runs_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Ingestion ----------
    if not args.skip_ingestion:
        cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "run_benchmark.py"),
            "-c", str(args.config),
            "--output-dir", str(benchmark_dir),
            "--ingest-only",
            "--resume",
            "--num-gpus", str(args.num_gpus),
        ]
        if args.video_ids:
            cmd.extend(["--video-ids", *args.video_ids])
        logger.info("Running ingestion: %s", " ".join(cmd))
        ret = subprocess.run(cmd, env=os.environ.copy())
        if ret.returncode != 0:
            logger.error("Ingestion failed (exit %d)", ret.returncode)
            sys.exit(ret.returncode)

    # ---------- Aggregate ----------
    predictions_path = _aggregate_predictions(benchmark_dir)
    if predictions_path.stat().st_size == 0:
        logger.error("No predictions to evaluate; aborting.")
        sys.exit(1)

    # ---------- Evaluate ----------
    eval_cmd = [
        sys.executable, "-m", "video_ingestion_agent.benchmark.evaluate_hot3d",
        "--predictions", str(predictions_path),
        "--ground-truth", str(args.ground_truth),
        "--output", str(benchmark_dir / "eval_results.json"),
        "--tiou-thresholds", *[str(t) for t in args.tiou_thresholds],
    ]
    if args.video_ids:
        eval_cmd.extend(["--video-filter", *args.video_ids])
    if args.no_object_similarity:
        eval_cmd.append("--no-object-similarity")
    logger.info("Running evaluation: %s", " ".join(eval_cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_PROJECT_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    ret = subprocess.run(eval_cmd, env=env)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
