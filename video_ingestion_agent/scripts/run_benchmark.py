#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run the video_ingestion_agent segmentation pipeline on EPIC-KITCHENS-100 videos.

Loops over downloaded videos, runs the pipeline per-video, and collects
all clips_stage1.jsonl outputs into a single aggregated predictions file.

Usage:
    python scripts/run_benchmark.py \\
        -c configs/benchmark_epic_kitchens.yaml

    # Run on specific videos only
    python scripts/run_benchmark.py \\
        -c configs/benchmark_epic_kitchens.yaml \\
        --video-ids P01_01 P01_02

    # Skip verification for faster iteration
    python scripts/run_benchmark.py \\
        -c configs/benchmark_epic_kitchens.yaml \\
        --no-verify

    # Resume a previous run (skip already-processed videos)
    python scripts/run_benchmark.py \\
        -c configs/benchmark_epic_kitchens.yaml \\
        --resume

    # Parallel processing across 8 GPUs
    python scripts/run_benchmark.py \\
        -c configs/benchmark_epic_kitchens.yaml \\
        --num-gpus 8
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src and project root to path
_PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "src"))
sys.path.insert(0, _PROJECT_ROOT)

from video_ingestion_agent.ingestion import load_config  # noqa: E402
from video_ingestion_agent.ingestion.io import read_jsonl  # noqa: E402
from video_ingestion_agent.utils.sharding import (  # noqa: E402
    discover_videos as _discover_videos,
)
from video_ingestion_agent.utils.sharding import (  # noqa: E402
    filter_processed_videos,
    process_videos,
    run_parallel_workers,
    shard_videos_lpt,
    video_id_from_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def discover_videos(videos_dir: Path, extensions: list[str] | None = None) -> list[Path]:
    """Discover video files in the given directory.

    Thin wrapper around the shared utility that keeps the benchmark's
    original non-recursive, non-symlink-resolving behaviour.
    """
    exts = set(extensions) if extensions else {".mp4", ".MP4"}
    return _discover_videos(videos_dir, extensions=exts, resolve_symlinks=False)


def is_video_processed(run_dir: Path) -> bool:
    """Check if a video has already been processed (has clips_stage1.jsonl)."""
    clips_file = run_dir / "clips_stage1.jsonl"
    return clips_file.exists() and clips_file.stat().st_size > 0


def aggregate_predictions(
    benchmark_dir: Path,
    video_ids: list[str],
) -> Path:
    """
    Aggregate per-video clip predictions into a single predictions file.

    Reads ``clips_final.jsonl`` (post-NMS deduplicated output) from each
    video's run directory.

    Args:
        benchmark_dir: Root benchmark runs directory
        video_ids: List of video IDs to aggregate

    Returns:
        Path to the aggregated predictions file
    """
    output_path = benchmark_dir / "all_predictions.jsonl"
    all_clips = []

    for video_id in sorted(video_ids):
        video_run_dir = benchmark_dir / video_id

        clips_file = video_run_dir / "clips_final.jsonl"
        if not clips_file.exists():
            logger.warning(f"No clips_final.jsonl found for {video_id}, skipping")
            continue

        logger.info(f"Aggregating {video_id} from {clips_file.name}")

        for clip_data in read_jsonl(clips_file):
            # Tag with video_id for downstream processing
            clip_data["_video_id"] = video_id
            clip_data["_source_file"] = str(clips_file)
            all_clips.append(clip_data)

    # Write aggregated file
    with open(output_path, "w", encoding="utf-8") as f:
        for clip in all_clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")

    logger.info(f"Aggregated {len(all_clips)} clips from {len(video_ids)} videos to {output_path}")
    return output_path


def _launch_parallel_benchmark(
    args: argparse.Namespace,
    all_videos: list[Path],
    benchmark_dir: Path,
    config_path: Path,
    num_gpus: int,
) -> None:
    """
    Launch N subprocess workers to process video shards in parallel.

    With the "local" VLM backend, each worker gets a dedicated GPU via
    CUDA_VISIBLE_DEVICES. With the "vllm" backend, the vLLM server owns
    all GPUs and workers are lightweight HTTP clients -- CUDA_VISIBLE_DEVICES
    is left unset so workers can share GPUs for the embedding model (SigLIP).
    """
    # Detect backend from config
    from video_ingestion_agent.ingestion.config import load_config as _load_config

    cfg = _load_config(str(config_path))
    use_vllm = cfg.models.vlm_backend == "vllm"

    shards = shard_videos_lpt(all_videos, num_gpus)

    # Build the base command to forward to each worker
    base_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "-c",
        str(config_path),
        "--output-dir",
        str(benchmark_dir),
        "--ingest-only",
        "--resume",
    ]
    if args.no_verify:
        base_cmd.append("--no-verify")
    if args.no_refine:
        base_cmd.append("--no-refine")
    if args.max_iterations is not None:
        base_cmd.extend(["--max-iterations", str(args.max_iterations)])
    if args.videos_dir:
        base_cmd.extend(["--videos-dir", str(args.videos_dir)])

    if use_vllm:
        logger.info(
            f"Using vLLM backend -- launching {num_gpus} workers as HTTP clients "
            f"(VLM served by vLLM, GPUs shared for SigLIP embedding)"
        )

    def _shard_args(_wid: int, shard: list[Path]) -> list[str]:
        return ["--video-ids"] + [video_id_from_path(v) for v in shard]

    def _env_for_worker(wid: int) -> dict[str, str]:
        env = os.environ.copy()
        if not use_vllm:
            env["CUDA_VISIBLE_DEVICES"] = str(wid)
        return env

    run_parallel_workers(
        shards=shards,
        base_cmd=base_cmd,
        shard_to_args=_shard_args,
        output_dir=benchmark_dir,
        env_for_worker=_env_for_worker,
        log_prefix="worker_gpu",
    )


def main():
    """Run benchmark pipeline on EPIC-KITCHENS videos."""
    parser = argparse.ArgumentParser(
        description="Run video_ingestion_agent pipeline on EPIC-KITCHENS-100 videos for benchmarking"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/benchmark_epic_kitchens.yaml",
        help="Path to benchmark config YAML",
    )
    parser.add_argument(
        "--video-ids",
        nargs="+",
        default=None,
        help="Specific video IDs to process (e.g., P01_01 P01_02). Default: all.",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Override videos directory from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: from config runs_dir)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification/refinement for faster processing",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Run verification but skip refinement loop",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max refinement iterations",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos that have already been processed",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Limit number of videos to process",
    )
    parser.add_argument(
        "--annotations-dir",
        type=str,
        default="data/benchmark/epic_kitchens/annotations",
        help="Path to EPIC-KITCHENS annotations directory",
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore computation (slow)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for parallel processing. Each GPU runs a separate "
        "worker subprocess. Default: 1 (sequential).",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only run video ingestion (segmentation/verification). "
        "Skip aggregation, adapter, evaluation, and report. "
        "Used internally by parallel worker subprocesses.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name to log results to (e.g., 'v2p-benchmark'). "
        "If not set, results are not logged to W&B.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name. Defaults to experiment_name from config.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="nvidia-isaac",
        help="W&B entity (team/org). Defaults to 'nvidia-isaac'.",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Apply CLI overrides
    if args.no_verify:
        config.enable_verification = False
        config.enable_refinement = False
    if args.no_refine:
        config.enable_refinement = False
    if args.max_iterations is not None:
        config.verification.max_iterations = args.max_iterations

    # Determine directories
    videos_dir = Path(args.videos_dir or config.paths.input_videos_dir)
    benchmark_dir = Path(args.output_dir or config.paths.runs_dir)

    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        sys.exit(1)

    # Discover videos
    all_videos = discover_videos(videos_dir)
    logger.info(f"Found {len(all_videos)} videos in {videos_dir}")

    if not all_videos:
        logger.error("No video files found!")
        sys.exit(1)

    # Filter to requested video IDs
    if args.video_ids:
        requested = set(args.video_ids)
        all_videos = [v for v in all_videos if video_id_from_path(v) in requested]
        missing = requested - {video_id_from_path(v) for v in all_videos}
        if missing:
            logger.warning(f"Requested videos not found: {missing}")
        logger.info(f"Filtered to {len(all_videos)} requested videos")

    # Limit
    if args.max_videos:
        all_videos = all_videos[: args.max_videos]
        logger.info(f"Limited to {len(all_videos)} videos")

    # Filter already-processed videos
    if args.resume:
        all_videos = filter_processed_videos(
            all_videos,
            lambda v: is_video_processed(benchmark_dir / video_id_from_path(v)),
        )

    # ----- Video processing -----
    if all_videos:
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        if args.num_gpus > 1 and len(all_videos) > 1:
            # ==============================================================
            # PARALLEL MODE: spawn one subprocess per GPU
            # ==============================================================
            logger.info("=" * 60)
            logger.info("EPIC-KITCHENS Benchmark Run (PARALLEL)")
            logger.info(f"  Videos: {len(all_videos)}")
            logger.info(f"  GPUs: {args.num_gpus}")
            logger.info(f"  Config: {config_path}")
            logger.info(f"  Output: {benchmark_dir}")
            logger.info(
                f"  Verification: {'enabled' if config.enable_verification else 'disabled'}"
            )
            logger.info(f"  Refinement: {'enabled' if config.enable_refinement else 'disabled'}")
            logger.info("=" * 60)

            # Save benchmark metadata
            meta = {
                "start_time": datetime.now().isoformat(),
                "config_path": str(config_path),
                "videos_dir": str(videos_dir),
                "total_videos": len(all_videos),
                "video_ids": [video_id_from_path(v) for v in all_videos],
                "num_gpus": args.num_gpus,
                "mode": "parallel",
            }
            meta_path = benchmark_dir / "benchmark_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            start_time = time.time()
            _launch_parallel_benchmark(args, all_videos, benchmark_dir, config_path, args.num_gpus)
            total_elapsed = time.time() - start_time
            logger.info(f"\nAll parallel workers finished in {total_elapsed:.1f}s")

        else:
            # ==============================================================
            # SEQUENTIAL MODE (original behavior)
            # ==============================================================
            start_time = time.time()
            logger.info("=" * 60)
            logger.info("EPIC-KITCHENS Benchmark Run")
            logger.info(f"  Videos: {len(all_videos)}")
            logger.info(f"  Config: {config_path}")
            logger.info(f"  Output: {benchmark_dir}")
            logger.info(
                f"  Verification: {'enabled' if config.enable_verification else 'disabled'}"
            )
            logger.info(f"  Refinement: {'enabled' if config.enable_refinement else 'disabled'}")
            logger.info(
                f"  Entity Graph: {'enabled' if config.enable_entity_graph else 'disabled'}"
            )
            logger.info("=" * 60)

            # Save benchmark metadata
            meta = {
                "start_time": datetime.now().isoformat(),
                "config_path": str(config_path),
                "videos_dir": str(videos_dir),
                "total_videos": len(all_videos),
                "video_ids": [video_id_from_path(v) for v in all_videos],
                "verification_enabled": config.enable_verification,
                "refinement_enabled": config.enable_refinement,
                "mode": "sequential",
            }
            meta_path = benchmark_dir / "benchmark_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Process each video using the shared sequential loop.
            # Benchmark doesn't use a shared graph DB, so DB paths are omitted.
            # per_video_subdir="" places run dirs directly under benchmark_dir.
            result_list = process_videos(
                videos=all_videos,
                output_dir=benchmark_dir,
                config=config,
                per_video_subdir="",
            )

            # Save results summary
            total_elapsed = time.time() - start_time
            successful = sum(1 for r in result_list if r["status"] == "success")
            failed = sum(1 for r in result_list if r["status"] == "error")
            per_video = {r["video_id"]: r for r in result_list}

            results_summary = {
                "total_elapsed_s": total_elapsed,
                "total_videos": len(all_videos),
                "successful": successful,
                "failed": failed,
                "per_video": per_video,
            }

            results_path = benchmark_dir / "benchmark_results.json"
            with open(results_path, "w") as f:
                json.dump(results_summary, f, indent=2)

            logger.info(f"\n{'=' * 60}")
            logger.info("Video Processing Complete!")
            logger.info(f"  Total time: {total_elapsed:.1f}s")
            logger.info(f"  Successful: {successful}/{len(all_videos)}")
            logger.info(f"  Failed: {failed}/{len(all_videos)}")
            logger.info(f"  Results: {results_path}")
            logger.info("=" * 60)
    else:
        logger.info("No videos to process.")

    # ------------------------------------------------------------------
    # Early exit for --ingest-only (used by parallel worker subprocesses)
    # ------------------------------------------------------------------
    if args.ingest_only:
        logger.info("Ingest-only mode complete. Exiting.")
        return

    # ==================================================================
    # POST-PROCESSING: aggregate + adapter + evaluation + report
    # ==================================================================

    # Gather all successfully processed video IDs from output directories
    if not benchmark_dir.exists():
        logger.info("No benchmark output directory found. Nothing to evaluate.")
        return

    successful_ids = sorted(
        [d.name for d in benchmark_dir.iterdir() if d.is_dir() and is_video_processed(d)]
    )

    if not successful_ids:
        logger.info("No successfully processed videos found. Nothing to evaluate.")
        return

    logger.info(f"Found {len(successful_ids)} successfully processed videos")
    agg_path = aggregate_predictions(benchmark_dir, successful_ids)
    logger.info(f"Aggregated predictions: {agg_path}")

    # ------------------------------------------------------------------
    # Adapter: map free-text to EPIC verb/noun class IDs
    # ------------------------------------------------------------------
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.exists():
        logger.warning(
            f"Annotations dir not found: {annotations_dir} -- "
            "skipping adapter/evaluation. Run evaluate.py manually."
        )
        return

    logger.info("\n" + "=" * 60)
    logger.info("ADAPTER: Mapping predictions to EPIC verb/noun classes")
    logger.info("=" * 60)

    try:
        from video_ingestion_agent.benchmark.adapter import EpicKitchensAdapter
        from video_ingestion_agent.benchmark.load_epic_kitchens import EpicKitchensGT

        gt = EpicKitchensGT(
            annotations_dir,
            split="validation",
            video_filter=successful_ids,
        )

        adapter = EpicKitchensAdapter(
            verb_classes=gt.verb_classes,
            noun_classes=gt.noun_classes,
            verb_instances=gt.verb_instances,
            noun_instances=gt.noun_instances,
            device="cuda",
        )

        mapped = adapter.convert_predictions(agg_path)

        mapped_path = benchmark_dir / "mapped_predictions.jsonl"
        adapter.save_mapped_predictions(mapped, mapped_path)

        c2_path = benchmark_dir / "c2_submission.json"
        adapter.to_c2_submission(mapped, c2_path)

        logger.info(f"Mapped predictions: {mapped_path}")
        logger.info(f"C2 submission: {c2_path}")
    except Exception as e:
        logger.error(f"Adapter failed: {e}", exc_info=True)
        logger.info("Run adapter.py manually after fixing the issue.")
        return

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION: Computing benchmark metrics")
    logger.info("=" * 60)

    try:
        from video_ingestion_agent.benchmark.evaluate import run_evaluation

        eval_results = run_evaluation(
            predictions_file=mapped_path,
            annotations_dir=annotations_dir,
            skip_bertscore=args.skip_bertscore,
            skip_semantic=False,
        )

        eval_path = benchmark_dir / "eval_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results.to_dict(), f, indent=2)

        # Print summary
        temporal = eval_results.temporal
        annotation = eval_results.annotation

        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)

        if "mAP" in temporal:
            map_data = temporal["mAP"]
            for key in sorted(map_data.keys()):
                if key.startswith("_"):
                    continue
                logger.info(f"  {key}: {map_data[key]:.4f}")

        if "segmentation_ratio" in temporal:
            sr = temporal["segmentation_ratio"]
            logger.info(f"  seg_ratio_mean: {sr.get('mean_ratio', 0):.2f}")

        if "accuracy" in annotation:
            acc = annotation["accuracy"]
            logger.info(f"  verb_top1_acc: {acc.get('verb_top1_acc', 0):.4f}")
            logger.info(f"  noun_top1_acc: {acc.get('noun_top1_acc', 0):.4f}")

        if "semantic_similarity" in annotation:
            ss = annotation["semantic_similarity"]
            logger.info(f"  semantic_sim: {ss.get('mean_similarity', 0):.4f}")

        if "soft_match" in annotation:
            sm = annotation["soft_match"]
            logger.info(f"  verb_soft_match: {sm.get('verb_soft_match_rate', 0):.4f}")
            logger.info(f"  noun_soft_match: {sm.get('noun_soft_match_rate', 0):.4f}")

        logger.info(f"\n  Full results: {eval_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        logger.info("Run evaluate.py manually after fixing the issue.")
        return

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("REPORT: Generating HTML benchmark report")
    logger.info("=" * 60)

    try:
        from video_ingestion_agent.benchmark.report import generate_benchmark_report

        report_path = generate_benchmark_report(
            eval_results_path=eval_path,
            output_path=benchmark_dir / "benchmark_report.html",
            predictions_path=mapped_path,
            annotations_dir=annotations_dir,
        )
        logger.info(f"  Report: file://{report_path.absolute()}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # W&B Logging
    # ------------------------------------------------------------------
    if args.wandb_project:
        logger.info("\n" + "=" * 60)
        logger.info("WANDB: Logging results to Weights & Biases")
        logger.info("=" * 60)

        # Load benchmark metadata
        meta_path = benchmark_dir / "benchmark_meta.json"
        benchmark_meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                benchmark_meta = json.load(f)

        # Merge timing from results summary if available
        results_path = benchmark_dir / "benchmark_results.json"
        if results_path.exists():
            with open(results_path) as f:
                results_summary = json.load(f)
            benchmark_meta.update(
                {
                    k: v
                    for k, v in results_summary.items()
                    if k in ("total_elapsed_s", "successful", "failed")
                }
            )

        try:
            from video_ingestion_agent.benchmark.wandb_logger import log_to_wandb

            run_name = args.wandb_run_name or config.experiment_name
            log_to_wandb(
                eval_results=eval_results.to_dict(),
                benchmark_meta=benchmark_meta,
                config=config,
                benchmark_dir=benchmark_dir,
                project=args.wandb_project,
                run_name=run_name,
                entity=args.wandb_entity,
            )
        except Exception as e:
            logger.error(f"W&B logging failed: {e}", exc_info=True)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
