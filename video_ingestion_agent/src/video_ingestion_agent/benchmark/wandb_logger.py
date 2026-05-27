# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Weights & Biases logging for benchmark results.

Logs evaluation metrics, pipeline config, and result artifacts to W&B
for experiment tracking and comparison.

Usage:
    from video_ingestion_agent.benchmark.wandb_logger import log_to_wandb

    log_to_wandb(
        eval_results=eval_results.to_dict(),
        benchmark_meta=meta,
        config=config,
        benchmark_dir=benchmark_dir,
        project="v2p-benchmark",
    )
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def log_to_wandb(
    eval_results: dict,
    benchmark_meta: dict,
    config: object,
    benchmark_dir: Path,
    project: str,
    run_name: str | None = None,
    entity: str = "nvidia-isaac",
) -> None:
    """Log benchmark results to Weights & Biases.

    Args:
        eval_results: Evaluation results dict (from eval_results.json).
        benchmark_meta: Benchmark metadata dict (from benchmark_meta.json).
        config: Pipeline config object.
        benchmark_dir: Path to benchmark output directory.
        project: W&B project name.
        run_name: Optional W&B run name.
        entity: W&B entity (team/org). Defaults to "nvidia-isaac".
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed. Skipping W&B logging. Install with: pip install wandb")
        return

    # Flatten config for W&B
    wandb_config = {
        "vlm_model": config.models.vlm_model,
        "vlm_backend": config.models.vlm_backend,
        "vlm_fps": config.models.vlm_fps,
        "vllm_tp_size": getattr(config.models, "vllm_tp_size", 1),
        "embedding_model": config.models.embedding_model,
        "chunk_size": config.segmentation.chunk_size,
        "chunk_overlap": config.segmentation.chunk_overlap,
        "min_clip_s": config.segmentation.min_clip_s,
        "max_clip_s": config.segmentation.max_clip_s,
        "verification_enabled": config.enable_verification,
        "refinement_enabled": config.enable_refinement,
        "max_iterations": config.verification.max_iterations,
        "num_gpus": benchmark_meta.get("num_gpus", 1),
        "total_videos": benchmark_meta.get("total_videos", 0),
        "mode": benchmark_meta.get("mode", "unknown"),
    }

    # Add evaluation metadata to config (n_videos, n_segments, thresholds, etc.)
    eval_meta = eval_results.get("metadata", {})
    for key, value in eval_meta.items():
        if isinstance(value, (int, float, str, bool)):
            wandb_config[f"eval/{key}"] = value
        elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
            wandb_config[f"eval/{key}"] = value

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        config=wandb_config,
        tags=["benchmark", "epic-kitchens", config.models.vlm_backend],
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = {}

    # Temporal metrics
    temporal = eval_results.get("temporal", {})
    if "mAP" in temporal:
        for key, value in temporal["mAP"].items():
            if not key.startswith("_") and isinstance(value, (int, float)):
                metrics[f"temporal/mAP/{key}"] = value

    if "boundary" in temporal:
        for key, value in temporal["boundary"].items():
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    if isinstance(v2, (int, float)):
                        metrics[f"temporal/boundary/{key}/{k2}"] = v2
            elif isinstance(value, (int, float)):
                metrics[f"temporal/boundary/{key}"] = value

    if "segmentation_ratio" in temporal:
        sr = temporal["segmentation_ratio"]
        for key, value in sr.items():
            if key == "per_video":
                continue  # skip verbose per-video breakdown
            if isinstance(value, (int, float)):
                metrics[f"temporal/seg_ratio/{key}"] = value

    # Annotation metrics
    annotation = eval_results.get("annotation", {})
    if "accuracy" in annotation:
        acc = annotation["accuracy"]
        for key, value in acc.items():
            if isinstance(value, (int, float)):
                metrics[f"annotation/accuracy/{key}"] = value

    if "semantic_similarity" in annotation:
        ss = annotation["semantic_similarity"]
        for key, value in ss.items():
            if isinstance(value, (int, float)):
                metrics[f"annotation/semantic_similarity/{key}"] = value

    if "soft_match" in annotation:
        sm = annotation["soft_match"]
        for key, value in sm.items():
            if isinstance(value, (int, float)):
                metrics[f"annotation/soft_match/{key}"] = value

    if "bertscore" in annotation:
        bs = annotation["bertscore"]
        for key, value in bs.items():
            if isinstance(value, (int, float)):
                metrics[f"annotation/bertscore/{key}"] = value

    # Timing
    if "total_elapsed_s" in benchmark_meta:
        metrics["timing/total_elapsed_s"] = benchmark_meta["total_elapsed_s"]
        total_videos = benchmark_meta.get("total_videos", 1)
        metrics["timing/avg_per_video_s"] = benchmark_meta["total_elapsed_s"] / max(total_videos, 1)

    wandb.log(metrics)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    artifact = wandb.Artifact(
        name=f"benchmark-results-{run.id}",
        type="benchmark",
    )
    for fname in [
        "eval_results.json",
        "benchmark_meta.json",
        "all_predictions.jsonl",
        "mapped_predictions.jsonl",
        "benchmark_report.html",
    ]:
        fpath = benchmark_dir / fname
        if fpath.exists():
            artifact.add_file(str(fpath))
    run.log_artifact(artifact)

    logger.info(f"W&B run: {run.url}")
    wandb.finish()
