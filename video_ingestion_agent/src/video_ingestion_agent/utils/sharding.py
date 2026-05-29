# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Shared video discovery, sharding, and parallel worker utilities.

Used by both the benchmark runner and the batch ingestion script to
distribute videos across parallel workers with near-optimal load balance.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default video extensions to discover
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".MP4", ".MOV", ".MKV"})


# ---------------------------------------------------------------------------
# Video Discovery
# ---------------------------------------------------------------------------


def discover_videos(
    input_dir: Path,
    extensions: set[str] | frozenset[str] | None = None,
    resolve_symlinks: bool = True,
) -> list[Path]:
    """Recursively discover video files under *input_dir*.

    Symlinks are resolved by default so that the database stores canonical
    (NFS/host) paths rather than container-local symlink paths.  This
    ensures ``--resume`` works across different container runs.

    Returns a **sorted** list so that sharding is deterministic across
    workers.

    Args:
        input_dir: Root directory to scan.
        extensions: File extensions to match (case-sensitive).
        resolve_symlinks: If True, resolve symlinks to real paths.

    Returns:
        Sorted list of discovered video file paths.
    """
    extensions = extensions or VIDEO_EXTENSIONS

    if resolve_symlinks:
        videos = sorted(
            p.resolve() for p in input_dir.rglob("*") if p.suffix in extensions and p.is_file()
        )
    else:
        videos = sorted(p for p in input_dir.rglob("*") if p.suffix in extensions and p.is_file())

    # Deduplicate (e.g. .mp4 and .MP4 match the same file on case-insensitive FS)
    seen: set[Path] = set()
    unique: list[Path] = []
    for v in videos:
        key = v.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(v)

    return unique


def video_id_from_path(video_path: Path) -> str:
    """Extract video ID from path (e.g. ``P01_01`` from ``P01_01.MP4``)."""
    return video_path.stem


# ---------------------------------------------------------------------------
# Duration Probing
# ---------------------------------------------------------------------------


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe (fast, header-only read).

    Returns 0.0 if probing fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Duration-aware LPT Sharding
# ---------------------------------------------------------------------------


def shard_videos_lpt(
    videos: list[Path],
    num_shards: int,
) -> list[list[Path]]:
    """Duration-aware greedy sharding (Longest Processing Time first).

    Probes each video's duration via ``ffprobe``, sorts longest-first,
    then greedily assigns each video to the worker with the smallest
    total duration.  This produces near-optimal load balance when video
    lengths vary.

    Falls back to round-robin if duration probing fails for all videos.

    Args:
        videos: List of video paths to distribute.
        num_shards: Number of shards (typically one per GPU).

    Returns:
        List of *num_shards* lists, each containing a subset of the
        videos.
    """
    # Probe durations (fast -- ffprobe reads headers only)
    durations: list[tuple[Path, float]] = []
    for v in videos:
        dur = get_video_duration(v)
        durations.append((v, dur))

    # If all durations are zero (probe failed), fall back to round-robin
    if all(d == 0.0 for _, d in durations):
        logger.warning("Could not probe video durations; falling back to round-robin sharding")
        shards: list[list[Path]] = [[] for _ in range(num_shards)]
        for i, video in enumerate(videos):
            shards[i % num_shards].append(video)
        return shards

    # Sort longest-first for better greedy packing
    durations.sort(key=lambda x: x[1], reverse=True)

    # Greedy assignment: always give the next video to the lightest worker
    shard_loads = [0.0] * num_shards
    shards = [[] for _ in range(num_shards)]

    for video, dur in durations:
        lightest = min(range(num_shards), key=lambda i: shard_loads[i])
        shards[lightest].append(video)
        shard_loads[lightest] += dur

    # Log balance info
    total_dur = sum(shard_loads)
    if total_dur > 0:
        for i, load in enumerate(shard_loads):
            logger.info(
                f"  Shard {i}: {len(shards[i])} videos, "
                f"{load:.0f}s total ({load / total_dur * 100:.1f}%)"
            )

    return shards


# ---------------------------------------------------------------------------
# Resume Filtering
# ---------------------------------------------------------------------------


def filter_processed_videos(
    videos: list[Path],
    is_done: set[str] | Callable[[Path], bool],
) -> list[Path]:
    """Remove already-processed videos from the list.

    Args:
        videos: Full list of candidate videos.
        is_done: Either a **set of strings** (e.g. video paths already in
            the database) to check with ``str(video) in is_done``, or a
            **callable** that receives a ``Path`` and returns ``True`` if
            the video has already been processed.

    Returns:
        Filtered list with processed videos removed.  Logs the count of
        skipped videos.
    """
    before = len(videos)

    if isinstance(is_done, set):
        pending = [v for v in videos if str(v) not in is_done]
    else:
        pending = [v for v in videos if not is_done(v)]

    skipped = before - len(pending)
    if skipped:
        logger.info(f"Resume: {skipped} already processed, {len(pending)} remaining")
    return pending


# ---------------------------------------------------------------------------
# Sequential Video Processing
# ---------------------------------------------------------------------------


def process_videos(
    videos: list[Path],
    output_dir: Path,
    config: Any,
    *,
    graph_db_path: Path | None = None,
    vector_db_path: Path | None = None,
    worker_id: int = 0,
    per_video_subdir: str = "per_video",
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Process a list of videos sequentially through the ingestion pipeline.

    This is the shared inner loop used by both the benchmark runner and
    batch ingestion script.

    Args:
        videos: Video file paths to process.
        output_dir: Root output directory.
        config: A ``PipelineConfig`` instance.
        graph_db_path: Optional path to shared graph DB (batch ingestion).
        vector_db_path: Optional path to shared vector DB (batch ingestion).
        worker_id: Worker ID for logging / progress filenames.
        per_video_subdir: Subdirectory under *output_dir* for per-video
            run dirs.  Use ``""`` to place run dirs directly under
            *output_dir* (as the benchmark does).
        resume: If False (default), truncate the per-worker progress
            JSONL at run start. Without this, re-running into a non-empty
            output_dir leaves stale entries that downstream readers (the
            webapp Ingest tab) sum into the current run's totals. If
            True, the file is preserved and new entries append, matching
            ``summary_worker_*.json``'s per-session contract.

    Returns:
        List of per-video result dicts with keys ``video``, ``status``,
        ``elapsed_s``, ``error``, and ``n_clips``.
    """
    # Lazy import to keep this module light at import time
    from video_ingestion_agent.ingestion import run_pipeline

    # Truncate progress file at start of a non-resume run so that re-running
    # into a populated output_dir doesn't cause readers to sum stale entries
    # from the prior run into the current run's totals.
    progress_path = output_dir / f"progress_worker_{worker_id}.jsonl"
    if not resume:
        progress_path.unlink(missing_ok=True)

    results: list[dict[str, Any]] = []
    total = len(videos)

    for idx, video_path in enumerate(videos, 1):
        vid = video_id_from_path(video_path)
        if per_video_subdir:
            run_dir = output_dir / per_video_subdir / vid
        else:
            run_dir = output_dir / vid
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"[Worker {worker_id}] Processing {idx}/{total}: {video_path}")
        logger.info("=" * 60)

        t_start = time.time()
        status = "success"
        error_msg = None
        n_clips = 0

        try:
            kwargs: dict[str, Any] = {
                "video_path": str(video_path),
                "run_dir": run_dir,
                "config": config,
            }
            if graph_db_path is not None:
                kwargs["graph_db_path"] = graph_db_path
            if vector_db_path is not None:
                kwargs["vector_db_path"] = vector_db_path

            final_state = run_pipeline(**kwargs)
            n_clips = len(final_state.get("clips", []))
            logger.info(f"Done: {vid} -> {n_clips} clips")
        except Exception as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"FAILED: {vid} -- {error_msg}")
            logger.error(traceback.format_exc())

        elapsed = time.time() - t_start
        result: dict[str, Any] = {
            "video": str(video_path),
            "video_id": vid,
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "n_clips": n_clips,
            "error": error_msg,
        }
        results.append(result)

        # Append progress incrementally (file already prepared at run start).
        with open(progress_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Free GPU memory between videos to reduce fragmentation and OOM risk
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Parallel Worker Orchestration
# ---------------------------------------------------------------------------


def run_parallel_workers(
    shards: list[list[Path]],
    base_cmd: list[str],
    shard_to_args: Callable[[int, list[Path]], list[str]],
    output_dir: Path,
    *,
    env_for_worker: Callable[[int], dict[str, str]] | None = None,
    log_prefix: str = "worker",
) -> list[int]:
    """Launch N subprocess workers and wait for them to finish.

    Args:
        shards: Pre-computed shard assignments (e.g. from
            :func:`shard_videos_lpt`).
        base_cmd: Base command list shared by all workers.
        shard_to_args: Callable ``(worker_id, shard) -> extra_args`` that
            returns worker-specific CLI arguments.
        output_dir: Directory where per-worker log files are written.
        env_for_worker: Optional callable ``(worker_id) -> env_dict``.
            If *None*, workers inherit ``os.environ``.
        log_prefix: Prefix for log file names (e.g. ``worker`` produces
            ``worker_0.log``).

    Returns:
        List of worker IDs that **failed** (non-zero exit code).
    """
    processes: list[tuple[int, subprocess.Popen, Path]] = []
    log_handles: list = []

    for worker_id, shard in enumerate(shards):
        if not shard:
            continue

        extra_args = shard_to_args(worker_id, shard)
        cmd = base_cmd + extra_args

        env = env_for_worker(worker_id) if env_for_worker else os.environ.copy()

        log_path = output_dir / f"{log_prefix}_{worker_id}.log"
        log_fh = open(log_path, "w")
        log_handles.append(log_fh)

        preview = ", ".join(video_id_from_path(v) for v in shard[:3])
        suffix = "..." if len(shard) > 3 else ""
        logger.info(f"Launching {log_prefix} {worker_id}: {len(shard)} videos ({preview}{suffix})")
        logger.info(f"  Log: {log_path}")

        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_id, p, log_path))

    # Wait for all workers to finish
    logger.info(f"Waiting for {len(processes)} workers to complete...")
    failed: list[int] = []
    for worker_id, p, log_path in processes:
        p.wait()
        if p.returncode != 0:
            logger.error(f"Worker {worker_id} exited with code {p.returncode}. See log: {log_path}")
            failed.append(worker_id)
        else:
            logger.info(f"Worker {worker_id} completed successfully")

    # Close log file handles
    for fh in log_handles:
        fh.close()

    if failed:
        logger.warning(
            f"{len(failed)} worker(s) failed (IDs: {failed}). "
            "Check individual worker logs for details."
        )

    return failed
