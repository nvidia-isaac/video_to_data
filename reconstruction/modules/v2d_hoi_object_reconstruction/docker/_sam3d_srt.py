# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel, resumable SAM3D SRT execution for the host orchestrator."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SRTConfig:
    """SRT settings shared with the OSMO fast path."""

    max_views: int = 25
    maxiter: int = 60
    top_k: int = 1
    parallel: int = 8


@dataclass(frozen=True)
class SRTOutcome:
    frame_id: str
    result: dict[str, Any]
    elapsed: float
    resumed: bool = False


_WORKER_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "V2D_SRT_CKDTREE_WORKERS": "1",
}


def _completed_result(job_dir: Path, frame_id: str) -> dict[str, Any] | None:
    """Return a valid completed result, or ``None`` when SRT must run."""
    srt_dir = job_dir / "sam3d" / frame_id / "srt"
    result_path = srt_dir / "srt_result.json"
    scaled_mesh_path = srt_dir / "output_scaled.glb"
    if not result_path.is_file() or not scaled_mesh_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict) or "scale" not in result:
        return None
    return result


def _run_candidate(
    job_dir: str,
    frame_id: str,
    use_depth: bool,
    stage1_end_frame: int | None,
    config: SRTConfig,
) -> SRTOutcome:
    """Run one candidate in an isolated process with bounded native threads."""
    # Import after worker thread limits are present in the environment.
    from v2d_hoi_object_reconstruction.lib.scale_mesh_srt import estimate_srt_for_frame

    job_path = Path(job_dir)
    frame_dir = job_path / "sam3d" / frame_id
    t0 = time.time()
    result = estimate_srt_for_frame(
        job_dir=job_path,
        glb_path=frame_dir / "mesh.glb",
        output_dir=frame_dir / "srt",
        use_depth=use_depth,
        stage1_end_frame=stage1_end_frame,
        max_views=config.max_views,
        maxiter=config.maxiter,
        top_k=config.top_k,
    )
    return SRTOutcome(frame_id=frame_id, result=result, elapsed=time.time() - t0)


def run_srt_candidates(
    job_dir: Path,
    frame_ids: Iterable[str],
    *,
    use_depth: bool,
    stage1_end_frame: int | None,
    config: SRTConfig,
    force: bool = False,
) -> list[SRTOutcome]:
    """Run independent candidates in parallel and reuse complete outputs."""
    frame_ids = list(frame_ids)
    outcomes: dict[str, SRTOutcome] = {}
    pending: list[str] = []

    for frame_id in frame_ids:
        mesh_path = job_dir / "sam3d" / frame_id / "mesh.glb"
        if not mesh_path.is_file():
            print(f"[warning] SAM3D output not found for frame {frame_id}: {mesh_path}")
            continue
        result = None if force else _completed_result(job_dir, frame_id)
        if result is None:
            pending.append(frame_id)
        else:
            print(f"[pipeline] SRT {frame_id}: reusing complete output")
            outcomes[frame_id] = SRTOutcome(
                frame_id=frame_id,
                result=result,
                elapsed=0.0,
                resumed=True,
            )

    if pending:
        workers = min(config.parallel, len(pending))
        print(
            f"[pipeline] SRT: {len(pending)} pending, {workers} parallel workers "
            f"(max_views={config.max_views}, maxiter={config.maxiter}, top_k={config.top_k})"
        )

        old_env = {key: os.environ.get(key) for key in _WORKER_ENV}
        os.environ.update(_WORKER_ENV)
        try:
            # Spawn makes each worker import numerical libraries only after the
            # per-process native thread limits above have been inherited.
            mp_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as pool:
                future_to_frame = {
                    pool.submit(
                        _run_candidate,
                        str(job_dir),
                        frame_id,
                        use_depth,
                        stage1_end_frame,
                        config,
                    ): frame_id
                    for frame_id in pending
                }
                for future in as_completed(future_to_frame):
                    outcome = future.result()
                    outcomes[outcome.frame_id] = outcome
                    scale = outcome.result.get("scale", float("nan"))
                    if isinstance(scale, list):
                        scale_str = "[" + ", ".join(f"{float(v):.4f}" for v in scale) + "]"
                    else:
                        scale_str = f"{float(scale):.4f}"
                    print(
                        f"[pipeline] SRT {outcome.frame_id} done in "
                        f"{outcome.elapsed:.1f}s  scale={scale_str}"
                    )
        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

    return [outcomes[frame_id] for frame_id in frame_ids if frame_id in outcomes]
