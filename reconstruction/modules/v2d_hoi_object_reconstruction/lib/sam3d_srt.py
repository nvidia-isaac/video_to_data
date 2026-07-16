# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel, resumable SAM3D SRT execution inside the HOI container."""

from __future__ import annotations

import argparse
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
    if (
        not result_path.is_file()
        or not scaled_mesh_path.is_file()
        or scaled_mesh_path.stat().st_size == 0
    ):
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
            f"(max_views={config.max_views}, maxiter={config.maxiter}, "
            f"top_k={config.top_k})"
        )

        old_env = {key: os.environ.get(key) for key in _WORKER_ENV}
        os.environ.update(_WORKER_ENV)
        try:
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
                        scale_str = "[" + ", ".join(
                            f"{float(value):.4f}" for value in scale
                        ) + "]"
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


def _read_selected_frames(path: Path) -> list[str]:
    try:
        selected = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read selected SAM3D frames from {path}") from exc
    if not isinstance(selected, list) or not all(
        isinstance(frame_id, str) for frame_id in selected
    ):
        raise ValueError(f"Selected SAM3D frames must be a JSON string list: {path}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job_dir", type=Path, required=True)
    parser.add_argument("--selected_frames", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--use_depth", action="store_true")
    parser.add_argument("--stage1_end_frame", type=int, default=None)
    parser.add_argument("--max_views", type=int, default=25)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in ("max_views", "maxiter", "top_k", "parallel"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")

    selected_frames = _read_selected_frames(args.selected_frames)
    outcomes = run_srt_candidates(
        args.job_dir,
        selected_frames,
        use_depth=args.use_depth,
        stage1_end_frame=args.stage1_end_frame,
        config=SRTConfig(
            max_views=args.max_views,
            maxiter=args.maxiter,
            top_k=args.top_k,
            parallel=args.parallel,
        ),
        force=args.force,
    )
    summary = {
        "requested_frames": selected_frames,
        "outcomes": [
            {
                "frame_id": outcome.frame_id,
                "elapsed": outcome.elapsed,
                "resumed": outcome.resumed,
            }
            for outcome in outcomes
        ],
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
