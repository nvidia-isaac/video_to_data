# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container-backed utility steps used by the host orchestrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from v2d.docker.container import run_in_container
from v2d.grounding_dino.docker.run_image_to_object_bboxes import (
    run_image_to_object_bboxes,
)


def run_grounding_dino(
    *,
    image_path: str | Path,
    output_path: str | Path,
    prompt: str,
    model_dir: str | Path,
    box_threshold: float,
) -> None:
    """Run Grounding DINO through its canonical container wrapper."""
    run_image_to_object_bboxes(
        image_path=str(image_path),
        output_path=str(output_path),
        prompt=prompt,
        model_dir=str(model_dir),
        box_threshold=box_threshold,
    )


def prepare_job(
    *,
    image: str,
    input_dir: str | Path,
    job_dir: str | Path,
    fps: int = 30,
    max_frames: int | None = None,
) -> None:
    """Prepare the reconstruction job directory inside ``image``."""
    input_dir = Path(input_dir).resolve()
    run_in_container(
        image=image,
        module="v2d_hoi_object_reconstruction.lib.prepare_FP_folder",
        inputs={"input_dir": str(input_dir)},
        outputs={"job_dir": str(job_dir)},
        extra_args={"fps": fps, "max_frames": max_frames},
        gpus=False,
    )


def convert_poses_to_matrix(
    *,
    image: str,
    poses_dir: str | Path,
) -> None:
    """Convert FoundationPose JSON files in place inside ``image``."""
    poses_dir = Path(poses_dir).resolve()
    run_in_container(
        image=image,
        module="v2d_hoi_object_reconstruction.lib.convert_poses_to_matrix",
        inputs={"poses_dir": str(poses_dir)},
        outputs={},
        gpus=False,
    )


def postprocess_masks(
    *,
    image: str,
    input_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Post-process SAM2 masks inside ``image`` and return its summary."""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir)
    summary_path = output_dir / "postprocess_summary.json"
    summary_path.unlink(missing_ok=True)
    run_in_container(
        image=image,
        module="v2d_hoi_object_reconstruction.lib.postprocess_masks",
        inputs={"input_dir": str(input_dir)},
        outputs={"output_dir": str(output_dir)},
        extra_args=config,
        gpus=False,
    )
    try:
        return json.loads(summary_path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Mask post-processing completed without producing {summary_path}."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Mask post-processing wrote an invalid {summary_path}."
        ) from exc


def select_sam3d_frames(
    *,
    image: str,
    job_dir: str | Path,
    bin_deg: float,
    fallback_count: int = 6,
) -> list[str]:
    """Select SAM3D source frames inside ``image``."""
    job_dir = Path(job_dir).resolve()
    output_path = job_dir / "sam3d" / "selected_frames.json"
    output_path.unlink(missing_ok=True)
    run_in_container(
        image=image,
        module="v2d_hoi_object_reconstruction.lib.select_sam3d_frames",
        inputs={"job_dir": str(job_dir)},
        outputs={"output_path": str(output_path)},
        extra_args={
            "bin_deg": bin_deg,
            "fallback_count": fallback_count,
        },
        gpus=False,
    )
    try:
        selected = json.loads(output_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"SAM3D frame selection did not produce valid JSON at {output_path}."
        ) from exc
    if not isinstance(selected, list) or not all(
        isinstance(frame_id, str) for frame_id in selected
    ):
        raise RuntimeError(f"Invalid SAM3D selected-frame list in {output_path}.")
    return selected


def run_sam3d_srt(
    *,
    image: str,
    job_dir: str | Path,
    use_depth: bool,
    stage1_end_frame: int | None,
    max_views: int,
    maxiter: int,
    top_k: int,
    parallel: int,
    force: bool,
) -> dict[str, Any]:
    """Run SAM3D SRT optimization inside ``image`` and return its summary."""
    job_dir = Path(job_dir).resolve()
    selected_frames = job_dir / "sam3d" / "selected_frames.json"
    summary_path = job_dir / "sam3d" / "srt_run_summary.json"
    summary_path.unlink(missing_ok=True)
    run_in_container(
        image=image,
        module="v2d_hoi_object_reconstruction.lib.sam3d_srt",
        inputs={
            "job_dir": str(job_dir),
            "selected_frames": str(selected_frames),
        },
        outputs={"summary_path": str(summary_path)},
        extra_args={
            "use_depth": use_depth,
            "stage1_end_frame": stage1_end_frame,
            "max_views": max_views,
            "maxiter": maxiter,
            "top_k": top_k,
            "parallel": parallel,
            "force": force,
        },
        gpus=False,
    )
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"SAM3D SRT did not produce valid JSON at {summary_path}."
        ) from exc
    if not isinstance(summary, dict) or not isinstance(summary.get("outcomes"), list):
        raise RuntimeError(f"Invalid SAM3D SRT summary in {summary_path}.")
    return summary


def select_best_sam3d(
    *,
    image: str,
    job_dir: str | Path,
) -> dict[str, Any]:
    """Rank and copy the best SAM3D candidate inside ``image``."""
    job_dir = Path(job_dir).resolve()
    output_dir = job_dir / "sam3d" / "best"
    summary_path = output_dir / "best_frame.json"
    summary_path.unlink(missing_ok=True)
    try:
        run_in_container(
            image=image,
            module="v2d_hoi_object_reconstruction.lib.select_sam3d_best",
            inputs={"job_dir": str(job_dir)},
            outputs={},
            gpus=False,
        )
    except subprocess.CalledProcessError as exc:
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
            else:
                if isinstance(summary, dict) and summary.get("best_frame") is None:
                    raise ValueError("No valid SAM3D SRT candidates found") from exc
        raise

    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"SAM3D best-candidate selection did not produce valid JSON at "
            f"{summary_path}."
        ) from exc
    if not isinstance(summary, dict) or not summary.get("best_frame"):
        raise RuntimeError(f"Invalid SAM3D best-candidate summary in {summary_path}.")
    return summary


def stitch_mp4(
    *,
    image: str,
    frames_dir: str | Path,
    output_mp4: str | Path,
    fps: int = 30,
) -> None:
    """Encode numbered JPEG frames into MP4 inside ``image``."""
    frames_dir = Path(frames_dir).resolve()
    run_in_container(
        image=image,
        module="v2d_hoi_object_reconstruction.lib.stitch_mp4",
        inputs={"frames_dir": str(frames_dir)},
        outputs={"output_mp4": str(output_mp4)},
        extra_args={"fps": fps},
        gpus=False,
    )
