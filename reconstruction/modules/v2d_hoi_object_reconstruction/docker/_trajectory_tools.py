# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container-backed CuSFM trajectory analysis for the host orchestrator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from v2d.docker.container import run_in_container


_QUALITY_DEFAULTS = {
    "min_keyframes": 30,
    "min_angle_span_deg": 600.0,
    "max_backtracking_fraction": 0.25,
    "max_translation_step_m": 2.0,
    "max_rotation_step_deg": 90.0,
    "robust_step_sigma": 6.0,
    "large_step_floor_m": 0.25,
    "max_large_step_fraction": 0.25,
}


def run_sfm_quality_check(
    *,
    image: str,
    sfm_keyframes: str,
    frames_meta: str,
    output_dir: Path,
    config: dict[str, Any],
    fail_on_error: bool,
) -> Path:
    """Run the CuSFM quality gate in ``image`` and return its result path."""
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    result_path.unlink(missing_ok=True)
    extra_args = {
        name: config.get(name, default)
        for name, default in _QUALITY_DEFAULTS.items()
    }
    extra_args["warn_only"] = not fail_on_error

    try:
        run_in_container(
            image=image,
            module="v2d_hoi_object_reconstruction.lib.check_sfm_scan_quality",
            inputs={
                "sfm_keyframes": sfm_keyframes,
                "frames_meta": frames_meta,
            },
            outputs={"output_dir": str(output_dir)},
            extra_args=extra_args,
            gpus=False,
        )
    except subprocess.CalledProcessError as exc:
        if result_path.is_file():
            raise RuntimeError(
                "CuSFM scan quality check failed. "
                f"See {result_path} and diagnostic plots."
            ) from exc
        raise RuntimeError(
            "CuSFM scan quality checker failed to execute in the HOI container. "
            "See the container log above."
        ) from exc

    if not result_path.is_file():
        raise RuntimeError(
            "CuSFM scan quality checker completed without producing "
            f"{result_path}."
        )
    return result_path


def run_stage1_detection(
    *,
    image: str,
    sfm_keyframes: str,
    frames_meta: str,
    output_dir: Path,
    buffer_deg: float,
) -> int | None:
    """Run Stage-1 detection in ``image`` and return the detected frame."""
    output_dir = Path(output_dir)
    result_path = output_dir / "result.json"
    result_path.unlink(missing_ok=True)

    try:
        run_in_container(
            image=image,
            module="v2d_hoi_object_reconstruction.lib.detect_stage1_end",
            inputs={
                "sfm_keyframes": sfm_keyframes,
                "frames_meta": frames_meta,
            },
            outputs={"output_dir": str(output_dir)},
            extra_args={"buffer_deg": buffer_deg},
            gpus=False,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Stage-1 detector failed to execute in the HOI container. "
            "See the container log above."
        ) from exc

    if not result_path.is_file():
        raise RuntimeError(
            "Stage-1 detection did not produce result.json "
            "(check stage1_detect_debug/ plots and the container log)."
        )
    try:
        result = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stage-1 detection wrote an invalid {result_path}.") from exc
    return result.get("stage1_end_frame")
