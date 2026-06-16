# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for experiment workflow generation.

Used by run_experiment.py and experiment-specific workflow.py modules.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_CONFIG_PATH = Path(__file__).resolve().parent / "experiment_config.yaml"


@lru_cache(maxsize=1)
def load_internal_experiment_config() -> dict[str, Any]:
    """Load optional internal launch settings removed from release builds."""
    if not EXPERIMENT_CONFIG_PATH.exists():
        return {}
    with open(EXPERIMENT_CONFIG_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid internal experiment config: {EXPERIMENT_CONFIG_PATH}"
        )
    return data


def get_internal_config_value(*keys: str, default: Any = None) -> Any:
    """Return a nested value from experiment_config.yaml when present."""
    value: Any = load_internal_experiment_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def require_internal_config_value(*keys: str) -> Any:
    """Return a required internal value or fail with release-safe guidance."""
    value = get_internal_config_value(*keys)
    if value is None:
        dotted = ".".join(keys)
        raise RuntimeError(
            f"Missing internal experiment config value: {dotted}. "
            f"Create {EXPERIMENT_CONFIG_PATH} or pass an explicit override."
        )
    return value


DEFAULT_WANDB_ENTITY = get_internal_config_value("wandb_entity")
DEFAULT_OSMO_IMAGE_REPO = get_internal_config_value("osmo", "image_repo")
DEFAULT_OSMO_IMAGE_LATEST = get_internal_config_value("osmo", "image_latest")


def sequence_to_object(sequence_id: str) -> str:
    """e.g. arctic_s01_capsulemachine_grab_01 -> capsulemachine."""
    parts = sequence_id.split("_")
    return parts[2] if len(parts) >= 3 else sequence_id


def is_local_path(path: str) -> bool:
    """True if path looks like a local file (not s3/http/gs)."""
    return not (
        path.startswith("s3://")
        or path.startswith("http://")
        or path.startswith("https://")
        or path.startswith("gs://")
    )


def _format_train_override_value(key: str, value: object) -> str:
    """Format a single train override for Hydra key=value syntax."""
    if isinstance(value, bool):
        return "true" if value else "false"
    # YAML often parses 1 / 0 as int; Hydra then infers int, but ConfigClass expects float.
    if isinstance(value, int) and not isinstance(value, bool):
        if "initial_virtual_object_control_curriculum_scale" in key:
            return str(float(value))
    return str(value)


def overrides_to_cli(overrides: dict[str, Any]) -> list[str]:
    """Convert overrides dict to train.py CLI args (key=value format)."""
    return [f"{k}={_format_train_override_value(k, v)}" for k, v in overrides.items()]


def build_train_command(
    run_name: str,
    overrides: dict[str, Any],
    *,
    resume_from: str | None = None,
    seed: int | None = None,
    motion_file: str | None = None,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    headless: bool = True,
    video: bool = True,
    eval_video_only: bool = False,
    video_length: int | None = None,
    video_interval: int | None = None,
    eval_episodes_per_save: int = 0,
    task: str = "Sharpa-V2P-v0",
    logger: str | None = None,
    log_project_name: str | None = None,
    zero_actor: bool = False,
    use_primitive_urdfs: bool = False,
) -> list[str]:
    """Build train.py command as list of args."""
    cmd = [
        "python",
        "scripts/rsl_rl/train.py",
        "--headless" if headless else "",
        "--video" if video else "",
        "--zero-actor" if zero_actor else "",
        "--use_primitive_urdfs" if use_primitive_urdfs else "",
        "--task",
        task,
        "--run_name",
        run_name,
    ]
    cmd = [c for c in cmd if c]  # drop empty
    if eval_video_only:
        cmd.append("--eval_video_only")
    if video_length is not None:
        cmd.extend(["--video_length", str(video_length)])
    if video_interval is not None:
        cmd.extend(["--video_interval", str(video_interval)])
    if eval_episodes_per_save > 0:
        cmd.extend(["--eval_episodes_per_save", str(eval_episodes_per_save)])
    if resume_from:
        cmd.extend(["--resume", "--checkpoint", resume_from])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if motion_file is not None:
        cmd.extend(["--motion_file", motion_file])
    if num_envs is not None:
        cmd.extend(["--num_envs", str(num_envs)])
    if max_iterations is not None:
        cmd.extend(["--max_iterations", str(max_iterations)])
    if logger:
        cmd.extend(["--logger", logger])
    if log_project_name:
        cmd.extend(["--log_project_name", log_project_name])
    cmd.extend(overrides_to_cli(overrides))
    return cmd


def build_eval_command(
    overrides: dict[str, Any],
    *,
    checkpoint: str | None = None,
    seed: int | None = None,
    motion_file: str | None = None,
    num_envs: int | None = None,
    video: bool = False,
    video_length: int | None = None,
    eval_episodes: int | None = None,
    task: str = "Sharpa-V2P-v0",
    logger: str | None = None,
    log_project_name: str | None = None,
    use_primitive_urdfs: bool = False,
    real_time: bool = False,
) -> list[str]:
    """Build eval.py command as list of args.

    Mirrors build_train_command but only forwards flags eval.py understands.
    Hydra-style train_overrides are forwarded verbatim so configs can scope
    e.g. motion_start_frame / motion_end_frame uniformly across train + eval.
    """
    cmd = [
        "python",
        "scripts/rsl_rl/eval.py",
        "--video" if video else "",
        "--use_primitive_urdfs" if use_primitive_urdfs else "",
        "--real-time" if real_time else "",
        "--task",
        task,
    ]
    cmd = [c for c in cmd if c]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if motion_file is not None:
        cmd.extend(["--motion_file", motion_file])
    if num_envs is not None:
        cmd.extend(["--num_envs", str(num_envs)])
    if video_length is not None:
        cmd.extend(["--video_length", str(video_length)])
    if eval_episodes is not None:
        cmd.extend(["--eval_episodes", str(eval_episodes)])
    if logger:
        cmd.extend(["--logger", logger])
    if log_project_name:
        cmd.extend(["--log_project_name", log_project_name])
    cmd.extend(overrides_to_cli(overrides))
    return cmd


def make_entry_script(
    run_name: str,
    overrides: dict[str, Any],
    *,
    resume_from: str | None = None,
    seed: int | None = None,
    motion_file: str | None = None,
    num_envs: int | None = None,
    max_iterations: int | None = None,
    use_timestamp: bool = True,
    video: bool = True,
    eval_video_only: bool = False,
    video_length: int | None = None,
    video_interval: int | None = None,
    eval_episodes_per_save: int = 0,
    task: str = "Sharpa-V2P-v0",
    logger: str = "wandb",
    log_project_name: str = "v2p_hands",
    zero_actor: bool = False,
    use_primitive_urdfs: bool = False,
    urdfs_src_path: str | None = None,
) -> str:
    """Generate /tmp/entry.sh content for OSMO."""
    # Pass run_name (suffix only) to train; train.py adds its own timestamp to avoid duplication.
    lines = [
        "set -ex",
        "",
    ]
    if urdfs_src_path:
        lines += [
            # Generate all TACO rigid URDFs + visual STL meshes from *_cm.obj files in
            # the image. Produces workspace assets/urdfs/taco/ + assets/meshes/taco/ with
            # correct relative mesh paths — no OSMO cp needed.
            "python scripts/generate_rigid_urdfs.py --dataset taco",
            "",
        ]
    cmd = build_train_command(
        run_name,
        overrides,
        resume_from=resume_from,
        seed=seed,
        motion_file=motion_file,
        num_envs=num_envs,
        max_iterations=max_iterations,
        video=video,
        eval_video_only=eval_video_only,
        video_length=video_length,
        video_interval=video_interval,
        eval_episodes_per_save=eval_episodes_per_save,
        task=task,
        logger=logger,
        log_project_name=log_project_name,
        zero_actor=zero_actor,
        use_primitive_urdfs=use_primitive_urdfs,
    )
    cmd_str = " \\\n  ".join(cmd)
    lines.append(cmd_str)
    return "\n".join(lines)
