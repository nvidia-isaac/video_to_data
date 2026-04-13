"""Shared utilities for experiment workflow generation.

Used by run_experiment.py and experiment-specific workflow.py modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


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
    task: str = "Sharpa-V2P-v0",
    logger: str | None = None,
    log_project_name: str | None = None,
    disable_robot_to_object_collisions: bool = False,
) -> list[str]:
    """Build train.py command as list of args."""
    cmd = [
        "python",
        "scripts/rsl_rl/train.py",
        "--headless" if headless else "",
        "--video" if video else "",
        (
            "--disable_robot_to_object_collisions"
            if disable_robot_to_object_collisions
            else ""
        ),
        "--task",
        task,
        "--run_name",
        run_name,
    ]
    cmd = [c for c in cmd if c]  # drop empty
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
    logger: str = "wandb",
    log_project_name: str = "v2p_hands",
) -> str:
    """Generate /tmp/entry.sh content for OSMO."""
    # Pass run_name (suffix only) to train; train.py adds its own timestamp to avoid duplication.
    lines = [
        "set -ex",
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
        logger=logger,
        log_project_name=log_project_name,
    )
    cmd_str = " \\\n  ".join(cmd)
    lines.append(cmd_str)
    return "\n".join(lines)
