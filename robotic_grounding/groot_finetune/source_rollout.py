# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for dynamics-source rollout export."""

from __future__ import annotations

from typing import Any

import numpy as np


def configure_timeout_only_terminations(terminations_cfg: Any) -> str:
    """Disable every non-timeout termination on a rollout environment config.

    Source collection measures whether an expert reaches the configured horizon.  The
    source task may also declare trajectory-error or safety terms for RL training, but
    those terms are not part of the post-training collection profile.  Mutating this
    environment-config instance before ``gym.make`` keeps that distinction local to
    collection and leaves the registered expert-training task unchanged.
    """
    configured = {
        name: value
        for name, value in vars(terminations_cfg).items()
        if not name.startswith("_") and value is not None and hasattr(value, "time_out")
    }
    timeout_names = [
        name
        for name, value in configured.items()
        if bool(getattr(value, "time_out", False))
    ]
    if len(timeout_names) != 1:
        raise ValueError(
            "source collection requires exactly one configured time_out=True "
            f"termination; got {timeout_names} from {sorted(configured)}"
        )
    timeout_name = timeout_names[0]
    for name in configured:
        if name != timeout_name:
            setattr(terminations_cfg, name, None)
    return timeout_name


def source_success_from_terminations(
    active_terms: tuple[str, ...] | list[str],
    terminated: np.ndarray,
    timeout_name: str,
) -> bool:
    """Return whether the episode reached its configured timeout."""
    names = tuple(active_terms)
    values = np.asarray(terminated, dtype=bool)
    if values.shape != (len(names),):
        raise ValueError(
            f"termination vector must have shape {(len(names),)}, got {values.shape}"
        )
    if names.count(timeout_name) != 1:
        raise ValueError(f"expected one timeout term {timeout_name!r}, got {names}")
    timeout_index = names.index(timeout_name)
    return bool(values[timeout_index])


def termination_reasons_for_env(
    termination_manager: Any, env_index: int
) -> tuple[str, ...]:
    """Return fired termination names through IsaacLab's public manager API."""
    active_terms = tuple(termination_manager.active_terms)
    reasons: list[str] = []
    for name, values in termination_manager.get_active_iterable_terms(env_index):
        if name not in active_terms:
            raise ValueError(f"termination manager returned an inactive term {name!r}")
        if any(bool(value) for value in values):
            reasons.append(str(name))
    return tuple(reasons)


def snapshot_applied_action_target(
    action_term: Any,
    action_indices: Any,
) -> np.ndarray:
    """Copy the absolute target actually applied by an action term.

    IsaacLab may reset a completed environment before ``env.step`` returns. Its reset
    clears ``processed_actions``, so reading that property after the step silently
    replaces a valid terminal target with zeros. Vega's residual action term exposes a
    reset-stable ``last_applied_actions`` buffer specifically for export.

    Raw policy actions are intentionally rejected: they are residuals around a moving
    reference and are not valid GR00T absolute-action labels.
    """
    applied = getattr(action_term, "last_applied_actions", None)
    if applied is None:
        raise TypeError(
            f"{type(action_term).__name__} does not expose last_applied_actions; "
            "cannot safely export terminal absolute targets across auto-reset"
        )
    snapshot = applied[:, action_indices].detach().cpu().numpy().copy()
    if snapshot.ndim != 2 or snapshot.shape[0] == 0:
        raise ValueError(
            f"applied action target must be a non-empty 2-D array, got {snapshot.shape}"
        )
    if not np.isfinite(snapshot).all():
        raise ValueError("applied action target contains non-finite values")
    return snapshot


def validate_terminal_action_target(action_target: np.ndarray) -> float:
    """Reject the characteristic reset-zero terminal label and return its L2 step."""
    values = np.asarray(action_target, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(
            f"action_target must contain at least two non-empty action rows, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("action_target contains non-finite values")
    previous = values[-2]
    terminal = values[-1]
    if np.all(np.abs(terminal) <= 1e-7) and np.linalg.norm(previous) > 1e-3:
        raise ValueError(
            "terminal action target is all zero after a nonzero target; this matches "
            "the auto-reset contamination signature"
        )
    return float(np.linalg.norm(terminal - previous))


def summarize_trajectory_diversity(
    joint_pos: np.ndarray,
    action_target: np.ndarray,
    object_pose: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Summarize phase-aligned variation across completed source trajectories.

    The first axis is the episode axis. All episodes must have the same horizon so the
    standard deviation is measured at the same task phase, rather than conflating timing
    or padding differences with behavioral diversity.
    """
    states = np.asarray(joint_pos, dtype=np.float64)
    actions = np.asarray(action_target, dtype=np.float64)
    for name, values in (("joint_pos", states), ("action_target", actions)):
        if values.ndim != 3 or min(values.shape) == 0:
            raise ValueError(
                f"{name} must have non-empty shape (episodes, time, dim), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    if states.shape[:2] != actions.shape[:2]:
        raise ValueError(
            f"joint_pos and action_target episode/time shapes differ: {states.shape[:2]} != {actions.shape[:2]}"
        )

    summary: dict[str, float | int] = {
        "episode_count": int(states.shape[0]),
        "horizon": int(states.shape[1]),
        "state_cross_episode_std_mean_rad": float(np.std(states, axis=0).mean()),
        "action_cross_episode_std_mean_rad": float(np.std(actions, axis=0).mean()),
        "initial_state_cross_episode_std_mean_rad": float(
            np.std(states[:, 0], axis=0).mean()
        ),
        "initial_action_cross_episode_std_mean_rad": float(
            np.std(actions[:, 0], axis=0).mean()
        ),
        "terminal_action_cross_episode_std_mean_rad": float(
            np.std(actions[:, -1], axis=0).mean()
        ),
    }

    if object_pose is not None:
        poses = np.asarray(object_pose, dtype=np.float64)
        if (
            poses.ndim != 4
            or poses.shape[:2] != states.shape[:2]
            or poses.shape[-1] != 7
        ):
            raise ValueError(
                "object_pose must have shape (episodes, time, objects, 7) with "
                f"matching episode/time axes, got {poses.shape}"
            )
        if not np.isfinite(poses).all():
            raise ValueError("object_pose contains non-finite values")
        positions = poses[..., :3]
        summary["object_position_cross_episode_std_mean_m"] = float(
            np.std(positions, axis=0).mean()
        )
        summary["initial_object_position_cross_episode_std_mean_m"] = float(
            np.std(positions[:, 0], axis=0).mean()
        )

    return summary
