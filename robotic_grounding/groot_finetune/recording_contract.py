# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embodiment-neutral contract used by semantic GR00T recording drivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecordingContract:
    """Action extraction and image-term mapping for one recording environment."""

    action_terms: tuple[str, ...]
    action_joint_order: tuple[str, ...] | None
    camera_terms: tuple[tuple[str, str], ...]


def contract_from_env_cfg(env_cfg: Any) -> RecordingContract:
    """Resolve an environment's recording contract."""
    required = (
        "gr00t_record_action_terms",
        "gr00t_record_camera_terms",
    )
    missing = [name for name in required if not hasattr(env_cfg, name)]
    if missing:
        raise ValueError(
            f"environment does not declare an explicit GR00T recording contract: missing={missing}"
        )
    action_terms = tuple(env_cfg.gr00t_record_action_terms)
    joint_order_value = getattr(env_cfg, "gr00t_record_action_joint_order", None)
    action_joint_order = (
        tuple(joint_order_value) if joint_order_value is not None else None
    )
    camera_terms = tuple(tuple(pair) for pair in env_cfg.gr00t_record_camera_terms)
    if not action_terms:
        raise ValueError("GR00T recording contract has no action terms")
    if not camera_terms:
        raise ValueError("GR00T recording contract has no camera terms")
    if len({term for term, _ in camera_terms}) != len(camera_terms):
        raise ValueError("GR00T recording contract repeats an image observation term")
    return RecordingContract(action_terms, action_joint_order, camera_terms)


def joint_reorder_indices(
    source_joint_names: list[str], target_joint_names: tuple[str, ...]
) -> list[int]:
    """Return indices mapping a processed-action tensor into canonical joint order."""
    if len(source_joint_names) != len(set(source_joint_names)):
        raise ValueError("processed action exposes duplicate source joint names")
    if len(target_joint_names) != len(set(target_joint_names)):
        raise ValueError("recording contract contains duplicate target joint names")
    missing = [name for name in target_joint_names if name not in source_joint_names]
    extra = [name for name in source_joint_names if name not in target_joint_names]
    if missing or extra:
        raise ValueError(
            f"record action joint mismatch: missing={missing}, extra={extra}"
        )
    return [source_joint_names.index(name) for name in target_joint_names]
