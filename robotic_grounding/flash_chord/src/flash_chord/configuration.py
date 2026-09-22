# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra composition at application boundaries; runtime modules consume concrete typed objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeVar

from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from flash_chord.lifecycle.curriculum import CurriculumStage, FixedCurriculum
from flash_chord.training.ppo.config import NetworkConfig

T = TypeVar("T")


def compose_config(config_name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose one packaged application config without taking ownership of process arguments."""
    with initialize_config_module(config_module="flash_chord.configs", version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))


def instantiate_typed(config: DictConfig, expected_type: type[T]) -> T:
    """Instantiate a Hydra target and enforce the typed runtime boundary."""
    value = instantiate(config, _convert_="all")
    if not isinstance(value, expected_type):
        raise TypeError(f"configured target produced {type(value).__name__}; expected {expected_type.__name__}")
    return value


def resolve_path(path: str | Path) -> Path:
    """Resolve a user config path against the invocation directory without changing process cwd."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def resolved_dict(config: DictConfig) -> dict[str, object]:
    """Convert a composed config to a plain resolved dictionary for metadata/logging."""
    value = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping config, got {type(value).__name__}")
    return value


def make_curriculum_stage(
    voc_scale: float,
    objective_weights: Mapping[str, float],
    reset_to_first_frame_probability: float | None = None,
    immediate_first_frame_probability: float | None = None,
) -> CurriculumStage:
    """Normalize Hydra mapping input at the typed curriculum boundary."""
    return CurriculumStage(
        voc_scale=voc_scale,
        objective_weights=dict(objective_weights),
        reset_to_first_frame_probability=reset_to_first_frame_probability,
        immediate_first_frame_probability=immediate_first_frame_probability,
    )


def make_fixed_curriculum(thresholds: Sequence[int], stages: Sequence[CurriculumStage]) -> FixedCurriculum:
    """Normalize Hydra list input at the typed fixed-curriculum boundary."""
    return FixedCurriculum(thresholds=tuple(thresholds), stages=tuple(stages))


def make_network_config(
    actor_hidden_dims: Sequence[int],
    critic_hidden_dims: Sequence[int],
    **kwargs,
) -> NetworkConfig:
    """Normalize Hydra list input at the typed network boundary."""
    return NetworkConfig(
        actor_hidden_dims=tuple(actor_hidden_dims),
        critic_hidden_dims=tuple(critic_hidden_dims),
        **kwargs,
    )
