# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actor-only, precompiled FlashSAC policy inference."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from flash_chord.training.flash_sac.checkpoint import ActorInferenceState, load_actor_checkpoint
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.network import (
    Actor,
    deterministic_action,
    project_unit_parameters,
    sample_policy,
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Checkpoint reconstruction and policy-sampling settings."""

    checkpoint: str
    deterministic: bool = True
    use_checkpoint_config: bool = True
    validate_policy_schema: bool = True
    world_count: int = 4096
    start_frame: int = 0
    reset_mode: str = "explicit"
    motion_start_frame: int | None = None
    motion_end_frame: int | None = None
    metrics_output: str | None = None
    source_root: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("evaluation checkpoint must be specified")
        if self.world_count <= 0:
            raise ValueError(f"evaluation world_count must be positive, got {self.world_count}")
        if self.start_frame < 0:
            raise ValueError(f"evaluation start_frame must be non-negative, got {self.start_frame}")
        if self.reset_mode not in ("explicit", "sampled_settled"):
            raise ValueError(f"evaluation reset_mode must be 'explicit' or 'sampled_settled', got {self.reset_mode!r}")
        if self.motion_start_frame is not None and self.motion_start_frame < 0:
            raise ValueError(
                f"evaluation motion_start_frame must be non-negative or None, got {self.motion_start_frame}"
            )
        if self.motion_end_frame is not None and self.motion_end_frame < -1:
            raise ValueError(
                f"evaluation motion_end_frame must be -1, non-negative, or None, got {self.motion_end_frame}"
            )


def create_actor_template(
    training: TrainingConfig,
    observation_dim: int,
    action_dim: int,
    *,
    device: jax.Device | None = None,
) -> tuple[Actor, ActorInferenceState]:
    """Create only the actor state tree needed to restore an inference checkpoint."""
    if observation_dim <= 0 or action_dim <= 0:
        raise ValueError(f"observation_dim and action_dim must be positive, got {observation_dim} and {action_dim}")
    compute_dtype = jnp.dtype(training.compute_dtype) if training.mixed_precision else jnp.dtype(jnp.float32)
    actor = Actor(action_dim=action_dim, config=training.network, dtype=compute_dtype)
    device_context = nullcontext() if device is None else jax.default_device(device)
    with device_context:
        variables = actor.init(
            jax.random.PRNGKey(training.seed),
            jnp.zeros((1, observation_dim), dtype=jnp.float32),
            training=False,
        )
        state = ActorInferenceState(
            params=project_unit_parameters(variables["params"], training.network.projection_epsilon),
            batch_stats=variables["batch_stats"],
        )
    return actor, state


def compile_policy_action(
    actor: Actor,
    state: ActorInferenceState,
    observation: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool,
    action_transform: str = "tanh",
    action_scale: float = 1.0,
) -> Callable:
    """Lower and compile one stable actor-only inference executable."""

    def policy_action(current: ActorInferenceState, current_observation: jax.Array, current_key: jax.Array):
        distribution = actor.apply(
            {"params": current.params, "batch_stats": current.batch_stats},
            current_observation,
            training=False,
        )
        if deterministic:
            return (
                deterministic_action(
                    distribution,
                    action_transform=action_transform,
                    action_scale=action_scale,
                ),
                current_key,
            )
        next_key, action_key = jax.random.split(current_key)
        return (
            sample_policy(
                distribution,
                action_key,
                action_transform=action_transform,
                action_scale=action_scale,
            ).action,
            next_key,
        )

    return jax.jit(policy_action).lower(state, observation, key).compile()


def load_policy_for_inference(
    checkpoint: str,
    training: TrainingConfig,
    observation: jax.Array,
    action_dim: int,
    *,
    deterministic: bool,
    device: jax.Device | None = None,
) -> tuple[ActorInferenceState, Callable, jax.Array, dict[str, str]]:
    """Restore a world-count-independent actor and precompile it for the evaluation batch shape."""
    if observation.ndim != 2:
        raise ValueError(f"evaluation observation must be rank two, got {observation.shape}")
    actor, template = create_actor_template(
        training,
        observation.shape[1],
        action_dim,
        device=device,
    )
    state, metadata = load_actor_checkpoint(checkpoint, template)
    key = jax.device_put(jax.random.PRNGKey(training.seed), device)
    policy_action = compile_policy_action(
        actor,
        state,
        observation,
        key,
        deterministic=deterministic,
        action_transform=training.network.action_transform,
        action_scale=training.network.action_scale,
    )
    return state, policy_action, key, metadata
