# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-policy inference shared by interactive and headless evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.learner import LearnerState
from flash_chord.training.ppo.normalization import normalize_observation


@dataclass(frozen=True)
class EvaluationConfig:
    """Policy sampling and checkpoint settings for evaluation."""

    checkpoint: str
    deterministic: bool = True
    use_checkpoint_config: bool = True
    validate_policy_schema: bool = True

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("evaluation checkpoint must be specified")


@jax.jit
def _policy_mean(state, observation: jax.Array) -> jax.Array:
    policy, _ = state.apply_fn(state.params, observation)
    return policy.mean()


@jax.jit
def _policy_sample(state, observation: jax.Array, key: jax.Array) -> jax.Array:
    policy, _ = state.apply_fn(state.params, observation)
    return policy.sample(seed=key)


def policy_action(
    learner: LearnerState,
    observation: jax.Array,
    training: TrainingConfig,
    evaluation: EvaluationConfig,
    key: jax.Array | None = None,
) -> jax.Array:
    """Compute deterministic means by default, with optional checkpoint-policy sampling."""
    policy_observation = jnp.asarray(observation)
    if training.observation_normalization:
        policy_observation = normalize_observation(
            learner.normalization,
            policy_observation,
            training.observation_normalization_epsilon,
        )
    if evaluation.deterministic:
        return _policy_mean(learner.train_state, policy_observation)
    if key is None:
        raise ValueError("stochastic evaluation requires a PRNG key")
    return _policy_sample(learner.train_state, policy_observation, key)
