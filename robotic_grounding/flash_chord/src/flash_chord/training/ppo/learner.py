# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composable learner state and one-iteration training operation."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flash_chord.training.environment import JAXVectorEnv
from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.network import ActorCritic
from flash_chord.training.ppo.normalization import (
    NormalizationState,
    create_normalization_state,
    normalize_observation,
)
from flash_chord.training.ppo.algorithm import PPOMetrics, create_train_state, update
from flash_chord.training.ppo.rollout import RolloutMetrics, collect_rollout


@flax.struct.dataclass
class LearnerState:
    """Complete device-resident learner state between iterations."""

    train_state: TrainState
    normalization: NormalizationState
    observation: jax.Array
    key: jax.Array
    learning_rate: jax.Array
    iteration: jax.Array


@flax.struct.dataclass
class IterationMetrics:
    """Rollout and optimization metrics from one learner iteration."""

    rollout: RolloutMetrics
    optimization: PPOMetrics


def create_learner(env: JAXVectorEnv, config: TrainingConfig) -> LearnerState:
    """Initialize policy, optimizer, normalization, and the first environment observation."""
    if env.world_count != config.world_count:
        raise ValueError(f"environment has {env.world_count} worlds; config requests {config.world_count}")
    key, initialization_key = jax.random.split(jax.random.PRNGKey(config.seed))
    model = ActorCritic(action_dim=env.action_dim, config=config.network)
    parameters = model.init(
        initialization_key,
        jnp.zeros((1, env.observation_dim), dtype=jnp.float32),
    )
    train_state = create_train_state(model, parameters, config.ppo)
    return LearnerState(
        train_state=train_state,
        normalization=create_normalization_state(env.observation_dim),
        observation=jnp.array(env.reset(), copy=True),
        key=key,
        learning_rate=jnp.asarray(config.ppo.learning_rate, dtype=jnp.float32),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )


def train_iteration(
    env: JAXVectorEnv,
    learner: LearnerState,
    config: TrainingConfig,
) -> tuple[LearnerState, IterationMetrics]:
    """Collect one rollout and apply all configured PPO epochs and mini-batches."""
    batch, observation, normalization, key, rollout_metrics = collect_rollout(
        env,
        learner.train_state,
        learner.normalization,
        learner.observation,
        learner.key,
        config,
    )
    policy_observation = observation
    if config.observation_normalization:
        policy_observation = normalize_observation(
            normalization,
            observation,
            config.observation_normalization_epsilon,
        )
    _, last_value = learner.train_state.apply_fn(
        learner.train_state.params,
        policy_observation,
    )
    key, update_key = jax.random.split(key)
    train_state, learning_rate, optimization_metrics = update(
        learner.train_state,
        batch,
        last_value,
        update_key,
        learner.learning_rate,
        config.ppo,
    )
    learner = learner.replace(
        train_state=train_state,
        normalization=normalization,
        observation=observation,
        key=key,
        learning_rate=learning_rate,
        iteration=learner.iteration + 1,
    )
    return learner, IterationMetrics(
        rollout=rollout_metrics,
        optimization=optimization_metrics,
    )
