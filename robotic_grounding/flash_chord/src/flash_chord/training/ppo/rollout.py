# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-device rollout collection across the Warp/JAX boundary."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp

from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.environment import JAXVectorEnv
from flash_chord.training.ppo.normalization import (
    NormalizationState,
    normalize_observation,
    update_normalization,
)
from flash_chord.training.ppo.algorithm import RolloutBatch


@flax.struct.dataclass
class RolloutMetrics:
    """Device-resident aggregate rollout metrics."""

    mean_step_reward: jax.Array
    reference_end_count: jax.Array
    episode_return_sum: jax.Array
    episode_length_sum: jax.Array
    episode_reference_progress_sum: jax.Array
    episode_count: jax.Array
    action_mean_abs_mean: jax.Array
    action_std_mean: jax.Array
    objective_term_mean: jax.Array | None = None
    termination_cause_count: jax.Array | None = None
    tracking_error_mean: jax.Array | None = None


@jax.jit
def sample_action(state, observation: jax.Array, key: jax.Array):
    """Sample one action batch and preserve its behavior-policy statistics."""
    policy, value = state.apply_fn(state.params, observation)
    action = policy.sample(seed=key)
    return action, policy.log_prob(action), value, policy.mean(), policy.stddev()


def collect_rollout(
    env: JAXVectorEnv,
    state,
    normalization: NormalizationState,
    observation: jax.Array,
    key: jax.Array,
    config: TrainingConfig,
) -> tuple[RolloutBatch, jax.Array, NormalizationState, jax.Array, RolloutMetrics]:
    """Collect one time-major on-policy rollout without host array transfers."""
    observations = []
    actions = []
    log_probabilities = []
    values = []
    rewards = []
    dones = []
    action_means = []
    action_stds = []
    raw_rewards = []
    reference_end_counts = []
    episode_returns = []
    episode_lengths = []
    episode_reference_progress = []
    episode_counts = []
    objective_terms = []
    termination_causes = []
    tracking_errors = []

    for _ in range(config.rollout_steps):
        raw_observation = jnp.array(observation, copy=True)
        policy_observation = raw_observation
        if config.observation_normalization:
            policy_observation = normalize_observation(
                normalization,
                raw_observation,
                config.observation_normalization_epsilon,
            )
        key, action_key = jax.random.split(key)
        action, log_probability, value, action_mean, action_std = sample_action(
            state,
            policy_observation,
            action_key,
        )
        transition = env.step(action)
        next_observation = jnp.array(transition.observation, copy=True)
        reward = jnp.array(transition.reward, copy=True)
        terminated = jnp.array(transition.terminated, copy=True)
        truncated = jnp.array(transition.truncated, copy=True)
        done = jnp.maximum(terminated, truncated).astype(jnp.float32)
        reference_end = jnp.logical_and(truncated.astype(bool), jnp.logical_not(terminated.astype(bool)))
        reward = reward + config.ppo.discount * value * reference_end.astype(jnp.float32)

        observations.append(policy_observation)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        action_means.append(action_mean)
        action_stds.append(action_std)
        raw_rewards.append(transition.reward.mean())
        reference_end_counts.append(reference_end.sum())
        completed = transition.episode_length > 0
        episode_returns.append(jnp.where(completed, transition.episode_return, 0.0).sum())
        episode_lengths.append(jnp.where(completed, transition.episode_length, 0).sum())
        episode_reference_progress.append(jnp.where(completed, transition.episode_reference_progress, 0.0).sum())
        episode_counts.append(completed.sum())
        if transition.objective_terms is not None:
            objective_terms.append(jnp.array(transition.objective_terms, copy=True))
        if transition.termination_causes is not None:
            remaining = terminated.astype(bool)
            exclusive_causes = []
            for cause in jnp.moveaxis(jnp.array(transition.termination_causes, copy=True), -1, 0):
                selected = jnp.logical_and(remaining, cause.astype(bool))
                exclusive_causes.append(selected)
                remaining = jnp.logical_and(remaining, jnp.logical_not(selected))
            termination_causes.append(jnp.stack(exclusive_causes, axis=-1))
        if transition.tracking_errors is not None:
            tracking_errors.append(jnp.array(transition.tracking_errors, copy=True))

        if config.observation_normalization:
            normalization = update_normalization(normalization, next_observation)
        observation = next_observation

    batch = RolloutBatch(
        observation=jnp.stack(observations),
        action=jnp.stack(actions),
        log_probability=jnp.stack(log_probabilities),
        value=jnp.stack(values),
        reward=jnp.stack(rewards),
        done=jnp.stack(dones),
        action_mean=jnp.stack(action_means),
        action_std=jnp.stack(action_stds),
    )
    stacked_action_means = jnp.stack(action_means)
    stacked_action_stds = jnp.stack(action_stds)
    metrics = RolloutMetrics(
        mean_step_reward=jnp.stack(raw_rewards).mean(),
        reference_end_count=jnp.stack(reference_end_counts).sum(),
        episode_return_sum=jnp.stack(episode_returns).sum(),
        episode_length_sum=jnp.stack(episode_lengths).sum(),
        episode_reference_progress_sum=jnp.stack(episode_reference_progress).sum(),
        episode_count=jnp.stack(episode_counts).sum(),
        action_mean_abs_mean=jnp.abs(stacked_action_means).mean(),
        action_std_mean=stacked_action_stds.mean(),
        objective_term_mean=jnp.stack(objective_terms).mean(axis=(0, 1)) if objective_terms else None,
        termination_cause_count=(jnp.stack(termination_causes).sum(axis=(0, 1)) if termination_causes else None),
        tracking_error_mean=jnp.stack(tracking_errors).mean(axis=(0, 1)) if tracking_errors else None,
    )
    return batch, observation, normalization, key, metrics
