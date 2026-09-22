# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpointable FlashSAC state and one donated end-to-end learner executable."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
import math
from typing import Any

import flax
from flax.core import FrozenDict
import jax
import jax.numpy as jnp
import optax

from flash_chord.lifecycle.reset import RESET_CONTEXT_NAMES
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.exploration import (
    ExplorationState,
    collection_action,
    create_exploration,
    truncated_zeta_cdf,
)
from flash_chord.training.flash_sac.network import (
    Actor,
    Critic,
    project_unit_parameters,
    sample_policy,
)
from flash_chord.training.flash_sac.replay import (
    ReplayInsert,
    ReplaySpec,
    ReplayState,
    _can_sample,
    _clear_replay,
    _flush_pending,
    _insert_replay,
    _relabel_rewards,
    _sample_replay,
    create_replay,
)
from flash_chord.training.flash_sac.reward_normalization import (
    RewardNormalizerState,
    create_reward_normalizer,
    normalize_rewards,
    reward_scale_denominator,
    update_reward_normalizer,
)
from flash_chord.training.flash_sac.update import (
    ActorLoss,
    LossScaleState,
    TemperatureLoss,
    actor_loss,
    categorical_cross_entropy,
    categorical_td_target,
    create_loss_scale,
    ema_parameters,
    gradients_are_finite,
    learning_rate,
    select_min_q_log_probability,
    temperature_loss,
    unscale_gradients,
    update_loss_scale,
)


def reset_context_indices(context_names: tuple[str, ...]) -> tuple[int, int, int]:
    """Validate the critic context prefix and return its reset/VOC scalar indices."""
    names = tuple(context_names)
    if names[: len(RESET_CONTEXT_NAMES)] != RESET_CONTEXT_NAMES:
        raise ValueError(f"critic context must begin with {RESET_CONTEXT_NAMES}, got {names}")
    if len(names) != len(set(names)):
        raise ValueError(f"critic context names must be unique, got {names}")
    return names.index(RESET_CONTEXT_NAMES[0]), names.index(RESET_CONTEXT_NAMES[1]), names.index(RESET_CONTEXT_NAMES[2])


@flax.struct.dataclass
class OptimizedNetworkState:
    """Parameters, BatchNorm state, Adam moments, and an independent scheduler clock."""

    params: Any
    batch_stats: Any
    optimizer_state: optax.OptState
    schedule_step: jax.Array


@flax.struct.dataclass
class TargetCriticState:
    """Target parameters and independently evolving BatchNorm state."""

    params: Any
    batch_stats: Any


@flax.struct.dataclass
class TemperatureState:
    """Direct log-temperature parameter, Adam state, and delayed scheduler clock."""

    log_temperature: jax.Array
    optimizer_state: optax.OptState
    schedule_step: jax.Array


@flax.struct.dataclass
class _MetricAccumulator:
    actor_loss: jax.Array
    actor_entropy: jax.Array
    actor_attempts: jax.Array
    actor_skips: jax.Array
    critic_loss: jax.Array
    critic_attempts: jax.Array
    critic_skips: jax.Array
    temperature_loss: jax.Array
    temperature_value: jax.Array
    temperature_attempts: jax.Array


@flax.struct.dataclass
class LoggingState:
    """Device-resident episode state and sums for the current external logging window."""

    running_episode_return: jax.Array
    running_episode_length: jax.Array
    objective_contribution_sum: jax.Array
    environment_step_count: jax.Array
    episode_return_sum: jax.Array
    episode_length_sum: jax.Array
    episode_reference_progress_sum: jax.Array
    episode_count: jax.Array
    reference_end_count: jax.Array
    termination_cause_count: jax.Array
    tracking_error_sum: jax.Array
    action_abs_sum: jax.Array
    optimizer: _MetricAccumulator


@flax.struct.dataclass
class LearnerState:
    """Complete device-resident state consumed and donated after each Warp step."""

    actor: OptimizedNetworkState
    critic: OptimizedNetworkState
    target_critic: TargetCriticState
    temperature: TemperatureState
    replay: ReplayState
    reward_normalizer: RewardNormalizerState
    exploration: ExplorationState
    loss_scale: LossScaleState
    logging: LoggingState
    observation: jax.Array
    critic_context: jax.Array
    key: jax.Array
    update_credit: jax.Array
    environment_steps: jax.Array
    global_update_step: jax.Array


@flax.struct.dataclass
class Transition:
    """One vector environment result, including terminal-aware and post-reset state."""

    action: jax.Array
    objective_terms: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    episode_reference_progress: jax.Array
    termination_diagnostics: jax.Array
    next_observation: jax.Array
    next_critic_context: jax.Array
    post_reset_observation: jax.Array
    post_reset_critic_context: jax.Array


@flax.struct.dataclass
class LearnerMetrics:
    """Small fixed-shape diagnostics synchronized only at runner logging cadence."""

    actor_loss_mean_per_update: jax.Array
    actor_entropy_mean_per_update: jax.Array
    actor_skipped_fraction: jax.Array
    critic_loss_mean_per_update: jax.Array
    critic_skipped_fraction: jax.Array
    temperature_loss_mean_per_update: jax.Array
    temperature_value_mean_per_update: jax.Array
    actor_update_count: jax.Array
    critic_update_count: jax.Array
    reward_mean_per_env_step: jax.Array
    objective_contribution_mean_per_env_step: jax.Array
    episode_return_mean_per_completed_episode: jax.Array
    episode_length_mean_per_completed_episode: jax.Array
    episode_reference_progress_mean_per_completed_episode: jax.Array
    episode_count: jax.Array
    reference_end_count: jax.Array
    termination_cause_count: jax.Array
    tracking_error_mean_per_env_step: jax.Array
    action_abs_mean_per_env_action: jax.Array
    replay_size: jax.Array
    reward_scale_denominator: jax.Array
    loss_scale: jax.Array


@dataclass(frozen=True)
class LearnerExecutables:
    """The two compiled JAX entry points surrounding the external Warp graph."""

    initial_action: Callable
    consume_transition: Callable


def _variables(state: OptimizedNetworkState | TargetCriticState) -> dict[str, Any]:
    return {"params": state.params, "batch_stats": state.batch_stats}


def _create_adam(config: TrainingConfig) -> optax.GradientTransformation:
    return optax.scale_by_adam(
        b1=config.optimization.adam_beta1,
        b2=config.optimization.adam_beta2,
        eps=config.optimization.adam_epsilon,
    )


def _apply_adam_if_finite(
    optimizer: optax.GradientTransformation,
    params,
    optimizer_state: optax.OptState,
    gradients,
    rate: jax.Array,
    finite: jax.Array,
    update_transform: Callable[[Any], Any] | None = None,
):
    """Skip only parameters and Adam moments; callers retain all other attempted-update state."""

    def apply(_):
        updates, next_optimizer_state = optimizer.update(gradients, optimizer_state, params)
        if update_transform is not None:
            updates = update_transform(updates)
        updates = jax.tree.map(lambda update: -rate * update, updates)
        return optax.apply_updates(params, updates), next_optimizer_state

    return jax.lax.cond(finite, apply, lambda _: (params, optimizer_state), operand=None)


def _scale_residual_mean_updates(updates, multiplier: float, mean_head: str = "unit_gain"):
    """Scale only residual mean amplitude updates, preserving the optimizer state tree."""
    if not isinstance(updates, Mapping):
        raise TypeError(f"actor updates must be a mapping, got {type(updates).__name__}")
    if "mean" not in updates or "mean_bias" not in updates:
        raise ValueError("residual actor updates require 'mean' and 'mean_bias' parameter paths")
    mean_updates = updates["mean"]
    if mean_head not in {"unit_gain", "zero_dense"}:
        raise ValueError(f"unsupported residual mean head {mean_head!r}")
    parameter_name = "gain" if mean_head == "unit_gain" else "kernel"
    if not isinstance(mean_updates, Mapping) or parameter_name not in mean_updates:
        raise ValueError(f"residual actor updates require the 'mean.{parameter_name}' parameter path")

    def replace(mapping, replacements):
        if isinstance(mapping, FrozenDict):
            return mapping.copy(replacements)
        if isinstance(mapping, dict):
            return {**mapping, **replacements}
        raise TypeError(f"actor update branch must be a dict or FrozenDict, got {type(mapping).__name__}")

    mean_updates = replace(mean_updates, {parameter_name: mean_updates[parameter_name] * multiplier})
    return replace(
        updates,
        {
            "mean": mean_updates,
            "mean_bias": updates["mean_bias"] * multiplier,
        },
    )


def _zero_metrics() -> _MetricAccumulator:
    zero = lambda: jnp.zeros((), dtype=jnp.float32)  # noqa: E731
    zero_count = lambda: jnp.zeros((), dtype=jnp.int32)  # noqa: E731
    return _MetricAccumulator(
        actor_loss=zero(),
        actor_entropy=zero(),
        actor_attempts=zero_count(),
        actor_skips=zero_count(),
        critic_loss=zero(),
        critic_attempts=zero_count(),
        critic_skips=zero_count(),
        temperature_loss=zero(),
        temperature_value=zero(),
        temperature_attempts=zero_count(),
    )


def _zero_logging_state(
    world_count: int,
    objective_term_count: int,
    termination_cause_count: int,
    tracking_error_count: int,
) -> LoggingState:
    return LoggingState(
        running_episode_return=jnp.zeros(world_count, dtype=jnp.float32),
        running_episode_length=jnp.zeros(world_count, dtype=jnp.int32),
        objective_contribution_sum=jnp.zeros(objective_term_count, dtype=jnp.float32),
        environment_step_count=jnp.zeros((), dtype=jnp.int32),
        episode_return_sum=jnp.zeros((), dtype=jnp.float32),
        episode_length_sum=jnp.zeros((), dtype=jnp.float32),
        episode_reference_progress_sum=jnp.zeros((), dtype=jnp.float32),
        episode_count=jnp.zeros((), dtype=jnp.int32),
        reference_end_count=jnp.zeros((), dtype=jnp.int32),
        termination_cause_count=jnp.zeros(termination_cause_count, dtype=jnp.int32),
        tracking_error_sum=jnp.zeros(tracking_error_count, dtype=jnp.float32),
        action_abs_sum=jnp.zeros((), dtype=jnp.float32),
        optimizer=_zero_metrics(),
    )


def reset_logging_window(state: LoggingState) -> LoggingState:
    """Clear completed-window sums while preserving in-progress episode state."""
    return state.replace(
        objective_contribution_sum=jnp.zeros_like(state.objective_contribution_sum),
        environment_step_count=jnp.zeros_like(state.environment_step_count),
        episode_return_sum=jnp.zeros_like(state.episode_return_sum),
        episode_length_sum=jnp.zeros_like(state.episode_length_sum),
        episode_reference_progress_sum=jnp.zeros_like(state.episode_reference_progress_sum),
        episode_count=jnp.zeros_like(state.episode_count),
        reference_end_count=jnp.zeros_like(state.reference_end_count),
        termination_cause_count=jnp.zeros_like(state.termination_cause_count),
        tracking_error_sum=jnp.zeros_like(state.tracking_error_sum),
        action_abs_sum=jnp.zeros_like(state.action_abs_sum),
        optimizer=_zero_metrics(),
    )


def _add_metrics(left: _MetricAccumulator, right: _MetricAccumulator) -> _MetricAccumulator:
    return jax.tree.map(jnp.add, left, right)


def _accumulate_logging(
    state: LoggingState,
    optimizer_metrics: _MetricAccumulator,
    transition: Transition,
    one_step_reward: jax.Array,
    objective_weights: jax.Array,
    frame_dt: jax.Array,
) -> LoggingState:
    """Accumulate one vector transition without synchronizing or launching a separate executable."""
    terminated = transition.terminated.astype(jnp.bool_)
    truncated = transition.truncated.astype(jnp.bool_)
    completed = jnp.logical_or(terminated, truncated)
    next_episode_return = state.running_episode_return + one_step_reward
    next_episode_length = state.running_episode_length + 1

    cause_count = state.termination_cause_count.shape[0]
    raw_causes = transition.termination_diagnostics[:, :cause_count].astype(jnp.bool_)
    remaining = terminated
    cause_increments = []
    for index in range(cause_count):
        selected = jnp.logical_and(remaining, raw_causes[:, index])
        cause_increments.append(selected.sum(dtype=jnp.int32))
        remaining = jnp.logical_and(remaining, jnp.logical_not(selected))
    if cause_increments:
        cause_increment = jnp.stack(cause_increments)
    else:
        cause_increment = jnp.zeros_like(state.termination_cause_count)

    reference_end = jnp.logical_and(truncated, jnp.logical_not(terminated))

    tracking_errors = transition.termination_diagnostics[:, cause_count:]
    return state.replace(
        running_episode_return=jnp.where(completed, 0.0, next_episode_return),
        running_episode_length=jnp.where(completed, 0, next_episode_length),
        objective_contribution_sum=(
            state.objective_contribution_sum + frame_dt * transition.objective_terms.sum(axis=0) * objective_weights
        ),
        environment_step_count=state.environment_step_count + one_step_reward.size,
        episode_return_sum=(state.episode_return_sum + jnp.where(completed, next_episode_return, 0.0).sum()),
        episode_length_sum=(
            state.episode_length_sum + jnp.where(completed, next_episode_length, 0).astype(jnp.float32).sum()
        ),
        episode_reference_progress_sum=(
            state.episode_reference_progress_sum
            + jnp.where(completed, transition.episode_reference_progress, 0.0).sum()
        ),
        episode_count=state.episode_count + completed.sum(dtype=jnp.int32),
        reference_end_count=(state.reference_end_count + reference_end.sum(dtype=jnp.int32)),
        termination_cause_count=state.termination_cause_count + cause_increment,
        tracking_error_sum=state.tracking_error_sum + tracking_errors.sum(axis=0),
        action_abs_sum=state.action_abs_sum + jnp.abs(transition.action).sum(),
        optimizer=_add_metrics(state.optimizer, optimizer_metrics),
    )


def _record_actor(
    metrics: _MetricAccumulator,
    actor: ActorLoss,
    temperature: TemperatureLoss,
    finite: jax.Array,
) -> _MetricAccumulator:
    return metrics.replace(
        actor_loss=metrics.actor_loss + actor.loss,
        actor_entropy=metrics.actor_entropy + actor.entropy,
        actor_attempts=metrics.actor_attempts + 1,
        actor_skips=metrics.actor_skips + (~finite).astype(jnp.int32),
        temperature_loss=metrics.temperature_loss + temperature.loss,
        temperature_value=metrics.temperature_value + temperature.value,
        temperature_attempts=metrics.temperature_attempts + 1,
    )


def _record_critic(
    metrics: _MetricAccumulator,
    loss: jax.Array,
    finite: jax.Array,
) -> _MetricAccumulator:
    return metrics.replace(
        critic_loss=metrics.critic_loss + loss,
        critic_attempts=metrics.critic_attempts + 1,
        critic_skips=metrics.critic_skips + (~finite).astype(jnp.int32),
    )


def _mean(total: jax.Array, count: jax.Array) -> jax.Array:
    return total / jnp.maximum(count.astype(jnp.float32), 1.0)


def _finalize_metrics(
    logging: LoggingState,
    state: LearnerState,
    config: TrainingConfig,
) -> LearnerMetrics:
    accumulator = logging.optimizer
    actor_count = accumulator.actor_attempts
    critic_count = accumulator.critic_attempts
    temperature_count = accumulator.temperature_attempts
    environment_step_count = logging.environment_step_count
    action_value_count = environment_step_count * state.exploration.noise.shape[1]
    episode_count = logging.episode_count
    return LearnerMetrics(
        actor_loss_mean_per_update=_mean(accumulator.actor_loss, actor_count),
        actor_entropy_mean_per_update=_mean(accumulator.actor_entropy, actor_count),
        actor_skipped_fraction=_mean(accumulator.actor_skips, actor_count),
        critic_loss_mean_per_update=_mean(accumulator.critic_loss, critic_count),
        critic_skipped_fraction=_mean(accumulator.critic_skips, critic_count),
        temperature_loss_mean_per_update=_mean(accumulator.temperature_loss, temperature_count),
        temperature_value_mean_per_update=_mean(accumulator.temperature_value, temperature_count),
        actor_update_count=actor_count,
        critic_update_count=critic_count,
        reward_mean_per_env_step=_mean(logging.objective_contribution_sum.sum(), environment_step_count),
        objective_contribution_mean_per_env_step=_mean(
            logging.objective_contribution_sum,
            environment_step_count,
        ),
        episode_return_mean_per_completed_episode=_mean(logging.episode_return_sum, episode_count),
        episode_length_mean_per_completed_episode=_mean(logging.episode_length_sum, episode_count),
        episode_reference_progress_mean_per_completed_episode=_mean(
            logging.episode_reference_progress_sum,
            episode_count,
        ),
        episode_count=episode_count,
        reference_end_count=logging.reference_end_count,
        termination_cause_count=logging.termination_cause_count,
        tracking_error_mean_per_env_step=_mean(logging.tracking_error_sum, environment_step_count),
        action_abs_mean_per_env_action=_mean(logging.action_abs_sum, action_value_count),
        replay_size=state.replay.size,
        reward_scale_denominator=reward_scale_denominator(
            state.reward_normalizer,
            bound=config.reward_normalization.normalized_return_bound,
        ),
        loss_scale=state.loss_scale.scale,
    )


def _validate_transition(state: LearnerState, transition: Transition, objective_weights: jax.Array) -> None:
    replay = state.replay.spec
    diagnostic_width = state.logging.termination_cause_count.size + state.logging.tracking_error_sum.size
    expected = {
        "action": ((replay.world_count, replay.action_dim), jnp.float32),
        "objective_terms": ((replay.world_count, replay.objective_term_count), jnp.float32),
        "terminated": ((replay.world_count,), jnp.int32),
        "truncated": ((replay.world_count,), jnp.int32),
        "episode_reference_progress": ((replay.world_count,), jnp.float32),
        "termination_diagnostics": ((replay.world_count, diagnostic_width), jnp.float32),
        "next_observation": ((replay.world_count, replay.observation_dim), jnp.float32),
        "next_critic_context": ((replay.world_count, replay.critic_context_dim), jnp.float32),
        "post_reset_observation": ((replay.world_count, replay.observation_dim), jnp.float32),
        "post_reset_critic_context": ((replay.world_count, replay.critic_context_dim), jnp.float32),
    }
    for name, (shape, dtype) in expected.items():
        value = getattr(transition, name)
        if value.shape != shape:
            raise ValueError(f"transition {name} has shape {value.shape}; expected {shape}")
        if value.dtype != jnp.dtype(dtype):
            raise TypeError(f"transition {name} has dtype {value.dtype}; expected {jnp.dtype(dtype)}")
    if objective_weights.shape != (replay.objective_term_count,):
        raise ValueError(
            f"objective_weights has shape {objective_weights.shape}; expected {(replay.objective_term_count,)}"
        )
    if objective_weights.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"objective_weights must be float32, got {objective_weights.dtype}")


def _collection_action(
    actor: Actor,
    actor_params,
    actor_batch_stats,
    replay_size: jax.Array,
    replay_minimum_size: int,
    warmup_action: str,
    observation: jax.Array,
    exploration: ExplorationState,
    key: jax.Array,
    cdf: jax.Array,
    action_transform: str,
    action_scale: float,
) -> tuple[jax.Array, ExplorationState, jax.Array]:
    """Select configured warm-up behavior and learned zeta-repeated collection."""

    def learned(operand):
        current_exploration, current_key = operand
        return collection_action(
            actor,
            {"params": actor_params, "batch_stats": actor_batch_stats},
            observation,
            current_exploration,
            current_key,
            cdf,
            stochastic=True,
            action_transform=action_transform,
            action_scale=action_scale,
        )

    def warmup(operand):
        current_exploration, current_key = operand
        current_key, action_key = jax.random.split(current_key)
        action = jax.random.uniform(
            action_key,
            current_exploration.noise.shape,
            minval=-action_scale,
            maxval=action_scale,
            dtype=jnp.float32,
        )
        return action, current_exploration, current_key

    if warmup_action == "policy":
        return learned((exploration, key))
    return jax.lax.cond(
        replay_size >= replay_minimum_size,
        learned,
        warmup,
        (exploration, key),
    )


def _actor_temperature_update(
    state: LearnerState,
    batch,
    metrics: _MetricAccumulator,
    key: jax.Array,
    actor: Actor,
    critic: Critic,
    optimizer: optax.GradientTransformation,
    config: TrainingConfig,
    loss_scaling: bool,
) -> tuple[LearnerState, _MetricAccumulator, jax.Array]:
    key, actor_key = jax.random.split(key)
    batch_size = batch.observation.shape[0]
    actor_observation = jnp.concatenate((batch.observation, batch.next_observation), axis=0)
    critic_observation = jnp.concatenate((batch.observation, batch.critic_context), axis=-1)

    def scaled_actor_objective(params):
        distribution, mutable = actor.apply(
            {"params": params, "batch_stats": state.actor.batch_stats},
            actor_observation,
            training=True,
            mutable=["batch_stats"],
        )
        sample = sample_policy(
            distribution,
            actor_key,
            action_transform=config.network.action_transform,
            action_scale=config.network.action_scale,
        )
        action = sample.action[:batch_size]
        log_probability = sample.log_probability[:batch_size]
        q = critic.apply(
            _variables(state.critic),
            critic_observation,
            action,
            training=False,
        ).q
        result = actor_loss(
            log_probability,
            jnp.minimum(q[0], q[1]),
            action,
            batch.action,
            jnp.exp(state.temperature.log_temperature),
            behavior_cloning_coefficient=config.optimization.behavior_cloning_coefficient,
        )
        return result.loss * state.loss_scale.scale, (result, mutable["batch_stats"])

    (_, (actor_result, actor_batch_stats)), scaled_gradients = jax.value_and_grad(
        scaled_actor_objective,
        has_aux=True,
    )(state.actor.params)
    actor_gradients = unscale_gradients(state.loss_scale, scaled_gradients)
    actor_finite = gradients_are_finite(actor_gradients)
    actor_rate = (
        learning_rate(
            state.actor.schedule_step,
            config.optimization,
            config.learning_rate_schedule_updates,
        )
        * config.optimization.actor_learning_rate_multiplier
    )
    actor_update_transform = None
    if (
        config.network.actor_head_mode == "residual"
        and config.optimization.residual_mean_learning_rate_multiplier != 1.0
    ):
        actor_update_transform = partial(
            _scale_residual_mean_updates,
            multiplier=config.optimization.residual_mean_learning_rate_multiplier,
            mean_head=config.network.residual_mean_head,
        )
    actor_params, actor_optimizer_state = _apply_adam_if_finite(
        optimizer,
        state.actor.params,
        state.actor.optimizer_state,
        actor_gradients,
        actor_rate,
        actor_finite,
        actor_update_transform,
    )
    actor_params = project_unit_parameters(actor_params, config.network.projection_epsilon)
    loss_scale = update_loss_scale(
        state.loss_scale,
        actor_finite,
        config.optimization,
        enabled=loss_scaling,
    )
    actor_state = state.actor.replace(
        params=actor_params,
        batch_stats=actor_batch_stats,
        optimizer_state=actor_optimizer_state,
        schedule_step=state.actor.schedule_step + 1,
    )

    def temperature_objective(log_temperature):
        result = temperature_loss(
            log_temperature,
            actor_result.entropy,
            config.exploration.target_entropy(
                batch.action.shape[-1],
                config.network.action_scale,
            ),
        )
        return result.loss, result

    (_, temperature_result), temperature_gradients = jax.value_and_grad(
        temperature_objective,
        has_aux=True,
    )(state.temperature.log_temperature)
    temperature_rate = learning_rate(
        state.temperature.schedule_step,
        config.optimization,
        config.learning_rate_schedule_updates,
    )
    temperature_updates, temperature_optimizer_state = optimizer.update(
        temperature_gradients,
        state.temperature.optimizer_state,
        state.temperature.log_temperature,
    )
    temperature_updates = jax.tree.map(lambda update: -temperature_rate * update, temperature_updates)
    log_temperature = optax.apply_updates(state.temperature.log_temperature, temperature_updates)
    temperature_state = state.temperature.replace(
        log_temperature=log_temperature,
        optimizer_state=temperature_optimizer_state,
        schedule_step=state.temperature.schedule_step + 1,
    )

    state = state.replace(actor=actor_state, temperature=temperature_state, loss_scale=loss_scale)
    return state, _record_actor(metrics, actor_result, temperature_result, actor_finite), key


def _critic_update(
    state: LearnerState,
    batch,
    rewards: jax.Array,
    metrics: _MetricAccumulator,
    key: jax.Array,
    actor: Actor,
    critic: Critic,
    optimizer: optax.GradientTransformation,
    config: TrainingConfig,
    loss_scaling: bool,
) -> tuple[LearnerState, _MetricAccumulator, jax.Array]:
    key, action_key = jax.random.split(key)
    next_distribution = actor.apply(
        _variables(state.actor),
        batch.next_observation,
        training=False,
    )
    next_sample = sample_policy(
        next_distribution,
        action_key,
        action_transform=config.network.action_transform,
        action_scale=config.network.action_scale,
    )
    current_observation = jnp.concatenate((batch.observation, batch.critic_context), axis=-1)
    next_observation = jnp.concatenate((batch.next_observation, batch.next_critic_context), axis=-1)
    observation = jnp.concatenate((current_observation, next_observation), axis=0)
    action = jnp.concatenate((batch.action, next_sample.action), axis=0)

    target_output, target_mutable = critic.apply(
        _variables(state.target_critic),
        observation,
        action,
        training=True,
        mutable=["batch_stats"],
    )
    next_q = target_output.q[:, batch.observation.shape[0] :]
    next_log_probability = target_output.log_probability[:, batch.observation.shape[0] :]
    selected_log_probability = select_min_q_log_probability(next_q, next_log_probability)
    temperature_value = jnp.exp(state.temperature.log_temperature)
    temperature_log_probability = temperature_value * next_sample.log_probability
    target_probability = categorical_td_target(
        selected_log_probability,
        rewards,
        batch.terminated,
        temperature_log_probability,
        bootstrap_discount=state.replay.spec.bootstrap_discount,
        value_min=-config.reward_normalization.normalized_return_bound,
        value_max=config.reward_normalization.normalized_return_bound,
    )

    def scaled_critic_objective(params):
        output, mutable = critic.apply(
            {"params": params, "batch_stats": state.critic.batch_stats},
            observation,
            action,
            training=True,
            mutable=["batch_stats"],
        )
        predicted_log_probability = output.log_probability[:, : batch.observation.shape[0]]
        loss = categorical_cross_entropy(target_probability, predicted_log_probability)
        return loss * state.loss_scale.scale, (loss, mutable["batch_stats"])

    (_, (critic_loss, critic_batch_stats)), scaled_gradients = jax.value_and_grad(
        scaled_critic_objective,
        has_aux=True,
    )(state.critic.params)
    critic_gradients = unscale_gradients(state.loss_scale, scaled_gradients)
    critic_finite = gradients_are_finite(critic_gradients)
    critic_rate = learning_rate(
        state.critic.schedule_step,
        config.optimization,
        config.learning_rate_schedule_updates,
    )
    critic_params, critic_optimizer_state = _apply_adam_if_finite(
        optimizer,
        state.critic.params,
        state.critic.optimizer_state,
        critic_gradients,
        critic_rate,
        critic_finite,
    )
    critic_params = project_unit_parameters(critic_params, config.network.projection_epsilon)
    loss_scale = update_loss_scale(
        state.loss_scale,
        critic_finite,
        config.optimization,
        enabled=loss_scaling,
    )
    critic_state = state.critic.replace(
        params=critic_params,
        batch_stats=critic_batch_stats,
        optimizer_state=critic_optimizer_state,
        schedule_step=state.critic.schedule_step + 1,
    )
    target_state = state.target_critic.replace(
        params=ema_parameters(
            state.target_critic.params,
            critic_params,
            config.optimization.target_tau,
        ),
        batch_stats=target_mutable["batch_stats"],
    )
    state = state.replace(
        critic=critic_state,
        target_critic=target_state,
        loss_scale=loss_scale,
        key=key,
        global_update_step=state.global_update_step + 1,
    )
    metrics = _record_critic(
        metrics,
        critic_loss,
        critic_finite,
    )
    return state, metrics, key


def _optimizer_attempt(
    state: LearnerState,
    metrics: _MetricAccumulator,
    objective_weights: jax.Array,
    frame_dt: jax.Array,
    actor: Actor,
    critic: Critic,
    optimizer: optax.GradientTransformation,
    config: TrainingConfig,
    loss_scaling: bool,
) -> tuple[LearnerState, _MetricAccumulator]:
    key, replay_key = jax.random.split(state.key)
    batch = _sample_replay(state.replay, replay_key)
    rewards = _relabel_rewards(batch, objective_weights, frame_dt)
    if config.reward_normalization.enabled:
        rewards = normalize_rewards(
            state.reward_normalizer,
            rewards,
            bound=config.reward_normalization.normalized_return_bound,
            epsilon=config.reward_normalization.epsilon,
        )
    state = state.replace(key=key)
    do_actor_update = state.global_update_step % config.optimization.actor_update_period == 0

    def update_actor(operand):
        current_state, current_metrics = operand
        current_state, current_metrics, current_key = _actor_temperature_update(
            current_state,
            batch,
            current_metrics,
            current_state.key,
            actor,
            critic,
            optimizer,
            config,
            loss_scaling,
        )
        return current_state.replace(key=current_key), current_metrics

    state, metrics = jax.lax.cond(
        do_actor_update,
        update_actor,
        lambda operand: operand,
        (state, metrics),
    )
    state, metrics, key = _critic_update(
        state,
        batch,
        rewards,
        metrics,
        state.key,
        actor,
        critic,
        optimizer,
        config,
        loss_scaling,
    )
    return state.replace(key=key), metrics


def create_learner(
    config: TrainingConfig,
    initial_observation: jax.Array,
    initial_critic_context: jax.Array,
    *,
    action_dim: int,
    objective_term_count: int,
    termination_cause_count: int = 0,
    tracking_error_count: int = 0,
    critic_context_names: tuple[str, ...] = RESET_CONTEXT_NAMES,
    device: jax.Device | None = None,
) -> tuple[LearnerState, LearnerExecutables]:
    """Initialize state and compile the two stable entry points for one task shape."""
    if action_dim <= 0 or objective_term_count <= 0:
        raise ValueError(
            f"action_dim and objective_term_count must be positive, got {action_dim} and {objective_term_count}"
        )
    if termination_cause_count < 0 or tracking_error_count < 0:
        raise ValueError(
            "termination_cause_count and tracking_error_count must be non-negative, got "
            f"{termination_cause_count} and {tracking_error_count}"
        )
    if initial_observation.shape[0] != config.world_count or initial_observation.ndim != 2:
        raise ValueError(
            f"initial_observation must have shape ({config.world_count}, observation_dim), "
            f"got {initial_observation.shape}"
        )
    if initial_critic_context.shape[0] != config.world_count or initial_critic_context.ndim != 2:
        raise ValueError(
            f"initial_critic_context must have shape ({config.world_count}, context_dim), "
            f"got {initial_critic_context.shape}"
        )
    context_names = tuple(critic_context_names)
    context_indices = reset_context_indices(context_names)
    if initial_critic_context.shape[1] != len(context_names):
        raise ValueError(
            f"initial_critic_context width {initial_critic_context.shape[1]} does not match "
            f"critic context names {context_names}"
        )
    if initial_observation.dtype != jnp.dtype(jnp.float32) or initial_critic_context.dtype != jnp.dtype(jnp.float32):
        raise TypeError("initial observation and critic context must be float32")

    observation_dim = initial_observation.shape[1]
    critic_context_dim = initial_critic_context.shape[1]
    compute_dtype = jnp.dtype(config.compute_dtype) if config.mixed_precision else jnp.dtype(jnp.float32)
    loss_scaling = config.mixed_precision and compute_dtype == jnp.dtype(jnp.float16)
    actor = Actor(action_dim=action_dim, config=config.network, dtype=compute_dtype)
    value_min, value_max = config.reward_normalization.value_support
    critic = Critic(value_min=value_min, value_max=value_max, config=config.network, dtype=compute_dtype)
    optimizer = _create_adam(config)
    replay_spec = ReplaySpec.from_config(
        config.replay,
        discount=config.discount,
        world_count=config.world_count,
        observation_dim=observation_dim,
        action_dim=action_dim,
        objective_term_count=objective_term_count,
        critic_context_dim=critic_context_dim,
    )

    device_context = nullcontext() if device is None else jax.default_device(device)
    with device_context:
        key = jax.random.PRNGKey(config.seed)
        key, actor_key, critic_key = jax.random.split(key, 3)
        actor_variables = actor.init(
            actor_key,
            jnp.zeros((1, observation_dim), dtype=jnp.float32),
            training=False,
        )
        critic_variables = critic.init(
            critic_key,
            jnp.zeros((1, observation_dim + critic_context_dim), dtype=jnp.float32),
            jnp.zeros((1, action_dim), dtype=jnp.float32),
            training=False,
        )
        actor_params = project_unit_parameters(actor_variables["params"], config.network.projection_epsilon)
        critic_params = project_unit_parameters(critic_variables["params"], config.network.projection_epsilon)
        target_params = jax.tree.map(lambda value: jnp.array(value, copy=True), critic_params)
        target_batch_stats = jax.tree.map(
            lambda value: jnp.array(value, copy=True),
            critic_variables["batch_stats"],
        )
        log_temperature = jnp.asarray(math.log(config.exploration.initial_temperature), dtype=jnp.float32)
        state = LearnerState(
            actor=OptimizedNetworkState(
                params=actor_params,
                batch_stats=actor_variables["batch_stats"],
                optimizer_state=optimizer.init(actor_params),
                schedule_step=jnp.asarray(0, dtype=jnp.int32),
            ),
            critic=OptimizedNetworkState(
                params=critic_params,
                batch_stats=critic_variables["batch_stats"],
                optimizer_state=optimizer.init(critic_params),
                schedule_step=jnp.asarray(0, dtype=jnp.int32),
            ),
            target_critic=TargetCriticState(
                params=target_params,
                batch_stats=target_batch_stats,
            ),
            temperature=TemperatureState(
                log_temperature=log_temperature,
                optimizer_state=optimizer.init(log_temperature),
                schedule_step=jnp.asarray(0, dtype=jnp.int32),
            ),
            replay=create_replay(replay_spec, device=device),
            reward_normalizer=create_reward_normalizer(config.world_count, device=device),
            exploration=create_exploration(config.world_count, action_dim, device=device),
            loss_scale=create_loss_scale(config.optimization, enabled=loss_scaling),
            logging=_zero_logging_state(
                config.world_count,
                objective_term_count,
                termination_cause_count,
                tracking_error_count,
            ),
            observation=jnp.array(initial_observation, dtype=jnp.float32, copy=True),
            critic_context=jnp.array(initial_critic_context, dtype=jnp.float32, copy=True),
            key=key,
            update_credit=jnp.asarray(0.0, dtype=jnp.float32),
            environment_steps=jnp.asarray(0, dtype=jnp.int32),
            global_update_step=jnp.asarray(0, dtype=jnp.int32),
        )
        cdf = truncated_zeta_cdf(
            config.exploration.zeta_exponent,
            config.exploration.maximum_noise_repeat,
        )

    def initial_action(
        actor_params,
        actor_batch_stats,
        replay_size,
        observation,
        exploration,
        key,
    ):
        return _collection_action(
            actor,
            actor_params,
            actor_batch_stats,
            replay_size,
            replay_spec.minimum_size,
            config.exploration.warmup_action,
            observation,
            exploration,
            key,
            cdf,
            config.network.action_transform,
            config.network.action_scale,
        )

    maximum_updates = math.ceil(config.updates_per_collection)

    def consume_transition(
        learner: LearnerState,
        transition: Transition,
        objective_weights: jax.Array,
        frame_dt: jax.Array,
        target_voc_scale: jax.Array,
        stage_changed: jax.Array,
        reset_logging: jax.Array,
    ) -> tuple[LearnerState, jax.Array, LearnerMetrics]:
        _validate_transition(learner, transition, objective_weights)
        if frame_dt.shape != () or frame_dt.dtype != jnp.dtype(jnp.float32):
            raise TypeError(f"frame_dt must be a float32 scalar, got {frame_dt.shape}/{frame_dt.dtype}")
        if target_voc_scale.shape != () or target_voc_scale.dtype != jnp.dtype(jnp.float32):
            raise TypeError(
                f"target_voc_scale must be a float32 scalar, got {target_voc_scale.shape}/{target_voc_scale.dtype}"
            )
        if stage_changed.shape != () or stage_changed.dtype != jnp.dtype(jnp.bool_):
            raise TypeError(f"stage_changed must be a bool scalar, got {stage_changed.shape}/{stage_changed.dtype}")
        if reset_logging.shape != () or reset_logging.dtype != jnp.dtype(jnp.bool_):
            raise TypeError(f"reset_logging must be a bool scalar, got {reset_logging.shape}/{reset_logging.dtype}")

        def prepare_stage(replay):
            if config.replay.clear_on_stage_change:
                return _clear_replay(replay)
            return _flush_pending(replay)

        replay = jax.lax.cond(stage_changed, prepare_stage, lambda current: current, learner.replay)
        terminated = transition.terminated.astype(jnp.bool_)
        truncated = transition.truncated.astype(jnp.bool_)

        def apply_stage_context(context):
            applied_index, target_index, settling_index = context_indices
            settling_complete = context[:, settling_index] >= 1.0
            applied_scale = jnp.where(settling_complete, target_voc_scale, context[:, applied_index])
            return context.at[:, applied_index].set(applied_scale).at[:, target_index].set(target_voc_scale)

        critic_context = jax.lax.cond(
            stage_changed,
            apply_stage_context,
            lambda context: context,
            learner.critic_context,
        )
        learner = learner.replace(critic_context=critic_context)
        one_step_reward = frame_dt * jnp.einsum("wt,t->w", transition.objective_terms, objective_weights)
        reward_normalizer = learner.reward_normalizer
        if config.reward_normalization.enabled:
            reward_normalizer = update_reward_normalizer(
                reward_normalizer,
                one_step_reward,
                terminated,
                truncated,
                discount=config.discount,
                moments_epsilon=config.reward_normalization.running_moments_epsilon,
            )
        replay = _insert_replay(
            replay,
            ReplayInsert(
                observation=learner.observation,
                action=transition.action,
                objective_terms=transition.objective_terms,
                terminated=terminated,
                truncated=truncated,
                next_observation=transition.next_observation,
                critic_context=learner.critic_context,
                next_critic_context=transition.next_critic_context,
            ),
        )
        learner = learner.replace(
            replay=replay,
            reward_normalizer=reward_normalizer,
            environment_steps=learner.environment_steps + config.world_count,
        )
        ready = _can_sample(replay)
        accrued_credit = jnp.where(
            ready,
            learner.update_credit + config.updates_per_collection,
            learner.update_credit,
        )
        update_count = jnp.floor(accrued_credit).astype(jnp.int32)

        def scan_slot(carry, slot):
            current_learner, current_metrics = carry
            return jax.lax.cond(
                slot < update_count,
                lambda operand: _optimizer_attempt(
                    operand[0],
                    operand[1],
                    objective_weights,
                    frame_dt,
                    actor,
                    critic,
                    optimizer,
                    config,
                    loss_scaling,
                ),
                lambda operand: operand,
                (current_learner, current_metrics),
            ), None

        (learner, accumulator), _ = jax.lax.scan(
            scan_slot,
            (learner, _zero_metrics()),
            jnp.arange(maximum_updates, dtype=jnp.int32),
        )
        logging = _accumulate_logging(
            learner.logging,
            accumulator,
            transition,
            one_step_reward,
            objective_weights,
            frame_dt,
        )
        learner = learner.replace(
            update_credit=accrued_credit - update_count.astype(jnp.float32),
            logging=jax.lax.cond(reset_logging, reset_logging_window, lambda current: current, logging),
            observation=learner.observation.at[:].set(transition.post_reset_observation),
            critic_context=learner.critic_context.at[:].set(transition.post_reset_critic_context),
        )
        action, exploration, key = _collection_action(
            actor,
            learner.actor.params,
            learner.actor.batch_stats,
            learner.replay.size,
            replay_spec.minimum_size,
            config.exploration.warmup_action,
            learner.observation,
            learner.exploration,
            learner.key,
            cdf,
            config.network.action_transform,
            config.network.action_scale,
        )
        learner = learner.replace(exploration=exploration, key=key)
        return learner, action, _finalize_metrics(logging, learner, config)

    return state, LearnerExecutables(
        initial_action=jax.jit(initial_action),
        consume_transition=jax.jit(consume_transition, donate_argnums=(0,)),
    )
