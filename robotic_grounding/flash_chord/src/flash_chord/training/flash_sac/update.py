# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure FlashSAC objectives, categorical projection, schedules, and target updates."""

from __future__ import annotations

import math

import flax
import jax
import jax.numpy as jnp

from flash_chord.training.flash_sac.config import OptimizationConfig


@flax.struct.dataclass
class ActorLoss:
    """Actor objective and interpretable component metrics."""

    loss: jax.Array
    entropy: jax.Array
    mean_action: jax.Array
    behavior_cloning_loss: jax.Array


@flax.struct.dataclass
class TemperatureLoss:
    """Direct-temperature objective and its pre-update value."""

    loss: jax.Array
    value: jax.Array


@flax.struct.dataclass
class LossScaleState:
    """Shared PyTorch-compatible dynamic loss scale for actor and critic."""

    scale: jax.Array
    finite_steps: jax.Array


def create_loss_scale(config: OptimizationConfig, *, enabled: bool) -> LossScaleState:
    """Create the shared scaler, or an inert unit scale for non-FP16 training."""
    return LossScaleState(
        scale=jnp.asarray(config.loss_scale_initial if enabled else 1.0, dtype=jnp.float32),
        finite_steps=jnp.asarray(0, dtype=jnp.int32),
    )


def unscale_gradients(state: LossScaleState, gradients):
    """Convert scaled gradients to float32 before the device-side finite check."""
    return jax.tree.map(lambda gradient: gradient.astype(jnp.float32) / state.scale, gradients)


def gradients_are_finite(gradients) -> jax.Array:
    """Return one device scalar covering every gradient leaf."""
    finite = jnp.asarray(True, dtype=jnp.bool_)
    for gradient in jax.tree.leaves(gradients):
        finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(gradient)))
    return finite


def update_loss_scale(
    state: LossScaleState,
    finite: jax.Array,
    config: OptimizationConfig,
    *,
    enabled: bool,
) -> LossScaleState:
    """Match PyTorch GradScaler growth/backoff timing after one optimizer attempt."""
    if not enabled:
        return state
    next_finite_steps = state.finite_steps + 1
    grow = next_finite_steps >= config.loss_scale_growth_interval
    finite_scale = jnp.where(grow, state.scale * config.loss_scale_growth_factor, state.scale)
    scale = jnp.where(finite, finite_scale, state.scale * config.loss_scale_backoff_factor)
    finite_steps = jnp.where(jnp.logical_or(~finite, grow), 0, next_finite_steps)
    return LossScaleState(scale=scale.astype(jnp.float32), finite_steps=finite_steps.astype(jnp.int32))


def select_min_q_log_probability(q: jax.Array, log_probability: jax.Array) -> jax.Array:
    """Select the whole categorical distribution from the lower expected-Q critic."""
    if q.ndim != 2 or q.shape[0] != 2:
        raise ValueError(f"q must have shape (2, batch), got {q.shape}")
    expected = (2, q.shape[1])
    if log_probability.ndim != 3 or log_probability.shape[:2] != expected:
        raise ValueError(
            f"log_probability must have shape (2, batch, atoms), got {log_probability.shape} for q {q.shape}"
        )
    minimum_index = jnp.argmin(q, axis=0)
    gather_index = jnp.broadcast_to(
        minimum_index[None, :, None],
        (1, q.shape[1], log_probability.shape[2]),
    )
    return jnp.take_along_axis(log_probability, gather_index, axis=0)[0]


def categorical_td_target(
    target_log_probability: jax.Array,
    reward: jax.Array,
    terminated: jax.Array,
    temperature_log_probability: jax.Array,
    *,
    bootstrap_discount: float,
    value_min: float,
    value_max: float,
) -> jax.Array:
    """Project the stopped distributional Bellman target onto fixed support atoms."""
    if target_log_probability.ndim != 2:
        raise ValueError(f"target_log_probability must have shape (batch, atoms), got {target_log_probability.shape}")
    batch_size, atom_count = target_log_probability.shape
    if atom_count < 2:
        raise ValueError(f"categorical target requires at least two atoms, got {atom_count}")
    expected = (batch_size,)
    for name, value in (
        ("reward", reward),
        ("terminated", terminated),
        ("temperature_log_probability", temperature_log_probability),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected}")
    if not math.isfinite(bootstrap_discount) or not 0.0 < bootstrap_discount <= 1.0:
        raise ValueError(f"bootstrap_discount must be finite and in (0, 1], got {bootstrap_discount}")
    if not math.isfinite(value_min) or not math.isfinite(value_max) or value_min >= value_max:
        raise ValueError(f"value support must be finite and increasing, got {value_min} and {value_max}")

    atom_width = (value_max - value_min) / (atom_count - 1)
    support = value_min + atom_width * jnp.arange(atom_count, dtype=target_log_probability.dtype)
    bootstrap_mask = 1.0 - terminated.astype(target_log_probability.dtype)
    target_value = (
        reward[:, None]
        + bootstrap_discount * (support[None, :] - temperature_log_probability[:, None]) * bootstrap_mask[:, None]
    )
    target_value = jnp.clip(target_value, value_min, value_max)
    fractional_index = (target_value - value_min) / atom_width
    lower = jnp.floor(fractional_index).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, atom_count - 1)
    upper_weight = fractional_index - lower.astype(fractional_index.dtype)
    probability = jnp.exp(target_log_probability)
    batch_index = jnp.broadcast_to(jnp.arange(batch_size, dtype=jnp.int32)[:, None], lower.shape)
    target_probability = jnp.zeros_like(target_log_probability)
    target_probability = target_probability.at[batch_index, lower].add(probability * (1.0 - upper_weight))
    target_probability = target_probability.at[batch_index, upper].add(probability * upper_weight)
    return jax.lax.stop_gradient(target_probability)


def categorical_cross_entropy(target_probability: jax.Array, predicted_log_probability: jax.Array) -> jax.Array:
    """Mean categorical cross-entropy across both critics and the replay batch."""
    if predicted_log_probability.ndim != 3 or predicted_log_probability.shape[0] != 2:
        raise ValueError(
            f"predicted_log_probability must have shape (2, batch, atoms), got {predicted_log_probability.shape}"
        )
    if target_probability.shape != predicted_log_probability.shape[1:]:
        raise ValueError(
            f"target_probability has shape {target_probability.shape}; expected {predicted_log_probability.shape[1:]}"
        )
    return -(target_probability[None, ...] * predicted_log_probability).sum(axis=-1).mean()


def actor_loss(
    log_probability: jax.Array,
    minimum_q: jax.Array,
    action: jax.Array,
    replay_action: jax.Array,
    temperature: jax.Array,
    *,
    behavior_cloning_coefficient: float,
) -> ActorLoss:
    """Compute the squashed-policy objective with optional upstream BC regularization."""
    if log_probability.shape != minimum_q.shape:
        raise ValueError(f"log_probability and minimum_q shapes differ: {log_probability.shape} and {minimum_q.shape}")
    if action.shape != replay_action.shape or action.shape[0] != log_probability.shape[0]:
        raise ValueError(
            f"action/replay shapes must match the batch: {action.shape}, {replay_action.shape}, {log_probability.shape}"
        )
    if not math.isfinite(behavior_cloning_coefficient) or behavior_cloning_coefficient < 0.0:
        raise ValueError(
            f"behavior_cloning_coefficient must be finite and non-negative, got {behavior_cloning_coefficient}"
        )

    temperature = jax.lax.stop_gradient(temperature)
    loss = (temperature * log_probability - minimum_q).mean()
    behavior_cloning_loss = jnp.mean(jnp.square(action - replay_action))
    if behavior_cloning_coefficient > 0.0:
        q_scale = jax.lax.stop_gradient(jnp.abs(minimum_q).mean())
        loss = loss + behavior_cloning_coefficient * q_scale * behavior_cloning_loss
    return ActorLoss(
        loss=loss,
        entropy=-log_probability.mean(),
        mean_action=action.mean(),
        behavior_cloning_loss=behavior_cloning_loss,
    )


def temperature_loss(log_temperature: jax.Array, entropy: jax.Array, target_entropy: float) -> TemperatureLoss:
    """Optimize ``exp(log_temperature)`` directly, matching upstream FlashSAC."""
    if not math.isfinite(target_entropy):
        raise ValueError(f"target_entropy must be finite, got {target_entropy}")
    if log_temperature.size != 1:
        raise ValueError(f"log_temperature must contain one scalar, got {log_temperature.shape}")
    value = jnp.exp(log_temperature).reshape(())
    loss = jnp.mean(value * (jax.lax.stop_gradient(entropy) - target_entropy))
    return TemperatureLoss(loss=loss, value=value)


def learning_rate(
    step: jax.Array,
    config: OptimizationConfig,
    total_optimizer_updates: int,
) -> jax.Array:
    """Exact upstream linear-warmup/cosine/end schedule for one optimizer clock."""
    warmup_steps, decay_steps = config.schedule_steps(total_optimizer_updates)
    step = jnp.asarray(step)
    step_f32 = step.astype(jnp.float32)
    if warmup_steps > 0:
        warmup = config.learning_rate_initial + (
            (config.learning_rate_peak - config.learning_rate_initial) * step_f32 / warmup_steps
        )
    else:
        warmup = jnp.asarray(config.learning_rate_peak, dtype=jnp.float32)
    progress = (step_f32 - warmup_steps) / (decay_steps - warmup_steps)
    cosine = config.learning_rate_end + (
        (config.learning_rate_peak - config.learning_rate_end) * 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    )
    return jnp.where(
        step < warmup_steps,
        warmup,
        jnp.where(step < decay_steps, cosine, config.learning_rate_end),
    ).astype(jnp.float32)


def ema_parameters(target_parameters, source_parameters, tau: float):
    """Interpolate target parameters only; BatchNorm state is intentionally separate."""
    if not math.isfinite(tau) or not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be finite and in (0, 1], got {tau}")
    return jax.tree.map(lambda target, source: target + tau * (source - target), target_parameters, source_parameters)
