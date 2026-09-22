# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive FlashSAC reward scaling with exact upstream running-moment semantics."""

from __future__ import annotations

from contextlib import nullcontext
import math

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class RunningMomentsState:
    """Float32 running population moments."""

    mean: jax.Array
    variance: jax.Array
    count: jax.Array


@flax.struct.dataclass
class RewardNormalizerState:
    """Per-world discounted-return state and global scaling statistics."""

    discounted_return: jax.Array
    maximum_absolute_return: jax.Array
    moments: RunningMomentsState


def create_reward_normalizer(world_count: int, device: jax.Device | None = None) -> RewardNormalizerState:
    """Create fixed-shape float32 state without upstream's lazy shape change."""
    if world_count <= 0:
        raise ValueError(f"world_count must be positive, got {world_count}")
    device_context = nullcontext() if device is None else jax.default_device(device)
    with device_context:
        return RewardNormalizerState(
            discounted_return=jnp.zeros((world_count,), dtype=jnp.float32),
            maximum_absolute_return=jnp.asarray(0.0, dtype=jnp.float32),
            moments=RunningMomentsState(
                mean=jnp.asarray(0.0, dtype=jnp.float32),
                variance=jnp.asarray(1.0, dtype=jnp.float32),
                count=jnp.asarray(0.0, dtype=jnp.float32),
            ),
        )


def update_running_moments(
    state: RunningMomentsState,
    samples: jax.Array,
    epsilon: float = 1.0e-4,
) -> RunningMomentsState:
    """Merge one vector batch using upstream's biased variance and epsilon ordering."""
    if samples.ndim != 1 or samples.shape[0] == 0:
        raise ValueError(f"running-moment samples must be a non-empty vector, got {samples.shape}")
    if samples.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"running-moment samples must be float32, got {samples.dtype}")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")

    sample_mean = jnp.mean(samples)
    sample_variance = jnp.var(samples)
    sample_count = jnp.asarray(samples.shape[0], dtype=jnp.float32)
    delta = sample_mean - state.mean
    total_count = state.count + sample_count
    ratio = sample_count / total_count
    mean = state.mean + delta * ratio
    previous_moment = state.variance * (state.count + epsilon)
    sample_moment = sample_variance * sample_count
    combined_moment = previous_moment + sample_moment + jnp.square(delta) * state.count * ratio
    variance = combined_moment / total_count
    return RunningMomentsState(mean=mean, variance=variance, count=total_count)


def update_reward_normalizer(
    state: RewardNormalizerState,
    reward: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    *,
    discount: float,
    moments_epsilon: float = 1.0e-4,
) -> RewardNormalizerState:
    """Update once from one collected vector transition, including replay warm-up."""
    expected = state.discounted_return.shape
    for name, value, dtype in (
        ("reward", reward, jnp.float32),
        ("terminated", terminated, jnp.bool_),
        ("truncated", truncated, jnp.bool_),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected}")
        if value.dtype != jnp.dtype(dtype):
            raise TypeError(f"{name} has dtype {value.dtype}; expected {jnp.dtype(dtype)}")
    if not math.isfinite(discount) or not 0.0 < discount <= 1.0:
        raise ValueError(f"discount must be finite and in (0, 1], got {discount}")

    done = jnp.logical_or(terminated, truncated)
    discounted_return = discount * (~done).astype(jnp.float32) * state.discounted_return + reward
    maximum_absolute_return = jnp.maximum(
        state.maximum_absolute_return,
        jnp.max(jnp.abs(discounted_return)),
    )
    moments = update_running_moments(state.moments, discounted_return, moments_epsilon)
    return RewardNormalizerState(
        discounted_return=discounted_return,
        maximum_absolute_return=maximum_absolute_return,
        moments=moments,
    )


def reward_scale_denominator(
    state: RewardNormalizerState,
    *,
    bound: float,
    epsilon: float = 1.0e-8,
) -> jax.Array:
    """Return the larger of variance scaling and the lifetime return bound."""
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"bound must be finite and positive, got {bound}")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
    variance_denominator = jnp.sqrt(state.moments.variance + epsilon)
    bound_denominator = state.maximum_absolute_return / bound
    return jnp.maximum(variance_denominator, bound_denominator)


def normalize_rewards(
    state: RewardNormalizerState,
    rewards: jax.Array,
    *,
    bound: float,
    epsilon: float = 1.0e-8,
) -> jax.Array:
    """Scale replay rewards without clipping, matching upstream FlashSAC."""
    if rewards.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"rewards must be float32, got {rewards.dtype}")
    return rewards / reward_scale_denominator(state, bound=bound, epsilon=epsilon)
