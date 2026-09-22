# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional observation normalization matching RSL-RL semantics."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class NormalizationState:
    """Device-resident running population moments."""

    mean: jax.Array
    variance: jax.Array
    count: jax.Array


def create_normalization_state(observation_dim: int) -> NormalizationState:
    """Create the zero-mean, unit-variance state used by RSL-RL."""
    if observation_dim <= 0:
        raise ValueError(f"observation_dim must be positive, got {observation_dim}")
    return NormalizationState(
        mean=jnp.zeros((observation_dim,), dtype=jnp.float32),
        variance=jnp.ones((observation_dim,), dtype=jnp.float32),
        count=jnp.asarray(0, dtype=jnp.int32),
    )


@jax.jit
def update_normalization(state: NormalizationState, observation: jax.Array) -> NormalizationState:
    """Merge one observation batch into the running population moments."""
    batch_count = observation.shape[0]
    count = state.count + batch_count
    rate = jnp.asarray(batch_count, dtype=observation.dtype) / count.astype(observation.dtype)
    batch_mean = observation.mean(axis=0)
    batch_variance = observation.var(axis=0)
    delta_mean = batch_mean - state.mean
    mean = state.mean + rate * delta_mean
    variance = state.variance + rate * (batch_variance - state.variance + delta_mean * (batch_mean - mean))
    return NormalizationState(mean=mean, variance=variance, count=count)


@jax.jit
def normalize_observation(
    state: NormalizationState,
    observation: jax.Array,
    epsilon: float = 1.0e-2,
) -> jax.Array:
    """Normalize observations with RSL-RL's standard-deviation epsilon."""
    return (observation - state.mean) / (jnp.sqrt(state.variance) + epsilon)
