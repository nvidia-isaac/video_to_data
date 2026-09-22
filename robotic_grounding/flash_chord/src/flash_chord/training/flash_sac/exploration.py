# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global zeta-repeated action noise used by FlashSAC collection."""

from __future__ import annotations

from contextlib import nullcontext
import math

import flax
import jax
import jax.numpy as jnp

from flash_chord.training.flash_sac.network import Actor, deterministic_action


@flax.struct.dataclass
class ExplorationState:
    """Cached vector-batch noise and its shared repeat counters."""

    noise: jax.Array
    repeat_count: jax.Array
    repeat_length: jax.Array


def create_exploration(
    world_count: int,
    action_dim: int,
    device: jax.Device | None = None,
) -> ExplorationState:
    """Create fixed-shape exploration state; the first stochastic call renews it."""
    if world_count <= 0 or action_dim <= 0:
        raise ValueError(f"world_count and action_dim must be positive, got {world_count} and {action_dim}")
    device_context = nullcontext() if device is None else jax.default_device(device)
    with device_context:
        return ExplorationState(
            noise=jnp.zeros((world_count, action_dim), dtype=jnp.float32),
            repeat_count=jnp.asarray(0, dtype=jnp.int32),
            repeat_length=jnp.asarray(0, dtype=jnp.int32),
        )


def truncated_zeta_cdf(exponent: float, maximum_repeat: int) -> jax.Array:
    """Return the normalized float32 CDF for repeat lengths ``1..maximum_repeat``."""
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError(f"exponent must be finite and positive, got {exponent}")
    if maximum_repeat <= 0:
        raise ValueError(f"maximum_repeat must be positive, got {maximum_repeat}")
    lengths = jnp.arange(1, maximum_repeat + 1, dtype=jnp.float32)
    probability = lengths ** (-exponent)
    return jnp.cumsum(probability / jnp.sum(probability))


def repeat_length_from_uniform(cdf: jax.Array, uniform: jax.Array) -> jax.Array:
    """Match upstream's literal first-true argmax, including its all-false fallback."""
    if cdf.ndim != 1 or cdf.shape[0] == 0:
        raise ValueError(f"cdf must be a non-empty vector, got {cdf.shape}")
    if cdf.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"cdf must be float32, got {cdf.dtype}")
    return (jnp.argmax((uniform < cdf).astype(jnp.int32)) + 1).astype(jnp.int32)


def apply_repeated_noise(
    mean: jax.Array,
    standard_deviation: jax.Array,
    state: ExplorationState,
    candidate_noise: jax.Array,
    candidate_repeat_length: jax.Array,
    *,
    noise_scale: float,
    action_transform: str = "tanh",
    action_scale: float = 1.0,
) -> tuple[jax.Array, ExplorationState]:
    """Select or retain one global noise tensor and return transformed actions."""
    expected = mean.shape
    for name, value in (
        ("standard_deviation", standard_deviation),
        ("state.noise", state.noise),
        ("candidate_noise", candidate_noise),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected}")
        if value.dtype != jnp.dtype(jnp.float32):
            raise TypeError(f"{name} has dtype {value.dtype}; expected float32")
    if mean.dtype != jnp.dtype(jnp.float32):
        raise TypeError(f"mean has dtype {mean.dtype}; expected float32")
    if not math.isfinite(noise_scale) or noise_scale < 0.0:
        raise ValueError(f"noise_scale must be finite and non-negative, got {noise_scale}")

    renew = jnp.logical_or(state.repeat_count == 0, state.repeat_count >= state.repeat_length)
    noise = jnp.where(renew, candidate_noise, state.noise)
    repeat_length = jnp.where(renew, candidate_repeat_length, state.repeat_length).astype(jnp.int32)
    repeat_count = jnp.where(renew, jnp.zeros_like(state.repeat_count), state.repeat_count) + 1
    raw_action = mean + standard_deviation * noise * noise_scale
    if action_transform == "tanh":
        action = jnp.tanh(raw_action) * action_scale
    elif action_transform == "identity":
        action = raw_action * action_scale
    else:
        raise ValueError(f"unsupported action transform {action_transform!r}")
    return action, ExplorationState(
        noise=noise,
        repeat_count=repeat_count,
        repeat_length=repeat_length,
    )


def collection_action(
    actor: Actor,
    variables,
    observation: jax.Array,
    state: ExplorationState,
    key: jax.Array,
    cdf: jax.Array,
    *,
    stochastic: bool,
    noise_scale: float = 1.0,
    action_transform: str = "tanh",
    action_scale: float = 1.0,
) -> tuple[jax.Array, ExplorationState, jax.Array]:
    """Run actor inference and the complete global repetition state transition."""
    distribution = actor.apply(variables, observation, training=False)
    if not stochastic:
        return (
            deterministic_action(
                distribution,
                action_transform=action_transform,
                action_scale=action_scale,
            ),
            state,
            key,
        )

    key, noise_key, repeat_key = jax.random.split(key, 3)
    candidate_noise = jax.random.normal(noise_key, distribution.mean.shape, dtype=jnp.float32)
    uniform = jax.random.uniform(repeat_key, (), dtype=jnp.float32)
    candidate_repeat_length = repeat_length_from_uniform(cdf, uniform)
    action, state = apply_repeated_noise(
        distribution.mean.astype(jnp.float32),
        distribution.std.astype(jnp.float32),
        state,
        candidate_noise,
        candidate_repeat_length,
        noise_scale=noise_scale,
        action_transform=action_transform,
        action_scale=action_scale,
    )
    return action, state, key
