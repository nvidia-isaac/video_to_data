# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashSAC actor, categorical double critic, and constrained layers."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import flax
from flax import linen as nn
from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp

from flash_chord.training.flash_sac.config import NetworkConfig


@flax.struct.dataclass
class ActorDistribution:
    """Parameters of the actor's unsquashed diagonal Gaussian."""

    mean: jax.Array
    log_std: jax.Array
    std: jax.Array


@flax.struct.dataclass
class PolicySample:
    """One reparameterized tanh-Gaussian sample."""

    action: jax.Array
    raw_action: jax.Array
    log_probability: jax.Array


@flax.struct.dataclass
class CategoricalCriticOutput:
    """Expected values and categorical distributions for every critic."""

    q: jax.Array
    logits: jax.Array
    log_probability: jax.Array


def _ensemble_orthogonal(key: jax.Array, shape: tuple[int, ...], dtype: Any) -> jax.Array:
    if len(shape) != 3:
        raise ValueError(f"ensemble orthogonal initialization expects rank three, got {shape}")
    initializer = nn.initializers.orthogonal()
    keys = jax.random.split(key, shape[0])
    return jax.vmap(lambda member_key: initializer(member_key, shape[1:], dtype))(keys)


class UnitLinear(nn.Module):
    """Bias-free low-precision matrix multiply with a float32 residual output."""

    features: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        kernel = self.param(
            "unit_kernel",
            nn.initializers.orthogonal(),
            (value.shape[-1], self.features),
            jnp.float32,
        )
        return jnp.matmul(value.astype(self.dtype), kernel.astype(self.dtype)).astype(jnp.float32)


class GainedUnitLinear(nn.Module):
    """Unit-direction projection with an independent trainable gain per output."""

    features: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        kernel = self.param(
            "unit_kernel",
            nn.initializers.orthogonal(),
            (value.shape[-1], self.features),
            jnp.float32,
        )
        gain = self.param("gain", nn.initializers.zeros_init(), (self.features,), jnp.float32)
        direction = jnp.matmul(value.astype(self.dtype), kernel.astype(self.dtype)).astype(jnp.float32)
        return direction * gain


class EnsembleUnitLinear(nn.Module):
    """Independent low-precision projections with float32 residual outputs."""

    ensemble_size: int
    features: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        if value.ndim != 3 or value.shape[0] != self.ensemble_size:
            raise ValueError(
                f"ensemble linear input must have shape ({self.ensemble_size}, batch, features), got {value.shape}"
            )
        kernel = self.param(
            "unit_kernel",
            _ensemble_orthogonal,
            (self.ensemble_size, value.shape[-1], self.features),
            jnp.float32,
        )
        return jnp.einsum("qbi,qio->qbo", value.astype(self.dtype), kernel.astype(self.dtype)).astype(jnp.float32)


class UnitBatchNorm(nn.Module):
    """Batch normalization whose affine pair is projected to norm ``sqrt(features)``."""

    update_rate: float = 0.01
    epsilon: float = 1.0e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array, *, training: bool) -> jax.Array:
        if value.ndim != 2:
            raise ValueError(f"actor batch normalization expects rank two, got {value.shape}")
        feature_count = value.shape[-1]
        scale = self.param("unit_scale", nn.initializers.ones_init(), (feature_count,), jnp.float32)
        bias = self.param("unit_bias", nn.initializers.zeros_init(), (feature_count,), jnp.float32)
        running_mean = self.variable(
            "batch_stats", "running_mean", lambda: jnp.zeros((feature_count,), dtype=jnp.float32)
        )
        running_variance = self.variable(
            "batch_stats", "running_variance", lambda: jnp.ones((feature_count,), dtype=jnp.float32)
        )
        value_f32 = value.astype(jnp.float32)
        if training:
            batch_size = value.shape[0]
            if batch_size <= 1:
                raise ValueError("training batch normalization requires at least two samples")
            mean = jnp.mean(value_f32, axis=0)
            variance = jnp.var(value_f32, axis=0)
            unbiased_variance = variance * (batch_size / (batch_size - 1))
            keep_rate = 1.0 - self.update_rate
            running_mean.value = keep_rate * running_mean.value + self.update_rate * jax.lax.stop_gradient(mean)
            running_variance.value = keep_rate * running_variance.value + self.update_rate * jax.lax.stop_gradient(
                unbiased_variance
            )
        else:
            mean = running_mean.value
            variance = running_variance.value
        normalized = (value_f32 - mean) * jax.lax.rsqrt(variance + self.epsilon)
        return normalized * scale + bias


class EnsembleUnitBatchNorm(nn.Module):
    """Per-critic batch normalization without cross-critic statistics."""

    ensemble_size: int
    update_rate: float = 0.01
    epsilon: float = 1.0e-5
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array, *, training: bool) -> jax.Array:
        if value.ndim != 3 or value.shape[0] != self.ensemble_size:
            raise ValueError(
                f"critic batch normalization expects ({self.ensemble_size}, batch, features), got {value.shape}"
            )
        feature_count = value.shape[-1]
        parameter_shape = (self.ensemble_size, feature_count)
        scale = self.param("unit_scale", nn.initializers.ones_init(), parameter_shape, jnp.float32)
        bias = self.param("unit_bias", nn.initializers.zeros_init(), parameter_shape, jnp.float32)
        running_mean = self.variable(
            "batch_stats", "running_mean", lambda: jnp.zeros(parameter_shape, dtype=jnp.float32)
        )
        running_variance = self.variable(
            "batch_stats", "running_variance", lambda: jnp.ones(parameter_shape, dtype=jnp.float32)
        )
        value_f32 = value.astype(jnp.float32)
        if training:
            batch_size = value.shape[1]
            if batch_size <= 1:
                raise ValueError("training batch normalization requires at least two samples")
            mean = jnp.mean(value_f32, axis=1)
            variance = jnp.var(value_f32, axis=1)
            unbiased_variance = variance * (batch_size / (batch_size - 1))
            keep_rate = 1.0 - self.update_rate
            running_mean.value = keep_rate * running_mean.value + self.update_rate * jax.lax.stop_gradient(mean)
            running_variance.value = keep_rate * running_variance.value + self.update_rate * jax.lax.stop_gradient(
                unbiased_variance
            )
        else:
            mean = running_mean.value
            variance = running_variance.value
        normalized = (value_f32 - mean[:, None, :]) * jax.lax.rsqrt(variance[:, None, :] + self.epsilon)
        return normalized * scale[:, None, :] + bias[:, None, :]


class UnitRMSNorm(nn.Module):
    """RMS normalization with a projected scale vector."""

    epsilon: float = 1.0e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        scale = self.param("rms_scale", nn.initializers.ones_init(), (value.shape[-1],), jnp.float32)
        value_f32 = value.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(jnp.square(value_f32), axis=-1, keepdims=True) + self.epsilon)
        return value_f32 / rms * scale


class EnsembleUnitRMSNorm(nn.Module):
    """Per-critic RMS normalization with projected scale vectors."""

    ensemble_size: int
    epsilon: float = 1.0e-6
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        if value.ndim != 3 or value.shape[0] != self.ensemble_size:
            raise ValueError(f"critic RMS normalization received invalid shape {value.shape}")
        scale = self.param(
            "rms_scale",
            nn.initializers.ones_init(),
            (self.ensemble_size, value.shape[-1]),
            jnp.float32,
        )
        value_f32 = value.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(jnp.square(value_f32), axis=-1, keepdims=True) + self.epsilon)
        return value_f32 / rms * scale[:, None, :]


class Block(nn.Module):
    """Actor inverted residual block."""

    hidden_dim: int
    expansion: int
    config: NetworkConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array, *, training: bool) -> jax.Array:
        residual = value
        value = UnitLinear(self.hidden_dim * self.expansion, self.dtype, name="expand")(value)
        value = UnitBatchNorm(
            self.config.batch_norm_update_rate,
            self.config.batch_norm_epsilon,
            self.dtype,
            name="expand_norm",
        )(value, training=training)
        value = nn.relu(value)
        value = UnitLinear(self.hidden_dim, self.dtype, name="contract")(value)
        value = UnitBatchNorm(
            self.config.batch_norm_update_rate,
            self.config.batch_norm_epsilon,
            self.dtype,
            name="contract_norm",
        )(value, training=training)
        return nn.relu(value) + residual


class EnsembleBlock(nn.Module):
    """Fused distributional-critic inverted residual block."""

    config: NetworkConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, value: jax.Array, *, training: bool) -> jax.Array:
        residual = value
        value = EnsembleUnitLinear(
            self.config.critic_count,
            self.config.critic_hidden_dim * self.config.expansion,
            self.dtype,
            name="expand",
        )(value)
        value = EnsembleUnitBatchNorm(
            self.config.critic_count,
            self.config.batch_norm_update_rate,
            self.config.batch_norm_epsilon,
            self.dtype,
            name="expand_norm",
        )(value, training=training)
        value = nn.relu(value)
        value = EnsembleUnitLinear(
            self.config.critic_count,
            self.config.critic_hidden_dim,
            self.dtype,
            name="contract",
        )(value)
        value = EnsembleUnitBatchNorm(
            self.config.critic_count,
            self.config.batch_norm_update_rate,
            self.config.batch_norm_epsilon,
            self.dtype,
            name="contract_norm",
        )(value, training=training)
        return nn.relu(value) + residual


class Actor(nn.Module):
    """Tanh-Gaussian FlashSAC actor."""

    action_dim: int
    config: NetworkConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, observation: jax.Array, *, training: bool) -> ActorDistribution:
        value = UnitBatchNorm(
            self.config.batch_norm_update_rate,
            self.config.batch_norm_epsilon,
            self.dtype,
            name="input_norm",
        )(observation, training=training)
        value = UnitLinear(self.config.actor_hidden_dim, self.dtype, name="embed")(value)
        for block_index in range(self.config.actor_blocks):
            value = Block(
                self.config.actor_hidden_dim,
                self.config.expansion,
                self.config,
                self.dtype,
                name=f"block_{block_index}",
            )(value, training=training)
        value = UnitRMSNorm(self.config.rms_norm_epsilon, self.dtype, name="post_norm")(value)
        if self.config.actor_head_mode == "upstream":
            mean = UnitLinear(self.action_dim, self.dtype, name="mean")(value)
            raw_log_std = UnitLinear(self.action_dim, self.dtype, name="log_std")(value)
            log_std_bias_initializer = nn.initializers.zeros_init()
            head_bias_dtype = self.dtype
        else:
            if self.config.residual_mean_head == "unit_gain":
                mean = GainedUnitLinear(self.action_dim, self.dtype, name="mean")(value)
            else:
                mean = nn.Dense(
                    self.action_dim,
                    use_bias=False,
                    dtype=self.dtype,
                    param_dtype=jnp.float32,
                    kernel_init=nn.initializers.zeros_init(),
                    name="mean",
                )(value).astype(jnp.float32)
            raw_log_std = GainedUnitLinear(self.action_dim, self.dtype, name="log_std")(value)
            initial_log_std = math.log(self.config.initial_normalized_std)
            normalized_log_std = (
                2.0 * (initial_log_std - self.config.log_std_min) / (self.config.log_std_max - self.config.log_std_min)
                - 1.0
            )
            log_std_bias_initializer = nn.initializers.constant(math.atanh(normalized_log_std))
            head_bias_dtype = jnp.float32
        mean_bias = self.param("mean_bias", nn.initializers.zeros_init(), (self.action_dim,), jnp.float32)
        log_std_bias = self.param(
            "log_std_bias",
            log_std_bias_initializer,
            (self.action_dim,),
            jnp.float32,
        )
        mean = mean + mean_bias.astype(head_bias_dtype)
        raw_log_std = raw_log_std + log_std_bias.astype(head_bias_dtype)
        log_std = self.config.log_std_min + (
            (self.config.log_std_max - self.config.log_std_min) * 0.5 * (1.0 + jnp.tanh(raw_log_std))
        )
        return ActorDistribution(mean=mean, log_std=log_std, std=jnp.exp(log_std))


class Critic(nn.Module):
    """Fused ensemble of categorical action-value critics."""

    value_min: float
    value_max: float
    config: NetworkConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(
        self,
        observation: jax.Array,
        action: jax.Array,
        *,
        training: bool,
    ) -> CategoricalCriticOutput:
        if observation.ndim != 2 or action.ndim != 2 or observation.shape[0] != action.shape[0]:
            raise ValueError(
                f"critic expects matching rank-two observation/action batches, got {observation.shape}/{action.shape}"
            )
        value = jnp.concatenate((observation, action), axis=-1)
        value = jnp.broadcast_to(value[None, ...], (self.config.critic_count, *value.shape))
        value = EnsembleUnitBatchNorm(
            self.config.critic_count,
            self.config.batch_norm_update_rate,
            self.config.batch_norm_epsilon,
            self.dtype,
            name="input_norm",
        )(value, training=training)
        value = EnsembleUnitLinear(
            self.config.critic_count,
            self.config.critic_hidden_dim,
            self.dtype,
            name="embed",
        )(value)
        for block_index in range(self.config.critic_blocks):
            value = EnsembleBlock(self.config, self.dtype, name=f"block_{block_index}")(value, training=training)
        value = EnsembleUnitRMSNorm(
            self.config.critic_count,
            self.config.rms_norm_epsilon,
            self.dtype,
            name="post_norm",
        )(value)
        logits = EnsembleUnitLinear(
            self.config.critic_count,
            self.config.atom_count,
            self.dtype,
            name="categorical",
        )(value)
        bias = self.param(
            "categorical_bias",
            nn.initializers.zeros_init(),
            (self.config.critic_count, self.config.atom_count),
            jnp.float32,
        )
        logits = logits + bias[:, None, :]
        log_probability = jax.nn.log_softmax(logits, axis=-1)
        atom_width = (self.value_max - self.value_min) / (self.config.atom_count - 1)
        support = self.value_min + atom_width * jnp.arange(self.config.atom_count, dtype=jnp.float32)
        q = jnp.sum(jnp.exp(log_probability) * support, axis=-1)
        return CategoricalCriticOutput(q=q, logits=logits, log_probability=log_probability)


def safe_tanh_log_det_jacobian(raw_action: jax.Array) -> jax.Array:
    """Stable elementwise log absolute determinant of the tanh Jacobian."""
    return 2.0 * (math.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action))


def squashed_gaussian_from_noise(distribution: ActorDistribution, noise: jax.Array) -> PolicySample:
    """Transform an explicit standard-normal sample through the actor distribution."""
    return policy_sample_from_noise(distribution, noise, action_transform="tanh", action_scale=1.0)


def policy_sample_from_noise(
    distribution: ActorDistribution,
    noise: jax.Array,
    *,
    action_transform: str,
    action_scale: float,
) -> PolicySample:
    """Transform an explicit Gaussian sample into the environment's action coordinates."""
    if noise.shape != distribution.mean.shape:
        raise ValueError(f"noise has shape {noise.shape}; expected {distribution.mean.shape}")
    if action_transform not in {"tanh", "identity"}:
        raise ValueError(f"unsupported action transform {action_transform!r}")
    if not math.isfinite(action_scale) or action_scale <= 0.0:
        raise ValueError(f"action_scale must be positive and finite, got {action_scale}")
    raw_action = distribution.mean + distribution.std * noise
    normal_log_probability = -0.5 * (jnp.square(noise) + 2.0 * distribution.log_std + math.log(2.0 * math.pi))
    if action_transform == "tanh":
        action = jnp.tanh(raw_action) * action_scale
        log_jacobian = safe_tanh_log_det_jacobian(raw_action) + math.log(action_scale)
    else:
        action = raw_action * action_scale
        log_jacobian = jnp.full_like(raw_action, math.log(action_scale))
    log_probability = jnp.sum(normal_log_probability - log_jacobian, axis=-1)
    return PolicySample(action=action, raw_action=raw_action, log_probability=log_probability)


def sample_squashed_gaussian(distribution: ActorDistribution, key: jax.Array) -> PolicySample:
    """Draw one reparameterized action from the actor distribution."""
    return sample_policy(distribution, key, action_transform="tanh", action_scale=1.0)


def sample_policy(
    distribution: ActorDistribution,
    key: jax.Array,
    *,
    action_transform: str,
    action_scale: float,
) -> PolicySample:
    """Draw one reparameterized action using the configured environment transform."""
    noise = jax.random.normal(key, distribution.mean.shape, dtype=distribution.mean.dtype)
    return policy_sample_from_noise(
        distribution,
        noise,
        action_transform=action_transform,
        action_scale=action_scale,
    )


def deterministic_action(
    distribution: ActorDistribution,
    *,
    action_transform: str = "tanh",
    action_scale: float = 1.0,
) -> jax.Array:
    """Return the transformed actor mean in environment action coordinates."""
    if action_transform == "tanh":
        return jnp.tanh(distribution.mean) * action_scale
    if action_transform == "identity":
        return distribution.mean * action_scale
    raise ValueError(f"unsupported action transform {action_transform!r}")


def project_unit_parameters(parameters, epsilon: float = 1.0e-8):
    """Project all unit-linear and normalization parameters onto their constraints."""
    was_frozen = isinstance(parameters, FrozenDict)
    mutable = unfreeze(parameters) if was_frozen else deepcopy(parameters)

    def project(module: dict[str, Any]) -> None:
        if "unit_kernel" in module:
            kernel = module["unit_kernel"]
            norm = jnp.linalg.norm(kernel, axis=-2, keepdims=True)
            module["unit_kernel"] = kernel / jnp.maximum(norm, epsilon)
        if "unit_scale" in module and "unit_bias" in module:
            scale = module["unit_scale"]
            bias = module["unit_bias"]
            feature_count = scale.shape[-1]
            squared_norm = jnp.sum(jnp.square(scale) + jnp.square(bias), axis=-1, keepdims=True)
            factor = math.sqrt(feature_count) * jax.lax.rsqrt(squared_norm + epsilon)
            module["unit_scale"] = scale * factor
            module["unit_bias"] = bias * factor
        if "rms_scale" in module:
            scale = module["rms_scale"]
            feature_count = scale.shape[-1]
            squared_norm = jnp.sum(jnp.square(scale), axis=-1, keepdims=True)
            module["rms_scale"] = scale * math.sqrt(feature_count) * jax.lax.rsqrt(squared_norm + epsilon)
        for value in module.values():
            if isinstance(value, dict):
                project(value)

    project(mutable)
    return freeze(mutable) if was_frozen else mutable
