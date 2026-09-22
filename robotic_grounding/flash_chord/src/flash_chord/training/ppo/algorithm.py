# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JIT-compiled PPO math for Warp-backed rollouts."""

from __future__ import annotations

from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from flash_chord.training.ppo.config import PPOConfig


@flax.struct.dataclass
class RolloutBatch:
    """Time-major on-policy data plus the behavior Gaussian parameters."""

    observation: jax.Array
    action: jax.Array
    log_probability: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    action_mean: jax.Array
    action_std: jax.Array


@flax.struct.dataclass
class PPOMetrics:
    """Mean optimization metrics over all epochs and mini-batches."""

    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    kl: jax.Array
    clip_fraction: jax.Array
    learning_rate: jax.Array
    explained_variance: jax.Array
    skipped_update_fraction: jax.Array


def create_train_state(model, parameters, config: PPOConfig) -> TrainState:
    """Create Adam state with clipping but a separately controlled live learning rate."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.scale_by_adam(),
    )
    state = TrainState.create(apply_fn=model.apply, params=parameters, tx=optimizer)
    return state.replace(step=jnp.asarray(0, dtype=jnp.int32))


def compute_gae(
    reward: jax.Array,
    value: jax.Array,
    done: jax.Array,
    last_value: jax.Array,
    discount: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute time-major generalized advantages and value targets."""

    def step(carry, transition):
        advantage, next_value = carry
        transition_reward, transition_value, transition_done = transition
        not_done = 1.0 - transition_done
        delta = transition_reward + discount * next_value * not_done - transition_value
        advantage = delta + discount * gae_lambda * not_done * advantage
        return (advantage, transition_value), advantage

    initial = (jnp.zeros_like(last_value), last_value)
    _, advantage = jax.lax.scan(step, initial, (reward, value, done), reverse=True)
    return advantage, advantage + value


def explained_variance(value_target: jax.Array, value_prediction: jax.Array) -> jax.Array:
    """Fraction of rollout target variance explained by pre-update value predictions."""
    target_variance = jnp.var(value_target)
    return jnp.where(
        target_variance > 1.0e-8,
        1.0 - jnp.var(value_target - value_prediction) / target_variance,
        0.0,
    )


def gaussian_kl(
    old_mean: jax.Array,
    old_std: jax.Array,
    new_mean: jax.Array,
    new_std: jax.Array,
) -> jax.Array:
    """Mean KL(old || new) for diagonal Gaussian policies."""
    old_std = jnp.maximum(old_std, 1.0e-8)
    new_std = jnp.maximum(new_std, 1.0e-8)
    per_dimension = (
        jnp.log(new_std / old_std)
        + (jnp.square(old_std) + jnp.square(old_mean - new_mean)) / (2.0 * jnp.square(new_std))
        - 0.5
    )
    return jnp.sum(per_dimension, axis=-1).mean()


def adapt_learning_rate(learning_rate: jax.Array, kl: jax.Array, config: PPOConfig) -> jax.Array:
    """Apply the RSL-RL adaptive-KL learning-rate rule."""
    if config.schedule == "fixed":
        return learning_rate
    lower = config.desired_kl * 0.5
    upper = config.desired_kl * 2.0
    decreased = jnp.maximum(learning_rate / config.adaptive_schedule_factor, config.minimum_learning_rate)
    increased = jnp.minimum(learning_rate * config.adaptive_schedule_factor, config.maximum_learning_rate)
    return jnp.where(kl > upper, decreased, jnp.where((kl > 0.0) & (kl < lower), increased, learning_rate))


def probability_ratio(
    new_log_probability: jax.Array,
    old_log_probability: jax.Array,
    log_ratio_clip: float | None = None,
) -> jax.Array:
    """Exponentiate the PPO log ratio, with an optional symmetric numerical bound."""
    log_ratio = new_log_probability - old_log_probability
    if log_ratio_clip is not None:
        log_ratio = jnp.clip(log_ratio, -log_ratio_clip, log_ratio_clip)
    return jnp.exp(log_ratio)


def _tree_all_finite(tree) -> jax.Array:
    """Return one device scalar describing every leaf without a host synchronization."""
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree)]))


@partial(jax.jit, static_argnames=("config",))
def update(
    state: TrainState,
    batch: RolloutBatch,
    last_value: jax.Array,
    key: jax.Array,
    learning_rate: jax.Array,
    config: PPOConfig,
) -> tuple[TrainState, jax.Array, PPOMetrics]:
    """Run all configured PPO epochs and mini-batches."""
    advantage, value_target = compute_gae(
        batch.reward,
        batch.value,
        batch.done,
        last_value,
        config.discount,
        config.gae_lambda,
    )
    rollout_explained_variance = explained_variance(value_target, batch.value)
    if not config.normalize_advantage_per_mini_batch:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1.0e-8)

    sample_count = batch.reward.shape[0] * batch.reward.shape[1]
    data = (
        batch.observation.reshape(sample_count, -1),
        batch.action.reshape(sample_count, -1),
        batch.log_probability.reshape(sample_count),
        batch.value.reshape(sample_count),
        value_target.reshape(sample_count),
        advantage.reshape(sample_count),
        batch.action_mean.reshape(sample_count, -1),
        batch.action_std.reshape(sample_count, -1),
    )

    def mini_batch(carry, mini_batch_data):
        current_state, current_learning_rate = carry
        observation, action, old_log_probability, old_value, target, mini_advantage, old_mean, old_std = mini_batch_data
        if config.normalize_advantage_per_mini_batch:
            mini_advantage = (mini_advantage - mini_advantage.mean()) / (mini_advantage.std() + 1.0e-8)

        def loss(parameters):
            policy, value = current_state.apply_fn(parameters, observation)
            new_log_probability = policy.log_prob(action)
            ratio = probability_ratio(
                new_log_probability,
                old_log_probability,
                config.log_ratio_clip,
            )
            unclipped = ratio * mini_advantage
            clipped = jnp.clip(ratio, 1.0 - config.clip_parameter, 1.0 + config.clip_parameter) * mini_advantage
            policy_loss = -jnp.minimum(unclipped, clipped).mean()
            clip_fraction = (jnp.abs(ratio - 1.0) > config.clip_parameter).mean()

            if config.use_clipped_value_loss:
                clipped_value = old_value + jnp.clip(
                    value - old_value,
                    -config.clip_parameter,
                    config.clip_parameter,
                )
                value_loss = jnp.maximum(jnp.square(value - target), jnp.square(clipped_value - target)).mean()
            else:
                value_loss = jnp.square(value - target).mean()
            entropy = policy.entropy().mean()
            kl = gaussian_kl(old_mean, old_std, policy.mean(), policy.stddev())
            total = policy_loss + config.value_loss_coefficient * value_loss - config.entropy_coefficient * entropy
            return total, (policy_loss, value_loss, entropy, kl, clip_fraction)

        (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(current_state.params)
        policy_loss, value_loss, entropy, kl, clip_fraction = metrics
        candidate_learning_rate = adapt_learning_rate(current_learning_rate, kl, config)
        updates, optimizer_state = current_state.tx.update(
            gradients,
            current_state.opt_state,
            current_state.params,
        )
        update_is_finite = _tree_all_finite(updates)
        updates = jax.tree.map(lambda value: -candidate_learning_rate * value, updates)
        parameters = optax.apply_updates(current_state.params, updates)
        candidate_state = current_state.replace(
            step=current_state.step + 1,
            params=parameters,
            opt_state=optimizer_state,
        )
        current_state = jax.lax.cond(
            update_is_finite,
            lambda: candidate_state,
            lambda: current_state,
        )
        current_learning_rate = jnp.where(update_is_finite, candidate_learning_rate, current_learning_rate)
        output = PPOMetrics(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            kl=kl,
            clip_fraction=clip_fraction,
            learning_rate=current_learning_rate,
            explained_variance=rollout_explained_variance,
            skipped_update_fraction=1.0 - update_is_finite.astype(jnp.float32),
        )
        return (current_state, current_learning_rate), output

    def epoch(carry, _):
        current_state, current_learning_rate, current_key = carry
        current_key, permutation_key = jax.random.split(current_key)
        permutation = jax.random.permutation(permutation_key, sample_count)
        shuffled = jax.tree.map(lambda value: value[permutation], data)
        mini_batches = jax.tree.map(
            lambda value: value.reshape((config.mini_batches, -1) + value.shape[1:]),
            shuffled,
        )
        (current_state, current_learning_rate), metrics = jax.lax.scan(
            mini_batch,
            (current_state, current_learning_rate),
            mini_batches,
        )
        return (current_state, current_learning_rate, current_key), metrics

    (state, learning_rate, key), metrics = jax.lax.scan(
        epoch,
        (state, learning_rate, key),
        None,
        length=config.learning_epochs,
    )
    metrics = jax.tree.map(lambda value: value.mean(), metrics)
    metrics = metrics.replace(learning_rate=learning_rate)
    return state, learning_rate, metrics
