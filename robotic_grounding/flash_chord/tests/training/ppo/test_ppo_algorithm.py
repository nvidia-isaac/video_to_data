# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical and integration tests for the PPO update."""

import numpy as np
import pytest


def _training_dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    return jax, jnp


def test_compute_gae_respects_episode_boundaries():
    _, jnp = _training_dependencies()

    from flash_chord.training.ppo.algorithm import compute_gae

    advantage, target = compute_gae(
        reward=jnp.asarray([[1.0], [2.0], [4.0]]),
        value=jnp.zeros((3, 1)),
        done=jnp.asarray([[0.0], [1.0], [0.0]]),
        last_value=jnp.zeros((1,)),
        discount=1.0,
        gae_lambda=1.0,
    )

    expected = np.asarray([[3.0], [2.0], [4.0]])
    np.testing.assert_allclose(np.asarray(advantage), expected)
    np.testing.assert_allclose(np.asarray(target), expected)


def test_gaussian_kl_matches_closed_form():
    _, jnp = _training_dependencies()

    from flash_chord.training.ppo.algorithm import gaussian_kl

    unit = jnp.ones((3, 2))
    zero = jnp.zeros((3, 2))
    np.testing.assert_allclose(np.asarray(gaussian_kl(zero, unit, zero, unit)), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(gaussian_kl(zero, unit, unit, unit)), 1.0, atol=1.0e-7)


def test_explained_variance_has_standard_rollout_semantics():
    jnp = pytest.importorskip("jax.numpy")

    from flash_chord.training.ppo.algorithm import explained_variance

    target = jnp.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(explained_variance(target, target)), 1.0)
    np.testing.assert_allclose(np.asarray(explained_variance(target, jnp.zeros_like(target))), 0.0)
    np.testing.assert_allclose(np.asarray(explained_variance(target, -target)), -3.0)
    np.testing.assert_allclose(np.asarray(explained_variance(jnp.ones(3), jnp.ones(3))), 0.0)


def test_adaptive_learning_rate_matches_rsl_rule_and_bounds():
    _, jnp = _training_dependencies()

    from flash_chord.training.ppo.config import PPOConfig
    from flash_chord.training.ppo.algorithm import adapt_learning_rate

    config = PPOConfig()
    rate = jnp.asarray(config.learning_rate)
    decreased = adapt_learning_rate(rate, jnp.asarray(0.02), config)
    increased = adapt_learning_rate(rate, jnp.asarray(0.001), config)
    unchanged = adapt_learning_rate(rate, jnp.asarray(0.005), config)

    np.testing.assert_allclose(np.asarray(decreased), 1.0e-3 / 1.5)
    np.testing.assert_allclose(np.asarray(increased), 1.5e-3)
    np.testing.assert_allclose(np.asarray(unchanged), 1.0e-3)
    np.testing.assert_allclose(
        np.asarray(adapt_learning_rate(jnp.asarray(1.0e-5), jnp.asarray(1.0), config)),
        config.minimum_learning_rate,
    )
    np.testing.assert_allclose(
        np.asarray(adapt_learning_rate(jnp.asarray(1.0e-2), jnp.asarray(1.0e-4), config)),
        config.maximum_learning_rate,
    )

    fixed = PPOConfig(schedule="fixed")
    np.testing.assert_allclose(np.asarray(adapt_learning_rate(rate, jnp.asarray(1.0), fixed)), rate)


def test_probability_ratio_is_finite_for_extreme_log_probability_changes():
    jax, jnp = _training_dependencies()

    from flash_chord.training.ppo.algorithm import probability_ratio

    new_log_probability = jnp.asarray([-1.0e3, 0.0, 1.0e3])
    old_log_probability = jnp.zeros_like(new_log_probability)
    ratio = probability_ratio(new_log_probability, old_log_probability, log_ratio_clip=20.0)
    gradient = jax.grad(lambda value: probability_ratio(value, old_log_probability, log_ratio_clip=20.0).sum())(
        new_log_probability
    )

    assert np.isfinite(np.asarray(ratio)).all()
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(float(ratio[-1]), np.exp(20.0), rtol=1.0e-6)


def test_probability_ratio_is_unbounded_exponential_by_default():
    _, jnp = _training_dependencies()

    from flash_chord.training.ppo.algorithm import probability_ratio

    new_log_probability = jnp.asarray([-2.0, 0.0, 2.0])
    old_log_probability = jnp.asarray([0.5, 0.0, -0.5])
    np.testing.assert_allclose(
        np.asarray(probability_ratio(new_log_probability, old_log_probability)),
        np.exp(np.asarray(new_log_probability - old_log_probability)),
    )


def test_update_changes_parameters_and_reports_finite_metrics():
    jax, jnp = _training_dependencies()

    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig
    from flash_chord.training.ppo.network import ActorCritic
    from flash_chord.training.ppo.algorithm import RolloutBatch, create_train_state, update

    network_config = NetworkConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,))
    ppo_config = PPOConfig(learning_epochs=2, mini_batches=2, schedule="fixed")
    model = ActorCritic(action_dim=2, config=network_config)
    key = jax.random.key(42)
    observation = jnp.arange(24, dtype=jnp.float32).reshape(4, 2, 3) / 24.0
    parameters = model.init(key, observation.reshape(8, 3))
    state = create_train_state(model, parameters, ppo_config)

    behavior_policy, behavior_value = model.apply(parameters, observation.reshape(8, 3))
    action = behavior_policy.mean().reshape(4, 2, 2)
    batch = RolloutBatch(
        observation=observation,
        action=action,
        log_probability=behavior_policy.log_prob(action.reshape(8, 2)).reshape(4, 2),
        value=behavior_value.reshape(4, 2),
        reward=jnp.asarray([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0], [0.0, 0.5]]),
        done=jnp.zeros((4, 2)),
        action_mean=behavior_policy.mean().reshape(4, 2, 2),
        action_std=behavior_policy.stddev().reshape(4, 2, 2),
    )
    state, learning_rate, metrics = update(
        state,
        batch,
        last_value=jnp.zeros((2,)),
        key=jax.random.key(7),
        learning_rate=jnp.asarray(ppo_config.learning_rate),
        config=ppo_config,
    )

    assert int(state.step) == ppo_config.learning_epochs * ppo_config.mini_batches
    np.testing.assert_allclose(np.asarray(learning_rate), ppo_config.learning_rate)
    assert 0.0 <= float(metrics.clip_fraction) <= 1.0
    assert float(metrics.skipped_update_fraction) == 0.0
    metric_values = jax.tree.leaves(metrics)
    assert all(np.isfinite(np.asarray(value)).all() for value in metric_values)
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(jax.tree.leaves(parameters), jax.tree.leaves(state.params), strict=True)
    )


def test_update_rejects_nonfinite_optimizer_updates_without_poisoning_state():
    jax, jnp = _training_dependencies()

    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig
    from flash_chord.training.ppo.network import ActorCritic
    from flash_chord.training.ppo.algorithm import RolloutBatch, create_train_state, update

    network_config = NetworkConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,))
    ppo_config = PPOConfig(learning_epochs=1, mini_batches=1, schedule="fixed")
    model = ActorCritic(action_dim=2, config=network_config)
    observation = jnp.zeros((2, 1, 3), dtype=jnp.float32)
    parameters = model.init(jax.random.key(42), observation.reshape(2, 3))
    state = create_train_state(model, parameters, ppo_config)
    behavior_policy, behavior_value = model.apply(parameters, observation.reshape(2, 3))
    action = behavior_policy.mean().reshape(2, 1, 2)
    batch = RolloutBatch(
        observation=observation,
        action=action,
        log_probability=behavior_policy.log_prob(action.reshape(2, 2)).reshape(2, 1),
        value=behavior_value.reshape(2, 1),
        reward=jnp.full((2, 1), jnp.nan),
        done=jnp.zeros((2, 1)),
        action_mean=behavior_policy.mean().reshape(2, 1, 2),
        action_std=behavior_policy.stddev().reshape(2, 1, 2),
    )

    updated, learning_rate, metrics = update(
        state,
        batch,
        last_value=jnp.zeros((1,)),
        key=jax.random.key(7),
        learning_rate=jnp.asarray(ppo_config.learning_rate),
        config=ppo_config,
    )

    assert int(updated.step) == 0
    assert float(metrics.skipped_update_fraction) == 1.0
    np.testing.assert_allclose(np.asarray(learning_rate), ppo_config.learning_rate)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(updated), strict=True):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
