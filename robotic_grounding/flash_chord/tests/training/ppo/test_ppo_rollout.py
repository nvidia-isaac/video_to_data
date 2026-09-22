# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for JAX rollout collection."""

import numpy as np
import pytest


class _DeterministicEnv:
    world_count = 2
    observation_dim = 3
    action_dim = 2
    objective_term_names = ("term_a", "term_b")
    termination_cause_names = ("wrist", "object")
    tracking_error_names = ("position_m", "orientation_rad")

    def __init__(self, jnp):
        self.jnp = jnp
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return self.jnp.zeros((self.world_count, self.observation_dim))

    def step(self, action):
        from flash_chord.training.environment import VectorStep

        assert action.shape == (self.world_count, self.action_dim)
        self.step_count += 1
        terminated = self.jnp.asarray([self.step_count == 2, False], dtype=self.jnp.int32)
        truncated = self.jnp.asarray([self.step_count == 2, self.step_count == 2], dtype=self.jnp.int32)
        episode_length = self.jnp.asarray([2, 2], dtype=self.jnp.int32) if self.step_count == 2 else self.jnp.zeros(2)
        episode_return = self.jnp.asarray([3.0, 4.0]) if self.step_count == 2 else self.jnp.zeros(2)
        episode_reference_progress = self.jnp.asarray([0.25, 0.75]) if self.step_count == 2 else self.jnp.zeros(2)
        step = float(self.step_count)
        return VectorStep(
            observation=self.jnp.full((self.world_count, self.observation_dim), float(self.step_count)),
            reward=self.jnp.asarray([float(self.step_count), float(self.step_count + 1)]),
            terminated=terminated,
            truncated=truncated,
            episode_return=episode_return,
            episode_length=episode_length,
            episode_reference_progress=episode_reference_progress,
            objective_terms=self.jnp.asarray([[step, 10.0 * step], [step + 1.0, 10.0 * step + 1.0]]),
            termination_causes=self.jnp.asarray(
                [[self.step_count == 2, self.step_count == 2], [False, self.step_count == 3]],
                dtype=self.jnp.int32,
            ),
            tracking_errors=self.jnp.asarray(
                [[step, 0.1 * step], [step + 0.5, 0.1 * step + 0.05]],
            ),
            terminal_observation=self.jnp.full((self.world_count, self.observation_dim), 999.0),
            critic_context=self.jnp.full((self.world_count, 3), 888.0),
            terminal_critic_context=self.jnp.full((self.world_count, 3), 777.0),
        )


def _learner(jax, jnp, rollout_steps, observation_normalization):
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")

    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig
    from flash_chord.training.ppo.network import ActorCritic
    from flash_chord.training.ppo.normalization import create_normalization_state
    from flash_chord.training.ppo.algorithm import create_train_state

    env = _DeterministicEnv(jnp)
    config = TrainingConfig(
        world_count=env.world_count,
        rollout_steps=rollout_steps,
        observation_normalization=observation_normalization,
        network=NetworkConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        ppo=PPOConfig(learning_epochs=1, mini_batches=2, schedule="fixed"),
    )
    model = ActorCritic(action_dim=env.action_dim, config=config.network)
    parameters = model.init(jax.random.key(42), jnp.zeros((1, env.observation_dim)))
    state = create_train_state(model, parameters, config.ppo)
    normalization = create_normalization_state(env.observation_dim)
    return env, config, state, normalization


def test_rollout_shapes_bootstraps_truncation_and_aggregates_metrics():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from flash_chord.training.ppo.rollout import collect_rollout

    env, config, state, normalization = _learner(jax, jnp, rollout_steps=3, observation_normalization=False)
    batch, observation, normalization, _, metrics = collect_rollout(
        env,
        state,
        normalization,
        env.reset(),
        jax.random.key(7),
        config,
    )

    assert batch.observation.shape == (3, 2, 3)
    assert batch.action.shape == (3, 2, 2)
    assert batch.reward.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(batch.observation[:, 0, 0]), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(observation), 3.0)
    np.testing.assert_allclose(np.asarray(batch.done[1]), [1.0, 1.0])
    np.testing.assert_allclose(np.asarray(batch.reward[1, 0]), 2.0)
    np.testing.assert_allclose(
        np.asarray(batch.reward[1, 1]),
        3.0 + config.ppo.discount * np.asarray(batch.value[1, 1]),
    )
    assert int(normalization.count) == 0
    np.testing.assert_allclose(np.asarray(metrics.mean_step_reward), 2.5)
    assert int(metrics.reference_end_count) == 1
    np.testing.assert_allclose(np.asarray(metrics.episode_return_sum), 7.0)
    assert int(metrics.episode_length_sum) == 4
    np.testing.assert_allclose(np.asarray(metrics.episode_reference_progress_sum), 1.0)
    assert int(metrics.episode_count) == 2
    np.testing.assert_allclose(np.asarray(metrics.action_mean_abs_mean), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(metrics.action_std_mean), 0.1, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(metrics.objective_term_mean), [2.5, 20.5])
    np.testing.assert_array_equal(np.asarray(metrics.termination_cause_count), [1, 0])
    np.testing.assert_allclose(np.asarray(metrics.tracking_error_mean), [2.25, 0.225])


def test_rollout_updates_normalization_from_each_next_observation():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from flash_chord.training.ppo.rollout import collect_rollout

    env, config, state, normalization = _learner(jax, jnp, rollout_steps=2, observation_normalization=True)
    batch, _, normalization, _, _ = collect_rollout(
        env,
        state,
        normalization,
        env.reset(),
        jax.random.key(7),
        config,
    )

    assert int(normalization.count) == 4
    np.testing.assert_allclose(np.asarray(normalization.mean), 1.5)
    np.testing.assert_allclose(np.asarray(normalization.variance), 0.25)
    np.testing.assert_allclose(np.asarray(batch.observation), 0.0, atol=1.0e-7)
