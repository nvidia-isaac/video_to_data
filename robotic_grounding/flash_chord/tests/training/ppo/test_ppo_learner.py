# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for one learner iteration."""

import numpy as np
import pytest


class _VectorEnv:
    world_count = 2
    observation_dim = 3
    action_dim = 2

    def __init__(self, jnp):
        self.jnp = jnp
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return self.jnp.zeros((self.world_count, self.observation_dim))

    def step(self, action):
        from flash_chord.training.environment import VectorStep

        self.step_count += 1
        observation = self.jnp.full((self.world_count, self.observation_dim), 0.1 * self.step_count)
        return VectorStep(
            observation=observation,
            reward=1.0 - self.jnp.square(action).mean(axis=-1),
            terminated=self.jnp.zeros(self.world_count, dtype=self.jnp.int32),
            truncated=self.jnp.zeros(self.world_count, dtype=self.jnp.int32),
            episode_return=self.jnp.zeros(self.world_count),
            episode_length=self.jnp.zeros(self.world_count, dtype=self.jnp.int32),
            episode_reference_progress=self.jnp.zeros(self.world_count),
        )


def _config(world_count=2):
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig

    return TrainingConfig(
        world_count=world_count,
        rollout_steps=2,
        network=NetworkConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        ppo=PPOConfig(learning_epochs=1, mini_batches=2),
    )


def test_two_training_iterations_advance_complete_learner_state():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")

    from flash_chord.training.ppo.learner import create_learner, train_iteration

    env = _VectorEnv(jnp)
    config = _config()
    learner = create_learner(env, config)
    initial_parameters = learner.train_state.params
    learner, first_metrics = train_iteration(env, learner, config)
    learner, second_metrics = train_iteration(env, learner, config)

    assert int(learner.iteration) == 2
    assert int(learner.train_state.step) == 2 * config.ppo.learning_epochs * config.ppo.mini_batches
    assert int(learner.normalization.count) == 2 * config.rollout_steps * config.world_count
    assert config.ppo.minimum_learning_rate <= float(learner.learning_rate) <= config.ppo.maximum_learning_rate
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree.leaves(initial_parameters),
            jax.tree.leaves(learner.train_state.params),
            strict=True,
        )
    )
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(first_metrics))
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(second_metrics))


def test_create_learner_rejects_world_count_mismatch():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")

    from flash_chord.training.ppo.learner import create_learner

    with pytest.raises(ValueError, match="environment has 2 worlds"):
        create_learner(_VectorEnv(jnp), _config(world_count=4))
