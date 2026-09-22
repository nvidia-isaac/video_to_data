# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for checkpoint-policy inference behavior."""

import numpy as np
import pytest


class _VectorEnv:
    world_count = 2
    observation_dim = 3
    action_dim = 2

    def __init__(self, jnp):
        self.jnp = jnp

    def reset(self):
        return self.jnp.zeros((self.world_count, self.observation_dim))


def test_policy_action_is_deterministic_by_default_and_stochastic_on_request():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")

    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig
    from flash_chord.training.ppo.evaluation import EvaluationConfig, policy_action
    from flash_chord.training.ppo.learner import create_learner

    config = TrainingConfig(
        world_count=2,
        rollout_steps=2,
        network=NetworkConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        ppo=PPOConfig(learning_epochs=1, mini_batches=2),
    )
    learner = create_learner(_VectorEnv(jnp), config)
    deterministic = policy_action(
        learner,
        learner.observation,
        config,
        EvaluationConfig(checkpoint="model.safetensors"),
    )
    stochastic_config = EvaluationConfig(checkpoint="model.safetensors", deterministic=False)
    stochastic = policy_action(
        learner,
        learner.observation,
        config,
        stochastic_config,
        key=jax.random.key(1),
    )

    assert deterministic.shape == stochastic.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(deterministic), 0.0)
    assert np.isfinite(np.asarray(stochastic)).all()
    with pytest.raises(ValueError, match="PRNG key"):
        policy_action(learner, learner.observation, config, stochastic_config)


def test_evaluation_config_validates_checkpoint():
    from flash_chord.training.ppo.evaluation import EvaluationConfig

    with pytest.raises(ValueError, match="checkpoint"):
        EvaluationConfig(checkpoint="")

    config = EvaluationConfig(checkpoint="model.safetensors")
    assert config.use_checkpoint_config is True
    assert config.validate_policy_schema is True
