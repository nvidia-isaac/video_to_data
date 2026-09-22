# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the JAX actor-critic model."""

import numpy as np
import pytest


def test_actor_critic_shapes_initial_std_and_jit_sampling():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("distrax")

    from flash_chord.training.ppo.config import NetworkConfig
    from flash_chord.training.ppo.network import ActorCritic

    config = NetworkConfig(actor_hidden_dims=(8, 4), critic_hidden_dims=(6, 3))
    model = ActorCritic(action_dim=2, config=config)
    key = jax.random.key(42)
    observation = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)
    parameters = model.init(key, observation)
    policy, value = model.apply(parameters, observation)

    assert policy.mean().shape == (5, 2)
    assert policy.stddev().shape == (5, 2)
    assert value.shape == (5,)
    np.testing.assert_array_equal(np.asarray(policy.mean()), 0.0)
    np.testing.assert_allclose(np.asarray(policy.stddev()), 0.1, atol=1e-7)

    nonzero_head = ActorCritic(
        action_dim=2,
        config=NetworkConfig(
            actor_hidden_dims=(8, 4),
            critic_hidden_dims=(6, 3),
            actor_output_gain=0.01,
        ),
    )
    nonzero_policy, _ = nonzero_head.apply(nonzero_head.init(key, observation), observation)
    assert np.any(np.asarray(nonzero_policy.mean()) != 0.0)

    @jax.jit
    def sample(params, obs, sample_key):
        distribution, critic_value = model.apply(params, obs)
        action = distribution.sample(seed=sample_key)
        return action, distribution.log_prob(action), critic_value

    action, log_probability, sampled_value = sample(parameters, observation, key)
    assert action.shape == (5, 2)
    assert log_probability.shape == (5,)
    assert sampled_value.shape == (5,)
    assert np.isfinite(np.asarray(action)).all()
    assert np.isfinite(np.asarray(log_probability)).all()
