# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for FlashSAC global zeta-repeated exploration."""

import numpy as np
import pytest


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    return jax, jnp


def _actor(jax, jnp, world_count=2):
    from flash_chord.training.flash_sac.config import NetworkConfig
    from flash_chord.training.flash_sac.network import Actor

    actor = Actor(
        action_dim=2,
        config=NetworkConfig(
            actor_blocks=1,
            actor_hidden_dim=4,
            critic_blocks=1,
            critic_hidden_dim=4,
            atom_count=5,
        ),
    )
    observation = jnp.zeros((world_count, 5), dtype=jnp.float32)
    variables = actor.init(jax.random.key(0), observation, training=False)
    return actor, variables, observation


def test_truncated_zeta_cdf_matches_upstream_distribution():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.exploration import truncated_zeta_cdf

    cdf = truncated_zeta_cdf(2.0, 16)
    lengths = np.arange(1, 17, dtype=np.float32)
    probability = lengths**-2
    probability /= probability.sum()
    np.testing.assert_allclose(np.asarray(cdf), np.cumsum(probability), rtol=1.0e-6)
    assert float(cdf[-1]) == pytest.approx(1.0)
    assert float(jnp.sum(jnp.asarray(probability) * jnp.asarray(lengths))) == pytest.approx(2.133831786, rel=1.0e-6)


def test_repeat_length_uses_strict_first_true_and_upstream_all_false_fallback():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.exploration import repeat_length_from_uniform

    sample = jax.jit(repeat_length_from_uniform)
    cdf = jnp.asarray([0.2, 0.5, 1.0], dtype=jnp.float32)
    assert int(sample(cdf, jnp.asarray(0.0, dtype=jnp.float32))) == 1
    assert int(sample(cdf, jnp.asarray(0.2, dtype=jnp.float32))) == 2
    assert int(sample(cdf, jnp.asarray(0.5, dtype=jnp.float32))) == 3
    assert int(sample(cdf, jnp.asarray(0.999, dtype=jnp.float32))) == 3
    assert int(sample(jnp.asarray([0.1, 0.2], dtype=jnp.float32), jnp.asarray(0.9, dtype=jnp.float32))) == 1


def test_noise_renewal_is_global_and_ignored_candidates_do_not_change_actions():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.exploration import (
        apply_repeated_noise,
        create_exploration,
    )

    mean = jnp.zeros((2, 2), dtype=jnp.float32)
    standard_deviation = jnp.ones((2, 2), dtype=jnp.float32)
    first_noise = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    ignored_noise = jnp.full((2, 2), 9.0, dtype=jnp.float32)
    renewed_noise = jnp.asarray([[-1.0, -2.0], [-3.0, -4.0]], dtype=jnp.float32)

    first_action, state = apply_repeated_noise(
        mean,
        standard_deviation,
        create_exploration(2, 2),
        first_noise,
        jnp.asarray(2, dtype=jnp.int32),
        noise_scale=1.0,
    )
    np.testing.assert_allclose(np.asarray(first_action), np.asarray(jnp.tanh(first_noise)))
    assert int(state.repeat_count) == 1
    assert int(state.repeat_length) == 2

    second_action, state = apply_repeated_noise(
        mean,
        standard_deviation,
        state,
        ignored_noise,
        jnp.asarray(4, dtype=jnp.int32),
        noise_scale=1.0,
    )
    np.testing.assert_allclose(np.asarray(second_action), np.asarray(first_action))
    np.testing.assert_array_equal(np.asarray(state.noise), np.asarray(first_noise))
    assert int(state.repeat_count) == 2
    assert int(state.repeat_length) == 2

    third_action, state = apply_repeated_noise(
        mean,
        standard_deviation,
        state,
        renewed_noise,
        jnp.asarray(3, dtype=jnp.int32),
        noise_scale=1.0,
    )
    np.testing.assert_allclose(np.asarray(third_action), np.asarray(jnp.tanh(renewed_noise)))
    np.testing.assert_array_equal(np.asarray(state.noise), np.asarray(renewed_noise))
    assert int(state.repeat_count) == 1
    assert int(state.repeat_length) == 3


def test_identity_exploration_does_not_squash_raw_actions():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.exploration import apply_repeated_noise, create_exploration

    mean = jnp.asarray([[2.0, -3.0]], dtype=jnp.float32)
    noise = jnp.asarray([[1.0, -1.0]], dtype=jnp.float32)
    action, _ = apply_repeated_noise(
        mean,
        jnp.ones_like(mean),
        create_exploration(1, 2),
        noise,
        jnp.asarray(1, dtype=jnp.int32),
        noise_scale=1.0,
        action_transform="identity",
        action_scale=1.0,
    )

    np.testing.assert_array_equal(np.asarray(action), [[3.0, -4.0]])


def test_stochastic_collection_inference_is_one_jit_and_always_advances_key():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.exploration import (
        ExplorationState,
        collection_action,
        truncated_zeta_cdf,
    )

    actor, variables, observation = _actor(jax, jnp)
    cdf = truncated_zeta_cdf(2.0, 16)
    state = ExplorationState(
        noise=jnp.full((2, 2), 0.25, dtype=jnp.float32),
        repeat_count=jnp.asarray(1, dtype=jnp.int32),
        repeat_length=jnp.asarray(16, dtype=jnp.int32),
    )

    @jax.jit
    def infer(values, obs, exploration, key):
        return collection_action(actor, values, obs, exploration, key, cdf, stochastic=True)

    key = jax.random.key(7)
    first_action, first_state, first_key = infer(variables, observation, state, key)
    second_action, second_state, second_key = infer(variables, observation, first_state, first_key)

    assert not np.array_equal(np.asarray(jax.random.key_data(first_key)), np.asarray(jax.random.key_data(key)))
    assert not np.array_equal(
        np.asarray(jax.random.key_data(second_key)),
        np.asarray(jax.random.key_data(first_key)),
    )
    np.testing.assert_array_equal(np.asarray(first_state.noise), np.asarray(state.noise))
    np.testing.assert_array_equal(np.asarray(second_state.noise), np.asarray(state.noise))
    assert int(first_state.repeat_count) == 2
    assert int(second_state.repeat_count) == 3
    np.testing.assert_allclose(np.asarray(first_action), np.asarray(second_action), rtol=1.0e-6)


def test_deterministic_collection_inference_preserves_key_and_exploration_state():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.exploration import (
        collection_action,
        create_exploration,
        truncated_zeta_cdf,
    )
    from flash_chord.training.flash_sac.network import deterministic_action

    actor, variables, observation = _actor(jax, jnp)
    state = create_exploration(2, 2)
    key = jax.random.key(11)
    cdf = truncated_zeta_cdf(2.0, 16)

    @jax.jit
    def infer(values, obs, exploration, action_key):
        return collection_action(actor, values, obs, exploration, action_key, cdf, stochastic=False)

    action, next_state, next_key = infer(variables, observation, state, key)
    expected = deterministic_action(actor.apply(variables, observation, training=False))
    np.testing.assert_allclose(np.asarray(action), np.asarray(expected))
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(next_state), strict=True):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
    np.testing.assert_array_equal(
        np.asarray(jax.random.key_data(next_key)),
        np.asarray(jax.random.key_data(key)),
    )


def test_exploration_configuration_and_shape_validation():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.exploration import (
        apply_repeated_noise,
        create_exploration,
        truncated_zeta_cdf,
    )

    with pytest.raises(ValueError, match="must be positive"):
        create_exploration(0, 2)
    with pytest.raises(ValueError, match="exponent"):
        truncated_zeta_cdf(0.0, 16)
    with pytest.raises(ValueError, match="candidate_noise has shape"):
        apply_repeated_noise(
            jnp.zeros((2, 2), dtype=jnp.float32),
            jnp.ones((2, 2), dtype=jnp.float32),
            create_exploration(2, 2),
            jnp.zeros((2, 3), dtype=jnp.float32),
            jnp.asarray(1, dtype=jnp.int32),
            noise_scale=1.0,
        )
