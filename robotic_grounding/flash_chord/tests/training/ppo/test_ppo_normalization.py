# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for device-resident empirical observation normalization."""

import numpy as np
import pytest


def test_normalization_matches_full_population_statistics_across_batches():
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")

    from flash_chord.training.ppo.normalization import create_normalization_state, update_normalization

    first = np.asarray([[1.0, 4.0], [3.0, 8.0]], dtype=np.float32)
    second = np.asarray([[5.0, 2.0], [7.0, 6.0], [9.0, 10.0]], dtype=np.float32)
    state = create_normalization_state(2)
    state = update_normalization(state, jnp.asarray(first))
    state = update_normalization(state, jnp.asarray(second))

    full = np.concatenate((first, second), axis=0)
    assert int(state.count) == full.shape[0]
    np.testing.assert_allclose(np.asarray(state.mean), full.mean(axis=0), rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(state.variance), full.var(axis=0), rtol=1.0e-6)


def test_normalization_uses_rsl_standard_deviation_epsilon():
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")

    from flash_chord.training.ppo.normalization import (
        create_normalization_state,
        normalize_observation,
        update_normalization,
    )

    observation = jnp.asarray([[1.0, 2.0], [3.0, 6.0]])
    state = update_normalization(create_normalization_state(2), observation)
    normalized = normalize_observation(state, observation, epsilon=1.0e-2)
    expected = (np.asarray(observation) - np.asarray(state.mean)) / (np.sqrt(np.asarray(state.variance)) + 1.0e-2)
    np.testing.assert_allclose(np.asarray(normalized), expected, rtol=1.0e-6)


def test_normalization_rejects_empty_feature_shape():
    pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")

    from flash_chord.training.ppo.normalization import create_normalization_state

    with pytest.raises(ValueError, match="observation_dim"):
        create_normalization_state(0)
