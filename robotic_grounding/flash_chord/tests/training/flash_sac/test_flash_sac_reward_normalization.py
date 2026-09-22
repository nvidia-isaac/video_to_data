# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for native FlashSAC adaptive reward scaling."""

import json
from pathlib import Path

import numpy as np
import pytest

_ORACLE_PATH = Path(__file__).with_name("fixtures") / "reward_normalization_oracle.json"


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    return jax, jnp


@pytest.mark.parametrize("compiled", [False, True])
def test_reward_normalizer_matches_frozen_upstream_torch_oracle(compiled):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.reward_normalization import (
        create_reward_normalizer,
        normalize_rewards,
        update_reward_normalizer,
    )

    oracle = json.loads(_ORACLE_PATH.read_text())
    assert oracle["upstream_commit"] == "87edc9061150ae9e962dd84e6544e27a1554b3ab"

    def update(state, reward, terminated, truncated):
        return update_reward_normalizer(
            state,
            reward,
            terminated,
            truncated,
            discount=oracle["discount"],
            moments_epsilon=1.0e-4,
        )

    def normalize(state, rewards):
        return normalize_rewards(state, rewards, bound=oracle["bound"], epsilon=1.0e-8)

    update = jax.jit(update) if compiled else update
    normalize = jax.jit(normalize) if compiled else normalize
    state = create_reward_normalizer(4)
    probe = jnp.asarray(oracle["probe"], dtype=jnp.float32)
    for transition, expected in zip(oracle["transitions"], oracle["states"], strict=True):
        state = update(
            state,
            jnp.asarray(transition["reward"], dtype=jnp.float32),
            jnp.asarray(transition["terminated"], dtype=jnp.bool_),
            jnp.asarray(transition["truncated"], dtype=jnp.bool_),
        )
        np.testing.assert_allclose(np.asarray(state.discounted_return), expected["discounted_return"], rtol=1.0e-6)
        assert float(state.maximum_absolute_return) == pytest.approx(expected["maximum_absolute_return"][0])
        assert float(state.moments.mean) == pytest.approx(expected["mean"][0], rel=1.0e-6)
        assert float(state.moments.variance) == pytest.approx(expected["variance"][0], rel=1.0e-6)
        assert float(state.moments.count) == pytest.approx(expected["count"])
        np.testing.assert_allclose(np.asarray(normalize(state, probe)), expected["normalized_probe"], rtol=1.0e-6)


def test_running_moment_merge_uses_biased_variance_and_repeated_epsilon():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.reward_normalization import RunningMomentsState, update_running_moments

    state = RunningMomentsState(
        mean=jnp.asarray(2.0, dtype=jnp.float32),
        variance=jnp.asarray(3.0, dtype=jnp.float32),
        count=jnp.asarray(5.0, dtype=jnp.float32),
    )
    samples = jnp.asarray([1.0, 4.0, 7.0], dtype=jnp.float32)
    result = update_running_moments(state, samples, epsilon=1.0e-4)

    sample_mean = np.mean([1.0, 4.0, 7.0])
    sample_variance = np.var([1.0, 4.0, 7.0])
    total_count = 8.0
    ratio = 3.0 / total_count
    delta = sample_mean - 2.0
    expected_variance = (3.0 * (5.0 + 1.0e-4) + sample_variance * 3.0 + delta**2 * 5.0 * ratio) / total_count
    assert float(result.mean) == pytest.approx(2.0 + delta * ratio)
    assert float(result.variance) == pytest.approx(expected_variance)
    assert float(result.count) == 8.0


def test_denominator_uses_larger_variance_or_lifetime_bound_without_clipping():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.reward_normalization import (
        RewardNormalizerState,
        RunningMomentsState,
        normalize_rewards,
        reward_scale_denominator,
    )

    def state(maximum):
        return RewardNormalizerState(
            discounted_return=jnp.zeros((2,), dtype=jnp.float32),
            maximum_absolute_return=jnp.asarray(maximum, dtype=jnp.float32),
            moments=RunningMomentsState(
                mean=jnp.asarray(0.0, dtype=jnp.float32),
                variance=jnp.asarray(4.0, dtype=jnp.float32),
                count=jnp.asarray(2.0, dtype=jnp.float32),
            ),
        )

    assert float(reward_scale_denominator(state(5.0), bound=5.0)) == pytest.approx(2.0)
    assert float(reward_scale_denominator(state(20.0), bound=5.0)) == pytest.approx(4.0)
    normalized = normalize_rewards(state(20.0), jnp.asarray([100.0], dtype=jnp.float32), bound=5.0)
    assert float(normalized[0]) == 25.0


def test_current_weight_relabel_and_normalization_compose_in_one_jit():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import ReplayBatch, relabel_rewards
    from flash_chord.training.flash_sac.reward_normalization import (
        RewardNormalizerState,
        RunningMomentsState,
        normalize_rewards,
    )

    batch = ReplayBatch(
        observation=jnp.zeros((2, 1), dtype=jnp.float32),
        action=jnp.zeros((2, 1), dtype=jnp.float32),
        discounted_objective_terms=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        terminated=jnp.zeros((2,), dtype=jnp.bool_),
        truncated=jnp.zeros((2,), dtype=jnp.bool_),
        next_observation=jnp.zeros((2, 1), dtype=jnp.float32),
        critic_context=jnp.zeros((2, 3), dtype=jnp.float32),
        next_critic_context=jnp.zeros((2, 3), dtype=jnp.float32),
    )
    normalizer = RewardNormalizerState(
        discounted_return=jnp.zeros((2,), dtype=jnp.float32),
        maximum_absolute_return=jnp.asarray(0.0, dtype=jnp.float32),
        moments=RunningMomentsState(
            mean=jnp.asarray(0.0, dtype=jnp.float32),
            variance=jnp.asarray(4.0, dtype=jnp.float32),
            count=jnp.asarray(2.0, dtype=jnp.float32),
        ),
    )

    @jax.jit
    def relabel_and_normalize(replay_batch, weights, state):
        rewards = relabel_rewards(replay_batch, weights, jnp.asarray(0.05, dtype=jnp.float32))
        return normalize_rewards(state, rewards, bound=5.0)

    first = relabel_and_normalize(batch, jnp.asarray([2.0, -1.0], dtype=jnp.float32), normalizer)
    second = relabel_and_normalize(batch, jnp.asarray([1.0, 1.0], dtype=jnp.float32), normalizer)
    np.testing.assert_allclose(np.asarray(first), [0.0, 0.05])
    np.testing.assert_allclose(np.asarray(second), [0.075, 0.175])


def test_reward_normalizer_state_is_checkpointable(tmp_path):
    jax, jnp = _dependencies()
    pytest.importorskip("safetensors")

    from flash_chord.training.checkpoint import load_checkpoint, save_checkpoint
    from flash_chord.training.flash_sac.reward_normalization import (
        create_reward_normalizer,
        update_reward_normalizer,
    )

    state = update_reward_normalizer(
        create_reward_normalizer(2),
        jnp.asarray([1.0, -2.0], dtype=jnp.float32),
        jnp.asarray([False, True], dtype=jnp.bool_),
        jnp.asarray([False, False], dtype=jnp.bool_),
        discount=0.99,
    )
    path = tmp_path / "normalizer.safetensors"
    save_checkpoint(path, state)
    restored, _ = load_checkpoint(path, create_reward_normalizer(2))

    for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_reward_normalizer_validates_shapes_and_float32_state():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.reward_normalization import (
        create_reward_normalizer,
        normalize_rewards,
        update_reward_normalizer,
    )

    state = create_reward_normalizer(2)
    assert all(leaf.dtype == jnp.float32 for leaf in (state.discounted_return, state.maximum_absolute_return))
    with pytest.raises(ValueError, match="reward has shape"):
        update_reward_normalizer(
            state,
            jnp.ones((3,), dtype=jnp.float32),
            jnp.zeros((2,), dtype=jnp.bool_),
            jnp.zeros((2,), dtype=jnp.bool_),
            discount=0.99,
        )
    with pytest.raises(TypeError, match="rewards must be float32"):
        normalize_rewards(state, jnp.ones((2,), dtype=jnp.float16), bound=5.0)
