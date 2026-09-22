# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for native JAX FlashSAC replay."""

import json
from pathlib import Path

import numpy as np
import pytest

_ORACLE_PATH = Path(__file__).with_name("fixtures") / "replay_oracle.json"


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    return jax, jnp


def _spec(**overrides):
    from flash_chord.training.flash_sac.replay import ReplaySpec

    values = {
        "capacity": 32,
        "minimum_size": 7,
        "batch_size": 7,
        "n_step": 3,
        "discount": 0.9,
        "world_count": 7,
        "observation_dim": 1,
        "action_dim": 1,
        "objective_term_count": 2,
        "critic_context_dim": 3,
    }
    values.update(overrides)
    return ReplaySpec(**values)


def _oracle_transition(step, jnp):
    from flash_chord.training.flash_sac.replay import ReplayInsert

    rewards = (
        [1, 1, 1, 1, 1, 1, 1],
        [2, 101, 2, 2, 104, 2, 2],
        [3, 201, 202, 3, 204, 205, 3],
    )
    terminated = (
        [False, True, False, False, False, False, False],
        [False, False, True, False, False, False, False],
        [False, False, False, True, False, False, False],
    )
    truncated = (
        [False, False, False, False, True, False, False],
        [False, False, False, False, False, True, False],
        [False, False, False, False, False, False, True],
    )
    world = jnp.arange(7, dtype=jnp.float32)
    reward = jnp.asarray(rewards[step], dtype=jnp.float32)
    observation = (world + 100.0 * step)[:, None]
    next_observation = (world + 10.0 + 10.0 * step)[:, None]
    context_offsets = jnp.asarray([0.0, 0.1, 0.2], dtype=jnp.float32)
    return ReplayInsert(
        observation=observation,
        action=(world + 10.0 * step)[:, None],
        objective_terms=jnp.stack((reward, -0.5 * reward), axis=-1),
        terminated=jnp.asarray(terminated[step], dtype=jnp.bool_),
        truncated=jnp.asarray(truncated[step], dtype=jnp.bool_),
        next_observation=next_observation,
        critic_context=observation + context_offsets,
        next_critic_context=next_observation + context_offsets,
    )


def _scalar_transition(jnp, observation, action, term, terminated=False, truncated=False, next_observation=None):
    from flash_chord.training.flash_sac.replay import ReplayInsert

    if next_observation is None:
        next_observation = observation + 1
    observation_array = jnp.asarray([[observation]], dtype=jnp.float32)
    next_observation_array = jnp.asarray([[next_observation]], dtype=jnp.float32)
    context_offsets = jnp.asarray([0.0, 0.1, 0.2], dtype=jnp.float32)
    return ReplayInsert(
        observation=observation_array,
        action=jnp.asarray([[action]], dtype=jnp.float32),
        objective_terms=jnp.asarray([[term]], dtype=jnp.float32),
        terminated=jnp.asarray([terminated], dtype=jnp.bool_),
        truncated=jnp.asarray([truncated], dtype=jnp.bool_),
        next_observation=next_observation_array,
        critic_context=observation_array + context_offsets,
        next_critic_context=next_observation_array + context_offsets,
    )


def test_mixer_memory_accounting_and_allocation_shapes_are_exact():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import create_replay, replay_memory

    mixer_1024 = _spec(
        capacity=1_000_000,
        minimum_size=100_000,
        batch_size=2_048,
        world_count=1_024,
        observation_dim=722,
        action_dim=56,
        objective_term_count=9,
    )
    memory = replay_memory(mixer_1024)
    assert memory.bytes_per_transition == 6_062
    assert memory.ring_bytes == 6_062_000_000
    assert memory.pending_bytes == 12_414_976
    assert memory.counter_bytes == 16
    assert memory.sampled_batch_bytes == 12_414_976
    assert memory.total_bytes == 6_074_414_992
    assert replay_memory(_spec(**{**mixer_1024.__dict__, "world_count": 4_096})).total_bytes == 6_111_659_920

    state = create_replay(_spec(capacity=8, minimum_size=2, batch_size=2, world_count=2))
    assert state.storage.observation.shape == (8, 1)
    assert state.storage.critic_context.shape == (8, 3)
    assert state.storage.terminated.dtype == jnp.bool_
    assert state.pending.observation.shape == (2, 2, 1)
    assert int(state.size) == 0
    assert int(state.pending.size) == 0


def test_float16_observation_storage_fits_ten_million_dexmate_transitions_and_samples_float32():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import create_replay, gather_replay, insert_replay, replay_memory

    dexmate_2048 = _spec(
        capacity=10_000_000,
        minimum_size=409_600,
        batch_size=2_048,
        world_count=2_048,
        observation_dim=778,
        action_dim=58,
        objective_term_count=9,
        critic_context_dim=4,
        observation_storage_dtype="float16",
    )
    memory = replay_memory(dexmate_2048)
    assert memory.bytes_per_transition == 3_414
    assert memory.ring_bytes == 34_140_000_000
    assert memory.pending_bytes == 13_983_744
    assert memory.counter_bytes == 16
    assert memory.sampled_batch_bytes == 13_365_248
    assert memory.total_bytes == 34_153_983_760

    state = create_replay(
        _spec(
            capacity=8,
            minimum_size=1,
            batch_size=1,
            n_step=1,
            world_count=1,
            objective_term_count=1,
            observation_storage_dtype="float16",
        )
    )
    assert state.storage.observation.dtype == jnp.float16
    assert state.storage.next_observation.dtype == jnp.float16
    assert state.storage.action.dtype == jnp.float32
    assert state.storage.discounted_objective_terms.dtype == jnp.float32
    assert state.storage.critic_context.dtype == jnp.float32
    assert state.pending.observation.dtype == jnp.float16

    state = insert_replay(state, _scalar_transition(jnp, 1.234567, 2.345678, 3.456789))
    batch = gather_replay(state, jnp.asarray([0], dtype=jnp.int32))
    assert batch.observation.dtype == jnp.float32
    assert batch.next_observation.dtype == jnp.float32
    assert float(batch.observation[0, 0]) == float(jnp.float16(1.234567))
    assert float(batch.action[0, 0]) == pytest.approx(2.345678)
    assert float(batch.discounted_objective_terms[0, 0]) == pytest.approx(3.456789)


def test_capacity_dominant_observation_leaves_allocate_before_smaller_payloads(monkeypatch):
    _dependencies()

    from flash_chord.training.flash_sac import replay

    allocation_shapes = []
    zeros = replay.jnp.zeros

    def record_zeros(shape, *args, **kwargs):
        allocation_shapes.append(tuple(shape))
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(replay.jnp, "zeros", record_zeros)
    replay.create_replay(
        _spec(
            capacity=8,
            minimum_size=1,
            batch_size=1,
            world_count=1,
            observation_dim=3,
            action_dim=2,
            objective_term_count=1,
            observation_storage_dtype="float16",
        )
    )

    assert allocation_shapes[:2] == [(8, 3), (8, 3)]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"capacity": 0}, "capacity must be positive"),
        ({"capacity": 6, "minimum_size": 6, "batch_size": 6}, "at least one"),
        ({"discount": 0.0}, "discount"),
        ({"observation_dim": 0}, "dimensions"),
        ({"observation_storage_dtype": "bfloat16"}, "observation_storage_dtype"),
    ],
)
def test_replay_spec_rejects_invalid_layouts(overrides, match):
    _dependencies()

    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


@pytest.mark.parametrize("observation_storage_dtype", ["float32", "float16"])
def test_vector_n_step_matches_frozen_upstream_torch_oracle(observation_storage_dtype):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import (
        can_sample,
        create_replay,
        gather_replay,
        insert_replay,
        relabel_rewards,
    )

    oracle = json.loads(_ORACLE_PATH.read_text())
    assert oracle["upstream_commit"] == "87edc9061150ae9e962dd84e6544e27a1554b3ab"
    state = create_replay(_spec(observation_storage_dtype=observation_storage_dtype))
    warmup = []
    for step in range(3):
        state = insert_replay(state, _oracle_transition(step, jnp))
        warmup.append({"size": int(state.size), "can_sample": bool(can_sample(state))})
    assert warmup == oracle["warmup"]

    batch = gather_replay(state, jnp.arange(7, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(batch.observation[:, 0]), oracle["observation"])
    np.testing.assert_array_equal(np.asarray(batch.action[:, 0]), oracle["action"])
    np.testing.assert_allclose(
        np.asarray(batch.discounted_objective_terms[:, 0]), oracle["discounted_reward"], rtol=1.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(batch.discounted_objective_terms[:, 1]),
        -0.5 * np.asarray(oracle["discounted_reward"]),
        rtol=1.0e-6,
    )
    np.testing.assert_array_equal(np.asarray(batch.terminated), oracle["terminated"])
    np.testing.assert_array_equal(np.asarray(batch.truncated), oracle["truncated"])
    np.testing.assert_array_equal(np.asarray(batch.next_observation[:, 0]), oracle["next_observation"])
    expected_next_context = np.asarray(oracle["next_observation"])[:, None] + np.asarray([0.0, 0.1, 0.2])
    np.testing.assert_allclose(np.asarray(batch.next_critic_context), expected_next_context, atol=1.0e-6)

    rewards = relabel_rewards(
        batch,
        jnp.asarray([2.0, -4.0], dtype=jnp.float32),
        jnp.asarray(0.05, dtype=jnp.float32),
    )
    np.testing.assert_allclose(np.asarray(rewards), 0.2 * np.asarray(oracle["discounted_reward"]), rtol=1.0e-6)


def test_autoreset_history_is_cut_at_boundary_without_flushing_pending():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import create_replay, gather_replay, insert_replay

    state = create_replay(_spec(capacity=8, minimum_size=1, batch_size=1, world_count=1, objective_term_count=1))
    transitions = (
        _scalar_transition(jnp, 0, 0, 1),
        _scalar_transition(jnp, 1, 1, 2, truncated=True, next_observation=99),
        _scalar_transition(jnp, 10, 2, 100, next_observation=11),
        _scalar_transition(jnp, 11, 3, 200, next_observation=12),
        _scalar_transition(jnp, 12, 4, 300, next_observation=13),
    )
    sizes = []
    for transition in transitions:
        state = insert_replay(state, transition)
        sizes.append(int(state.size))
    assert sizes == [0, 0, 1, 2, 3]

    batch = gather_replay(state, jnp.arange(3, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(batch.observation[:, 0]), [0, 1, 10])
    np.testing.assert_array_equal(np.asarray(batch.action[:, 0]), [0, 1, 2])
    np.testing.assert_allclose(np.asarray(batch.discounted_objective_terms[:, 0]), [2.8, 2.0, 523.0])
    np.testing.assert_array_equal(np.asarray(batch.terminated), [False, False, False])
    np.testing.assert_array_equal(np.asarray(batch.truncated), [True, True, False])
    np.testing.assert_array_equal(np.asarray(batch.next_observation[:, 0]), [99, 99, 13])


def test_nondivisible_ring_wrap_matches_upstream_physical_layout():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import ReplayInsert, create_replay, insert_replay

    spec = _spec(
        capacity=5,
        minimum_size=1,
        batch_size=2,
        n_step=1,
        world_count=3,
        objective_term_count=1,
    )
    state = create_replay(spec)
    for base in (0, 10, 20):
        observation = jnp.arange(base, base + 3, dtype=jnp.float32)[:, None]
        state = insert_replay(
            state,
            ReplayInsert(
                observation=observation,
                action=observation,
                objective_terms=observation,
                terminated=jnp.zeros((3,), dtype=jnp.bool_),
                truncated=jnp.zeros((3,), dtype=jnp.bool_),
                next_observation=observation + 1,
                critic_context=jnp.broadcast_to(observation, (3, 3)),
                next_critic_context=jnp.broadcast_to(observation + 1, (3, 3)),
            ),
        )
        if base == 10:
            np.testing.assert_array_equal(np.asarray(state.storage.observation[:, 0]), [12, 1, 2, 10, 11])
            assert int(state.write_index) == 1
            assert int(state.size) == 5

    np.testing.assert_array_equal(np.asarray(state.storage.observation[:, 0]), [12, 20, 21, 22, 11])
    assert int(state.write_index) == 4
    assert int(state.size) == 5


def test_uniform_sampling_uses_only_valid_entries_and_preserves_dtypes():
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import create_replay, insert_replay, sample_replay

    spec = _spec(capacity=8, minimum_size=1, batch_size=8, n_step=1, world_count=2, objective_term_count=1)
    state = create_replay(spec)
    for base in (0, 10):
        first = _scalar_transition(jnp, base, base, base + 1)
        second = _scalar_transition(jnp, base + 1, base + 1, base + 2)
        transition = jax.tree.map(lambda left, right: jnp.concatenate((left, right), axis=0), first, second)
        state = insert_replay(state, transition)

    batch = sample_replay(state, jax.random.key(7))
    assert batch.observation.shape == (8, 1)
    assert batch.terminated.dtype == jnp.bool_
    assert set(np.asarray(batch.observation[:, 0]).tolist()) <= {0.0, 1.0, 10.0, 11.0}


def test_flush_pending_retains_fifo_and_prevents_cross_stage_windows():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import create_replay, flush_pending, gather_replay, insert_replay

    state = create_replay(_spec(capacity=8, minimum_size=1, batch_size=1, world_count=1, objective_term_count=1))
    for value in (0, 1, 2):
        state = insert_replay(state, _scalar_transition(jnp, value, value, 1))
    assert int(state.size) == 1
    assert int(state.pending.size) == 2

    state = flush_pending(state)
    assert int(state.size) == 1
    assert int(state.pending.size) == 0
    for value in (100, 101, 102):
        state = insert_replay(state, _scalar_transition(jnp, value, value, 10))

    batch = gather_replay(state, jnp.arange(2, dtype=jnp.int32))
    np.testing.assert_allclose(np.asarray(batch.discounted_objective_terms[:, 0]), [2.71, 27.1])


def test_logical_clear_hides_old_entries_and_reuses_payload_values():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import can_sample, clear_replay, create_replay, insert_replay

    state = create_replay(_spec(capacity=8, minimum_size=1, batch_size=1, world_count=1, objective_term_count=1))
    for value in (0, 1, 2):
        state = insert_replay(state, _scalar_transition(jnp, value, value, 1))
    payload = np.asarray(state.storage.observation).copy()

    state = clear_replay(state)
    assert int(state.size) == 0
    assert int(state.write_index) == 0
    assert int(state.pending.size) == 0
    assert not bool(can_sample(state))
    np.testing.assert_array_equal(np.asarray(state.storage.observation), payload)

    for value in (100, 101, 102):
        state = insert_replay(state, _scalar_transition(jnp, value, value, 10))
    assert int(state.size) == 1
    assert float(state.storage.observation[0, 0]) == 100.0


def test_insert_validates_static_shapes_and_dtypes():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.replay import create_replay, insert_replay

    state = create_replay(_spec(capacity=8, minimum_size=1, batch_size=1, world_count=1, objective_term_count=1))
    transition = _scalar_transition(jnp, 0, 0, 1)
    with pytest.raises(ValueError, match="action has shape"):
        insert_replay(state, transition.replace(action=jnp.zeros((1, 2), dtype=jnp.float32)))


def test_fixed_bootstrap_discount_matches_upstream_without_effective_horizon():
    _dependencies()

    from flash_chord.training.flash_sac.replay import ReplayStorage

    spec = _spec()
    assert spec.bootstrap_discount == pytest.approx(0.9**3)
    assert "effective_horizon" not in ReplayStorage.__dataclass_fields__
