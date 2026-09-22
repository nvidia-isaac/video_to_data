# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint separation and restore tests for native FlashSAC."""

import numpy as np
import pytest


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("safetensors")
    return jax, jnp


def _learner(jnp, world_count=2, observation_storage_dtype="float32"):
    from flash_chord.training.flash_sac.config import NetworkConfig, ReplayConfig, TrainingConfig
    from flash_chord.training.flash_sac.learner import create_learner

    config = TrainingConfig(
        world_count=world_count,
        total_environment_steps=10 * world_count,
        updates_per_collection=1.0,
        mixed_precision=False,
        replay=ReplayConfig(
            capacity=8,
            minimum_size=2,
            batch_size=2,
            n_step=3,
            observation_storage_dtype=observation_storage_dtype,
        ),
        network=NetworkConfig(
            actor_blocks=1,
            actor_hidden_dim=8,
            critic_blocks=1,
            critic_hidden_dim=8,
            expansion=2,
            atom_count=11,
        ),
    )
    observation = jnp.arange(world_count * 3, dtype=jnp.float32).reshape(world_count, 3)
    context = jnp.zeros((world_count, 3), dtype=jnp.float32)
    return create_learner(
        config,
        observation,
        context,
        action_dim=2,
        objective_term_count=2,
    )[0]


def _assert_tree_equal(jax, actual, expected):
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf))


def test_policy_and_training_state_are_separate_and_restore_without_replay(tmp_path):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.checkpoint import (
        actor_inference_state,
        load_actor_checkpoint,
        restore_learner_checkpoint,
        save_learner_checkpoint,
        training_checkpoint,
    )

    learner = _learner(jnp)
    learner = learner.replace(
        reward_normalizer=learner.reward_normalizer.replace(
            discounted_return=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        ),
        logging=learner.logging.replace(
            running_episode_return=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            objective_contribution_sum=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
            environment_step_count=jnp.asarray(2, dtype=jnp.int32),
        ),
        environment_steps=jnp.asarray(12, dtype=jnp.int32),
        global_update_step=jnp.asarray(7, dtype=jnp.int32),
        update_credit=jnp.asarray(0.5, dtype=jnp.float32),
    )
    paths = save_learner_checkpoint(tmp_path, learner, metadata={"resolved_config_json": "{}"})

    assert paths.policy.name == "policy_12.safetensors"
    assert paths.training_state.name == "state_12.safetensors"
    assert paths.replay is None
    actor, actor_metadata = load_actor_checkpoint(paths.policy, actor_inference_state(_learner(jnp, world_count=1)))
    _assert_tree_equal(jax, actor, actor_inference_state(learner))
    assert actor_metadata["algorithm"] == "flash_sac"
    assert actor_metadata["checkpoint_kind"] == "policy"
    assert actor_metadata["environment_steps"] == "12"

    fresh = _learner(jnp)
    fresh_observation = np.asarray(fresh.observation).copy()
    restored, state_metadata = restore_learner_checkpoint(paths.training_state, fresh)
    _assert_tree_equal(
        jax,
        training_checkpoint(restored),
        training_checkpoint(learner).replace(
            reward_normalizer=learner.reward_normalizer.replace(
                discounted_return=jnp.zeros_like(learner.reward_normalizer.discounted_return)
            )
        ),
    )
    np.testing.assert_array_equal(np.asarray(restored.observation), fresh_observation)
    _assert_tree_equal(jax, restored.logging, fresh.logging)
    assert int(restored.replay.size) == 0
    assert state_metadata["checkpoint_kind"] == "training_state"
    assert state_metadata["checkpoint_set_id"] == actor_metadata["checkpoint_set_id"]


@pytest.mark.parametrize("observation_storage_dtype", ["float32", "float16"])
def test_optional_replay_is_separate_and_pending_is_flushed_on_restore(tmp_path, observation_storage_dtype):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.checkpoint import restore_learner_checkpoint, save_learner_checkpoint

    learner = _learner(jnp, observation_storage_dtype=observation_storage_dtype)
    observation_dtype = jnp.dtype(observation_storage_dtype)
    replay = learner.replay.replace(
        size=jnp.asarray(2, dtype=jnp.int32),
        write_index=jnp.asarray(2, dtype=jnp.int32),
        storage=learner.replay.storage.replace(
            observation=learner.replay.storage.observation.at[:2].set(
                jnp.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=observation_dtype)
            )
        ),
        pending=learner.replay.pending.replace(
            size=jnp.asarray(1, dtype=jnp.int32),
            write_index=jnp.asarray(1, dtype=jnp.int32),
        ),
    )
    learner = learner.replace(replay=replay, environment_steps=jnp.asarray(4, dtype=jnp.int32))
    paths = save_learner_checkpoint(tmp_path, learner, save_replay=True)

    assert paths.replay is not None
    assert paths.replay.name == "replay_4.safetensors"
    restored, _ = restore_learner_checkpoint(
        paths.training_state,
        _learner(jnp, observation_storage_dtype=observation_storage_dtype),
        replay_path=paths.replay,
    )
    assert int(restored.replay.size) == 2
    assert int(restored.replay.write_index) == 2
    assert int(restored.replay.pending.size) == 0
    assert int(restored.replay.pending.write_index) == 0
    assert restored.replay.storage.observation.dtype == observation_dtype
    np.testing.assert_array_equal(
        np.asarray(restored.replay.storage.observation[:2]),
        [[10, 11, 12], [20, 21, 22]],
    )


def test_replay_restore_rejects_storage_dtype_mismatch(tmp_path):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.checkpoint import restore_learner_checkpoint, save_learner_checkpoint

    source = _learner(jnp, observation_storage_dtype="float16")
    paths = save_learner_checkpoint(tmp_path, source, save_replay=True)

    with pytest.raises(ValueError, match="float16; expected .*float32"):
        restore_learner_checkpoint(
            paths.training_state,
            _learner(jnp, observation_storage_dtype="float32"),
            replay_path=paths.replay,
        )


def test_checkpoint_kind_is_validated_before_restore(tmp_path):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.checkpoint import (
        actor_inference_state,
        load_actor_checkpoint,
        save_learner_checkpoint,
    )

    learner = _learner(jnp)
    paths = save_learner_checkpoint(tmp_path, learner)
    with pytest.raises(ValueError, match="checkpoint kind"):
        load_actor_checkpoint(paths.training_state, actor_inference_state(learner))


def test_resume_flags_keep_fresh_optimizer_scaler_and_reward_normalizer(tmp_path):
    jax, jnp = _dependencies()

    from flash_chord.training.flash_sac.checkpoint import restore_learner_checkpoint, save_learner_checkpoint

    source = _learner(jnp)
    shifted_optimizer = jax.tree.map(lambda value: value + 1, source.actor.optimizer_state)
    source = source.replace(
        actor=source.actor.replace(
            params=jax.tree.map(lambda value: value + 0.25, source.actor.params),
            optimizer_state=shifted_optimizer,
            schedule_step=jnp.asarray(9, dtype=jnp.int32),
        ),
        critic=source.critic.replace(schedule_step=jnp.asarray(8, dtype=jnp.int32)),
        temperature=source.temperature.replace(schedule_step=jnp.asarray(7, dtype=jnp.int32)),
        reward_normalizer=source.reward_normalizer.replace(
            discounted_return=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
            maximum_absolute_return=jnp.asarray(12.0, dtype=jnp.float32),
            moments=source.reward_normalizer.moments.replace(
                mean=jnp.asarray(5.0, dtype=jnp.float32),
                variance=jnp.asarray(6.0, dtype=jnp.float32),
                count=jnp.asarray(7.0, dtype=jnp.float32),
            ),
        ),
        loss_scale=source.loss_scale.replace(
            scale=jnp.asarray(128.0, dtype=jnp.float32),
            finite_steps=jnp.asarray(11, dtype=jnp.int32),
        ),
        update_credit=jnp.asarray(0.75, dtype=jnp.float32),
        environment_steps=jnp.asarray(4, dtype=jnp.int32),
        global_update_step=jnp.asarray(13, dtype=jnp.int32),
    )
    paths = save_learner_checkpoint(tmp_path, source)
    fresh = _learner(jnp)
    restored, _ = restore_learner_checkpoint(
        paths.training_state,
        fresh,
        load_optimizer=False,
        load_reward_normalizer=False,
    )

    _assert_tree_equal(jax, restored.actor.params, source.actor.params)
    _assert_tree_equal(jax, restored.actor.optimizer_state, fresh.actor.optimizer_state)
    _assert_tree_equal(jax, restored.critic.optimizer_state, fresh.critic.optimizer_state)
    _assert_tree_equal(jax, restored.temperature.optimizer_state, fresh.temperature.optimizer_state)
    _assert_tree_equal(jax, restored.loss_scale, fresh.loss_scale)
    _assert_tree_equal(jax, restored.reward_normalizer, fresh.reward_normalizer)
    assert int(restored.actor.schedule_step) == 0
    assert int(restored.critic.schedule_step) == 0
    assert int(restored.temperature.schedule_step) == 0
    assert int(restored.global_update_step) == 0
    assert float(restored.update_credit) == 0.0
    assert int(restored.environment_steps) == 4


def test_resume_rejects_replay_from_another_checkpoint_set(tmp_path):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.checkpoint import restore_learner_checkpoint, save_learner_checkpoint

    learner = _learner(jnp).replace(environment_steps=jnp.asarray(4, dtype=jnp.int32))
    first = save_learner_checkpoint(tmp_path / "first", learner, save_replay=True)
    second = save_learner_checkpoint(tmp_path / "second", learner, save_replay=True)

    with pytest.raises(ValueError, match="different checkpoint_set_id"):
        restore_learner_checkpoint(
            first.training_state,
            _learner(jnp),
            replay_path=second.replay,
        )
