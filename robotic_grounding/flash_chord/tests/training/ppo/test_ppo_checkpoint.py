# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PPO learner checkpoint round-trip and resume tests."""

import stat

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
        return VectorStep(
            observation=self.jnp.full((self.world_count, self.observation_dim), self.step_count / 10.0),
            reward=1.0 - self.jnp.square(action).mean(axis=-1),
            terminated=self.jnp.zeros(self.world_count, dtype=self.jnp.int32),
            truncated=self.jnp.zeros(self.world_count, dtype=self.jnp.int32),
            episode_return=self.jnp.zeros(self.world_count),
            episode_length=self.jnp.zeros(self.world_count, dtype=self.jnp.int32),
            episode_reference_progress=self.jnp.zeros(self.world_count),
        )


def _config(hidden_dim=8):
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig

    return TrainingConfig(
        world_count=2,
        rollout_steps=2,
        network=NetworkConfig(actor_hidden_dims=(hidden_dim,), critic_hidden_dims=(hidden_dim,)),
        ppo=PPOConfig(learning_epochs=1, mini_batches=2, schedule="fixed"),
    )


def test_checkpoint_round_trip_restores_all_leaves_and_can_resume(tmp_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    pytest.importorskip("safetensors")

    from flash_chord.training.checkpoint import load_checkpoint, read_checkpoint_metadata, save_checkpoint
    from flash_chord.training.ppo.learner import create_learner, train_iteration

    config = _config()
    env = _VectorEnv(jnp)
    learner, _ = train_iteration(env, create_learner(env, config), config)
    path = tmp_path / "learner.safetensors"
    save_checkpoint(path, learner, metadata={"voc_scale": 0.75, "source_run": "sizirwre"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert read_checkpoint_metadata(path) == {"voc_scale": "0.75", "source_run": "sizirwre"}

    template_env = _VectorEnv(jnp)
    template = create_learner(template_env, config)
    restored, metadata = load_checkpoint(path, template)
    assert metadata == {"voc_scale": "0.75", "source_run": "sizirwre"}
    for expected, actual in zip(jax.tree.leaves(learner), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    restored, metrics = train_iteration(template_env, restored, config)
    assert int(restored.iteration) == 2
    assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(metrics))


def test_checkpoint_rejects_incompatible_template(tmp_path):
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    pytest.importorskip("safetensors")

    from flash_chord.training.checkpoint import load_checkpoint, save_checkpoint
    from flash_chord.training.ppo.learner import create_learner

    env = _VectorEnv(jnp)
    path = tmp_path / "learner.safetensors"
    save_checkpoint(path, create_learner(env, _config(hidden_dim=8)))

    incompatible = create_learner(_VectorEnv(jnp), _config(hidden_dim=4))
    with pytest.raises(ValueError, match="checkpoint leaf"):
        load_checkpoint(path, incompatible)


def test_inference_restore_keeps_observation_for_different_world_count(tmp_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    pytest.importorskip("safetensors")

    from flash_chord.training.checkpoint import load_checkpoint_for_inference, save_checkpoint
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig
    from flash_chord.training.ppo.learner import create_learner

    source = create_learner(_VectorEnv(jnp), _config())
    path = tmp_path / "learner.safetensors"
    save_checkpoint(path, source)

    class SingleWorldEnv(_VectorEnv):
        world_count = 1

        def reset(self):
            return self.jnp.ones((self.world_count, self.observation_dim))

    single_world_config = TrainingConfig(
        world_count=1,
        rollout_steps=4,
        network=NetworkConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        ppo=PPOConfig(learning_epochs=1, mini_batches=4, schedule="fixed"),
    )
    template = create_learner(SingleWorldEnv(jnp), single_world_config)
    restored, _ = load_checkpoint_for_inference(path, template)

    np.testing.assert_array_equal(np.asarray(restored.observation), np.asarray(template.observation))
    for expected, actual in zip(
        jax.tree.leaves(source.train_state.params),
        jax.tree.leaves(restored.train_state.params),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
