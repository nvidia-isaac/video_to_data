# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for training orchestration, logging, curriculum, and resume."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest


class _VectorEnv:
    world_count = 2
    observation_dim = 3
    action_dim = 2
    frame_dt = 0.05
    reference_num_frames = 10
    objective_term_names = (
        "object_keypoints",
        "hand_keypoints",
        "hand_joint_pos",
        "contact_wrench_support",
        "missed_contact",
        "unintended_contact",
        "termination",
        "action_rate_l2",
        "action_l2",
        "contact_force_l2",
        "force_closure",
    )
    termination_cause_names = ("wrist", "object")
    tracking_error_names = (
        "wrist_position_m",
        "wrist_orientation_rad",
        "object_position_m",
        "object_orientation_rad",
    )

    def __init__(self, jnp):
        self.jnp = jnp
        self.step_count = 0
        self.reset_count = 0
        self.stages = []

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        return self.jnp.zeros((self.world_count, self.observation_dim))

    def step(self, action):
        from flash_chord.training.environment import VectorStep

        self.step_count += 1
        completed = self.step_count % 2 == 0
        cycle = self.step_count // 2
        return VectorStep(
            observation=self.jnp.full((self.world_count, self.observation_dim), self.step_count / 10.0),
            reward=1.0 - self.jnp.square(action).mean(axis=-1),
            terminated=self.jnp.asarray([completed, False], dtype=self.jnp.int32),
            truncated=self.jnp.asarray([completed and cycle == 1, completed], dtype=self.jnp.int32),
            episode_return=self.jnp.asarray([3.0, 4.0]) if completed else self.jnp.zeros(self.world_count),
            episode_length=(
                self.jnp.asarray([2, 2], dtype=self.jnp.int32)
                if completed
                else self.jnp.zeros(self.world_count, dtype=self.jnp.int32)
            ),
            episode_reference_progress=(
                self.jnp.asarray([0.2, 0.6]) if completed else self.jnp.zeros(self.world_count)
            ),
            objective_terms=self.jnp.full(
                (self.world_count, len(self.objective_term_names)),
                float(self.step_count),
            ),
            termination_causes=self.jnp.asarray(
                [
                    [completed and cycle == 1, completed],
                    [False, False],
                ],
                dtype=self.jnp.int32,
            ),
            tracking_errors=self.jnp.full((self.world_count, 4), self.step_count / 10.0),
        )

    def set_curriculum(self, stage):
        self.stages.append(stage)


class _Logger:
    def __init__(self):
        self.records = []
        self.checkpoints = []
        self.closed = False

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def log_checkpoint(self, path, *, iteration, metadata):
        assert path.is_file()
        self.checkpoints.append((path, iteration, dict(metadata)))

    def close(self):
        self.closed = True


class _Progress:
    def __init__(self):
        self.started = []
        self.updated = []

    def start(self, environment_steps):
        self.started.append(environment_steps)

    def update(self, environment_steps):
        self.updated.append(environment_steps)


def _config(max_iterations):
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig

    return TrainingConfig(
        world_count=2,
        rollout_steps=2,
        max_iterations=max_iterations,
        save_interval=1,
        network=NetworkConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,)),
        ppo=PPOConfig(learning_epochs=1, mini_batches=2, schedule="fixed"),
    )


def _reference_checkpoint_metadata(friction: float):
    return {
        "resolved_config_json": json.dumps(
            {
                "reference": {
                    "object_articulations": {
                        "version": 1,
                        "entries": [{"physics": {"armature": 0.01, "friction": friction}}],
                    }
                }
            }
        )
    }


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    pytest.importorskip("safetensors")
    return jax, jnp


def test_runner_logs_applies_override_and_saves_each_iteration(tmp_path):
    _, jnp = _dependencies()

    from flash_chord.training.ppo.runner import metric_definitions, run_training

    env = _VectorEnv(jnp)
    logger = _Logger()
    progress = _Progress()
    config = _config(max_iterations=2)
    learner = run_training(
        env,
        config,
        logger=logger,
        checkpoint_dir=tmp_path,
        voc_scale_override=0.3,
        checkpoint_metadata={"resolved_config_json": '{"seed":42}'},
        progress=progress,
    )

    assert int(learner.iteration) == 2
    assert [step for step, _ in logger.records] == [4, 8]
    assert logger.closed is True
    assert progress.started == [0]
    assert progress.updated == [4, 8]
    assert [(path.name, iteration) for path, iteration, _ in logger.checkpoints] == [
        ("model_1.safetensors", 1),
        ("model_2.safetensors", 2),
    ]
    assert [metadata["voc_scale"] for _, _, metadata in logger.checkpoints] == [0.3, 0.3]
    assert len(env.stages) == 2
    assert all(stage.voc_scale == 0.3 for stage in env.stages)
    assert all(stage.objective_weights["contact_wrench_support"] == 10.0 for stage in env.stages)
    assert (tmp_path / "model_1.safetensors").is_file()
    assert (tmp_path / "model_2.safetensors").is_file()
    from flash_chord.training.checkpoint import read_checkpoint_metadata

    saved_metadata = read_checkpoint_metadata(tmp_path / "model_2.safetensors")
    assert saved_metadata["resolved_config_json"] == '{"seed":42}'
    assert saved_metadata["iteration"] == "2"
    assert saved_metadata["environment_steps"] == "8"
    assert "resolved_config_json" not in logger.checkpoints[-1][2]
    definitions = metric_definitions(env)
    assert len(definitions) == 34
    for environment_steps, metrics in logger.records:
        iteration = environment_steps // (config.world_count * config.rollout_steps)
        expected_term = 1.5 + 2.0 * (iteration - 1)
        expected_tracking_mean = 0.15 + 0.2 * (iteration - 1)
        assert metrics["curriculum/stage_index"] == 0.0
        assert metrics["curriculum/voc_scale"] == 0.3
        assert metrics["throughput/env_steps_per_second"] > 0.0
        assert metrics["ppo/explained_variance_per_rollout"] <= 1.0
        assert 0.0 <= metrics["ppo/clip_fraction_mean_per_optimizer_sample"] <= 1.0
        assert metrics["ppo/skipped_update_fraction_per_iteration"] == 0.0
        assert metrics["reward_components/hand_keypoints_weighted_mean_per_env_step"] == pytest.approx(
            0.05 * 0.25 * expected_term
        )
        termination_contribution = metrics["reward_components/termination_weighted_mean_per_env_step"]
        assert metrics["reward/dense_mean_per_env_step"] == pytest.approx(
            metrics["reward/total_mean_per_env_step"] - termination_contribution
        )
        assert metrics["episode/return_mean_per_completed_episode"] == 3.5
        assert metrics["episode/length_mean_per_completed_episode"] == 2.0
        assert metrics["episode/reference_progress_mean_per_completed_episode"] == pytest.approx(0.4)
        assert metrics["termination/reference_end_fraction_per_completed_episode"] == 0.5
        assert metrics["termination/wrist_fraction_per_completed_episode"] == 0.5 * (iteration == 1)
        assert metrics["termination/object_fraction_per_completed_episode"] == 0.5 * (iteration == 2)
        assert metrics["tracking_error/wrist_position_m_mean_env_max_per_step"] == pytest.approx(expected_tracking_mean)
        assert set(metrics) == set(definitions)
        assert "episode/completed_count_per_rollout" not in metrics
        assert not any(key.startswith("observation/") for key in metrics)
        assert not any("per_1000" in key for key in metrics)
        assert np.isfinite(tuple(metrics.values())).all()


def test_wandb_logger_versions_checkpoint_artifact_with_iteration_alias(monkeypatch, tmp_path):
    artifacts = []
    init_options = []

    class _Artifact:
        def __init__(self, name, type, metadata):
            self.name = name
            self.type = type
            self.metadata = metadata
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    class _Run:
        id = "test123"

        def log_artifact(self, artifact, aliases):
            artifacts.append((artifact, aliases))

        def finish(self):
            self.finished = True

    run = _Run()

    def _init(**options):
        init_options.append(options)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=_init, Artifact=_Artifact))

    from flash_chord.training.ppo.runner import WandbLogger

    logger = WandbLogger(_config(max_iterations=1), upload_checkpoints=True, mode="offline")
    checkpoint = tmp_path / "model_200.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    logger.log_checkpoint(checkpoint, iteration=200, metadata={"iteration": 200, "voc_scale": 0.25})
    logger.close()

    assert init_options[0]["mode"] == "offline"
    assert len(artifacts) == 1
    artifact, aliases = artifacts[0]
    assert artifact.name == "run-test123-checkpoint"
    assert artifact.type == "model"
    assert artifact.metadata == {"iteration": 200, "voc_scale": 0.25}
    assert artifact.files == [(str(checkpoint), checkpoint.name)]
    assert aliases == ["latest", "iteration-200"]
    assert run.finished is True


def test_runner_resume_restores_learner_but_resets_environment(tmp_path):
    _, jnp = _dependencies()

    from flash_chord.training.ppo.runner import run_training

    first_env = _VectorEnv(jnp)
    metadata = _reference_checkpoint_metadata(0.1)
    run_training(first_env, _config(max_iterations=1), checkpoint_dir=tmp_path, checkpoint_metadata=metadata)
    checkpoint = tmp_path / "model_1.safetensors"

    resumed_config = replace(
        _config(max_iterations=2),
        resume=True,
        checkpoint=str(checkpoint),
        rollout_steps=4,
    )
    resumed_env = _VectorEnv(jnp)
    logger = _Logger()
    learner = run_training(resumed_env, resumed_config, logger=logger, checkpoint_metadata=metadata)

    assert int(learner.iteration) == 2
    assert resumed_env.reset_count == 2
    assert len(resumed_env.stages) == 1
    assert [step for step, _ in logger.records] == [12]

    with pytest.raises(ValueError, match="object articulation config does not match"):
        run_training(
            _VectorEnv(jnp),
            resumed_config,
            checkpoint_metadata=_reference_checkpoint_metadata(0.0),
        )
