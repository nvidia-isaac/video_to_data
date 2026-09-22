# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic FlashSAC runner, metrics, curriculum, and checkpoint tests."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("safetensors")
    return jax, jnp


def _checkpoint_metadata(*, input_mode: str = "raw", friction: float | None = None):
    action = {
        "dimension": 2,
        "blocks": [{"name": "action", "start": 0, "end": 2}],
    }
    action["input_mode"] = input_mode
    action["input_mapping"] = "identity" if input_mode == "raw" else "linear"
    observation = {
        "dimension": 3,
        "blocks": [{"name": "observation", "start": 0, "end": 3}],
    }
    policy = {
        "action": action,
        "observation": observation,
    }
    context = {
        "dimension": 3,
        "blocks": [
            {"name": "applied_voc_scale", "start": 0, "end": 1},
            {"name": "target_voc_scale", "start": 1, "end": 2},
            {"name": "normalized_settling_progress", "start": 2, "end": 3},
        ],
    }
    critic = {
        "input_order": ["observation", "context", "action"],
        "input_dimension": 8,
        "observation": observation,
        "context": context,
        "action": action,
    }
    resolved_config = {}
    if friction is not None:
        resolved_config = {
            "reference": {
                "object_articulations": {
                    "version": 1,
                    "entries": [{"physics": {"armature": 0.01, "friction": friction}}],
                }
            }
        }
    return {
        "resolved_config_json": json.dumps(resolved_config),
        "policy_schema_json": json.dumps(policy),
        "critic_schema_json": json.dumps(critic),
    }


class _Logger:
    def __init__(self):
        self.records = []
        self.checkpoints = []
        self.closed = False

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def log_checkpoint(self, paths, *, environment_steps, metadata):
        self.checkpoints.append((environment_steps, paths, dict(metadata)))

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


class _Env:
    world_count = 2
    observation_dim = 3
    action_dim = 2
    frame_dt = 0.05
    objective_term_names = ("tracking", "termination")
    termination_cause_names = ("failure",)
    tracking_error_names = ("position",)
    critic_context_names = (
        "applied_voc_scale",
        "target_voc_scale",
        "normalized_settling_progress",
    )
    critic_context_dim = 3

    def __init__(self, jnp):
        self.jnp = jnp
        self.step_count = 0
        self.stage = None
        self.stage_history = []
        self.reset_to_first_frame_probability = 0.1
        self.reset_probability_history = []
        self.immediate_first_frame_probability = 0.0
        self.immediate_probability_history = []
        self.initial_critic_context = jnp.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=jnp.float32)

    def set_curriculum(self, stage):
        self.stage = stage
        self.stage_history.append(stage.voc_scale)
        probability = stage.reset_to_first_frame_probability
        self.reset_to_first_frame_probability = 0.1 if probability is None else probability
        self.reset_probability_history.append(self.reset_to_first_frame_probability)
        immediate_probability = stage.immediate_first_frame_probability
        self.immediate_first_frame_probability = 0.0 if immediate_probability is None else immediate_probability
        self.immediate_probability_history.append(self.immediate_first_frame_probability)

    def reset(self):
        self.step_count = 0
        target = self.stage.voc_scale
        self.initial_critic_context = self.jnp.asarray(
            [[1.0, target, 0.0], [1.0, target, 0.0]],
            dtype=self.jnp.float32,
        )
        return self.jnp.zeros((2, 3), dtype=self.jnp.float32)

    def step(self, action):
        from flash_chord.training.environment import VectorStep

        assert action.shape == (2, 2)
        self.step_count += 1
        value = float(self.step_count)
        objective_terms = self.jnp.asarray([[value, 0.0], [value + 1.0, 1.0]], dtype=self.jnp.float32)
        weights = self.jnp.asarray(self.stage.weights_for(self.objective_term_names), dtype=self.jnp.float32)
        reward = self.frame_dt * self.jnp.einsum("wt,t->w", objective_terms, weights)
        terminated = self.jnp.asarray([self.step_count == 4, False], dtype=self.jnp.int32)
        truncated = self.jnp.asarray([False, self.step_count == 2], dtype=self.jnp.int32)
        completed = self.jnp.maximum(terminated, truncated)
        observation = self.jnp.full((2, 3), value, dtype=self.jnp.float32)
        terminal_observation = observation + completed[:, None] * 100.0
        target = self.stage.voc_scale
        terminal_context = self.jnp.asarray(
            [[target, target, 1.0], [target, target, 1.0]],
            dtype=self.jnp.float32,
        )
        reset_context = self.jnp.asarray([1.0, target, 0.0], dtype=self.jnp.float32)
        context = self.jnp.where(completed[:, None].astype(bool), reset_context, terminal_context)
        episode_length = completed * self.step_count
        return VectorStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_return=completed.astype(self.jnp.float32) * reward,
            episode_length=episode_length,
            episode_reference_progress=completed.astype(self.jnp.float32) * 0.5,
            objective_terms=objective_terms,
            terminal_observation=terminal_observation,
            critic_context=context,
            terminal_critic_context=terminal_context,
        )

    def transition_diagnostics(self):
        causes = self.jnp.asarray([[self.step_count == 4], [False]], dtype=self.jnp.int32)
        errors = self.jnp.asarray([[self.step_count], [self.step_count + 1]], dtype=self.jnp.float32)
        return causes, errors

    def transition_logging_inputs(self):
        completed = self.jnp.asarray(
            [self.step_count == 4, self.step_count == 2],
            dtype=self.jnp.float32,
        )
        packed = self.jnp.asarray(
            [
                [self.step_count == 4, self.step_count],
                [False, self.step_count + 1],
            ],
            dtype=self.jnp.float32,
        )
        return completed * 0.5, packed


class _DeferredMetricEnv(_Env):
    def step(self, action):
        step = super().step(action)
        self.latest_step_metrics = (
            step.reward,
            step.episode_return,
            step.episode_length,
            step.episode_reference_progress,
        )
        return step.replace(
            reward=None,
            episode_return=None,
            episode_length=None,
            episode_reference_progress=None,
        )

    def transition_metrics(self):
        raise AssertionError("FlashSAC must not sample boundary-step metrics outside the compiled window")

    def transition_diagnostics(self):
        raise AssertionError("FlashSAC must not launch boundary-step diagnostic reductions")


def _config():
    from flash_chord.lifecycle.curriculum import CurriculumStage, FixedCurriculum
    from flash_chord.training.flash_sac.config import (
        CheckpointConfig,
        NetworkConfig,
        ReplayConfig,
        TrainingConfig,
    )

    stages = (
        CurriculumStage(1.0, {"tracking": 1.0, "termination": -5.0}),
        CurriculumStage(
            0.25,
            {"tracking": 2.0, "termination": -10.0},
            reset_to_first_frame_probability=1.0,
            immediate_first_frame_probability=0.5,
        ),
    )
    return TrainingConfig(
        world_count=2,
        total_environment_steps=8,
        updates_per_collection=1.0,
        mixed_precision=False,
        replay=ReplayConfig(capacity=8, minimum_size=2, batch_size=2, n_step=1),
        network=NetworkConfig(
            actor_blocks=1,
            actor_hidden_dim=8,
            critic_blocks=1,
            critic_hidden_dim=8,
            expansion=2,
            atom_count=11,
        ),
        checkpoint=CheckpointConfig(save_interval_environment_steps=4),
        curriculum=FixedCurriculum(thresholds=(4, 6), stages=stages),
    )


def test_runner_uses_environment_step_curriculum_logs_and_saves(tmp_path, monkeypatch):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac import runner

    ready_shapes = []
    original_block_until_ready = runner.jax.block_until_ready

    def record_ready(value):
        ready_shapes.append("pytree" if isinstance(value, runner.LearnerState) else value.shape)
        return original_block_until_ready(value)

    compile_calls = 0
    original_compile = runner._compile_executables

    def record_compile(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(runner.jax, "block_until_ready", record_ready)
    monkeypatch.setattr(runner, "_compile_executables", record_compile)

    env = _Env(jnp)
    logger = _Logger()
    progress = _Progress()
    config = _config()
    learner = runner.run_training(
        env,
        config,
        logger=logger,
        checkpoint_dir=tmp_path,
        log_interval=2,
        checkpoint_metadata=_checkpoint_metadata(),
        progress=progress,
    )

    assert int(learner.environment_steps) == 8
    assert int(learner.global_update_step) == 4
    assert int(learner.actor.schedule_step) == 2
    assert int(learner.critic.schedule_step) == 4
    assert int(learner.temperature.schedule_step) == 2
    assert learner.observation.device == runner.resolve_jax_device(config.device)
    assert compile_calls == 1
    assert ready_shapes.count("pytree") == 1
    assert logger.closed
    assert progress.started == [0]
    assert progress.updated == [2, 4, 6, 8]
    assert [step for step, _ in logger.records] == [4, 8]
    assert [step for step, _, _ in logger.checkpoints] == [4, 8]
    assert env.stage_history.count(0.25) >= 1
    assert 1.0 in env.reset_probability_history

    final = logger.records[-1][1]
    assert final["curriculum/stage_index"] == 1.0
    assert final["curriculum/voc_scale"] == 0.25
    assert final["curriculum/reset_to_first_frame_probability"] == 1.0
    assert final["curriculum/immediate_first_frame_probability"] == 0.5
    assert final["policy/action_abs_mean_per_env_action"] >= 0.0
    assert final["replay/size_transitions"] == 8.0
    assert final["replay/fill_fraction"] == 1.0
    assert final["throughput/env_steps_per_second"] > 0.0
    assert final["reward_components/tracking_weighted_mean_per_env_step"] == pytest.approx(0.4)
    assert final["reward_components/termination_weighted_mean_per_env_step"] == pytest.approx(-0.25)
    assert final["tracking_error/position_mean_env_max_per_step"] == pytest.approx(4.0)
    assert final["termination/failure_fraction_per_completed_episode"] == pytest.approx(1.0)
    assert np.isfinite(list(final.values())).all()

    assert (Path(tmp_path) / "policy_4.safetensors").exists()
    assert (Path(tmp_path) / "state_4.safetensors").exists()
    assert (Path(tmp_path) / "policy_8.safetensors").exists()
    assert (Path(tmp_path) / "state_8.safetensors").exists()
    definitions = runner.metric_definitions(env)
    assert "reward/total_mean_per_env_step" in definitions
    assert "sac/critic_loss_mean_per_update" in definitions
    assert "reward_components/tracking_weighted_mean_per_env_step" in definitions
    assert "curriculum/reset_to_first_frame_probability" in definitions
    assert "curriculum/immediate_first_frame_probability" in definitions
    assert "policy/action_abs_mean_per_env_action" in definitions
    assert logger.checkpoints[-1][2]["reset_to_first_frame_probability"] == 1.0
    assert logger.checkpoints[-1][2]["immediate_first_frame_probability"] == 0.5


@pytest.mark.parametrize("upload_training_state", [False, True])
def test_wandb_logger_uploads_learner_state_only_when_requested(monkeypatch, tmp_path, upload_training_state):
    _dependencies()

    artifacts = []

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
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **options: run, Artifact=_Artifact))

    from flash_chord.training.flash_sac.checkpoint import CheckpointPaths
    from flash_chord.training.flash_sac.runner import WandbLogger

    logger = WandbLogger(_config(), upload_training_state=upload_training_state, mode="offline")
    paths = CheckpointPaths(
        policy=tmp_path / "policy_4.safetensors",
        training_state=tmp_path / "state_4.safetensors",
        replay=tmp_path / "replay_4.safetensors",
    )
    logger.log_checkpoint(paths, environment_steps=4, metadata={"curriculum_stage": 1, "voc_scale": 0.25})
    logger.close()

    assert len(artifacts) == 1
    artifact, aliases = artifacts[0]
    assert artifact.name == "run-test123-checkpoint"
    assert artifact.metadata == {"environment_steps": 4, "curriculum_stage": 1, "voc_scale": 0.25}
    expected_files = [(str(paths.policy), paths.policy.name)]
    if upload_training_state:
        expected_files += [
            (str(paths.training_state), paths.training_state.name),
            (str(paths.replay), paths.replay.name),
        ]
    assert artifact.files == expected_files
    assert aliases == ["latest", "environment-steps-4"]
    assert run.finished is True


def test_runner_requires_positive_log_interval():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.runner import run_training

    logger = _Logger()
    with pytest.raises(ValueError, match="log_interval"):
        run_training(_Env(jnp), _config(), logger=logger, log_interval=0)
    assert logger.closed


@pytest.mark.parametrize(
    "checkpoint_metadata, message",
    [
        (None, "requires checkpoint metadata"),
        ({"critic_schema_json": _checkpoint_metadata()["critic_schema_json"]}, "policy_schema_json"),
        ({"policy_schema_json": _checkpoint_metadata()["policy_schema_json"]}, "critic_schema_json"),
    ],
)
def test_runner_requires_complete_schemas_before_checkpointing(tmp_path, checkpoint_metadata, message):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.runner import run_training

    with pytest.raises(ValueError, match=message):
        run_training(
            _Env(jnp),
            _config(),
            checkpoint_dir=tmp_path,
            checkpoint_metadata=checkpoint_metadata,
        )


@pytest.mark.parametrize(
    "checkpoint_metadata, message",
    [
        (None, "requires checkpoint metadata"),
        ({"critic_schema_json": _checkpoint_metadata()["critic_schema_json"]}, "policy_schema_json"),
        ({"policy_schema_json": _checkpoint_metadata()["policy_schema_json"]}, "critic_schema_json"),
    ],
)
def test_runner_requires_complete_schemas_before_resume(checkpoint_metadata, message):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.runner import run_training

    config = replace(_config(), resume=True, checkpoint_path="missing-state.safetensors")
    with pytest.raises(ValueError, match=message):
        run_training(_Env(jnp), config, checkpoint_metadata=checkpoint_metadata)


@pytest.mark.parametrize(
    "saved_metadata, message",
    [
        ({"critic_schema_json": _checkpoint_metadata()["critic_schema_json"]}, "policy_schema_json"),
        ({"policy_schema_json": _checkpoint_metadata()["policy_schema_json"]}, "critic_schema_json"),
    ],
)
def test_resume_schema_validation_requires_complete_saved_schemas(saved_metadata, message):
    from flash_chord.training.checkpoint_metadata import checkpoint_critic_schema, checkpoint_policy_schema
    from flash_chord.training.flash_sac.runner import _validate_resume_schemas

    current = _checkpoint_metadata()
    current_policy = checkpoint_policy_schema(current)
    current_critic = checkpoint_critic_schema(current)
    assert current_policy is not None
    assert current_critic is not None

    with pytest.raises(ValueError, match=message):
        _validate_resume_schemas(saved_metadata, current_policy, current_critic)


def test_runner_uses_compiled_window_without_legacy_boundary_reductions():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.runner import run_training

    env = _DeferredMetricEnv(jnp)
    logger = _Logger()
    learner = run_training(env, _config(), logger=logger, log_interval=2)

    assert int(learner.environment_steps) == 8
    assert [step for step, _ in logger.records] == [4, 8]
    assert logger.records[-1][1]["reward/total_mean_per_env_step"] == pytest.approx(0.15)


def test_jax_device_resolution_matches_warp_style_names():
    jax, _ = _dependencies()

    from flash_chord.training.flash_sac.runner import resolve_jax_device

    assert resolve_jax_device("cuda:0") == jax.devices("gpu")[0]
    assert resolve_jax_device("gpu:0") == jax.devices("gpu")[0]
    assert resolve_jax_device("cpu") == jax.devices("cpu")[0]
    with pytest.raises(ValueError, match="unsupported JAX device"):
        resolve_jax_device("tpu:0")
    with pytest.raises(ValueError, match="cannot select"):
        resolve_jax_device("cuda:999")


def test_runner_rejects_reordered_critic_context_before_compilation():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.runner import run_training

    env = _Env(jnp)
    env.critic_context_names = (
        "target_voc_scale",
        "applied_voc_scale",
        "normalized_settling_progress",
    )

    with pytest.raises(ValueError, match="critic context must begin"):
        run_training(env, _config())


def test_runner_accepts_named_critic_context_extensions():
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.runner import run_training

    class ExpandedContextEnv(_Env):
        critic_context_names = _Env.critic_context_names + (
            "normalized_reference_phase",
            "object_body_0_linear_velocity_w_x",
            "object_body_0_angular_velocity_w_x",
        )
        critic_context_dim = len(critic_context_names)

        def _expand_context(self, context):
            extra = self.jnp.full((2, 3), float(self.step_count), dtype=self.jnp.float32)
            return self.jnp.concatenate((context, extra), axis=-1)

        def reset(self):
            observation = super().reset()
            self.initial_critic_context = self._expand_context(self.initial_critic_context)
            return observation

        def step(self, action):
            step = super().step(action)
            return step.replace(
                critic_context=self._expand_context(step.critic_context),
                terminal_critic_context=self._expand_context(step.terminal_critic_context),
            )

    learner = run_training(ExpandedContextEnv(jnp), _config())

    assert learner.critic_context.shape == (2, 6)
    assert learner.replay.storage.critic_context.shape == (8, 6)


def test_runner_resumes_compact_state_and_validates_saved_schemas(tmp_path):
    _, jnp = _dependencies()

    from flash_chord.training.flash_sac.runner import run_training

    first_metadata = _checkpoint_metadata(friction=0.1)
    current_metadata = _checkpoint_metadata(friction=0.1)
    first_config = replace(_config(), total_environment_steps=4)
    first = run_training(
        _Env(jnp),
        first_config,
        checkpoint_dir=tmp_path / "first",
        checkpoint_metadata=first_metadata,
    )
    assert int(first.environment_steps) == 4
    assert int(first.global_update_step) == 2

    resumed_config = replace(
        _config(),
        resume=True,
        checkpoint_path=str(tmp_path / "first" / "state_4.safetensors"),
    )
    resumed_env = _Env(jnp)
    logger = _Logger()
    resumed = run_training(
        resumed_env,
        resumed_config,
        logger=logger,
        log_interval=1,
        checkpoint_dir=tmp_path / "resumed",
        checkpoint_metadata=current_metadata,
    )

    assert int(resumed.environment_steps) == 8
    assert int(resumed.global_update_step) == 4
    assert resumed_env.stage_history[-1] == 0.25
    assert [step for step, _ in logger.records] == [6, 8]
    assert (tmp_path / "resumed" / "state_8.safetensors").exists()

    with pytest.raises(ValueError, match="saved=.*raw.*current=.*normalized"):
        run_training(
            _Env(jnp),
            resumed_config,
            checkpoint_dir=tmp_path / "normalized",
            checkpoint_metadata=_checkpoint_metadata(input_mode="normalized"),
        )
    with pytest.raises(ValueError, match="object articulation config does not match"):
        run_training(
            _Env(jnp),
            resumed_config,
            checkpoint_metadata=_checkpoint_metadata(friction=0.0),
        )
