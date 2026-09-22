# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reference-matched RL training configuration."""

from dataclasses import replace

import pytest

from flash_chord.lifecycle.curriculum import reference_curriculum
from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig


def test_training_defaults_match_reference_run():
    config = TrainingConfig()

    assert config.seed == 42
    assert config.world_count == 4096
    assert config.rollout_steps == 24
    assert config.max_iterations == 20_000
    assert config.save_interval == 200
    assert config.observation_normalization is True
    assert config.observation_normalization_epsilon == 1.0e-2
    assert config.reward_normalization is False
    assert config.device == "cuda:0"
    assert config.resume is False
    assert config.checkpoint is None
    assert config.wandb_entity == "nvidia-isaac"
    assert config.wandb_project == "v2d-flash-chord"
    assert config.experiment_name == "sharpa"
    assert config.run_name == "ppo"
    assert config.curriculum == reference_curriculum()

    assert config.network == NetworkConfig(
        actor_hidden_dims=(1024, 512, 256, 128),
        critic_hidden_dims=(1024, 512, 256, 128),
        activation="elu",
        initial_noise_std=0.1,
        noise_std_type="scalar",
    )
    assert config.ppo == PPOConfig(
        value_loss_coefficient=1.0,
        use_clipped_value_loss=True,
        clip_parameter=0.1,
        entropy_coefficient=0.001,
        learning_epochs=5,
        mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        minimum_learning_rate=1.0e-5,
        maximum_learning_rate=1.0e-2,
        adaptive_schedule_factor=1.5,
        discount=0.99,
        gae_lambda=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        log_ratio_clip=None,
    )


def test_training_config_validates_batch_and_resume():
    with pytest.raises(ValueError, match="rollout batch"):
        TrainingConfig(world_count=3, rollout_steps=3)

    with pytest.raises(ValueError, match="resume requires"):
        TrainingConfig(resume=True)

    resumed = TrainingConfig(resume=True, checkpoint="checkpoint.safetensors")
    assert resumed.checkpoint == "checkpoint.safetensors"
    assert replace(resumed, resume=False).resume is False


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: NetworkConfig(actor_hidden_dims=(0,)), "actor_hidden_dims"),
        (lambda: NetworkConfig(actor_output_gain=-0.1), "actor_output_gain"),
        (lambda: PPOConfig(clip_parameter=0.0), "clip_parameter"),
        (lambda: PPOConfig(schedule="cosine"), "schedule"),
        (lambda: PPOConfig(minimum_learning_rate=2.0e-3), "learning rates"),
        (lambda: PPOConfig(adaptive_schedule_factor=1.0), "adaptive_schedule_factor"),
        (lambda: PPOConfig(discount=1.1), "discount"),
        (lambda: PPOConfig(log_ratio_clip=0.0), "log_ratio_clip"),
        (lambda: PPOConfig(log_ratio_clip=float("nan")), "log_ratio_clip"),
        (lambda: TrainingConfig(observation_normalization_epsilon=0.0), "observation_normalization_epsilon"),
    ],
)
def test_nested_config_validation(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
