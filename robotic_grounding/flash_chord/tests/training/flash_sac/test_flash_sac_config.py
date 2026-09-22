# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for native FlashSAC configuration and derived reference defaults."""

import math

import pytest

from flash_chord.training.flash_sac.config import (
    CheckpointConfig,
    ExplorationConfig,
    NetworkConfig,
    OptimizationConfig,
    ReplayConfig,
    RewardNormalizationConfig,
    TrainingConfig,
)


def test_reference_defaults_match_executable_isaaclab_launcher():
    config = TrainingConfig()

    assert config.seed == 42
    assert config.world_count == 1_024
    assert config.total_environment_steps == 50_000_896
    assert config.updates_per_collection == 2.0
    assert config.discount == 0.99
    assert config.collection_count == 48_829
    assert config.total_optimizer_updates == 97_658
    assert config.optimization.learning_rate_schedule_updates is None
    assert config.learning_rate_schedule_updates == 97_658
    assert config.schedule_steps == (0, 97_658)
    assert config.mixed_precision is True
    assert config.compute_dtype == "float16"

    assert config.network == NetworkConfig()
    assert config.network.actor_blocks == 2
    assert config.network.actor_hidden_dim == 128
    assert config.network.actor_head_mode == "upstream"
    assert config.network.residual_mean_head == "unit_gain"
    assert config.network.initial_normalized_std == 0.1
    assert config.network.critic_blocks == 2
    assert config.network.critic_hidden_dim == 256
    assert config.network.critic_count == 2
    assert config.network.atom_count == 101

    assert config.optimization.actor_update_period == 2
    assert config.optimization.actor_learning_rate_multiplier == 1.0
    assert config.optimization.residual_mean_learning_rate_multiplier == 1.0
    assert config.optimization.target_tau == 0.01
    assert config.optimization.loss_scale_initial == 65_536.0
    assert config.optimization.loss_scale_growth_factor == 2.0
    assert config.optimization.loss_scale_backoff_factor == 0.5
    assert config.optimization.loss_scale_growth_interval == 2_000
    assert config.replay.capacity == 1_000_000
    assert config.replay.minimum_size == 100_000
    assert config.replay.batch_size == 2_048
    assert config.replay.n_step == 3
    assert config.replay.sampling == "uniform"
    assert config.replay.clear_on_stage_change is False
    assert config.replay.observation_storage_dtype == "float32"
    assert config.exploration == ExplorationConfig()
    assert config.exploration.warmup_action == "uniform"
    assert config.reward_normalization.value_support == (-5.0, 5.0)
    assert config.checkpoint.save_replay is False
    assert config.checkpoint.load_replay_path is None


def test_entropy_target_is_derived_from_action_dimension_and_sigma():
    config = ExplorationConfig(target_sigma=0.15)
    action_dim = 56

    expected = 0.5 * action_dim * math.log(2.0 * math.pi * math.e * 0.15**2)
    assert config.target_entropy(action_dim) == pytest.approx(expected)
    assert config.target_entropy(action_dim, 2.0) == pytest.approx(expected + action_dim * math.log(2.0))

    with pytest.raises(ValueError, match="action_dim must be positive"):
        config.target_entropy(0)
    with pytest.raises(ValueError, match="action_scale"):
        config.target_entropy(action_dim, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"actor_blocks": 0}, "dimensions and block counts"),
        ({"critic_hidden_dim": 0}, "dimensions and block counts"),
        ({"critic_count": 3}, "exactly two critics"),
        ({"atom_count": 1}, "at least two"),
        ({"actor_head_mode": "zero"}, "actor_head_mode"),
        ({"residual_mean_head": "zero"}, "residual_mean_head"),
        ({"residual_mean_head": "zero_dense"}, "requires actor_head_mode='residual'"),
        ({"initial_normalized_std": 0.0}, "initial_normalized_std"),
        ({"action_transform": "clip"}, "action_transform"),
        ({"action_scale": 0.0}, "action_scale"),
        (
            {"actor_head_mode": "residual", "initial_normalized_std": math.exp(2.0)},
            "strictly inside",
        ),
        (
            {"actor_head_mode": "residual", "initial_normalized_std": math.exp(-10.0)},
            "strictly inside",
        ),
        ({"log_std_min": 2.0}, "less than log_std_max"),
        ({"log_std_max": math.inf}, "must be finite"),
        ({"batch_norm_update_rate": 0.0}, "must be in"),
        ({"batch_norm_update_rate": 1.1}, "must be in"),
        ({"rms_norm_epsilon": 0.0}, "epsilons must be"),
    ],
)
def test_network_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NetworkConfig(**kwargs)


def test_zero_dense_mean_head_is_available_only_for_residual_actor():
    config = NetworkConfig(actor_head_mode="residual", residual_mean_head="zero_dense")

    assert config.residual_mean_head == "zero_dense"


def test_identity_action_transform_supports_raw_residual_policies():
    config = NetworkConfig(action_transform="identity", action_scale=2.0)

    assert config.action_transform == "identity"
    assert config.action_scale == 2.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"learning_rate_initial": 0.0}, "learning rates"),
        ({"warmup_fraction": 0.5, "decay_fraction": 0.5}, "schedule fractions"),
        ({"decay_fraction": 1.1}, "schedule fractions"),
        ({"adam_beta1": 1.0}, "Adam betas"),
        ({"adam_epsilon": 0.0}, "adam_epsilon"),
        ({"loss_scale_initial": 0.0}, "loss_scale_initial"),
        ({"loss_scale_growth_factor": 1.0}, "loss_scale_growth_factor"),
        ({"loss_scale_backoff_factor": 1.0}, "loss_scale_backoff_factor"),
        ({"loss_scale_growth_interval": 0}, "loss_scale_growth_interval"),
        ({"actor_update_period": 0}, "actor_update_period"),
        ({"actor_learning_rate_multiplier": 0.0}, "actor_learning_rate_multiplier"),
        ({"actor_learning_rate_multiplier": math.nan}, "actor_learning_rate_multiplier"),
        ({"actor_learning_rate_multiplier": math.inf}, "actor_learning_rate_multiplier"),
        ({"residual_mean_learning_rate_multiplier": 0.0}, "residual_mean_learning_rate_multiplier"),
        ({"residual_mean_learning_rate_multiplier": math.nan}, "residual_mean_learning_rate_multiplier"),
        ({"residual_mean_learning_rate_multiplier": math.inf}, "residual_mean_learning_rate_multiplier"),
        ({"behavior_cloning_coefficient": -1.0}, "behavior_cloning_coefficient"),
        ({"target_tau": 0.0}, "target_tau"),
    ],
)
def test_optimization_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OptimizationConfig(**kwargs)


def test_schedule_step_conversion_uses_upstream_integer_truncation():
    config = OptimizationConfig(warmup_fraction=0.1, decay_fraction=0.9)

    assert config.schedule_steps(11) == (1, 9)
    with pytest.raises(ValueError, match="must be positive"):
        config.schedule_steps(0)
    with pytest.raises(ValueError, match="non-positive decay span"):
        OptimizationConfig(warmup_fraction=0.4, decay_fraction=0.6).schedule_steps(1)


def test_learning_rate_schedule_horizon_can_be_shorter_or_longer_than_run():
    short = TrainingConfig(optimization=OptimizationConfig(learning_rate_schedule_updates=50_000))
    long = TrainingConfig(optimization=OptimizationConfig(learning_rate_schedule_updates=150_000))

    assert short.total_optimizer_updates == long.total_optimizer_updates == 97_658
    assert short.learning_rate_schedule_updates == 50_000
    assert short.schedule_steps == (0, 50_000)
    assert long.learning_rate_schedule_updates == 150_000
    assert long.schedule_steps == (0, 150_000)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_learning_rate_schedule_horizon_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="learning_rate_schedule_updates"):
        OptimizationConfig(learning_rate_schedule_updates=value)


@pytest.mark.parametrize("multiplier", [0.1, 2.0])
def test_residual_mean_learning_rate_multiplier_requires_residual_actor_head(multiplier):
    optimization = OptimizationConfig(residual_mean_learning_rate_multiplier=multiplier)

    with pytest.raises(ValueError, match="requires network.actor_head_mode='residual'"):
        TrainingConfig(optimization=optimization)

    config = TrainingConfig(
        network=NetworkConfig(actor_head_mode="residual"),
        optimization=optimization,
    )
    assert config.optimization.residual_mean_learning_rate_multiplier == multiplier


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"capacity": 0}, "capacity must be positive"),
        ({"capacity": 8, "minimum_size": 0}, "minimum_size"),
        ({"capacity": 8, "minimum_size": 9}, "minimum_size"),
        ({"capacity": 8, "minimum_size": 4, "batch_size": 0}, "batch_size"),
        ({"capacity": 8, "minimum_size": 4, "batch_size": 9}, "batch_size"),
        ({"n_step": 0}, "n_step"),
        ({"sampling": "prioritized"}, "unsupported replay sampling"),
        ({"observation_storage_dtype": "bfloat16"}, "observation_storage_dtype"),
    ],
)
def test_replay_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ReplayConfig(**kwargs)


@pytest.mark.parametrize(
    ("config_type", "kwargs", "match"),
    [
        (ExplorationConfig, {"target_sigma": 0.0}, "exploration values"),
        (ExplorationConfig, {"initial_temperature": -1.0}, "exploration values"),
        (ExplorationConfig, {"zeta_exponent": math.nan}, "exploration values"),
        (ExplorationConfig, {"maximum_noise_repeat": 0}, "maximum_noise_repeat"),
        (ExplorationConfig, {"warmup_action": "zero"}, "warmup_action"),
        (RewardNormalizationConfig, {"normalized_return_bound": 0.0}, "reward-normalization values"),
        (RewardNormalizationConfig, {"epsilon": math.inf}, "reward-normalization values"),
        (CheckpointConfig, {"save_interval_environment_steps": 0}, "must be positive"),
    ],
)
def test_component_configs_reject_invalid_values(config_type, kwargs, match):
    with pytest.raises(ValueError, match=match):
        config_type(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"world_count": 0}, "world_count"),
        ({"total_environment_steps": 0}, "total_environment_steps must be positive"),
        ({"world_count": 3}, "must be divisible"),
        ({"updates_per_collection": 0.0}, "updates_per_collection"),
        ({"discount": 0.0}, "discount"),
        ({"compute_dtype": "float64"}, "compute_dtype"),
        ({"resume": True}, "resume requires"),
        ({"world_count": 2_000_000, "total_environment_steps": 2_000_000}, "must hold at least one"),
        ({"total_environment_steps": 50_176}, "cannot reach replay minimum"),
    ],
)
def test_training_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TrainingConfig(**kwargs)


def test_configs_are_hashable_when_curriculum_is_not_attached():
    assert isinstance(hash(NetworkConfig()), int)
    assert isinstance(hash(ReplayConfig()), int)
    assert isinstance(hash(TrainingConfig()), int)
