# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release-recipe and current-schema configuration tests."""

from pathlib import Path

import pytest
from omegaconf import MissingMandatoryValue

from flash_chord.configuration import compose_config, instantiate_typed, resolve_path
from flash_chord.evaluation.view import PolicyViewConfig
from flash_chord.training.flash_sac.config import TrainingConfig as FlashSACTrainingConfig
from flash_chord.training.ppo.config import TrainingConfig as PPOTrainingConfig
from flash_chord.visualization.viewer import ViewerConfig

_PARQUET = "task.parquet=/tmp/reference.parquet"


@pytest.mark.parametrize(
    ("root", "experiment", "training_type", "world_count", "input_mode", "mapping"),
    (
        ("train", "sharpa_ppo", PPOTrainingConfig, 4096, "raw", "linear"),
        ("train_flash_sac", "sharpa_flash_sac", FlashSACTrainingConfig, 4096, "normalized", "rational"),
        (
            "train_flash_sac",
            "dexmate_sharpa_flash_sac",
            FlashSACTrainingConfig,
            2048,
            "normalized",
            "rational",
        ),
        ("train", "g1_recon_body_ppo", PPOTrainingConfig, 4096, "raw", "linear"),
        ("train_flash_sac", "g1_recon_body_flash_sac", FlashSACTrainingConfig, 2048, "raw", "linear"),
    ),
)
def test_public_training_recipes_compose(
    root,
    experiment,
    training_type,
    world_count,
    input_mode,
    mapping,
):
    config = compose_config(root, [f"experiment={experiment}", _PARQUET])
    training = instantiate_typed(config.training, training_type)

    assert config.task.parquet == "/tmp/reference.parquet"
    assert config.scene.world_count == world_count
    assert training.world_count == world_count
    assert config.action.input_mode == input_mode
    assert config.action.normalized_mapping == mapping
    assert config.logging.run_name == experiment


def test_sharpa_flash_sac_uses_validated_final_mixed_reset_curriculum():
    config = compose_config("train_flash_sac", ["experiment=sharpa_flash_sac", _PARQUET])
    training = instantiate_typed(config.training, FlashSACTrainingConfig)
    curriculum = training.curriculum

    assert training.total_environment_steps == 250_003_456
    assert curriculum.thresholds[-2:] == (200_007_680, 250_003_456)
    assert curriculum.stage_at(200_007_679).reset_to_first_frame_probability is None
    assert curriculum.stage_at(200_007_680).reset_to_first_frame_probability == 0.5
    assert curriculum.stage_at(250_003_455).reset_to_first_frame_probability == 0.5


def test_dexmate_recipe_retains_derived_controller_without_a_versioned_public_config():
    config = compose_config("train_flash_sac", ["experiment=dexmate_sharpa_flash_sac", _PARQUET])
    training = instantiate_typed(config.training, FlashSACTrainingConfig)

    assert config.embodiment.config.asset_id == "vega_sharpa_v2"
    assert config.embodiment.config.urdf_path._args_[0].endswith("v2/vega_sharpa_58dof.urdf")
    assert config.task.motion_speed == 0.5
    assert config.action.delay.min_steps == config.action.delay.max_steps == 2
    assert config.action.arm_scale == 0.075
    assert config.action.finger_scale == 0.15
    assert config.action.final_target.mode == "ema"
    assert config.action.final_target.arm_ema == config.action.final_target.finger_ema == 0.75
    assert config.objective.action_rate_l2.weight == -0.05
    assert config.objective.contact_wrench_support.weight == 20.0
    assert config.objective.contact_force_l2.mode == "per_hand_log"
    assert config.collision.robot_support_collision is True
    assert config.env.arm_position_integral.gain_s_inv2 == 0.02
    assert config.env.arm_position_integral.max_effort_fraction == 0.15
    assert training.replay.capacity == 10_000_000
    assert training.replay.observation_storage_dtype == "float16"
    assert training.curriculum.thresholds[-2:] == (200_007_680, 250_009_600)
    assert training.curriculum.stage_at(200_007_680).reset_to_first_frame_probability == 0.5


def test_object_scale_randomization_remains_publicly_configurable():
    config = compose_config(
        "train_flash_sac",
        [
            "experiment=sharpa_flash_sac",
            _PARQUET,
            "scene.object_scale_min=0.8",
            "scene.object_scale_max=1.2",
            "scene.object_scale_seed=7",
        ],
    )

    assert (config.scene.object_scale_min, config.scene.object_scale_max, config.scene.object_scale_seed) == (
        0.8,
        1.2,
        7,
    )


def test_force_closure_reward_configs_remain_available():
    sharpa = compose_config("train_flash_sac", ["experiment=sharpa_flash_sac", _PARQUET])
    g1 = compose_config("train", ["experiment=g1_recon_body_ppo", _PARQUET])

    assert sharpa.objective.force_closure._target_.endswith("ForceClosureObjectiveTermConfig")
    assert g1.objective.force_closure_weight == 5.0
    assert g1.objective.force_closure_min_support == 0.01


def test_task_parquet_is_required_in_the_release_config():
    config = compose_config("train_flash_sac", ["experiment=sharpa_flash_sac"])
    with pytest.raises(MissingMandatoryValue):
        _ = config.task.parquet


def test_partitioned_parquet_path_composes_when_value_is_quoted():
    parquet = "/data/sequence_id=clip/robot_name=g1/data.parquet"
    config = compose_config("train", ["experiment=g1_recon_body_ppo", f"task.parquet='{parquet}'"])

    assert config.task.parquet == parquet


def test_viewer_config_supports_only_release_backends():
    for backend in ("gl", "mp4", "viser", "null", "none"):
        output = "policy.mp4" if backend == "mp4" else "output.mp4"
        assert ViewerConfig(backend=backend, output_path=output).backend == backend


def test_policy_viewer_uses_algorithm_neutral_checkpoint_config():
    config = compose_config("view_policy", ["evaluation.checkpoint=/tmp/policy.safetensors"])
    evaluation = instantiate_typed(config.evaluation, PolicyViewConfig)

    assert evaluation.checkpoint == "/tmp/policy.safetensors"
    assert evaluation.deterministic is True
    assert evaluation.reset_mode == "explicit"


def test_resolve_path_uses_invocation_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("checkpoint.ckpt") == Path(tmp_path, "checkpoint.ckpt")
