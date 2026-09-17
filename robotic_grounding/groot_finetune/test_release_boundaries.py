# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release-boundary regressions for the GR00T post-training workflow."""

from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
EXPORTER = REPOSITORY / "robotic_grounding/scripts/rsl_rl/export_parallel_rollouts.py"
MANIP_CFG = REPOSITORY / (
    "robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/"
    "v2d_whole_body/config/vega_sharpa/vega_sharpa_manip_env_cfg.py"
)
GROOT_CFG = REPOSITORY / (
    "robotic_grounding/source/robotic_grounding/robotic_grounding/tasks/"
    "v2d_whole_body/config/vega_sharpa/vega_sharpa_gr00t_env_cfg.py"
)
EXPERT_STAGE_START = "def train_expert_stages("
EXPERT_STAGE_END = "def collection_env_count("


def test_exporter_configures_events_before_environment_creation() -> None:
    """Keep reset randomization declarative and construction-time only."""
    source = EXPORTER.read_text(encoding="utf-8")
    assert source.index("configure_timeout_only_terminations(") < source.index(
        "gym.make(args_cli.task"
    )
    assert source.index("_configure_reset_noise(env_cfg, contract)") < source.index(
        "gym.make(args_cli.task"
    )
    assert "write_joint_state_to_sim" not in source
    assert "write_root_pose_to_sim" not in source
    assert "_reset_idx" not in source
    for hardcoded_slice in ("_RIGHT_ARM", "_LEFT_ARM", "_RIGHT_FINGER", "_LEFT_FINGER"):
        assert hardcoded_slice not in source


def test_reset_noise_defaults_are_off_and_not_in_recording_cfg() -> None:
    """Keep the shared manipulation reset exact unless collection opts in."""
    manip_source = MANIP_CFG.read_text(encoding="utf-8")
    assert manip_source.count('"range": (0.0, 0.0)') == 2
    assert '"object_xy_noise_range": (0.0, 0.0)' in manip_source
    assert '"object_yaw_noise_range": (0.0, 0.0)' in manip_source

    recording_source = GROOT_CFG.read_text(encoding="utf-8")
    assert "joint_position_noise_groups" not in recording_source
    assert "object_xy_noise_range" not in recording_source
    assert "object_yaw_noise_range" not in recording_source


def test_task_example_is_not_embedded_in_runtime_scripts() -> None:
    """Keep task-specific text in explicit task profiles only."""
    paths = (
        EXPORTER,
        REPOSITORY / "robotic_grounding/scripts/rsl_rl/replay_record.py",
        REPOSITORY / "robotic_grounding/scripts/rsl_rl/gr00t_infer.py",
        REPOSITORY / "robotic_grounding/groot_finetune/convert_to_gr00t.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8").lower()
        assert "tissue" not in source
        assert "lift the box" not in source


def test_expert_training_stage_does_not_depend_on_post_training_contracts() -> None:
    """Keep RL expert training isolated from GR00T collection contracts."""
    stages = (REPOSITORY / "tools/e2e/stages.py").read_text(encoding="utf-8")
    expert = stages[stages.index(EXPERT_STAGE_START) : stages.index(EXPERT_STAGE_END)]
    assert "scripts/rsl_rl/train.py" in expert
    assert "--contract" not in expert
    assert "--task-profile" not in expert
    assert "groot_finetune" not in expert
    assert "--zero-actor" not in expert
    assert "agent.num_steps_per_env" not in expert
    assert "agent.save_interval" not in expert
    assert "env.rewards." not in expert
