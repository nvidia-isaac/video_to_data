# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for released embodiment and task contracts."""

from dataclasses import replace

import pytest

from groot_finetune.contracts import (
    SHARPA_DUAL_HAND_THREE_CAMERA,
    VEGA_SHARPA_JOINT,
    load_embodiment_contract,
)
from groot_finetune.task_profile import TargetObject, load_task_profile


def test_vega_contract_has_one_exact_layout() -> None:
    contract = load_embodiment_contract("vega_sharpa_joint")
    assert contract is VEGA_SHARPA_JOINT
    assert contract.source_task == "VegaSharpa-WholeBody-Manip-v0"
    assert contract.modality_config == "groot_finetune/vega_sharpa_joint_config.py"
    assert contract.action_horizon == 16
    assert contract.state_dim == 58
    assert contract.action_dim == 58
    assert contract.source_terminations == ("timeout",)
    assert [field.key for field in contract.action_fields] == [
        "right_arm",
        "right_finger",
        "left_arm",
        "left_finger",
    ]
    assert [camera.key for camera in contract.cameras] == [
        "front",
        "right_wrist_view",
        "left_wrist_view",
    ]
    assert contract.evaluation_terminations == ("timeout", "robot_state_diverged")


def test_contract_hash_changes_with_semantics() -> None:
    assert replace(VEGA_SHARPA_JOINT, fps=10).sha256 != VEGA_SHARPA_JOINT.sha256


def test_contract_requires_complete_release_routing() -> None:
    with pytest.raises(ValueError, match="inference_task"):
        replace(VEGA_SHARPA_JOINT, inference_task="")
    with pytest.raises(ValueError, match="evaluation_terminations"):
        replace(VEGA_SHARPA_JOINT, evaluation_terminations=())


def test_floating_contract_matches_registered_recording_layout() -> None:
    assert [camera.key for camera in SHARPA_DUAL_HAND_THREE_CAMERA.cameras] == [
        "front",
        "right_wrist_view",
        "left_wrist_view",
    ]
    assert SHARPA_DUAL_HAND_THREE_CAMERA.source_terminations == ("time_out",)
    assert SHARPA_DUAL_HAND_THREE_CAMERA.evaluation_terminations == ("time_out",)


def test_example_task_profile_is_explicit() -> None:
    profile = load_task_profile(
        "groot_finetune/task_profiles/tissue_box_lift_hold.json"
    )
    assert profile.task_id == "tissue_box_lift_hold"
    assert profile.evaluator.evaluator_id == "lift_hold"


def test_target_selector_is_strict() -> None:
    assert TargetObject("primary").resolve(("box", "tray")) == 0
    assert TargetObject("name", "tray").resolve(("box", "tray")) == 1
    with pytest.raises(ValueError, match="not present"):
        TargetObject("name", "missing").resolve(("box",))
