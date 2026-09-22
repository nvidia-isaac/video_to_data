# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic evaluation-composition tests for task-specific termination layouts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("non_finite_worlds", [(1,), (0, 1)])
def test_rollout_mppe_excludes_invalid_worlds(non_finite_worlds):
    from flash_chord.evaluation.harness import score_rollout
    from flash_chord.evaluation.metrics import TRACKING_ERROR_NAMES, TrackingThresholds

    reference = np.zeros((2, 1, 7))
    reference[..., 0] = 1.0
    reference[..., 6] = 1.0
    achieved = np.repeat(reference[:, None], 2, axis=1)
    achieved[:, 0, :, 0] += 0.02
    achieved[:, list(non_finite_worlds)] = 0.0  # Recorder's invalid-world placeholders.
    rollout = SimpleNamespace(
        achieved_pose_w=achieved,
        reference_pose_w=reference,
        object_vertices_o=np.zeros((1, 1, 3)),
        body_object_ids=np.array([0]),
        tracking_error=np.zeros((2, 2, 4)),
        tracking_error_names=TRACKING_ERROR_NAMES,
        completion=SimpleNamespace(
            terminated=np.zeros(2, dtype=bool),
            truncated=np.ones(2, dtype=bool),
            reference_progress=np.ones(2),
        ),
        object_body_names=("object",),
        non_finite_worlds=non_finite_worlds,
    )
    if len(non_finite_worlds) == 2:
        with pytest.raises(ValueError, match="finite world"):
            score_rollout(rollout, TrackingThresholds())
    else:
        assert score_rollout(rollout, TrackingThresholds()).mppe_cm == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("flash_chord.training.flash_sac.config.TrainingConfig", "flash_sac"),
        ("flash_chord.training.ppo.config.TrainingConfig", "ppo"),
    ],
)
def test_checkpoint_training_algorithm_uses_exact_typed_target(target, expected):
    from flash_chord.evaluation.harness import checkpoint_training_algorithm

    assert checkpoint_training_algorithm({"training": {"_target_": target}}) == expected


@pytest.mark.parametrize(
    ("config", "error_type"),
    [
        ({}, TypeError),
        ({"training": {}}, ValueError),
        ({"training": {"_target_": "third_party.TrainingConfig"}}, ValueError),
    ],
)
def test_checkpoint_training_algorithm_rejects_missing_or_unknown_target(config, error_type):
    from flash_chord.evaluation.harness import checkpoint_training_algorithm

    with pytest.raises(error_type, match="training mapping|training target"):
        checkpoint_training_algorithm(config)


def _tracking_termination():
    return {
        "wrist_position": {"enabled": True, "threshold": 0.15},
        "wrist_orientation": {"enabled": False, "threshold": 0.0},
        "object_position": {"enabled": True, "threshold": 0.10},
        "object_orientation": {"enabled": True, "threshold": 0.50},
        "reference_end": {"enabled": True},
    }


def _recon_body_termination():
    return {
        "pelvis_position_threshold": 0.70,
        "pelvis_orientation_threshold": 1.50,
        "palm_position_threshold": 0.15,
        "palm_orientation_threshold": 1.50,
        "object_position_threshold": 0.10,
        "object_orientation_threshold": 1.50,
        "reset_freeze_steps": 1,
        "truncate_at_reference_end": True,
    }


def _config(termination):
    return {
        "scene": {"world_count": 1},
        "env": {
            "termination": deepcopy(termination),
            "reset": {},
            "voc": {},
        },
        "termination": deepcopy(termination),
    }


@pytest.mark.parametrize("termination", [_tracking_termination(), _recon_body_termination()])
def test_evaluation_composition_disables_failure_without_changing_saved_criteria(termination):
    from flash_chord.evaluation.harness import (
        apply_evaluation_composition,
        reference_tracking_criteria,
    )

    config = _config(termination)
    criteria = reference_tracking_criteria(config)
    apply_evaluation_composition(config, world_count=8)

    assert config["scene"]["world_count"] == 8
    assert config["env"]["auto_reset"] is False
    assert config["env"]["voc_scale"] == 0.0
    if "pelvis_position_threshold" in termination:
        assert criteria.error_names[0] == "pelvis_position_m"
        assert criteria.active_after_step == (-1, -1, 1, 1, 1, 1)
        assert config["env"]["termination"]["pelvis_position_threshold"] > 1_000.0
        assert config["termination"]["object_orientation_threshold"] > 1_000.0
    else:
        assert criteria.thresholds == (0.15, None, 0.10, 0.50)
        assert not config["env"]["termination"]["wrist_position"]["enabled"]
        assert not config["termination"]["object_orientation"]["enabled"]


def test_sampled_settled_composition_preserves_reset_assistance_only_for_preamble():
    from flash_chord.evaluation.harness import apply_evaluation_composition

    config = _config(_recon_body_termination())
    config["env"]["voc"] = {"max_force": 150.0, "max_torque": 10.0}
    config["env"]["reset"]["voc_decay_steps"] = 10

    apply_evaluation_composition(config, world_count=8, reset_mode="sampled_settled")

    assert config["env"]["voc_scale"] == 0.0
    assert config["env"]["voc"] == {"max_force": 150.0, "max_torque": 10.0}
    assert config["env"]["reset"]["voc_decay_steps"] == 10
    assert config["env"]["reset"]["always_reset_to_first_frame"] is True


def test_preamble_shifts_termination_activation_into_scored_step_coordinates():
    from flash_chord.evaluation.harness import _shift_criteria_for_preamble, reference_tracking_criteria

    criteria = reference_tracking_criteria(_config(_recon_body_termination()))

    shifted = _shift_criteria_for_preamble(criteria, settling_steps=1)

    assert shifted.active_after_step == (-1, -1, 0, 0, 0, 0)


def test_sampled_settled_curriculum_disables_immediate_starts_during_preamble():
    from flash_chord.evaluation.harness import _evaluation_curriculum_stage
    from flash_chord.lifecycle.curriculum import CurriculumStage

    stage = CurriculumStage(
        voc_scale=0.25,
        objective_weights={"tracking": 1.0},
        reset_to_first_frame_probability=0.5,
        immediate_first_frame_probability=0.5,
    )

    sampled = _evaluation_curriculum_stage(stage, "sampled_settled")
    explicit = _evaluation_curriculum_stage(stage, "explicit")

    assert sampled.voc_scale == 0.0
    assert sampled.reset_to_first_frame_probability == 0.5
    assert sampled.immediate_first_frame_probability == 0.0
    assert explicit.voc_scale == 0.0
    assert explicit.immediate_first_frame_probability == 0.5


def test_recon_body_reference_success_uses_all_causes_and_reset_freeze():
    from flash_chord.evaluation.harness import reference_tracking_criteria, score_reference_tracking

    criteria = reference_tracking_criteria(_config(_recon_body_termination()))
    errors = np.zeros((4, 3, 6), dtype=np.float64)
    errors[0, 1, 0] = 0.71  # Pelvis is active immediately.
    errors[1, 2, 2] = 0.16  # Palm is still masked through reset_freeze_steps == 1.
    errors[2, 2, 4] = 0.11  # Object is active after the freeze.
    rollout = SimpleNamespace(
        tracking_error=errors,
        tracking_error_names=criteria.error_names,
        non_finite_worlds=(),
        completion=SimpleNamespace(
            terminated=np.zeros(3, dtype=np.bool_),
            truncated=np.ones(3, dtype=np.bool_),
            reference_progress=np.ones(3, dtype=np.float64),
        ),
    )

    metrics = score_reference_tracking(rollout, criteria)

    assert metrics.success_rate == pytest.approx(1.0 / 3.0)
    assert metrics.reference_end_fraction == 1.0
    assert metrics.failure_cause_fraction == pytest.approx({"pelvis": 1.0 / 3.0, "palm": 0.0, "object": 1.0 / 3.0})


def test_reference_success_requires_reaching_reference_end():
    from flash_chord.evaluation.harness import reference_tracking_criteria, score_reference_tracking

    criteria = reference_tracking_criteria(_config(_recon_body_termination()))
    rollout = SimpleNamespace(
        tracking_error=np.zeros((4, 2, 6), dtype=np.float64),
        tracking_error_names=criteria.error_names,
        non_finite_worlds=(),
        completion=SimpleNamespace(
            terminated=np.zeros(2, dtype=np.bool_),
            truncated=np.asarray([True, False]),
            reference_progress=np.asarray([1.0, 0.8]),
        ),
    )

    metrics = score_reference_tracking(rollout, criteria)

    assert metrics.reference_end_fraction == 0.5
    assert metrics.success_rate == 0.5
