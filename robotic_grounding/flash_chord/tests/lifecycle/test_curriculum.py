# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fixed-schedule VOC curriculum."""

import pytest

from flash_chord.lifecycle.curriculum import (
    CurriculumStage,
    FixedCurriculum,
    VOCCurriculum,
    reference_curriculum,
)
from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES


def _zero_weights():
    return dict.fromkeys(OBJECTIVE_TERM_NAMES, 0.0)


def test_scale_steps_through_schedule():
    c = VOCCurriculum(schedule=((1000, 0.5), (2000, 0.2), (3000, 0.0)), initial_scale=1.0)
    assert c.scale_at(0) == 1.0  # before the first threshold -> initial
    assert c.scale_at(999) == 1.0
    assert c.scale_at(1000) == 0.5  # at a threshold -> switch
    assert c.scale_at(1999) == 0.5
    assert c.scale_at(2000) == 0.2
    assert c.scale_at(3000) == 0.0
    assert c.scale_at(10**9) == 0.0  # past the last threshold -> clamped


def test_default_initial_scale_is_one():
    c = VOCCurriculum(schedule=((500, 0.3),))
    assert c.scale_at(0) == 1.0
    assert c.scale_at(500) == 0.3


def test_rejects_non_increasing_thresholds():
    with pytest.raises(ValueError):
        VOCCurriculum(schedule=((2000, 0.5), (1000, 0.2)))
    with pytest.raises(ValueError):
        VOCCurriculum(schedule=((1000, 0.5), (1000, 0.2)))  # duplicate threshold


@pytest.mark.parametrize(
    "curriculum",
    (
        lambda: VOCCurriculum(schedule=((-1, 0.5),)),
        lambda: VOCCurriculum(schedule=((1, -0.5),)),
        lambda: VOCCurriculum(schedule=((1, float("nan")),)),
        lambda: VOCCurriculum(schedule=(), initial_scale=float("inf")),
        lambda: CurriculumStage(float("nan"), _zero_weights()),
    ),
)
def test_voc_scales_must_be_finite_and_non_negative(curriculum):
    with pytest.raises(ValueError, match="non-negative"):
        curriculum()


def test_fixed_curriculum_uses_reference_threshold_semantics():
    weights = _zero_weights()
    curriculum = FixedCurriculum(
        thresholds=(10, 20, 30),
        stages=(
            CurriculumStage(1.0, weights),
            CurriculumStage(0.5, weights),
            CurriculumStage(0.0, weights),
        ),
    )

    assert curriculum.stage_index(0) == 0
    assert curriculum.stage_index(9) == 0
    assert curriculum.stage_index(10) == 1
    assert curriculum.stage_index(20) == 2
    assert curriculum.stage_index(10**9) == 2


@pytest.mark.parametrize("probability", (0.0, 0.25, 1.0))
def test_curriculum_stage_accepts_first_frame_reset_probability(probability):
    stage = CurriculumStage(
        1.0,
        _zero_weights(),
        reset_to_first_frame_probability=probability,
    )

    assert stage.reset_to_first_frame_probability == probability


@pytest.mark.parametrize("probability", (0.0, 0.25, 1.0))
def test_curriculum_stage_accepts_immediate_first_frame_probability(probability):
    stage = CurriculumStage(
        1.0,
        _zero_weights(),
        immediate_first_frame_probability=probability,
    )

    assert stage.immediate_first_frame_probability == probability


@pytest.mark.parametrize("probability", (-0.01, 1.01, float("nan"), float("inf")))
def test_curriculum_stage_rejects_invalid_first_frame_reset_probability(probability):
    with pytest.raises(ValueError, match="reset_to_first_frame_probability"):
        CurriculumStage(
            1.0,
            _zero_weights(),
            reset_to_first_frame_probability=probability,
        )


@pytest.mark.parametrize("probability", (-0.01, 1.01, float("nan"), float("inf")))
def test_curriculum_stage_rejects_invalid_immediate_first_frame_probability(probability):
    with pytest.raises(ValueError, match="immediate_first_frame_probability"):
        CurriculumStage(
            1.0,
            _zero_weights(),
            immediate_first_frame_probability=probability,
        )


def test_reference_curriculum_matches_wandb_run():
    curriculum = reference_curriculum()

    assert curriculum.thresholds == (2000, 3500, 5000, 6500, 8000, 9500, 11000, 12500, 14000, 15500)
    assert tuple(stage.voc_scale for stage in curriculum.stages) == (
        1.0,
        0.75,
        0.5,
        0.25,
        0.1,
        0.05,
        0.025,
        0.01,
        0.0,
        0.0,
    )
    assert tuple(stage.objective_weights["object_keypoints"] for stage in curriculum.stages) == (
        0.0,
        0.1,
        0.25,
        0.25,
        0.5,
        0.5,
        1.0,
        1.0,
        1.0,
        20.0,
    )
    assert all(stage.reset_to_first_frame_probability is None for stage in curriculum.stages)
    assert curriculum.stages[0].objective_weights == {
        "object_keypoints": 0.0,
        "hand_keypoints": 0.25,
        "hand_joint_pos": 0.25,
        "contact_wrench_support": 10.0,
        "missed_contact": -1.0,
        "unintended_contact": -10.0,
        "termination": -100.0,
        "action_rate_l2": -0.005,
        "action_l2": -0.002,
        "contact_force_l2": 0.0,
        "force_closure": 0.0,
    }


def test_fixed_curriculum_validates_named_weights_and_schedule():
    stage = CurriculumStage(1.0, {"known": 1.0})
    with pytest.raises(ValueError, match="missing=.*other"):
        stage.weights_for(("known", "other"))
    with pytest.raises(ValueError, match="unexpected=.*known"):
        stage.weights_for(("other",))
    with pytest.raises(ValueError, match="finite"):
        CurriculumStage(1.0, {"known": float("nan")})
    with pytest.raises(ValueError, match="equal length"):
        FixedCurriculum(thresholds=(10,), stages=(CurriculumStage(1.0, _zero_weights()),) * 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        FixedCurriculum(
            thresholds=(10, 10),
            stages=(CurriculumStage(1.0, _zero_weights()),) * 2,
        )
