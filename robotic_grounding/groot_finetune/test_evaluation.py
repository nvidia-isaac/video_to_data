# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for profile-driven lift-and-hold evaluation."""

from dataclasses import replace

import numpy as np
import pytest

from groot_finetune.closed_loop.evaluation import (
    LiftHoldTracker,
    evaluate_lift_hold,
)
from groot_finetune.task_profile import LiftHoldEvaluator, load_task_profile

CONFIG = LiftHoldEvaluator(
    lift_threshold_m=0.10,
    hold_threshold_m=0.05,
    min_hold_steps=3,
)


def complete_trace() -> np.ndarray:
    return np.r_[0.0, [0.11] * 3, np.linspace(0.105, 0.0, 22), np.zeros(20)]


def test_lift_hold_success() -> None:
    trace = complete_trace() + 0.2
    result = evaluate_lift_hold(trace, config=CONFIG)
    assert result.success
    assert result.max_lift_m == pytest.approx(0.11)
    assert result.lift_hold_success
    assert result.sample_count == len(trace) - 1


@pytest.mark.parametrize(
    "trace",
    [
        np.r_[0.0, np.full(40, 0.11)],  # Never placed.
        np.r_[complete_trace()[:-20], np.tile([0.0, 0.006], 10)],  # Jitter.
        np.r_[complete_trace()[:-20], np.linspace(0.0, 0.02, 20)],  # Drift.
        np.r_[0.0, [0.11] * 3, np.zeros(20)],  # Abrupt drop, then stable.
        np.r_[0.0, [0.11] * 3],  # Incomplete placement window.
        np.r_[
            complete_trace()[:-20],
            np.linspace(0.0, -0.02, 5),
            np.full(20, -0.02),
        ],  # Below existing final-height floor.
    ],
)
def test_lift_hold_alone_is_not_success(trace: np.ndarray) -> None:
    result = evaluate_lift_hold(trace, config=CONFIG)
    assert result.lift_hold_success
    assert not result.success


def test_placement_thresholds_round_trip_and_affect_success() -> None:
    profile = load_task_profile(
        "groot_finetune/task_profiles/tissue_box_lift_hold.json"
    )
    custom = replace(CONFIG, placement_threshold_m=0.015, placement_window_steps=5)
    profile = replace(profile, evaluator=custom)
    assert load_task_profile(profile.as_dict()) == profile
    trace = np.r_[complete_trace()[:-20], np.full(20, 0.02)]
    assert evaluate_lift_hold(trace, config=CONFIG).success
    assert not evaluate_lift_hold(trace, config=custom).success
    legacy = profile.as_dict()
    for key in (
        "placement_threshold_m",
        "placement_window_steps",
        "max_descent_step_m",
        "max_terminal_step_m",
        "max_terminal_range_m",
    ):
        del legacy["evaluator"][key]
    assert load_task_profile(legacy).evaluator == CONFIG


@pytest.mark.parametrize(
    "field",
    [
        "placement_threshold_m",
        "max_descent_step_m",
        "max_terminal_step_m",
        "max_terminal_range_m",
    ],
)
@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_invalid_placement_thresholds_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        replace(CONFIG, **{field: value})


def test_invalid_placement_window_and_floor_rejected() -> None:
    with pytest.raises(ValueError, match="placement_window_steps"):
        replace(CONFIG, placement_window_steps=1)
    with pytest.raises(ValueError, match="final_min_lift_m"):
        replace(CONFIG, final_min_lift_m=0.04)


def test_hold_must_be_consecutive() -> None:
    result = evaluate_lift_hold(
        np.asarray([0.20, 0.31, 0.20, 0.31, 0.20, 0.31]), config=CONFIG
    )
    assert not result.success
    assert result.max_lift_m == pytest.approx(0.11)
    assert result.max_hold_steps == 1


def test_parallel_completion_resets_only_selected_environment() -> None:
    tracker = LiftHoldTracker(np.asarray([0.1, 0.2]), config=CONFIG)
    for height in complete_trace()[1:]:
        tracker.update(np.asarray([0.1 + height, 0.2 + height]))
    first = tracker.complete(0, next_initial_z=0.4)
    assert first.success
    tracker.update(np.asarray([0.4, 0.2]))
    second = tracker.complete(1, next_initial_z=0.5)
    assert second.success
    assert second.max_hold_steps == first.max_hold_steps
    assert not tracker.complete(0, next_initial_z=0.4).success
    # A drop in one episode must not poison the next episode's descent/window.
    for height in [0.11, 0.11, 0.11, 0.0]:
        tracker.update(np.asarray([0.4 + height, 0.5]))
    assert not tracker.complete(0, next_initial_z=0.4).success
    for height in complete_trace()[1:]:
        tracker.update(np.asarray([0.4 + height, 0.5]))
    assert tracker.complete(0, next_initial_z=0.4).success


@pytest.mark.parametrize(
    "trace",
    [np.asarray([]), np.asarray([[0.0]]), np.asarray([0.0, np.nan])],
)
def test_invalid_trace_rejected(trace: np.ndarray) -> None:
    with pytest.raises(ValueError):
        evaluate_lift_hold(trace, config=CONFIG)
