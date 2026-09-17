# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for profile-driven lift-and-hold evaluation."""

import numpy as np
import pytest

from groot_finetune.closed_loop.evaluation import (
    LiftHoldTracker,
    evaluate_lift_hold,
)
from groot_finetune.task_profile import LiftHoldEvaluator

CONFIG = LiftHoldEvaluator(
    lift_threshold_m=0.10,
    hold_threshold_m=0.05,
    min_hold_steps=3,
)


def test_lift_hold_success() -> None:
    result = evaluate_lift_hold(
        np.asarray([0.20, 0.26, 0.31, 0.30, 0.29]), config=CONFIG
    )
    assert result.success
    assert result.max_lift_m == pytest.approx(0.11)
    assert result.max_hold_steps == 4
    assert result.sample_count == 4


def test_hold_must_be_consecutive() -> None:
    result = evaluate_lift_hold(
        np.asarray([0.20, 0.31, 0.20, 0.31, 0.20, 0.31]), config=CONFIG
    )
    assert not result.success
    assert result.max_lift_m == pytest.approx(0.11)
    assert result.max_hold_steps == 1


def test_parallel_completion_resets_only_selected_environment() -> None:
    tracker = LiftHoldTracker(np.asarray([0.1, 0.2]), config=CONFIG)
    for sample in ([0.21, 0.20], [0.21, 0.26], [0.21, 0.26]):
        tracker.update(np.asarray(sample))
    first = tracker.complete(0, next_initial_z=0.4)
    assert first.success
    tracker.update(np.asarray([0.4, 0.26]))
    second = tracker.complete(1, next_initial_z=0.5)
    assert second.max_hold_steps == 3


@pytest.mark.parametrize(
    "trace",
    [np.asarray([]), np.asarray([[0.0]]), np.asarray([0.0, np.nan])],
)
def test_invalid_trace_rejected(trace: np.ndarray) -> None:
    with pytest.raises(ValueError):
        evaluate_lift_hold(trace, config=CONFIG)
