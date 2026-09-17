# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for strict-interior retargeting joint-limit projection."""

from __future__ import annotations

import numpy as np
import pytest
from robotic_grounding.retarget.joint_limits import clamp_position_limits


def test_clamp_position_limits_projects_batches_strictly_inside() -> None:
    values = np.array([[-1.1, 0.2], [0.5, 1.2]])

    actual = clamp_position_limits(
        values,
        lower=np.array([-1.0, 0.0]),
        upper=np.array([1.0, 1.0]),
        margin=0.1,
    )

    np.testing.assert_allclose(actual, [[-0.9, 0.2], [0.5, 0.9]])


def test_clamp_position_limits_preserves_unbounded_sides() -> None:
    actual = clamp_position_limits(
        np.array([-10.0, 10.0]),
        lower=np.array([-np.inf, 0.0]),
        upper=np.array([0.0, np.inf]),
        margin=0.1,
    )

    np.testing.assert_allclose(actual, [-10.0, 10.0])


def test_clamp_position_limits_collapses_narrow_interval_to_midpoint() -> None:
    actual = clamp_position_limits(
        np.array([2.0]),
        lower=np.array([0.0]),
        upper=np.array([0.1]),
        margin=0.2,
    )

    np.testing.assert_allclose(actual, [0.05])


@pytest.mark.parametrize("margin", [-1.0, np.inf, np.nan, True])
def test_clamp_position_limits_rejects_invalid_margin(margin: float) -> None:
    with pytest.raises(ValueError, match="margin"):
        clamp_position_limits(
            np.array([0.0]),
            lower=np.array([-1.0]),
            upper=np.array([1.0]),
            margin=margin,
        )
