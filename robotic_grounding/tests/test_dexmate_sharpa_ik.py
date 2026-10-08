# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Vega IK stopping tolerances with unchanged posture regularization."""

import inspect

import numpy as np
import pytest
from scripts.retarget.mano_to_dexmate_sharpa import DexmateSharpaIK


def test_default_tolerances() -> None:
    ik = DexmateSharpaIK()
    assert ik.position_tolerance == pytest.approx(0.002)
    assert ik.orientation_tolerance == pytest.approx(0.01)
    assert inspect.signature(DexmateSharpaIK).parameters["posture_cost"].default == 0.01


@pytest.mark.parametrize(
    ("position_error", "orientation_error", "expected"),
    [(0.002, 0.01, True), (0.002001, 0.01, False), (0.002, 0.010001, False)],
)
def test_convergence_thresholds(position_error, orientation_error, expected) -> None:
    ik = DexmateSharpaIK()
    errors = {name: np.zeros(6) for name in ik.task_frame_names}
    errors["left_hand_C_MC"][[0, 3]] = [position_error, orientation_error]
    assert all(
        b.compute_barrier(ik.configuration)[0] >= 0 for b in ik.collision_barriers
    )
    assert ik._converged(errors) is expected


@pytest.mark.parametrize("start_at_target", [False, True])
@pytest.mark.parametrize("strict_tolerances", [False, True])
@pytest.mark.parametrize("offset", [0.15, 1.0])
def test_reachable_target_convergence(
    start_at_target: bool,
    strict_tolerances: bool,
    offset: float,
) -> None:
    # The new tolerances accept the 1-rad reproducer's residual; the original
    # strict tolerances still saturate. The posture objective is unchanged.
    expected_convergence = not (strict_tolerances and offset == 1.0)
    ik = (
        DexmateSharpaIK(position_tolerance=5e-4, orientation_tolerance=2e-3)
        if strict_tolerances
        else DexmateSharpaIK()
    )
    reference = ik._clamp_q_to_limits(ik.q_reference)
    target_q = reference.copy()
    for name in ("L_arm_j4", "R_arm_j4"):
        joint = ik.robot.model.joints[ik.robot.model.getJointId(name)]
        target_q[joint.idx_q] += offset
    assert np.all(target_q >= ik.robot.model.lowerPositionLimit)
    assert np.all(target_q <= ik.robot.model.upperPositionLimit)
    targets = ik.fk(target_q)
    ik.configuration.update(target_q)
    assert all(
        b.compute_barrier(ik.configuration)[0] >= 0 for b in ik.collision_barriers
    )

    result = ik.solve(targets, q_init=target_q if start_at_target else reference)

    assert result.converged is expected_convergence
    if expected_convergence:
        assert result.iterations < ik.max_iters
    else:
        assert result.iterations == ik.max_iters
    assert np.all(result.q >= ik.robot.model.lowerPositionLimit)
    assert np.all(result.q <= ik.robot.model.upperPositionLimit)
    assert all(
        b.compute_barrier(ik.configuration)[0] >= 0 for b in ik.collision_barriers
    )
    within_tolerances = all(
        np.linalg.norm(error[:3]) <= ik.position_tolerance
        and (
            ik._frame_task_specs[name].orientation_cost == 0
            or np.linalg.norm(error[3:]) <= ik.orientation_tolerance
        )
        for name, error in result.frame_errors.items()
    )
    assert within_tolerances is expected_convergence
