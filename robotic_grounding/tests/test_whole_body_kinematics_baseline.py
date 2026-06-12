# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical-regression test for the unified ``WholeBodyKinematics``.

This is V2 of the unify-WholeBodyKinematics refactor: load the fixture
captured at V0 by ``tests/fixtures/regenerate_whole_body_kin_baseline.py``
against the pre-refactor class, replay the same synthetic SOMA input
through the post-refactor unified ``WholeBodyKinematics``, and assert
per-frame ``q``, ``frame_pose``, ``frame_task_errors``, and
``num_optimization_iterations`` match within a tight tolerance.

Tolerance: ``atol=1e-10`` for the float arrays and EXACT equality for
the iteration counts. The merge is a structural flattening — no
floating math should change. A failure here means the merged
``__init__`` ordering or a method body has drifted; loosening the
tolerance is the wrong fix.

Usage (pytest):
    pytest tests/test_whole_body_kinematics_baseline.py -x -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from robotic_grounding.retarget.robot_config import load_robot_config
from robotic_grounding.retarget.whole_body_kinematics import WholeBodyKinematics

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "whole_body_kin_baseline.npz"


def test_whole_body_kinematics_matches_baseline() -> None:
    """Replay V0 fixture through new WholeBodyKinematics; expect identical output."""
    if not FIXTURE_PATH.is_file():
        pytest.skip(
            f"Baseline fixture not found at {FIXTURE_PATH}. Regenerate via "
            f"`python tests/fixtures/regenerate_whole_body_kin_baseline.py` "
            f"BEFORE the WholeBodyKinematics merge lands."
        )

    fixture = np.load(FIXTURE_PATH, allow_pickle=False)
    positions = fixture["positions"]
    wxyz = fixture["wxyz"]
    q_baseline = fixture["q"]
    frame_pose_baseline = fixture["frame_pose"]
    frame_task_errors_baseline = fixture["frame_task_errors"]
    num_iters_baseline = fixture["num_iters"]
    scale = float(fixture["scale"])
    n_frames = positions.shape[0]

    config = load_robot_config("g1")
    kin = WholeBodyKinematics(config=config)

    q = kin.robot.q0.copy()
    for t in range(n_frames):
        result = kin.compute(
            source_joints=positions[t],
            source_joints_wxyz=wxyz[t],
            source_to_robot_scale=scale,
            qpos=q,
        )
        q_new = result["q"]
        frame_pose_new = np.asarray(result["frame_pose"])
        frame_task_errors_new = np.asarray(
            result["frame_task_errors"], dtype=np.float64
        )
        num_iters_new = int(result["num_optimization_iterations"])

        q_diff = float(np.abs(q_new - q_baseline[t]).max())
        assert np.allclose(
            q_new, q_baseline[t], atol=1e-10, rtol=0.0
        ), f"frame {t}: q drift max|diff|={q_diff:.3e} > 1e-10"

        pose_diff = float(np.abs(frame_pose_new - frame_pose_baseline[t]).max())
        assert np.allclose(
            frame_pose_new, frame_pose_baseline[t], atol=1e-10, rtol=0.0
        ), f"frame {t}: frame_pose drift max|diff|={pose_diff:.3e} > 1e-10"

        err_diff = float(
            np.abs(frame_task_errors_new - frame_task_errors_baseline[t]).max()
        )
        assert np.allclose(
            frame_task_errors_new,
            frame_task_errors_baseline[t],
            atol=1e-12,
            rtol=0.0,
        ), f"frame {t}: frame_task_errors drift max|diff|={err_diff:.3e} > 1e-12"

        assert num_iters_new == int(num_iters_baseline[t]), (
            f"frame {t}: num_optimization_iterations diverged "
            f"({num_iters_new} vs baseline {int(num_iters_baseline[t])}); the "
            f"IK convergence path is not identical."
        )

        q = result["q"].copy()


if __name__ == "__main__":
    test_whole_body_kinematics_matches_baseline()
    print("OK: WholeBodyKinematics matches V0 baseline within tolerance.")
