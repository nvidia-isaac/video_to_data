# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the wrist-orientation effort kernel."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_wrist_effort_proportional_to_orientation_error():
    """Current orientation = identity, target = 90 deg about +z: the ball effort should be
    kp * rotvec(error) = kp * (0, 0, pi/2), clamped to tau_max."""
    import warp as wp

    from flash_chord.runtime.control import control_wrist_orientation

    kp, tau_max = 25.0, 45.0
    # one world, one wrist: ball quat at coord 0..3 (identity xyzw), ball rotvec target at dof 0..2
    with wp.ScopedDevice("cuda:0"):
        joint_q = wp.array([0.0, 0.0, 0.0, 1.0], dtype=wp.float32)  # identity xyzw
        targets = wp.array([0.0, 0.0, np.pi / 2], dtype=wp.float32)  # 90 deg about +z
        joint_f = wp.zeros(3, dtype=wp.float32)
        wp.launch(
            control_wrist_orientation,
            dim=1,
            inputs=[
                joint_q, targets, 4, 3, 1,
                wp.array([0], dtype=wp.int32), wp.array([0], dtype=wp.int32),
                wp.array([kp], dtype=wp.float32), wp.array([tau_max], dtype=wp.float32),
            ],
            outputs=[joint_f],
        )
        tau = joint_f.numpy()

    assert np.allclose(tau[:2], 0.0, atol=1e-4)  # no x/y torque
    assert abs(tau[2] - kp * (np.pi / 2)) < 1e-3  # z torque = kp * (pi/2), under the clamp
