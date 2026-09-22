# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for hand keypoint + finger-joint tracking objectives."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_hand_keypoints_objective():
    """Track a rotated semantic fingertip offset rather than only its retained body origin."""
    import warp as wp

    from flash_chord.objectives.keypoints import hand_keypoints_objective
    from flash_chord.objectives.shaping import ObjectiveShape

    var = 0.1
    with wp.ScopedDevice("cuda:0"):
        body_q = wp.array(
            [
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(
                    wp.vec3(0.1, 0.0, 0.0),
                    wp.quat_rpy(0.0, 0.0, np.pi / 2.0),
                ),
            ],
            dtype=wp.transform,
        )
        kp_ids = wp.array([0, 1], dtype=wp.int32)
        local_pos = wp.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=wp.vec3)
        target = wp.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]], dtype=wp.vec3)
        rew = wp.zeros(1, dtype=wp.float32)
        wp.launch(
            hand_keypoints_objective,
            dim=2,
            inputs=[
                body_q,
                kp_ids,
                local_pos,
                target,
                2,
                1,
                2,
                var,
                int(ObjectiveShape.GAUSSIAN),
                0.0,
            ],
            outputs=[rew],
        )
        per_hand = float(rew.numpy()[0]) / 2.0  # divide by n_kp

    expected = (1.0 + np.exp(-(0.05**2) / var)) / 2.0  # wrist err 0 -> 1; fingertip err 0.05^2
    assert abs(per_hand - expected) < 1e-5


def test_hand_joint_objective():
    """One hand, 3 fingers, one joint off by 0.1: objective = shaped(0.01)."""
    import warp as wp

    from flash_chord.objectives.joints import hand_joint_objective
    from flash_chord.objectives.shaping import ObjectiveShape

    with wp.ScopedDevice("cuda:0"):
        joint_q = wp.array([0.1, 0.2, 0.0], dtype=wp.float32)
        finger_q_ids = wp.array([0, 1, 2], dtype=wp.int32)
        target_q = wp.array([0.0, 0.2, 0.0], dtype=wp.float32)  # only joint 0 off, by 0.1 -> sse 0.01

        def run(kind, var):
            rew = wp.zeros(1, dtype=wp.float32)
            wp.launch(
                hand_joint_objective,
                dim=1,
                inputs=[joint_q, finger_q_ids, target_q, 3, 1, 3, var, int(kind), 0.0],
                outputs=[rew],
            )
            return float(rew.numpy()[0])

        g = run(ObjectiveShape.GAUSSIAN, 1.0)
        lap = run(ObjectiveShape.LAPLACIAN, 1.0)

    assert abs(g - np.exp(-0.01 / 1.0)) < 1e-6  # Gaussian: exp(-sse/var)
    assert abs(lap - np.exp(-np.sqrt(0.01) / 1.0)) < 1e-6  # Laplacian: exp(-sqrt(sse)/var)
