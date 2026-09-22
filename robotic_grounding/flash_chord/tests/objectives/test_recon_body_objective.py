# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic BodyRecon objective and force-closure checks."""

import numpy as np


def test_recon_body_objective_terms_match_analytic_values():
    import warp as wp

    from flash_chord.objectives.recon_body import compose_recon_body_objective

    with wp.ScopedDevice("cpu"):
        reference_joint_q = np.zeros((11, 10), dtype=np.float32)
        reference_joint_q[:, 6] = 1.0
        wrist_position = np.tile(np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (11, 1, 1))
        wrist_quat = np.zeros((11, 2, 4), dtype=np.float32)
        wrist_quat[..., 3] = 1.0
        terms = wp.zeros(12, dtype=wp.float32)
        score = wp.zeros(1, dtype=wp.float32)
        wp.launch(
            compose_recon_body_objective,
            dim=1,
            inputs=[
                wp.array(
                    [
                        wp.transform_identity(),
                        wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
                        wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity()),
                        wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
                    ],
                    dtype=wp.transform,
                ),
                wp.zeros(10, dtype=wp.float32),
                wp.array([5], dtype=wp.int32),
                wp.zeros(1, dtype=wp.int32),
                wp.zeros(1, dtype=wp.int32),
                wp.array([3.0], dtype=wp.float32),
                wp.array([2.0], dtype=wp.float32),
                wp.array([0.25], dtype=wp.float32),
                wp.zeros(10, dtype=wp.float32),
                wp.array(reference_joint_q, dtype=wp.float32),
                wp.array(wrist_position.reshape(-1, 3), dtype=wp.vec3),
                wp.array(wrist_quat.reshape(-1, 4), dtype=wp.quat),
                wp.array([1, 2], dtype=wp.int32),
                wp.zeros(2, dtype=wp.vec3),
                wp.array([wp.quat_identity(), wp.quat_identity()], dtype=wp.quat),
                wp.array([7, 8], dtype=wp.int32),
                wp.array([7], dtype=wp.int32),
                wp.array([-1.0, -1.0], dtype=wp.float32),
                wp.array([1.0, 1.0], dtype=wp.float32),
                3,
                wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3),
                0,
                11,
                4,
                10,
                2,
                1,
                0.3,
                0.4,
                1.0,
                0.2,
                0.2,
                0.4,
                wp.ones(12, dtype=wp.float32),
                0.02,
            ],
            outputs=[terms, score],
        )

    expected = np.asarray([0.0, 3.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.25])
    np.testing.assert_allclose(terms.numpy(), expected, atol=1.0e-6)
    np.testing.assert_allclose(score.numpy(), [expected.sum() * 0.02], atol=1.0e-6)


def test_force_closure_uses_binary_labels_and_maximum_support_across_bodies():
    import warp as wp

    from flash_chord.objectives.recon_body import reduce_recon_body_force_closure

    # W=2, H=2, B=2, K=4. Supports are flattened world/hand/body/basis.
    support = np.zeros((2, 2, 2, 4), dtype=np.float32)
    support[0, 0, 0, :3] = 0.02
    support[0, 1, :, :] = 1.0  # ignored because this hand's label is inactive.
    support[1, 0, 0, :2] = 0.02
    support[1, 1, 1, :] = 0.02
    labels = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    with wp.ScopedDevice("cpu"):
        result = wp.zeros(2, dtype=wp.float32)
        wp.launch(
            reduce_recon_body_force_closure,
            dim=2,
            inputs=[
                wp.array([0, 1], dtype=wp.int32),
                wp.array(labels.reshape(-1), dtype=wp.float32),
                wp.array(support.reshape(-1), dtype=wp.float32),
                2,
                2,
                2,
                4,
                0.01,
            ],
            outputs=[result],
        )

    np.testing.assert_allclose(result.numpy(), [0.75, 0.75], atol=1.0e-6)
