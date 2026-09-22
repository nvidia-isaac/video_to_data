# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the wrench-space support function."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _support(contact_pos, contact_normal, basis, mu, rc, num_edges=8, enabled=None):
    import warp as wp

    from flash_chord.objectives.wrench import friction_cone_angles, wrench_support

    cos_t, sin_t = friction_cone_angles(num_edges)
    nb = len(basis)
    enabled = np.ones(len(contact_pos), dtype=np.int32) if enabled is None else np.asarray(enabled, dtype=np.int32)
    with wp.ScopedDevice("cuda:0"):
        cp = wp.array(np.asarray(contact_pos, dtype=np.float32), dtype=wp.vec3)
        cn = wp.array(np.asarray(contact_normal, dtype=np.float32), dtype=wp.vec3)
        b = wp.array(np.asarray(basis, dtype=np.float32), dtype=wp.spatial_vector)
        out = wp.zeros(nb, dtype=wp.float32)
        wp.launch(
            wrench_support,
            dim=nb,
            inputs=[
                cp,
                cn,
                wp.array(enabled, dtype=wp.int32),
                b,
                wp.array(cos_t, dtype=wp.float32),
                wp.array(sin_t, dtype=wp.float32),
                float(mu),
                wp.array(np.asarray(rc, dtype=np.float32), dtype=wp.float32),
                len(contact_pos),
                num_edges,
                nb,
            ],
            outputs=[out],
        )
        return out.numpy()


def test_frictionless_normal_only_support():
    # 1 body, 1 contact at origin, normal +z, mu=0 -> only force (0,0,1), zero torque (p=0).
    sup = _support(
        contact_pos=[[0.0, 0.0, 0.0]],
        contact_normal=[[0.0, 0.0, 1.0]],
        basis=[[0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0], [1, 0, 0, 0, 0, 0]],
        mu=0.0,
        rc=[1.0],
    )
    assert np.allclose(sup, [1.0, 0.0, 0.0], atol=1e-5)  # +z force dir supported=1; -z and x clamped to 0


def test_torque_support_with_lever_arm():
    # contact at (1,0,0), normal +z, mu=0, rc=2 -> f=(0,0,1), torque=(p×f)/rc=(0,-1,0)/2=(0,-0.5,0).
    sup = _support(
        contact_pos=[[1.0, 0.0, 0.0]],
        contact_normal=[[0.0, 0.0, 1.0]],
        basis=[[0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 1, 0]],
        mu=0.0,
        rc=[2.0],
    )
    assert np.allclose(sup, [0.5, 0.0], atol=1e-5)


def test_inactive_contact_gives_zero_support():
    sup = _support([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0, 0, 1, 0, 0, 0]], mu=0.0, rc=[1.0])
    assert np.allclose(sup, [0.0], atol=1e-6)  # zero normal => inactive => no support


def test_batched_support_uses_each_hands_contact_range():
    import warp as wp

    from flash_chord.objectives.wrench import batched_wrench_support, friction_cone_angles

    basis = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    cos_t, sin_t = friction_cone_angles(4)
    with wp.ScopedDevice("cuda:0"):
        support = wp.zeros(6, dtype=wp.float32)
        wp.launch(
            batched_wrench_support,
            dim=6,
            inputs=[
                wp.zeros(3, dtype=wp.vec3),
                wp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3),
                wp.ones(3, dtype=wp.int32),
                wp.array([0, 1], dtype=wp.int32),
                wp.array([1, 2], dtype=wp.int32),
                3,
                2,
                1,
                wp.array(basis, dtype=wp.spatial_vector),
                wp.array(cos_t, dtype=wp.float32),
                wp.array(sin_t, dtype=wp.float32),
                0.0,
                wp.array([1.0], dtype=wp.float32),
                4,
                3,
            ],
            outputs=[support],
        )

    np.testing.assert_allclose(support.numpy(), [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1.0e-6)
