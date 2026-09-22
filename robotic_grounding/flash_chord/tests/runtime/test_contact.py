# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for world-frame contact readout."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

G = 9.81


def _build_box_on_ground():
    import newton

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2

    body = builder.add_body(mass=0.5, label="box")
    builder.add_joint_free(child=body, label="box_free")
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, label="box_geom")
    q = np.array(builder.joint_q, dtype=float)
    q[:3] = [0.0, 0.0, 0.06]
    q[3:7] = [0.0, 0.0, 0.0, 1.0]
    builder.joint_q = q.tolist()

    builder.add_ground_plane()
    builder.request_contact_attributes("force")
    return builder.finalize()


def test_resting_box_contact_force_equals_weight():
    """A box resting on the ground: contacts on the box/world, vertical normals at
    z~0, and total contact force = the box's weight m*g (closed-form)."""
    import warp as wp

    import newton
    from flash_chord.runtime.contact import read_contacts_w

    with wp.ScopedDevice("cuda:0"):
        model = _build_box_on_ground()
        solver = newton.solvers.SolverMuJoCo(
            model, solver="newton", integrator="implicitfast",
            njmax=50, nconmax=50, iterations=20, ls_iterations=10, use_mujoco_contacts=True,
        )
        s0, s1 = model.state(), model.state()
        control = model.control()
        contacts = model.contacts()
        dt = 1.0 / 100.0 / 5.0
        for _ in range(200):
            model.collide(s0, contacts)
            for _ in range(5):
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, dt)
                s0, s1 = s1, s0
        solver.update_contacts(contacts, s0)  # populate contacts.force

        c = read_contacts_w(model, s0, contacts)
        mass = float(model.body_mass.numpy()[0])

    assert len(c) >= 1
    # all contacts are between the box (body 0) and the world/ground (body -1)
    assert set(c.body0_ids.tolist()) | set(c.body1_ids.tolist()) <= {-1, 0}
    # normals vertical; contact points lie on the ground plane (z ~ 0)
    assert np.allclose(np.abs(c.contact_normal_w[:, 2]), 1.0, atol=1e-3)
    assert np.allclose(c.contact_pos_w[:, 2], 0.0, atol=0.01)
    # total contact force supports the weight, purely vertical
    total = c.contact_force_w.sum(axis=0)
    assert abs(abs(total[2]) - mass * G) < 0.5
    assert np.linalg.norm(total[:2]) < 0.5
