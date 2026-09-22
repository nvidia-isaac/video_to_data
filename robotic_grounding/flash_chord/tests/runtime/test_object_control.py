# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Virtual Object Controller."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _command(
    *,
    world_count,
    num_joint_q,
    num_joint_dof,
    bodies_per_world=0,
    body_ids=(),
    object_body_offsets=None,
    object_body_ids=None,
    target_pos=(),
    target_quat=(),
    articulation_dof_ids=(),
    articulation_target_pos=(),
):
    import warp as wp

    num_objects = len(body_ids) // world_count
    num_articulations = len(articulation_dof_ids)
    if object_body_offsets is None:
        object_body_offsets = tuple(range(num_objects + 1))
    if object_body_ids is None:
        object_body_ids = tuple(body_ids[:num_objects])
    assert len(body_ids) == world_count * num_objects
    assert len(articulation_target_pos) == world_count * num_articulations
    return SimpleNamespace(
        world_count=world_count,
        num_joint_q=num_joint_q,
        num_joint_dof=num_joint_dof,
        bodies_per_world=bodies_per_world,
        layout=SimpleNamespace(num_objects=num_objects, num_articulations=num_articulations),
        voc_body_ids_w=wp.array(body_ids, dtype=wp.int32),
        voc_object_body_offsets=wp.array(object_body_offsets, dtype=wp.int32),
        voc_object_body_ids=wp.array(object_body_ids, dtype=wp.int32),
        voc_target_pos_w=wp.array(target_pos, dtype=wp.vec3),
        voc_target_quat_w=wp.array(target_quat, dtype=wp.quat),
        articulation_dof_ids=wp.array(articulation_dof_ids, dtype=wp.int32),
        articulation_target_pos=wp.array(articulation_target_pos, dtype=wp.float32),
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"k_lin": -1.0},
        {"d_ang": float("nan")},
        {"max_force": float("inf")},
        {"gravity": -9.81},
    ),
)
def test_voc_parameters_must_be_finite_and_non_negative(kwargs):
    from flash_chord.runtime.object_control import VOCParams

    with pytest.raises(ValueError, match="finite and non-negative"):
        VOCParams(**kwargs)


def test_voc_holds_object_at_target_against_gravity():
    """A free box, initialized above a target, is pulled to and held at the target by the VOC
    (gravity compensation + PD) instead of falling to the ground."""
    import warp as wp

    import newton
    from flash_chord.runtime.object_control import apply_object_control

    target_z = 0.5
    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        body = builder.add_body(mass=0.5, label="obj")
        builder.add_joint_free(child=body)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        q = np.array(builder.joint_q, dtype=float)
        q[:3] = [0.0, 0.0, target_z + 0.05]  # start above target
        q[3:7] = [0.0, 0.0, 0.0, 1.0]
        builder.joint_q = q.tolist()
        builder.add_ground_plane()
        model = builder.finalize()

        solver = newton.solvers.SolverMuJoCo(
            model,
            solver="newton",
            integrator="implicitfast",
            njmax=50,
            nconmax=50,
            iterations=20,
            ls_iterations=10,
            use_mujoco_contacts=False,
        )
        s0, s1 = model.state(), model.state()
        control, contacts = model.control(), model.contacts()
        target_pos = wp.array([wp.vec3(0.0, 0.0, target_z)], dtype=wp.vec3)
        target_quat = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat)
        command = _command(
            world_count=1,
            num_joint_q=model.joint_coord_count,
            num_joint_dof=model.joint_dof_count,
            bodies_per_world=model.body_count,
            body_ids=(0,),
            target_pos=target_pos.numpy(),
            target_quat=target_quat.numpy(),
        )
        scale = wp.array([1.0], dtype=wp.float32)

        dt = 1.0 / 100.0 / 5.0
        for _ in range(150):
            model.collide(s0, contacts)
            for _ in range(5):
                s0.clear_forces()
                apply_object_control(model, s0, control, command, scale=scale)
                solver.step(s0, s1, control, contacts, dt)
                s0, s1 = s1, s0
        pos = s0.body_q.numpy()[0][:3]

    assert abs(pos[2] - target_z) < 0.03  # held at the target height (would be ~0.05 if it fell)
    assert np.linalg.norm(pos[:2]) < 0.02


def test_voc_applies_each_world_scale_to_all_object_roots():
    import warp as wp

    import newton
    from flash_chord.runtime.object_control import apply_object_control

    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        bodies = [builder.add_body(mass=0.5, label=f"obj_{index}") for index in range(4)]
        for body in bodies:
            builder.add_joint_free(child=body)
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize()
        state = model.state()
        control = model.control()
        command = _command(
            world_count=2,
            num_joint_q=model.joint_coord_count // 2,
            num_joint_dof=model.joint_dof_count // 2,
            bodies_per_world=model.body_count // 2,
            body_ids=tuple(bodies),
            target_pos=(wp.vec3(0.0, 0.0, 0.5),) * 4,
            target_quat=(wp.quat_identity(),) * 4,
        )
        state.clear_forces()
        apply_object_control(
            model,
            state,
            control,
            command,
            scale=wp.array([1.0, 0.0], dtype=wp.float32),
        )
        force = state.body_f.numpy()

    assert np.linalg.norm(force[0]) > 1.0
    assert np.linalg.norm(force[1]) > 1.0
    np.testing.assert_allclose(force[2:], 0.0)


def test_voc_uses_each_objects_total_mass_and_root_origin_velocity():
    import warp as wp

    from flash_chord.runtime.object_control import VOCParams, apply_object_control

    with wp.ScopedDevice("cuda:0"):
        model = SimpleNamespace(
            body_mass=wp.array([0.25, 0.75, 2.0], dtype=wp.float32),
            body_com=wp.array(
                [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
                dtype=wp.vec3,
            ),
            body_inertia=wp.zeros(3, dtype=wp.mat33),
        )
        state = SimpleNamespace(
            body_q=wp.array(
                [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())] * 3,
                dtype=wp.transform,
            ),
            body_qd=wp.array(
                [
                    wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                ],
                dtype=wp.spatial_vector,
            ),
            joint_q=wp.zeros(0, dtype=wp.float32),
            joint_qd=wp.zeros(0, dtype=wp.float32),
            body_f=wp.zeros(3, dtype=wp.spatial_vector),
        )
        control = SimpleNamespace(
            joint_target_pos=wp.zeros(0, dtype=wp.float32),
            joint_target_vel=wp.zeros(0, dtype=wp.float32),
        )
        command = _command(
            world_count=1,
            num_joint_q=0,
            num_joint_dof=0,
            bodies_per_world=3,
            body_ids=(0, 2),
            object_body_offsets=(0, 2, 3),
            object_body_ids=(0, 1, 2),
            target_pos=(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)),
            target_quat=(wp.quat_identity(), wp.quat_identity()),
        )
        apply_object_control(
            model,
            state,
            control,
            command,
            scale=wp.ones(1, dtype=wp.float32),
            params=VOCParams(k_lin=0.0, d_lin=10.0, k_ang=0.0, d_ang=0.0),
        )
        force = state.body_f.numpy()[:, :3]

    np.testing.assert_allclose(force[0], [0.0, 0.0, 9.81], atol=1.0e-5)
    np.testing.assert_allclose(force[1], 0.0)
    np.testing.assert_allclose(force[2], [0.0, 0.0, 19.62], atol=1.0e-5)


def test_voc_direct_gain_mode_matches_force_torque_controller_units():
    import warp as wp

    from flash_chord.runtime.object_control import VOCParams, apply_object_control

    with wp.ScopedDevice("cuda:0"):
        model = SimpleNamespace(
            body_mass=wp.array([2.0], dtype=wp.float32),
            body_com=wp.zeros(1, dtype=wp.vec3),
            body_inertia=wp.array([wp.mat33(3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0)], dtype=wp.mat33),
        )
        state = SimpleNamespace(
            body_q=wp.array(
                [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
                dtype=wp.transform,
            ),
            body_qd=wp.zeros(1, dtype=wp.spatial_vector),
            body_f=wp.zeros(1, dtype=wp.spatial_vector),
        )
        control = SimpleNamespace(
            joint_target_pos=wp.zeros(0, dtype=wp.float32),
            joint_target_vel=wp.zeros(0, dtype=wp.float32),
        )
        command = _command(
            world_count=1,
            num_joint_q=0,
            num_joint_dof=0,
            bodies_per_world=1,
            body_ids=(0,),
            target_pos=(wp.vec3(0.5, 0.0, 0.0),),
            target_quat=(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.2),),
        )
        apply_object_control(
            model,
            state,
            control,
            command,
            scale=wp.ones(1, dtype=wp.float32),
            params=VOCParams(
                k_lin=50.0,
                d_lin=10.0,
                k_ang=10.0,
                d_ang=0.1,
                max_force=60.0,
                max_torque=60.0,
                gravity=9.81,
                mass_normalized_gains=False,
            ),
        )
        wrench = state.body_f.numpy()[0]

    np.testing.assert_allclose(wrench[:3], [25.0, 0.0, 19.62], atol=1.0e-5)
    np.testing.assert_allclose(wrench[3:], [0.0, 0.0, 2.0], atol=1.0e-5)


def test_articulation_voc_writes_targets_and_updates_a_captured_graph():
    import warp as wp

    from flash_chord.runtime.object_control import apply_object_control

    world_count = 4
    num_joint_dof = 3
    targets = np.asarray((1.0, -1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0), dtype=np.float32)

    with wp.ScopedDevice("cuda:0"):
        model = SimpleNamespace(
            body_mass=wp.zeros(0, dtype=wp.float32),
            body_com=wp.zeros(0, dtype=wp.vec3),
            body_inertia=wp.zeros(0, dtype=wp.mat33),
        )
        state = SimpleNamespace(
            body_q=wp.zeros(0, dtype=wp.transform),
            body_qd=wp.zeros(0, dtype=wp.spatial_vector),
            body_f=wp.zeros(0, dtype=wp.spatial_vector),
        )
        control = SimpleNamespace(
            joint_target_pos=wp.full(world_count * num_joint_dof, value=7.0, dtype=wp.float32),
            joint_target_vel=wp.full(world_count * num_joint_dof, value=-7.0, dtype=wp.float32),
        )
        command = _command(
            world_count=world_count,
            num_joint_q=0,
            num_joint_dof=num_joint_dof,
            bodies_per_world=0,
            articulation_dof_ids=(1, 2),
            articulation_target_pos=targets,
        )
        scale = wp.array([0.0, 0.25, 1.0, 1.0], dtype=wp.float32)

        apply_object_control(model, state, control, command, scale=scale)
        target_pos = control.joint_target_pos.numpy().reshape(world_count, num_joint_dof)
        target_vel = control.joint_target_vel.numpy().reshape(world_count, num_joint_dof)

        with wp.ScopedCapture("cuda:0") as capture:
            apply_object_control(model, state, control, command, scale=scale)
        updated_targets = targets + 0.5
        command.articulation_target_pos.assign(updated_targets)
        wp.capture_launch(capture.graph)
        captured_pos = control.joint_target_pos.numpy().reshape(world_count, num_joint_dof)
        captured_vel = control.joint_target_vel.numpy().reshape(world_count, num_joint_dof)

    np.testing.assert_array_equal(target_pos[:, 0], 7.0)
    np.testing.assert_array_equal(target_pos[:, 1:], targets.reshape(world_count, 2))
    np.testing.assert_array_equal(target_vel[:, 0], -7.0)
    np.testing.assert_array_equal(target_vel[:, 1:], 0.0)
    np.testing.assert_array_equal(captured_pos[:, 0], 7.0)
    np.testing.assert_array_equal(captured_pos[:, 1:], updated_targets.reshape(world_count, 2))
    np.testing.assert_array_equal(captured_vel[:, 0], -7.0)
    np.testing.assert_array_equal(captured_vel[:, 1:], 0.0)
