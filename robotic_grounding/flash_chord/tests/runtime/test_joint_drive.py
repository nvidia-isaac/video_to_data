# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for per-world implicit joint-drive scaling."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _build(*, target_mode, world_count=1):
    import newton

    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    body = builder.add_link(label="body", mass=1.0)
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Z,
        label="joint",
    )
    builder.add_articulation([joint], label="articulation")
    assert builder.joint_coord_count == builder.joint_dof_count == 1
    builder.joint_q[0] = 0.8
    builder.joint_target_mode[0] = int(target_mode)
    builder.joint_target_ke[0] = 50.0
    builder.joint_target_kd[0] = 2.0
    builder.joint_effort_limit[0] = 50.0

    worlds = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(worlds)
    worlds.replicate(builder, world_count)
    model = worlds.finalize(device="cuda:0")
    solver = newton.solvers.SolverMuJoCo(
        model,
        solver="newton",
        integrator="implicitfast",
        cone="pyramidal",
        njmax=32,
        nconmax=16,
        iterations=100,
        ls_iterations=50,
        disable_contacts=True,
        use_mujoco_contacts=True,
    )
    return model, solver


def test_scaled_joint_target_drive_updates_per_world_and_captured_graph():
    import newton
    import warp as wp

    from flash_chord.runtime.joint_drive import ScaledJointTargetDrive

    with wp.ScopedDevice("cuda:0"):
        _, solver = _build(target_mode=newton.JointTargetMode.POSITION_VELOCITY, world_count=2)
        base_gain = solver.mjw_model.actuator_gainprm.numpy().copy()
        base_bias = solver.mjw_model.actuator_biasprm.numpy().copy()
        drive = ScaledJointTargetDrive.build(solver, (0,), world_count=2)
        position_id = int(drive.position_actuator_ids.numpy()[0])
        velocity_id = int(drive.velocity_actuator_ids.numpy()[0])
        actuator_ids = [position_id, velocity_id]
        scale = wp.array([0.0, 1.0], dtype=wp.float32)

        drive.apply(scale)
        gain = solver.mjw_model.actuator_gainprm.numpy()
        bias = solver.mjw_model.actuator_biasprm.numpy()
        np.testing.assert_array_equal(gain[0, actuator_ids], 0.0)
        np.testing.assert_array_equal(bias[0, actuator_ids], 0.0)
        np.testing.assert_array_equal(gain[1, actuator_ids], base_gain[1, actuator_ids])
        np.testing.assert_array_equal(bias[1, actuator_ids], base_bias[1, actuator_ids])

        with wp.ScopedCapture("cuda:0") as capture:
            drive.apply(scale)
        scale.assign([0.25, 0.5])
        wp.capture_launch(capture.graph)
        gain = solver.mjw_model.actuator_gainprm.numpy()
        bias = solver.mjw_model.actuator_biasprm.numpy()
        np.testing.assert_allclose(gain[0, actuator_ids], 0.25 * base_gain[0, actuator_ids])
        np.testing.assert_allclose(bias[0, actuator_ids], 0.25 * base_bias[0, actuator_ids])
        np.testing.assert_allclose(gain[1, actuator_ids], 0.5 * base_gain[1, actuator_ids])
        np.testing.assert_allclose(bias[1, actuator_ids], 0.5 * base_bias[1, actuator_ids])


def test_scaled_joint_target_drive_does_not_modify_unselected_joint_actuators():
    import newton
    import warp as wp

    from flash_chord.runtime.joint_drive import ScaledJointTargetDrive

    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        for joint in range(2):
            body = builder.add_link(label=f"body_{joint}", mass=1.0)
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            joint_id = builder.add_joint_revolute(
                parent=-1,
                child=body,
                axis=newton.Axis.Z,
                label=f"joint_{joint}",
            )
            builder.add_articulation([joint_id], label=f"articulation_{joint}")
            builder.joint_target_mode[joint] = int(newton.JointTargetMode.POSITION_VELOCITY)
            builder.joint_target_ke[joint] = 10.0 + joint
            builder.joint_target_kd[joint] = 1.0 + joint
            builder.joint_effort_limit[joint] = 20.0
        model = builder.finalize(device="cuda:0")
        solver = newton.solvers.SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_contacts=True,
            njmax=16,
            nconmax=8,
        )
        base_gain = solver.mjw_model.actuator_gainprm.numpy().copy()
        base_bias = solver.mjw_model.actuator_biasprm.numpy().copy()
        mapping = solver.mjc_actuator_to_newton_idx.numpy()
        unselected_ids = np.flatnonzero((mapping == 0) | (mapping == -2))
        selected_ids = np.flatnonzero((mapping == 1) | (mapping == -3))

        drive = ScaledJointTargetDrive.build(solver, (1,), world_count=1)
        drive.apply(wp.array([0.0], dtype=wp.float32))
        gain = solver.mjw_model.actuator_gainprm.numpy()
        bias = solver.mjw_model.actuator_biasprm.numpy()

    np.testing.assert_array_equal(gain[:, unselected_ids], base_gain[:, unselected_ids])
    np.testing.assert_array_equal(bias[:, unselected_ids], base_bias[:, unselected_ids])
    np.testing.assert_array_equal(gain[:, selected_ids], 0.0)
    np.testing.assert_array_equal(bias[:, selected_ids], 0.0)


def test_joint_actuator_limits_clamp_velocity():
    import newton
    import warp as wp

    from flash_chord.runtime.joint_drive import JointActuatorLimits

    with wp.ScopedDevice("cuda:0"):
        model, _ = _build(target_mode=newton.JointTargetMode.POSITION, world_count=2)
        model.joint_velocity_limit.assign([2.0, 2.0])
        state = model.state()
        state.joint_q.assign([0.8, 0.8])
        state.joint_qd.assign([3.0, -4.0])
        limits = JointActuatorLimits.build(model, (0,), (0,), world_count=2)
        limits.enforce_velocity(state)

    np.testing.assert_allclose(state.joint_qd.numpy(), [2.0, -2.0])


def _torque_pulse(*, passive: bool):
    import newton
    import warp as wp

    from flash_chord.runtime.joint_drive import ScaledJointTargetDrive

    target_mode = newton.JointTargetMode.EFFORT if passive else newton.JointTargetMode.POSITION_VELOCITY
    model, solver = _build(target_mode=target_mode)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()
    control.joint_target_pos.assign([0.8])
    control.joint_target_vel.zero_()
    if not passive:
        drive = ScaledJointTargetDrive.build(solver, (0,), world_count=1)
        drive.apply(wp.array([0.0], dtype=wp.float32))

    q = [state_0.joint_q.numpy().copy()]
    qd = [state_0.joint_qd.numpy().copy()]
    actuator_force = []
    for step in range(50):
        state_0.clear_forces()
        control.joint_f.assign([0.005 if step < 5 else 0.0])
        solver.step(state_0, state_1, control, contacts, 0.01)
        actuator_force.append(solver.mjw_data.qfrc_actuator.numpy().copy())
        state_0, state_1 = state_1, state_0
        q.append(state_0.joint_q.numpy().copy())
        qd.append(state_0.joint_qd.numpy().copy())
    return np.asarray(q), np.asarray(qd), np.asarray(actuator_force)


def test_zero_scaled_implicit_drive_exactly_matches_passive_joint():
    import warp as wp

    with wp.ScopedDevice("cuda:0"):
        scaled_q, scaled_qd, scaled_force = _torque_pulse(passive=False)
        passive_q, passive_qd, _ = _torque_pulse(passive=True)

    np.testing.assert_array_equal(scaled_q, passive_q)
    np.testing.assert_array_equal(scaled_qd, passive_qd)
    np.testing.assert_array_equal(scaled_force, 0.0)


def test_full_scaled_implicit_drive_is_bounded_at_100_hz():
    import newton
    import warp as wp

    from flash_chord.runtime.joint_drive import ScaledJointTargetDrive

    with wp.ScopedDevice("cuda:0"):
        model, solver = _build(target_mode=newton.JointTargetMode.POSITION_VELOCITY)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        contacts = model.contacts()
        control.joint_target_pos.assign([0.5])
        control.joint_target_vel.zero_()
        drive = ScaledJointTargetDrive.build(solver, (0,), world_count=1)
        drive.apply(wp.array([1.0], dtype=wp.float32))

        q = []
        qd = []
        for _ in range(100):
            state_0.clear_forces()
            control.joint_f.zero_()
            solver.step(state_0, state_1, control, contacts, 0.01)
            state_0, state_1 = state_1, state_0
            q.append(float(state_0.joint_q.numpy()[0]))
            qd.append(float(state_0.joint_qd.numpy()[0]))

    assert np.isfinite(q).all()
    assert np.isfinite(qd).all()
    assert max(abs(value) for value in qd) < 10.0
    assert q[-1] == pytest.approx(0.5, abs=1.0e-5)
