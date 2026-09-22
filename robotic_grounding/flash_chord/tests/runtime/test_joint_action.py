# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact tests for reference-relative scalar-joint actions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _frames(side: str) -> dict:
    from flash_chord.embodiments.base import BodyFrame

    return {
        "palm_frame": BodyFrame(f"{side}_palm", 0),
        "dp_frames": (BodyFrame(f"{side}_index_DP", 0),),
        "fingertip_frames": (BodyFrame(f"{side}_index_fingertip", 0),),
    }


def _build_scene(wp, world_count=2, *, lower=None, upper=None):
    import newton

    from flash_chord.embodiments.base import EmbodimentLayout, HandLayout
    from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding

    layout = EmbodimentLayout(
        num_joint_q=6,
        num_joint_dof=6,
        hands=(
            HandLayout(
                side="left",
                **_frames("left"),
                arm_q_ids=(4, 1),
                arm_dof_ids=(0, 5),
                arm_joint_names=("la4", "la1"),
                finger_q_ids=(5,),
                finger_dof_ids=(2,),
                finger_joint_names=("lf",),
            ),
            HandLayout(
                side="right",
                **_frames("right"),
                arm_q_ids=(3, 0),
                arm_dof_ids=(4, 1),
                arm_joint_names=("ra3", "ra0"),
                finger_q_ids=(2,),
                finger_dof_ids=(3,),
                finger_joint_names=("rf",),
            ),
        ),
    )
    joint_q = np.array(
        [[11.0, 15.0, 13.0, 14.0, 10.0, 12.0], [21.0, 25.0, 23.0, 24.0, 20.0, 22.0]],
        dtype=np.float32,
    )
    joint_target = np.array(
        [[10.0, 11.0, 12.0, 13.0, 14.0, 15.0], [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]],
        dtype=np.float32,
    )
    keypoint_position = np.zeros((2, 1, 3), dtype=np.float32)
    keypoint_orientation = np.zeros((2, 1, 4), dtype=np.float32)
    keypoint_orientation[..., 0] = 1.0
    hands = []
    for side in layout.sides:
        hand = layout.hand(side)
        hands.append(
            BoundHandReference(
                side=side,
                wrist_pos_w=np.zeros((2, 3), dtype=np.float32),
                wrist_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32),
                arm_joint_pos=joint_q[:, hand.arm_q_ids],
                finger_joint_pos=joint_q[:, hand.finger_q_ids],
                dp_pos_w=keypoint_position,
                dp_quat_w=keypoint_orientation,
                fingertip_pos_w=keypoint_position,
                fingertip_quat_w=keypoint_orientation,
            )
        )
    reference = RobotReferenceBinding(
        num_frames=2,
        fps=20.0,
        num_joint_q=6,
        num_joint_dof=6,
        joint_q=joint_q,
        joint_target=joint_target,
        hands=tuple(hands),
    )

    values_per_world = 8
    total_dof = world_count * values_per_world
    lower = np.full(total_dof, -100.0, dtype=np.float32) if lower is None else np.asarray(lower, np.float32)
    upper = np.full(total_dof, 100.0, dtype=np.float32) if upper is None else np.asarray(upper, np.float32)
    model = SimpleNamespace(
        body_count=world_count,
        joint_coord_count=total_dof,
        joint_dof_count=total_dof,
        joint_limit_lower=wp.array(lower, dtype=wp.float32),
        joint_limit_upper=wp.array(upper, dtype=wp.float32),
        joint_target_mode=wp.array(
            [int(newton.JointTargetMode.POSITION_VELOCITY)] * total_dof,
            dtype=wp.int32,
        ),
    )
    return SimpleNamespace(
        model=model,
        layout=layout,
        robot_reference=reference,
        world_count=world_count,
    )


def _build_finger_only_scene(wp):
    import newton

    from flash_chord.embodiments.base import EmbodimentLayout, HandLayout
    from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding

    layout = EmbodimentLayout(
        num_joint_q=2,
        num_joint_dof=2,
        hands=(
            HandLayout(
                side="left",
                **_frames("left"),
                finger_q_ids=(0,),
                finger_dof_ids=(0,),
                finger_joint_names=("left_finger",),
            ),
            HandLayout(
                side="right",
                **_frames("right"),
                finger_q_ids=(1,),
                finger_dof_ids=(1,),
                finger_joint_names=("right_finger",),
            ),
        ),
    )
    joint_q = np.array([[1.0, 2.0]], dtype=np.float32)
    keypoint_position = np.zeros((1, 1, 3), dtype=np.float32)
    keypoint_orientation = np.zeros((1, 1, 4), dtype=np.float32)
    keypoint_orientation[..., 0] = 1.0
    hands = tuple(
        BoundHandReference(
            side=side,
            wrist_pos_w=np.zeros((1, 3), dtype=np.float32),
            wrist_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            arm_joint_pos=np.empty((1, 0), dtype=np.float32),
            finger_joint_pos=joint_q[:, layout.hand(side).finger_q_ids],
            dp_pos_w=keypoint_position,
            dp_quat_w=keypoint_orientation,
            fingertip_pos_w=keypoint_position,
            fingertip_quat_w=keypoint_orientation,
        )
        for side in layout.sides
    )
    reference = RobotReferenceBinding(1, 20.0, 2, 2, joint_q, joint_q, hands)
    model = SimpleNamespace(
        body_count=1,
        joint_coord_count=2,
        joint_dof_count=2,
        joint_limit_lower=wp.array([-10.0, -10.0], dtype=wp.float32),
        joint_limit_upper=wp.array([10.0, 10.0], dtype=wp.float32),
        joint_target_mode=wp.array(
            [int(newton.JointTargetMode.POSITION)] * 2,
            dtype=wp.int32,
        ),
    )
    return SimpleNamespace(model=model, layout=layout, robot_reference=reference, world_count=1)


def test_zero_residual_uses_policy_order_and_scattered_simulation_order():
    import warp as wp

    from flash_chord.runtime.actions import (
        Action,
        ActionDiagnostics,
        ActionSpec,
        PolicyAction,
        ResidualJointPositionActionConfig,
    )

    with wp.ScopedDevice("cuda:0"):
        scene = _build_scene(wp)
        config = ResidualJointPositionActionConfig(ema=0.0)
        action = config.build(scene)
        action.process(wp.zeros(12, dtype=wp.float32), wp.array([-4, 99], dtype=wp.int32))
        processed = action.processed_target.numpy().reshape(2, 6)
        target = action.joint_target.numpy().reshape(2, 8)

    assert isinstance(config, ActionSpec)
    assert isinstance(action, Action)
    assert isinstance(action, PolicyAction)
    assert isinstance(action, ActionDiagnostics)
    assert action.sides == ("right", "left")
    assert action.action_dim == action.processed_dim == 6
    assert action.selection.q_ids == (3, 0, 2, 4, 1, 5)
    assert action.selection.dof_ids == (4, 1, 3, 0, 5, 2)
    assert action.block_names == (
        "right_arm_joint_residual[ra3,ra0]",
        "right_finger_joint_residual[rf]",
        "left_arm_joint_residual[la4,la1]",
        "left_finger_joint_residual[lf]",
    )
    assert action.block_ranges == ((0, 2), (2, 3), (3, 5), (5, 6))
    np.testing.assert_allclose(processed, [[14, 11, 13, 10, 15, 12], [24, 21, 23, 20, 25, 22]])
    np.testing.assert_allclose(target, [[10, 11, 12, 13, 14, 15, 0, 0], [20, 21, 22, 23, 24, 25, 0, 0]])
    np.testing.assert_allclose(action.action_l2.numpy(), 0.0)
    np.testing.assert_allclose(action.action_rate_l2.numpy(), 0.0)


def test_raw_residual_routes_group_scale_clip_ema_and_regularization():
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig

    policy_action = np.array([2.0, -2.0, 2.0, 2.0, -2.0, -2.0], dtype=np.float32)
    expected_filtered = np.array([0.15, -0.15, 0.3, 0.15, -0.15, -0.3], dtype=np.float32)
    reference_policy_order = np.array([14.0, 11.0, 13.0, 10.0, 15.0, 12.0], dtype=np.float32)
    config = ResidualJointPositionActionConfig(
        arm_scale=0.1,
        finger_scale=0.25,
        arm_clip=0.15,
        finger_clip=0.3,
        ema=0.25,
        clip_to_joint_limits=False,
    )

    with wp.ScopedDevice("cuda:0"):
        action = config.build(_build_scene(wp, world_count=1))
        timestep = wp.zeros(1, dtype=wp.int32)
        action.process(wp.array(policy_action, dtype=wp.float32), timestep)
        first_raw = action.raw_action.numpy().copy()
        first_filtered = action.filtered_action.numpy().copy()
        first_target = action.processed_target.numpy().copy()
        first_l2 = action.action_l2.numpy()[0]
        first_rate = action.action_rate_l2.numpy()[0]
        action.process(wp.zeros(6, dtype=wp.float32), timestep)
        second_filtered = action.filtered_action.numpy().copy()
        second_target = action.processed_target.numpy().copy()
        second_l2 = action.action_l2.numpy()[0]
        second_rate = action.action_rate_l2.numpy()[0]

    np.testing.assert_array_equal(first_raw, policy_action)
    np.testing.assert_allclose(first_filtered, expected_filtered)
    np.testing.assert_allclose(first_target, reference_policy_order + expected_filtered)
    assert first_l2 == pytest.approx(24.0)
    assert first_rate == pytest.approx(24.0)
    np.testing.assert_allclose(second_filtered, 0.25 * expected_filtered)
    np.testing.assert_allclose(second_target, reference_policy_order + 0.25 * expected_filtered)
    assert second_l2 == pytest.approx(0.0)
    assert second_rate == pytest.approx(24.0)


def test_processed_target_rate_uses_final_absolute_target_and_skips_reset_transition():
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig

    with wp.ScopedDevice("cuda:0"):
        action = ResidualJointPositionActionConfig(
            ema=0.0,
            clip_to_joint_limits=False,
            rate_semantics="processed_target",
        ).build(_build_scene(wp, world_count=1))
        zero = wp.zeros(6, dtype=wp.float32)
        action.process(zero, wp.array([0], dtype=wp.int32))
        first_rate = action.action_rate_l2.numpy()[0]
        action.process(zero, wp.array([1], dtype=wp.int32))
        reference_step_rate = action.action_rate_l2.numpy()[0]
        action.process(zero, wp.array([1], dtype=wp.int32))
        held_rate = action.action_rate_l2.numpy()[0]
        action.reset(wp.array([1], dtype=wp.int32))
        action.process(zero, wp.array([1], dtype=wp.int32))
        post_reset_rate = action.action_rate_l2.numpy()[0]

    assert first_rate == pytest.approx(0.0)
    assert reference_step_rate == pytest.approx(600.0)
    assert held_rate == pytest.approx(0.0)
    assert post_reset_rate == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("mapping", "expected_input", "expected_arm", "expected_finger"),
    [("linear", 2.0, 0.15, 0.375), ("rational", 8.0 / 3.0, 0.2, 0.5)],
)
def test_normalized_residual_routes_shared_mapping_by_joint_group(
    mapping,
    expected_input,
    expected_arm,
    expected_finger,
):
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig

    policy_action = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    expected_filtered = np.array(
        [expected_arm, -expected_arm, expected_finger, -expected_arm, expected_arm, -expected_finger],
        dtype=np.float32,
    )
    config = ResidualJointPositionActionConfig(
        arm_scale=0.1,
        finger_scale=0.25,
        arm_clip=0.2,
        finger_clip=0.5,
        ema=0.25,
        clip_to_joint_limits=False,
        input_mode="normalized",
        normalized_mapping=mapping,
    )

    with wp.ScopedDevice("cuda:0"):
        action = config.build(_build_scene(wp, world_count=1))
        action.process(wp.array(policy_action, dtype=wp.float32), wp.zeros(1, dtype=wp.int32))
        input_scale = action.input_scale.numpy()
        raw_action = action.raw_action.numpy()
        filtered_action = action.filtered_action.numpy()

    assert action.input_mapping == mapping
    assert action.input_scale_values == pytest.approx((expected_input,) * 6)
    np.testing.assert_allclose(input_scale, expected_input)
    np.testing.assert_allclose(raw_action, expected_input * policy_action)
    np.testing.assert_allclose(filtered_action, expected_filtered)


def test_joint_limits_are_world_local_and_configurable():
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig

    reference = np.array([[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]], dtype=np.float32)
    upper_offset = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float32)
    lower_offset = np.array([0.11, 0.12, 0.13, 0.14, 0.15, 0.16], dtype=np.float32)
    lower = np.full((2, 8), -100.0, dtype=np.float32)
    upper = np.full((2, 8), 100.0, dtype=np.float32)
    upper[0, :6] = reference[0] + upper_offset
    lower[1, :6] = reference[1] - lower_offset
    policy_action = np.concatenate((np.ones(6, dtype=np.float32), -np.ones(6, dtype=np.float32)))

    with wp.ScopedDevice("cuda:0"):
        scene = _build_scene(wp, lower=lower.reshape(-1), upper=upper.reshape(-1))
        clipped = ResidualJointPositionActionConfig(
            arm_scale=1.0,
            finger_scale=1.0,
            arm_clip=2.0,
            finger_clip=2.0,
            ema=0.0,
        ).build(scene)
        clipped.process(wp.array(policy_action, dtype=wp.float32), wp.array([0, 1], dtype=wp.int32))
        clipped_sim = clipped.joint_target.numpy().reshape(2, 8)
        clipped_policy = clipped.processed_target.numpy().reshape(2, 6)

        unclipped = ResidualJointPositionActionConfig(
            arm_scale=1.0,
            finger_scale=1.0,
            arm_clip=2.0,
            finger_clip=2.0,
            ema=0.0,
            clip_to_joint_limits=False,
        ).build(scene)
        unclipped.process(wp.array(policy_action, dtype=wp.float32), wp.array([0, 1], dtype=wp.int32))
        unclipped_sim = unclipped.joint_target.numpy().reshape(2, 8)

    np.testing.assert_allclose(clipped_sim[0, :6], reference[0] + upper_offset)
    np.testing.assert_allclose(clipped_sim[1, :6], reference[1] - lower_offset)
    np.testing.assert_allclose(clipped_policy[0], clipped_sim[0, [4, 1, 3, 0, 5, 2]])
    np.testing.assert_allclose(clipped_policy[1], clipped_sim[1, [4, 1, 3, 0, 5, 2]])
    np.testing.assert_allclose(unclipped_sim[0, :6], reference[0] + 1.0)
    np.testing.assert_allclose(unclipped_sim[1, :6], reference[1] - 1.0)


def test_masked_reset_clears_owned_state_without_touching_command_targets():
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig

    with wp.ScopedDevice("cuda:0"):
        action = ResidualJointPositionActionConfig(ema=0.0).build(_build_scene(wp))
        policy_action = np.concatenate((np.ones(6, dtype=np.float32), -np.ones(6, dtype=np.float32)))
        action.process(wp.array(policy_action, dtype=wp.float32), wp.array([0, 1], dtype=wp.int32))
        target = action.joint_target.numpy().reshape(2, 8)
        target[:, 6:] = [[60.0, 70.0], [80.0, 90.0]]
        action.joint_target.assign(target.reshape(-1))
        world_one = {
            "raw": action.raw_action.numpy().reshape(2, 6)[1].copy(),
            "filtered": action.filtered_action.numpy().reshape(2, 6)[1].copy(),
            "processed": action.processed_target.numpy().reshape(2, 6)[1].copy(),
            "target": target[1].copy(),
            "l2": action.action_l2.numpy()[1],
            "rate": action.action_rate_l2.numpy()[1],
        }
        action.reset(wp.array([1, 0], dtype=wp.int32))
        raw = action.raw_action.numpy().reshape(2, 6)
        filtered = action.filtered_action.numpy().reshape(2, 6)
        processed = action.processed_target.numpy().reshape(2, 6)
        target = action.joint_target.numpy().reshape(2, 8)
        l2 = action.action_l2.numpy()
        rate = action.action_rate_l2.numpy()

    np.testing.assert_allclose(raw[0], 0.0)
    np.testing.assert_allclose(filtered[0], 0.0)
    np.testing.assert_allclose(processed[0], 0.0)
    np.testing.assert_allclose(target[0], [0, 0, 0, 0, 0, 0, 60, 70])
    assert l2[0] == rate[0] == 0.0
    np.testing.assert_allclose(raw[1], world_one["raw"])
    np.testing.assert_allclose(filtered[1], world_one["filtered"])
    np.testing.assert_allclose(processed[1], world_one["processed"])
    np.testing.assert_allclose(target[1], world_one["target"])
    assert l2[1] == world_one["l2"]
    assert rate[1] == world_one["rate"]


def test_immediate_control_writes_position_targets_without_effort_control():
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig

    with wp.ScopedDevice("cuda:0"):
        action = ResidualJointPositionActionConfig(ema=0.0).build(_build_scene(wp, world_count=1))
        action.process(wp.zeros(6, dtype=wp.float32), wp.zeros(1, dtype=wp.int32))
        target = action.joint_target.numpy()
        target[6:] = [100.0, 200.0]
        action.joint_target.assign(target)
        control = SimpleNamespace(
            joint_target_pos=wp.zeros(8, dtype=wp.float32),
            joint_f=wp.full(8, value=7.0, dtype=wp.float32),
        )
        action.prepare_control(control)
        prepared = control.joint_target_pos.numpy().copy()
        force = control.joint_f.numpy().copy()
        action.apply_control(object(), control)
        applied = control.joint_target_pos.numpy().copy()

    np.testing.assert_allclose(prepared, target)
    np.testing.assert_allclose(applied, target)
    np.testing.assert_allclose(control.joint_f.numpy(), force)


def test_fixed_arm_delay_advances_per_physics_substep_and_leaves_other_targets_immediate():
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig
    from flash_chord.runtime.delay import TargetDelayConfig

    target_a = np.array([10, 11, 12, 13, 14, 15, 100, 200], dtype=np.float32)
    target_b = np.array([20, 21, 22, 23, 24, 25, 101, 201], dtype=np.float32)
    arm_dofs = (0, 1, 4, 5)
    immediate_dofs = (2, 3, 6, 7)

    with wp.ScopedDevice("cuda:0"):
        action = ResidualJointPositionActionConfig(
            ema=0.0,
            delay=TargetDelayConfig(min_steps=4, max_steps=4),
        ).build(_build_scene(wp, world_count=1))
        assert action.delay is not None
        assert action.delay.delayed_dof_ids == (4, 1, 0, 5)
        action.reset(wp.ones(1, dtype=wp.int32))
        control = SimpleNamespace(
            joint_target_pos=wp.zeros(8, dtype=wp.float32),
            joint_f=wp.full(8, value=7.0, dtype=wp.float32),
        )
        action.process(wp.zeros(6, dtype=wp.float32), wp.array([0], dtype=wp.int32))
        action.joint_target.assign(target_a)
        action.prepare_control(control)
        np.testing.assert_allclose(control.joint_target_pos.numpy(), 0.0)
        action.apply_control(None, control)
        np.testing.assert_allclose(control.joint_target_pos.numpy(), target_a)

        action.process(wp.zeros(6, dtype=wp.float32), wp.array([1], dtype=wp.int32))
        immediate_processed = action.processed_target.numpy().copy()
        action.joint_target.assign(target_b)
        outputs = []
        for _ in range(5):
            action.apply_control(None, control)
            outputs.append(control.joint_target_pos.numpy().copy())
        outputs = np.stack(outputs)

    np.testing.assert_allclose(immediate_processed, target_b[[4, 1, 3, 0, 5, 2]])
    np.testing.assert_allclose(outputs[:4, arm_dofs], np.broadcast_to(target_a[list(arm_dofs)], (4, 4)))
    np.testing.assert_allclose(
        outputs[:, immediate_dofs],
        np.broadcast_to(target_b[list(immediate_dofs)], (5, 4)),
    )
    np.testing.assert_allclose(outputs[4], target_b)
    np.testing.assert_allclose(control.joint_f.numpy(), 7.0)


def test_build_rejects_partial_or_unnamed_control_topology():
    import newton
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig

    with wp.ScopedDevice("cuda:0"):
        scene = _build_scene(wp, world_count=1)
        with pytest.raises(ValueError, match="exact permutation"):
            ResidualJointPositionActionConfig(side_order=("right",)).build(scene)

        with pytest.raises(ValueError, match="finger names must match"):
            replace(scene.layout.hand("right"), finger_joint_names=())

        effort_model = SimpleNamespace(**vars(scene.model))
        effort_mode = effort_model.joint_target_mode.numpy()
        effort_mode[4] = int(newton.JointTargetMode.EFFORT)
        effort_model.joint_target_mode = wp.array(effort_mode, dtype=wp.int32)
        effort_scene = SimpleNamespace(**vars(scene))
        effort_scene.model = effort_model
        with pytest.raises(ValueError, match="position-capable target modes"):
            ResidualJointPositionActionConfig().build(effort_scene)


def test_finger_only_control_supports_no_delay_and_rejects_arm_delay():
    import warp as wp

    from flash_chord.runtime.actions import ResidualJointPositionActionConfig
    from flash_chord.runtime.delay import TargetDelayConfig

    with wp.ScopedDevice("cuda:0"):
        scene = _build_finger_only_scene(wp)
        action = ResidualJointPositionActionConfig(ema=0.0).build(scene)
        action.process(wp.zeros(2, dtype=wp.float32), wp.zeros(1, dtype=wp.int32))
        with pytest.raises(ValueError, match="arm target delay requires"):
            ResidualJointPositionActionConfig(delay=TargetDelayConfig(min_steps=1, max_steps=1)).build(scene)

    assert action.block_names == (
        "right_finger_joint_residual[right_finger]",
        "left_finger_joint_residual[left_finger]",
    )
    assert action.block_ranges == ((0, 1), (1, 2))
    np.testing.assert_allclose(action.joint_target.numpy(), [1.0, 2.0])
