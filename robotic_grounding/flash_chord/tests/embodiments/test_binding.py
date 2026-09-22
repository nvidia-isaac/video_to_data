# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for setup-time robot reference binding and semantic FK."""

from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _vega(preserve_palm_bodies=True):
    import newton

    from flash_chord.embodiments.vega_sharpa import VegaSharpa, VegaSharpaConfig

    builder = newton.ModelBuilder()
    layout = VegaSharpa(VegaSharpaConfig(preserve_palm_bodies=preserve_palm_bodies)).build(builder)
    return builder, layout


def _trajectory(builder):
    first = np.asarray(builder.joint_q, dtype=np.float32)
    second = np.linspace(-0.1, 0.1, len(first), dtype=np.float32)
    return np.stack((first, second))


def test_binding_exposes_readonly_dp_and_fingertip_trajectories(monkeypatch):
    import newton

    from flash_chord.embodiments.binding import build_robot_reference_binding

    builder, layout = _vega()
    monkeypatch.setattr(
        newton.ModelBuilder,
        "replicate",
        lambda *_args, **_kwargs: pytest.fail("reference FK must not replicate collision geometry"),
    )
    joint_q = _trajectory(builder)
    binding = build_robot_reference_binding(builder, layout, joint_q, joint_q * 2.0, fps=20.0)

    assert binding.num_frames == 2
    assert binding.num_joint_q == binding.num_joint_dof == 58
    assert binding.fps == 20.0
    np.testing.assert_array_equal(binding.joint_q, joint_q)
    np.testing.assert_array_equal(binding.joint_target, joint_q * 2.0)
    assert not binding.joint_q.flags.writeable
    assert not binding.joint_target.flags.writeable

    for side in layout.sides:
        hand = binding.hand(side)
        assert hand.wrist_pos_w.shape == (2, 3)
        assert hand.wrist_quat_w.shape == (2, 4)
        assert hand.arm_joint_pos.shape == (2, 7)
        np.testing.assert_array_equal(hand.arm_joint_pos, joint_q[:, layout.hand(side).arm_q_ids])
        assert hand.finger_joint_pos.shape == (2, 22)
        assert hand.dp_pos_w.shape == hand.fingertip_pos_w.shape == (2, 5, 3)
        assert hand.dp_quat_w.shape == hand.fingertip_quat_w.shape == (2, 5, 4)
        assert hand.keypoint_pos_w("dp").shape == (2, 6, 3)
        assert hand.keypoint_pos_w("fingertip").shape == (2, 6, 3)
        distance = np.linalg.norm(hand.dp_pos_w - hand.fingertip_pos_w, axis=-1)
        assert np.all(distance > 0.025)
        assert all(
            not array.flags.writeable
            for array in (
                hand.wrist_pos_w,
                hand.wrist_quat_w,
                hand.arm_joint_pos,
                hand.finger_joint_pos,
                hand.dp_pos_w,
                hand.dp_quat_w,
                hand.fingertip_pos_w,
                hand.fingertip_quat_w,
            )
        )

    with pytest.raises(ValueError, match="must be 'dp' or 'fingertip'"):
        binding.hand("left").keypoint_pos_w("distal")
    with pytest.raises(KeyError, match="no bound hand"):
        binding.hand("center")


def test_vega_named_frame_validation_accepts_v2d_dp_frame_set():
    from flash_chord.data.motion_v1 import MotionV1Reference
    from flash_chord.data.reference import ReferenceMetadata
    from flash_chord.embodiments.binding import build_robot_reference_binding
    from flash_chord.embodiments.vega_sharpa import (
        VegaSharpaConfig,
        _validate_named_frame_poses,
    )

    builder, layout = _vega()
    joint_q = _trajectory(builder)
    binding = build_robot_reference_binding(builder, layout, joint_q, joint_q, fps=20.0)
    frame_names = [frame.name for side in layout.sides for frame in layout.hand(side).dp_frames]
    frame_pos = np.concatenate(
        [binding.hand(side).dp_pos_w for side in layout.sides],
        axis=1,
    )
    frame_quat = np.concatenate(
        [binding.hand(side).dp_quat_w for side in layout.sides],
        axis=1,
    )
    reference = MotionV1Reference(
        _metadata=ReferenceMetadata(source_path="v2d.parquet", source_fps=20.0),
        _fps=20.0,
        _sides=layout.sides,
        _robot_joint_names=[],
        _robot_joint_pos=np.empty((2, 0), dtype=np.float32),
        _robot_root_pos_w=np.zeros((2, 3), dtype=np.float32),
        _robot_root_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        _robot_frame_names=frame_names,
        _robot_frame_pos_w=frame_pos,
        _robot_frame_quat_w=frame_quat,
        _object_body_pos_w=np.empty((2, 0, 3), dtype=np.float32),
        _object_body_quat_w=np.empty((2, 0, 4), dtype=np.float32),
        _object_articulation=np.empty((2, 0), dtype=np.float32),
        _contact_pos_w={},
        _contact_normal_w={},
        _contact_part_ids={},
        _contact_active={},
        _object_name="",
        _object_body_names=[],
        _object_mesh_paths=[],
        _object_urdf_paths=[],
        _object_mesh_radius=np.empty(0, dtype=np.float32),
        _object_assets=(),
    )

    config = VegaSharpaConfig()
    assert config.dp_frame_orientation_tolerance == pytest.approx(1.0e-3)
    _validate_named_frame_poses(binding, layout, reference, config)

    invalid_position = frame_pos.copy()
    invalid_position[1, 0, 0] += 2.0e-3
    with pytest.raises(ValueError, match="native named frame 'left_thumb_DP'"):
        _validate_named_frame_poses(
            binding,
            layout,
            replace(reference, _robot_frame_pos_w=invalid_position),
            config,
        )


def test_semantic_fk_is_independent_of_palm_mount_collapse_choice():
    from flash_chord.embodiments.binding import build_robot_reference_binding

    retained_builder, retained_layout = _vega(preserve_palm_bodies=True)
    collapsed_builder, collapsed_layout = _vega(preserve_palm_bodies=False)
    joint_q = _trajectory(retained_builder)
    retained = build_robot_reference_binding(retained_builder, retained_layout, joint_q, joint_q, fps=20.0)
    collapsed = build_robot_reference_binding(collapsed_builder, collapsed_layout, joint_q, joint_q, fps=20.0)

    for side in retained_layout.sides:
        retained_topology = retained_layout.hand(side)
        collapsed_topology = collapsed_layout.hand(side)
        assert collapsed_topology.collision_shape_ids == retained_topology.collision_shape_ids
        assert collapsed_topology.contact_shape_ids == retained_topology.contact_shape_ids
        collapsed_wrist_shapes = {
            shape_id
            for shape_id, body_id in enumerate(collapsed_builder.shape_body)
            if body_id == collapsed_topology.wrist_body_id
        }
        assert collapsed_wrist_shapes.intersection(collapsed_topology.contact_shape_ids)
        assert collapsed_wrist_shapes - set(collapsed_topology.collision_shape_ids)

        retained_hand = retained.hand(side)
        collapsed_hand = collapsed.hand(side)
        np.testing.assert_allclose(retained_hand.wrist_pos_w, collapsed_hand.wrist_pos_w, atol=2e-6)
        np.testing.assert_allclose(retained_hand.wrist_quat_w, collapsed_hand.wrist_quat_w, atol=2e-6)
        np.testing.assert_allclose(retained_hand.dp_pos_w, collapsed_hand.dp_pos_w, atol=2e-6)
        np.testing.assert_allclose(retained_hand.dp_quat_w, collapsed_hand.dp_quat_w, atol=2e-6)
        np.testing.assert_allclose(retained_hand.fingertip_pos_w, collapsed_hand.fingertip_pos_w, atol=2e-6)
        np.testing.assert_allclose(
            retained_hand.fingertip_quat_w,
            collapsed_hand.fingertip_quat_w,
            atol=2e-6,
        )


def test_device_reference_preserves_named_binding_and_requested_side_order():
    import warp as wp

    from flash_chord.embodiments.binding import DeviceRobotReference, build_robot_reference_binding

    builder, layout = _vega()
    joint_q = _trajectory(builder)
    binding = build_robot_reference_binding(builder, layout, joint_q, joint_q * 2.0, fps=20.0)
    with wp.ScopedDevice("cuda:0"):
        device_reference = DeviceRobotReference.build(binding, layout, sides=("right", "left"))
        joint_target = device_reference.joint_target.numpy().reshape(2, 58)
        arm_joint_pos = device_reference.arm_joint_pos.numpy().reshape(2, 14)
        finger_joint_pos = device_reference.finger_joint_pos.numpy().reshape(2, 44)
        wrist_pos = device_reference.wrist_pos_w.numpy().reshape(2, 2, 3)
        wrist_quat = device_reference.wrist_quat_w.numpy().reshape(2, 2, 4)

    assert device_reference.sides == ("right", "left")
    assert device_reference.arm_counts == (7, 7)
    assert device_reference.finger_counts == (22, 22)
    np.testing.assert_array_equal(joint_target, binding.joint_target)
    np.testing.assert_array_equal(
        arm_joint_pos,
        np.concatenate(
            (binding.hand("right").arm_joint_pos, binding.hand("left").arm_joint_pos),
            axis=1,
        ),
    )
    np.testing.assert_array_equal(
        finger_joint_pos,
        np.concatenate(
            (binding.hand("right").finger_joint_pos, binding.hand("left").finger_joint_pos),
            axis=1,
        ),
    )
    np.testing.assert_array_equal(wrist_pos[:, 0], binding.hand("right").wrist_pos_w)
    np.testing.assert_array_equal(wrist_quat[:, 0], binding.hand("right").wrist_quat_w[:, (1, 2, 3, 0)])

    with pytest.raises(ValueError, match="device reference sides"):
        DeviceRobotReference.build(binding, layout, sides=("left",))
    bad_left = replace(binding.hand("left"), arm_joint_pos=np.empty((2, 6), dtype=np.float32))
    bad_binding = replace(binding, hands=(bad_left, binding.hand("right")))
    with pytest.raises(ValueError, match="left arm_joint_pos has width"):
        DeviceRobotReference.build(bad_binding, layout)


def test_bound_reference_dataclasses_normalize_and_reject_invalid_ownership():
    from flash_chord.embodiments.binding import build_robot_reference_binding

    builder, layout = _vega()
    joint_q = _trajectory(builder)
    binding = build_robot_reference_binding(builder, layout, joint_q, joint_q, fps=20.0)
    left = binding.hand("left")

    scaled_quaternion = replace(left, wrist_quat_w=left.wrist_quat_w * 2.0)
    np.testing.assert_allclose(np.linalg.norm(scaled_quaternion.wrist_quat_w, axis=1), 1.0)
    assert not scaled_quaternion.wrist_quat_w.flags.writeable

    invalid_fingers = left.finger_joint_pos.copy()
    invalid_fingers[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite values"):
        replace(left, finger_joint_pos=invalid_fingers)
    with pytest.raises(ValueError, match="sides must be unique"):
        replace(binding, hands=(left, left))
    with pytest.raises(ValueError, match="declares 3 frames"):
        replace(binding, num_frames=3)


def test_named_joint_trajectory_is_permutation_invariant():
    from flash_chord.embodiments.binding import build_named_joint_trajectory

    builder, layout = _vega()
    scalar = layout.scalar_joints
    lower = np.asarray(builder.joint_limit_lower, dtype=np.float32)[list(scalar.dof_ids)]
    upper = np.asarray(builder.joint_limit_upper, dtype=np.float32)[list(scalar.dof_ids)]
    reference_values = np.stack((0.5 * lower + 0.5 * upper, 0.25 * lower + 0.75 * upper))
    permutation = np.arange(len(scalar.names))[::-1]
    joint_q, joint_target = build_named_joint_trajectory(
        builder,
        layout,
        [scalar.names[index] for index in permutation],
        reference_values[:, permutation],
        joint_limit_tolerance=0.0,
    )

    np.testing.assert_array_equal(joint_q[:, scalar.q_ids], reference_values)
    np.testing.assert_array_equal(joint_target[:, scalar.dof_ids], reference_values)


def test_named_joint_trajectory_rejects_nonbijective_or_incomplete_layouts():
    from flash_chord.embodiments.binding import build_named_joint_trajectory

    builder, layout = _vega()
    names = list(layout.scalar_joints.names)
    joint_pos = np.zeros((2, len(names)), dtype=np.float32)

    with pytest.raises(ValueError, match="duplicate joints"):
        build_named_joint_trajectory(
            builder,
            layout,
            [names[0], *names[1:-1], names[0]],
            joint_pos,
            joint_limit_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="missing=.*unexpected"):
        build_named_joint_trajectory(
            builder,
            layout,
            [*names[:-1], "unexpected_joint"],
            joint_pos,
            joint_limit_tolerance=0.0,
        )
    with pytest.raises(ValueError, match="does not define named scalar joints"):
        build_named_joint_trajectory(
            builder,
            replace(layout, scalar_joints=None),
            names,
            joint_pos,
            joint_limit_tolerance=0.0,
        )


def test_named_joint_trajectory_preserves_q_and_clamps_only_controller_limit_noise():
    from flash_chord.embodiments.binding import build_named_joint_trajectory

    builder, layout = _vega()
    scalar = layout.scalar_joints
    lower = np.asarray(builder.joint_limit_lower, dtype=np.float32)[list(scalar.dof_ids)]
    upper = np.asarray(builder.joint_limit_upper, dtype=np.float32)[list(scalar.dof_ids)]
    source = (0.5 * lower + 0.5 * upper)[None, :]
    source[0, 0] = upper[0] + 5.0e-6
    source[0, 1] = lower[1] - 5.0e-6

    joint_q, joint_target = build_named_joint_trajectory(
        builder,
        layout,
        scalar.names,
        source,
        joint_limit_tolerance=1.0e-5,
    )

    np.testing.assert_array_equal(joint_q[:, scalar.q_ids], source)
    assert joint_target[0, scalar.dof_ids[0]] == upper[0]
    assert joint_target[0, scalar.dof_ids[1]] == lower[1]
    with pytest.raises(ValueError, match="named joint .* frame 0 .* tolerance"):
        build_named_joint_trajectory(
            builder,
            layout,
            scalar.names,
            source,
            joint_limit_tolerance=1.0e-7,
        )
    for invalid_tolerance in (-1.0, np.nan):
        with pytest.raises(ValueError, match="tolerance must be nonnegative and finite"):
            build_named_joint_trajectory(
                builder,
                layout,
                scalar.names,
                source,
                joint_limit_tolerance=invalid_tolerance,
            )


@pytest.mark.parametrize(
    ("joint_q_shape", "target_shape", "fps", "message"),
    [
        ((2, 57), (2, 58), 20.0, "joint_q must have shape"),
        ((2, 58), (3, 58), 20.0, "joint_q has 2 frames"),
        ((2, 58), (2, 58), 0.0, "fps must be positive"),
    ],
)
def test_binding_rejects_invalid_trajectory_contract(joint_q_shape, target_shape, fps, message):
    from flash_chord.embodiments.binding import build_robot_reference_binding

    builder, layout = _vega()
    joint_q = np.zeros(joint_q_shape, dtype=np.float32)
    joint_target = np.zeros(target_shape, dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        build_robot_reference_binding(builder, layout, joint_q, joint_target, fps)
