# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for residual action processing and reference target gathering."""

from dataclasses import replace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.gpu


def _frames(side: str) -> dict:
    from flash_chord.embodiments.base import BodyFrame

    return {
        "palm_frame": BodyFrame(f"{side}_palm", 0),
        "dp_frames": (BodyFrame(f"{side}_index_DP", 0),),
        "fingertip_frames": (BodyFrame(f"{side}_index_fingertip", 0),),
    }


def _map_normalized_action(policy_action, input_scale, mapping):
    normalized = np.clip(np.asarray(policy_action), -1.0, 1.0)
    if mapping == "linear":
        return normalized * input_scale
    magnitude = np.abs(normalized)
    return normalized / ((1.0 - magnitude) + magnitude / input_scale)


def _reference():
    from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding

    wrist_pos_w = {
        "right": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "left": np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32),
    }
    wrist_quat_w = np.array(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    finger_joint_pos = {
        "right": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        "left": np.array([[-10.0, -20.0], [-30.0, -40.0]], dtype=np.float32),
    }
    joint_q = np.zeros((2, 18), dtype=np.float32)
    joint_target = np.zeros((2, 16), dtype=np.float32)
    for side, q_ids, dof_ids in (
        ("left", (7, 8), (6, 7)),
        ("right", (16, 17), (14, 15)),
    ):
        joint_q[:, q_ids] = finger_joint_pos[side]
        joint_target[:, dof_ids] = finger_joint_pos[side]
    joint_q[:, (0, 1, 2)] = wrist_pos_w["left"]
    joint_q[:, (9, 10, 11)] = wrist_pos_w["right"]
    joint_q[:, 3] = 1.0
    joint_q[:, 12] = 1.0
    joint_target[:, (0, 1, 2)] = wrist_pos_w["left"]
    joint_target[:, (8, 9, 10)] = wrist_pos_w["right"]
    keypoint_position = np.zeros((2, 1, 3), dtype=np.float32)
    keypoint_orientation = np.zeros((2, 1, 4), dtype=np.float32)
    keypoint_orientation[..., 0] = 1.0
    hands = tuple(
        BoundHandReference(
            side=side,
            wrist_pos_w=wrist_pos_w[side],
            wrist_quat_w=wrist_quat_w,
            arm_joint_pos=np.empty((2, 0), dtype=np.float32),
            finger_joint_pos=finger_joint_pos[side],
            dp_pos_w=keypoint_position,
            dp_quat_w=keypoint_orientation,
            fingertip_pos_w=keypoint_position,
            fingertip_quat_w=keypoint_orientation,
        )
        for side in ("left", "right")
    )
    return RobotReferenceBinding(
        num_frames=2,
        fps=20.0,
        num_joint_q=18,
        num_joint_dof=16,
        joint_q=joint_q,
        joint_target=joint_target,
        hands=hands,
    )


def _layout():
    from flash_chord.embodiments.base import EmbodimentLayout, HandLayout

    left = HandLayout(
        side="left",
        **_frames("left"),
        wrist_pos_dof_ids=(0, 1, 2),
        wrist_orient_dof_ids=(3, 4, 5),
        wrist_orient_q_id=3,
        finger_dof_ids=(6, 7),
        wrist_pos_q_ids=(0, 1, 2),
        finger_q_ids=(7, 8),
        finger_joint_names=("left_a", "left_b"),
    )
    right = HandLayout(
        side="right",
        **_frames("right"),
        wrist_pos_dof_ids=(8, 9, 10),
        wrist_orient_dof_ids=(11, 12, 13),
        wrist_orient_q_id=12,
        finger_dof_ids=(14, 15),
        wrist_pos_q_ids=(9, 10, 11),
        finger_q_ids=(16, 17),
        finger_joint_names=("right_a", "right_b"),
    )
    return EmbodimentLayout(num_joint_q=18, num_joint_dof=16, hands=(left, right))


def test_zero_action_gathers_reference_in_policy_and_sim_order():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(_layout(), _reference(), world_count=2)
        action = wp.zeros(32, dtype=wp.float32)
        timestep = wp.array([0, 1], dtype=wp.int32)
        buffers.process(action, timestep)
        target = buffers.joint_target.numpy().reshape(2, 16)
        processed = buffers.processed_target.numpy().reshape(2, 18)
        finger_reference = buffers.reference.finger_joint_pos.numpy().reshape(2, 4)

    np.testing.assert_allclose(target[0, 8:11], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(target[0, 14:16], [10.0, 20.0])
    np.testing.assert_allclose(target[0, 0:3], [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(target[0, 6:8], [-10.0, -20.0])
    np.testing.assert_allclose(target[1, 8:11], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(processed[0, :7], [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(processed[0, 7:9], [10.0, 20.0])
    np.testing.assert_allclose(finger_reference, [[10.0, 20.0, -10.0, -20.0], [30.0, 40.0, -30.0, -40.0]])
    np.testing.assert_allclose(buffers.action_l2.numpy(), 0.0)
    np.testing.assert_allclose(buffers.action_rate_l2.numpy(), 0.0)


def test_action_scaling_clip_ema_orientation_and_penalties():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig, ResidualHandPoseAction

    config = ActionConfig(
        wrist_pos_scale=1.0,
        wrist_ori_scale=1.0,
        finger_joint_pos_scale=1.0,
        wrist_pos_clip=0.2,
        wrist_ori_clip=1.0,
        finger_joint_pos_clip=1.0,
        ema=0.25,
        normalized_mapping="rational",
    )
    raw = np.zeros(16, dtype=np.float32)
    raw[0] = 2.0
    raw[1] = -0.0
    raw[3:6] = [0.2, -0.1, 0.3]
    raw[6:8] = [0.4, -0.4]
    expected_filtered = 0.75 * raw
    expected_filtered[0] = 0.2

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(_layout(), _reference(), world_count=1, config=config)
        timestep = wp.zeros(1, dtype=wp.int32)
        buffers.process(wp.array(raw, dtype=wp.float32), timestep)
        stored_raw = buffers.raw_action.numpy()
        filtered = buffers.filtered_action.numpy()
        processed = buffers.processed_target.numpy()
        target = buffers.joint_target.numpy()
        first_l2 = buffers.action_l2.numpy()[0]
        first_rate = buffers.action_rate_l2.numpy()[0]

        buffers.process(wp.zeros(16, dtype=wp.float32), timestep)
        second_filtered = buffers.filtered_action.numpy()
        second_l2 = buffers.action_l2.numpy()[0]
        second_rate = buffers.action_rate_l2.numpy()[0]
        action_strategy = ResidualHandPoseAction(buffers=buffers, wrist=None)  # type: ignore[arg-type]

    np.testing.assert_array_equal(stored_raw.view(np.uint32), raw.view(np.uint32))
    assert action_strategy.input_mode == "raw"
    assert action_strategy.normalized_mapping == "rational"
    assert action_strategy.input_mapping == "identity"
    np.testing.assert_allclose(filtered, expected_filtered, atol=1e-6)
    np.testing.assert_allclose(processed[:3], [1.2, 2.0, 3.0], atol=1e-6)
    expected_quat_xyzw = Rotation.from_euler("xyz", expected_filtered[3:6]).as_quat()
    expected_quat_wxyz = expected_quat_xyzw[[3, 0, 1, 2]]
    np.testing.assert_allclose(processed[3:7], expected_quat_wxyz, atol=1e-6)
    np.testing.assert_allclose(target[11:14], Rotation.from_quat(expected_quat_xyzw).as_rotvec(), atol=1e-6)
    np.testing.assert_allclose(processed[7:9], [10.3, 19.7], atol=1e-6)
    assert first_l2 == pytest.approx(float(np.dot(raw, raw)))
    assert first_rate == pytest.approx(float(np.dot(raw, raw)))
    np.testing.assert_allclose(second_filtered, 0.25 * expected_filtered, atol=1e-6)
    assert second_l2 == pytest.approx(0.0)
    assert second_rate == pytest.approx(float(np.dot(raw, raw)))


def test_normalized_linear_action_clamps_and_maps_each_policy_block_to_its_physical_clip():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig, ResidualHandPoseAction

    config = ActionConfig(
        wrist_pos_scale=0.1,
        wrist_ori_scale=0.25,
        finger_joint_pos_scale=0.5,
        wrist_pos_clip=0.2,
        wrist_ori_clip=1.0,
        finger_joint_pos_clip=1.5,
        ema=0.0,
        input_mode="normalized",
        normalized_mapping="linear",
    )
    policy_action = np.concatenate(
        [
            np.full(8, 2.0, dtype=np.float32),
            np.full(8, -2.0, dtype=np.float32),
        ]
    )
    hand_input_scale = np.array([2.0] * 3 + [4.0] * 3 + [3.0] * 2, dtype=np.float32)
    expected_input_scale = np.tile(hand_input_scale, 2)
    expected_filtered = np.array([0.2] * 3 + [1.0] * 3 + [1.5] * 2, dtype=np.float32)
    expected_filtered = np.concatenate([expected_filtered, -expected_filtered])

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(_layout(), _reference(), world_count=1, config=config)
        buffers.process(wp.array(policy_action, dtype=wp.float32), wp.zeros(1, dtype=wp.int32))
        input_scale = buffers.input_scale.numpy()
        raw_action = buffers.raw_action.numpy()
        filtered_action = buffers.filtered_action.numpy()
        processed_target = buffers.processed_target.numpy()
        action_strategy = ResidualHandPoseAction(buffers=buffers, wrist=None)  # type: ignore[arg-type]

    np.testing.assert_allclose(input_scale, expected_input_scale)
    assert action_strategy.input_mode == "normalized"
    assert action_strategy.normalized_mapping == "linear"
    assert action_strategy.input_mapping == "linear"
    assert action_strategy.input_scale_values == pytest.approx(tuple(expected_input_scale))
    np.testing.assert_allclose(raw_action, np.sign(policy_action) * expected_input_scale)
    np.testing.assert_allclose(filtered_action, expected_filtered)
    np.testing.assert_allclose(processed_target[:3], [1.2, 2.2, 3.2], atol=1e-6)
    np.testing.assert_allclose(processed_target[7:9], [11.5, 21.5], atol=1e-6)
    np.testing.assert_allclose(processed_target[9:12], [-1.2, -2.2, -3.2], atol=1e-6)
    np.testing.assert_allclose(processed_target[16:18], [-11.5, -21.5], atol=1e-6)


def test_normalized_action_regularizes_transformed_residuals_and_rate():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    config = ActionConfig(
        wrist_pos_scale=0.1,
        wrist_ori_scale=0.25,
        finger_joint_pos_scale=0.5,
        wrist_pos_clip=0.2,
        wrist_ori_clip=1.0,
        finger_joint_pos_clip=1.5,
        ema=0.0,
        input_mode="normalized",
        normalized_mapping="linear",
    )
    first_policy_action = np.linspace(-0.75, 0.75, 16, dtype=np.float32)
    second_policy_action = np.linspace(0.5, -0.5, 16, dtype=np.float32)
    input_scale = np.tile(np.array([2.0] * 3 + [4.0] * 3 + [3.0] * 2, dtype=np.float32), 2)
    first_residual = first_policy_action * input_scale
    second_residual = second_policy_action * input_scale

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(_layout(), _reference(), world_count=1, config=config)
        timestep = wp.zeros(1, dtype=wp.int32)
        buffers.process(wp.array(first_policy_action, dtype=wp.float32), timestep)
        first_raw = buffers.raw_action.numpy()
        first_l2 = buffers.action_l2.numpy()[0]
        first_rate = buffers.action_rate_l2.numpy()[0]
        buffers.process(wp.array(second_policy_action, dtype=wp.float32), timestep)
        second_raw = buffers.raw_action.numpy()
        second_l2 = buffers.action_l2.numpy()[0]
        second_rate = buffers.action_rate_l2.numpy()[0]

    np.testing.assert_allclose(first_raw, first_residual)
    assert first_l2 == pytest.approx(float(np.dot(first_residual, first_residual)), rel=1e-6)
    assert first_rate == pytest.approx(first_l2)
    np.testing.assert_allclose(second_raw, second_residual)
    assert second_l2 == pytest.approx(float(np.dot(second_residual, second_residual)), rel=1e-6)
    delta = second_residual - first_residual
    assert second_rate == pytest.approx(float(np.dot(delta, delta)), rel=1e-6)


def test_normalized_rational_endpoints_reach_physical_clip_on_first_control_step():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig, ResidualHandPoseAction

    config = ActionConfig(
        wrist_pos_scale=0.1,
        wrist_ori_scale=0.25,
        finger_joint_pos_scale=0.5,
        wrist_pos_clip=0.2,
        wrist_ori_clip=1.0,
        finger_joint_pos_clip=1.5,
        ema=0.3,
        input_mode="normalized",
        normalized_mapping="rational",
    )
    policy_action = np.concatenate([np.ones(8, dtype=np.float32), -np.ones(8, dtype=np.float32)])
    hand_input_scale = np.array(
        [0.2 / (0.7 * 0.1)] * 3 + [1.0 / (0.7 * 0.25)] * 3 + [1.5 / (0.7 * 0.5)] * 2,
        dtype=np.float32,
    )
    expected_input_scale = np.tile(hand_input_scale, 2)
    expected_filtered = np.array([0.2] * 3 + [1.0] * 3 + [1.5] * 2, dtype=np.float32)
    expected_filtered = np.concatenate([expected_filtered, -expected_filtered])

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(_layout(), _reference(), world_count=1, config=config)
        buffers.process(wp.array(policy_action, dtype=wp.float32), wp.zeros(1, dtype=wp.int32))
        raw_action = buffers.raw_action.numpy()
        filtered_action = buffers.filtered_action.numpy()
        action_strategy = ResidualHandPoseAction(buffers=buffers, wrist=None)  # type: ignore[arg-type]

    assert action_strategy.input_mapping == "rational"
    assert action_strategy.input_scale_values == pytest.approx(tuple(expected_input_scale))
    np.testing.assert_allclose(raw_action, np.sign(policy_action) * expected_input_scale, rtol=1e-6)
    np.testing.assert_allclose(filtered_action, expected_filtered, rtol=1e-6, atol=1e-6)


def test_normalized_rational_mapping_is_monotonic_bounded_and_locally_raw_scaled():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    config = ActionConfig(
        wrist_pos_scale=0.1,
        wrist_ori_scale=0.25,
        finger_joint_pos_scale=0.5,
        wrist_pos_clip=0.2,
        wrist_ori_clip=1.0,
        finger_joint_pos_clip=1.5,
        ema=0.3,
        input_mode="normalized",
        normalized_mapping="rational",
    )
    levels = np.array([-2.0, -1.0, -0.5, -1.0e-4, 0.0, 1.0e-4, 0.5, 1.0, 2.0], dtype=np.float32)
    policy_action = np.broadcast_to(levels[:, None], (len(levels), 16)).copy()

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(_layout(), _reference(), world_count=len(levels), config=config)
        buffers.process(
            wp.array(policy_action.reshape(-1), dtype=wp.float32),
            wp.zeros(len(levels), dtype=wp.int32),
        )
        input_scale = buffers.input_scale.numpy()
        scale = buffers.scale.numpy()
        clip = buffers.clip.numpy()
        raw_action = buffers.raw_action.numpy().reshape(len(levels), 16)
        filtered_action = buffers.filtered_action.numpy().reshape(len(levels), 16)

    expected_raw = _map_normalized_action(policy_action, input_scale, "rational")
    np.testing.assert_allclose(raw_action, expected_raw, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(filtered_action, (1.0 - config.ema) * expected_raw * scale, atol=2e-6)
    assert np.all(np.diff(raw_action, axis=0) >= 0.0)
    assert np.all(np.abs(raw_action) <= input_scale[None, :] * (1.0 + 1e-6))
    assert np.all(np.abs(filtered_action) <= clip[None, :] * (1.0 + 1e-6))
    np.testing.assert_allclose(raw_action[[0, 1]], np.broadcast_to(-input_scale, (2, 16)), rtol=2e-6)
    np.testing.assert_allclose(raw_action[[7, 8]], np.broadcast_to(input_scale, (2, 16)), rtol=2e-6)
    np.testing.assert_allclose(raw_action[3] / levels[3], 1.0, atol=1e-3)
    np.testing.assert_allclose(raw_action[5] / levels[5], 1.0, atol=1e-3)


def test_normalized_rational_action_regularizes_transformed_residuals_and_rate():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    config = ActionConfig(
        wrist_pos_scale=0.1,
        wrist_ori_scale=0.25,
        finger_joint_pos_scale=0.5,
        wrist_pos_clip=0.2,
        wrist_ori_clip=1.0,
        finger_joint_pos_clip=1.5,
        ema=0.3,
        input_mode="normalized",
        normalized_mapping="rational",
    )
    first_policy_action = np.linspace(-0.75, 0.75, 16, dtype=np.float32)
    second_policy_action = np.linspace(0.5, -0.5, 16, dtype=np.float32)

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(_layout(), _reference(), world_count=1, config=config)
        input_scale = buffers.input_scale.numpy()
        first_residual = _map_normalized_action(first_policy_action, input_scale, "rational")
        second_residual = _map_normalized_action(second_policy_action, input_scale, "rational")
        timestep = wp.zeros(1, dtype=wp.int32)
        buffers.process(wp.array(first_policy_action, dtype=wp.float32), timestep)
        first_raw = buffers.raw_action.numpy()
        first_l2 = buffers.action_l2.numpy()[0]
        first_rate = buffers.action_rate_l2.numpy()[0]
        buffers.process(wp.array(second_policy_action, dtype=wp.float32), timestep)
        second_raw = buffers.raw_action.numpy()
        second_l2 = buffers.action_l2.numpy()[0]
        second_rate = buffers.action_rate_l2.numpy()[0]

    np.testing.assert_allclose(first_raw, first_residual, rtol=2e-6)
    assert first_l2 == pytest.approx(float(np.dot(first_residual, first_residual)), rel=2e-6)
    assert first_rate == pytest.approx(first_l2)
    np.testing.assert_allclose(second_raw, second_residual, rtol=2e-6)
    assert second_l2 == pytest.approx(float(np.dot(second_residual, second_residual)), rel=2e-6)
    delta = second_residual - first_residual
    assert second_rate == pytest.approx(float(np.dot(delta, delta)), rel=2e-6)


def test_action_reset_does_not_modify_normalized_input_scale():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(
            _layout(),
            _reference(),
            world_count=2,
            config=ActionConfig(input_mode="normalized", normalized_mapping="rational", ema=0.3),
        )
        expected_input_scale = buffers.input_scale.numpy().copy()
        buffers.process(wp.ones(32, dtype=wp.float32), wp.zeros(2, dtype=wp.int32))
        buffers.reset(wp.array([1, 0], dtype=wp.int32))
        input_scale = buffers.input_scale.numpy()
        raw_action = buffers.raw_action.numpy().reshape(2, 16)

    np.testing.assert_array_equal(input_scale, expected_input_scale)
    assert buffers.input_mapping == "rational"
    np.testing.assert_allclose(raw_action[0], 0.0)
    assert np.any(raw_action[1] != 0.0)


def test_action_config_rejects_unknown_input_mode():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    with wp.ScopedDevice("cuda:0"), pytest.raises(ValueError, match="input_mode"):
        ActionBuffers.build(
            _layout(),
            _reference(),
            world_count=1,
            config=ActionConfig(input_mode="bounded"),  # type: ignore[arg-type]
        )


def test_action_config_rejects_unknown_normalized_mapping():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    with wp.ScopedDevice("cuda:0"), pytest.raises(ValueError, match="normalized_mapping"):
        ActionBuffers.build(
            _layout(),
            _reference(),
            world_count=1,
            config=ActionConfig(normalized_mapping="cubic"),  # type: ignore[arg-type]
        )


def test_normalized_rational_mapping_rejects_fully_frozen_ema():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    with wp.ScopedDevice("cuda:0"), pytest.raises(ValueError, match="rational mapping requires ema"):
        ActionBuffers.build(
            _layout(),
            _reference(),
            world_count=1,
            config=ActionConfig(input_mode="normalized", normalized_mapping="rational", ema=1.0),
        )


def test_normalized_mapping_rejects_unrepresentable_float32_scale():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    with wp.ScopedDevice("cuda:0"), pytest.raises(ValueError, match="residual scales"):
        ActionBuffers.build(
            _layout(),
            _reference(),
            world_count=1,
            config=ActionConfig(
                input_mode="normalized",
                normalized_mapping="rational",
                wrist_pos_scale=1.0e-300,
            ),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("wrist_pos_scale", 0.0),
        ("wrist_ori_scale", -1.0),
        ("finger_joint_pos_scale", np.inf),
        ("wrist_pos_clip", 0.0),
        ("wrist_ori_clip", -1.0),
        ("finger_joint_pos_clip", np.nan),
    ],
)
def test_action_config_requires_positive_finite_scales_and_clips(name, value):
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    with wp.ScopedDevice("cuda:0"), pytest.raises(ValueError, match=name):
        ActionBuffers.build(
            _layout(),
            _reference(),
            world_count=1,
            config=replace(ActionConfig(), **{name: value}),
        )


def test_masked_action_reset_changes_only_selected_world():
    import warp as wp

    from flash_chord.runtime.actions import ActionBuffers, ActionConfig

    with wp.ScopedDevice("cuda:0"):
        buffers = ActionBuffers.build(_layout(), _reference(), world_count=2, config=ActionConfig(ema=0.0))
        action = wp.array(np.ones(32, dtype=np.float32), dtype=wp.float32)
        timestep = wp.zeros(2, dtype=wp.int32)
        buffers.process(action, timestep)
        before = buffers.filtered_action.numpy().reshape(2, 16).copy()
        buffers.reset(wp.array([1, 0], dtype=wp.int32))
        raw = buffers.raw_action.numpy().reshape(2, 16)
        filtered = buffers.filtered_action.numpy().reshape(2, 16)
        processed = buffers.processed_target.numpy().reshape(2, 18)

    np.testing.assert_allclose(raw[0], 0.0)
    np.testing.assert_allclose(filtered[0], 0.0)
    np.testing.assert_allclose(processed[0, [3, 12]], 1.0)
    np.testing.assert_allclose(raw[1], 1.0)
    np.testing.assert_allclose(filtered[1], before[1])


def test_residual_action_routes_delayed_target_to_all_control_paths():
    import warp as wp

    from flash_chord.runtime.actions import ResidualHandPoseAction
    from flash_chord.runtime.delay import TargetDelayBuffer, TargetDelayConfig

    class _Buffers:
        def __init__(self, target):
            self.joint_target = target

        def reset(self, reset_mask):
            pass

    class _Wrist:
        def __init__(self):
            self.position_targets = []
            self.effort_targets = []

        def write_position_targets(self, control, target):
            self.position_targets.append(target.numpy().copy())

        def apply_wrist_effort(self, state, control, target):
            self.effort_targets.append(target.numpy().copy())

    with wp.ScopedDevice("cuda:0"):
        target = wp.array([1.0, 2.0, 100.0], dtype=wp.float32)
        delay = TargetDelayBuffer.build(
            world_count=1,
            num_joint_dof=3,
            delayed_dof_ids=(0, 1),
            config=TargetDelayConfig(min_steps=2, max_steps=2),
        )
        wrist = _Wrist()
        action = ResidualHandPoseAction(buffers=_Buffers(target), wrist=wrist, delay=delay)
        action.reset(wp.ones(1, dtype=wp.int32))
        action.prepare_control(None)
        action.apply_control(None, None)
        target.assign(np.array([3.0, 4.0, 101.0], dtype=np.float32))
        action.apply_control(None, None)
        action.apply_control(None, None)
        action.apply_control(None, None)

    assert len(wrist.position_targets) == 4
    np.testing.assert_allclose(wrist.position_targets[0], [1.0, 2.0, 100.0])
    np.testing.assert_allclose(wrist.position_targets[1], [1.0, 2.0, 101.0])
    np.testing.assert_allclose(wrist.position_targets[2], [1.0, 2.0, 101.0])
    np.testing.assert_allclose(wrist.position_targets[3], [3.0, 4.0, 101.0])
    for position_target, effort_target in zip(wrist.position_targets, wrist.effort_targets, strict=True):
        np.testing.assert_allclose(position_target, effort_target)
