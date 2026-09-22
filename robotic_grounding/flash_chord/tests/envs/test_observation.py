# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact tests for device-native policy observation assembly."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _frames(side: str, body_id: int, palm=None) -> dict:
    from flash_chord.embodiments.base import BodyFrame

    return {
        "palm_frame": palm or BodyFrame(f"{side}_palm", body_id),
        "dp_frames": (BodyFrame(f"{side}_index_DP", body_id),),
        "fingertip_frames": (BodyFrame(f"{side}_index_fingertip", body_id),),
    }


class _PolicyAction:
    sides = ("right", "left")
    action_dim = 4
    processed_dim = 6

    def __init__(self, wp):
        self.joint_target = wp.zeros(2, dtype=wp.float32)
        self.raw_action = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
        self.processed_target = wp.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=wp.float32)
        self.action_l2 = wp.zeros(1, dtype=wp.float32)
        self.action_rate_l2 = wp.zeros(1, dtype=wp.float32)

    def process(self, action, timestep, state=None):
        pass

    def prepare_control(self, control):
        pass

    def apply_control(self, state, control):
        pass

    def reset(self, reset_mask):
        pass


def _build_inputs(wp, *, right_palm=None, left_palm=None):
    from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout
    from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding

    right_palm = right_palm or BodyFrame("right_palm", 0)
    left_palm = left_palm or BodyFrame("left_palm", 1)
    embodiment = EmbodimentLayout(
        num_joint_q=2,
        num_joint_dof=2,
        hands=(
            HandLayout(
                side="left",
                **_frames("left", 1, left_palm),
                finger_dof_ids=(1,),
                finger_q_ids=(1,),
                finger_joint_names=("left_finger",),
            ),
            HandLayout(
                side="right",
                **_frames("right", 0, right_palm),
                finger_dof_ids=(0,),
                finger_q_ids=(0,),
                finger_joint_names=("right_finger",),
            ),
        ),
    )
    model = SimpleNamespace(
        body_count=3,
        joint_coord_count=2,
        joint_dof_count=2,
        joint_limit_lower=wp.array([-1.0, 0.0], dtype=wp.float32),
        joint_limit_upper=wp.array([1.0, 2.0], dtype=wp.float32),
    )
    root = np.sqrt(0.5)
    keypoint_position = np.zeros((2, 1, 3), dtype=np.float32)
    keypoint_orientation = np.zeros((2, 1, 4), dtype=np.float32)
    keypoint_orientation[..., 0] = 1.0
    bound_hands = (
        BoundHandReference(
            side="left",
            wrist_pos_w=np.array([[0.0, 0.0, 0.0], [-1.0, 1.0, 2.0]], dtype=np.float32),
            wrist_quat_w=np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            arm_joint_pos=np.empty((2, 0), dtype=np.float32),
            finger_joint_pos=np.array([[0.0], [1.5]], dtype=np.float32),
            dp_pos_w=keypoint_position,
            dp_quat_w=keypoint_orientation,
            fingertip_pos_w=keypoint_position,
            fingertip_quat_w=keypoint_orientation,
        ),
        BoundHandReference(
            side="right",
            wrist_pos_w=np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 3.0]], dtype=np.float32),
            wrist_quat_w=np.array(
                [[1.0, 0.0, 0.0, 0.0], [root, 0.0, 0.0, root]],
                dtype=np.float32,
            ),
            arm_joint_pos=np.empty((2, 0), dtype=np.float32),
            finger_joint_pos=np.array([[0.0], [0.75]], dtype=np.float32),
            dp_pos_w=keypoint_position,
            dp_quat_w=keypoint_orientation,
            fingertip_pos_w=keypoint_position,
            fingertip_quat_w=keypoint_orientation,
        ),
    )
    robot_reference = RobotReferenceBinding(
        num_frames=2,
        fps=20.0,
        num_joint_q=2,
        num_joint_dof=2,
        joint_q=np.array([[0.0, 0.0], [0.75, 1.5]], dtype=np.float32),
        joint_target=np.array([[0.0, 0.0], [0.75, 1.5]], dtype=np.float32),
        hands=bound_hands,
    )
    scene = SimpleNamespace(
        model=model,
        layout=embodiment,
        robot_reference=robot_reference,
        world_count=1,
    )
    action = _PolicyAction(wp)
    command = SimpleNamespace(
        world_count=1,
        layout=SimpleNamespace(num_bodies=1),
        body_ids=wp.array([2], dtype=wp.int32),
        reference=SimpleNamespace(
            num_frames=2,
            body_pos_w=wp.array([[0.0, 0.0, 0.0], [3.0, -1.0, 0.5]], dtype=wp.vec3),
            body_quat_w=wp.array(
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]],
                dtype=wp.quat,
            ),
        ),
    )
    contact_layout = SimpleNamespace(
        sides=("right", "left"),
        num_objects=1,
        object_body_ids=(2,),
        link_counts=(2, 1),
        hand_slot_starts=(0, 2),
        slots_per_world=3,
    )
    contact = SimpleNamespace(
        layout=contact_layout,
        contact_pos_b=wp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]], dtype=wp.vec3),
        contact_force_direction_b=wp.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=wp.vec3,
        ),
    )
    state = SimpleNamespace(
        body_q=wp.array(
            [
                wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat(0.0, 0.0, root, root)),
                wp.transform(wp.vec3(-1.0, 0.0, 2.0), wp.quat(0.0, 0.0, 0.0, -1.0)),
                wp.transform(wp.vec3(2.0, -1.0, 0.5), wp.quat_identity()),
            ],
            dtype=wp.transform,
        ),
        body_qd=wp.array(
            [
                wp.spatial_vector(0.0, 1.0, 0.0, -1.0, 0.0, 0.0),
                wp.spatial_vector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
                wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ],
            dtype=wp.spatial_vector,
        ),
        joint_q=wp.array([0.5, 0.5], dtype=wp.float32),
        joint_qd=wp.array([1.5, -2.5], dtype=wp.float32),
    )
    timestep = wp.array([1], dtype=wp.int32)
    episode_step = wp.array([1], dtype=wp.int32)
    return scene, action, command, contact, state, timestep, episode_step


def _build_articulated_inputs(wp):
    from flash_chord.embodiments.base import EmbodimentLayout, HandLayout
    from flash_chord.embodiments.binding import RobotReferenceBinding

    original, action, command, contact, state, timestep, episode_step = _build_inputs(wp)
    layout = EmbodimentLayout(
        num_joint_q=6,
        num_joint_dof=6,
        hands=(
            HandLayout(
                side="left",
                **_frames("left", 1),
                arm_q_ids=(4, 1),
                arm_dof_ids=(0, 5),
                arm_joint_names=("left_arm_4", "left_arm_1"),
                finger_q_ids=(5,),
                finger_dof_ids=(2,),
                finger_joint_names=("left_finger",),
            ),
            HandLayout(
                side="right",
                **_frames("right", 0),
                arm_q_ids=(3, 0),
                arm_dof_ids=(4, 1),
                arm_joint_names=("right_arm_3", "right_arm_0"),
                finger_q_ids=(2,),
                finger_dof_ids=(3,),
                finger_joint_names=("right_finger",),
            ),
        ),
    )
    joint_q = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        ],
        dtype=np.float32,
    )
    left = original.robot_reference.hand("left")
    right = original.robot_reference.hand("right")
    robot_reference = RobotReferenceBinding(
        num_frames=2,
        fps=20.0,
        num_joint_q=6,
        num_joint_dof=6,
        joint_q=joint_q,
        joint_target=np.zeros((2, 6), dtype=np.float32),
        hands=(
            replace(
                left,
                arm_joint_pos=joint_q[:, (4, 1)],
                finger_joint_pos=joint_q[:, (5,)],
            ),
            replace(
                right,
                arm_joint_pos=joint_q[:, (3, 0)],
                finger_joint_pos=joint_q[:, (2,)],
            ),
        ),
    )
    model = SimpleNamespace(
        body_count=3,
        joint_coord_count=6,
        joint_dof_count=6,
        joint_limit_lower=wp.array([-2.0] * 6, dtype=wp.float32),
        joint_limit_upper=wp.array([2.0] * 6, dtype=wp.float32),
    )
    scene = SimpleNamespace(
        model=model,
        layout=layout,
        robot_reference=robot_reference,
        world_count=1,
    )
    state.joint_q = wp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=wp.float32)
    state.joint_qd = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=wp.float32)
    return scene, action, command, contact, state, timestep, episode_step


def test_policy_observation_exact_layout_and_values():
    import warp as wp

    from flash_chord.envs.observation import PolicyObservation

    with wp.ScopedDevice("cuda:0"):
        scene, action, command, contact, state, timestep, episode_step = _build_inputs(wp)
        policy = PolicyObservation.build(scene, action, command, contact, timestep, episode_step)
        observation = policy.compute(state).numpy()
        layout = policy.layout
        episode_step.assign(np.array([0], dtype=np.int32))
        reset_observation = policy.compute(state).numpy()

    assert layout.observation_dim == 88
    assert layout.arm_position_start == -1
    assert layout.arm_velocity_start == -1
    assert layout.arm_reference_delta_start == -1
    assert layout.object_velocity_start == -1
    assert policy.block_names == (
        "wrist_position_e_m",
        "wrist_orientation_e_wxyz",
        "wrist_velocity_b_mps_radps",
        "finger_joint_position_normalized",
        "finger_joint_velocity_radps",
        "object_position_e_m",
        "object_orientation_e_wxyz",
        "command_wrist_relative_pose",
        "command_finger_delta_rad",
        "command_object_relative_position_m",
        "command_object_relative_orientation_wxyz",
        "action_raw",
        "action_processed",
        "contact_left_position_wrist_m",
        "contact_left_force_direction_wrist_unit",
        "contact_right_position_wrist_m",
        "contact_right_force_direction_wrist_unit",
    )
    assert policy.block_ranges[0][0] == 0
    assert policy.block_ranges[-1][1] == layout.observation_dim
    assert all(left[1] == right[0] for left, right in zip(policy.block_ranges, policy.block_ranges[1:]))
    np.testing.assert_allclose(observation[layout.wrist_pos_start : layout.wrist_quat_start], [1, 2, 3, -1, 0, 2])
    np.testing.assert_allclose(
        observation[layout.wrist_quat_start : layout.wrist_velocity_start],
        [np.sqrt(0.5), 0, 0, np.sqrt(0.5), 1, 0, 0, 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        observation[layout.wrist_velocity_start : layout.finger_pos_start],
        [1, 0, 0, 0, 1, 0, 1, 2, 3, 4, 5, 6],
        atol=1e-6,
    )
    np.testing.assert_allclose(observation[layout.finger_pos_start : layout.finger_velocity_start], [0.5, -0.5])
    np.testing.assert_allclose(observation[layout.finger_velocity_start : layout.object_pos_start], [1.5, -2.5])
    np.testing.assert_allclose(observation[layout.object_pos_start : layout.object_quat_start], [2, -1, 0.5])
    np.testing.assert_allclose(observation[layout.object_quat_start : layout.command_wrist_start], [1, 0, 0, 0])
    np.testing.assert_allclose(
        observation[layout.command_wrist_start : layout.command_finger_start],
        [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(observation[layout.command_finger_start : layout.command_object_pos_start], [0.25, 1])
    np.testing.assert_allclose(
        observation[layout.command_object_pos_start : layout.command_object_quat_start], [1, 0, 0]
    )
    np.testing.assert_allclose(
        observation[layout.command_object_quat_start : layout.raw_action_start],
        [np.sqrt(0.5), 0, 0, np.sqrt(0.5)],
        atol=1e-6,
    )
    np.testing.assert_allclose(observation[layout.raw_action_start : layout.processed_action_start], [1, 2, 3, 4])
    np.testing.assert_allclose(
        observation[layout.processed_action_start : layout.contact_position_starts[0]], [5, 6, 7, 8, 9, 10]
    )
    np.testing.assert_allclose(
        observation[layout.contact_position_starts[0] : layout.contact_direction_starts[0]],
        [30, 0, 0],
    )
    np.testing.assert_allclose(
        observation[layout.contact_direction_starts[0] : layout.contact_position_starts[1]],
        [0, 0, 1],
    )
    np.testing.assert_allclose(
        observation[layout.contact_position_starts[1] : layout.contact_direction_starts[1]],
        [10, 0, 0, 20, 0, 0],
    )
    np.testing.assert_allclose(
        observation[layout.contact_direction_starts[1] : layout.observation_dim],
        [1, 0, 0, 0, 1, 0],
    )
    np.testing.assert_allclose(reset_observation[layout.contact_position_starts[0] :], 0.0)


def test_object_body_velocity_term_uses_mapped_world_velocity_and_scale():
    import warp as wp

    from flash_chord.envs.observation import ObservationTermConfig, PolicyObservation, PolicyObservationConfig

    config = PolicyObservationConfig(object_body_velocity=ObservationTermConfig(scale=0.5))
    with wp.ScopedDevice("cuda:0"):
        scene, action, command, contact, state, timestep, episode_step = _build_inputs(wp)
        state.body_qd = wp.array(
            [
                wp.spatial_vector(0.0),
                wp.spatial_vector(0.0),
                wp.spatial_vector(2.0, -4.0, 6.0, 8.0, -10.0, 12.0),
            ],
            dtype=wp.spatial_vector,
        )
        policy = PolicyObservation.build(
            scene,
            action,
            command,
            contact,
            timestep,
            episode_step,
            config=config,
        )
        observation = policy.compute(state).numpy()
        layout = policy.layout

    assert policy.observation_dim == 94
    assert policy.block_names[7] == "object_body_velocity_w_mps_radps"
    assert layout.object_velocity_start == layout.object_quat_start + 4
    assert layout.command_wrist_start == layout.object_velocity_start + 6
    np.testing.assert_allclose(
        observation[layout.object_velocity_start : layout.command_wrist_start],
        [1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
    )


def test_articulated_arm_terms_follow_named_layout_and_reference_order():
    import warp as wp

    from flash_chord.envs.observation import ObservationTermConfig, PolicyObservation, PolicyObservationConfig

    config = PolicyObservationConfig(
        arm_joint_position=ObservationTermConfig(scale=2.0),
        arm_joint_velocity=ObservationTermConfig(scale=0.1),
        arm_joint_reference_delta=ObservationTermConfig(scale=-3.0),
    )
    with wp.ScopedDevice("cuda:0"):
        scene, action, command, contact, state, timestep, episode_step = _build_articulated_inputs(wp)
        policy = PolicyObservation.build(
            scene,
            action,
            command,
            contact,
            timestep,
            episode_step,
            config=config,
        )
        observation = policy.compute(state).numpy()
        layout = policy.layout

    arm_order = "right_arm_3,right_arm_0,left_arm_4,left_arm_1"
    assert policy.block_names[3:6] == (
        f"arm_joint_position_rad[{arm_order}]",
        f"arm_joint_velocity_radps[{arm_order}]",
        f"arm_joint_reference_delta_rad[{arm_order}]",
    )
    np.testing.assert_allclose(
        observation[layout.arm_position_start : layout.arm_velocity_start],
        [0.8, 0.2, 1.0, 0.4],
    )
    np.testing.assert_allclose(
        observation[layout.arm_velocity_start : layout.arm_reference_delta_start],
        [0.5, 0.2, 0.1, 0.6],
    )
    np.testing.assert_allclose(
        observation[layout.arm_reference_delta_start : layout.finger_pos_start],
        [-3.0, -3.0, -3.0, -3.0],
    )


def test_arm_terms_reject_an_armless_embodiment():
    import warp as wp

    from flash_chord.envs.observation import ObservationTermConfig, PolicyObservationConfig, PolicyObservationLayout

    config = PolicyObservationConfig(arm_joint_position=ObservationTermConfig())
    with wp.ScopedDevice("cuda:0"):
        scene, action, command, contact, _, _, _ = _build_inputs(wp)
        with pytest.raises(ValueError, match="articulated arm joints"):
            PolicyObservationLayout.build(scene, action, command, contact, config)


def test_policy_observation_uses_semantic_palm_pose_command_and_offset_velocity():
    import warp as wp

    from flash_chord.embodiments.base import BodyFrame
    from flash_chord.envs.observation import PolicyObservation

    sine = float(np.sqrt(0.5))
    right_palm = BodyFrame(
        "right_palm",
        body_id=0,
        body_to_frame_pos=(1.0, 0.0, 0.0),
        body_to_frame_quat_xyzw=(0.0, 0.0, sine, sine),
    )
    with wp.ScopedDevice("cuda:0"):
        scene, action, command, contact, state, timestep, episode_step = _build_inputs(
            wp,
            right_palm=right_palm,
        )
        state.body_q = wp.array(
            [wp.transform_identity(), wp.transform_identity(), wp.transform_identity()],
            dtype=wp.transform,
        )
        state.body_qd = wp.array(
            [
                wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                wp.spatial_vector(0.0),
                wp.spatial_vector(0.0),
            ],
            dtype=wp.spatial_vector,
        )
        policy = PolicyObservation.build(scene, action, command, contact, timestep, episode_step)
        observation = policy.compute(state).numpy()
        layout = policy.layout

    np.testing.assert_allclose(observation[layout.wrist_pos_start : layout.wrist_pos_start + 3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        observation[layout.wrist_quat_start : layout.wrist_quat_start + 4],
        [sine, 0.0, 0.0, sine],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        observation[layout.wrist_velocity_start : layout.wrist_velocity_start + 6],
        [1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        observation[layout.command_wrist_start : layout.command_wrist_start + 7],
        [3.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        atol=1.0e-6,
    )


def test_policy_observation_can_write_a_caller_buffer_without_touching_primary():
    import warp as wp

    from flash_chord.envs.observation import ObservationInto, PolicyObservation

    with wp.ScopedDevice("cuda:0"):
        scene, action, command, contact, state, timestep, episode_step = _build_inputs(wp)
        policy = PolicyObservation.build(scene, action, command, contact, timestep, episode_step)
        expected = policy.compute(state).numpy().copy()
        policy.observation.assign(np.full(policy.world_count * policy.observation_dim, -123.0, dtype=np.float32))
        alternate = wp.zeros(policy.world_count * policy.observation_dim, dtype=wp.float32)
        result = policy.compute_into(state, alternate)

        assert isinstance(policy, ObservationInto)
        assert result is alternate
        np.testing.assert_allclose(alternate.numpy(), expected, atol=1.0e-6)
        np.testing.assert_array_equal(policy.observation.numpy(), -123.0)
        with pytest.raises(ValueError, match="observation output has shape"):
            policy.compute_into(state, wp.zeros(policy.observation_dim - 1, dtype=wp.float32))


def test_layout_can_remove_contacts_without_removing_objects():
    import warp as wp

    from flash_chord.envs.observation import PolicyObservationConfig, PolicyObservationLayout

    with wp.ScopedDevice("cuda:0"):
        scene, action, command, _, _, _, _ = _build_inputs(wp)
        layout = PolicyObservationLayout.build(
            scene,
            action,
            command,
            contact=None,
            config=PolicyObservationConfig(contact_sides=()),
        )

    assert layout.num_objects == 1
    assert layout.contact_position_starts == ()
    assert layout.observation_dim == 70


def test_named_terms_can_be_removed_or_masked_without_reordering_remaining_blocks():
    import warp as wp

    from flash_chord.envs.observation import ObservationTermConfig, PolicyObservation, PolicyObservationConfig

    config = PolicyObservationConfig(
        wrist_velocity=ObservationTermConfig(enabled=False),
        contact_position=ObservationTermConfig(enabled=False),
        contact_direction=ObservationTermConfig(scale=0.0),
    )
    with wp.ScopedDevice("cuda:0"):
        scene, action, command, contact, state, timestep, episode_step = _build_inputs(wp)
        policy = PolicyObservation.build(
            scene,
            action,
            command,
            contact,
            timestep,
            episode_step,
            config=config,
        )
        observation = policy.compute(state).numpy()

    assert policy.observation_dim == 67
    assert "wrist_velocity_b_mps_radps" not in policy.block_names
    assert not any("position_wrist" in name for name in policy.block_names)
    direction_ranges = [
        bounds
        for name, bounds in zip(policy.block_names, policy.block_ranges, strict=True)
        if "force_direction" in name
    ]
    assert len(direction_ranges) == 2
    for start, end in direction_ranges:
        np.testing.assert_array_equal(observation[start:end], 0.0)
    assert all(left[1] == right[0] for left, right in zip(policy.block_ranges, policy.block_ranges[1:]))


def test_disabled_contact_terms_do_not_require_contact_tracker():
    import warp as wp

    from flash_chord.envs.observation import ObservationTermConfig, PolicyObservationConfig, PolicyObservationLayout

    config = PolicyObservationConfig(
        contact_position=ObservationTermConfig(enabled=False),
        contact_direction=ObservationTermConfig(enabled=False),
    )
    with wp.ScopedDevice("cuda:0"):
        scene, action, command, _, _, _, _ = _build_inputs(wp)
        layout = PolicyObservationLayout.build(scene, action, command, contact=None, config=config)

    assert layout.observation_dim == 70
    assert not any(name.startswith("contact_") for name in layout.block_names)


def test_observation_term_config_rejects_nonfinite_scale():
    from flash_chord.envs.observation import ObservationTermConfig

    with pytest.raises(ValueError, match="finite"):
        ObservationTermConfig(scale=float("nan"))
