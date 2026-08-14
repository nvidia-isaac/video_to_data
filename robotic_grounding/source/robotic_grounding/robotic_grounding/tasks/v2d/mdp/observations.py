# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv


def finger_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Finger joint positions.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, num_fingers) with finger joint positions.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            math_utils.scale_transform(
                command.right_hand_finger_joint_pos,
                command.right_robot.data.joint_pos_limits[..., 0],
                command.right_robot.data.joint_pos_limits[..., 1],
            ),
            math_utils.scale_transform(
                command.left_hand_finger_joint_pos,
                command.left_robot.data.joint_pos_limits[..., 0],
                command.left_robot.data.joint_pos_limits[..., 1],
            ),
        ],
        dim=-1,
    ).float()


def finger_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Finger joint velocity.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, num_fingers) with finger joint velocities.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.right_hand_finger_joint_vel,
            command.left_hand_finger_joint_vel,
        ],
        dim=-1,
    ).float()


def wrist_position_e(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Wrist position in the environment frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, 3 x NUM_HANDS) with wrist position in the environment frame.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.right_hand_wrist_position_e,
            command.left_hand_wrist_position_e,
        ],
        dim=-1,
    )


def wrist_orientation_e(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Wrist orientation in the environment frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, 4 x NUM_HANDS) with wrist orientation in the environment frame.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.right_hand_wrist_wxyz_e,
            command.left_hand_wrist_wxyz_e,
        ],
        dim=-1,
    )


def wrist_velocity_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Wrist velocity in the body frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.right_hand_wrist_velocity_b,
            command.left_hand_wrist_velocity_b,
        ],
        dim=-1,
    ).float()


def object_position_e(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object position in the environment frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.
    """
    command = env.command_manager.get_term(command_name)
    return command.object_position_e.reshape(command.num_envs, -1)


def object_orientation_e(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object orientation in the environment frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.
    """
    command = env.command_manager.get_term(command_name)
    return command.object_orientation_e.reshape(command.num_envs, -1)


def prev_action(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """The previous actions to the environment.

    The name of the action term for which the action is required.
    """
    return env.action_manager.get_term(action_name).prev_actions


def processed_action(env: ManagerBasedEnv, action_name: str) -> torch.Tensor:
    """The processed actions to the environment.

    The name of the action term for which the action is required.
    """
    return env.action_manager.get_term(action_name).processed_actions


def contact_position_direction_in_wrist(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """Wrist position in contact position.

    Args:
        env: The environment instance.
        command_name: The name of the command.
    """
    command = env.command_manager.get_term(command_name)

    # Extract and expand the wrist position and orientation
    left_hand_wrist_position_w = command.left_hand_wrist_position_w.reshape(
        env.num_envs, 1, 1, 3
    ).expand(
        -1, command.num_bodies, command.num_robot_contacts_left, 3
    )  # (num_envs, num_bodies, num_hand_link_w_sensor, 3)
    left_hand_wrist_wxyz_w = command.left_hand_wrist_wxyz_e.reshape(
        env.num_envs, 1, 1, 4
    ).expand(
        -1, command.num_bodies, command.num_robot_contacts_left, 4
    )  # (num_envs, num_bodies, num_hand_link_w_sensor, 4)

    right_hand_wrist_position_w = command.right_hand_wrist_position_w.reshape(
        env.num_envs, 1, 1, 3
    ).expand(
        -1, command.num_bodies, command.num_robot_contacts_right, 3
    )  # (num_envs, num_bodies, num_hand_link_w_sensor, 3)
    right_hand_wrist_wxyz_w = command.right_hand_wrist_wxyz_e.reshape(
        env.num_envs, 1, 1, 4
    ).expand(
        -1, command.num_bodies, command.num_robot_contacts_right, 4
    )  # (num_envs, num_bodies, num_hand_link_w_sensor, 4)

    # Get valid contact mask
    left_active_contact = (
        command.left_hand_object_contact_positions_w.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies, num_hand_link_w_sensor)
    right_active_contact = (
        command.right_hand_object_contact_positions_w.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies, num_hand_link_w_sensor)

    # Compute contact position in wrist frame
    wrist_p_left_contact_positions, _ = math_utils.subtract_frame_transforms(
        left_hand_wrist_position_w,  # world_t_wrist
        left_hand_wrist_wxyz_w,
        command.left_hand_object_contact_positions_w,  # world_t_contact
        q02=None,
    )  # (num_envs, num_bodies, num_hand_link_w_sensor, 3)
    wrist_p_left_contact_positions *= left_active_contact.unsqueeze(-1)

    wrist_p_right_contact_positions, _ = math_utils.subtract_frame_transforms(
        right_hand_wrist_position_w,  # world_t_wrist
        right_hand_wrist_wxyz_w,
        command.right_hand_object_contact_positions_w,  # world_t_contact
        q02=None,
    )  # (num_envs, num_bodies, num_hand_link_w_sensor, 3)
    wrist_p_right_contact_positions *= right_active_contact.unsqueeze(-1)

    # Compute contact force direction in wrist frame
    left_contact_direction_w = command.left_hand_object_contact_forces_w[
        :, 0
    ]  # (num_envs, num_bodies, num_hand_link_w_sensor, 3)
    left_contact_direction_w = left_contact_direction_w / left_contact_direction_w.norm(
        dim=-1
    ).unsqueeze(-1).clamp(min=1e-5)
    wrist_q_left_contact_direction, _ = math_utils.subtract_frame_transforms(
        torch.zeros_like(left_hand_wrist_position_w),
        left_hand_wrist_wxyz_w,
        left_contact_direction_w,
        q02=None,
    )
    wrist_q_left_contact_direction *= left_active_contact.unsqueeze(-1)

    right_contact_direction_w = command.right_hand_object_contact_forces_w[
        :, 0
    ]  # (num_envs, num_bodies, num_hand_link_w_sensor, 3)
    right_contact_direction_w = (
        right_contact_direction_w
        / right_contact_direction_w.norm(dim=-1).unsqueeze(-1).clamp(min=1e-5)
    )
    wrist_q_right_contact_direction, _ = math_utils.subtract_frame_transforms(
        torch.zeros_like(right_hand_wrist_position_w),
        right_hand_wrist_wxyz_w,
        right_contact_direction_w,
        q02=None,
    )
    wrist_q_right_contact_direction *= right_active_contact.unsqueeze(-1)

    # Combine contact position and direction
    return torch.cat(
        [
            wrist_p_left_contact_positions.reshape(env.num_envs, -1),
            wrist_q_left_contact_direction.reshape(env.num_envs, -1),
            wrist_p_right_contact_positions.reshape(env.num_envs, -1),
            wrist_q_right_contact_direction.reshape(env.num_envs, -1),
        ],
        dim=-1,
    )
