# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hand_to_object_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when hands deviate too far from the commanded trajectory.

    Compares per-hand wrist-to-object distances against the commanded
    wrist-to-object distances and terminates if any hand exceeds
    `threshold` times its commanded distance.

    Args:
        env: The environment instance.
        command_name: The name of the command term.
        threshold: Ratio threshold for termination.

    Returns:
        Tensor of shape (num_envs,) indicating whether to terminate.
    """
    command = env.command_manager.get_term(command_name)

    right_hand_wrist_object_position_difference_command = torch.norm(
        command.right_hand_wrist_position_command_e
        - command.object_body_position_command_e,
        dim=-1,
    )
    left_hand_wrist_object_position_difference_command = torch.norm(
        command.left_hand_wrist_position_command_e
        - command.object_body_position_command_e,
        dim=-1,
    )

    right_hand_wrist_object_position_difference = torch.norm(
        command.right_robot.data.body_link_pos_w[:, command.right_wrist_body_id]
        - command.object_position_w,
        dim=-1,
    ).squeeze()
    left_hand_wrist_object_position_difference = torch.norm(
        command.left_robot.data.body_link_pos_w[:, command.left_wrist_body_id]
        - command.object_position_w,
        dim=-1,
    ).squeeze()

    right_hand_difference_ratio = (
        right_hand_wrist_object_position_difference
        / right_hand_wrist_object_position_difference_command
    )
    left_hand_difference_ratio = (
        left_hand_wrist_object_position_difference
        / left_hand_wrist_object_position_difference_command
    )

    return torch.logical_and(
        right_hand_difference_ratio > threshold,
        left_hand_difference_ratio > threshold,
    )


def hand_wrist_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when the hands are away from the trajectory."""
    command = env.command_manager.get_term(command_name)
    right_hand_position_difference = torch.norm(
        command.right_hand_wrist_position_command_e
        - command.right_hand_wrist_position_e,
        dim=-1,
    )
    left_hand_position_difference = torch.norm(
        command.left_hand_wrist_position_command_e - command.left_hand_wrist_position_e,
        dim=-1,
    )
    return torch.logical_or(
        right_hand_position_difference > threshold,
        left_hand_position_difference > threshold,
    )


def object_away_from_trajectory_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when the object is away from the trajectory.

    Args:
        env: The environment instance.
        command_name: The name of the command.
        threshold: The threshold for the termination.

    Returns:
        Tensor of shape (num_envs,) indicating whether to terminate.
    """
    command = env.command_manager.get_term(command_name)
    object_position_z_difference = torch.abs(
        command.object_body_position_command_e[..., 2]
        - command.object_position_e[..., 2].squeeze()
    )
    return object_position_z_difference > threshold


def object_away_from_trajectory(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
) -> torch.Tensor:
    """Terminate when the object is away from the trajectory."""
    command = env.command_manager.get_term(command_name)
    object_position_difference = torch.norm(
        command.object_body_position_command_e - command.object_position_e.squeeze(),
        dim=-1,
    )
    return object_position_difference > threshold


def timestep_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the command is completed."""
    command = env.command_manager.get_term(command_name)
    return command.timestep_counter >= command.retargeted_horizon - 1
