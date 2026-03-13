"""Observation functions for SONIC encoder/tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import matrix_from_quat, quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from robotic_grounding.tasks.v2p_whole_body.mdp.commands import TrackingCommand


def motion_anchor_pos_b(
    env: ManagerBasedRLEnv, command_name: str = "motion", num_future_frames: int = 10
) -> torch.Tensor:
    """Get future anchor positions.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future anchor positions (num_envs, num_future_frames * 3) - flattened 3D position
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    anchor_pos = command.command_anchor_pos_multi_future
    return anchor_pos[:, :num_future_frames, :].reshape(env.num_envs, -1)


def motion_joint_pos_delta(
    env: ManagerBasedRLEnv, command_name: str = "motion", num_future_frames: int = 10
) -> torch.Tensor:
    """
    Get future joint position deltas from the robot's current joint position.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future joint position deltas (num_envs, num_future_frames * num_joints) - flattened joint position deltas
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos_delta = command.command_joint_pos_multi_future[
        :, :num_future_frames, :
    ] - command.robot_joint_pos.unsqueeze(1)
    return joint_pos_delta.reshape(env.num_envs, -1)


def motion_ee_pos_delta(
    env: ManagerBasedRLEnv, command_name: str = "motion", num_future_frames: int = 10
) -> torch.Tensor:
    """Get future EE position deltas."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ee_pos_delta = command.command_ee_pos_multi_future[
        :, :num_future_frames, :
    ] - command.robot_ee_pos_w.unsqueeze(1)
    return ee_pos_delta.reshape(env.num_envs, -1)


def motion_ee_quat_delta(
    env: ManagerBasedRLEnv, command_name: str = "motion", num_future_frames: int = 10
) -> torch.Tensor:
    """Get future EE quaternion deltas as 6D rotation representation.

    Computes the rotation difference between desired EE quaternion trajectory and current EE quaternion.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future EE rotation deltas in 6D representation (num_envs, num_ee_links * num_future_frames * 6)
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)

    # Get reference EE quaternions: (num_envs, num_future_frames, num_ee_links, 4)
    ref_ee_quat = command.command_ee_quat_multi_future[:, :num_future_frames, :, :]

    # Get current robot EE quaternions: (num_envs, num_ee_links, 4)
    robot_ee_quat = command.robot_ee_quat_w

    num_envs = env.num_envs
    num_ee_links = robot_ee_quat.shape[1]

    # Compute rotation difference: quat_inv(robot_quat) * ref_quat
    robot_ee_quat_inv = quat_inv(robot_ee_quat)  # (num_envs, num_ee_links, 4)
    robot_ee_quat_inv = robot_ee_quat_inv.unsqueeze(1).expand(
        -1, num_future_frames, -1, -1
    )

    # Compute quaternion difference
    ee_quat_diff = quat_mul(
        robot_ee_quat_inv, ref_ee_quat
    )  # (num_envs, num_future_frames, num_ee_links, 4)

    # Convert to 6D rotation representation (first two columns of rotation matrix)
    mat = matrix_from_quat(
        ee_quat_diff
    )  # (num_envs, num_future_frames, num_ee_links, 3, 3)
    ee_rot_6d = mat[..., :2].reshape(num_envs, num_future_frames, num_ee_links, 6)

    return ee_rot_6d.reshape(num_envs, -1)


def object_pos_delta(
    env: ManagerBasedRLEnv, command_name: str = "motion", num_future_frames: int = 10
) -> torch.Tensor:
    """Get future object position deltas.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future object position deltas (num_envs, num_future_frames * 3) - flattened object position deltas
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    object_pos_delta = command.command_object_pos_multi_future[
        :, :num_future_frames, :
    ] - command.object_pos_w.unsqueeze(1)
    return object_pos_delta.reshape(env.num_envs, -1)


def command_trajectory_progress(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """
    Normalized trajectory progress.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term

    Returns:
        Normalized trajectory progress (num_envs, 1)
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    trajectory_progress = command.timestep.float() / max(command.num_timesteps - 1, 1)
    return trajectory_progress.unsqueeze(-1)
