"""Observation functions for SONIC encoder/tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from robotic_grounding.tasks.v2p_whole_body.mdp.actions import SONICActionBase
from robotic_grounding.tasks.v2p_whole_body.mdp.commands import TrackingCommand


def encoder_mode(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Get encoder mode scalar index from tracking command.

    Returns:
        Encoder mode selection (num_envs, 4) [0,0,0,0] - G1 mode active
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    return command.encoder_mode


def command_joint_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    sonic_joints_only: bool = False,
    action_name: str | None = None,
    num_future_frames: int = 10,
) -> torch.Tensor:
    """Get future joint positions.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        sonic_joints_only: If True, filter to SONIC-controlled joints only (requires SONIC action term)
        action_name: Name of the action term (required when sonic_joints_only=True)
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future joint positions (num_envs, num_future_frames * num_joints)
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    joint_pos = (
        command.command_joint_pos_multi_future
    )  # (num_envs, num_future_frames, num_joints)
    joint_pos = joint_pos[:, :num_future_frames, :]

    if sonic_joints_only:
        if action_name is None:
            raise ValueError("action_name must be provided when sonic_joints_only=True")
        action_term = env.action_manager.get_term(action_name)
        if isinstance(action_term, SONICActionBase):
            sonic_joint_indices = action_term.get_sonic_joint_indices()
            joint_pos = joint_pos[:, :, sonic_joint_indices]

    return joint_pos.reshape(env.num_envs, -1)


def command_joint_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "motion",
    sonic_joints_only: bool = False,
    action_name: str | None = None,
    num_future_frames: int = 10,
) -> torch.Tensor:
    """Get future joint velocities.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        sonic_joints_only: If True, filter to SONIC-controlled joints only (requires SONIC action term)
        action_name: Name of the action term (required when sonic_joints_only=True)
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future joint velocities (num_envs, num_future_frames * num_joints)
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    joint_vel = (
        command.command_joint_vel_multi_future
    )  # (num_envs, num_future_frames, num_joints)
    joint_vel = joint_vel[:, :num_future_frames, :]

    if sonic_joints_only:
        if action_name is None:
            raise ValueError("action_name must be provided when sonic_joints_only=True")
        action_term = env.action_manager.get_term(action_name)
        if isinstance(action_term, SONICActionBase):
            sonic_joint_indices = action_term.get_sonic_joint_indices()
            joint_vel = joint_vel[:, :, sonic_joint_indices]

    return joint_vel.reshape(env.num_envs, -1)


def command_joint_pos_vel(
    env: ManagerBasedRLEnv, command_name: str = "motion", num_future_frames: int = 10
) -> torch.Tensor:
    """Get future joint positions and velocities.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future joint pos+vel (num_envs, num_future_frames * num_joints * 2)
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    multi_future = (
        command.command_multi_future
    )  # (num_envs, num_future_frames, num_joints * 2)
    multi_future = multi_future[:, :num_future_frames, :]
    return multi_future.reshape(env.num_envs, -1)


def motion_anchor_ori_b(
    env: ManagerBasedRLEnv, command_name: str = "motion", num_future_frames: int = 10
) -> torch.Tensor:
    """Get future orientation differences.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future orientation diffs (num_envs, num_future_frames * 6) - flattened 6D rotation
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    ori_diffs = (
        command.command_root_rot_dif_l_multi_future
    )  # (num_envs, num_future_frames, 6)
    ori_diffs = ori_diffs[:, :num_future_frames, :]
    return ori_diffs.reshape(env.num_envs, -1)  # Flatten


def command_z(
    env: ManagerBasedRLEnv, command_name: str = "motion", num_future_frames: int = 10
) -> torch.Tensor:
    """Get future z (height) positions.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        num_future_frames: Number of future frames to include (default: 10)

    Returns:
        Future z positions (num_envs, num_future_frames)
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    z_positions = command.command_z_multi_future  # (num_envs, num_future_frames)
    return z_positions[:, :num_future_frames]


def encoder_padding(env: ManagerBasedRLEnv, dim: int) -> torch.Tensor:
    """Zero padding for formatting tokenizer input.

    Returns:
        Zero padding (num_envs, dim)
    """
    return torch.zeros(env.num_envs, dim, device=env.device)
