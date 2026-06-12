# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observation functions for SONIC encoder/tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

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
    anchor_pos = command.command_anchor_pos_w_multi_future
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
    ee_pos_delta = command.command_ee_pos_w_multi_future[
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
    ref_ee_quat = command.command_ee_quat_w_multi_future[:, :num_future_frames, :, :]

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
    object_pos_delta = command.command_object_pos_w_multi_future[
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


def object_pose_delta(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Object pose delta (pos + quat) in current object frame. Shape: (num_envs, 7)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_delta, quat_delta = subtract_frame_transforms(
        command.object_pos_w,
        command.object_quat_w,
        command.command_object_pos_w,
        command.command_object_quat_w,
    )
    return torch.cat([pos_delta, quat_delta], dim=-1)


def wrist_position_b(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Wrist positions in body (pelvis) frame. (num_envs, 6) = [right(3), left(3)]."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pelvis_pos_w = command.robot_anchor_pos_w
    pelvis_quat_inv = quat_inv(command.robot_anchor_quat_w)
    rot = matrix_from_quat(pelvis_quat_inv)

    def _to_body(pos_w: torch.Tensor) -> torch.Tensor:
        delta = pos_w - pelvis_pos_w
        return torch.bmm(rot, delta.unsqueeze(-1)).squeeze(-1)

    return torch.cat(
        [
            _to_body(command.right_hand_wrist_position_e + env.scene.env_origins),
            _to_body(command.left_hand_wrist_position_e + env.scene.env_origins),
        ],
        dim=-1,
    )


def wrist_orientation_b(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Wrist orientations relative to pelvis as 6D rotation. (num_envs, 12) = [right(6), left(6)]."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pelvis_quat_inv = quat_inv(command.robot_anchor_quat_w)

    def _to_body_6d(wrist_quat_w: torch.Tensor) -> torch.Tensor:
        rel_quat = quat_mul(pelvis_quat_inv, wrist_quat_w)
        return matrix_from_quat(rel_quat)[..., :2].reshape(-1, 6)

    right_idx = 1 if command.robot_ee_quat_w.shape[1] > 1 else 0
    return torch.cat(
        [
            _to_body_6d(command.robot_ee_quat_w[:, right_idx]),
            _to_body_6d(command.robot_ee_quat_w[:, 0]),
        ],
        dim=-1,
    )


def wrist_velocity_b(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Wrist linear velocities in body (pelvis) frame. (num_envs, 6) = [right(3), left(3)]."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot = command.robot
    pelvis_quat_inv = quat_inv(command.robot_anchor_quat_w)
    rot = matrix_from_quat(pelvis_quat_inv)

    ee_ids = command.ee_link_ids or []
    if len(ee_ids) >= 2:
        left_vel_w = robot.data.body_vel_w[:, ee_ids[0], :3]
        right_vel_w = robot.data.body_vel_w[:, ee_ids[1], :3]
    else:
        return torch.zeros(env.num_envs, 6, device=env.device)

    right_vel_b = torch.bmm(rot, right_vel_w.unsqueeze(-1)).squeeze(-1)
    left_vel_b = torch.bmm(rot, left_vel_w.unsqueeze(-1)).squeeze(-1)
    return torch.cat([right_vel_b, left_vel_b], dim=-1)


def object_position_b(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Object position in body (pelvis) frame. (num_envs, 3)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pelvis_pos_w = command.robot_anchor_pos_w
    pelvis_quat_inv = quat_inv(command.robot_anchor_quat_w)
    rot = matrix_from_quat(pelvis_quat_inv)
    obj_pos_w = command.object.data.root_pos_w
    delta = obj_pos_w - pelvis_pos_w
    return torch.bmm(rot, delta.unsqueeze(-1)).squeeze(-1)


def object_orientation_b(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Object orientation relative to pelvis as 6D rotation. (num_envs, 6)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pelvis_quat_inv = quat_inv(command.robot_anchor_quat_w)
    obj_quat_w = command.object.data.root_quat_w
    rel_quat = quat_mul(pelvis_quat_inv, obj_quat_w)
    return matrix_from_quat(rel_quat)[..., :2].reshape(env.num_envs, 6)


def object_pose_delta_6d(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Object pose delta (pos + 6D rot) in current object frame. Shape: (num_envs, 9)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    pos_delta, quat_delta = subtract_frame_transforms(
        command.object_pos_w,
        command.object_quat_w,
        command.command_object_pos_w,
        command.command_object_quat_w,
    )
    rot_6d = matrix_from_quat(quat_delta)[..., :2].reshape(env.num_envs, 6)
    return torch.cat([pos_delta, rot_6d], dim=-1)


def hand_object_transform_6d(
    env: ManagerBasedRLEnv, frame_transform_cfg: Any, threshold: float = 10.0
) -> torch.Tensor:
    """Hand-object transform with 6D rotation. (num_envs, 9) = pos(3) + 6D rot(6).

    Zeroed when distance exceeds threshold.
    """
    frame_transform = env.scene[frame_transform_cfg.name]
    pos = frame_transform.data.target_pos_source  # (E, 1, 3)
    quat = frame_transform.data.target_quat_source  # (E, 1, 4)
    rot_6d = matrix_from_quat(quat)[..., :2].reshape(pos.shape[0], pos.shape[1], 6)
    transform = torch.cat([pos, rot_6d], dim=-1)  # (E, 1, 9)

    distance = torch.norm(pos, dim=-1)  # (E, 1)
    mask = (distance < threshold).unsqueeze(-1)
    return torch.where(mask, transform, torch.zeros_like(transform)).reshape(
        env.num_envs, -1
    )


def action_history(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Past actions from the tracking command's action history buffer.

    Shape: (num_envs, history_len * action_dim). Zeroed on reset.
    The action term must call command.update_action_history() each step.
    """
    command: TrackingCommand = env.command_manager.get_term(command_name)
    return command.action_history


def contact_desired_positions_e(
    env: ManagerBasedRLEnv, command_name: str = "motion"
) -> torch.Tensor:
    """Desired contact positions in env frame (left + right, flattened)."""
    command: TrackingCommand = env.command_manager.get_term(command_name)
    left = command.left_hand_object_contact_command_positions_e
    right = command.right_hand_object_contact_command_positions_e
    return torch.cat(
        [left.reshape(env.num_envs, -1), right.reshape(env.num_envs, -1)], dim=-1
    )
