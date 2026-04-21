from __future__ import annotations

from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_error_magnitude


def _mask_freeze(command: Any, terminated: torch.Tensor) -> torch.Tensor:
    """Suppress termination during the post-reset freeze period."""
    freeze_steps = getattr(command.cfg, "reset_freeze_steps", 0)
    if freeze_steps > 0:
        in_freeze = command.steps_since_last_reset <= freeze_steps
        return terminated & ~in_freeze
    return terminated


def timestep_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the trajectory is completed."""
    command = env.command_manager.get_term(command_name)
    return command.timestep >= command.num_timesteps - 1


def anchor_pos_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.7,
) -> torch.Tensor:
    """Terminate when the anchor position is too far from the command."""
    command = env.command_manager.get_term(command_name)
    error = torch.norm(
        command.robot_anchor_pos_w - command.command_anchor_pos_w, dim=-1
    )
    return error > threshold


def anchor_quat_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.7,
) -> torch.Tensor:
    """Terminate when the anchor orientation is too far from the command."""
    command = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(
        command.robot_anchor_quat_w, command.command_anchor_quat_w
    )
    return error > threshold


def ee_position_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.15,
) -> torch.Tensor:
    """Terminate when either EE is too far from the command. Freeze-aware."""
    command = env.command_manager.get_term(command_name)
    # Per-link error, terminate if ANY link exceeds threshold
    error = torch.norm(
        command.command_ee_pos_w - command.robot_ee_pos_w, dim=-1
    )  # (num_envs, num_ee)
    terminated = error.max(dim=-1).values > threshold
    return _mask_freeze(command, terminated)


def ee_quat_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 1.5,
) -> torch.Tensor:
    """Terminate when either EE orientation is too far from the command. Freeze-aware."""
    command = env.command_manager.get_term(command_name)
    cmd_flat = command.command_ee_quat_w.reshape(-1, 4)
    robot_flat = command.robot_ee_quat_w.reshape(-1, 4)
    error = quat_error_magnitude(cmd_flat, robot_flat).reshape(command.num_envs, -1)
    terminated = error.max(dim=-1).values > threshold
    return _mask_freeze(command, terminated)


def joint_pos_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Terminate when joint positions are too far from the command. Freeze-aware."""
    command = env.command_manager.get_term(command_name)
    error = torch.norm(command.robot_joint_pos - command.command_joint_pos, dim=-1)
    terminated = error > threshold
    return _mask_freeze(command, terminated)


def object_pos_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when the object position is too far from the command. Freeze-aware."""
    command = env.command_manager.get_term(command_name)
    error = torch.norm(command.object_pos_w - command.command_object_pos_w, dim=-1)
    terminated = error > threshold
    return _mask_freeze(command, terminated)


def object_quat_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Terminate when the object orientation is too far from the command. Freeze-aware."""
    command = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.object_quat_w, command.command_object_quat_w)
    terminated = error > threshold
    return _mask_freeze(command, terminated)
