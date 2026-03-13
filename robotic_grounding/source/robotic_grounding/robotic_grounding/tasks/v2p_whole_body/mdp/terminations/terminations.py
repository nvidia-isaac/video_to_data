import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_error_magnitude


def timestep_termination(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """
    Terminate when the command is completed.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term

    Returns:
        Termination tensor (num_envs,)
    """
    command = env.command_manager.get_term(command_name)
    return command.timestep >= command.num_timesteps - 1


def anchor_pos_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Terminate when the anchor position is too far from the command.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        threshold: Threshold for the anchor position error

    Returns:
        Termination tensor (num_envs,)
    """
    command = env.command_manager.get_term(command_name)
    return (
        torch.norm(command.robot_anchor_pos_w - command.command_anchor_pos_w, dim=-1)
        > threshold
    )


def anchor_quat_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when the anchor quaternion is too far from the command.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        threshold: Threshold for the anchor quaternion error

    Returns:
        Termination tensor (num_envs,)
    """
    command = env.command_manager.get_term(command_name)
    return (
        quat_error_magnitude(command.robot_anchor_quat_w, command.command_anchor_quat_w)
        > threshold
    )


def ee_position_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when the EE position is too far from the command."""
    command = env.command_manager.get_term(command_name)
    # command_ee_pos_w: (num_envs, num_ee_links, 3), robot_ee_pos_w: (num_envs, num_ee_links, 3)
    num_envs = command.command_ee_pos_w.shape[0]
    num_ee_links = command.command_ee_pos_w.shape[1]

    # Flatten to (num_envs * num_ee_links, 3)
    cmd_pos_flat = command.command_ee_pos_w.reshape(-1, 3)
    robot_pos_flat = command.robot_ee_pos_w.reshape(-1, 3)

    error_flat = torch.sum(
        torch.square(cmd_pos_flat - robot_pos_flat), dim=-1
    )  # (num_envs * num_ee_links,)
    error_per_ee = error_flat.reshape(
        num_envs, num_ee_links
    )  # (num_envs, num_ee_links)
    error = torch.sum(error_per_ee, dim=-1)  # (num_envs,) - sum over EE links
    return error > threshold


def ee_quat_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when the EE quaternion is too far from the command."""
    command = env.command_manager.get_term(command_name)
    # command_ee_quat_w: (num_envs, num_ee_links, 4), robot_ee_quat_w: (num_envs, num_ee_links, 4)
    num_envs = command.command_ee_quat_w.shape[0]
    num_ee_links = command.command_ee_quat_w.shape[1]

    # Reshape to (num_envs * num_ee_links, 4) for quat_error_magnitude
    cmd_quat_flat = command.command_ee_quat_w.reshape(-1, 4)
    robot_quat_flat = command.robot_ee_quat_w.reshape(-1, 4)

    error_flat = (
        quat_error_magnitude(cmd_quat_flat, robot_quat_flat) ** 2
    )  # (num_envs * num_ee_links,)
    error_per_ee = error_flat.reshape(
        num_envs, num_ee_links
    )  # (num_envs, num_ee_links)
    error = torch.sum(error_per_ee, dim=-1)  # (num_envs,) - sum over EE links
    return error > threshold


def joint_pos_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when the joint position is too far from the command."""
    command = env.command_manager.get_term(command_name)
    return (
        torch.norm(
            command.robot_joint_pos - command.command_joint_pos_multi_future[:, 0, :],
            dim=-1,
        )
        > threshold
    )


def object_pos_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when the object position is too far from the command.

    Args:
        env: The environment instance
        command_name: Name of the tracking command term
        threshold: Threshold for the object position error

    Returns:
        Termination tensor (num_envs,)
    """
    command = env.command_manager.get_term(command_name)
    return (
        torch.norm(command.object_pos_w - command.command_object_pos_w, dim=-1)
        > threshold
    )
