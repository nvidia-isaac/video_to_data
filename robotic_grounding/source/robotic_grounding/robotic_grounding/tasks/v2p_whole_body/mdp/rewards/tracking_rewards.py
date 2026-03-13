import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_error_magnitude


def motion_global_anchor_position_error_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    Reward for tracking the global anchor position using exponential kernel.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.
        std: Standard deviation of the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(command.command_anchor_pos_w - command.robot_anchor_pos_w), dim=-1
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    Reward for tracking the global anchor orientation using exponential kernel.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.
        std: Standard deviation of the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command = env.command_manager.get_term(command_name)
    error = (
        quat_error_magnitude(command.command_anchor_quat_w, command.robot_anchor_quat_w)
        ** 2
    )
    return torch.exp(-error / std**2)


def motion_ee_position_error_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    Reward for tracking the EE position using exponential kernel.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.
        std: Standard deviation of the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
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
    return torch.exp(-error / std**2)


def motion_ee_orientation_error_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    Reward for tracking the EE orientation using exponential kernel.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.
        std: Standard deviation of the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
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
    return torch.exp(-error / std**2)


def motion_joint_pos_error_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    Reward for tracking the joint positions using exponential kernel.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.
        std: Standard deviation of the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(command.command_joint_pos - command.robot_joint_pos), dim=-1
    )
    return torch.exp(-error / std**2)


def motion_object_position_error_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """
    Reward for tracking object position using exponential kernel.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.
        std: Standard deviation of the exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    command = env.command_manager.get_term(command_name)
    object_pos_w = env.scene["object"].data.root_pos_w
    error = torch.sum(torch.square(object_pos_w - command.command_object_pos_w), dim=-1)
    rew = torch.exp(-error / std**2)

    return rew


def motion_object_lifted(
    env: ManagerBasedEnv,
    command_name: str,
) -> torch.Tensor:
    """
    Reward for object being lifted, but only when reference trajectory says it should be.

    Args:
        env: The environment object.
        command_name: Name of the tracking command term.

    Returns:
        Reward tensor of shape (num_envs,) with 1.0 if lifted when should be, 0.0 otherwise.
    """
    command = env.command_manager.get_term(command_name)
    actual_object_height = env.scene["object"].data.root_pos_w[:, 2]
    minimal_height = torch.min(command._object_pos_w[:, 2])

    # Check if object is lifted
    is_lifted = actual_object_height > minimal_height

    # Check if object should be lifted
    ref_object_height = command.command_object_pos_w[:, 2]  # Z coordinate
    should_be_lifted = ref_object_height > minimal_height

    return (is_lifted & should_be_lifted).float()


def motion_object_orientation_error_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """Reward for tracking object orientation using exponential kernel."""
    command = env.command_manager.get_term(command_name)
    object_quat_w = env.scene["object"].data.root_quat_w
    error = quat_error_magnitude(object_quat_w, command.command_object_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_progress(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reward for tracking the progress of the motion.

    Progress is normalized relative to where each environment started from.
    """
    command = env.command_manager.get_term(command_name)
    # Compute progress relative to reset point
    steps_taken = (command.timestep - command.reset_timestep).float()
    steps_remaining = (command.num_timesteps - 1 - command.reset_timestep).float()
    return steps_taken / steps_remaining.clamp(min=1.0)
