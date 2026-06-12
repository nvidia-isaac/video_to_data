# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils.math import quat_error_magnitude

from robotic_grounding.tasks.v2p.mdp.utils import chamfer_distance


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


class motion_joint_pos_error_exp(ManagerTermBase):  # noqa: N801
    """Reward for tracking joint positions using an exponential kernel."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv) -> None:
        """Resolve optional joint filters once when the reward term is created."""
        super().__init__(cfg, env)

        self._command = env.command_manager.get_term(cfg.params["command_name"])
        joint_names: list[str] | None = cfg.params.get("joint_names")
        self._joint_ids: torch.Tensor | None = None

        if joint_names is not None:
            _, resolved_joint_names = self._command.robot.find_joints(joint_names)
            tracked_joint_name_to_idx = {
                name: idx for idx, name in enumerate(self._command._tracked_joint_names)
            }
            missing_joint_names = [
                name
                for name in resolved_joint_names
                if name not in tracked_joint_name_to_idx
            ]
            if missing_joint_names:
                raise ValueError(
                    "Reward joint_names must be included in the tracking command joints. "
                    f"Missing joints: {missing_joint_names}"
                )
            self._joint_ids = torch.tensor(
                [tracked_joint_name_to_idx[name] for name in resolved_joint_names],
                device=env.device,
                dtype=torch.long,
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        command_name: str,
        std: float,
        joint_names: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Compute joint position tracking reward.

        Args:
            env: The environment object.
            command_name: Name of the tracking command term.
            std: Standard deviation of the exponential kernel.
            joint_names: Optional list of joint names or name patterns to track. If not
                provided, all joints tracked by the command are used.

        Returns:
            Reward tensor of shape (num_envs,).
        """
        del env, command_name, joint_names
        command_joint_pos = self._command.command_joint_pos
        robot_joint_pos = self._command.robot_joint_pos
        if self._joint_ids is not None:
            command_joint_pos = command_joint_pos[:, self._joint_ids]
            robot_joint_pos = robot_joint_pos[:, self._joint_ids]

        error = torch.sum(torch.square(command_joint_pos - robot_joint_pos), dim=-1)
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


def motion_finger_joint_pos_gaussian_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """Finger joint tracking: exp(-||error||^2 / std^2). Returns sum L+R (max 2.0)."""
    command = env.command_manager.get_term(command_name)
    right_error = torch.sum(
        torch.square(
            command.right_hand_finger_joint_pos_command
            - command.right_hand_finger_joint_pos
        ),
        dim=-1,
    )
    left_error = torch.sum(
        torch.square(
            command.left_hand_finger_joint_pos_command
            - command.left_hand_finger_joint_pos
        ),
        dim=-1,
    )
    return torch.exp(-right_error / std**2) + torch.exp(-left_error / std**2)


def motion_progress(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reward for tracking the progress of the motion.

    Progress is normalized relative to where each environment started from.
    """
    command = env.command_manager.get_term(command_name)
    # Compute progress relative to reset point
    steps_taken = (command.timestep - command.reset_timestep).float()
    steps_remaining = (command.num_timesteps - 1 - command.reset_timestep).float()
    return steps_taken / steps_remaining.clamp(min=1.0)


def motion_hand_keypoints_gaussian_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """Hand keypoints tracking: exp(-||error||^2 / std^2). Returns sum L+R (max 2.0)."""
    command = env.command_manager.get_term(command_name)

    left_kp_cmd = torch.cat(
        [
            command.left_hand_wrist_pose_command_e[:, :3].unsqueeze(1),
            command.left_hand_fingertip_position_command_e[..., :3],
        ],
        dim=1,
    )
    right_kp_cmd = torch.cat(
        [
            command.right_hand_wrist_pose_command_e[:, :3].unsqueeze(1),
            command.right_hand_fingertip_position_command_e[..., :3],
        ],
        dim=1,
    )
    left_kp = torch.cat(
        [
            command.left_hand_wrist_position_e.unsqueeze(1),
            command.left_hand_fingertip_position_e[..., :3],
        ],
        dim=1,
    )
    right_kp = torch.cat(
        [
            command.right_hand_wrist_position_e.unsqueeze(1),
            command.right_hand_fingertip_position_e[..., :3],
        ],
        dim=1,
    )

    left_err = torch.sum(torch.square(left_kp_cmd - left_kp), dim=-1).sum(dim=-1)
    right_err = torch.sum(torch.square(right_kp_cmd - right_kp), dim=-1).sum(dim=-1)
    return torch.exp(-left_err / std**2) + torch.exp(-right_err / std**2)


def motion_contact_tracking_gaussian_exp(
    env: ManagerBasedEnv, command_name: str, std: float, mask_zero_contact: bool = True
) -> torch.Tensor:
    """Chamfer contact tracking: exp(-dist^2 / std^2). Returns sum L+R."""
    command = env.command_manager.get_term(command_name)
    result = torch.zeros(env.num_envs, device=env.device)

    for side in ("right", "left"):
        pos_e = getattr(command, f"{side}_hand_object_contact_positions_e")
        valid = (
            getattr(command, f"{side}_hand_object_contact_positions_w").sum(dim=-1)
            > 1e-5
        )
        cmd_e = getattr(command, f"{side}_hand_object_contact_command_positions_e")
        cmd_valid = getattr(command, f"retargeted_{side}_object_contact_is_valid")[
            command.timestep_counter
        ]

        dist = chamfer_distance(pos_e, cmd_e, valid, cmd_valid)
        rew = torch.exp(-(dist**2) / std**2)
        if mask_zero_contact:
            both_zero = (valid.sum(dim=-1) == 0) & (cmd_valid.sum(dim=-1) == 0)
            rew[both_zero] = 0.0
        result += rew

    return result


def motion_contact_force_gaussian_exp(
    env: ManagerBasedEnv, command_name: str, std: float
) -> torch.Tensor:
    """Contact force reward: exp(-force^2 / std^2). Mean over in-contact points."""
    command = env.command_manager.get_term(command_name)

    total_rew = torch.zeros(env.num_envs, device=env.device)
    total_contacts = torch.zeros(env.num_envs, device=env.device)

    for side in ("right", "left"):
        forces = (
            getattr(command, f"{side}_hand_object_contact_forces_w")
            .norm(dim=1)
            .norm(dim=-1)
        )
        in_contact = forces > 1e-3
        total_rew += (in_contact * torch.exp(-(forces**2) / std**2)).sum(dim=-1)
        total_contacts += in_contact.sum(dim=-1)

    return torch.nan_to_num(total_rew / total_contacts, nan=0.0)
