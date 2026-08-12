# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv

from robotic_grounding.tasks.v2d.mdp.utils_jit import (
    contact_wrench_support_reward_jit,
    hand_keypoints_tracking_jit,
    missed_contact_penalty_jit,
    unintended_contact_penalty_jit,
)


def object_position_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.3,
) -> torch.Tensor:
    """Compute task reward based on object tracking (num_envs,).

    Args:
        env: The RL environment.
        command_name: Name of the command term to get demo data from.
        var: Variance for the exponential reward.

    Returns:
        Task reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)

    # Get current object state and position error
    object_position_e = command.object_position_e_sq
    object_position_error_e = torch.sum(
        torch.square(command.object_body_position_command_e - object_position_e),
        dim=-1,
    )

    # Compute exponential rewards
    object_position_tracking_rew = torch.exp(-object_position_error_e / var)

    return object_position_tracking_rew


def object_wxyz_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.3,
) -> torch.Tensor:
    """Compute task reward based on object tracking (num_envs,).

    Args:
        env: The RL environment.
        command_name: Name of the command term to get demo data from.
        var: Variance for the exponential reward.

    Returns:
        Task reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)

    # Get current object state and orientation error
    object_wxyz = command.object_wxyz_e_sq
    object_orientation_error_e = math_utils.quat_error_magnitude(
        command.object_body_wxyz_command_e, object_wxyz
    )

    # Compute exponential rewards
    object_orientation_tracking_rew = torch.exp(-object_orientation_error_e / var)

    return object_orientation_tracking_rew


def object_keypoints_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.1,
) -> torch.Tensor:
    """
    Compute the exponential reward for object keypoints tracking.

    This reward encourages the agent to align the object's pose with the demonstration target pose by matching key positions ("keypoints") in the object frame.
    Keypoints are defined as the 6 principal unit vectors (+X, +Y, +Z, -X, -Y, -Z) transformed to world space using the current and command object poses.
    The reward decays exponentially with the aggregate error between the tracked keypoints and their respective targets in the demonstration trajectory.

    Args:
        env (ManagerBasedRLEnv): The RL environment instance.
        command_name (str, optional): The name of the command term providing trajectory data. Defaults to "dual_hands_object_tracking_command".
        var (float, optional): Variance (decay scale) for the exponential reward. Smaller values penalize deviations more sharply.

    Returns:
        torch.Tensor: A tensor of shape (num_envs,) containing the keypoints tracking reward for each environment.
    """
    command = env.command_manager.get_term(command_name)

    # Get current object state
    object_position = command.object_position_e.unsqueeze(2).expand(
        -1, -1, 6, -1
    )  # (num_envs, k, 6, 3)
    object_wxyz = command.object_orientation_e.unsqueeze(2).expand(
        -1, -1, 6, -1
    )  # (num_envs, k, 6, 4)

    # Compute keypoints
    object_keypoints, _ = math_utils.combine_frame_transforms(
        object_position,
        object_wxyz,
        command.KEYPOINT_VECS,
        q12=None,
    )  # (num_envs, k, 6, 3)
    object_command_keypoints, _ = math_utils.combine_frame_transforms(
        command.object_body_position_command_e.unsqueeze(2).expand(-1, -1, 6, -1),
        command.object_body_wxyz_command_e.unsqueeze(2).expand(-1, -1, 6, -1),
        command.KEYPOINT_VECS,
        q12=None,
    )  # (num_envs, k, 6, 3)

    # Compute keypoints error
    object_keypoints_error = torch.sum(
        torch.square(object_keypoints - object_command_keypoints), dim=-1
    )  # (num_envs, k, 6)
    object_keypoints_tracking_rew = torch.exp(-object_keypoints_error / var).mean(
        dim=(-2, -1)
    )

    return object_keypoints_tracking_rew


def hand_keypoints_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.1,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Compute the exponential imitation reward for hand keypoints tracking, including both wrist and fingertip positions.

    This reward encourages the agent's hand (both wrists and fingertips) to closely follow the demonstration trajectory.
    Specifically, the wrist position and all fingertips are treated as "keypoints," and their current positions are compared
    against the corresponding target positions from the demonstration. For each keypoint, the Euclidean distance to the
    reference is computed, and an exponential penalty is applied based on the provided variance.

    The final reward is the sum of the exponentiated negative errors for:
      - Right wrist position
      - Left wrist position
      - Right-hand fingertip positions
      - Left-hand fingertip positions

    Args:
        env (ManagerBasedRLEnv): The RL environment instance.
        command_name (str, optional): Name of the command term providing demonstration data. Defaults to "dual_hands_object_tracking_command".
        var (float, optional): Variance (decay scale) for the exponential reward. Smaller values penalize deviations more sharply.
        threshold (float, optional): Threshold to saturate the reward. Errors below this threshold are set to 1.0.

    Returns:
        torch.Tensor: A tensor of shape (num_envs,) containing the imitation reward for each environment.
    """
    command = env.command_manager.get_term(command_name)
    return hand_keypoints_tracking_jit(
        left_wrist_cmd=command.left_hand_wrist_pose_command_e[:, :3],
        right_wrist_cmd=command.right_hand_wrist_pose_command_e[:, :3],
        left_fingertip_cmd=command.left_hand_fingertip_position_command_e[..., :3],
        right_fingertip_cmd=command.right_hand_fingertip_position_command_e[..., :3],
        left_wrist_cur=command.left_hand_wrist_position_e,
        right_wrist_cur=command.right_hand_wrist_position_e,
        left_fingertip_cur=command.left_hand_fingertip_position_e[..., :3],
        right_fingertip_cur=command.right_hand_fingertip_position_e[..., :3],
        var=float(var),
        threshold=float(threshold),
    )


def hand_joint_pos_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.1,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Compute imitation reward based on hand joint positions (num_envs,)."""
    command = env.command_manager.get_term(command_name)

    # Compute keypoints error for hand joint positions
    left_hand_joint_pos_error = torch.sum(
        torch.square(
            command.left_hand_finger_joint_pos_command
            - command.left_hand_finger_joint_pos
        ),
        dim=-1,
    )
    right_hand_joint_pos_error = torch.sum(
        torch.square(
            command.right_hand_finger_joint_pos_command
            - command.right_hand_finger_joint_pos
        ),
        dim=-1,
    )

    return (
        torch.exp(-(left_hand_joint_pos_error - threshold).clamp(min=0.0) / var)
        + torch.exp(-(right_hand_joint_pos_error - threshold).clamp(min=0.0) / var)
    ) / 2.0


def contact_force_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 1.0,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Contact force reward.

    If no contact is present, reward is 0.
    If contact is present, reward forces within the range of lower_force_squared and upper_force_squared.

    Args:
        env: The RL environment.
        command_name: Command term that provides contact forces.
        var: Scale for exp(-contact_force / var).
        threshold: Threshold for contact allowrance.

    Returns:
        Reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)

    right_hand_object_contact_forces_norm = command.right_force_sq_per_link
    right_hand_link_in_contact = command.right_link_in_contact
    num_right_hand_links_in_contact = right_hand_link_in_contact.sum(dim=-1)

    left_hand_object_contact_forces_norm = command.left_force_sq_per_link
    left_hand_link_in_contact = command.left_link_in_contact
    num_left_hand_links_in_contact = left_hand_link_in_contact.sum(dim=-1)

    contact_force_reward = (
        right_hand_link_in_contact
        * torch.exp(
            -(right_hand_object_contact_forces_norm - threshold).clamp(min=0.0) / var
        )
        + left_hand_link_in_contact
        * torch.exp(
            -(left_hand_object_contact_forces_norm - threshold).clamp(min=0.0) / var
        )
    ).sum(dim=-1) / (
        num_right_hand_links_in_contact + num_left_hand_links_in_contact
    ).clamp(
        min=1e-5
    )

    return contact_force_reward


def contact_force_range_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 1.0,
    lower_force_squared: float = 4.0,
    upper_force_squared: float = 16.0,
) -> torch.Tensor:
    """Contact force reward.

    If no contact is present, reward is 0.
    If contact is present, reward forces within the range of lower_force_squared and upper_force_squared.

    Args:
        env: The RL environment.
        command_name: Command term that provides contact forces.
        var: Scale for exp(-contact_force / var).
        lower_force_squared: Lower force squared to be rewarded.
        upper_force_squared: Upper force squared to be rewarded.

    Returns:
        Reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)

    right_hand_object_contact_forces_norm = command.right_force_sq_per_link
    right_hand_link_in_contact = command.right_link_in_contact
    num_right_hand_links_in_contact = right_hand_link_in_contact.sum(dim=-1)

    left_hand_object_contact_forces_norm = command.left_force_sq_per_link
    left_hand_link_in_contact = command.left_link_in_contact
    num_left_hand_links_in_contact = left_hand_link_in_contact.sum(dim=-1)

    contact_force_reward = (
        right_hand_link_in_contact
        * torch.exp(
            -(lower_force_squared - right_hand_object_contact_forces_norm).clamp(
                min=0.0
            )
            / var
        )
        * torch.exp(
            -(right_hand_object_contact_forces_norm - upper_force_squared).clamp(
                min=0.0
            )
            / var
        )
        + left_hand_link_in_contact
        * torch.exp(
            -(lower_force_squared - left_hand_object_contact_forces_norm).clamp(min=0.0)
            / var
        )
        * torch.exp(
            -(left_hand_object_contact_forces_norm - upper_force_squared).clamp(min=0.0)
            / var
        )
    ).sum(dim=-1) / (
        num_right_hand_links_in_contact + num_left_hand_links_in_contact
    ).clamp(
        min=1e-5
    )

    return contact_force_reward


def contact_force_rate_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 1.0,
) -> torch.Tensor:
    """Contact force rate reward.

    Args:
        env: The RL environment.
        command_name: Command term that provides contact forces.
        var: Scale for exp(-contact_force_rate / var).
        threshold: Threshold for contact force rate.

    Returns:
        Reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)

    right_hand_object_contact_forces_norm = (
        command.right_hand_object_contact_forces_w.norm(dim=-1)
    ).sum(
        dim=2
    )  # (num_envs, timesteps, num_hand_link_w_sensor)
    left_hand_object_contact_forces_norm = (
        command.left_hand_object_contact_forces_w.norm(dim=-1)
    ).sum(
        dim=2
    )  # (num_envs, timesteps, num_hand_link_w_sensor)

    right_hand_link_in_contact = (
        right_hand_object_contact_forces_norm.sum(dim=1) > 1e-3
    )  # (num_envs, num_hand_link_w_sensor)
    left_hand_link_in_contact = (
        left_hand_object_contact_forces_norm.sum(dim=1) > 1e-3
    )  # (num_envs, num_hand_link_w_sensor)

    num_right_hand_links_in_contact = right_hand_link_in_contact.sum(dim=-1)
    num_left_hand_links_in_contact = left_hand_link_in_contact.sum(dim=-1)

    right_hand_object_contact_forces_norm_diff = torch.abs(
        torch.diff(right_hand_object_contact_forces_norm, dim=1)
    ).mean(
        dim=1
    )  # (num_envs, num_hand_link_w_sensor)
    left_hand_object_contact_forces_norm_diff = torch.abs(
        torch.diff(left_hand_object_contact_forces_norm, dim=1)
    ).mean(
        dim=1
    )  # (num_envs, num_hand_link_w_sensor)

    contact_force_rate_reward = (
        right_hand_link_in_contact
        * torch.exp(-right_hand_object_contact_forces_norm_diff / var)
        + left_hand_link_in_contact
        * torch.exp(-left_hand_object_contact_forces_norm_diff / var)
    ).sum(dim=1) / (
        num_right_hand_links_in_contact + num_left_hand_links_in_contact
    ).clamp(
        min=1e-5
    )

    return contact_force_rate_reward


def termination_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for termination (num_envs,)."""
    return env.termination_manager.terminated


def action_norm(
    env: ManagerBasedRLEnv,
    action_names: list[str],
) -> torch.Tensor:
    """Lp norm of the actions (num_envs,)."""
    for i, action_name in enumerate(action_names):
        if i == 0:
            action_norm = torch.sum(
                torch.square(env.action_manager.get_term(action_name).raw_actions),
                dim=-1,
            )
        else:
            action_norm += torch.sum(
                torch.square(env.action_manager.get_term(action_name).raw_actions),
                dim=-1,
            )
    return action_norm


def contact_wrench_support_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    tolerance: float = 0.1,
    var: float = 0.1,
) -> torch.Tensor:
    """Contact wrench support reward.

    Args:
        env: The RL environment.
        command_name: Command term that provides contact wrench supports.
        tolerance: Tolerance for current wrench support compared to command wrench support.
        var: Scale for exp(-contact_loss / var).

    Returns:
        Reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)
    return contact_wrench_support_reward_jit(
        right_cmd_active=command.right_wrench_cmd_active,
        right_cur_active=command.right_wrench_cur_active,
        left_cmd_active=command.left_wrench_cmd_active,
        left_cur_active=command.left_wrench_cur_active,
        right_cmd_active_per_body=command.right_wrench_cmd_active_per_body,
        left_cmd_active_per_body=command.left_wrench_cmd_active_per_body,
        right_cmd_supports=command.right_hand_contact_wrench_supports_command,
        right_cur_supports=command.right_hand_contact_wrench_supports,
        left_cmd_supports=command.left_hand_contact_wrench_supports_command,
        left_cur_supports=command.left_hand_contact_wrench_supports,
        tolerance=float(tolerance),
        var=float(var),
    )


def unintended_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
) -> torch.Tensor:
    """Unintended contact penalty where the command has no contact but current has contact.

    Args:
        env: The RL environment.
        command_name: Command term that provides contact wrench supports.
        penalty: Penalty for unintended contact.
    """
    command = env.command_manager.get_term(command_name)
    return unintended_contact_penalty_jit(
        right_cmd_active_per_body=command.right_wrench_cmd_active_per_body,
        right_cur_active_per_body=command.right_wrench_cur_active_per_body,
        left_cmd_active_per_body=command.left_wrench_cmd_active_per_body,
        left_cur_active_per_body=command.left_wrench_cur_active_per_body,
        right_cur_supports=command.right_hand_contact_wrench_supports,
        left_cur_supports=command.left_hand_contact_wrench_supports,
        num_bodies=int(command.num_bodies),
    )


def missed_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
) -> torch.Tensor:
    """Missed contact penalty.

    Args:
        env: The RL environment.
        command_name: Command term that provides contact wrench supports.
        penalty: Penalty for missed contact.
    """
    command = env.command_manager.get_term(command_name)
    return missed_contact_penalty_jit(
        right_cmd_active=command.right_wrench_cmd_active,
        right_cur_active=command.right_wrench_cur_active,
        left_cmd_active=command.left_wrench_cmd_active,
        left_cur_active=command.left_wrench_cur_active,
        right_cmd_active_per_body=command.right_wrench_cmd_active_per_body,
        left_cmd_active_per_body=command.left_wrench_cmd_active_per_body,
    )


def relative_object_pose_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    pos_sigma: float = 0.05,
    rot_sigma: float = 0.5,
) -> torch.Tensor:
    """Proximity-gated reward for inter-object relative pose tracking (num_envs,).

    Measures how well the policy maintains the correct spatial relationship between
    object0 and object1 as demonstrated in the reference motion. Only active when
    the demo shows the objects within ``relative_object_proximity_threshold`` of each
    other. Returns zeros for single-object tasks.

    Args:
        env: The RL environment.
        command_name: Name of the command term to get demo data from.
        pos_sigma: Position tolerance in metres; reward = 1/e at this error. Default 5 cm.
        rot_sigma: Rotation tolerance in radians; reward = 1/e at this error. Default ~29 deg.
    """
    command = env.command_manager.get_term(command_name)
    if not getattr(command, "_has_multi_object", False):
        return torch.zeros(env.num_envs, device=env.device)
    pos_err, rot_err = command.relative_object_pose_error
    proximity_mask = command.relative_object_proximity_mask
    reward = torch.exp(-pos_err / pos_sigma) * torch.exp(-rot_err / rot_sigma)
    return reward * proximity_mask.float()


def relative_object_pos_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    pos_sigma: float = 0.02,
) -> torch.Tensor:
    """Proximity-gated position component of inter-object relative pose reward (num_envs,).

    Decoupled from rotation so pos and rot weights can be tuned independently.
    Returns zeros for single-object tasks.

    Args:
        env: The RL environment.
        command_name: Name of the command term to get demo data from.
        pos_sigma: Position tolerance in metres; reward = 1/e at this error. Default 2 cm.
    """
    command = env.command_manager.get_term(command_name)
    if not getattr(command, "_has_multi_object", False):
        return torch.zeros(env.num_envs, device=env.device)
    pos_err, _ = command.relative_object_pose_error
    proximity_mask = command.relative_object_proximity_mask
    return torch.exp(-pos_err / pos_sigma) * proximity_mask.float()


def relative_object_rot_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    rot_sigma: float = 0.3,
) -> torch.Tensor:
    """Proximity-gated rotation component of inter-object relative pose reward (num_envs,).

    Decoupled from position so rot weight can be tuned independently. Default rot_sigma
    is tighter (0.3 rad ≈ 17°) than the combined form to give stronger gradient signal.
    Returns zeros for single-object tasks.

    Args:
        env: The RL environment.
        command_name: Name of the command term to get demo data from.
        rot_sigma: Rotation tolerance in radians; reward = 1/e at this error. Default ~17 deg.
    """
    command = env.command_manager.get_term(command_name)
    if not getattr(command, "_has_multi_object", False):
        return torch.zeros(env.num_envs, device=env.device)
    _, rot_err = command.relative_object_pose_error
    proximity_mask = command.relative_object_proximity_mask
    return torch.exp(-rot_err / rot_sigma) * proximity_mask.float()


def inter_object_proximity_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    dist_sigma: float = 0.05,
) -> torch.Tensor:
    """Scalar inter-object distance tracking reward (num_envs,).

    Always-on companion to the proximity-gated relative pose rewards: encourages
    |‖obj1 − obj0‖ − demo_distance| → 0 even when the two objects are far apart
    in the demo. Provides a long-range signal so the policy is incentivised to
    close the gap (e.g. spoon moving toward pan) before the proximity-gated
    pose rewards activate. Returns zeros for single-object tasks.

    Args:
        env: The RL environment.
        command_name: Name of the command term to get demo data from.
        dist_sigma: Distance tolerance in metres; reward = 1/e at this error. Default 5 cm.
    """
    command = env.command_manager.get_term(command_name)
    if not getattr(command, "_has_multi_object", False):
        return torch.zeros(env.num_envs, device=env.device)
    obj0_pos = command.object_position_e[:, 0, :]
    obj1_pos = command.object_position_e[:, command._obj1_root_body_idx, :]
    cur_dist = torch.norm(obj1_pos - obj0_pos, dim=-1)
    demo_dist = command._demo_inter_object_dist[command.timestep_counter]
    return torch.exp(-(cur_dist - demo_dist).abs() / dist_sigma)


def object_meshvert_tracking_fine(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.001,
) -> torch.Tensor:
    """Fine object-pose reward from sampled mesh vertices (num_envs,)."""
    command = env.command_manager.get_term(command_name)
    if not getattr(command, "_meshvert_reward_enabled", False):
        return torch.zeros(env.num_envs, device=env.device)

    num_verts = command._meshvert_num_verts
    verts = command.OBJECT_MESHVERT_SAMPLED_VERTS
    pos_current = command.object_position_e.unsqueeze(2).expand(-1, -1, num_verts, -1)
    quat_current = command.object_orientation_e.unsqueeze(2).expand(
        -1, -1, num_verts, -1
    )
    pos_target = command.object_body_position_command_e.unsqueeze(2).expand(
        -1, -1, num_verts, -1
    )
    quat_target = command.object_body_wxyz_command_e.unsqueeze(2).expand(
        -1, -1, num_verts, -1
    )

    verts_current, _ = math_utils.combine_frame_transforms(
        pos_current,
        quat_current,
        verts,
    )
    verts_target, _ = math_utils.combine_frame_transforms(
        pos_target,
        quat_target,
        verts,
    )
    add_per_env = torch.norm(verts_current - verts_target, dim=-1).mean(dim=(-2, -1))
    return torch.exp(-(add_per_env**2) / var)
