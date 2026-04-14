from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from robotic_grounding.tasks.v2p.mdp.observations import (
    finger_contact_force_vectors,
    finger_contact_forces,
)
from robotic_grounding.tasks.v2p.mdp.utils import chamfer_distance


def contact_force_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg | None = None,
    max_force: float = 50.0,
) -> torch.Tensor:
    """Penalty for excessive contact forces (num_envs,)."""
    force_magnitudes = finger_contact_forces(env, sensor_cfg)
    excess_forces = torch.clamp(force_magnitudes - max_force, min=0.0)
    return -excess_forces.sum(dim=-1)


def grasp_force_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg | None = None,
    target_force: float = 5.0,
) -> torch.Tensor:
    """Reward for maintaining target grasp force (num_envs,)."""
    force_magnitudes = finger_contact_forces(env, sensor_cfg)
    total_force = force_magnitudes.sum(dim=-1)
    return -torch.abs(total_force - target_force)


def maniptrans_contact_reward(
    env: ManagerBasedRLEnv,
    contact_range_min: float = 0.02,
    contact_range_max: float = 0.03,
    decay_constant: float = 1.0,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """ManipTrans-style contact reward with distance-weighted force masking (num_envs,).

    This reward encourages meaningful contact by weighting fingertip forces based on
    distance to the object. Forces from fingertips close to the object contribute more
    to the reward than forces from fingertips far away.

    Pre-computed reference distances are loaded from ``env.cfg.tips_distance_data``
    (set by the task-specific env config from parquet data). Episode step is used to
    index into the reference trajectory with FPS ratio scaling.

    The soft distance weight is computed as:
        weight = clamp((contact_range_max - distance) / (contact_range_max - contact_range_min), 0, 1)

    The reward is computed as:
        reward = exp(-decay_constant / (total_masked_force + epsilon))

    Args:
        env: The environment instance.
        contact_range_min: Distance below which weight is 1.0 (in contact). Default: 0.02m.
        contact_range_max: Distance above which weight is 0.0 (too far). Default: 0.03m.
        decay_constant: Controls reward sensitivity to force magnitude. Default: 1.0.
        epsilon: Small constant to prevent division by zero. Default: 1e-5.

    Returns:
        Reward tensor of shape (num_envs,) with values in [0, 1).
        Higher total masked force leads to higher reward (approaches 1.0).
        Returns zeros if tips_distance data is not available.
    """
    # Lazy-cache the tips_distance tensor on GPU
    if not hasattr(env, "_tips_distance_tensor"):
        if (
            hasattr(env.cfg, "tips_distance_data")
            and env.cfg.tips_distance_data is not None
        ):
            env._tips_distance_tensor = (
                torch.from_numpy(env.cfg.tips_distance_data).float().to(env.device)
            )
        else:
            env._tips_distance_tensor = None

    if env._tips_distance_tensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    # Index by episode step, accounting for FPS difference between source data and env
    source_fps = getattr(env.cfg, "tips_distance_fps", 30.0)
    env_fps = 1.0 / env.step_dt
    fps_ratio = source_fps / env_fps
    source_indices = (env.episode_length_buf.float() * fps_ratio).long()
    source_indices = source_indices.clamp(0, env._tips_distance_tensor.shape[0] - 1)
    distances = env._tips_distance_tensor[source_indices]  # (num_envs, 10)

    # Soft distance weights: (num_envs, 10)
    weights = torch.clamp(
        (contact_range_max - distances) / (contact_range_max - contact_range_min),
        min=0.0,
        max=1.0,
    )

    # Get 3D force vectors from contact sensors and apply distance mask: (num_envs, num_fingers, 3)
    force_vectors = finger_contact_force_vectors(env)
    masked_forces = force_vectors * weights.unsqueeze(-1)

    # Sum of masked force magnitudes: (num_envs,)
    total_force = torch.norm(masked_forces, dim=-1).sum(dim=-1)

    # Reward: exp(-decay_constant / (total_force + epsilon))
    # Higher force -> higher reward (approaches 1.0)
    return torch.exp(-decay_constant / (total_force + epsilon))


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
    object_position_e = command.object_position_e.squeeze(1)
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
    object_wxyz = command.object_orientation_e.squeeze(1)
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

    # Combine wrist and fingertip positions
    left_hand_keypoints_position_command_e = torch.cat(
        [
            command.left_hand_wrist_pose_command_e[:, :3].unsqueeze(1),
            command.left_hand_fingertip_position_command_e[..., :3],
        ],
        dim=1,
    )
    right_hand_keypoints_position_command_e = torch.cat(
        [
            command.right_hand_wrist_pose_command_e[:, :3].unsqueeze(1),
            command.right_hand_fingertip_position_command_e[..., :3],
        ],
        dim=1,
    )
    left_hand_keypoints_position_e = torch.cat(
        [
            command.left_hand_wrist_position_e.unsqueeze(1),
            command.left_hand_fingertip_position_e[..., :3],
        ],
        dim=1,
    )
    right_hand_keypoints_position_e = torch.cat(
        [
            command.right_hand_wrist_position_e.unsqueeze(1),
            command.right_hand_fingertip_position_e[..., :3],
        ],
        dim=1,
    )

    # Compute keypoints error
    left_hand_keypoints_position_error = torch.sum(
        torch.square(
            left_hand_keypoints_position_command_e - left_hand_keypoints_position_e
        ),
        dim=-1,
    )  # (num_envs, num_keypoints)
    right_hand_keypoints_position_error = torch.sum(
        torch.square(
            right_hand_keypoints_position_command_e - right_hand_keypoints_position_e
        ),
        dim=-1,
    )  # (num_envs, num_keypoints)

    left_hand_keypoints_reward = torch.exp(
        -(left_hand_keypoints_position_error - threshold).clamp(min=0.0) / var
    ).mean(dim=-1)
    right_hand_keypoints_reward = torch.exp(
        -(right_hand_keypoints_position_error - threshold).clamp(min=0.0) / var
    ).mean(dim=-1)

    return (left_hand_keypoints_reward + right_hand_keypoints_reward) / 2.0


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


def dexmachina_contact_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.03,
    mask_zero_contact: bool = True,
) -> torch.Tensor:
    """Chamfer-style contact reward: match policy contact points to demo contact.

    Demo contact comes from command (contact_links_left/right_command_e); policy contact
    from env contact sensors (per-link, per-object-part).

    Args:
        env: The RL environment.
        command_name: Command term that provides demo contact and object pose.
        var: Scale for exp(-chamfer_dist / var).
        mask_zero_contact: If True, reward is 0 when both demo and policy have no contact.

    Returns:
        Reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)

    # Extract contact positions and validity
    right_hand_object_contact_positions_e = (
        command.right_hand_object_contact_positions_e
    )  # (num_envs, num_bodies, num_hand_link_w_sensor, 3)
    right_hand_object_contact_positions_is_valid = (
        command.right_hand_object_contact_positions_w.sum(dim=-1) > 1e-5
    )  # (num_envs, num_bodies, num_hand_link_w_sensor)
    left_hand_object_contact_positions_e = command.left_hand_object_contact_positions_e
    # (num_envs, num_bodies, num_hand_link_w_sensor, 3)
    left_hand_object_contact_positions_is_valid = (
        command.left_hand_object_contact_positions_w.sum(dim=-1) > 1e-5
    )  # (num_envs, num_bodies, num_hand_link_w_sensor)

    # Extract desired contact part id for each hand and contact positions for each hand and desired part
    right_hand_object_contact_command_part_ids_per_hand = (
        command.get_command_contact_part_id("right")
    )
    right_hand_object_contact_positions_e = right_hand_object_contact_positions_e[
        command.all_env_ids, right_hand_object_contact_command_part_ids_per_hand
    ]  # (num_envs, num_hand_link_w_sensor, 3)
    right_hand_object_contact_positions_is_valid = (
        right_hand_object_contact_positions_is_valid[
            command.all_env_ids, right_hand_object_contact_command_part_ids_per_hand
        ]
    )  # (num_envs, num_hand_link_w_sensor)

    left_hand_object_contact_command_part_ids_per_hand = (
        command.get_command_contact_part_id("left")
    )
    left_hand_object_contact_positions_e = left_hand_object_contact_positions_e[
        command.all_env_ids, left_hand_object_contact_command_part_ids_per_hand
    ]  # (num_envs, num_hand_link_w_sensor, 3)
    left_hand_object_contact_positions_is_valid = (
        left_hand_object_contact_positions_is_valid[
            command.all_env_ids, left_hand_object_contact_command_part_ids_per_hand
        ]
    )  # (num_envs, num_hand_link_w_sensor)

    # Extract desired contact positions and validity for each hand
    right_hand_object_contact_command_positions_e = (
        command.right_hand_object_contact_command_positions_and_normals_e[..., :3]
    )
    right_hand_object_contact_command_positions_is_valid = (
        command.retargeted_right_object_contact_is_valid[command.timestep_counter]
    )
    left_hand_object_contact_command_positions_e = (
        command.left_hand_object_contact_command_positions_and_normals_e[..., :3]
    )
    left_hand_object_contact_command_positions_is_valid = (
        command.retargeted_left_object_contact_is_valid[command.timestep_counter]
    )

    right_hand_object_contact_dist = chamfer_distance(
        right_hand_object_contact_positions_e,
        right_hand_object_contact_command_positions_e,
        right_hand_object_contact_positions_is_valid,
        right_hand_object_contact_command_positions_is_valid,
    )
    left_hand_object_contact_dist = chamfer_distance(
        left_hand_object_contact_positions_e,
        left_hand_object_contact_command_positions_e,
        left_hand_object_contact_positions_is_valid,
        left_hand_object_contact_command_positions_is_valid,
    )

    right_hand_object_contact_rew = torch.exp(-right_hand_object_contact_dist / var)
    left_hand_object_contact_rew = torch.exp(-left_hand_object_contact_dist / var)

    if mask_zero_contact:
        right_hand_object_contact_rew[right_hand_object_contact_dist < 1e-5] = 0.0
        left_hand_object_contact_rew[left_hand_object_contact_dist < 1e-5] = 0.0

    return right_hand_object_contact_rew + left_hand_object_contact_rew


def contact_force_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 1.0,
    threshold: float = 0.0,
    in_contact_force_threshold: float = 1e-3,
) -> torch.Tensor:
    """Contact force reward.

    If no contact is present, reward is 0.
    If contact is present, reward forces within the range of lower_force_squared and upper_force_squared.

    Args:
        env: The RL environment.
        command_name: Command term that provides contact forces.
        var: Scale for exp(-contact_force / var).
        threshold: Threshold for contact allowrance.
        in_contact_force_threshold: Threshold for contact force to be considered in contact.

    Returns:
        Reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)

    right_hand_object_contact_forces_norm = (
        command.right_hand_object_contact_forces_w.square().sum(dim=-1).mean(dim=1)
    ).sum(
        dim=1
    )  # (num_envs, num_hand_link_w_sensor)
    right_hand_link_in_contact = (
        right_hand_object_contact_forces_norm > in_contact_force_threshold
    )  # (num_envs, num_hand_link_w_sensor)
    num_right_hand_links_in_contact = right_hand_link_in_contact.sum(
        dim=-1
    )  # (num_envs,)

    left_hand_object_contact_forces_norm = (
        command.left_hand_object_contact_forces_w.square().sum(dim=-1).mean(dim=1)
    ).sum(
        dim=1
    )  # (num_envs, num_hand_link_w_sensor)
    left_hand_link_in_contact = (
        left_hand_object_contact_forces_norm > in_contact_force_threshold
    )  # (num_envs, num_hand_link_w_sensor)
    num_left_hand_links_in_contact = left_hand_link_in_contact.sum(
        dim=-1
    )  # (num_envs,)

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
    in_contact_force_threshold: float = 1e-3,
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
        in_contact_force_threshold: Threshold for contact force to be considered in contact.

    Returns:
        Reward tensor (num_envs,).
    """
    command = env.command_manager.get_term(command_name)

    right_hand_object_contact_forces_norm = (
        command.right_hand_object_contact_forces_w.square().sum(dim=-1).mean(dim=1)
    ).sum(
        dim=1
    )  # (num_envs, num_hand_link_w_sensor)
    right_hand_link_in_contact = (
        right_hand_object_contact_forces_norm > in_contact_force_threshold
    )  # (num_envs, num_hand_link_w_sensor)
    num_right_hand_links_in_contact = right_hand_link_in_contact.sum(
        dim=-1
    )  # (num_envs,)

    left_hand_object_contact_forces_norm = (
        command.left_hand_object_contact_forces_w.square().sum(dim=-1).mean(dim=1)
    ).sum(
        dim=1
    )  # (num_envs, num_hand_link_w_sensor)
    left_hand_link_in_contact = (
        left_hand_object_contact_forces_norm > in_contact_force_threshold
    )  # (num_envs, num_hand_link_w_sensor)
    num_left_hand_links_in_contact = left_hand_link_in_contact.sum(
        dim=-1
    )  # (num_envs,)

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


def contact_wrench_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    in_contact_force_threshold: float = 1e-3,
) -> torch.Tensor:
    r"""Contact wrench reward based on per-direction alignment with the reference.

    Reference and agent wrench supports are evaluated **per hand** (left / right)
    against ``retargeted_{left,right}_contact_wrench_supports`` and
    ``{left,right}_hand_contact_wrench_supports``; combined activity is derived
    as the union of left and right reference supports.

    For each hand and sampled wrench-space basis direction ``b_i``:

    * If that hand's reference is zero (``ref_{h,i} \approx 0``) and its agent
      support is zero: **no contribution** from that hand for this direction.
    * If ``ref_{h,i} \approx 0`` but the agent support is non-zero: a per-cell
      penalty of **-1** (spurious wrench support where the demo has none).
    * If ``ref_{h,i} > 0`` and the agent support is non-zero: alignment
      ``clamp(agent_{h,i} / ref_{h,i}, 0, 1)`` (partial credit, capped at 1 when
      the agent meets or exceeds the reference).
    * If ``ref_{h,i} > 0`` but the agent support is zero: per-cell penalty **-1**
      (missing support where the demo requires it).

    Per basis direction, scores from hands with ``ref_{h,i} > 0`` are averaged.
    If **both** hands have zero reference on that direction but either hand shows
    non-zero agent support, the direction scores **-1**; if both agent supports
    are zero, the score is **0**.

    Alignment contributions are averaged over directions where the **combined**
    demo reference is non-zero; spurious-support penalties on directions where
    the combined reference is zero are averaged separately over all basis
    directions (so each such fault contributes ``-1 / B`` on average).

    Reward is only positive when the agent has at least one hand in contact with
    the object (contact force exceeds ``in_contact_force_threshold``).  When the
    reference requires contact but the agent has none at all, a **negative**
    penalty of ``-1`` is added, so the overall signal can span ``[-1, +1]``.

    Args:
        env: The RL environment.
        command_name: Command term that provides reference contact wrench repr.
        in_contact_force_threshold: Minimum contact force magnitude (N) to count
            a link as in contact.

    Returns:
        Reward tensor of shape (num_envs,) in [-1, 1].
    """
    cmd = env.command_manager.get_term(command_name)

    # ── reference and current wrench supports ─────────────────────────────────
    # Shapes: (N, num_bodies, B) — one support scalar per body per basis direction.
    ref_L = cmd.retargeted_left_contact_wrench_supports[cmd.timestep_counter]
    ref_R = cmd.retargeted_right_contact_wrench_supports[cmd.timestep_counter]
    curr_L = cmd.left_hand_contact_wrench_supports
    curr_R = cmd.right_hand_contact_wrench_supports

    eps = 1e-6

    def _hand_cell_score(ref_h: torch.Tensor, curr_h: torch.Tensor) -> torch.Tensor:
        """Per-hand per-body per-direction score in {-1, 0} ∪ (0, 1]."""
        pos_ref = ref_h > eps
        pos_curr = curr_h > eps
        align = (curr_h / ref_h.clamp(min=eps)).clamp(0.0, 1.0)
        return torch.where(
            pos_ref & pos_curr,
            align,
            torch.where(
                pos_ref & ~pos_curr,
                torch.full_like(ref_h, -1.0),
                torch.where(
                    ~pos_ref & pos_curr,
                    torch.full_like(ref_h, -1.0),
                    torch.zeros_like(ref_h),
                ),
            ),
        )

    mask_L = ref_L > eps  # (N, num_bodies, B)
    mask_R = ref_R > eps  # (N, num_bodies, B)
    w = mask_L.float() + mask_R.float()  # (N, num_bodies, B)
    hs_L = _hand_cell_score(ref_L, curr_L)
    hs_R = _hand_cell_score(ref_R, curr_R)
    # Combined activity: either hand has non-zero reference for this body/direction
    active_c = (ref_L > eps) | (ref_R > eps)  # (N, num_bodies, B)
    s_dir = torch.where(
        w > 0,
        (hs_L * mask_L.float() + hs_R * mask_R.float()) / w.clamp(min=1.0),
        torch.zeros_like(ref_L),
    )  # (N, num_bodies, B)
    num_active = active_c.float().sum(dim=-1)  # (N, num_bodies)
    has_ref_dir = num_active > 0  # (N, num_bodies)
    mean_from_ref = torch.where(
        has_ref_dir,
        (s_dir * active_c.float()).sum(dim=-1) / num_active.clamp(min=1.0),
        torch.zeros_like(num_active),
    )  # (N, num_bodies)
    spurious_inactive = (~active_c) & ((curr_L > eps) | (curr_R > eps))
    mean_spurious = -spurious_inactive.float().mean(dim=-1)  # (N, num_bodies)
    mean_alignment = mean_from_ref + mean_spurious  # (N, num_bodies)

    # ── contact gating ────────────────────────────────────────────────────────
    right_in_contact = (
        (
            cmd.right_hand_object_contact_forces_w.norm(dim=1).norm(dim=-1)
            > in_contact_force_threshold
        )
        .any(dim=-1)
        .any(dim=-1)
    )  # (N,)
    left_in_contact = (
        (
            cmd.left_hand_object_contact_forces_w.norm(dim=1).norm(dim=-1)
            > in_contact_force_threshold
        )
        .any(dim=-1)
        .any(dim=-1)
    )  # (N,)
    in_contact = right_in_contact | left_in_contact  # (N,)

    # Per-body: is the demo requesting contact on this body?
    ref_active_per_body = active_c.any(dim=-1)  # (N, num_bodies)

    # Average quality and penalty over bodies then return (N,)
    quality = (
        mean_alignment * in_contact.unsqueeze(-1).float() * ref_active_per_body.float()
    ).mean(dim=-1)
    penalty = -(ref_active_per_body.float() * (~in_contact).unsqueeze(-1).float()).mean(
        dim=-1
    )

    return quality + penalty


def contact_wrench_cumulative_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    eps: float = 1e-6,
    streak_scale: float = 20.0,
) -> torch.Tensor:
    """Per-body streak reward encouraging sustained contact on the correct object bodies.

    For each object body, tracks how many consecutive steps the policy has maintained
    wrench support where the reference requires it.  Streaks are per-body, so touching
    the wrong body does not inflate the reward for a body the demo actually needs.

    * **Reference active & policy active (per body):** positive streak increments;
      reward is ``+tanh(streak / streak_scale)`` in ``(0, 1)``.
    * **Reference inactive & policy active (per body):** spurious contact — negative
      streak increments; reward is ``-tanh(streak / streak_scale)`` in ``(-1, 0)``.
    * **Reference active & policy inactive:** streak resets; reward is **0**.
    * **Both inactive:** streak resets; reward is **0**.

    Per-body rewards are averaged over bodies where the reference wants contact.
    Streaks reset at episode boundaries.

    Args:
        env: The RL environment.
        command_name: Command term providing retargeted wrench repr and current
            hand wrench repr.
        eps: Threshold for treating a support scalar as non-zero.
        streak_scale: Step count at which tanh reaches ~0.76.  Larger values
            make the reward grow more slowly with streak length.

    Returns:
        Reward tensor of shape ``(num_envs,)`` in ``(-1, 1)``.
    """
    cmd = env.command_manager.get_term(command_name)

    # Shapes: (N, num_bodies, B)
    ref_L = cmd.retargeted_left_contact_wrench_supports[cmd.timestep_counter]
    ref_R = cmd.retargeted_right_contact_wrench_supports[cmd.timestep_counter]
    curr_L = cmd.left_hand_contact_wrench_supports
    curr_R = cmd.right_hand_contact_wrench_supports

    # Collapse only basis directions → (N, num_bodies), keeping per-body resolution
    ref_active = ((ref_L > eps) | (ref_R > eps)).any(dim=-1)  # (N, num_bodies)
    policy_active = (curr_L > eps).any(dim=-1) | (curr_R > eps).any(
        dim=-1
    )  # (N, num_bodies)

    N, num_bodies = ref_active.shape

    # Lazy init: streaks are (N, num_bodies) so we can track persistence per body
    if not hasattr(env, "_cwc_good_steps") or env._cwc_good_steps.shape != (
        N,
        num_bodies,
    ):
        env._cwc_good_steps = torch.zeros(
            N, num_bodies, dtype=torch.long, device=env.device
        )
        env._cwc_bad_steps = torch.zeros(
            N, num_bodies, dtype=torch.long, device=env.device
        )
        env._cwc_prev_ep_len = torch.zeros(N, dtype=torch.long, device=env.device)

    g = env._cwc_good_steps
    b = env._cwc_bad_steps

    # Reset streaks at episode boundaries; unsqueeze to broadcast over num_bodies
    el = env.episode_length_buf
    reset_episode = ((el == 0) | (el < env._cwc_prev_ep_len)).unsqueeze(-1)  # (N, 1)
    env._cwc_prev_ep_len = el.clone()
    g = torch.where(reset_episode, torch.zeros_like(g), g)
    b = torch.where(reset_episode, torch.zeros_like(b), b)

    good = ref_active & policy_active  # (N, num_bodies)
    bad_spurious = (~ref_active) & policy_active  # (N, num_bodies)

    g_new = torch.where(good, g + 1, torch.zeros_like(g))
    b_new = torch.where(bad_spurious, b + 1, torch.zeros_like(b))

    env._cwc_good_steps = g_new
    env._cwc_bad_steps = b_new

    scale = max(float(streak_scale), 1e-6)
    good_mag = torch.tanh(g_new.float() / scale)  # (N, num_bodies)
    bad_mag = torch.tanh(b_new.float() / scale)  # (N, num_bodies)
    per_body_reward = torch.where(
        good,
        good_mag,
        torch.where(bad_spurious, -bad_mag, torch.zeros_like(good_mag)),
    )  # (N, num_bodies)

    # Average over bodies where reference wants contact
    num_ref_bodies = ref_active.float().sum(dim=-1).clamp(min=1.0)  # (N,)
    return (per_body_reward * ref_active.float()).sum(dim=-1) / num_ref_bodies


def contact_wrench_continuous_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    approach_var: float = 0.05,
    in_contact_force_threshold: float = 1e-3,
) -> torch.Tensor:
    """Continuous contact wrench reward combining wrench alignment (A) and approach distance (B).

    ``total = A + B`` where:

    * **A** = ``contact_wrench_reward``: quality in (0, 1] when in contact with the
      object as the demo requires; ``-1`` flat penalty when the reference is active but
      the agent has no contact at all.
    * **B** = ``exp(-avg_min_dist / approach_var)``, gated to **zero** whenever the
      agent is already in contact.  ``avg_min_dist`` is the average over all valid
      reference contact points of the distance to the nearest fingertip.

    Args:
        env: The RL environment.
        command_name: Command term that provides reference contact wrench repr and
            contact positions.
        approach_var: Distance scale for the exponential approach reward (metres).
        in_contact_force_threshold: Minimum contact force (N) to count a link as
            in contact.

    Returns:
        Reward tensor of shape (num_envs,) in [-1, 1].
    """
    A = contact_wrench_reward(env, command_name, in_contact_force_threshold)

    cmd = env.command_manager.get_term(command_name)

    right_in_contact = (
        (
            cmd.right_hand_object_contact_forces_w.norm(dim=1).norm(dim=-1)
            > in_contact_force_threshold
        )
        .any(dim=-1)
        .any(dim=-1)
    )  # (N,)
    left_in_contact = (
        (
            cmd.left_hand_object_contact_forces_w.norm(dim=1).norm(dim=-1)
            > in_contact_force_threshold
        )
        .any(dim=-1)
        .any(dim=-1)
    )  # (N,)
    in_contact = right_in_contact | left_in_contact  # (N,)

    ref_L = cmd.retargeted_left_contact_wrench_supports[
        cmd.timestep_counter
    ]  # (N, num_bodies, B)
    ref_R = cmd.retargeted_right_contact_wrench_supports[
        cmd.timestep_counter
    ]  # (N, num_bodies, B)
    ref_active = ((ref_L > 1e-6) | (ref_R > 1e-6)).any(dim=-1).any(dim=-1)  # (N,)

    right_ref_pts = cmd.right_hand_object_contact_command_positions_e  # (N, P_r, 3)
    right_ref_valid = cmd.retargeted_right_object_contact_is_valid[
        cmd.timestep_counter
    ]  # (N, P_r)
    left_ref_pts = cmd.left_hand_object_contact_command_positions_e  # (N, P_l, 3)
    left_ref_valid = cmd.retargeted_left_object_contact_is_valid[
        cmd.timestep_counter
    ]  # (N, P_l)

    right_tips = cmd.right_hand_fingertip_position_e[..., :3]  # (N, F_r, 3)
    left_tips = cmd.left_hand_fingertip_position_e[..., :3]  # (N, F_l, 3)

    right_pair_dist = torch.norm(
        right_ref_pts.unsqueeze(2) - right_tips.unsqueeze(1), dim=-1
    )  # (N, P_r, F_r)
    right_min_dist = right_pair_dist.min(dim=-1).values  # (N, P_r)
    right_num_valid = right_ref_valid.float().sum(dim=-1).clamp(min=1.0)  # (N,)
    right_avg_dist = (right_min_dist * right_ref_valid.float()).sum(
        dim=-1
    ) / right_num_valid  # (N,)

    left_pair_dist = torch.norm(
        left_ref_pts.unsqueeze(2) - left_tips.unsqueeze(1), dim=-1
    )  # (N, P_l, F_l)
    left_min_dist = left_pair_dist.min(dim=-1).values  # (N, P_l)
    left_num_valid = left_ref_valid.float().sum(dim=-1).clamp(min=1.0)  # (N,)
    left_avg_dist = (left_min_dist * left_ref_valid.float()).sum(
        dim=-1
    ) / left_num_valid  # (N,)

    has_right_valid = right_ref_valid.any(dim=-1)  # (N,)
    has_left_valid = left_ref_valid.any(dim=-1)  # (N,)
    num_valid_hands = (has_right_valid.float() + has_left_valid.float()).clamp(min=1.0)
    avg_dist = (
        right_avg_dist * has_right_valid.float()
        + left_avg_dist * has_left_valid.float()
    ) / num_valid_hands  # (N,)

    needs_approach = ref_active & ~in_contact & (has_right_valid | has_left_valid)
    B = torch.exp(-avg_dist / approach_var) * needs_approach.float()  # (N,) in (0, 1]

    return A + B


def contact_wrench_support_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    tolerance: float = 0.1,
    var: float = 60.0,
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

    right_command_has_contact = (
        command.right_hand_contact_wrench_supports_command.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies)
    right_has_contact = (
        command.right_hand_contact_wrench_supports.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies)
    right_command_num_contact = right_command_has_contact.sum(dim=-1)  # (num_envs,)

    left_command_has_contact = (
        command.left_hand_contact_wrench_supports_command.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies)
    left_has_contact = (
        command.left_hand_contact_wrench_supports.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies)
    left_command_num_contact = left_command_has_contact.sum(dim=-1)  # (num_envs,)

    # Current supports needs to be better than command supports by at least tolerance
    right_better_than_command = (
        (1.0 - tolerance) * command.right_hand_contact_wrench_supports_command
        - command.right_hand_contact_wrench_supports
    ).clamp(min=0.0)
    left_better_than_command = (
        (1.0 - tolerance) * command.left_hand_contact_wrench_supports_command
        - command.left_hand_contact_wrench_supports
    ).clamp(min=0.0)

    # Current supports should not be arbitrarily larger than command supports
    right_not_arbitrarily_large = (
        command.right_hand_contact_wrench_supports
        - (1.0 + tolerance) * command.right_hand_contact_wrench_supports_command
    ).clamp(min=0.0)
    left_not_arbitrarily_large = (
        command.left_hand_contact_wrench_supports
        - (1.0 + tolerance) * command.left_hand_contact_wrench_supports_command
    ).clamp(min=0.0)

    # Rewards when command has contact (inclusion contact reward)
    right_contact_loss = right_better_than_command.square().sum(
        dim=-1
    ) + right_not_arbitrarily_large.square().sum(dim=-1)
    right_contact_reward = (
        (right_command_has_contact & right_has_contact)
        * torch.exp(-right_contact_loss / var)
    ).sum(dim=-1) / right_command_num_contact.clamp(min=1e-3)

    left_contact_loss = left_better_than_command.square().sum(
        dim=-1
    ) + left_not_arbitrarily_large.square().sum(dim=-1)
    left_contact_reward = (
        (left_command_has_contact & left_has_contact)
        * torch.exp(-left_contact_loss / var)
    ).sum(dim=-1) / left_command_num_contact.clamp(min=1e-3)

    return (right_contact_reward + left_contact_reward) / 2.0


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

    right_command_has_contact = (
        command.right_hand_contact_wrench_supports_command.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies)
    right_has_contact = (
        command.right_hand_contact_wrench_supports.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies)
    right_command_num_contact = right_command_has_contact.sum(dim=-1)  # (num_envs,)

    left_command_has_contact = (
        command.left_hand_contact_wrench_supports_command.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies)
    left_has_contact = (
        command.left_hand_contact_wrench_supports.amax(dim=-1) > 1e-3
    )  # (num_envs, num_bodies)
    left_command_num_contact = left_command_has_contact.sum(dim=-1)  # (num_envs,)

    # Unintended contact
    right_unintended_contact = torch.logical_and(
        ~right_command_has_contact, right_has_contact
    )
    left_unintended_contact = torch.logical_and(
        ~left_command_has_contact, left_has_contact
    )

    # Continuous penalty when command has no contact but current has contact
    right_unintended_wrench_support = (
        ~right_command_has_contact
    ).float() * command.right_hand_contact_wrench_supports.clamp(min=0.0).square().mean(
        dim=-1
    )
    right_unintended_wrench_support = right_unintended_wrench_support.sum(dim=-1) / (
        command.num_bodies - right_command_num_contact
    ).clamp(min=1e-3)

    left_unintended_wrench_support = (
        ~left_command_has_contact
    ).float() * command.left_hand_contact_wrench_supports.clamp(min=0.0).square().mean(
        dim=-1
    )
    left_unintended_wrench_support = left_unintended_wrench_support.sum(dim=-1) / (
        command.num_bodies - left_command_num_contact
    ).clamp(min=1e-3)

    return (
        right_unintended_contact.float().mean(dim=-1)
        + right_unintended_wrench_support
        + left_unintended_contact.float().mean(dim=-1)
        + left_unintended_wrench_support
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

    right_command_has_contact = (
        command.right_hand_contact_wrench_supports_command > 1e-3
    )  # (num_envs, num_bodies, num_wrench_space_basis_samples)
    right_has_contact = (
        command.right_hand_contact_wrench_supports > 1e-3
    )  # (num_envs, num_bodies, num_wrench_space_basis_samples)

    left_command_has_contact = (
        command.left_hand_contact_wrench_supports_command > 1e-3
    )  # (num_envs, num_bodies)
    left_has_contact = (
        command.left_hand_contact_wrench_supports > 1e-3
    )  # (num_envs, num_bodies)

    # Penalize missed contact
    right_missed_contact = torch.logical_and(
        right_command_has_contact, ~right_has_contact
    )
    left_missed_contact = torch.logical_and(left_command_has_contact, ~left_has_contact)

    return (
        right_missed_contact.sum(dim=-1).any(dim=-1).float()
        + left_missed_contact.sum(dim=-1).any(dim=-1).float()
    )
