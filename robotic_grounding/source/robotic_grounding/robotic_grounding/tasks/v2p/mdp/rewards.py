from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from robotic_grounding.tasks.v2p.mdp.observations import (
    finger_contact_force_vectors,
    finger_contact_forces,
)


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
    object_position_e = command.object_position_e.squeeze()
    object_position_error_e = torch.norm(
        command.object_body_position_command_e - object_position_e,
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
    object_wxyz = command.object_orientation_e.squeeze()
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
    object_position = command.object_position_e  # (num_envs, 1, 3)
    object_wxyz = command.object_orientation_e  # (num_envs, 1, 4)

    # Compute keypoints
    object_keypoints, _ = math_utils.combine_frame_transforms(
        object_position.repeat(1, 6, 1),
        object_wxyz.repeat(1, 6, 1),
        command.KEYPOINT_VECS,
        command.QUAT_UNIT_VEC,
    )  # (num_envs, 6, 3)
    object_command_keypoints, _ = math_utils.combine_frame_transforms(
        command.object_body_position_command_e.unsqueeze(1).repeat(1, 6, 1),
        command.object_body_wxyz_command_e.unsqueeze(1).repeat(1, 6, 1),
        command.KEYPOINT_VECS,
        command.QUAT_UNIT_VEC,
    )  # (num_envs, 6, 3)

    # Compute keypoints error
    object_keypoints_error = torch.norm(
        object_keypoints - object_command_keypoints, p=2, dim=-1
    ).norm(dim=-1)
    object_keypoints_tracking_rew = torch.exp(-object_keypoints_error / var)

    return object_keypoints_tracking_rew


def hand_keypoints_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.1,
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

    Returns:
        torch.Tensor: A tensor of shape (num_envs,) containing the imitation reward for each environment.
    """
    command = env.command_manager.get_term(command_name)

    # Extract command wrist positions
    left_hand_wrist_position_command_e = (
        command.left_hand_wrist_position_command_e
    )  # (num_envs, 3)
    right_hand_wrist_position_command_e = (
        command.right_hand_wrist_position_command_e
    )  # (num_envs, 3)

    # Compute keypoints error for wrist positions
    right_hand_wrist_position_error = torch.norm(
        right_hand_wrist_position_command_e - command.right_hand_wrist_position_e,
        dim=-1,
    )
    left_hand_wrist_position_error = torch.norm(
        left_hand_wrist_position_command_e - command.left_hand_wrist_position_e, dim=-1
    )

    # Extract command fingertip positions
    left_hand_fingertip_position_command_e = (
        command.left_hand_fingertip_position_command_e[..., :3]
    )  # (num_envs, NUM_FINGERTIPS, 3)
    right_hand_fingertip_position_command_e = (
        command.right_hand_fingertip_position_command_e[..., :3]
    )  # (num_envs, NUM_FINGERTIPS, 3)

    # Compute keypoints error for fingertip positions
    right_hand_fingertip_position_error = torch.norm(
        right_hand_fingertip_position_command_e
        - command.right_hand_fingertip_position_e,
        p=2,
        dim=-1,
    ).norm(dim=-1)
    left_hand_fingertip_position_error = torch.norm(
        left_hand_fingertip_position_command_e - command.left_hand_fingertip_position_e,
        p=2,
        dim=-1,
    ).norm(dim=-1)

    hand_keypoints_tracking_rew = (
        torch.exp(-right_hand_wrist_position_error / var)
        + torch.exp(-left_hand_wrist_position_error / var)
        + torch.exp(-right_hand_fingertip_position_error / var)
        + torch.exp(-left_hand_fingertip_position_error / var)
    )

    return hand_keypoints_tracking_rew


def hand_joint_pos_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "dual_hands_object_tracking_command",
    var: float = 0.1,
) -> torch.Tensor:
    """Compute imitation reward based on hand joint positions (num_envs,)."""
    command = env.command_manager.get_term(command_name)

    # Compute keypoints error for hand joint positions
    left_hand_joint_pos_error = torch.norm(
        command.left_hand_finger_joint_pos_command - command.left_hand_finger_joint_pos,
        dim=-1,
    )
    right_hand_joint_pos_error = torch.norm(
        command.right_hand_finger_joint_pos_command
        - command.right_hand_finger_joint_pos,
        dim=-1,
    )

    return torch.exp(-left_hand_joint_pos_error / var) + torch.exp(
        -right_hand_joint_pos_error / var
    )


def termination_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for termination (num_envs,)."""
    return env.termination_manager.terminated


def action_norm(
    env: ManagerBasedRLEnv,
    action_names: list[str],
    p: int = 1,
) -> torch.Tensor:
    """Lp norm of the actions (num_envs,)."""
    for i, action_name in enumerate(action_names):
        if i == 0:
            action_norm = torch.norm(
                env.action_manager.get_term(action_name).raw_actions, p=p, dim=-1
            )
        else:
            action_norm += torch.norm(
                env.action_manager.get_term(action_name).raw_actions, p=p, dim=-1
            )
    return action_norm
