from __future__ import annotations

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
