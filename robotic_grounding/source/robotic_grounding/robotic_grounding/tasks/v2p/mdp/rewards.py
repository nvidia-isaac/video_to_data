from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from robotic_grounding.tasks.v2p.mdp.observations import finger_contact_forces


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
