from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

robot_scene_entity_cfg = SceneEntityCfg("robot")


def fall(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = robot_scene_entity_cfg,
) -> torch.Tensor:
    """Terminate when the asset falls."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    asset_height_w = asset.data.root_pos_w[:, 2]
    return asset_height_w < threshold


def timestep_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the command is completed."""
    command = env.command_manager.get_term(command_name)
    return command.timestep_counter >= command.num_timesteps - 1
