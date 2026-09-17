# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-free observation terms for whole-body VLA recording."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_pos_ordered(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return raw joint radians in the explicit order resolved by ``asset_cfg``."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def body_position_e(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened body positions in each environment frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
    return (positions - env.scene.env_origins.unsqueeze(1)).reshape(env.num_envs, -1)


def body_orientation_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return flattened body wxyz orientations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_quat_w[:, asset_cfg.body_ids].reshape(env.num_envs, -1)
