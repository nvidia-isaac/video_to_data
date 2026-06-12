# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observation functions for SONIC policy/decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

from robotic_grounding.tasks.v2p_whole_body.mdp.actions import SONICActionBase


def joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sonic_joints_only: bool = False,
    action_name: str | None = None,
) -> torch.Tensor:
    """Get current joint positions (absolute).

    Args:
        env: The environment instance
        asset_cfg: The asset configuration
        sonic_joints_only: If True, filter to SONIC-controlled joints only (requires SONIC action term)
        action_name: Name of the action term (required when sonic_joints_only=True)

    Returns:
        Joint positions (num_envs, num_joints)
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos

    if sonic_joints_only:
        if action_name is None:
            raise ValueError("action_name must be provided when sonic_joints_only=True")
        action_term = env.action_manager.get_term(action_name)
        if isinstance(action_term, SONICActionBase):
            sonic_joint_ids = action_term.get_sonic_joint_ids()
            joint_pos = joint_pos[:, sonic_joint_ids]

    return joint_pos


def joint_pos_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sonic_joints_only: bool = False,
    action_name: str | None = None,
) -> torch.Tensor:
    """Get current joint positions (relative to default).

    Args:
        env: The environment instance
        asset_cfg: The asset configuration
        sonic_joints_only: If True, filter to SONIC-controlled joints only (requires SONIC action term)
        action_name: Name of the action term (required when sonic_joints_only=True)

    Returns:
        Joint positions (num_envs, num_joints)
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos - asset.data.default_joint_pos

    if sonic_joints_only:
        if action_name is None:
            raise ValueError("action_name must be provided when sonic_joints_only=True")
        action_term = env.action_manager.get_term(action_name)
        if isinstance(action_term, SONICActionBase):
            sonic_joint_ids = action_term.get_sonic_joint_ids()
            joint_pos = joint_pos[:, sonic_joint_ids]

    return joint_pos


def joint_vel_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sonic_joints_only: bool = False,
    action_name: str | None = None,
) -> torch.Tensor:
    """Get current joint velocities.

    Args:
        env: The environment instance
        asset_cfg: The asset configuration
        sonic_joints_only: If True, filter to SONIC-controlled joints only (requires SONIC action term)
        action_name: Name of the action term (required when sonic_joints_only=True)

    Returns:
        Joint velocities (num_envs, num_joints)
    """
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel

    if sonic_joints_only:
        if action_name is None:
            raise ValueError("action_name must be provided when sonic_joints_only=True")
        action_term = env.action_manager.get_term(action_name)
        if isinstance(action_term, SONICActionBase):
            sonic_joint_ids = action_term.get_sonic_joint_ids()
            joint_vel = joint_vel[:, sonic_joint_ids]

    return joint_vel


def last_action(
    env: ManagerBasedRLEnv,
    sonic_joints_only: bool = False,
    action_name: str | None = None,
) -> torch.Tensor:
    """Get last actions.

    Args:
        env: The environment instance
        sonic_joints_only: If True, return SONIC output actions only (requires SONIC action term)
                          If False, return raw actions from the action manager
        action_name: Name of the action term (required when sonic_joints_only=True)

    Returns:
        Last actions (num_envs, num_joints)
    """
    if action_name is None:
        action_name = "joint_pos"

    action_term = env.action_manager.get_term(action_name)

    if sonic_joints_only:
        if isinstance(action_term, SONICActionBase):
            return action_term.get_last_sonic_actions()
        else:
            # Fallback to raw actions if not a SONIC action term
            return action_term.raw_actions
    else:
        return action_term.raw_actions
