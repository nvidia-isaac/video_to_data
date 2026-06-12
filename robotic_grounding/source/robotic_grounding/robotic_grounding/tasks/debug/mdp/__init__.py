# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MDP components for debug environments."""

from .joint_gui import JointGUIAction
from .joint_gui_cfg import JointGUIActionCfg
from .object_pose_gui_action import (
    ArticulatedObjectPoseGUIAction,
    ObjectPoseGUIAction,
    RigidObjectPoseGUIAction,
)
from .object_pose_gui_action_cfg import (
    ArticulatedObjectPoseGUIActionCfg,
    ObjectPoseGUIActionCfg,
    RigidObjectPoseGUIActionCfg,
)
from .reward_visualizer import RewardVisualizer
from .reward_visualizer_cfg import RewardVisualizerCfg
from .rewards import (
    contact_force,
    contact_pos,
)

__all__ = [
    "JointGUIAction",
    "JointGUIActionCfg",
    # Unified object pose GUI action
    "ObjectPoseGUIAction",
    "ObjectPoseGUIActionCfg",
    # Backwards compatibility aliases
    "ArticulatedObjectPoseGUIAction",
    "ArticulatedObjectPoseGUIActionCfg",
    "RigidObjectPoseGUIAction",
    "RigidObjectPoseGUIActionCfg",
    # Reward visualizer
    "RewardVisualizer",
    "RewardVisualizerCfg",
    # Reward functions
    "contact_force",
    "contact_pos",
]
