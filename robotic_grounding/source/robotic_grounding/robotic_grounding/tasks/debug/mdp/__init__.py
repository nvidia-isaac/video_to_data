# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""MDP components for debug environments."""

from .joint_pos_gui_action import JointPositionGUIAction
from .joint_pos_gui_action_cfg import JointPositionGUIActionCfg
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
    "JointPositionGUIAction",
    "JointPositionGUIActionCfg",
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
