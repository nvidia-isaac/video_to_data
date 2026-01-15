# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaaclab.utils import configclass

from robotic_grounding.assets.arctic_object import ARCTIC_OBJECT_CFG
from robotic_grounding.assets.dual_sharpa_wave import (
    DUAL_SHARPA_WAVE_ACTION_SCALE,
    DUAL_SHARPA_WAVE_CFG,
)
from robotic_grounding.tasks.v2p.v2p_hand_env_cfg import V2PHandEnvCfg


@configclass
class SharpaV2PEnvCfg(V2PHandEnvCfg):
    """Configuration for the Sharpa V2P environment."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        self.scene.robot = DUAL_SHARPA_WAVE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "right_wrist_x": -0.4,
            "right_wrist_y": 0.25,
            "right_wrist_roll": 2.08,
            "right_wrist_pitch": -0.48,
            "right_wrist_yaw": -0.317,
            "left_wrist_x": 0.4,
            "left_wrist_y": 0.35,
            "left_wrist_roll": -0.6,
            "left_wrist_pitch": 2.24,
            "left_wrist_yaw": -0.46,
            ".*_wrist_z": 1.0,
        }

        self.actions.joint_pos.scale = DUAL_SHARPA_WAVE_ACTION_SCALE

        self.scene.object = ARCTIC_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")
        self.scene.object.init_state.joint_pos = {
            "base_x": 0.0,
            "base_y": 0.0,
            "base_z": 2.0,
            "base_roll": 0.0,
            "base_pitch": 0.0,
            "base_yaw": 0.0,
            "rotation": 0.0,
        }


@configclass
class SharpaV2PEnvCfgPlay(SharpaV2PEnvCfg):
    """Configuration for the Sharpa V2P environment for playing."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        self.scene.num_envs = 16
