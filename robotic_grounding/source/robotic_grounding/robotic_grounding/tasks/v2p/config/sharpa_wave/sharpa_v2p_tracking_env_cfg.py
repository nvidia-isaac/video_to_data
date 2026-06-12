# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sharpa Wave hand-only tracking environment configuration (no object)."""

import os

from isaaclab.utils import configclass

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.assets.sharpa_wave import (
    FINGER_JOINTS,
    FINGERTIP_BODY_NAME,
    LEFT_SHARPA_WAVE_CFG,
    RIGHT_SHARPA_WAVE_CFG,
    WRIST_BODY_NAME,
    WRIST_JOINTS,
)
from robotic_grounding.tasks.v2p.v2p_hand_tracking_env_cfg import V2PHandTrackingEnvCfg


@configclass
class SharpaV2PTrackingEnvCfg(V2PHandTrackingEnvCfg):
    """Configuration for the Sharpa V2P tracking-only environment (no object)."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # Set robots
        self.scene.right_robot = RIGHT_SHARPA_WAVE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RightRobot"
        )
        self.scene.left_robot = LEFT_SHARPA_WAVE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/LeftRobot"
        )

        # Set commands
        self.commands.dual_hands_tracking_command.motion_folder = os.path.join(
            ASSET_DIR, "human_motion_data", "arctic", "arctic_processed"
        )
        self.commands.dual_hands_tracking_command.wrist_joint_names = WRIST_JOINTS
        self.commands.dual_hands_tracking_command.finger_joint_names = FINGER_JOINTS
        self.commands.dual_hands_tracking_command.wrist_body_name = WRIST_BODY_NAME
        self.commands.dual_hands_tracking_command.fingertip_body_name = (
            FINGERTIP_BODY_NAME
        )
        self.commands.dual_hands_tracking_command.motion_filters = [
            ("robot_name", "=", "sharpa_wave"),
            ("sequence_id", "contains", "box_grab"),
        ]
        self.commands.dual_hands_tracking_command.motion_id = 0


@configclass
class SharpaV2PTrackingEnvCfgPlay(SharpaV2PTrackingEnvCfg):
    """Configuration for the Sharpa V2P tracking-only environment for playing."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        self.scene.num_envs = 16
