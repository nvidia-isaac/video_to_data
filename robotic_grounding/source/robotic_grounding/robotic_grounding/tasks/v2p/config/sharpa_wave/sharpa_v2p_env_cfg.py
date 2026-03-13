# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.assets.rigid_object import (
    RIGID_OBJECT_CFG,
)
from robotic_grounding.assets.sharpa_wave import (
    FINGER_JOINTS,
    FINGERTIP_BODY_NAME,
    HAND_CONTACT_BODIES,
    LEFT_SHARPA_WAVE_CFG,
    RIGHT_SHARPA_WAVE_CFG,
    WRIST_BODY_NAME,
    WRIST_JOINTS,
)
from robotic_grounding.tasks.v2p.v2p_hand_env_cfg import V2PHandEnvCfg


@configclass
class SharpaV2PEnvCfg(V2PHandEnvCfg):
    """Configuration for the Sharpa V2P environment."""

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

        # Set object
        self.scene.object = RIGID_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")

        # Set commands
        self.commands.dual_hands_object_tracking_command.motion_folder = os.path.join(
            ASSET_DIR, "human_motion_data", "arctic_processed"
        )
        self.commands.dual_hands_object_tracking_command.reset_state_file_path = (
            os.path.join(ASSET_DIR, "human_motion_data", "reset_trajectory.pt")
        )
        self.commands.dual_hands_object_tracking_command.wrist_joint_names = (
            WRIST_JOINTS
        )
        self.commands.dual_hands_object_tracking_command.finger_joint_names = (
            FINGER_JOINTS
        )
        self.commands.dual_hands_object_tracking_command.wrist_body_name = (
            WRIST_BODY_NAME
        )
        self.commands.dual_hands_object_tracking_command.fingertip_body_name = (
            FINGERTIP_BODY_NAME
        )
        self.commands.dual_hands_object_tracking_command.motion_filters = [
            ("robot_name", "=", "sharpa_wave"),
            ("sequence_id", "contains", "box_grab"),
        ]
        self.commands.dual_hands_object_tracking_command.motion_id = 0

        # FIXME: this needs to be more general.
        # Contact sensors per (object part, hand link) for contact-link reward.
        # The scene uses RIGID_OBJECT_CFG (box_rigid.urdf) with a single link named "object".
        # We filter by that body for both "bottom" and "top" so sensors resolve; part semantics
        # are preserved in reward/command (demo part_id 1/2) while policy sees one body.
        # Part order: bottom, top.
        # self.hand_contact_sensor_names = []
        # Actual rigid body link name from box_rigid.urdf (single body, no bottom/top)
        self.object_contact_sensor_names = []
        right_filter_prim_paths_expr = []
        left_filter_prim_paths_expr = []
        for body_name in HAND_CONTACT_BODIES:
            side_body_name = body_name.replace(".*", "right")
            right_filter_prim_paths_expr.append(
                f"{{ENV_REGEX_NS}}/RightRobot/{side_body_name}"
            )
            side_body_name = body_name.replace(".*", "left")
            left_filter_prim_paths_expr.append(
                f"{{ENV_REGEX_NS}}/LeftRobot/{side_body_name}"
            )
        for side in ["right", "left"]:
            sensor_name = f"object_{side}_contact_sensor"
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Object/object",
                    track_pose=True,
                    debug_vis=False,
                    force_threshold=1.0,
                    history_length=3,
                    filter_prim_paths_expr=(
                        right_filter_prim_paths_expr
                        if side == "right"
                        else left_filter_prim_paths_expr
                    ),
                    track_contact_points=True,
                    track_air_time=True,
                    max_contact_data_count_per_prim=128,
                ),
            )
            self.object_contact_sensor_names.append(sensor_name)


@configclass
class SharpaV2PEnvCfgPlay(SharpaV2PEnvCfg):
    """Configuration for the Sharpa V2P environment for playing."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        self.scene.num_envs = 16
