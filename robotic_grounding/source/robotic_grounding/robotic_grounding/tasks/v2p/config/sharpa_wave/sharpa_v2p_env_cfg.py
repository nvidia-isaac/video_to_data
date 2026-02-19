# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

from isaaclab.utils import configclass

from robotic_grounding.assets import ASSET_DIR
from robotic_grounding.assets.rigid_object import (
    RIGID_OBJECT_CFG,
)
from robotic_grounding.assets.sharpa_wave import (
    FINGER_JOINTS,
    FINGERTIP_BODY_NAME,
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

        # FIXME: need to read contact info from commands
        # # Dynamically set contact sensors
        # # Note: elastomer links are merged into parent *_DP (distal phalange) rigid bodies
        # # PhysX limitation: Each filter pattern must match exactly 1 prim
        # finger_sensor_names = []

        # # Create sensors for bottom part of object
        # for body_name in FINGERTIP_CONTACT_BODIES:
        #     sensor_name = f"{body_name.replace('_DP', '')}_contact_sensor_bottom"
        #     setattr(
        #         self.scene,
        #         sensor_name,
        #         ContactSensorCfg(
        #             prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        #             track_pose=True,
        #             debug_vis=False,
        #             force_threshold=0.1,
        #             filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/bottom"],
        #             track_contact_points=True,
        #             max_contact_data_count_per_prim=8,
        #         ),
        #     )
        #     finger_sensor_names.append(sensor_name)

        # # Create sensors for top part of object
        # for body_name in FINGERTIP_CONTACT_BODIES:
        #     sensor_name = f"{body_name.replace('_DP', '')}_contact_sensor_top"
        #     setattr(
        #         self.scene,
        #         sensor_name,
        #         ContactSensorCfg(
        #             prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        #             track_pose=True,
        #             debug_vis=False,
        #             force_threshold=0.1,
        #             filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/top"],
        #             track_contact_points=True,
        #             max_contact_data_count_per_prim=8,
        #         ),
        #     )
        #     finger_sensor_names.append(sensor_name)

        # # Store sensor names on env config for observations to access
        # # This provides the contract: robot-specific config provides sensor names
        # self.finger_sensor_names = finger_sensor_names

        # # FIXME: We need to unify the data loading once we have a proper command or reset logic.
        # # Load tips_distance from processed parquet for ManipTrans contact reward
        # processed_dir = os.path.join(ASSET_DIR, "human_motion_data", "arctic_processed")
        # try:
        #     motion_data = ManoSharpaData.from_parquet(
        #         processed_dir,
        #         filters=[
        #             ("robot_name", "=", "sharpa_wave"),
        #             ("sequence_id", "contains", "box_grab"),
        #         ],
        #     )
        #     if (
        #         motion_data.mano_right_tips_distance
        #         and motion_data.mano_left_tips_distance
        #     ):
        #         right = np.array(motion_data.mano_right_tips_distance)  # (T, 5)
        #         left = np.array(motion_data.mano_left_tips_distance)  # (T, 5)
        #         self.tips_distance_data = np.concatenate(
        #             [right, left], axis=1
        #         )  # (T, 10)
        #         self.tips_distance_fps = motion_data.fps
        #     else:
        #         logger.warning(
        #             "tips_distance columns empty in parquet, contact reward disabled"
        #         )
        #         self.tips_distance_data = None
        # except Exception as e:
        #     logger.warning("Could not load tips_distance from parquet: %s", e)
        #     self.tips_distance_data = None


@configclass
class SharpaV2PEnvCfgPlay(SharpaV2PEnvCfg):
    """Configuration for the Sharpa V2P environment for playing."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        self.scene.num_envs = 16
