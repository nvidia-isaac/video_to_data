# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2p.mdp.commands.commands import (
    DualHandsObjectTrackingCommand,
    TrackingCommand,
)


@configclass
class TrackingCommandCfg(CommandTermCfg):
    """Configuration for the tracking command term."""

    class_type: type = TrackingCommand
    resampling_time_range: tuple[float, float] = (
        1e6,
        1e6,
    )  # no resampling based on time

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    joint_names: list[str] = [""]
    """The names of the joints to track."""

    make_quat_unique: bool = True
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    file_path: str = ""
    """Path to the file containing the motion data."""

    file_joint_order: list[str] = [""]
    """The order of the joints to track in the motion data file."""

    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Offset of the root pose in the environment frame."""

    update_goal_on_reach: bool = False
    """Whether to update the goal when the goal is reached."""

    goal_reach_threshold: float = 0.1
    """Threshold for considering the goal as reached."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/goal_marker"
    )
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)

    tips_distance_file_path: str | None = None
    """Path to the tips_distance file (optional).

    If provided, the file should contain a numpy array of shape (T, F) where T is the
    number of timesteps and F is the number of fingertips or finger sensors, as defined
    by the environment configuration. The order of fingers should match that used by the
    environment for contact sensing and observation.

    This follows the ManipTrans approach of pre-computing reference fingertip-to-object
    surface distances during data processing for use in contact rewards.
    """


@configclass
class DualHandsObjectTrackingCommandCfg(CommandTermCfg):
    """Configuration for the tracking command term."""

    class_type: type = DualHandsObjectTrackingCommand
    resampling_time_range: tuple[float, float] = (
        1e6,
        1e6,
    )  # no resampling based on time

    right_robot_name: str = "right_robot"
    """Name of the robot in the environment for which the commands are generated."""

    left_robot_name: str = "left_robot"
    """Name of the left robot in the environment for which the commands are generated."""

    wrist_joint_names: list[str] = [""]
    """Names of the wrist joints in the robot."""

    finger_joint_names: list[str] = [""]
    """Names of the finger joints in the robot."""

    wrist_body_name: str = ""
    """Name of the hand body in the robot."""

    fingertip_body_name: str = ""
    """Name of the fingertip body in the robot."""

    object_name: str = "object"
    """Name of the object in the environment for which the commands are generated."""

    object_body_names: list[str] = ["object"]
    """Names of the object body in the environment for which the commands are generated."""

    make_quat_unique: bool = True
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    motion_folder: str = ""
    """Path to the folder containing the motion data."""

    motion_filters: list[tuple[str, str, str]] = [
        ("robot_name", "=", "sharpa_wave"),
        ("sequence_id", "contains", "box_grab"),
    ]
    """Filters to apply to the motion data."""

    motion_id: int = 0
    """Index of the motion to use after applying the filters."""

    reset_state_file_path: str = ""
    """Path to the file containing the reset states."""

    target_fps: float | None = None
    """Target FPS to interpolate the motion data to. If None, use the 1 / env.step_dt."""

    reset_to_initial: bool = False
    """Whether to reset the command to the initial frame of the motion."""

    ###################################################
    # Visualizer markers
    ###################################################

    object_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/object_marker"
    )
    object_pose_visualizer_cfg.markers["frame"].scale = (0.14, 0.14, 0.14)
    """Visualizer for the object pose."""

    right_hand_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/right_hand_marker"
    )
    right_hand_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    """Visualizer for the right hand pose."""

    left_hand_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/left_hand_marker"
    )
    left_hand_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    """Visualizer for the left hand pose."""

    object_goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/object_goal_marker"
    )
    object_goal_pose_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    """Visualizer for the object goal pose."""

    right_hand_goal_pose_visualizer_cfg: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/right_hand_goal_marker")
    )
    right_hand_goal_pose_visualizer_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
    """Visualizer for the right hand goal pose."""

    left_hand_goal_pose_visualizer_cfg: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/left_hand_goal_marker")
    )
    left_hand_goal_pose_visualizer_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
    """Visualizer for the left hand goal pose."""
