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

from robotic_grounding.tasks.v2p.mdp.commands.commands import TrackingCommand


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
