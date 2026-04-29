# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2p.mdp.commands.commands import (
    DualHandsTrackingCommand,
    TrackingCommand,
)
from robotic_grounding.tasks.v2p.mdp.commands.hand_object_commands import (
    DualHandsObjectTrackingCommand,
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
    resampling_time_range: tuple[float, float] = (1e6, 1e6)
    """No resampling based on time."""

    debug_vis: bool = False
    """Whether to visualize the debug markers."""

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

    motion_speed: float = 0.5
    """Speed factor to interpolate the motion data to."""

    reset_finger_openness: float = 0.7
    """Max interpolation factor for finger joints at reset.

    At reset, each env samples a uniform factor in [0, reset_finger_openness].
    The finger joint positions are then: factor * reference_finger_joints.
    0.0 = fully open, 1.0 = fully matching reference.
    """

    always_reset_to_first_frame: bool = False
    """Whether to always reset to the first frame of the motion data."""

    initial_virtual_object_control_curriculum_scale: float = 1.0
    """Initial virtual object control curriculum scale."""

    virtual_object_control_decay_steps: int = 20
    """Number of steps over which the virtual object control factor decays after reset."""

    virtual_object_control_decay_mode: str = "step"
    """Decay mode for the virtual object control factor after reset.

    - ``"linear"``: linearly decay from 1 to ``virtual_object_control_curriculum_scale``
      over ``virtual_object_control_decay_steps``.
    - ``"step"``: hold at 1 for ``virtual_object_control_decay_steps``, then drop to
      ``virtual_object_control_decay_steps`` instantly.
    """

    recompute_hand_keypoints_from_object: bool = True
    """Whether to recompute the hand keypoints based on the object frame."""

    num_friction_cone_edges: int = 8
    """Number of friction cone edges to sample."""

    num_wrench_space_basis_samples: int = 512
    """Number of basis samples to draw for the wrench space support function."""

    friction_coefficients: float = 0.1
    """Friction coefficient for the wrench space support function."""

    enable_additional_metrics: bool = False
    """Whether to log additional diagnostic metrics (contact wrench CV, coverage, etc.).

    Enables contact_wrench_support_reward_cv and related W&B metrics.
    Off by default to avoid compute overhead in production runs.
    """

    ###################################################
    # Visualizer markers
    ###################################################

    object_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/object_marker"
    )
    object_pose_visualizer_cfg.markers["frame"].scale = (0.07, 0.07, 0.07)
    """Visualizer for the object pose."""

    object_com_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/object_com_marker"
    )
    object_com_pose_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    """Visualizer for the object COM frame."""

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
    object_goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
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

    target_contact_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/TargetContact",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.4, 1.0)
                ),
            ),
        },
    )
    """Visualizer for demo target contact link positions (from parquet). Same idea as arctic_to_sharpa add_icosphere(radius=0.005)."""

    current_contact_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/CurrentContact",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.4, 0.0, 1.0)
                ),
            ),
        },
    )
    """Visualizer for current contact link positions. """

    target_fingertip_position_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/Command/TargetFingertip",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),
            },
        )
    )

    current_fingertip_position_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/Command/CurrentFingertip",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0)
                    ),
                ),
            },
        )
    )


@configclass
class DualHandsTrackingCommandCfg(CommandTermCfg):
    """Configuration for the dual-hand tracking command term (no object)."""

    class_type: type = DualHandsTrackingCommand
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

    motion_speed: float = 0.2
    """Speed factor to interpolate the motion data to."""

    reset_finger_openness: float = 0.7
    """Max interpolation factor for finger joints at reset.

    At reset, each env samples a uniform factor in [0, reset_finger_openness].
    The finger joint positions are then: factor * reference_finger_joints.
    0.0 = fully open, 1.0 = fully matching reference.
    """

    ###################################################
    # Visualizer markers
    ###################################################

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

    target_fingertip_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/TargetFingertip",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
        },
    )
    """Visualizer for target fingertip positions (red)."""

    current_fingertip_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/CurrentFingertip",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                ),
            ),
        },
    )
    """Visualizer for current fingertip positions (green)."""
