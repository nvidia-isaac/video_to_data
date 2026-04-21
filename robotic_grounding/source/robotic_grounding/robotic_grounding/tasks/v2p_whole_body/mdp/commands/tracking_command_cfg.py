from __future__ import annotations

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .tracking_command import TrackingCommand


@configclass
class TrackingCommandCfg(CommandTermCfg):
    """Configuration for the whole-body tracking command term.

    Loads motion data from a planner parquet file containing body qpos,
    EE targets, hand keypoints, contacts, and object trajectories.
    """

    class_type: type = TrackingCommand

    # Asset names in the scene
    asset_name: str = "robot"
    """Name of the robot articulation asset in the scene."""

    object_name: str = "object"
    """Name of the primary object asset in the scene."""

    object_body_names: list[str] = []
    """Scene attribute names for object assets. Set by apply_scene_config()."""

    # Motion data source (planner parquet)
    motion_file: str = ""
    """Path to the planner parquet file or Hive-partitioned directory."""

    # Offsets applied to loaded trajectories
    object_pos_offset: list[float] = [0.0, 0.0, 0.0]
    """Offset applied to object position trajectory."""

    robot_anchor_pos_offset: list[float] = [0.0, 0.0, 0.0]
    """Offset applied to robot root position trajectory."""

    # Timing
    dt: float = 0.02
    """Time step of the motion data (50 Hz = 0.02s)."""

    # Body tracking
    anchor_body_name: str = "pelvis"
    """Name of the root/anchor body on the robot."""

    joint_names: list[str] = [".*"]
    """Joint name patterns to track (IsaacLab ordering)."""

    file_joint_names: list[str] | None = None
    """Joint names in the motion file (for reordering). Auto-detected from parquet."""

    # EE tracking
    ee_link_names: list[str] = []
    """EE link names on the robot for tracking. Auto-detected from parquet."""

    # Future frame observations (for SONIC encoder)
    num_future_frames: int = 10
    """Number of future frames in encoder observations."""

    dt_future_frames: float = 0.1
    """Time step between future frames."""

    # Hand skeleton configuration
    fingertip_body_name: str = ""
    """Regex for fingertip body names on the robot (e.g. '.*_DP' for Sharpa)."""

    finger_joint_names: list[str] = []
    """Names of finger joints on the robot."""

    # Contact and hand-object configuration
    hand_contact_bodies: list[str] = []
    """Hand body patterns for contact sensors (e.g. ['.*_hand_palm_link', ...]).
    Set by env cfg. Used by apply_scene_config to create contact sensors."""

    hand_frame_target_bodies: list[str] = []
    """Left and right hand body names for FrameTransformer targets.
    E.g. ['left_hand_palm_link', 'right_hand_palm_link']. Set by env cfg."""

    object_contact_sensor_names: list[str] = []
    """Scene sensor names. Populated by apply_scene_config, not set manually."""

    # Virtual Object Control (VOC) curriculum
    initial_virtual_object_control_curriculum_scale: float = 0.0
    """VOC scale at episode start. 0.0 = disabled."""

    voc_decay_steps: int = 0
    """Steps to decay VOC from voc_reset_scale toward curriculum scale."""

    voc_reset_scale: float = 1.0
    """VOC scale to set on reset (before decay)."""

    # Action history
    action_history_length: int = 3
    """Number of past processed actions to store for policy observation."""

    # Reset behavior
    always_reset_to_first_frame: bool = False
    """Force reset to frame 0 (for eval). Overrides trajectory_time_index."""

    reset_freeze_steps: int = 0
    """Steps after reset to freeze the timestep counter (robot settling time)."""

    reset_shoulder_spread: float = 0.0
    """Shoulder yaw spread (radians) applied on reset when freeze_steps > 0.
    Widens arms + zeros fingers to prevent hand-object collision during settling."""

    reset_root_height_min: float | None = None
    """Clamp root Z to this minimum on reset. None = no clamp. Use 0.795 for G1."""

    reset_yaw_only: bool = False
    """Zero roll/pitch from root quaternion on reset, keeping only yaw."""

    target_fps: float | None = None
    """Resample V2P data to this FPS. If None, uses data as-is."""

    # Wrench computation
    num_wrench_space_basis_samples: int = 512
    """Number of basis samples for wrench space support function."""

    num_friction_cone_edges: int = 8
    """Number of edges in the friction cone approximation."""

    friction_coefficients: float = 0.1
    """Friction coefficient for contact wrench computation."""

    # Visualization
    debug_vis: bool = True
    """Whether to enable debug visualization of tracking targets."""

    pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_marker"
    )
    pose_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)

    resampling_time_range: tuple[float, float] = (1e6, 1e6)
    """No time-based resampling."""
