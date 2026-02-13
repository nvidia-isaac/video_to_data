from __future__ import annotations

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from .tracking_command import TrackingCommand


@configclass
class TrackingCommandCfg(CommandTermCfg):
    """Configuration for the tracking command term."""

    class_type: type = TrackingCommand

    # Asset
    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    # Object
    object_name: str = "object"
    """Name of the object asset in the scene."""

    # Motion data
    motion_file: str = ""
    """Path to the HDF5 file containing motion data."""

    object_pos_offset: list[float] = [0.0, 0.0, 0.0]
    """Offset of the object position in the world frame."""

    robot_anchor_pos_offset: list[float] = [0.0, 0.0, 0.0]
    """Offset of the robot anchor position in the world frame."""

    object_position_key: str = "apple_position"
    """HDF5 key for object position trajectory."""

    object_quaternion_key: str = "apple_wxyz"
    """HDF5 key for object quaternion trajectory."""

    ee_link_names: list[str] = []
    """Names of the end effector links on the robot. If empty, no EE tracking is performed."""

    ee_file_names: list[str] | None = None
    """Names of the EE links in the motion file (for YAML key lookup). If None, uses ee_link_names."""

    dt: float = 0.02
    """Time step of the motion data (50 Hz = 0.02s)."""

    # Tracking configuration
    anchor_body_name: str = "pelvis"
    """Name of the anchor body (usually pelvis for humanoids)."""

    joint_names: list[str] = [".*"]  # Track all joints
    """Names of the joints to track (in IsaacLab ordering)."""

    file_joint_names: list[str] | None = None
    """Names of the joints in the motion file. If None, assumes same order as IsaacLab."""

    is_ee_motion: bool = False
    """Whether the motion data is EE-based (floating-base hands) vs full-body joint positions."""

    num_future_frames: int = 10
    """Number of future frames to include in encoder observations."""

    dt_future_frames: float = 0.1
    """Time step between future frames."""

    # Visualization
    debug_vis: bool = True
    """Whether to enable debug visualization of tracking target."""

    pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_marker"
    )
    pose_visualizer_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)

    resampling_time_range: tuple[float, float] = (
        1e6,
        1e6,
    )  # no resampling based on time
