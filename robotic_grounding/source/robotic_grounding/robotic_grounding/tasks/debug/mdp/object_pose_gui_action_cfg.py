# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Configuration for the Unified Object Pose GUI Action."""

from __future__ import annotations

import math
from dataclasses import field
from typing import TYPE_CHECKING, Literal

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass


@configclass
class ObjectPoseGUIActionCfg(ActionTermCfg):
    """Configuration for the unified object pose GUI action term.

    This action term allows interactive control of an object's 6DoF base pose
    via a DearPyGui window. It supports:

    1. **RigidObject**: Direct pose via `write_root_link_pose_to_sim()`
    2. **Articulation with floating base joints**: Pose via floating base joints
    3. **Articulation without floating base joints**: Direct root pose write

    The control mode is determined by the `position_joint_names` and `rotation_joint_names`
    configuration:
    - If joint names are provided, control is via floating base joints (for Articulation)
    - If joint names are None, control is via direct root pose write (for both RigidObject
      and Articulation without floating base)

    The user can adjust position (x, y, z) and orientation (roll, pitch, yaw) using sliders.

    Attributes:
        asset_name: Name of the object in the scene to control.
        position_limits: Dictionary mapping axis names to (min, max) tuples in meters.
        rotation_limits: Dictionary mapping axis names to (min, max) tuples in radians.
        position_joint_names: Optional mapping from axis names to joint names for position.
        rotation_joint_names: Optional mapping from axis names to joint names for rotation.
        control_mode: Control mode for floating base joints ("kinematic" or "pd_target").
        gui_window_title: Title of the DearPyGui window.
    """

    class_type: type[ActionTerm] = field(
        default_factory=lambda: _get_object_pose_gui_action_class()
    )
    """The class type for this action term."""

    # Object to control (uses asset_name from base ActionTermCfg)
    asset_name: str = "object"
    """Name of the object entity in the scene. Defaults to 'object'."""

    # Position limits (meters)
    position_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "x": (-2.0, 2.0),
            "y": (-2.0, 2.0),
            "z": (0.0, 3.0),
        }
    )
    """Position limits for each axis in meters. Defaults to +/-2m for x/y, 0-3m for z."""

    # Rotation limits (radians)
    rotation_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "roll": (-math.pi, math.pi),
            "pitch": (-math.pi, math.pi),
            "yaw": (-math.pi, math.pi),
        }
    )
    """Rotation limits for each Euler angle in radians. Defaults to +/-pi for all axes."""

    # Optional joint name mapping for floating base (Articulation only)
    # If None, pose is applied directly to root (works for both Rigid and Articulation)
    position_joint_names: dict[str, str] | None = None
    """Optional mapping from axis names to joint names for position control.

    If provided, the object must be an Articulation with matching floating base joints.
    Example: {"x": "base_x", "y": "base_y", "z": "base_z"}

    If None, pose is applied directly via write_root_link_pose_to_sim().
    """

    rotation_joint_names: dict[str, str] | None = None
    """Optional mapping from axis names to joint names for rotation control.

    If provided, the object must be an Articulation with matching floating base joints.
    Example: {"roll": "base_roll", "pitch": "base_pitch", "yaw": "base_yaw"}

    If None, pose is applied directly via write_root_link_pose_to_sim().
    """

    # Control mode for floating base joints
    control_mode: Literal["kinematic", "pd_target"] = "kinematic"
    """Control mode when using floating base joints.

    - "kinematic": Directly write joint positions (immediate, no dynamics)
    - "pd_target": Set joint position targets (follows PD controller dynamics)

    Defaults to "kinematic" for direct pose control.
    """

    # GUI settings
    gui_window_title: str = "Object Pose Controller"
    """Title of the DearPyGui window. Defaults to 'Object Pose Controller'."""

    gui_window_width: int = 500
    """Width of the GUI window in pixels. Defaults to 500."""

    gui_window_height: int = 400
    """Height of the GUI window in pixels. Defaults to 400."""


def _get_object_pose_gui_action_class() -> type[ActionTerm]:
    """Lazy import to avoid circular dependency."""
    from .object_pose_gui_action import ObjectPoseGUIAction  # noqa: PLC0415

    return ObjectPoseGUIAction


# Backwards compatibility aliases
ArticulatedObjectPoseGUIActionCfg = ObjectPoseGUIActionCfg
RigidObjectPoseGUIActionCfg = ObjectPoseGUIActionCfg
