# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Configuration for the Joint Position GUI Action."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Literal

from isaaclab.envs import mdp
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass


@configclass
class JointPositionGUIActionCfg(mdp.JointActionCfg):
    """Configuration for the joint position GUI action term.

    This action term allows interactive control of robot joint positions via a
    DearPyGui window. The user can adjust joint positions, stiffness (P-gain),
    and damping (D-gain) using sliders.

    Attributes:
        use_default_offset: Whether to use default joint positions as offset.
        control_mode: How to apply joint positions ("kinematic", "pd_target", or "both").
        show_gains_sliders: Whether to show P/D gain sliders in the GUI.
        max_stiffness: Maximum stiffness for the P-gain slider.
        max_damping: Maximum damping for the D-gain slider.
    """

    class_type: type[ActionTerm] = field(
        default_factory=lambda: _get_joint_position_gui_action_class()
    )
    """The class type for this action term."""

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """

    control_mode: Literal["kinematic", "pd_target"] = "kinematic"
    """How to apply joint positions. Defaults to "kinematic".

    - "kinematic": Directly write joint state to simulation (bypasses physics/PD controller).
      Best for debugging and manual positioning. Joints move instantly.
    - "pd_target": Set joint position targets for implicit PD controller.
      Joint movement depends on actuator stiffness/damping. Physics-based movement.
    """

    show_gains_sliders: bool = False
    """Whether to show P/D gain sliders in the GUI. Defaults to False.

    Only relevant when control_mode is "pd_target".
    When True, users can adjust stiffness (P-gain) and damping (D-gain) per joint.
    """

    max_stiffness: float = 200.0
    """Maximum stiffness for the P-gain slider. Defaults to 200.0."""

    max_damping: float = 25.0
    """Maximum damping for the D-gain slider. Defaults to 25.0."""


def _get_joint_position_gui_action_class() -> type[ActionTerm]:
    """Lazy import to avoid circular dependency."""
    from .joint_pos_gui_action import JointPositionGUIAction  # noqa: PLC0415

    return JointPositionGUIAction
