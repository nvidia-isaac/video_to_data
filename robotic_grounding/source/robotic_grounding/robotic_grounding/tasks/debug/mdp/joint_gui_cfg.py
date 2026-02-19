# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Configuration for the Joint GUI Action term."""

from __future__ import annotations

from dataclasses import field
from typing import Literal

from isaaclab.envs import mdp
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


@configclass
class JointGUIActionCfg(mdp.JointActionCfg):
    """Configuration for the joint GUI action term.

    This action term provides interactive control of robot joints via a DearPyGui
    window. Position-controlled joints are adjusted with position sliders, while
    velocity-controlled joints (e.g. virtual base DOFs) get separate velocity
    sliders. A separate GUI thread is spawned so that the physics simulation can
    continue to run on the main thread.

    Control modes (for position-controlled joints):

    - ``"kinematic"``: Directly write joint state to simulation (bypasses the PD
      controller). Joints move instantly—best for debugging and manual posing.
    - ``"pd_target"``: Set joint position targets for the implicit PD controller.
      Joints move according to actuator stiffness/damping.
    """

    class_type: type[ActionTerm] = field(
        default_factory=lambda: _get_joint_gui_action_class()
    )
    """The class type for this action term."""

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset
    as offset. Defaults to ``True``.

    When enabled, the values of :attr:`offset` are overwritten with the default
    joint positions from the articulation asset.
    """

    control_mode: Literal["kinematic", "pd_target"] = "kinematic"
    """How to apply joint positions. Defaults to ``"kinematic"``.

    - ``"kinematic"``: Directly write joint state to simulation (bypasses
      physics/PD controller). Best for debugging and manual positioning.
    - ``"pd_target"``: Set joint position targets for the implicit PD controller.
      Joint movement depends on actuator stiffness/damping.
    """

    show_gains_sliders: bool = False
    """Whether to show P/D gain sliders in the GUI. Defaults to ``False``.

    Only relevant when ``control_mode`` is ``"pd_target"``. When enabled, users
    can adjust stiffness (P-gain) and damping (D-gain) per joint.
    """

    max_stiffness: float = 200.0
    """Maximum stiffness for the P-gain slider. Defaults to ``200.0``."""

    max_damping: float = 25.0
    """Maximum damping for the D-gain slider. Defaults to ``25.0``."""

    velocity_joint_names: list[str] | None = None
    """Optional list of joint names controlled via velocity sliders.

    When set, the listed joints are shown in a dedicated *Velocity Control*
    section of the GUI. Velocity targets are applied via
    ``set_joint_velocity_target``; all other joints still use position control.
    """


def _get_joint_gui_action_class() -> type[ActionTerm]:
    """Lazy import to avoid circular dependency."""
    from .joint_gui import JointGUIAction  # noqa: PLC0415

    return JointGUIAction
