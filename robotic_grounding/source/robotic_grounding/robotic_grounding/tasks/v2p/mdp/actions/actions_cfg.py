# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2p.mdp.actions.action_chunks import (
    JointPositionActionChunk,
)
from robotic_grounding.tasks.v2p.mdp.actions.action_direct import (
    JointDirectPositionAction,
)
from robotic_grounding.tasks.v2p.mdp.actions.action_track_residual import (
    JointResidualWithTrackingAction,
)
from robotic_grounding.tasks.v2p.mdp.actions.virtual_articulated_object_control import (
    VirtualArticulatedObjectControl,
)
from robotic_grounding.tasks.v2p.mdp.actions.virtual_rigid_object_control import (
    VirtualRigidObjectControl,
)


@configclass
class JointPositionActionChunkCfg(ActionTermCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionActionChunk` for more details.
    """

    class_type: type[ActionTerm] = JointPositionActionChunk

    joint_names: list[str] = [""]
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 0.1
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""

    num_prediction_nodes: int = 4
    """The number of prediction nodes in the action chunk. Defaults to 4."""

    num_execution_steps: int = 100
    """The number of execution steps, expand prediction nodes to execution steps for execution. Defaults to 100."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""

    use_relative_actions: bool = True
    """Whether to use relative actions. Defaults to True."""


@configclass
class JointResidualWithTrackingActionCfg(ActionTermCfg):
    """Configuration for the residual joint position with tracking controller action term.

    See :class:`JointResidualWithTrackingAction` for more details.
    """

    class_type: type[ActionTerm] = JointResidualWithTrackingAction

    joint_names: list[str] = [""]
    """List of joint names or regex expressions that the action will be mapped to."""

    command_name: str = "dual_hands_object_tracking_command"
    """Name of the command to use for the action."""

    wrist_position_scale: float = 0.05
    """Scale factor for policy's residual action for wrist position."""

    wrist_orientation_scale: float = 0.1
    """Scale factor for policy's residual action for wrist orientation."""

    finger_joint_scale: float = 0.1
    """Scale factor for policy's residual action for finger joints."""

    wrist_position_clip: float = 0.2
    """Clip factor for policy's residual action for wrist position."""

    wrist_orientation_clip: float = 1.0
    """Clip factor for policy's residual action for wrist orientation."""

    finger_joint_clip: float = 1.0
    """Clip factor for policy's residual action for finger joints."""

    ema_factor: float = 0.1
    """The EMA decay factor for the actions. The higher the factor, the more weight is given to the previous actions. Defaults to 0.1."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""

    clip_to_torque_limit: bool = True
    """Whether to clip the actions to the torque limit. Defaults to True."""

    tracking_controller_linear_stiffness: float = 100.0
    """Stiffness gain for the tracking controller in linear direction."""

    tracking_controller_linear_damping: float = 10.0
    """Damping gain for the tracking controller in linear direction."""

    tracking_controller_angular_stiffness: float = 50.0
    """Stiffness gain for the tracking controller in angular direction."""

    tracking_controller_angular_damping: float = 0.0
    """Damping gain for the tracking controller in angular direction."""

    max_force: float = 100.0
    """Maximum force for the tracking controller."""

    max_torque: float = 100.0
    """Maximum torque for the tracking controller."""


@configclass
class JointDirectPositionActionCfg(ActionTermCfg):
    """Configuration for the direct joint position action term.

    See :class:`JointDirectPositionAction` for more details.
    """

    class_type: type[ActionTerm] = JointDirectPositionAction

    joint_names: list[str] = [""]
    """List of joint names or regex expressions that the action will be mapped to."""

    command_name: str = "dual_hands_object_tracking_command"
    """Name of the command to use for the action."""

    wrist_position_scale: float = 0.05
    """Scale factor for policy's action for wrist position delta."""

    wrist_orientation_scale: float = 0.15
    """Scale factor for policy's action for wrist orientation delta."""

    finger_joint_scale: float = 0.15
    """Scale factor for policy's action for finger joint delta."""

    wrist_position_clip: float = 0.2
    """Clip factor for policy's action for wrist position."""

    wrist_orientation_clip: float = 0.8
    """Clip factor for policy's action for wrist orientation."""

    finger_joint_clip: float = 100.0
    """Clip factor for policy's action for finger joints."""

    ema_factor: float = 0.9
    """The EMA decay factor for the actions."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output."""

    clip_to_torque_limit: bool = True
    """Whether to clip the actions to the torque limit."""

    tracking_controller_linear_stiffness: float = 100.0
    """Stiffness gain for the PD controller in linear direction."""

    tracking_controller_linear_damping: float = 10.0
    """Damping gain for the PD controller in linear direction."""

    tracking_controller_angular_stiffness: float = 50.0
    """Stiffness gain for the PD controller in angular direction."""

    tracking_controller_angular_damping: float = 0.0
    """Damping gain for the PD controller in angular direction."""

    max_force: float = 100.0
    """Maximum force for the PD controller."""

    max_torque: float = 100.0
    """Maximum torque for the PD controller."""


@configclass
class VirtualRigidObjectControlCfg(ActionTermCfg):
    """Configuration for the virtual rigid object control action term.

    See :class:`VirtualRigidObjectControl` for more details.
    """

    class_type: type[ActionTerm] = VirtualRigidObjectControl

    command_name: str = "dual_hands_object_tracking_command"
    """Name of the command to use for the action."""

    tracking_controller_linear_stiffness: float = 50.0
    """Stiffness gain for the tracking controller in linear direction."""

    tracking_controller_linear_damping: float = 10.0
    """Damping gain for the tracking controller in linear direction."""

    tracking_controller_angular_stiffness: float = 10.0
    """Stiffness gain for the tracking controller in angular direction."""

    tracking_controller_angular_damping: float = 0.1
    """Damping gain for the tracking controller in angular direction."""

    max_force: float = 60.0
    """Maximum force for the tracking controller."""

    max_torque: float = 60.0
    """Maximum torque for the tracking controller."""


@configclass
class VirtualArticulatedObjectControlCfg(ActionTermCfg):
    """Configuration for the virtual articulated object control action term.

    See :class:`VirtualArticulatedObjectControl` for more details.
    """

    class_type: type[ActionTerm] = VirtualArticulatedObjectControl

    command_name: str = "dual_hands_object_tracking_command"
    """Name of the command to use for the action."""

    tracking_controller_linear_stiffness: float = 50.0
    """Stiffness gain for the tracking controller in linear direction."""

    tracking_controller_linear_damping: float = 10.0
    """Damping gain for the tracking controller in linear direction."""

    tracking_controller_angular_stiffness: float = 10.0
    """Stiffness gain for the tracking controller in angular direction."""

    tracking_controller_angular_damping: float = 0.1
    """Damping gain for the tracking controller in angular direction."""

    max_force: float = 60.0
    """Maximum force for the tracking controller."""

    max_torque: float = 60.0
    """Maximum torque for the tracking controller."""
