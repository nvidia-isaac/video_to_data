# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2p.mdp.actions.action_chunks import (
    JointPositionActionChunk,
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
