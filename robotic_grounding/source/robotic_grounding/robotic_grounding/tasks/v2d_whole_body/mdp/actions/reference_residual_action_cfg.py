# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config for the reference-anchored residual joint-position action."""

from __future__ import annotations

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .reference_residual_action import ReferenceResidualJointPositionAction


@configclass
class ReferenceResidualJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`ReferenceResidualJointPositionAction`.

    Arm command delay is supplied by ``DelayedPDActuator`` rather than this action term.
    """

    class_type: type[ActionTerm] = ReferenceResidualJointPositionAction

    joint_names: list[str] = [".*"]
    """List of joint names or regex expressions that the action will be mapped to."""

    command_name: str = "motion"
    """Tracking command term supplying the reference ``command_joint_pos``."""

    scale: float | dict[str, float] = 0.15
    """Scale applied to the raw policy output before filtering."""

    clip: float | dict[str, float] = 1.0
    """Symmetric bound on the filtered residual."""

    ema_factor: float = 0.3
    """EMA decay on the residual. Higher keeps more of the previous residual."""

    clip_to_joint_limits: bool = True
    """Clamp the composed target into the asset's joint limits."""
