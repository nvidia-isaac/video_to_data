# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415 — lazy imports in __post_init__ for circular dep avoidance
from dataclasses import MISSING
from enum import Enum

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass


class SONICActionType(Enum):
    """Types of SONIC action terms."""

    HIERARCHICAL = "hierarchical"
    HIERARCHICAL_RESIDUAL = "hierarchical_residual"  # residuals BEFORE SONIC
    JOINT_RESIDUAL = "joint_residual"  # residuals AFTER SONIC
    LATENT_RESIDUAL = "latent_residual"
    LATENT = "latent"
    LATENT_HAND_POLICY = "latent_hand_policy"


@configclass
class SONICActionCfg(ActionTermCfg):
    """Common configuration for all SONIC action terms."""

    action_type: SONICActionType = SONICActionType.HIERARCHICAL

    policy_dir: str = MISSING  # type: ignore[assignment]
    """Path to directory containing SONIC ONNX models."""

    asset_name: str = "robot"
    joint_names: list[str] = [".*"]
    sonic_joint_names: list[str] = MISSING  # type: ignore[assignment]
    command_name: str = "motion"
    use_default_offset: bool = True
    scale: float | dict[str, float] = 1.0

    # Hand policy (for LATENT_HAND_POLICY)
    hand_policy_class: type | None = None
    hand_policy_cfg: object = None

    # Joint residual params (for JOINT_RESIDUAL)
    residual_scale: float = 0.1
    """Scale applied to RL residuals before adding to SONIC output."""

    residual_joint_names: list[str] | None = None
    """If set, RL only outputs residuals for these joints (subset of sonic joints)."""

    finger_residual: bool = False
    """If True, RL also outputs residuals for non-SONIC (finger) joints."""

    finger_residual_scale: float = -1.0
    """Scale for finger residuals. -1.0 means use residual_scale."""

    use_tanh: bool = True
    """Whether to apply tanh squashing to residuals."""

    debug: bool = False

    def __post_init__(self) -> None:
        """Dispatch class_type from action_type enum."""
        from .sonic_hierarchical_action import SONICHierachicalAction
        from .sonic_hierarchical_residual_action import (
            SONICHierarchicalResidualAction,
        )
        from .sonic_joint_residual_action import SONICJointResidualAction
        from .sonic_latent_action import SONICLatentAction
        from .sonic_latent_hand_policy_action import SONICLatentHandPolicyAction
        from .sonic_latent_residual_action import SONICLatentResidualAction

        _dispatch = {
            SONICActionType.HIERARCHICAL: SONICHierachicalAction,
            SONICActionType.HIERARCHICAL_RESIDUAL: SONICHierarchicalResidualAction,
            SONICActionType.JOINT_RESIDUAL: SONICJointResidualAction,
            SONICActionType.LATENT_RESIDUAL: SONICLatentResidualAction,
            SONICActionType.LATENT: SONICLatentAction,
            SONICActionType.LATENT_HAND_POLICY: SONICLatentHandPolicyAction,
        }

        if self.action_type not in _dispatch:
            raise ValueError(f"Unknown SONIC action type: {self.action_type}")
        self.class_type = _dispatch[self.action_type]

        super().__post_init__()
