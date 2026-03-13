# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaaclab.utils import configclass

from robotic_grounding.assets.rigid_object import RIGID_OBJECT_NO_COLLISION_CFG
from robotic_grounding.tasks.v2p import mdp
from robotic_grounding.tasks.v2p.config.sharpa_wave.sharpa_v2p_env_cfg import (
    SharpaV2PEnvCfg,
)


@configclass
class SharpaV2PDirectEnvCfg(SharpaV2PEnvCfg):
    """Sharpa V2P environment with direct position control (no tracking controller).

    The policy directly outputs PD targets instead of residuals on top
    of a reference-tracking controller. Wrist targets accumulate deltas;
    finger targets are set directly.
    """

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # Use no-collision object for testing
        self.scene.object = RIGID_OBJECT_NO_COLLISION_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Object"
        )

        # Replace residual action terms with direct position action terms.
        # Attribute names are kept the same so observation/reward references still work.
        self.actions.right_joint_residual_action = mdp.JointDirectPositionActionCfg(
            asset_name="right_robot",
            joint_names=[".*"],
            tracking_controller_linear_stiffness=50.0,
            tracking_controller_linear_damping=10.0,
            tracking_controller_angular_stiffness=12.0,
            tracking_controller_angular_damping=0.5,
            wrist_position_scale=0.05,
            wrist_orientation_scale=0.15,
            finger_joint_scale=0.15,
            ema_factor=0.9,
        )

        self.actions.left_joint_residual_action = mdp.JointDirectPositionActionCfg(
            asset_name="left_robot",
            joint_names=[".*"],
            tracking_controller_linear_stiffness=50.0,
            tracking_controller_linear_damping=10.0,
            tracking_controller_angular_stiffness=12.0,
            tracking_controller_angular_damping=0.5,
            wrist_position_scale=0.05,
            wrist_orientation_scale=0.15,
            finger_joint_scale=0.15,
            ema_factor=0.9,
        )


@configclass
class SharpaV2PDirectEnvCfgPlay(SharpaV2PDirectEnvCfg):
    """Configuration for playing."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        self.scene.num_envs = 16
