# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaaclab.utils import configclass

from robotic_grounding.tasks.v2p import mdp
from robotic_grounding.tasks.v2p.v2p_hand_env_cfg import V2PHandEnvCfg

_DEFAULT_MOTION_FILE = "arctic_processed/arctic_s01_box_grab_01/sharpa_wave"


@configclass
class SharpaV2PDirectEnvCfg(V2PHandEnvCfg):
    """Sharpa V2P environment with direct position control (no tracking controller).

    The policy directly outputs PD targets instead of residuals on top
    of a reference-tracking controller. Wrist targets accumulate deltas;
    finger targets are set directly.
    """

    motion_file: str = _DEFAULT_MOTION_FILE

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # Replace residual action terms with direct position action terms.
        # Attribute names are kept the same so observation/reward references still work.
        self.actions.right_joint_residual_action = mdp.JointDirectPositionActionCfg(
            asset_name="right_robot",
            joint_names=[".*"],
            tracking_controller_linear_stiffness=1000.0,
            tracking_controller_linear_damping=100.0,
            tracking_controller_angular_stiffness=40.0,
            tracking_controller_angular_damping=0.01,
            wrist_position_scale=0.05,
            wrist_orientation_scale=0.15,
            finger_joint_scale=0.15,
            finger_joint_clip=100.0,
            ema_factor=0.0,
        )

        self.actions.left_joint_residual_action = mdp.JointDirectPositionActionCfg(
            asset_name="left_robot",
            joint_names=[".*"],
            tracking_controller_linear_stiffness=1000.0,
            tracking_controller_linear_damping=100.0,
            tracking_controller_angular_stiffness=40.0,
            tracking_controller_angular_damping=0.01,
            wrist_position_scale=0.05,
            wrist_orientation_scale=0.15,
            finger_joint_scale=0.15,
            finger_joint_clip=100.0,
            ema_factor=0.0,
        )


@configclass
class SharpaV2PDirectEnvCfgPlay(SharpaV2PDirectEnvCfg):
    """Configuration for playing."""

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()
        self.scene.num_envs = 16
