# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vega Sharpa whole-body env — a direct-joint tracker deriving from V2DEnvCfg.

The Vega Sharpa robot is FIXED-BASE and dual-arm, so this env tracks the reference motion in
joint space with a direct relative-joint-position action. Because the base is bolted down,
world-frame anchor/EE/object tracking carries frame-convention risk and adds nothing for a
free-space motion; the reward is the frame-invariant 58-DOF joint trajectory error, which
is exactly what reproducing the reference pick-and-place motion requires.

The MotionTrackingCommand intentionally carries no object/contact/wrench state and can
sample a bank of reference motions independently for every simulated environment.
"""

from dataclasses import MISSING

import isaaclab.envs.mdp as il_mdp
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robotic_grounding.assets.vega_sharpa import VEGA_SHARPA_SYSID_CFG
from robotic_grounding.tasks.v2d_whole_body.base_env_cfg import (
    BaseCommandsCfg,
    V2DEnvCfg,
)
from robotic_grounding.tasks.v2d_whole_body.mdp import observations as obs
from robotic_grounding.tasks.v2d_whole_body.mdp.commands import (
    MotionTrackingCommandCfg,
)
from robotic_grounding.tasks.v2d_whole_body.mdp.rewards import tracking_rewards
from robotic_grounding.tasks.v2d_whole_body.mdp.terminations import (
    joint_pos_error,
    timestep_termination,
)

# Fixed base link — importing vega_sharpa_reduced.urdf merges the fixed torso/arm_center
# links into `root`, so `root` is the surviving root body. (The reduced_FIXED URDF keeps
# `base` instead; do not swap one without the other.) Anchor tracking is not rewarded
# (the base is fixed), so this only needs to resolve for the command to initialize.
VEGA_ANCHOR_BODY = "root"


# ---------------------------------------------------------------------------
# Actions — direct relative joint position over all 58 DOF (14 arm + 44 finger)
# ---------------------------------------------------------------------------


@configclass
class VegaWholeBodyActionsCfg:
    """Relative joint-position action over the whole body."""

    joint_pos = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.02,
    )


# ---------------------------------------------------------------------------
# Observations — proprioception + joint-tracking command (all frame-invariant)
# ---------------------------------------------------------------------------


@configclass
class VegaPolicyCfg(ObsGroup):
    """Policy observations for whole-body joint tracking."""

    joint_pos_rel = ObsTerm(
        func=obs.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sonic_joints_only": False,
            "action_name": "joint_pos",
        },
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel_rel = ObsTerm(
        func=obs.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sonic_joints_only": False,
            "action_name": "joint_pos",
        },
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    motion_joint_pos_delta = ObsTerm(
        func=obs.motion_joint_pos_delta,
        params={"command_name": "motion", "num_future_frames": 3},
    )
    trajectory_progress = ObsTerm(func=obs.command_trajectory_progress)
    last_action = ObsTerm(
        func=obs.last_action,
        params={"action_name": "joint_pos", "sonic_joints_only": False},
    )
    concatenate_terms = True


@configclass
class VegaObservationsCfg:
    """Observation config — a single policy group (no SONIC encoder/decoder groups)."""

    policy: VegaPolicyCfg = VegaPolicyCfg()


@configclass
class VegaCommandsCfg(BaseCommandsCfg):
    """Reference-only command with optional multi-motion sampling."""

    motion: MotionTrackingCommandCfg = MotionTrackingCommandCfg(
        asset_name="robot",
        motion_file=MISSING,
        anchor_body_name=VEGA_ANCHOR_BODY,
        dt=0.02,
        num_future_frames=10,
        dt_future_frames=0.1,
        debug_vis=True,
    )


# ---------------------------------------------------------------------------
# Rewards — joint-space tracking + regularization
# ---------------------------------------------------------------------------


@configclass
class VegaRewardsCfg:
    """Rewards for whole-body joint tracking."""

    motion_joint_pos_error_exp = RewTerm(
        func=tracking_rewards.motion_joint_pos_error_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 1.0, "joint_names": [".*"]},
    )
    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-1e-4)
    action_l2 = RewTerm(func=il_mdp.action_l2, weight=-1e-4)
    joint_pos_limit = RewTerm(
        func=il_mdp.joint_pos_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    termination_penalty = RewTerm(func=il_mdp.is_terminated, weight=-100.0)


# ---------------------------------------------------------------------------
# Terminations — trajectory end + joint-tracking divergence (frame-invariant)
# ---------------------------------------------------------------------------


@configclass
class VegaTerminationsCfg:
    """Terminations for whole-body joint tracking."""

    timeout = DoneTerm(
        func=timestep_termination,
        params={"command_name": "motion"},
        time_out=True,
    )
    joint_pos_error = DoneTerm(
        func=joint_pos_error,
        params={"command_name": "motion", "threshold": 2.0},
    )


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


@configclass
class VegaSharpaWholeBodyEnvCfg(V2DEnvCfg):
    """Vega Sharpa whole-body joint-tracking env."""

    actions: VegaWholeBodyActionsCfg = VegaWholeBodyActionsCfg()
    commands: VegaCommandsCfg = VegaCommandsCfg()
    observations: VegaObservationsCfg = VegaObservationsCfg()
    rewards: VegaRewardsCfg = VegaRewardsCfg()
    terminations: VegaTerminationsCfg = VegaTerminationsCfg()

    def __post_init__(self) -> None:
        """Assign the Vega robot and configure the tracking command for a fixed base."""
        self.scene.robot = VEGA_SHARPA_SYSID_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.commands.motion.anchor_body_name = VEGA_ANCHOR_BODY
        super().__post_init__()
        # Upper bound; apply_scene_config() clips this down to the trajectory length so a
        # full episode can replay the whole motion (with a random start frame on reset).
        self.episode_length_s = 20.0
