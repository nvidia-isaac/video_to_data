# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vega Sharpa whole-body MANIPULATION env — object-tracking + force-closure grasping.

Unlike the joint-space-only ``VegaSharpaWholeBodyEnvCfg``, which tracks robot references
with no object, this env spawns the manipulated object and rewards a force-closure grasp
that tracks the reference object trajectory. It reproduces natively in IsaacLab/PhysX what
the external policy that generated those references trains under a different physics
backend, so the two can be compared without a sim2sim gap in the GR00T finetune data.

The recipe: residual joint-position action (scale 0.15, EMA 0.3), 20 Hz control / 100 Hz
physics, gravity compensated on the robot (gravity off on robot links, on for the object),
object-keypoint + hand-keypoint + force-closure rewards (contact-wrench / missed /
unintended contact terms zeroed), a linear VOC decay curriculum, and NO domain
randomization.

The object, per-side contact sensors, VOC action, and hand-object FrameTransformers are all
auto-wired by ``apply_scene_config``'s whole-body branch once the motion file resolves; this
cfg only supplies the command's body/joint names and the physics/timing settings.
"""

# TODO: Consider to merge this EnvCfg with the one in v2d_hand_env_cfg.py

from __future__ import annotations

import isaaclab.envs.mdp as il_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robotic_grounding.assets.vega_sharpa import (
    HAND_CONTACT_BODIES as VEGA_HAND_CONTACT_BODIES,
)
from robotic_grounding.assets.vega_sharpa import (
    VEGA_ARM_DEFAULT_POSE,
    VEGA_ARM_JOINT_ORDER,
    VEGA_FINGER_JOINT_ORDER,
    VEGA_SHARPA_SYSID_CFG,
    VEGA_SYSID_FINGER_ACTUATOR_CFG,
)
from robotic_grounding.tasks.v2d.mdp.events import configure_collision_groups
from robotic_grounding.tasks.v2d_whole_body.base_env_cfg import BaseEventsCfg, V2DEnvCfg
from robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.vega_sharpa_env_cfg import (
    VEGA_ANCHOR_BODY,
)
from robotic_grounding.tasks.v2d_whole_body.mdp import observations as obs
from robotic_grounding.tasks.v2d_whole_body.mdp.actions.reference_residual_action_cfg import (
    ReferenceResidualJointPositionActionCfg,
)
from robotic_grounding.tasks.v2d_whole_body.mdp.curriculum import (
    FixedTimestepCurriculum,
)
from robotic_grounding.tasks.v2d_whole_body.mdp.events import (
    reset_robot_to_trajectory_start,
)
from robotic_grounding.tasks.v2d_whole_body.mdp.rewards import (
    contact_rewards,
    tracking_rewards,
)
from robotic_grounding.tasks.v2d_whole_body.mdp.terminations import (
    hand_wrist_away_from_trajectory,
    object_pos_error,
    object_quat_error,
    robot_state_diverged,
    timestep_termination,
)

# Vega wrist bodies: the URDF importer merges the fixed *_hand_C_MC palm links into the arm
# flange links, so the wrist/palm frame is addressed via L_arm_l7 / R_arm_l7.
VEGA_WRIST_BODIES = ("L_arm_l7", "R_arm_l7")

# ---------------------------------------------------------------------------
# Actions — residual joint position over all 58 DOF
# ---------------------------------------------------------------------------


@configclass
class VegaManipActionsCfg:
    """Joint-position residual around the motion reference."""

    joint_pos = ReferenceResidualJointPositionActionCfg(
        asset_name="robot",
        command_name="motion",
        joint_names=[".*"],
        scale=0.15,
        clip=1.0,
        ema_factor=0.3,
        clip_to_joint_limits=True,
    )


# ---------------------------------------------------------------------------
# Observations — proprioception + object/hand/command state (no SONIC groups)
# ---------------------------------------------------------------------------


@configclass
class VegaManipPolicyCfg(ObsGroup):
    """Single policy group: joint state + reference deltas + object/wrist state."""

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
    motion_ee_pos_delta = ObsTerm(
        func=obs.motion_ee_pos_delta,
        params={"command_name": "motion", "num_future_frames": 3},
    )
    motion_ee_quat_delta = ObsTerm(
        func=obs.motion_ee_quat_delta,
        params={"command_name": "motion", "num_future_frames": 3},
    )
    left_hand_object_transform = ObsTerm(
        func=obs.hand_object_reference_transform,
        params={"side": "left", "command_name": "motion", "threshold": 10.0},
    )
    right_hand_object_transform = ObsTerm(
        func=obs.hand_object_reference_transform,
        params={"side": "right", "command_name": "motion", "threshold": 10.0},
    )
    object_pose_delta = ObsTerm(
        func=obs.object_pose_delta,
        params={"command_name": "motion"},
    )
    wrist_position_e = ObsTerm(
        func=obs.wrist_position_e,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    wrist_wxyz_e = ObsTerm(
        func=obs.wrist_wxyz_e,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    object_position_e = ObsTerm(
        func=obs.object_position_e,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    object_wxyz_e = ObsTerm(
        func=obs.object_wxyz_e,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    trajectory_progress = ObsTerm(func=obs.command_trajectory_progress)
    last_action = ObsTerm(
        func=obs.last_action,
        params={"action_name": "joint_pos", "sonic_joints_only": False},
    )
    concatenate_terms = True


@configclass
class VegaManipObservationsCfg:
    """Observation config — a single policy group (no SONIC encoder/decoder groups)."""

    policy: VegaManipPolicyCfg = VegaManipPolicyCfg()


# ---------------------------------------------------------------------------
# Rewards — force-closure tracking recipe
# ---------------------------------------------------------------------------


@configclass
class VegaManipRewardsCfg:
    """Object-keypoint + hand-keypoint tracking + force-closure grasp; contacts zeroed."""

    termination_penalty = RewTerm(func=il_mdp.is_terminated, weight=-100.0)
    motion_object_keypoints_tracking_exp = RewTerm(
        func=tracking_rewards.motion_object_keypoints_tracking_exp,
        weight=2.0,
        params={"command_name": "motion", "var": 0.1},
    )
    motion_hand_keypoints_gaussian_exp = RewTerm(
        func=tracking_rewards.motion_hand_keypoints_gaussian_exp,
        # Six keypoints per hand: std ~= sqrt(6 * 0.1); half-weight averages both hands.
        weight=0.5,
        params={"command_name": "motion", "std": 0.775},
    )
    force_closure = RewTerm(
        func=contact_rewards.force_closure_reward,
        weight=5.0,
        params={"command_name": "motion", "min_support": 0.01},
    )
    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-5e-4)
    action_l2 = RewTerm(func=il_mdp.action_l2, weight=-5e-3)
    # Noisy monocular contacts: keep the wrench/contact terms present but off.
    motion_finger_joint_pos_gaussian_exp = RewTerm(
        func=tracking_rewards.motion_finger_joint_pos_gaussian_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_contact_tracking_gaussian_exp = RewTerm(
        func=tracking_rewards.motion_contact_tracking_gaussian_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 0.05},
    )
    contact_wrench_support_reward = RewTerm(
        func=contact_rewards.contact_wrench_support_reward,
        weight=0.0,
        params={"command_name": "motion", "tolerance": 0.1, "var": 0.1},
    )
    unintended_contact_penalty = RewTerm(
        func=contact_rewards.unintended_contact_penalty,
        weight=0.0,
        params={"command_name": "motion"},
    )
    missed_contact_penalty = RewTerm(
        func=contact_rewards.missed_contact_penalty,
        weight=0.0,
        params={"command_name": "motion"},
    )


# ---------------------------------------------------------------------------
# Terminations — timeout + wrist/object divergence (fixed base: no anchor/EE terms)
# ---------------------------------------------------------------------------


@configclass
class VegaManipTerminationsCfg:
    """Horizon + hand/object trajectory-divergence terminations."""

    timeout = DoneTerm(
        func=timestep_termination,
        time_out=True,
        params={"command_name": "motion"},
    )
    hand_wrist_away = DoneTerm(
        func=hand_wrist_away_from_trajectory,
        params={"command_name": "motion", "threshold": 0.15},
    )
    object_pos_error = DoneTerm(
        func=object_pos_error,
        params={"command_name": "motion", "threshold": 0.10},
    )
    object_quat_error = DoneTerm(
        func=object_quat_error,
        params={"command_name": "motion", "threshold": 0.50},
    )
    # Safety guard (mirrors the floating-hand env's robot_state_diverged): reset any env
    # whose joint/root state goes non-finite or implausibly fast. Without it a single
    # diverged env poisons the batch and the PPO update dies with
    # "normal expects all elements of std >= 0.0" -- which killed the baseline AND fixed
    # runs at ~5.6-6.3k of 20k iterations. NOT freeze-masked: divergence must terminate
    # even during the reset freeze.
    robot_state_diverged = DoneTerm(
        func=robot_state_diverged,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_joint_vel": 100.0},
    )


# ---------------------------------------------------------------------------
# Curriculum — linear VOC decay 1.0 -> 0.0 over PPO updates
# ---------------------------------------------------------------------------


@configclass
class VegaManipCurriculumCfg:
    """Fixed-timestep VOC decay (PPO-update indices; num_steps_per_env must match the PPO cfg)."""

    # LINEAR 11-stage decay, matching the reference policy's successful force-closure run
    # (ckpt/xyjljbk2, tissue-box-simple-fc-ppo, success=true) -- the "linear" in that
    # (a linear VOC schedule). The reference stores stage END thresholds
    # [1250, 2500, ... 12500], and this curriculum wants exactly that: it selects with
    # ``bisect_right(thresholds, step)``, so 10 thresholds + 11 scales puts scales[0]=1.0
    # in force before the first threshold (the ``len+1`` branch of the length assert in
    # FixedTimestepCurriculum). A leading 0 instead makes bisect_right return 1 at step 0,
    # so scales[0] is unreachable and the run silently starts at 0.9 with the whole ladder
    # shifted one stage early.
    # The previous [1.0, 0.75, 0.5, 0.25, ...] cliff schedule was inherited from the
    # non-force-closure recipe, which pairs it with a very different reward set.
    voc_curriculum = CurrTerm(
        func=FixedTimestepCurriculum,
        params={
            "command_name": "motion",
            "num_steps_per_env": 24,
            "timestep_schedule": [
                1250,
                2500,
                3750,
                5000,
                6250,
                7500,
                8750,
                10000,
                11250,
                12500,
            ],
            "virtual_object_control_scale_factor": [
                1.0,
                0.9,
                0.8,
                0.7,
                0.6,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0.0,
            ],
        },
    )


# ---------------------------------------------------------------------------
# Events — collision grouping only (NO domain randomization)
# ---------------------------------------------------------------------------


@configclass
class VegaManipEventsCfg(BaseEventsCfg):
    """Reset-to-frame (inherited) + collision grouping.

    The robot collides with the object but not with the fixed support surface
    (the reference policy trains with robot/support collision disabled).
    """

    setup_collision_groups = EventTerm(
        func=configure_collision_groups,
        mode="prestartup",
        params={
            "robot_names": ["Robot"],
            "object_names": [],
            "fixed_object_names": [],
            "disable_robot_to_object_collisions": False,
            "disable_robot_to_fixed_object_collisions": True,
        },
    )

    # Grasp friction, set on the HAND only and left at the PhysX default (0.5) on the
    # object -- the same convention as the floating-hand env (`v2d_hand_env_cfg.py`), which
    # raises its hands to 2.0 and never touches the object. Effective pair coefficient is
    # then PhysX's default `average` combine of 2.0 and 0.5, i.e. 1.25.
    #
    # This must be an event, not spawn config: `physics_material` is not a field on
    # `UrdfFileCfg`/`UsdFileCfg` (only on `GroundPlaneCfg`/`ShapeCfg`), so assigning it
    # there silently creates a stray attribute the spawner never reads. The event writes
    # through the PhysX view, which also sidesteps the object collider being an instance
    # proxy (`isaaclab.sim.utils.apply_nested` skips instanced prims). Degenerate ranges
    # are how the floating-hand env pins a constant through this randomization API.
    hand_physics_material = EventTerm(
        func=il_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=VEGA_HAND_CONTACT_BODIES),
            "static_friction_range": (2.0, 2.01),
            "dynamic_friction_range": (2.0, 2.01),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Collection tools may opt into reset perturbations by changing these ranges on a
    # task-local cfg before environment construction. Zero ranges are exact no-ops.
    reset_to_trajectory_frame = EventTerm(
        func=reset_robot_to_trajectory_start,
        params={
            "command_name": "motion",
            "trajectory_time_index": (0, 999999),
            "joint_position_noise_groups": {
                "arm": {
                    "joint_names": list(VEGA_ARM_JOINT_ORDER),
                    "range": (0.0, 0.0),
                },
                "finger": {
                    "joint_names": list(VEGA_FINGER_JOINT_ORDER),
                    "range": (0.0, 0.0),
                },
            },
            "object_xy_noise_range": (0.0, 0.0),
            "object_yaw_noise_range": (0.0, 0.0),
        },
        mode="reset",
    )


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


@configclass
class VegaSharpaManipEnvCfg(V2DEnvCfg):
    """Vega Sharpa whole-body manipulation env (object tracking + force-closure grasp)."""

    actions: VegaManipActionsCfg = VegaManipActionsCfg()
    observations: VegaManipObservationsCfg = VegaManipObservationsCfg()
    rewards: VegaManipRewardsCfg = VegaManipRewardsCfg()
    terminations: VegaManipTerminationsCfg = VegaManipTerminationsCfg()
    curriculum: VegaManipCurriculumCfg = VegaManipCurriculumCfg()
    events: VegaManipEventsCfg = VegaManipEventsCfg()

    # No object_contact_friction / object_contact_offset here on purpose: the object stays
    # at the PhysX defaults and the grasp friction is raised on the hand instead, via the
    # `hand_physics_material` event. See that term for why the spawn-cfg route cannot work.

    # Clamp the manipulated object to solver 8/0 (matching the 8/0 robot). PhysX steps the
    # whole GPU scene at the MAX iteration count over all bodies, so an object left at the
    # 16/1 default silently drags the tuned 8/0 robot back up to 16/1 (floating-hand env
    # measured +31% throughput from this clamp). Read by apply_scene_objects.
    object_solver_position_iteration_count: int = 8
    object_solver_velocity_iteration_count: int = 0

    def __post_init__(self) -> None:
        """Vega robot + object-tracking command + physics/timing settings."""
        # Use the REDUCED URDF (vega_sharpa_reduced.urdf) — its torso/head are baked at the
        # retargeting pose (torso_j1 pitch -0.78 rad), matching the IK reference world in which
        # ee_pose_w / hand keypoints / object poses were computed. The reduced_FIXED URDF zeroes
        # that torso pitch, so the same joint angles offset the hands ~45 deg from the reference —
        # the fundamental inconsistency. Init the arms at the RL-training standing pose so the
        # frozen non-arm/finger joints and the pre-reset default posture are reference-consistent.
        self.scene.robot = VEGA_SHARPA_SYSID_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                joint_pos=VEGA_ARM_DEFAULT_POSE,
                joint_vel={".*": 0.0},
            ),
            # Sysid DelayedPD arms (soft per-joint kp/kd, measured armature, 4-physics-step
            # command delay) + the official sysid per-group finger calibration. The default
            # VEGA_SHARPA_CFG arm is an ImplicitActuator at kp 1000/1000/500x5 with armature
            # 0.001 and NO delay -- a near-rigid zero-lag servo that transmits every bit of
            # residual action straight to the wrist instead of mechanically low-passing it.
            actuators={
                "arm": VEGA_SHARPA_SYSID_CFG.actuators["arm"],
                "fingers": VEGA_SYSID_FINGER_ACTUATOR_CFG,
            },
        )
        # The reduced URDF importer merges base -> root (unlike reduced_fixed which keeps
        # `base`), so the fixed-base anchor body is `root` here. Shared with
        # VegaSharpaWholeBodyEnvCfg so the two cannot drift apart from the URDF.
        self.commands.motion.anchor_body_name = VEGA_ANCHOR_BODY
        super().__post_init__()

        # Timing: 20 Hz control / 100 Hz physics. The DelayedPD arm delay of 4
        # physics steps equals the sysid'd 40 ms only at 100 Hz, so dt=0.01 is required.
        self.decimation = 5
        self.sim.dt = 0.01
        self.sim.render_interval = 5
        self.scene.replicate_physics = False
        self.scene.filter_collisions = False

        # PhysX GPU buffer caps sized for 4096 envs of a 58-DOF robot + contact-rich grasp
        # (matches the floating-hand recipe). Avoids rigid-contact/patch buffer overflow
        # stalls at scale. Pure allocation caps — no contact-dynamics change (compliant
        # contact / bounce threshold deliberately NOT ported, to keep grasp physics fixed).
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23

        # Gravity OFF on robot links (matches the gravity-compensated real arm);
        # the sysid PD gains are not sized to hold a raised pose against gravity. Global
        # gravity stays on so the object still rests/lifts.
        spawn = self.scene.robot.spawn
        if getattr(spawn, "rigid_props", None) is not None:
            self.scene.robot = self.scene.robot.replace(
                spawn=spawn.replace(
                    rigid_props=spawn.rigid_props.replace(disable_gravity=True)
                )
            )

        # Object-tracking command (object/contact/VOC auto-wired by apply_scene_config).
        motion = self.commands.motion
        # One reference frame is consumed per 20 Hz environment step. Source
        # trajectories (including the 50 Hz tissue-box sequence) are resampled
        # to this playback rate, adjusted by motion.motion_speed.
        motion.dt = 0.05  # = decimation * sim.dt
        # The Vega manipulation policy contract uses half-speed reference playback.
        motion.motion_speed = 0.5
        motion.ee_link_names = list(VEGA_WRIST_BODIES)
        motion.left_wrist_body_name = VEGA_WRIST_BODIES[0]
        motion.right_wrist_body_name = VEGA_WRIST_BODIES[1]
        motion.fingertip_body_name = ".*_DP"
        motion.finger_joint_names = list(VEGA_FINGER_JOINT_ORDER)
        motion.hand_contact_bodies = list(VEGA_HAND_CONTACT_BODIES)
        motion.hand_frame_target_bodies = list(VEGA_WRIST_BODIES)
        motion.reset_freeze_steps = 50
        # Fully random start frames (no forced frame-0 resets). The frame-0 collapse that
        # originally motivated mixing them in was caused by the self-integrating action --
        # a zero action froze the robot while the reference walked away -- and that is now
        # fixed by the reference-anchored action term, under which a reset lands the policy
        # on-trajectory from any start frame. 0.0 is the A/B arm against the runs using 0.5.
        motion.reset_to_first_frame_prob = 0.0
        motion.initial_virtual_object_control_curriculum_scale = 1.0
        motion.reset_shoulder_spread = 0.0  # vega has no shoulder-yaw joints
        # Start each episode from a random partial finger aperture so the grasp is closed
        # under physics rather than teleported onto the reference (the reference run used
        # reset_finger_openness 0.7).
        motion.reset_finger_openness = 0.7
        motion.voc_decay_steps = 10
        motion.voc_reset_scale = 1.0

        # Upper bound; apply_scene_config() clips this to the trajectory length + freeze.
        self.episode_length_s = 20.0
