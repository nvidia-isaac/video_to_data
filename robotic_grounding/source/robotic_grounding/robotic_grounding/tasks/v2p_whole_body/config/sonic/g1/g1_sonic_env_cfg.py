# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import isaaclab.envs.mdp as il_mdp
from isaaclab.envs.mdp import observations
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robotic_grounding.assets import POLICY_ASSET_DIR
from robotic_grounding.assets.g1 import (
    G1_CYLINDER_MODEL_12_HANDS_DEX_DELAYED_CFG,
    G1_DEX_CONTACT_BODIES,
    G1_HAND_JOINT_NAMES,
    G1_MODEL_12_ACTION_SCALE,
)
from robotic_grounding.tasks.v2p.mdp.events import configure_collision_groups
from robotic_grounding.tasks.v2p_whole_body.base_env_cfg import BaseEventsCfg, V2PEnvCfg
from robotic_grounding.tasks.v2p_whole_body.mdp import observations as obs
from robotic_grounding.tasks.v2p_whole_body.mdp.actions import (
    SONICActionCfg,
    SONICActionType,
)
from robotic_grounding.tasks.v2p_whole_body.mdp.curriculum import (
    WholeBodyFixedTimestepVOCCurriculum,
)
from robotic_grounding.tasks.v2p_whole_body.mdp.rewards import (
    contact_rewards,
    tracking_rewards,
)
from robotic_grounding.tasks.v2p_whole_body.mdp.terminations import (
    anchor_pos_error,
    anchor_quat_error,
    ee_position_error,
    ee_quat_error,
    hand_wrist_away_from_trajectory,
    object_pos_error,
    object_quat_error,
    timestep_termination,
)

POLICY_DIR = f"{POLICY_ASSET_DIR}/sonic"

# G1 joints that SONIC controls
G1_SONIC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


# ---------------------------------------------------------------------------
# Observation groups
# ---------------------------------------------------------------------------


@configclass
class G1SONICEncoderCfg(ObsGroup):
    """SONIC tokenizer observations (29 body joints only)."""

    encoder_index = ObsTerm(func=obs.encoder_mode, params={"command_name": "motion"})
    command_joint_pos_multi_future = ObsTerm(
        func=obs.command_joint_pos,
        params={
            "command_name": "motion",
            "sonic_joints_only": True,
            "action_name": "joint_pos",
        },
    )
    command_joint_vel_multi_future = ObsTerm(
        func=obs.command_joint_vel,
        params={
            "command_name": "motion",
            "sonic_joints_only": True,
            "action_name": "joint_pos",
        },
    )
    padding_1 = ObsTerm(func=obs.encoder_padding, params={"dim": 17})
    motion_anchor_ori_b = ObsTerm(
        func=obs.motion_anchor_ori_b, params={"command_name": "motion"}
    )
    padding_2 = ObsTerm(func=obs.encoder_padding, params={"dim": 1762 - 17 - 644})
    concatenate_terms = True


@configclass
class G1SONICDecoderCfg(ObsGroup):
    """SONIC decoder observations (29 body joints only, history_length=10)."""

    base_ang_vel = ObsTerm(
        func=observations.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    joint_pos = ObsTerm(
        func=obs.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sonic_joints_only": True,
            "action_name": "joint_pos",
        },
    )
    joint_vel = ObsTerm(
        func=obs.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sonic_joints_only": True,
            "action_name": "joint_pos",
        },
    )
    actions = ObsTerm(
        func=obs.last_action,
        params={"action_name": "joint_pos", "sonic_joints_only": True},
    )
    gravity_dir = ObsTerm(
        func=observations.projected_gravity,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    concatenate_terms = True
    history_length = 10


@configclass
class G1HandPolicyCfg(ObsGroup):
    """Hand-object transform observations."""

    left_hand_object_transform = ObsTerm(
        func=obs.hand_object_transform,
        params={
            "frame_transform_cfg": SceneEntityCfg("left_hand_object_transform"),
            "threshold": 10.0,
        },
    )
    right_hand_object_transform = ObsTerm(
        func=obs.hand_object_transform,
        params={
            "frame_transform_cfg": SceneEntityCfg("right_hand_object_transform"),
            "threshold": 10.0,
        },
    )


@configclass
class G1PolicyCfg(ObsGroup):
    """Unified policy observations for whole-body tracking.

    Egocentric (body frame) for hand/object state, 6D rotation throughout,
    and legacy absolute anchor positions for checkpoint compatibility.
    """

    wrist_position_b = ObsTerm(
        func=obs.wrist_position_b,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    wrist_orientation_b = ObsTerm(
        func=obs.wrist_orientation_b,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    wrist_velocity_b = ObsTerm(
        func=obs.wrist_velocity_b,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    object_position_b = ObsTerm(
        func=obs.object_position_b,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    object_orientation_b = ObsTerm(
        func=obs.object_orientation_b,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
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
    motion_anchor_pos_b = ObsTerm(
        func=obs.motion_anchor_pos_b,
        params={
            "command_name": "motion",
            "num_future_frames": 3,
            "frame": "absolute",
        },
    )
    motion_anchor_ori_b = ObsTerm(
        func=obs.motion_anchor_ori_b,
        params={"command_name": "motion", "num_future_frames": 3},
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
        func=obs.hand_object_transform_6d,
        params={
            "frame_transform_cfg": SceneEntityCfg("left_hand_object_transform"),
            "threshold": 10.0,
        },
    )
    right_hand_object_transform = ObsTerm(
        func=obs.hand_object_transform_6d,
        params={
            "frame_transform_cfg": SceneEntityCfg("right_hand_object_transform"),
            "threshold": 10.0,
        },
    )
    object_pose_delta = ObsTerm(func=obs.object_pose_delta_6d)
    trajectory_progress = ObsTerm(func=obs.command_trajectory_progress)
    action_history = ObsTerm(func=obs.action_history, params={"command_name": "motion"})
    concatenate_terms = True


@configclass
class G1SonicObservationsCfg:
    """Complete observation config with all groups."""

    policy: G1PolicyCfg = G1PolicyCfg()
    sonic_tokenizer: G1SONICEncoderCfg = G1SONICEncoderCfg()
    sonic_policy: G1SONICDecoderCfg = G1SONICDecoderCfg()
    hand_policy: G1HandPolicyCfg = G1HandPolicyCfg()


# ---------------------------------------------------------------------------
# ReconHand observations
# ---------------------------------------------------------------------------


@configclass
class G1ReconHandPolicyCfg(ObsGroup):
    """Policy (actor) observations for the hand-recon whole-body task.

    Shape: 385 for single-body objects, plus 14 dims per additional object body.
    """

    wrist_velocity_b = ObsTerm(
        func=obs.wrist_velocity_full_b,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
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
    motion_anchor_pos_b = ObsTerm(
        func=obs.motion_anchor_pos_b,
        params={
            "command_name": "motion",
            "num_future_frames": 3,
            "frame": "relative",
        },
    )
    motion_anchor_ori_b = ObsTerm(
        func=obs.motion_anchor_ori_b,
        params={"command_name": "motion", "num_future_frames": 3},
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
        params={
            "side": "left",
            "command_name": "motion",
            "threshold": 10.0,
        },
    )
    right_hand_object_transform = ObsTerm(
        func=obs.hand_object_reference_transform,
        params={
            "side": "right",
            "command_name": "motion",
            "threshold": 10.0,
        },
    )
    object_pose_delta = ObsTerm(
        func=obs.object_pose_delta,
        params={"command_name": "motion"},
    )
    trajectory_progress = ObsTerm(func=obs.command_trajectory_progress)
    base_ang_vel = ObsTerm(
        func=observations.base_ang_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    actions = ObsTerm(
        func=obs.last_action,
        params={"action_name": "joint_pos", "sonic_joints_only": False},
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
    concatenate_terms = True


@configclass
class G1SonicReconHandObservationsCfg:
    """Observation config for the hand-recon whole-body task."""

    policy: G1ReconHandPolicyCfg = G1ReconHandPolicyCfg()
    sonic_tokenizer: G1SONICEncoderCfg = G1SONICEncoderCfg()
    sonic_policy: G1SONICDecoderCfg = G1SONICDecoderCfg()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class G1SonicActionsCfg:
    """JOINT_RESIDUAL: SONIC encodes reference, RL adds residuals after decode."""

    joint_pos = SONICActionCfg(
        action_type=SONICActionType.JOINT_RESIDUAL,
        policy_dir=POLICY_DIR,
        asset_name="robot",
        joint_names=[".*"],
        sonic_joint_names=G1_SONIC_JOINT_NAMES,
        command_name="motion",
        use_default_offset=True,
        residual_scale=0.5,
        use_tanh=False,
        finger_residual=True,
        finger_residual_scale=0.15,
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class G1SonicRewardsCfg:
    """Base reward config — termination penalty and regularization only."""

    termination_penalty = RewTerm(func=il_mdp.is_terminated, weight=-300.0)
    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-1e-6)
    action_l2 = RewTerm(func=il_mdp.action_l2, weight=-1e-6)
    joint_pos_limit = RewTerm(
        func=il_mdp.joint_pos_limits,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class G1SonicTerminationsCfg:
    """Shared termination config."""

    timeout = DoneTerm(
        func=timestep_termination,
        params={"command_name": "motion"},
        time_out=True,
    )
    anchor_pos_error = DoneTerm(
        func=anchor_pos_error,
        params={"command_name": "motion", "threshold": 0.70},
    )
    anchor_quat_error = DoneTerm(
        func=anchor_quat_error,
        params={"command_name": "motion", "threshold": 1.50},
    )
    ee_pos_error = DoneTerm(
        func=ee_position_error,
        params={"command_name": "motion", "threshold": 0.15},
    )
    ee_quat_error = DoneTerm(
        func=ee_quat_error,
        params={"command_name": "motion", "threshold": 1.50},
    )
    object_pos_error = DoneTerm(
        func=object_pos_error,
        params={"command_name": "motion", "threshold": 0.10},
    )
    object_quat_error = DoneTerm(
        func=object_quat_error,
        params={"command_name": "motion", "threshold": 1.50},
    )


# ---------------------------------------------------------------------------
# Base G1 SONIC env
# ---------------------------------------------------------------------------


@configclass
class G1SonicEnvCfg(V2PEnvCfg):
    """Base G1 whole-body env with SONIC JOINT_RESIDUAL action and unified observations."""

    actions: G1SonicActionsCfg = G1SonicActionsCfg()
    observations: G1SonicObservationsCfg = G1SonicObservationsCfg()
    rewards: G1SonicRewardsCfg = G1SonicRewardsCfg()
    terminations: G1SonicTerminationsCfg = G1SonicTerminationsCfg()

    def __post_init__(self) -> None:
        """Configure G1 robot, action scale, and hand sensor bodies."""
        self.scene.robot = G1_CYLINDER_MODEL_12_HANDS_DEX_DELAYED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.actions.joint_pos.scale = G1_MODEL_12_ACTION_SCALE
        self.commands.motion.hand_contact_bodies = list(G1_DEX_CONTACT_BODIES)
        self.commands.motion.hand_frame_target_bodies = [
            "left_hand_palm_link",
            "right_hand_palm_link",
        ]
        super().__post_init__()


# ---------------------------------------------------------------------------
# ReconBody: body-accurate reference (MHR)
# ---------------------------------------------------------------------------


@configclass
class G1SonicReconBodyRewardsCfg(G1SonicRewardsCfg):
    """Rewards for body-accurate references. Emphasizes body/joint/object tracking."""

    motion_anchor_position_error_exp = RewTerm(
        func=tracking_rewards.motion_global_anchor_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_anchor_orientation_error_exp = RewTerm(
        func=tracking_rewards.motion_global_anchor_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_joint_pos_error_exp = RewTerm(
        func=tracking_rewards.motion_joint_pos_error_exp,
        weight=5.0,
        params={
            "command_name": "motion",
            "std": 1.0,
            "joint_names": G1_SONIC_JOINT_NAMES,
        },
    )
    motion_object_position_error_exp = RewTerm(
        func=tracking_rewards.motion_object_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.2},
    )
    motion_progress = RewTerm(
        func=tracking_rewards.motion_progress,
        weight=1.0,
        params={"command_name": "motion"},
    )
    motion_ee_position_error_exp = RewTerm(
        func=tracking_rewards.motion_ee_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.2},
    )
    motion_ee_orientation_error_exp = RewTerm(
        func=tracking_rewards.motion_ee_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    force_closure = RewTerm(
        func=contact_rewards.force_closure_reward,
        weight=5.0,
        params={"command_name": "motion", "min_support": 0.01},
    )
    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-0.0001)


@configclass
class G1SonicReconBodyVOCCurriculumCfg:
    """Fixed-timestep curriculum that decays the VOC target scale over PPO updates.

    The schedule is expressed in PPO update indices and converted to env steps
    via ``num_steps_per_env``. Reward weights are intentionally not scheduled
    here; see ``WholeBodyFixedTimestepVOCCurriculum`` for the rationale.
    """

    voc_curriculum = CurrTerm(
        func=WholeBodyFixedTimestepVOCCurriculum,
        params={
            "command_name": "motion",
            # Match num_steps_per_env in agents/rsl_rl_ppo_cfg.py::G1SonicRslRlPpoCfg.
            "num_steps_per_env": 24,
            # PPO update indices (not seconds, not raw env steps).
            "timestep_schedule": [0, 2000, 4000, 6000, 8000, 10000, 12000],
            # VOC target scale per stage; final stage drives VOC fully off.
            "virtual_object_control_scale_factor": [
                1.0,
                0.75,
                0.5,
                0.25,
                0.1,
                0.05,
                0.0,
            ],
        },
    )


@configclass
class G1SonicReconBodyEnvCfg(G1SonicEnvCfg):
    """Body-accurate reference env (MHR pipeline)."""

    rewards: G1SonicReconBodyRewardsCfg = G1SonicReconBodyRewardsCfg()
    curriculum: G1SonicReconBodyVOCCurriculumCfg = G1SonicReconBodyVOCCurriculumCfg()

    def __post_init__(self) -> None:
        """Set residual scales for body-accurate tracking."""
        super().__post_init__()
        self.actions.joint_pos.residual_scale = 0.15
        self.actions.joint_pos.finger_residual_scale = 0.15


# ---------------------------------------------------------------------------
# ReconHand: hand-accurate reference (planner)
# ---------------------------------------------------------------------------


@configclass
class G1SonicReconHandRewardsCfg(G1SonicRewardsCfg):
    """Rewards for the hand-recon whole-body task."""

    termination_penalty = RewTerm(func=il_mdp.is_terminated, weight=-100.0)
    motion_hand_keypoints_gaussian_exp = RewTerm(
        func=tracking_rewards.motion_hand_keypoints_gaussian_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.1},
    )
    motion_finger_joint_pos_gaussian_exp = RewTerm(
        func=tracking_rewards.motion_finger_joint_pos_gaussian_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_object_keypoints_tracking_exp = RewTerm(
        func=tracking_rewards.motion_object_keypoints_tracking_exp,
        weight=0.0,
        params={"command_name": "motion", "var": 0.1},
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
    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-0.001)
    action_l2 = RewTerm(func=il_mdp.action_l2, weight=-0.0001)
    joint_pos_limit = RewTerm(
        func=il_mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class G1SonicReconHandCurriculumCfg:
    """ReconHand curriculum hooks.

    Defaults to no-op. Experiment configs can enable fixed VOC schedules by
    overriding `timestep_schedule` and `virtual_object_control_scale_factor`.
    """

    voc_curriculum = CurrTerm(
        func=WholeBodyFixedTimestepVOCCurriculum,
        params={
            "command_name": "motion",
            "num_steps_per_env": 24,
            # Single disabled stage (VOC target 0.0). Experiments override these two
            # equal-length lists to enable a PPO-update VOC decay schedule.
            "timestep_schedule": [0],
            "virtual_object_control_scale_factor": [0.0],
        },
    )


@configclass
class G1SonicReconHandTerminationsCfg:
    """Termination config for the hand-recon task.

    Drops EE pos/quat terminations (redundant: the reference EE depends on the
    current object pose, covered by hand_wrist_away_from_trajectory) and adds the
    hand-away termination. Object terminations are effectively disabled.
    """

    timeout = DoneTerm(
        func=timestep_termination,
        time_out=True,
        params={"command_name": "motion"},
    )
    anchor_pos_error = DoneTerm(
        func=anchor_pos_error,
        params={"command_name": "motion", "threshold": 0.70},
    )
    anchor_quat_error = DoneTerm(
        func=anchor_quat_error,
        params={"command_name": "motion", "threshold": 1.50},
    )
    hand_wrist_away = DoneTerm(
        func=hand_wrist_away_from_trajectory,
        params={"command_name": "motion", "threshold": 0.15},
    )
    object_pos_error = DoneTerm(
        func=object_pos_error,
        params={"command_name": "motion", "threshold": 100.0},
    )
    object_quat_error = DoneTerm(
        func=object_quat_error,
        params={"command_name": "motion", "threshold": 100.0},
    )


@configclass
class G1SonicReconHandEventsCfg(BaseEventsCfg):
    """Events for hand-recon scene collision grouping."""

    setup_collision_groups = EventTerm(
        func=configure_collision_groups,
        mode="prestartup",
        params={
            "robot_names": ["Robot"],
            "object_names": [],
            "fixed_object_names": [],
            "disable_robot_to_object_collisions": False,
            "disable_robot_to_fixed_object_collisions": False,
        },
    )


@configclass
class G1SonicReconHandEnvCfg(G1SonicEnvCfg):
    """Hand-accurate reference env (planner pipeline)."""

    # Hand-recon actor observations; overrides the inherited generic obs config.
    events: G1SonicReconHandEventsCfg = G1SonicReconHandEventsCfg()  # type: ignore[assignment]
    observations: G1SonicReconHandObservationsCfg = G1SonicReconHandObservationsCfg()  # type: ignore[assignment]
    rewards: G1SonicReconHandRewardsCfg = G1SonicReconHandRewardsCfg()
    terminations: G1SonicReconHandTerminationsCfg = G1SonicReconHandTerminationsCfg()  # type: ignore[assignment]
    curriculum: G1SonicReconHandCurriculumCfg = G1SonicReconHandCurriculumCfg()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Configure Dex3 hand tracking: EE links, fingertips, VOC, freeze."""
        super().__post_init__()
        self.scene.replicate_physics = False
        self.scene.filter_collisions = False
        self.commands.motion.ee_link_names = [
            "left_hand_palm_link",
            "right_hand_palm_link",
        ]
        self.commands.motion.fingertip_body_name = ".*_(thumb_2|index_1|middle_1)_link"
        self.commands.motion.finger_joint_names = G1_HAND_JOINT_NAMES
        self.commands.motion.reset_freeze_steps = 50
        self.commands.motion.initial_virtual_object_control_curriculum_scale = 1.0
        self.commands.motion.reset_shoulder_spread = 0.5
        self.commands.motion.voc_decay_steps = 10
        self.commands.motion.voc_reset_scale = 1.0
        # Upper bound; apply_scene_config() clips this down to the trajectory length.
        self.episode_length_s = 70.0
