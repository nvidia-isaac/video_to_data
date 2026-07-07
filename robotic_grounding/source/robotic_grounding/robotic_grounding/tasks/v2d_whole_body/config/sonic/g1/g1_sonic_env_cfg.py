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
from robotic_grounding.tasks.v2d.mdp.events import configure_collision_groups
from robotic_grounding.tasks.v2d_whole_body.base_env_cfg import BaseEventsCfg, V2DEnvCfg
from robotic_grounding.tasks.v2d_whole_body.mdp import observations as obs
from robotic_grounding.tasks.v2d_whole_body.mdp.actions import (
    SONICActionCfg,
    SONICActionType,
)
from robotic_grounding.tasks.v2d_whole_body.mdp.curriculum import (
    FixedTimestepCurriculum,
)
from robotic_grounding.tasks.v2d_whole_body.mdp.rewards import (
    contact_rewards,
    tracking_rewards,
)
from robotic_grounding.tasks.v2d_whole_body.mdp.terminations import (
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

# Joints the ReconHand policy adds residuals to: the SONIC joints minus the 12
# leg joints, so the legs stay on the SONIC prior and residuals shape the torso
# and arms.
RECON_HAND_RESIDUAL_JOINT_NAMES = G1_SONIC_JOINT_NAMES[12:]


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
class G1SonicEnvCfg(V2DEnvCfg):
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
    """Fixed-timestep curriculum that decays the VOC target scale.

    The scale list is one entry longer than ``timestep_schedule``: ``scale[0]``
    applies from step 0 and each threshold advances to the next scale.
    """

    voc_curriculum = CurrTerm(
        func=FixedTimestepCurriculum,
        params={
            "command_name": "motion",
            # Must match the PPO runner's num_steps_per_env.
            "num_steps_per_env": 24,
            "timestep_schedule": [2000, 4000, 6000, 8000, 10000, 12000],
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
    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-0.01)
    action_l2 = RewTerm(func=il_mdp.action_l2, weight=-0.001)
    joint_pos_limit = RewTerm(
        func=il_mdp.joint_pos_limits,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class G1SonicReconHandCurriculumCfg:
    """ReconHand curriculum hooks (no-op by default).

    Enable a fixed VOC schedule by overriding ``timestep_schedule`` and
    ``virtual_object_control_scale_factor``.
    """

    fixed_timestep_curriculum = CurrTerm(
        func=FixedTimestepCurriculum,
        params={
            "command_name": "motion",
            "num_steps_per_env": 24,
            # Empty schedule disables the curriculum (no-op); the VOC scale stays
            # at the command's initial value. Override to enable a schedule.
            "timestep_schedule": [],
            "virtual_object_control_scale_factor": [],
            "reward_weight_schedules": {
                # Pre-declared so these keys can be overridden under Hydra strict mode
                # without a `+` prefix. Override with a schedule matching the VOC scale length.
                "motion_object_keypoints_tracking_exp": [],
                "motion_hand_keypoints_gaussian_exp": [],
                "motion_finger_joint_pos_gaussian_exp": [],
                "motion_contact_tracking_gaussian_exp": [],
                "contact_wrench_support_reward": [],
                "unintended_contact_penalty": [],
                "missed_contact_penalty": [],
            },
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
        self.actions.joint_pos.residual_joint_names = list(
            RECON_HAND_RESIDUAL_JOINT_NAMES
        )
        self.commands.motion.reset_freeze_steps = 50
        self.commands.motion.initial_virtual_object_control_curriculum_scale = 1.0
        self.commands.motion.reset_shoulder_spread = 0.5
        self.commands.motion.voc_decay_steps = 10
        self.commands.motion.voc_reset_scale = 1.0
        # Upper bound; apply_scene_config() clips this down to the trajectory length.
        self.episode_length_s = 70.0


@configclass
class G1SonicReconHandEpisodeTimeoutTerminationsCfg(G1SonicReconHandTerminationsCfg):
    """ReconHand terminations using a fixed episode-length (wall-clock) timeout.

    Replaces the base trajectory-end ``timestep_termination`` with
    ``il_mdp.time_out`` driven by ``episode_length_s``.
    """

    timeout = DoneTerm(func=il_mdp.time_out, time_out=True)


@configclass
class G1SonicReconHandEpisodeTimeoutEnvCfg(G1SonicReconHandEnvCfg):
    """ReconHand variant that uses the episode-length (wall-clock) timeout."""

    terminations: G1SonicReconHandEpisodeTimeoutTerminationsCfg = (
        G1SonicReconHandEpisodeTimeoutTerminationsCfg()
    )  # type: ignore[assignment]


def _apply_contact_rewards(cfg: G1SonicReconHandEnvCfg) -> None:
    """Contact-grounding reward mix + object terminations (stages 2 and 3)."""
    cfg.rewards.motion_hand_keypoints_gaussian_exp.weight = 0.10
    cfg.rewards.motion_finger_joint_pos_gaussian_exp.weight = 0.10
    cfg.rewards.motion_contact_tracking_gaussian_exp.weight = 0.10
    cfg.rewards.contact_wrench_support_reward.weight = 5.0
    cfg.rewards.unintended_contact_penalty.weight = -2.5
    cfg.rewards.missed_contact_penalty.weight = -5.0
    cfg.terminations.object_pos_error.params["threshold"] = 0.15
    cfg.terminations.object_quat_error.params["threshold"] = 1.0


@configclass
class G1SonicReconHandStage1EnvCfg(G1SonicReconHandEpisodeTimeoutEnvCfg):
    """Stage 1 — no-collision warm-up.

    Robot↔object collisions off (support surfaces stay solid), VOC always on,
    only hand-keypoint and finger-joint tracking rewards. Train with --zero-actor.
    """

    def __post_init__(self) -> None:
        """Apply the no-collision warm-up deltas."""
        super().__post_init__()
        self.episode_length_s = 10.0
        self.commands.motion.voc_decay_steps = 0
        self.events.setup_collision_groups.params[
            "disable_robot_to_object_collisions"
        ] = True


@configclass
class G1SonicReconHandStage2EnvCfg(G1SonicReconHandStage1EnvCfg):
    """Stage 2 — contact grounding (resume from stage 1).

    Robot↔object collisions on, VOC decays 0.5→0 over a fixed-timestep
    curriculum, and contact wrench-support + unintended/missed-contact rewards
    are enabled; object-keypoint weight ramps in over the curriculum.
    """

    def __post_init__(self) -> None:
        """Apply the contact-grounding stage deltas."""
        super().__post_init__()
        self.events.setup_collision_groups.params[
            "disable_robot_to_object_collisions"
        ] = False
        self.commands.motion.initial_virtual_object_control_curriculum_scale = 0.5
        self.commands.motion.voc_decay_steps = 10
        _apply_contact_rewards(self)
        sched = self.curriculum.fixed_timestep_curriculum.params
        sched["timestep_schedule"] = [5000, 10000, 15000, 20000]
        sched["virtual_object_control_scale_factor"] = [0.5, 0.1, 0.02, 0.004, 0.0]
        rw = sched["reward_weight_schedules"]
        rw["motion_object_keypoints_tracking_exp"] = [0.0, 0.4, 0.48, 0.496, 0.5]
        rw["motion_hand_keypoints_gaussian_exp"] = [0.10] * 5
        rw["motion_finger_joint_pos_gaussian_exp"] = [0.10] * 5
        rw["motion_contact_tracking_gaussian_exp"] = [0.10] * 5
        rw["contact_wrench_support_reward"] = [5.0] * 5
        rw["unintended_contact_penalty"] = [-2.5] * 5
        rw["missed_contact_penalty"] = [-5.0] * 5


@configclass
class G1SonicReconHandStage3EnvCfg(G1SonicReconHandEnvCfg):
    """Stage 3 — full-sequence finetune (resume from stage 2).

    Full-length episodes (trajectory-end timeout, so it extends the base
    ReconHand env rather than the fixed episode-length variant), VOC off,
    curriculum disabled, object-keypoint tracking weighted 10x.
    """

    def __post_init__(self) -> None:
        """Apply the full-sequence finetune deltas."""
        super().__post_init__()
        _apply_contact_rewards(self)
        self.commands.motion.always_reset_to_first_frame = True
        self.commands.motion.initial_virtual_object_control_curriculum_scale = 0.0
        self.rewards.motion_object_keypoints_tracking_exp.weight = 5.0
