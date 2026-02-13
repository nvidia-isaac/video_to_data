"""Base V2P EE-focused environment configuration."""

import isaaclab.envs.mdp as il_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.manipulation.lift.mdp.observations import (
    object_position_in_robot_root_frame,
)

from robotic_grounding.tasks.v2p_whole_body.mdp import observations as obs
from robotic_grounding.tasks.v2p_whole_body.mdp.rewards import tracking_rewards
from robotic_grounding.tasks.v2p_whole_body.mdp.terminations import (
    anchor_pos_error,
    anchor_quat_error,
    ee_position_error,
    ee_quat_error,
)

from .base_env_cfg import (
    BaseObservationsCfg,
    BaseRewardsCfg,
    BaseTerminationsCfg,
    V2PEnvCfg,
)


@configclass
class EEObservationsCfg(BaseObservationsCfg):
    """Observation configuration using EE positions instead of joint positions."""

    @configclass
    class EEPolicyCfg(ObsGroup):
        """Policy observations with EE tracking instead of joint tracking."""

        motion_anchor_pos_b = ObsTerm(
            func=obs.motion_anchor_pos_b,
            params={"command_name": "motion", "num_future_frames": 10},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        motion_anchor_ori_b = ObsTerm(
            func=obs.motion_anchor_ori_b,
            params={"command_name": "motion", "num_future_frames": 10},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        motion_ee_pos_delta = ObsTerm(
            func=obs.motion_ee_pos_delta,
            params={"command_name": "motion", "num_future_frames": 10},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        motion_ee_quat_delta = ObsTerm(
            func=obs.motion_ee_quat_delta,
            params={"command_name": "motion", "num_future_frames": 10},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        object_pos_b = ObsTerm(
            func=object_position_in_robot_root_frame,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        object_pos_delta = ObsTerm(
            func=obs.object_pos_delta, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        trajectory_progress = ObsTerm(func=obs.command_trajectory_progress)
        base_lin_vel = ObsTerm(func=il_mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=il_mdp.base_ang_vel)
        joint_pos_rel = ObsTerm(func=il_mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=il_mdp.joint_vel_rel)
        actions = ObsTerm(func=il_mdp.last_action)

        concatenate_terms = True

    # Override policy with EE-focused observations
    policy: EEPolicyCfg = EEPolicyCfg()


@configclass
class EERewardsCfg(BaseRewardsCfg):
    """Reward configuration focused on root and EE tracking."""

    # Remove joint tracking reward from base (set to None or override)
    motion_joint_pos_error_exp = None
    motion_lifting_object = None

    # EE tracking rewards
    motion_ee_position_coarse_error_exp = RewTerm(
        func=tracking_rewards.motion_ee_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_ee_position_fine_error_exp = RewTerm(
        func=tracking_rewards.motion_ee_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.2},
    )
    motion_ee_orientation_error_exp = RewTerm(
        func=tracking_rewards.motion_ee_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )

    # Action rate penalty
    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-0.01)

    # Joint position limits
    joint_pos_limit = RewTerm(
        func=il_mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # Optional: Kernel similarity reward for natural motion (disabled by default)
    natural_motion_similarity: RewTerm | None = None


@configclass
class EETerminationsCfg(BaseTerminationsCfg):
    """Termination configuration focused on root and EE tracking."""

    # Override base thresholds for EE tracking
    anchor_pos_error = DoneTerm(
        func=anchor_pos_error, params={"command_name": "motion", "threshold": 0.70}
    )
    anchor_quat_error = DoneTerm(
        func=anchor_quat_error, params={"command_name": "motion", "threshold": 0.70}
    )

    # Remove joint-based termination
    joint_pos_error = None

    # Add EE-based terminations
    ee_pos_error = DoneTerm(
        func=ee_position_error, params={"command_name": "motion", "threshold": 0.25}
    )
    ee_quat_error = DoneTerm(
        func=ee_quat_error, params={"command_name": "motion", "threshold": 1.20}
    )


@configclass
class V2PEEEnvCfg(V2PEnvCfg):
    """V2P environment focused on root and EE tracking."""

    observations: EEObservationsCfg = EEObservationsCfg()
    rewards: EERewardsCfg = EERewardsCfg()
    terminations: EETerminationsCfg = EETerminationsCfg()
