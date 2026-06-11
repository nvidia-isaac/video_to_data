# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omegaconf import MISSING

from robotic_grounding.tasks.v2p import mdp

#################################################
# Scene definition
#################################################


@configclass
class V2PSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with two robots."""

    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", debug_vis=False
    )

    # robots
    right_robot: ArticulationCfg = MISSING
    left_robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


#################################################
# MDP settings
#################################################


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Patched by apply_scene_commands with scene-specific fields.
    dual_hands_object_tracking_command = mdp.DualHandsObjectTrackingCommandCfg(
        motion_speed=0.5,
        reset_finger_openness=0.7,
        initial_virtual_object_control_curriculum_scale=1.0,
        virtual_object_control_decay_steps=20,
        virtual_object_control_decay_mode="step",
        recompute_hand_keypoints_from_object=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Virtual object control is added in apply_scene_commands for each object

    right_joint_residual_action = mdp.JointResidualWithTrackingActionCfg(
        asset_name="right_robot",
        joint_names=[".*"],
        tracking_controller_linear_stiffness=50.0,
        tracking_controller_linear_damping=10.0,
        tracking_controller_angular_stiffness=12.0,
        tracking_controller_angular_damping=0.1,
        wrist_position_scale=0.05,
        wrist_orientation_scale=0.15,
        finger_joint_scale=0.15,
        ema_factor=0.3,
    )

    left_joint_residual_action = mdp.JointResidualWithTrackingActionCfg(
        asset_name="left_robot",
        joint_names=[".*"],
        tracking_controller_linear_stiffness=50.0,
        tracking_controller_linear_damping=10.0,
        tracking_controller_angular_stiffness=12.0,
        tracking_controller_angular_damping=0.1,
        wrist_position_scale=0.05,
        wrist_orientation_scale=0.15,
        finger_joint_scale=0.15,
        ema_factor=0.3,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group. Order preserved."""

        wrist_position_e = ObsTerm(
            func=mdp.wrist_position_e,
            params={"command_name": "dual_hands_object_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        wrist_orientation_e = ObsTerm(
            func=mdp.wrist_orientation_e,
            params={"command_name": "dual_hands_object_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        wrist_velocity_b = ObsTerm(
            func=mdp.wrist_velocity_b,
            params={"command_name": "dual_hands_object_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        finger_joint_pos = ObsTerm(
            func=mdp.finger_joint_pos,
            params={"command_name": "dual_hands_object_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        finger_joint_vel = ObsTerm(
            func=mdp.finger_joint_vel,
            params={"command_name": "dual_hands_object_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_position_e = ObsTerm(
            func=mdp.object_position_e,
            params={"command_name": "dual_hands_object_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_orientation_e = ObsTerm(
            func=mdp.object_orientation_e,
            params={"command_name": "dual_hands_object_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        # object_t_wrist = ObsTerm(
        #     func=mdp.object_t_wrist,
        #     params={"command_name": "dual_hands_object_tracking_command"},
        # )

        # object_p_fingertip = ObsTerm(
        #     func=mdp.object_p_fingertip,
        #     params={"command_name": "dual_hands_object_tracking_command"},
        # )

        command = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "dual_hands_object_tracking_command"},
        )

        actions = ObsTerm(func=isaac_mdp.last_action)
        processed_right_actions = ObsTerm(
            func=mdp.processed_action,
            params={"action_name": "right_joint_residual_action"},
        )
        processed_left_actions = ObsTerm(
            func=mdp.processed_action,
            params={"action_name": "left_joint_residual_action"},
        )

        contact_position_direction_in_wrist = ObsTerm(
            func=mdp.contact_position_direction_in_wrist,
            params={"command_name": "dual_hands_object_tracking_command"},
        )

        def __post_init__(self) -> None:
            """Post initialization."""
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # prestartup
    setup_collision_groups = EventTerm(
        func=mdp.configure_collision_groups,
        mode="prestartup",
        params={
            "robot_names": [],
            "object_names": [],
            "fixed_object_names": [],
            "disable_robot_to_object_collisions": False,
            "disable_robot_to_fixed_object_collisions": True,
        },
    )

    # startup
    right_physics_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("right_robot", body_names=".*"),
            "static_friction_range": (2.0, 2.01),
            "dynamic_friction_range": (2.0, 2.01),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    left_physics_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("left_robot", body_names=".*"),
            "static_friction_range": (2.0, 2.01),
            "dynamic_friction_range": (2.0, 2.01),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Object physics material event term disabled — SceneEntityCfg("object") does
    # not match any real scene entity (objects spawn per-sequence with dynamic
    # names). Objects use IsaacLab's default RigidBodyMaterialCfg (static=0.5,
    # dynamic=0.5, restitution=0.0).
    # object_physics_material = EventTerm(
    #     func=isaac_mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object", body_names=".*"),
    #         "static_friction_range": (0.99, 1.01),
    #         "dynamic_friction_range": (0.99, 1.01),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-5e-3)
    action_l1 = RewTerm(
        func=mdp.action_norm,
        weight=-2e-3,
        params={
            "action_names": [
                "right_joint_residual_action",
                "left_joint_residual_action",
            ]
        },
    )

    # right_joint_limit = RewTerm(
    #     func=isaac_mdp.joint_pos_limits,
    #     weight=-10.0,
    #     params={"asset_cfg": SceneEntityCfg("right_robot", joint_names=[".*"])},
    # )
    # left_joint_limit = RewTerm(
    #     func=isaac_mdp.joint_pos_limits,
    #     weight=-10.0,
    #     params={"asset_cfg": SceneEntityCfg("left_robot", joint_names=[".*"])},
    # )

    object_keypoints_tracking_exp = RewTerm(
        func=mdp.object_keypoints_tracking_exp,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.1,
        },
    )

    object_meshvert_tracking_fine = RewTerm(
        func=mdp.object_meshvert_tracking_fine,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.001,
        },
    )

    hand_keypoints_tracking_exp = RewTerm(
        func=mdp.hand_keypoints_tracking_exp,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.1,
        },
    )

    hand_joint_pos_tracking_exp = RewTerm(
        func=mdp.hand_joint_pos_tracking_exp,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 1.0,
        },
    )

    termination_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-100.0,
    )

    contact_wrench_support_reward = RewTerm(
        func=mdp.contact_wrench_support_reward,
        weight=10.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "tolerance": 0.1,
            "var": 0.1,
        },
    )

    contact_wrench_continuous_reward = RewTerm(
        func=mdp.contact_wrench_continuous_reward,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "approach_var": 0.05,
            "in_contact_force_threshold": 1e-3,
        },
    )

    contact_wrench_cumulative_reward = RewTerm(
        func=mdp.contact_wrench_cumulative_reward,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "eps": 1e-6,
            "streak_scale": 20.0,
        },
    )

    unintended_contact_penalty = RewTerm(
        func=mdp.unintended_contact_penalty,
        weight=-2.5,
        params={
            "command_name": "dual_hands_object_tracking_command",
        },
    )
    missed_contact_penalty = RewTerm(
        func=mdp.missed_contact_penalty,
        weight=-0.25,
        params={
            "command_name": "dual_hands_object_tracking_command",
        },
    )

    dexmachina_contact_tracking_reward = RewTerm(
        func=mdp.dexmachina_contact_tracking_reward,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.03,
            "mask_zero_contact": True,
        },
    )

    relative_object_pose_reward = RewTerm(
        func=mdp.relative_object_pose_reward,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "pos_sigma": 0.05,
            "rot_sigma": 0.5,
        },
    )

    relative_object_pos_reward = RewTerm(
        func=mdp.relative_object_pos_reward,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "pos_sigma": 0.02,
        },
    )

    relative_object_rot_reward = RewTerm(
        func=mdp.relative_object_rot_reward,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "rot_sigma": 0.3,
        },
    )

    inter_object_proximity_reward = RewTerm(
        func=mdp.inter_object_proximity_reward,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "dist_sigma": 0.05,
        },
    )

    contact_force_reward = RewTerm(
        func=mdp.contact_force_reward,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 2.0,
            "threshold": 0.0,
        },
    )

    # contact_force_range_reward = RewTerm(
    #     func=mdp.contact_force_range_reward,
    #     weight=0.0,
    #     params={
    #         "command_name": "dual_hands_object_tracking_command",
    #         "var": 1.0,
    #         "lower_force_squared": 4.0,
    #         "upper_force_squared": 16.0,
    #     },
    # )

    # contact_force_rate_reward = RewTerm(
    #     func=mdp.contact_force_rate_reward,
    #     weight=0.0,
    #     params={
    #         "command_name": "dual_hands_object_tracking_command",
    #         "var": 1.0,
    #     },
    # )

    # contact_slippage_reward = RewTerm(
    #     func=mdp.contact_slippage_reward,
    #     weight=1.0,
    #     params={
    #         "command_name": "dual_hands_object_tracking_command",
    #         "var": 1.0,
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.timestep_timeout,
        time_out=True,
        params={
            "command_name": "dual_hands_object_tracking_command",
        },
    )

    hand_wrist_away_from_trajectory = DoneTerm(
        func=mdp.hand_wrist_away_from_trajectory,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "threshold": 0.25,
        },
    )

    object_away_from_trajectory = DoneTerm(
        func=mdp.object_away_from_trajectory,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "position_threshold": 0.15,
            "orientation_threshold": 0.7,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    virtual_object_control_curriculum = CurrTerm(
        func=mdp.VirtualObjectControlCurriculum,
        params={
            "reward_thresholds": {
                "object_keypoints_tracking_exp": 0.1,
                "hand_keypoints_tracking_exp": 0.0,
                "contact_wrench_support_reward": 1.6,
                "contact_force_reward": 0.0,
            },
            "episode_length_ratio_threshold": 0.95,
            "decay_mode": "exponential",
            "deque_maxlen": 500,
            "command_name": "dual_hands_object_tracking_command",
            "zero_scale_factor_threshold": 0.05,
            "initial_wait_env_steps": 2000,
            "wait_env_steps_since_last_decay": 1000,
            "exponential_decay_factor": 0.9,
            "linear_decay_step": 10.0,
            "fixed_schedule_steps": None,
            "fixed_schedule_values": None,
            # 0.0 = disabled (filtered out at runtime); set > 0 to activate gate
            "metric_thresholds": {
                "contact_wrench_support_ratio_right": 0.0,
                "contact_wrench_support_ratio_left": 0.0,
                "contact_bodies_coverage_frac_right": 0.0,
                "contact_bodies_coverage_frac_left": 0.0,
            },
            # 0.0 = disabled; set > 0 to require current deque mean >= baseline * ratio
            "reward_baseline_retention": {
                "contact_wrench_support_reward": 0.0,
            },
            # custom_schedule mode: explicit VOC levels + paired reward weight changes.
            # Each list has one entry per decay event. Empty = custom_schedule disabled.
            "custom_voc_schedule": [],
            "custom_reward_schedules": {
                "object_keypoints_tracking_exp": [],
                "hand_keypoints_tracking_exp": [],
                "hand_joint_pos_tracking_exp": [],
                "object_meshvert_tracking_fine": [],
            },
            # Force decay after this many env steps of being gate-eligible without firing.
            # 0 = disabled. Only active in custom_schedule mode.
            "max_eligible_wait_env_steps": 0,
            # 0.0 = disabled; set > 0 to require metric <= threshold before decay.
            "metric_upper_thresholds": {
                "contact_wrench_support_reward_cv": 0.0,
                # For multi-object tasks: gate VOC decay on relative orientation quality.
                # Set to e.g. 0.15 (rad) to prevent decay while rot_err is too large.
                # Ignored (with a warning suppressed by 0.0) on single-object tasks.
                "relative_object_rot_error": 0.0,
            },
            # If True, metric_upper_thresholds gate only applies before the first decay.
            "metric_upper_thresholds_initial_only": False,
        },
    )


@configclass
class FixedTimestepCurriculumCfg:
    """Curriculum for virtual object control with a fixed timestep decay."""

    fixed_timestep_curriculum = CurrTerm(
        func=mdp.FixedTimestepCurriculum,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "num_steps_per_env": 24,
            # Last entry (16500) is a post-training reward boost: VOC has been 0
            # since iter 14500, so the policy is already self-driving; the jump
            # from 0.5 -> 20.0 on object_keypoints_tracking_exp amplifies the
            # tracking signal for fine-grained object pose convergence.
            "timestep_schedule": [
                2000,
                4000,
                5500,
                7000,
                8500,
                10000,
                11500,
                13000,
                14500,
                16000,
                16500,
            ],
            "virtual_object_control_scale_factor": [
                1.0,
                1.0,
                0.75,
                0.5,
                0.25,
                0.1,
                0.05,
                0.025,
                0.01,
                0.0,
                0.0,
            ],
            "rewards_object_keypoints_tracking_exp": [
                0.0,
                0.0,
                0.1,
                0.1,
                0.2,
                0.25,
                0.25,
                0.5,
                0.5,
                0.5,
                20.0,
            ],
            "rewards_hand_keypoints_tracking_exp": [
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.25,
            ],
            "rewards_hand_joint_pos_tracking_exp": [
                0.25,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.25,
            ],
            "rewards_contact_wrench_support_reward": 10.0,
            "rewards_unintended_contact_penalty": -20.0,
            "rewards_missed_contact_penalty": -5.0,
            # Optional per-step scheduled rewards — 0.0 scalar expands to match
            # the timestep_schedule length at runtime.
            "rewards_object_meshvert_tracking_fine": 0.0,
            "rewards_dexmachina_contact_tracking_reward": 0.0,
            "rewards_relative_object_pos_reward": 0.0,
            "rewards_relative_object_rot_reward": 0.0,
            "rewards_inter_object_proximity_reward": 0.0,
            # Optional per-step termination thresholds — None means no override.
            "termination_object_away_from_trajectory_position_threshold": None,
            "termination_object_away_from_trajectory_orientation_threshold": None,
            "termination_hand_wrist_away_from_trajectory_threshold": None,
        },
    )


#################################################
# Environment configuration
#################################################


@configclass
class V2PHandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: V2PSceneCfg = V2PSceneCfg(
        num_envs=4096,
        env_spacing=1.5,
        replicate_physics=False,
        filter_collisions=False,
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: FixedTimestepCurriculumCfg = FixedTimestepCurriculumCfg()

    max_contact_data_count_per_prim: int = 1024

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 5  # 20 Hz control
        self.episode_length_s = 20.0  # Overridden by the scene config
        # simulation settings
        self.sim.dt = 0.01  # 100 Hz simulation
        self.sim.render_interval = self.decimation  # 20 Hz rendering
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23

        # Make the environment more compliant
        self.sim.physics_material.compliant_contact_stiffness = 10.0
        self.sim.physics_material.compliant_contact_damping = 1.0

        # viewer settings
        self.viewer.eye = (-0.5, 0.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.2)
        # self.viewer.resolution = (3840, 2160)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 6


@configclass
class V2PHandEnvCfgEnvOnly(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: V2PSceneCfg = V2PSceneCfg(num_envs=4096, env_spacing=1.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: FixedTimestepCurriculumCfg = FixedTimestepCurriculumCfg()

    max_contact_data_count_per_prim: int = 1024

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 5  # 20 Hz control
        self.episode_length_s = 20.0  # Overridden by the scene config
        # simulation settings
        self.sim.dt = 0.01  # 100 Hz simulation
        self.sim.render_interval = self.decimation  # 20 Hz rendering
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23

        # Make the environment more compliant
        self.sim.physics_material.compliant_contact_stiffness = 10.0
        self.sim.physics_material.compliant_contact_damping = 1.0
