# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from robotic_grounding.tasks.v2d import mdp

#################################################
# Scene definition
#################################################


@configclass
class V2DSceneCfg(InteractiveSceneCfg):
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
    )

    left_joint_residual_action = mdp.JointResidualWithTrackingActionCfg(
        asset_name="left_robot",
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

    object_keypoints_tracking_exp = RewTerm(
        func=mdp.object_keypoints_tracking_exp,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.1,
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

    unintended_contact_penalty = RewTerm(
        func=mdp.unintended_contact_penalty,
        weight=-10.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
        },
    )

    missed_contact_penalty = RewTerm(
        func=mdp.missed_contact_penalty,
        weight=-1.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
        },
    )


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
            "threshold": 0.2,
        },
    )

    object_away_from_trajectory = DoneTerm(
        func=mdp.object_away_from_trajectory,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "position_threshold": 0.2,
            "orientation_threshold": 0.7,
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
            "timestep_schedule": [
                2000,
                3500,
                5000,
                6500,
                8000,
                9500,
                11000,
                12500,
                14000,
                15500,
            ],
            "virtual_object_control_scale_factor": [
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
                0.1,
                0.25,
                0.25,
                0.5,
                0.5,
                1.0,
                1.0,
                1.0,
                20.0,
            ],
            "rewards_hand_keypoints_tracking_exp": 0.25,
            "rewards_hand_joint_pos_tracking_exp": 0.25,
            "rewards_contact_wrench_support_reward": 10.0,
            "rewards_unintended_contact_penalty": -10.0,
            "rewards_missed_contact_penalty": -1.0,
        },
    )


#################################################
# Environment configuration
#################################################


@configclass
class V2DHandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: V2DSceneCfg = V2DSceneCfg(
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

        # viewer settings
        self.viewer.eye = (-0.5, 0.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 1.2)
        # self.viewer.resolution = (3840, 2160)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 6
