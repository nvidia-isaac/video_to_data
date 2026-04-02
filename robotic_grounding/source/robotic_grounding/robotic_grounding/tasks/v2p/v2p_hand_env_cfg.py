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

    # # table
    # table = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=[0.0, -0.14, 0.475], rot=[1.0, 0.0, 0.0, 0.0]
    #     ),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.15, 0.15, 0.952),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.14, 0.14, 0.14), metallic=0.7
    #         ),
    #     ),
    # )

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
    dual_hands_object_tracking_command: mdp.DualHandsObjectTrackingCommandCfg = (
        mdp.DualHandsObjectTrackingCommandCfg()
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
        tracking_controller_angular_damping=0.5,
        wrist_position_scale=0.05,
        wrist_orientation_scale=0.15,
        finger_joint_scale=0.15,
        ema_factor=0.1,
    )

    left_joint_residual_action = mdp.JointResidualWithTrackingActionCfg(
        asset_name="left_robot",
        joint_names=[".*"],
        tracking_controller_linear_stiffness=50.0,
        tracking_controller_linear_damping=10.0,
        tracking_controller_angular_stiffness=12.0,
        tracking_controller_angular_damping=0.5,
        wrist_position_scale=0.05,
        wrist_orientation_scale=0.15,
        finger_joint_scale=0.15,
        ema_factor=0.1,
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
        prev_right_actions = ObsTerm(
            func=mdp.prev_action, params={"action_name": "right_joint_residual_action"}
        )
        prev_left_actions = ObsTerm(
            func=mdp.prev_action, params={"action_name": "left_joint_residual_action"}
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

    # startup
    right_physics_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("right_robot", body_names=".*"),
            "static_friction_range": (0.99, 1.01),
            "dynamic_friction_range": (0.99, 1.01),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    left_physics_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("left_robot", body_names=".*"),
            "static_friction_range": (0.99, 1.01),
            "dynamic_friction_range": (0.99, 1.01),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

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

    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-5e-4)
    action_l1 = RewTerm(
        func=mdp.action_norm,
        weight=-5e-3,
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
        weight=1.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.1,
        },
    )

    hand_keypoints_tracking_exp = RewTerm(
        func=mdp.hand_keypoints_tracking_exp,
        weight=1.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.05,
            "threshold": 0.02,  # FIXME: Need to be 0 during tracking training
        },
    )

    hand_joint_pos_tracking_exp = RewTerm(
        func=mdp.hand_joint_pos_tracking_exp,
        weight=0.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.05,
        },
    )

    termination_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-100.0,
    )

    # Contact tracking (chamfer) reward
    contact_tracking = RewTerm(
        func=mdp.contact_tracking_reward,
        weight=1.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 0.03,
            "mask_zero_contact": True,
        },
    )

    contact_force = RewTerm(
        func=mdp.contact_force_reward,
        weight=1.0,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 1.0,
            "threshold": 2.0,
        },
    )

    contact_force_rate = RewTerm(
        func=mdp.contact_force_rate_reward,
        weight=0.25,
        params={
            "command_name": "dual_hands_object_tracking_command",
            "var": 1.0,
        },
    )

    # FIXME: add appropriate contact reward
    # contact_force_penalty = RewTerm(
    #     func=mdp.contact_force_penalty, weight=-0.05, params={}
    # )

    # # ManipTrans-style contact reward
    # maniptrans_contact = RewTerm(
    #     func=mdp.maniptrans_contact_reward,
    #     weight=1.0,
    #     params={
    #         "contact_range_min": 0.02,
    #         "contact_range_max": 0.03,
    #         "decay_constant": 1.0,
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
            "threshold": 0.15,
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
                "hand_keypoints_tracking_exp": 1.0,
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
        },
    )


#################################################
# Environment configuration
#################################################


@configclass
class V2PHandEnvCfg(ManagerBasedRLEnvCfg):
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
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        """Post initialization."""
        # general settings
        self.decimation = 10  # 20 Hz control
        self.episode_length_s = 20.0  # Overridden by the scene config
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz simulation
        self.sim.render_interval = 4  # 50 Hz rendering
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 17 * 2**15

        # Make the environment more compliant
        self.sim.physics_material.compliant_contact_stiffness = 10.0
        self.sim.physics_material.compliant_contact_damping = 1.0

        # viewer settings
        self.viewer.eye = (1.5, 1.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.5)
        # self.viewer.resolution = (3840, 2160)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 6


@configclass
class V2PHandEnvCfgEnvOnly(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: V2PSceneCfg = V2PSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        """Post initialization."""
        # general settings
        self.decimation = 10  # 20 Hz control
        self.episode_length_s = 20.0  # Overridden by the scene config
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz simulation
        self.sim.render_interval = 4  # 50 Hz rendering
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 17 * 2**15

        # Make the environment more compliant
        self.sim.physics_material.compliant_contact_stiffness = 10.0
        self.sim.physics_material.compliant_contact_damping = 1.0
