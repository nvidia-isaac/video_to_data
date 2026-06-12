# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand-only tracking environment configuration (no object)."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robotic_grounding.tasks.v2p import mdp

#################################################
# Scene definition
#################################################


@configclass
class V2PTrackingSceneCfg(InteractiveSceneCfg):
    """Configuration for the tracking-only scene with two robots (no object)."""

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

    dual_hands_tracking_command = mdp.DualHandsTrackingCommandCfg(
        debug_vis=False,
        motion_speed=0.2,
        reset_finger_openness=0.7,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    right_joint_direct_action = mdp.JointDirectPositionActionCfg(
        asset_name="right_robot",
        joint_names=[".*"],
        command_name="dual_hands_tracking_command",
        tracking_controller_linear_stiffness=50.0,
        tracking_controller_linear_damping=10.0,
        tracking_controller_angular_stiffness=12.0,
        tracking_controller_angular_damping=0.1,
        wrist_position_scale=0.05,
        wrist_orientation_scale=0.15,
        finger_joint_scale=0.15,
        finger_joint_clip=100.0,
        ema_factor=0.9,
    )

    left_joint_direct_action = mdp.JointDirectPositionActionCfg(
        asset_name="left_robot",
        joint_names=[".*"],
        command_name="dual_hands_tracking_command",
        tracking_controller_linear_stiffness=50.0,
        tracking_controller_linear_damping=10.0,
        tracking_controller_angular_stiffness=12.0,
        tracking_controller_angular_damping=0.1,
        wrist_position_scale=0.05,
        wrist_orientation_scale=0.15,
        finger_joint_scale=0.15,
        finger_joint_clip=100.0,
        ema_factor=0.9,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group. Order preserved."""

        wrist_position_e = ObsTerm(
            func=mdp.wrist_position_e,
            params={"command_name": "dual_hands_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        wrist_orientation_e = ObsTerm(
            func=mdp.wrist_orientation_e,
            params={"command_name": "dual_hands_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        finger_joint_pos = ObsTerm(
            func=mdp.finger_joint_pos,
            params={"command_name": "dual_hands_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        finger_joint_vel = ObsTerm(
            func=mdp.finger_joint_vel,
            params={"command_name": "dual_hands_tracking_command"},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        command = ObsTerm(
            func=isaac_mdp.generated_commands,
            params={"command_name": "dual_hands_tracking_command"},
        )

        actions = ObsTerm(func=isaac_mdp.last_action)
        prev_right_actions = ObsTerm(
            func=mdp.prev_action, params={"action_name": "right_joint_direct_action"}
        )
        prev_left_actions = ObsTerm(
            func=mdp.prev_action, params={"action_name": "left_joint_direct_action"}
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-5e-4)
    action_l1 = RewTerm(
        func=mdp.action_norm,
        weight=-5e-3,
        params={
            "action_names": [
                "right_joint_direct_action",
                "left_joint_direct_action",
            ]
        },
    )

    right_joint_limit = RewTerm(
        func=isaac_mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("right_robot", joint_names=[".*"])},
    )
    left_joint_limit = RewTerm(
        func=isaac_mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("left_robot", joint_names=[".*"])},
    )

    hand_keypoints_tracking_exp = RewTerm(
        func=mdp.hand_keypoints_tracking_exp,
        weight=1.0,
        params={
            "command_name": "dual_hands_tracking_command",
            "var": 0.05,
            "threshold": 0.02,
        },
    )

    hand_joint_pos_tracking_exp = RewTerm(
        func=mdp.hand_joint_pos_tracking_exp,
        weight=1.0,
        params={
            "command_name": "dual_hands_tracking_command",
            "var": 0.05,
        },
    )

    termination_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-300.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.timestep_timeout,
        time_out=True,
        params={
            "command_name": "dual_hands_tracking_command",
        },
    )

    hand_wrist_away_from_trajectory = DoneTerm(
        func=mdp.hand_wrist_away_from_trajectory,
        params={
            "command_name": "dual_hands_tracking_command",
            "threshold": 0.15,
        },
    )


#################################################
# Environment configuration
#################################################


@configclass
class V2PHandTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the hand-only tracking environment (no object)."""

    # Scene settings
    scene: V2PTrackingSceneCfg = V2PTrackingSceneCfg(num_envs=4096, env_spacing=1.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 41.4
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 17 * 2**15

        # Make the environment more compliant
        self.sim.physics_material.compliant_contact_stiffness = 10.0
        self.sim.physics_material.compliant_contact_damping = 1.0

        # viewer settings
        self.viewer.eye = (1.5, 1.5, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.5)
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
