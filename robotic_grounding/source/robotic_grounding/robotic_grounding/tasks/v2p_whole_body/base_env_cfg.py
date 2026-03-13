"""Base V2P whole-body environment configuration."""

from dataclasses import MISSING

import isaaclab.envs.mdp as il_mdp
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
from isaaclab_tasks.manager_based.manipulation.lift.mdp.observations import (
    object_position_in_robot_root_frame,
)

from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from robotic_grounding.tasks.v2p_whole_body.mdp import observations as obs
from robotic_grounding.tasks.v2p_whole_body.mdp.commands import TrackingCommandCfg
from robotic_grounding.tasks.v2p_whole_body.mdp.events import (
    reset_robot_to_trajectory_start,
)
from robotic_grounding.tasks.v2p_whole_body.mdp.rewards import tracking_rewards
from robotic_grounding.tasks.v2p_whole_body.mdp.terminations import (
    anchor_pos_error,
    anchor_quat_error,
    joint_pos_error,
    object_pos_error,
)


@configclass
class V2PSceneCfg(InteractiveSceneCfg):
    """Scene configuration for V2P whole-body tasks. Objects are added dynamically via scene config."""

    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", debug_vis=False
    )

    # Robot articulation - must be set by child config
    robot: ArticulationCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class BaseObservationsCfg:
    """Base observation configuration. Controllers should add their specific observation groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Base policy observations for motion tracking."""

        motion_anchor_pos_b = ObsTerm(
            func=obs.motion_anchor_pos_b,
            params={"command_name": "motion", "num_future_frames": 1},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        motion_anchor_ori_b = ObsTerm(
            func=obs.motion_anchor_ori_b,
            params={"command_name": "motion", "num_future_frames": 1},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        motion_joint_pos_delta = ObsTerm(
            func=obs.motion_joint_pos_delta,
            params={"command_name": "motion", "num_future_frames": 10},
            noise=Unoise(n_min=-0.1, n_max=0.1),
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

    # Learned policy observations
    policy: PolicyCfg = PolicyCfg()


@configclass
class BaseRewardsCfg:
    """Base reward configuration for motion tracking."""

    termination_penalty = RewTerm(func=il_mdp.is_terminated, weight=-500.0)

    # Tracking rewards
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
        params={"command_name": "motion", "std": 0.5},
    )
    motion_object_position_error_exp = RewTerm(
        func=tracking_rewards.motion_object_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.2},
    )
    motion_lifting_object = RewTerm(
        func=tracking_rewards.motion_object_lifted,
        weight=1.0,
        params={"command_name": "motion"},
    )
    motion_progress = RewTerm(
        func=tracking_rewards.motion_progress,
        weight=1.0,
        params={"command_name": "motion"},
    )

    # Regularization
    action_rate = RewTerm(func=il_mdp.action_rate_l2, weight=-0.001)


@configclass
class BaseTerminationsCfg:
    """Base termination configuration."""

    timeout = DoneTerm(func=il_mdp.time_out, time_out=True)
    anchor_pos_error = DoneTerm(
        func=anchor_pos_error, params={"command_name": "motion", "threshold": 0.7}
    )
    anchor_quat_error = DoneTerm(
        func=anchor_quat_error, params={"command_name": "motion", "threshold": 0.7}
    )
    joint_pos_error = DoneTerm(
        func=joint_pos_error, params={"command_name": "motion", "threshold": 2.0}
    )
    object_pos_error = DoneTerm(
        func=object_pos_error, params={"command_name": "motion", "threshold": 0.10}
    )


@configclass
class BaseCurriculumsCfg:
    """Base curriculum configuration."""

    pass


@configclass
class BaseEventsCfg:
    """Base event configuration."""

    reset_to_motion_start = EventTerm(
        func=reset_robot_to_trajectory_start,
        params={"command_name": "motion", "trajectory_time_index": (0, 1)},
        mode="reset",
    )

    randomize_physics_material = EventTerm(
        func=il_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.9, 1.1),
            "restitution_range": (0.0, 0.25),
            "num_buckets": 64,
        },
    )

    randomize_base_com = EventTerm(
        func=il_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {
                "x": (-0.003, 0.003),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
            },
        },
    )

    randomize_joint_pos = EventTerm(
        func=il_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.005, 0.005),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=il_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.5, 1.5),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.1, 0.1),
                "roll": (-0.26, 0.26),
                "pitch": (-0.26, 0.26),
                "yaw": (-0.39, 0.39),
            },
        },
    )


@configclass
class BaseCommandsCfg:
    """Base command configuration."""

    motion: TrackingCommandCfg = TrackingCommandCfg(
        asset_name="robot",
        motion_file=MISSING,  # Motion file set via scene config
        anchor_body_name="pelvis",
        dt=0.02,
        num_future_frames=10,
        dt_future_frames=0.1,
        debug_vis=True,
    )


@configclass
class V2PEnvCfg(ManagerBasedRLEnvCfg):
    """Base V2P whole-body environment configuration."""

    scene: V2PSceneCfg = V2PSceneCfg(num_envs=1, env_spacing=3.0)
    observations: BaseObservationsCfg = BaseObservationsCfg()
    rewards: BaseRewardsCfg = BaseRewardsCfg()
    terminations: BaseTerminationsCfg = BaseTerminationsCfg()
    events: BaseEventsCfg = BaseEventsCfg()
    curriculums: BaseCurriculumsCfg = BaseCurriculumsCfg()
    commands: BaseCommandsCfg = BaseCommandsCfg()
    scene_config_path: str | None = None

    def __post_init__(self) -> None:
        """Post-initialization setup."""
        self.decimation = 4
        self.episode_length_s = 10.0

        # Sim settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # Viewer
        self.viewer.eye = (-2.5, -5.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.viewer.origin_type = "world"

        # Load scene config if a path is provided
        if isinstance(self.scene_config_path, str):
            scene_config = SceneConfig.from_yaml(self.scene_config_path)
            apply_scene_config(self, scene_config)
