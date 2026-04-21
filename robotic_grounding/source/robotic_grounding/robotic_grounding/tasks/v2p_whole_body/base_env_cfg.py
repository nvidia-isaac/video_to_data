"""Base V2P whole-body environment configuration.

Provides scene, commands, and events. Child configs (G1SonicEnvCfg etc.)
add robot, actions, observations, rewards, and terminations.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from robotic_grounding.tasks.v2p_whole_body.mdp.commands import TrackingCommandCfg
from robotic_grounding.tasks.v2p_whole_body.mdp.events import (
    reset_robot_to_trajectory_start,
)


@configclass
class V2PSceneCfg(InteractiveSceneCfg):
    """Scene configuration for V2P whole-body tasks. Objects are added dynamically via scene config."""

    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", debug_vis=False
    )

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
class BaseEventsCfg:
    """Base event configuration."""

    reset_to_trajectory_frame = EventTerm(
        func=reset_robot_to_trajectory_start,
        params={"command_name": "motion", "trajectory_time_index": (0, 999999)},
        mode="reset",
    )


@configclass
class BaseCommandsCfg:
    """Base command configuration."""

    motion: TrackingCommandCfg = TrackingCommandCfg(
        asset_name="robot",
        motion_file=MISSING,
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
    events: BaseEventsCfg = BaseEventsCfg()
    commands: BaseCommandsCfg = BaseCommandsCfg()
    scene_config_path: str | None = None

    def __post_init__(self) -> None:
        """Configure simulation defaults and apply scene config."""
        self.decimation = 4
        self.episode_length_s = 5.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.viewer.eye = (-2.5, -5.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.viewer.origin_type = "world"

        if isinstance(self.scene_config_path, str):
            scene_config = SceneConfig.from_motion_file(self.scene_config_path)
            apply_scene_config(self, scene_config)
