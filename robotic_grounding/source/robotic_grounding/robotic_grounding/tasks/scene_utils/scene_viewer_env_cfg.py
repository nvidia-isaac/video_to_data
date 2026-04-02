"""Minimal env config for verifying object spawning from a scene motion file."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_objects


def _dummy_obs(env: object) -> torch.Tensor:
    return torch.zeros(env.num_envs, 1, device=env.device)  # type: ignore[attr-defined]


@configclass
class SceneViewerSceneCfg(InteractiveSceneCfg):
    """Scene with terrain and lights for the scene viewer."""

    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", debug_vis=False
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class SceneViewerObsCfg:
    """Observation config with a single dummy term."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Single dummy observation group."""

        dummy = ObsTerm(func=_dummy_obs)

        def __post_init__(self) -> None:
            """Post-init: disable corruption and concatenate terms."""
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class SceneViewerActionsCfg:
    """Empty actions config for passive scene viewing."""


@configclass
class SceneViewerEnvCfg(ManagerBasedEnvCfg):
    """Spawns object + support surface from a SceneConfig YAML. No robot, no RL."""

    scene: SceneViewerSceneCfg = SceneViewerSceneCfg(num_envs=1, env_spacing=3.0)
    observations: SceneViewerObsCfg = SceneViewerObsCfg()
    actions: SceneViewerActionsCfg = SceneViewerActionsCfg()

    motion_file: str | None = None

    def __post_init__(self) -> None:
        """Post-init: configure simulation and load scene from motion file."""
        self.decimation = 1
        self.sim.dt = 0.01
        self.sim.render_interval = 1

        if self.motion_file is not None:
            scene_config = SceneConfig.from_motion_file(self.motion_file)
            apply_scene_objects(self, scene_config)
