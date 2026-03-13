# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Debug environment configuration for Vega Sharpa with GUI joint control.

Based on SharpaDebugEnvCfg; overrides scene (Vega robot only, no object/table)
and actions (joint GUI only) while reusing the same debug patterns.
"""

import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from robotic_grounding.assets.vega_sharpa import VEGA_SHARPA_PLANAR_CFG
from robotic_grounding.tasks.debug.mdp import JointGUIActionCfg
from robotic_grounding.tasks.debug.sharpa_debug_env_cfg import SharpaDebugEnvCfg


@configclass
class VegaSharpaDebugSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground + Vega Sharpa robot + lights."""

    terrain = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False,
    )

    robot: ArticulationCfg = VEGA_SHARPA_PLANAR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
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
class VegaSharpaDebugActionsCfg:
    """GUI control for all Vega Sharpa joints."""

    joint_pos = JointGUIActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
        velocity_joint_names=[
            "virtual_x",
            "virtual_y",
            "virtual_yaw",
        ],
    )


@configclass
class VegaSharpaDebugRewardsCfg:
    """Minimal rewards for debug."""

    is_alive = RewTerm(func=isaac_mdp.is_alive, weight=1.0)


@configclass
class VegaSharpaDebugTerminationsCfg:
    """Minimal terminations for debug."""

    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)


@configclass
class VegaSharpaDebugEnvCfg(SharpaDebugEnvCfg):
    """Debug environment for Vega Sharpa with interactive GUI control."""

    scene: VegaSharpaDebugSceneCfg = VegaSharpaDebugSceneCfg(
        num_envs=1, env_spacing=4.0
    )
    actions: VegaSharpaDebugActionsCfg = VegaSharpaDebugActionsCfg()
    rewards: VegaSharpaDebugRewardsCfg = VegaSharpaDebugRewardsCfg()
    terminations: VegaSharpaDebugTerminationsCfg = VegaSharpaDebugTerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        self.sim.dt = 0.005
        self.sim.decimation = 4

        # Use the planar Vega Sharpa robot
        self.scene.robot = VEGA_SHARPA_PLANAR_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {".*": 0.0}

        # Remove object and table from parent scene
        if hasattr(self.scene, "object"):
            delattr(self.scene, "object")
        if hasattr(self.scene, "table"):
            delattr(self.scene, "table")

        # Disable contact sensors for now
        for name in list(getattr(self, "finger_sensor_names", [])):
            if hasattr(self.scene, name):
                delattr(self.scene, name)
        self.finger_sensor_names: list[str] = []

        # Reset action scale
        self.actions.joint_pos.scale = 1.0

        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
