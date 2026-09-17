# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal env config for verifying object spawning from a scene motion file.

Uses ManagerBasedRLEnvCfg (with empty rewards/terminations) so the env can
be created via ``gym.make()`` and wrapped with ``RecordVideo`` for MP4 export.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from scipy.spatial.transform import Rotation as R

from robotic_grounding.tasks.scene_utils import (
    SceneConfig,
    apply_scene_objects,
    apply_scene_robot,
)
from robotic_grounding.tasks.scene_utils.replay_data import (
    DualHandTrajectory,
    SingleRobotTrajectory,
    load_replay_trajectory,
)
from robotic_grounding.tasks.v2d import mdp


def _dummy_obs(env: object) -> torch.Tensor:
    return torch.zeros(env.num_envs, 1, device=env.device)  # type: ignore[attr-defined]


def _zero_reward(env: object) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)  # type: ignore[attr-defined]


def _never_done(env: object) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)  # type: ignore[attr-defined]


def _seed_robot_init_state_from_motion(env_cfg: object, motion_file: str) -> None:
    """Seed robot articulation init states from frame 0 of a replay motion.

    Without this, the viewer spawns the URDF at the asset default pose
    (e.g. world origin facing +X) and lets gravity drop it. The retargeted
    object trajectory is stored in the same world frame as the retargeted
    robot, so re-zeroing the robot rotates the relative robot/object
    heading and the object can end up behind a forward-facing URDF when
    the original body had a different yaw.
    """
    try:
        replay = load_replay_trajectory(motion_file)
    except Exception:  # noqa: BLE001 -- viewer-only path, missing fields are non-fatal
        return

    scene = env_cfg.scene  # type: ignore[attr-defined]
    if isinstance(replay, SingleRobotTrajectory):
        robot_cfg = getattr(scene, "robot", None)
        if robot_cfg is None or replay.num_frames == 0:
            return

        pos = tuple(float(value) for value in replay.robot_root_position[0].tolist())
        rot = tuple(float(value) for value in replay.robot_root_wxyz[0].tolist())
        joint_pos = {
            name: float(value)
            for name, value in zip(
                replay.robot_joint_names,
                replay.robot_joint_positions[0],
                strict=True,
            )
        }
        scene.robot = robot_cfg.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos,
                rot=rot,
                joint_pos=joint_pos,
                joint_vel={".*": 0.0},
            ),
        )
        return

    if not isinstance(replay, DualHandTrajectory) or replay.num_frames == 0:
        return

    right_robot_cfg = getattr(scene, "right_robot", None)
    left_robot_cfg = getattr(scene, "left_robot", None)
    if right_robot_cfg is None or left_robot_cfg is None:
        return

    right_pos = tuple(float(value) for value in replay.right_wrist_position[0].tolist())
    left_pos = tuple(float(value) for value in replay.left_wrist_position[0].tolist())
    if replay.wrist_orientation_format == "wxyz":
        right_rot_values = replay.right_wrist_orientation[0].tolist()
        left_rot_values = replay.left_wrist_orientation[0].tolist()
    else:
        right_rot_values = R.from_euler(
            "XYZ", replay.right_wrist_orientation[0], degrees=False
        ).as_quat(scalar_first=True)
        left_rot_values = R.from_euler(
            "XYZ", replay.left_wrist_orientation[0], degrees=False
        ).as_quat(scalar_first=True)
    right_rot = tuple(float(value) for value in right_rot_values)
    left_rot = tuple(float(value) for value in left_rot_values)

    scene.right_robot = right_robot_cfg.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=right_pos,
            rot=right_rot,
            joint_vel={".*": 0.0},
        ),
    )
    scene.left_robot = left_robot_cfg.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=left_pos,
            rot=left_rot,
            joint_vel={".*": 0.0},
        ),
    )


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
class SceneViewerEventCfg:
    """Prestartup collision group setup; same contract as ``apply_scene_objects`` expects."""

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


@configclass
class SceneViewerRewardsCfg:
    """No-op reward for passive scene viewing."""

    dummy = RewTerm(func=_zero_reward, weight=0.0)


@configclass
class SceneViewerTerminationsCfg:
    """Never terminate for passive scene viewing."""

    dummy = DoneTerm(func=_never_done)


@configclass
class SceneViewerEnvCfg(ManagerBasedRLEnvCfg):
    """Spawns object + support surface from a SceneConfig. No policy, no RL reward.

    Extends ManagerBasedRLEnvCfg so the env can be created with ``gym.make()``
    and wrapped with ``gym.wrappers.RecordVideo`` for MP4 export. Keeps an
    event cfg so ``apply_scene_robot`` can register robot names for
    collision-group configuration.
    """

    scene: SceneViewerSceneCfg = SceneViewerSceneCfg(
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=False,
    )
    observations: SceneViewerObsCfg = SceneViewerObsCfg()
    actions: SceneViewerActionsCfg = SceneViewerActionsCfg()
    events: SceneViewerEventCfg = SceneViewerEventCfg()
    rewards: SceneViewerRewardsCfg = SceneViewerRewardsCfg()
    terminations: SceneViewerTerminationsCfg = SceneViewerTerminationsCfg()

    episode_length_s: float = 1000.0  # effectively infinite for passive viewing
    motion_file: str | None = None

    def __post_init__(self) -> None:
        """Post-init: configure simulation and load scene from motion file."""
        self.decimation = 1
        self.sim.dt = 0.01
        self.sim.render_interval = 1

        if self.motion_file is not None:
            scene_config = SceneConfig.from_motion_file(self.motion_file)
            apply_scene_objects(self, scene_config)
            if scene_config.robot_name is not None:
                apply_scene_robot(self, scene_config, static=False)
                _seed_robot_init_state_from_motion(self, self.motion_file)
