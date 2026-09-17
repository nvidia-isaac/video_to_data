# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render external embodiment rollouts into the semantic GR00T HDF5 contract.

The external simulator remains the source of dynamics and absolute action labels.
IsaacLab is used only as a calibrated, visually randomized renderer. Input is a
``manifest.json`` plus ``episode_*.npz`` files; output is compatible with
``python -m groot_finetune.convert_to_gr00t``.

The explicit embodiment contract defines tensor order, dimensions, cameras, and
provenance; replay must preserve it exactly.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--export_dir", required=True)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--contract", required=True)
parser.add_argument("--task_profile", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_demos", type=int, default=None)
parser.add_argument("--record_output", default="out/groot_replay")
parser.add_argument("--visual_warmup_steps", type=int, default=60)
parser.add_argument(
    "--save_on", choices=("source_success", "all"), default="source_success"
)
parser.add_argument("--verify", action="store_true")
parser.add_argument("--no_dr", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim launches."""

import json
import sys
from pathlib import Path
from typing import TypedDict

import gymnasium as gym
import h5py
import isaaclab_tasks  # noqa: F401
import numpy as np
import robotic_grounding.tasks  # noqa: F401
import torch
from pxr import PhysxSchema, Usd, UsdPhysics
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from robotic_grounding.tasks.v2d_whole_body.config.vega_sharpa.vega_sharpa_gr00t_env_cfg import (
    VegaSharpaGr00tRecordEnvCfg,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from groot_finetune.contracts import (  # noqa: E402
    VEGA_SHARPA_JOINT,
    load_embodiment_contract,
)
from groot_finetune.replay_export import ReplayEpisode, load_replay_export  # noqa: E402
from groot_finetune.task_profile import load_task_profile  # noqa: E402

_VERIFY_JOINT_TOL = 1e-3
_VERIFY_OBJECT_TOL = 1e-3


class _ReplayBuffer(TypedDict):
    """In-memory frames for one parallel replay slot."""

    observations: dict[str, list[np.ndarray]]
    actions: list[np.ndarray]


def _disable_replay_dynamics(env, robot) -> None:
    """Disable drives/colliders and relax limits for exact kinematic rendering."""
    zeros = torch.zeros_like(robot.data.joint_pos)
    robot.write_joint_stiffness_to_sim(zeros)
    robot.write_joint_damping_to_sim(zeros)

    wide_limits = robot.data.joint_pos_limits.clone()
    wide_limits[..., 0] -= 0.5
    wide_limits[..., 1] += 0.5
    robot.write_joint_position_limit_to_sim(wide_limits)

    stage = env.sim.stage
    for env_prim in stage.GetPrimAtPath("/World/envs").GetChildren():
        robot_prim = stage.GetPrimAtPath(f"{env_prim.GetPath()}/Robot")
        if robot_prim.IsValid():
            for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    PhysxSchema.PhysxArticulationAPI.Apply(
                        prim
                    ).GetEnabledSelfCollisionsAttr().Set(False)
        for child in env_prim.GetChildren():
            if child.GetName() == "Robot":
                continue
            for prim in Usd.PrimRange(child):
                if prim.IsInstance():
                    prim.SetInstanceable(False)
            for prim in Usd.PrimRange(child):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)


def _as_record_array(name: str, value: torch.Tensor, slot: int) -> np.ndarray:
    """Copy one slot's observation, normalizing camera tensors to uint8 RGB."""
    array = value[slot].detach().cpu().numpy()
    if name.startswith("image"):
        array = array[..., :3]
        if array.dtype != np.uint8:
            array = (
                (np.asarray(array, dtype=np.float32) * 255.0)
                .clip(0, 255)
                .astype(np.uint8)
            )
    return array.copy()


def _write_demo(
    data_group: h5py.Group,
    demo_index: int,
    observations: dict[str, list[np.ndarray]],
    actions: list[np.ndarray],
    source_name: str,
    source_success: bool,
) -> None:
    """Write one buffered replay slot and release its memory."""
    demo = data_group.create_group(f"demo_{demo_index}")
    demo.attrs["num_samples"] = len(actions)
    demo.attrs["source_episode"] = source_name
    demo.attrs["source_success"] = source_success
    obs_group = demo.create_group("obs")
    for name, values in observations.items():
        kwargs = {"compression": "gzip"} if name.startswith("image") else {}
        obs_group.create_dataset(name, data=np.stack(values), **kwargs)
    demo.create_dataset("actions", data=np.stack(actions).astype(np.float32))


def _configure_replay_cfg(
    scene_config: SceneConfig,
) -> VegaSharpaGr00tRecordEnvCfg:
    """Build the native record task in dynamics-free replay mode."""
    cfg = VegaSharpaGr00tRecordEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.commands.motion.motion_file = args_cli.motion_file
    cfg.commands.motion.always_reset_to_first_frame = True
    cfg.commands.motion.debug_vis = False
    cfg.commands.motion.initial_virtual_object_control_curriculum_scale = 0.0
    cfg.commands.motion.voc_reset_scale = 0.0
    cfg.commands.motion.voc_decay_steps = 0
    cfg.curriculum = None
    cfg.terminations.timeout = None
    cfg.terminations.hand_wrist_away = None
    cfg.terminations.object_pos_error = None
    cfg.terminations.object_quat_error = None
    cfg.terminations.robot_state_diverged = None
    apply_scene_config(cfg, scene_config)
    # Replay is a renderer, not a second dynamics rollout. With drives and collisions
    # disabled below, zero gravity prevents the manipulated object from drifting during
    # the five 100 Hz substeps between a teleport and camera capture.
    cfg.sim.gravity = (0.0, 0.0, 0.0)
    if args_cli.no_dr:
        for name in list(vars(cfg.events)):
            if name.startswith("visual_"):
                setattr(cfg.events, name, None)
    return cfg


def main() -> None:
    """Replay every selected episode and write semantic HDF5 demos."""
    contract = load_embodiment_contract(args_cli.contract)
    if contract != VEGA_SHARPA_JOINT:
        raise ValueError(
            f"Vega replay task requires embodiment contract {VEGA_SHARPA_JOINT.contract_id!r}"
        )
    task_profile = load_task_profile(args_cli.task_profile)
    scene_config = SceneConfig.from_motion_file(args_cli.motion_file)
    object_names = tuple(
        scene_object.name for scene_object in scene_config.scene_objects
    )
    export = load_replay_export(
        args_cli.export_dir,
        contract,
        object_names,
        source_success_only=args_cli.save_on == "source_success",
    )
    episodes = list(export.episodes)
    eligible_episode_count = len(episodes)
    if args_cli.num_demos is not None:
        episodes = episodes[: args_cli.num_demos]
    print(
        "[FILTER] "
        f"input={export.total_episode_count} "
        f"source_successful={export.source_successful_episode_count} "
        f"source_unsuccessful={export.source_unsuccessful_episode_count} "
        f"eligible={eligible_episode_count} "
        f"selected={len(episodes)} "
        f"mode={args_cli.save_on}"
    )
    if not episodes:
        raise RuntimeError("no episodes selected for replay")

    if not scene_config.scene_objects:
        raise ValueError("Vega GR00T replay requires at least one rigid scene object")
    cfg = _configure_replay_cfg(scene_config)
    env = gym.make(contract.record_task, cfg=cfg, render_mode=None)
    raw_env = env.unwrapped
    raw_env.reset()

    robot = raw_env.scene["robot"]
    sim_names = list(robot.joint_names)
    joint_ids = torch.tensor(
        [sim_names.index(name) for name in contract.joint_names],
        dtype=torch.long,
        device=raw_env.device,
    )
    unsupported = [
        name for name in object_names if name not in raw_env.scene.rigid_objects
    ]
    if unsupported:
        raise NotImplementedError(
            f"semantic replay supports rigid scene objects only; unsupported objects={unsupported}"
        )
    scene_objects = [raw_env.scene.rigid_objects[name] for name in object_names]
    _disable_replay_dynamics(raw_env, robot)

    output_dir = Path(args_cli.record_output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.h5"
    num_envs = raw_env.num_envs
    origins = raw_env.scene.env_origins
    zero_action = torch.zeros(
        num_envs, raw_env.action_manager.total_action_dim, device=raw_env.device
    )

    cursors: list[list[int] | None] = [None] * num_envs
    next_episode = 0
    for slot in range(num_envs):
        if next_episode < len(episodes):
            cursors[slot] = [next_episode, 0]
            next_episode += 1

    def write_frame(slot: int, episode: ReplayEpisode, frame: int) -> None:
        all_joint_pos = robot.data.joint_pos.clone()
        all_joint_pos[slot, joint_ids] = torch.as_tensor(
            episode.joint_pos[frame], device=raw_env.device
        )
        env_ids = torch.tensor([slot], device=raw_env.device)
        robot.write_joint_state_to_sim(
            all_joint_pos[env_ids],
            torch.zeros_like(all_joint_pos[env_ids]),
            env_ids=env_ids,
        )
        robot.set_joint_position_target(all_joint_pos[env_ids], env_ids=env_ids)

        for object_index, scene_object in enumerate(scene_objects):
            pose = torch.as_tensor(
                episode.object_pose[frame, object_index],
                dtype=torch.float32,
                device=raw_env.device,
            ).clone()
            pose[:3] += origins[slot]
            scene_object.write_root_pose_to_sim(pose.unsqueeze(0), env_ids=env_ids)
            scene_object.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=raw_env.device), env_ids=env_ids
            )

    def render_frame() -> dict[str, torch.Tensor]:
        """Render observations without applying an action or advancing dynamics."""
        # A normal env.step() processes the record task's residual action before
        # observations are computed. Even a zero action therefore pulls the robot
        # toward the motion reference and corrupts the teleported rollout state.
        # Forward propagates the just-written articulation/object poses without a
        # physics step; scene.update marks camera buffers stale, and render refreshes
        # them before the observation manager reads the record group.
        raw_env.sim.forward()
        raw_env.scene.update(raw_env.step_dt)
        if "interval" in raw_env.event_manager.available_modes:
            raw_env.event_manager.apply(mode="interval", dt=raw_env.step_dt)
        raw_env.sim.render()
        return raw_env.observation_manager.compute_group("record")

    for _ in range(args_cli.visual_warmup_steps):
        for slot, cursor in enumerate(cursors):
            if cursor is not None:
                write_frame(slot, episodes[cursor[0]], 0)
        raw_env.step(zero_action)

    buffers: list[_ReplayBuffer] = [
        {"observations": {}, "actions": []} for _ in range(num_envs)
    ]
    saved = 0
    max_joint_error = 0.0
    max_object_error = 0.0
    with h5py.File(output_path, "w") as h5:
        data_group = h5.create_group("data")
        data_group.attrs["source"] = "external_rollout_replay"
        data_group.attrs["fps"] = contract.fps
        data_group.attrs["motion_file"] = args_cli.motion_file
        data_group.attrs["manifest"] = json.dumps(export.manifest, sort_keys=True)
        data_group.attrs["schema_version"] = 1
        data_group.attrs["embodiment_contract"] = contract.contract_id
        data_group.attrs["embodiment_contract_sha256"] = contract.sha256
        data_group.attrs["task_profile"] = task_profile.task_id
        data_group.attrs["task_profile_sha256"] = task_profile.sha256
        data_group.attrs["selection_policy"] = args_cli.save_on
        data_group.attrs["source_episode_count"] = export.total_episode_count
        data_group.attrs["source_successful_episode_count"] = (
            export.source_successful_episode_count
        )
        data_group.attrs["source_unsuccessful_episode_count"] = (
            export.source_unsuccessful_episode_count
        )
        data_group.attrs["selected_episode_count"] = len(episodes)

        while any(cursor is not None for cursor in cursors):
            for slot, cursor in enumerate(cursors):
                if cursor is not None:
                    write_frame(slot, episodes[cursor[0]], cursor[1])

            observations = render_frame()

            for slot, cursor in enumerate(cursors):
                if cursor is None:
                    continue
                episode_index, frame = cursor
                episode = episodes[episode_index]
                buffer = buffers[slot]
                for name, value in observations.items():
                    buffer["observations"].setdefault(name, []).append(
                        _as_record_array(name, value, slot)
                    )
                buffer["actions"].append(episode.action_target[frame].copy())

                if args_cli.verify:
                    achieved = (
                        robot.data.joint_pos[slot, joint_ids].detach().cpu().numpy()
                    )
                    max_joint_error = max(
                        max_joint_error,
                        float(np.max(np.abs(achieved - episode.joint_pos[frame]))),
                    )
                    for object_index, scene_object in enumerate(scene_objects):
                        object_position = (
                            (scene_object.data.root_pos_w[slot] - origins[slot])
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        max_object_error = max(
                            max_object_error,
                            float(
                                np.max(
                                    np.abs(
                                        object_position
                                        - episode.object_pose[frame, object_index, :3]
                                    )
                                )
                            ),
                        )

                cursor[1] += 1
                if cursor[1] < len(episode.joint_pos):
                    continue

                _write_demo(
                    data_group,
                    saved,
                    buffer["observations"],
                    buffer["actions"],
                    episode.path.name,
                    episode.source_success,
                )
                saved += 1
                buffers[slot] = {"observations": {}, "actions": []}
                if next_episode < len(episodes):
                    cursors[slot] = [next_episode, 0]
                    next_episode += 1
                else:
                    cursors[slot] = None

        data_group.attrs["num_demos"] = saved
        data_group.attrs["total"] = sum(len(episode.joint_pos) for episode in episodes)

    env.close()
    print(f"[INFO] wrote {saved} demos to {output_path}")
    if args_cli.verify:
        print(
            f"[VERIFY] max joint error={max_joint_error:.3e} rad, max object error={max_object_error:.3e} m"
        )
        if max_joint_error >= _VERIFY_JOINT_TOL:
            raise RuntimeError(
                f"replay joint error {max_joint_error:.3e} exceeds {_VERIFY_JOINT_TOL:.1e}"
            )
        if max_object_error >= _VERIFY_OBJECT_TOL:
            raise RuntimeError(
                f"replay object error {max_object_error:.3e} exceeds {_VERIFY_OBJECT_TOL:.1e}"
            )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
