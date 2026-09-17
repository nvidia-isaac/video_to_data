# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visually-augmented demo dataset with BIT-IDENTICAL state/action trajectories.

Two phases in one Isaac session:

1. **Source rollout**: run the policy closed-loop (deterministic, frame-0 reset, all
   randomization off) until one episode reaches the motion horizon (= success),
   caching per step the ``record`` obs group, the absolute action targets, and the
   raw sim states (robot root poses + joint positions, rigid-object poses).
2. **Re-render**: for each demo, re-randomize every ``visual_*`` DR term (dome +
   distant light, ground/object/hand/table textures), then kinematically replay the
   cached sim states frame by frame (teleport + render, no policy, no managers) and
   capture every camera image in the ``record`` observation group.

Every demo shares the source episode's state/action arrays verbatim — the dataset is
N visually-distinct copies of one trajectory, exactly identical by construction. That
isolates visual variation from trajectory variation, unlike the live recording path
(``record_dataset.py``), where an episode's visuals change mid-rollout and each episode
also follows a different trajectory.

Output schema is ``data/demo_i/obs/<term>`` + ``data/demo_i/actions`` — named groups,
written directly here rather than through IsaacLab's RecorderManager (which writes a
flat ``obs`` vector plus ``camera/<sensor>/<type>``).

The source action and camera mappings come from the task's explicit GR00T recording
contract.
Successful horizons are detected from termination metadata rather than term names.

Example (inside the container, headless):
    HEADLESS=1 python scripts/rsl_rl/rerender_demo_visuals.py \
        --task Sharpa-V2D-Gr00t-Record-v0 \
        --contract sharpa_dual_hand_three_camera \
        --task_profile /path/to/task_profile.json \
        --checkpoint <path>/model_19999.pt \
        --motion_file ego_recon/processed/sequence_id=<sequence>/robot_name=<robot> \
        --num_demos 100 --record_output out/dr_rerender
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from typing import Any

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Re-render one successful demo under N random backgrounds."
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Task name (must have a camera + 'record' observation group).",
)
parser.add_argument("--contract", required=True)
parser.add_argument("--task_profile", required=True)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument(
    "--motion_file", type=str, default=None, help="Motion file to load (scene source)."
)
parser.add_argument(
    "--record_output",
    type=str,
    default="out/groot_bgdr_rerender",
    help="Directory for the output data.h5.",
)
parser.add_argument("--num_demos", type=int, default=100, help="Total demos to write.")
parser.add_argument(
    "--max_source_attempts",
    type=int,
    default=10,
    help="Give up if no successful source episode within this many episodes.",
)
parser.add_argument(
    "--random_source_start",
    action="store_true",
    default=False,
    help="Sample source-episode start frames using the fixed environment seed instead "
    "of forcing frame 0. Use this for policies trained with random-start curricula. "
    "The accepted episode and all visual rerenders remain deterministic.",
)
parser.add_argument(
    "--min_source_steps",
    type=int,
    default=1,
    help="Reject successful random-start source episodes shorter than this many steps.",
)
parser.add_argument(
    "--settle_render_steps",
    type=int,
    default=32,
    help="Render ticks after each background randomization before capturing frame 0. "
    "Must cover the renderer's auto-exposure adaptation (~30 frames), which otherwise "
    "brightens the first captured frames after a dome swap.",
)
parser.add_argument(
    "--visual_dr_include",
    type=str,
    default=None,
    help="Comma-separated substrings selecting WHICH visual DR terms are randomized "
    "per demo (matched against the visual_* event term names, e.g. "
    "'dome_light,ground_texture' for a background-only study). Non-matching visual "
    "terms (object/hand/table textures, key light) are disabled entirely. "
    "Default: all visual terms.",
)
parser.add_argument(
    "--use_primitive_urdfs",
    action="store_true",
    default=False,
    help="Use primitive URDFs for the robot.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import h5py
import isaaclab_tasks  # noqa: F401
import numpy as np
import robotic_grounding.tasks  # noqa: F401
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from robotic_grounding.rendering.dr.isaaclab_events import prewarm_textures
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from rsl_rl.runners import OnPolicyRunner

from groot_finetune.contracts import load_embodiment_contract
from groot_finetune.recording_contract import (
    RecordingContract,
    contract_from_env_cfg,
    joint_reorder_indices,
)
from groot_finetune.source_rollout import (
    configure_timeout_only_terminations,
    termination_reasons_for_env,
)
from groot_finetune.task_profile import load_task_profile

_TRACKING_COMMAND = "dual_hands_object_tracking_command"


def _tracking_command_cfg(env_cfg):
    """Return the floating-hand or whole-body tracking command config."""
    commands = getattr(env_cfg, "commands", None)
    if commands is None:
        return None
    return getattr(commands, _TRACKING_COMMAND, None) or getattr(
        commands, "motion", None
    )


def _tracking_command_term(command_manager):
    """Return the task's sole supported trajectory-tracking command term."""
    names = [
        name
        for name in (_TRACKING_COMMAND, "motion")
        if name in command_manager.active_terms
    ]
    if len(names) != 1:
        raise RuntimeError(
            "Task must expose exactly one supported tracking command "
            f"({_TRACKING_COMMAND!r} or 'motion'); got {names or 'none'} from "
            f"{command_manager.active_terms}"
        )
    return command_manager.get_term(names[0])


def _tracking_timestep(command) -> torch.Tensor:
    """Return the per-environment source-frame counter for either task family."""
    for attr in ("timestep", "timestep_counter"):
        value = getattr(command, attr, None)
        if value is not None:
            return value
    raise RuntimeError(
        f"Tracking command {type(command).__name__} exposes neither 'timestep' nor 'timestep_counter'"
    )


def _processed_record_action(
    action_manager, contract: RecordingContract
) -> torch.Tensor:
    """Concatenate processed targets and optionally reorder joint columns by name."""
    terms = [action_manager.get_term(name) for name in contract.action_terms]
    actions = torch.cat([term.processed_actions for term in terms], dim=-1)
    if contract.action_joint_order is None:
        return actions

    source_joint_names: list[str] = []
    for name, term in zip(contract.action_terms, terms, strict=True):
        joint_names = list(getattr(term, "_joint_names", ()))
        if not joint_names:
            raise ValueError(
                f"GR00T action term {name!r} does not expose resolved joint names; "
                "cannot enforce the canonical joint order"
            )
        if len(joint_names) != term.processed_actions.shape[-1]:
            raise ValueError(
                f"GR00T action term {name!r} has {len(joint_names)} joint names but "
                f"{term.processed_actions.shape[-1]} processed columns"
            )
        source_joint_names.extend(joint_names)
    indices = joint_reorder_indices(source_joint_names, contract.action_joint_order)
    return actions[:, indices]


def _apply_deterministic_defaults(env_cfg) -> None:
    """Frame-0 reset, VOC off, no curriculum, no reset/startup randomization.

    Mirrors ``record.py``'s ``_apply_record_defaults`` + ``_apply_identical_episodes``
    (kept local: record.py launches the app at import time, so it cannot be imported).
    """
    cmd = _tracking_command_cfg(env_cfg)
    if cmd is not None:
        if hasattr(cmd, "always_reset_to_first_frame"):
            cmd.always_reset_to_first_frame = not args_cli.random_source_start
        if hasattr(cmd, "reset_to_first_frame_prob"):
            cmd.reset_to_first_frame_prob = 0.0 if args_cli.random_source_start else 1.0
        if hasattr(cmd, "initial_virtual_object_control_curriculum_scale"):
            cmd.initial_virtual_object_control_curriculum_scale = 0.0
        if hasattr(cmd, "reset_finger_openness"):
            cmd.reset_finger_openness = 0.0
        if hasattr(cmd, "voc_reset_scale"):
            cmd.voc_reset_scale = 0.0
        if hasattr(cmd, "voc_decay_steps"):
            cmd.voc_decay_steps = 0
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None
    env_cfg.num_rerenders_on_reset = max(
        1, int(getattr(env_cfg, "num_rerenders_on_reset", 0))
    )
    for term_name in ("right_physics_material", "left_physics_material"):
        term = getattr(env_cfg.events, term_name, None)
        if term is not None:
            term.params["static_friction_range"] = (2.0, 2.0)
            term.params["dynamic_friction_range"] = (2.0, 2.0)
    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is not None:
            group.enable_corruption = False
    # Swap the selected visual DR terms to RESET mode: they then fire only on env
    # resets (none happen during the phase-2 kinematic replay), the manager still
    # instantiates the class-based texture terms (needed for the explicit per-demo
    # invocation in _randomize_visuals), and mid-rollout flips are gone. Do NOT null
    # an INCLUDED texture term: un-instantiated, it has no material binding to write
    # to. Terms excluded by --visual_dr_include are never invoked, so nulling them
    # (interval + startup twins) is safe and keeps their visuals at spawn defaults.
    include = (
        [tok.strip() for tok in args_cli.visual_dr_include.split(",") if tok.strip()]
        if args_cli.visual_dr_include
        else None
    )
    for name, term in list(vars(env_cfg.events).items()):
        if not name.startswith("visual_") or not isinstance(term, EventTerm):
            continue
        base = name[: -len("_startup")] if name.endswith("_startup") else name
        included = include is None or any(tok in base for tok in include)
        if not included:
            setattr(env_cfg.events, name, None)
        elif not name.endswith("_startup"):
            setattr(
                env_cfg.events,
                name,
                EventTerm(func=term.func, mode="reset", params=term.params),
            )


def _visual_term_cfgs(env) -> list:
    """The instantiated reset-mode ``visual_*`` event term cfgs, via the event manager."""
    manager = env.event_manager
    names = [
        n for n in manager.active_terms.get("reset", []) if n.startswith("visual_")
    ]
    return [manager.get_term_cfg(n) for n in names]


def _randomize_visuals(env, num_render_ticks: int) -> None:
    """Invoke every visual DR term (lights + all textures) and let replicator settle.

    Only the ``visual_*`` terms are invoked - applying the whole reset/startup event
    mode would replay PHYSICS randomization (materials, gains, mass) mid-replay.
    """
    for cfg in _visual_term_cfgs(env):
        cfg.func(env, None, **cfg.params)
    for _ in range(num_render_ticks):
        env.sim.render()


def _capture_rgb(env, sensor_name: str) -> np.ndarray:
    """Grab ``sensor_name``'s RGB frame for env 0 as uint8 HxWx3."""
    rgb = env.scene[sensor_name].data.output["rgb"][0, ..., :3]
    arr = rgb.detach().cpu().numpy()
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    """Roll out one successful episode, then re-render it under random backgrounds."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.env_index = 0

    embodiment_contract = load_embodiment_contract(args_cli.contract)
    task_profile = load_task_profile(args_cli.task_profile)
    if args_cli.task != embodiment_contract.record_task:
        raise ValueError(
            "rerender task does not match the embodiment contract: "
            f"{args_cli.task!r} != {embodiment_contract.record_task!r}"
        )
    recording_contract = contract_from_env_cfg(env_cfg)
    expected_cameras = tuple(
        (camera.observation_term, camera.sensor_name)
        for camera in embodiment_contract.cameras
    )
    if recording_contract.action_terms != embodiment_contract.source_action_terms:
        raise ValueError(
            "record environment action terms do not match the embodiment contract"
        )
    if recording_contract.camera_terms != expected_cameras:
        raise ValueError(
            "record environment cameras do not match the embodiment contract"
        )

    # Validate the declared sensor mapping before the first capture.
    image_term_to_sensor = dict(recording_contract.camera_terms)
    installed = sorted(k for k in vars(env_cfg.scene) if "cam" in k.lower())
    missing_sensors = [
        sensor
        for sensor in image_term_to_sensor.values()
        if not hasattr(env_cfg.scene, sensor)
    ]
    if missing_sensors:
        raise SystemExit(
            f"Task '{args_cli.task}' is missing camera sensors {missing_sensors}. "
            f"Installed cameras: {installed or 'none'}."
        )

    if args_cli.motion_file is not None:
        env_cfg.motion_file = args_cli.motion_file
    if getattr(env_cfg, "motion_file", None) is not None:
        scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
        apply_scene_config(
            env_cfg, scene_config, use_primitive_urdfs=args_cli.use_primitive_urdfs
        )

    _apply_deterministic_defaults(env_cfg)
    configure_timeout_only_terminations(env_cfg.terminations)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required.")
    resume_path = retrieve_file_path(args_cli.checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=raw_env.device)

    # Sim entities to snapshot/teleport: both hands + every rigid object.
    robots = {
        name: raw_env.scene.articulations[name] for name in raw_env.scene.articulations
    }
    rigid_objects = {
        name: raw_env.scene.rigid_objects[name] for name in raw_env.scene.rigid_objects
    }
    print(f"[INFO]: articulations={list(robots)}, rigid_objects={list(rigid_objects)}")

    term_mgr = raw_env.termination_manager
    active_terminations = set(term_mgr.active_terms)
    expected_terminations = set(embodiment_contract.source_terminations)
    if active_terminations != expected_terminations:
        raise RuntimeError(
            "record environment terminations do not match the embodiment contract: "
            f"active={sorted(active_terminations)}, expected={sorted(expected_terminations)}"
        )
    timeout_terms = [
        name
        for name in term_mgr.active_terms
        if bool(getattr(term_mgr.get_term_cfg(name), "time_out", False))
    ]
    if len(timeout_terms) != 1:
        raise RuntimeError(
            f"Task must expose exactly one time_out=True termination; got {timeout_terms}"
        )
    timeout_name = timeout_terms[0]
    action_manager = raw_env.action_manager
    tracking_command = _tracking_command_term(raw_env.command_manager)

    # ---------------- Phase 1: closed-loop source rollout until success ----------------
    obs = env.get_observations()
    episode: list[dict[str, Any]] | None = None
    buffer: list[dict[str, Any]] = []
    attempts = 0
    while simulation_app.is_running() and episode is None:
        with torch.inference_mode():
            step_data: dict[str, Any] = {
                "record": {k: v[0].detach().clone() for k, v in obs["record"].items()},
                "source_timestep": int(_tracking_timestep(tracking_command)[0].item()),
                "states": {
                    **{
                        f"art/{n}": (
                            a.data.root_state_w[0, :7].clone(),
                            a.data.joint_pos[0].clone(),
                        )
                        for n, a in robots.items()
                    },
                    **{
                        f"obj/{n}": o.data.root_state_w[0, :7].clone()
                        for n, o in rigid_objects.items()
                    },
                },
            }
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            step_data["action"] = (
                _processed_record_action(action_manager, recording_contract)[0]
                .detach()
                .clone()
            )
            buffer.append(step_data)

            if bool(dones[0]):
                fired = list(termination_reasons_for_env(term_mgr, 0))
                timed_out = timeout_name in fired
                long_enough = len(buffer) >= args_cli.min_source_steps
                if timed_out and long_enough:
                    episode = buffer
                    print(
                        f"[INFO]: source episode succeeded: {len(episode)} steps "
                        f"(motion frames {episode[0]['source_timestep']}.."
                        f"{episode[-1]['source_timestep']})"
                    )
                else:
                    attempts += 1
                    if timed_out and not long_enough:
                        fired.append(
                            f"short_source_episode={len(buffer)}<{args_cli.min_source_steps}"
                        )
                    print(
                        f"[INFO]: source attempt {attempts} failed ({fired}); retrying"
                    )
                    if attempts >= args_cli.max_source_attempts:
                        raise RuntimeError(
                            f"No successful source episode in {attempts} attempts."
                        )
                    buffer = []

    if episode is None:
        raise RuntimeError("Simulation stopped before a source episode completed")

    num_steps = len(episode)
    term_names = list(episode[0]["record"].keys())
    image_terms = [n for n in term_names if "image" in n]
    unmapped_image_terms = [n for n in image_terms if n not in image_term_to_sensor]
    if unmapped_image_terms:
        raise SystemExit(
            f"No scene-sensor mapping for record image terms {unmapped_image_terms}; "
            f"known mappings: {image_term_to_sensor}."
        )
    image_term_to_sensor = {term: image_term_to_sensor[term] for term in image_terms}
    print(f"[INFO]: re-render camera mapping: {image_term_to_sensor}")
    sim_dt = raw_env.physics_dt

    # Stacked source arrays shared by every demo.
    source_obs = {
        name: np.stack([s["record"][name].cpu().numpy() for s in episode], axis=0)
        for name in term_names
    }
    for image_term in image_terms:
        if source_obs[image_term].dtype in (np.float32, np.float64):
            source_obs[image_term] = (
                (source_obs[image_term] * 255).clip(0, 255).astype(np.uint8)
            )
    source_actions = np.stack([s["action"].cpu().numpy() for s in episode], axis=0)

    # ---------------- Phase 2: kinematic re-render under fresh backgrounds ----------------
    os.makedirs(args_cli.record_output, exist_ok=True)
    output_path = os.path.join(args_cli.record_output, "data.h5")
    zero6 = torch.zeros(1, 6, device=raw_env.device)

    with h5py.File(output_path, "w") as h5:
        data_group = h5.create_group("data")
        data_group.attrs["schema_version"] = 1
        data_group.attrs["fps"] = embodiment_contract.fps
        data_group.attrs["embodiment_contract"] = embodiment_contract.contract_id
        data_group.attrs["embodiment_contract_sha256"] = embodiment_contract.sha256
        data_group.attrs["task_profile"] = task_profile.task_id
        data_group.attrs["task_profile_sha256"] = task_profile.sha256

        def write_demo(idx: int, images: dict[str, np.ndarray]) -> None:
            grp = data_group.create_group(f"demo_{idx}")
            grp.attrs["num_samples"] = num_steps
            grp.attrs["env_idx"] = 0
            obs_grp = grp.create_group("obs")
            for image_term, frames in images.items():
                obs_grp.create_dataset(image_term, data=frames, compression="gzip")
            for name in term_names:
                if name not in images:
                    obs_grp.create_dataset(name, data=source_obs[name])
            grp.create_dataset("actions", data=source_actions)

        # ALL demos (incl. 0) are re-rendered: the source rollout's own images start on
        # the pre-randomization background (replicator applies startup textures with a
        # few-frame latency), so they are used only for shape and then discarded.
        with torch.inference_mode():
            # Pre-warm the texture pools of the ACTIVE visual terms only (RTX streams
            # texture files asynchronously; the cache is keyed by file path, so
            # cycling them through the already bound ground material warms them for
            # every material). Deriving pools from the instantiated terms respects
            # --visual_dr_include automatically.
            dome_textures: list = []
            material_textures: list = []
            for cfg in _visual_term_cfgs(raw_env):
                params = cfg.params
                if "prim_path_pattern" in params:
                    material_textures += list(params.get("textures") or [])
                elif "texture_paths" in params:
                    material_textures += list(params["texture_paths"])
                elif "textures" in params:  # dome light
                    dome_textures += list(params.get("textures") or [])

            set_material_texture = None
            active_reset = raw_env.event_manager.active_terms.get("reset", [])
            if "visual_ground_texture" in active_reset:
                ground_cfg = raw_env.event_manager.get_term_cfg("visual_ground_texture")

                def _set_ground_texture(tex: str) -> None:
                    ground_cfg.func(
                        raw_env,
                        None,
                        prim_path_pattern=ground_cfg.params["prim_path_pattern"],
                        textures=[tex],
                    )

                set_material_texture = _set_ground_texture

            prewarm_textures(
                raw_env,
                dome_textures=dome_textures,
                material_textures=material_textures,
                set_material_texture=set_material_texture,
                renders_per_texture=4,
            )
            for demo_idx in range(0, args_cli.num_demos):
                _randomize_visuals(raw_env, args_cli.settle_render_steps)
                frames_by_term = {
                    term: np.empty_like(source_obs[term]) for term in image_terms
                }
                for t, step_data in enumerate(episode):
                    for n, art in robots.items():
                        pose, jpos = step_data["states"][f"art/{n}"]
                        art.write_root_pose_to_sim(pose.unsqueeze(0))
                        art.write_root_velocity_to_sim(zero6)
                        art.write_joint_state_to_sim(
                            jpos.unsqueeze(0), torch.zeros_like(jpos).unsqueeze(0)
                        )
                    for n, obj in rigid_objects.items():
                        obj.write_root_pose_to_sim(
                            step_data["states"][f"obj/{n}"].unsqueeze(0)
                        )
                        obj.write_root_velocity_to_sim(zero6)
                    raw_env.scene.write_data_to_sim()
                    raw_env.sim.step(render=True)
                    raw_env.scene.update(dt=sim_dt)
                    for image_term, sensor_name in image_term_to_sensor.items():
                        frames_by_term[image_term][t] = _capture_rgb(
                            raw_env, sensor_name
                        )
                write_demo(demo_idx, frames_by_term)
                if demo_idx % 10 == 0 or demo_idx == args_cli.num_demos - 1:
                    print(
                        f"[INFO]: re-rendered demo {demo_idx + 1}/{args_cli.num_demos}"
                    )

        data_group.attrs["num_demos"] = args_cli.num_demos
        data_group.attrs["total"] = args_cli.num_demos * num_steps
        data_group.attrs["source_start_timestep"] = episode[0]["source_timestep"]
        data_group.attrs["source_end_timestep"] = episode[-1]["source_timestep"]
        data_group.attrs["random_source_start"] = args_cli.random_source_start

    print(
        f"[INFO]: Wrote {args_cli.num_demos} demos x {num_steps} steps "
        f"(identical states/actions, per-demo backgrounds) to {output_path}"
    )
    try:
        from data_recorder import save_sample_grid

        for image_term in image_terms:
            save_sample_grid(
                output_path,
                out_dir=os.path.join(args_cli.record_output, f"samples_{image_term}"),
                image_key=f"obs/{image_term}",
            )
    except ImportError:
        print("[WARN]: data_recorder.py unavailable; skipping sample grids.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
