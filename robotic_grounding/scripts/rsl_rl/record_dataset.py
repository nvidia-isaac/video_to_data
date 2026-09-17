# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roll out a trained RSL-RL policy and record a dataset via Isaac Lab's RecorderManager.

By default the output is **LeRobot v3 format** (a directory).  Pass
``--output_format hdf5`` to keep the raw HDF5 instead.

Use a task whose env cfg adds a camera + recorders, e.g. ``Sharpa-V2D-Record-v0``.

Example (LeRobot, default):
  python scripts/rsl_rl/record_dataset.py --task Sharpa-V2D-Record-v0 \\
      --checkpoint logs/rsl_rl/<exp>/<run>/model_<n>.pt \\
      --motion_file arctic_processed/arctic_s01_box_grab_01/sharpa_wave \\
      --num_episodes 64 --num_envs 16 \\
      --output_file datasets/arctic_s01_box_grab_01.hdf5

Example (HDF5 only):
  python scripts/rsl_rl/record_dataset.py ... --output_format hdf5
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Record an HDF5 dataset from an RSL-RL policy rollout."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Sharpa-V2D-Record-v0", help="Name of the task."
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--scene_config",
    type=str,
    default=None,
    help="Path to the scene configuration file.",
)
parser.add_argument(
    "--motion_file", type=str, default=None, help="Motion file to load."
)
parser.add_argument(
    "--wandb_id", type=str, default=None, help="Wandb run ID to resume from."
)
parser.add_argument(
    "--use_primitive_urdfs",
    action="store_true",
    default=False,
    help="Use primitive URDFs for the robot.",
)
# dataset / recording options
parser.add_argument(
    "--output_file",
    type=str,
    default="datasets/dataset.hdf5",
    help="Output HDF5 dataset path (dir is created if missing).",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=32,
    help="Number of completed episodes to record before exiting.",
)
parser.add_argument(
    "--export_mode",
    type=str,
    default="all",
    choices=["all", "succeeded"],
    help="Export all episodes or only episodes flagged successful.",
)
parser.add_argument(
    "--output_format",
    type=str,
    default="lerobot",
    choices=["lerobot", "hdf5"],
    help="Output format: 'lerobot' (default, LeRobot v3 directory) or 'hdf5' (raw HDF5).",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Control frequency (Hz) used for LeRobot timestamps and video encoding.",
)
parser.add_argument(
    "--task_name",
    type=str,
    default="robot manipulation",
    help="Natural-language task description written to the LeRobot dataset.",
)
parser.add_argument(
    "--keep_hdf5",
    action="store_true",
    default=False,
    help="Keep the intermediate HDF5 file after LeRobot conversion (default: delete it).",
)
parser.add_argument(
    "--camera_width", type=int, default=None, help="Override camera width."
)
parser.add_argument(
    "--camera_height", type=int, default=None, help="Override camera height."
)
parser.add_argument(
    "--debug_world_axes",
    action="store_true",
    default=False,
    help="DEBUG: render RGB=XYZ world-frame axes at each env origin to help tune camera "
    "poses. Bakes geometry into the RGB/segmentation; capped at 4 episodes.",
)
parser.add_argument(
    "--no_failure_terminations",
    action="store_true",
    default=False,
    help="Disable BOTH failure terminations (hand_wrist_away + object_away) so episodes run "
    "the full horizon (time_out only) even after divergence — for probing whether the policy "
    "recovers. NOTE: makes completion_ratio meaningless (episodes always reach horizon).",
)
parser.add_argument(
    "--disable_terminations",
    nargs="*",
    default=[],
    metavar="TERM",
    help="Disable specific termination terms by name (e.g. hand_wrist_away_from_trajectory). "
    "Keeps the others. Episodes then end on the remaining terms (e.g. object_away / time_out).",
)
parser.add_argument(
    "--replay_motion",
    action="store_true",
    default=False,
    help="Drive the hands by KINEMATIC replay of the reference wrist+finger trajectory "
    "instead of a policy (no checkpoint needed). Pair with --voc_scale 1.0 to let the "
    "virtual object controller carry the object. For validating trajectories/cameras.",
)
parser.add_argument(
    "--voc_scale",
    type=float,
    default=0.0,
    help="Virtual-object-control curriculum scale during recording (0.0 = off, the "
    "deterministic-eval default; 1.0 = object fully driven toward its reference).",
)
parser.add_argument(
    "--voc_decay_steps",
    type=int,
    default=0,
    help="VOC curriculum decay steps = the warmup window where the object is assisted "
    "before the policy is on its own (env-default eval uses 20). Also used as the warmup "
    "subtracted in the completion-ratio metric. 0 = no warmup (pure deterministic).",
)
parser.add_argument(
    "--domain_randomization",
    action="store_true",
    default=False,
    help="Randomize object materials (color/roughness/metallic), support surface material, "
    "and scene lighting per episode.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Guard: the world-axes overlay is a debug aid and contaminates the RGB/segmentation, so
# refuse to bake it into anything but a tiny inspection run. Fails fast before app launch.
if args_cli.debug_world_axes and args_cli.num_episodes > 4:
    raise SystemExit(
        f"--debug_world_axes is for debugging only; refusing to record "
        f"{args_cli.num_episodes} episodes (max 4). Drop the flag for real generation."
    )

# --- visual-DR collision guard (keep as one block; PR #66 also edits this file) -------
# The imperative SceneMaterialRandomizer and the EventTerm visual DR both bind OmniPBR
# materials via rep.functional.create_batch.material() on overlapping prims. Whichever
# binds last wins the USD binding, and the loser's modify.attribute() writes land on
# unbound materials with no error. The two bind at different times (one after gym.make,
# the other lazily mid-rollout), so the result looks nondeterministic. Refuse rather than
# silently produce a dataset with half its randomization missing.
if args_cli.domain_randomization and "-DR-" in args_cli.task:
    raise SystemExit(
        f"--domain_randomization (imperative material DR) conflicts with the EventTerm "
        f"visual DR built into '{args_cli.task}': both rebind OmniPBR materials on the "
        f"same prims. Use one or the other -- drop the flag to keep the task's visual "
        f"DR, or use the non-DR task id to use the flag."
    )
# --------------------------------------------------------------------------------------

# recording always needs the camera rendering pipeline
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import sys
import torch
from download_from_wandb import download_run

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import robotic_grounding.tasks  # noqa: F401
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from robotic_grounding.tasks.v2d.config.sharpa_wave.recording.sharpa_v2d_record_env_cfg import (
    add_world_axes,
    aim_camera_at_scene,
)
from robotic_grounding.tasks.scene_utils.replay_kinematics import (
    DualHandReplay,
    build_joint_reorder,
    disable_gravity_in_articulation_cfg,
    write_dual_hand_frame_per_env,
)
from viewer_utils import autoframe_viewer
from domain_randomization import SceneMaterialRandomizer

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

_EXPORT_MODES = {
    "all": DatasetExportMode.EXPORT_ALL,
    "succeeded": DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
}


def _report_completion_metrics(hdf5_path: str, horizon: int, warmup: int = 0) -> None:
    """Compute from-frame-0 completion metrics, store them per episode, print the aggregate.

    For each recorded episode: ``completion_ratio = clamp((ep_len - warmup) /
    (horizon - warmup), 0, 1)`` where ``ep_len`` is the episode's recorded sample count and
    ``horizon`` is the reference trajectory length (command ``retargeted_horizon``);
    ``full_completion = completion_ratio >= 0.99``. This mirrors the Pass-A (reset-to-first-
    frame) logic of the removed ``eval_callback.py``. Assumes a frame-0 rollout with
    ``virtual_object_control_decay_steps == 0`` (warmup 0), which ``record_dataset`` enforces.

    Writes ``completion_ratio`` + ``full_completion`` attrs onto each ``data/<demo>`` group
    and prints ``completion_ratio_mean`` + ``full_completion_pct`` over all episodes.
    """
    import h5py

    denom = max(horizon - warmup, 1)
    ratios = []
    with h5py.File(hdf5_path, "a") as f:
        if "data" not in f:
            return
        for _name, ep in f["data"].items():
            ep_len = int(ep.attrs.get("num_samples", 0))
            ratio = min(max(ep_len - warmup, 0), denom) / denom
            ep.attrs["completion_ratio"] = float(ratio)
            ep.attrs["full_completion"] = bool(ratio >= 0.99)
            ratios.append(ratio)
    if not ratios:
        return
    mean_ratio = sum(ratios) / len(ratios)
    full_pct = 100.0 * sum(1 for r in ratios if r >= 0.99) / len(ratios)
    print(
        f"[completion] from-frame-0 over {len(ratios)} episode(s) (horizon={horizon}): "
        f"completion_ratio_mean={mean_ratio:.4f}  full_completion_pct={full_pct:.1f}%",
        flush=True,
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """Record an HDF5 dataset from a trained policy."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Record", "").replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    if args_cli.camera_width is not None:
        env_cfg.camera_width = args_cli.camera_width
    if args_cli.camera_height is not None:
        env_cfg.camera_height = args_cli.camera_height
    # The camera cfgs were already built in __post_init__ at the default resolution, so the
    # attribute overrides above don't reach them — push the new resolution onto each recorded
    # camera cfg directly (else the override is a silent no-op).
    if args_cli.camera_width is not None or args_cli.camera_height is not None:
        for _cam in getattr(env_cfg.recorders.record_camera, "sensor_names", []):
            _cam_cfg = getattr(env_cfg.scene, _cam, None)
            if _cam_cfg is not None:
                if args_cli.camera_width is not None:
                    _cam_cfg.width = args_cli.camera_width
                if args_cli.camera_height is not None:
                    _cam_cfg.height = args_cli.camera_height

    # Apply scene config: motion_file (Hydra override) > --scene_config YAML > env default.
    env_cfg.motion_file = args_cli.motion_file
    scene_config = None
    if getattr(env_cfg, "motion_file", None) is not None:
        scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
        apply_scene_config(
            env_cfg, scene_config, use_primitive_urdfs=args_cli.use_primitive_urdfs
        )
        autoframe_viewer(env_cfg, scene_config.motion_file)
    elif args_cli.scene_config is not None:
        env_cfg.scene_config_path = args_cli.scene_config
        scene_config = SceneConfig.from_yaml(args_cli.scene_config)
        apply_scene_config(
            env_cfg, scene_config, use_primitive_urdfs=args_cli.use_primitive_urdfs
        )
        autoframe_viewer(env_cfg, scene_config.motion_file)

    # Aim every recorded camera at the scene's mean object position. Bakes the look-at
    # into each camera's cfg offset before env creation, so it holds across episode resets.
    if scene_config is not None:
        for cam_name in env_cfg.recorders.record_camera.sensor_names:
            target = aim_camera_at_scene(env_cfg, scene_config, sensor_name=cam_name)
            print(
                f"[INFO] {cam_name} aimed at scene object mean (env-relative): {target}"
            )

    # DEBUG: overlay world-frame axes (RGB=XYZ) at each env origin to aid camera tuning.
    if args_cli.debug_world_axes:
        add_world_axes(env_cfg)
        print("[INFO] DEBUG world axes enabled (RGB=XYZ at env origin).")

    # Deterministic eval-style rollout: reset to first frame. VOC scale is configurable
    # (default 0.0 = off; 1.0 = object driven toward its reference, e.g. for replay mode).
    cmd = getattr(
        getattr(env_cfg, "commands", None), "dual_hands_object_tracking_command", None
    )
    if cmd is not None:
        cmd.always_reset_to_first_frame = True
        cmd.initial_virtual_object_control_curriculum_scale = args_cli.voc_scale
        cmd.virtual_object_control_decay_steps = args_cli.voc_decay_steps
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None

    # Optionally disable termination terms (probe recovery / relax early kills).
    _to_disable = set(args_cli.disable_terminations or [])
    if args_cli.no_failure_terminations:
        _to_disable |= {
            "hand_wrist_away_from_trajectory",
            "object_away_from_trajectory",
        }
    if _to_disable and hasattr(env_cfg, "terminations"):
        for _term in sorted(_to_disable):
            if getattr(env_cfg.terminations, _term, None) is not None:
                setattr(env_cfg.terminations, _term, None)
        print(f"[INFO] Disabled terminations: {sorted(_to_disable)}")

    # Replay mode: the hands are teleported each step, so disable gravity on the hand
    # articulations (the object stays physics/VOC-driven and is NOT teleported).
    if args_cli.replay_motion:
        for _rn in ("left_robot", "right_robot"):
            _rc = getattr(env_cfg.scene, _rn, None)
            if _rc is not None:
                setattr(env_cfg.scene, _rn, disable_gravity_in_articulation_cfg(_rc))

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # Configure the RecorderManager dataset export.
    output_file = os.path.abspath(args_cli.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not hasattr(env_cfg, "recorders"):
        raise RuntimeError(
            f"Task '{args_cli.task}' has no 'recorders' cfg. Use a recording task "
            "(e.g. Sharpa-V2D-Record-v0) whose env cfg adds a camera + RecorderManager."
        )
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(output_file)
    env_cfg.recorders.dataset_filename = os.path.basename(output_file)
    env_cfg.recorders.dataset_export_mode = _EXPORT_MODES[args_cli.export_mode]
    env_cfg.recorders.export_in_record_pre_reset = True

    # Resolve the checkpoint (skipped in replay mode — no policy needed).
    resume_path = None
    if not args_cli.replay_motion:
        log_root_path = os.path.abspath(
            os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        )
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        elif args_cli.wandb_id is not None:
            resume_path = download_run(args_cli.wandb_id)
            agent_cfg.load_checkpoint = resume_path
        else:
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        env_cfg.log_dir = os.path.dirname(resume_path)
    else:
        env_cfg.log_dir = os.path.dirname(output_file)

    # Create env (render_mode rgb_array so the camera pipeline renders).
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Reference trajectory length, for from-frame-0 completion metrics (reported per episode).
    completion_horizon = 0
    try:
        _cmd_term = env.unwrapped.command_manager.get_term(
            "dual_hands_object_tracking_command"
        )
        completion_horizon = int(getattr(_cmd_term, "retargeted_horizon", 0))
    except Exception:
        completion_horizon = 0

    if args_cli.replay_motion:
        # --- Kinematic wrist-replay rollout (no policy); object carried by VOC. ---
        if scene_config is None:
            raise RuntimeError(
                "--replay_motion requires --motion_file (no scene config loaded)."
            )
        base_env = env.unwrapped
        device = base_env.device
        num_envs = base_env.num_envs
        right_robot = base_env.scene["right_robot"]
        left_robot = base_env.scene["left_robot"]
        right_names = list(right_robot.joint_names)
        left_names = list(left_robot.joint_names)
        traj = DualHandReplay(scene_config.motion_file, device)
        right_reorder = build_joint_reorder(traj.right_joint_names, right_names)
        left_reorder = build_joint_reorder(traj.left_joint_names, left_names)
        zero_action = torch.zeros(
            num_envs, base_env.action_manager.total_action_dim, device=device
        )
        env_origins = base_env.scene.env_origins
        print(
            f"[INFO] REPLAY: {traj.num_frames} frames @ {traj.fps:.0f}fps, VOC={args_cli.voc_scale}; "
            f"recording up to {args_cli.num_episodes} episodes ({num_envs} envs) -> {output_file}"
        )
        base_env.reset()
        dr = (
            SceneMaterialRandomizer(base_env, scene_config)
            if args_cli.domain_randomization
            else None
        )
        frame_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        completed = 0
        while simulation_app.is_running() and completed < args_cli.num_episodes:
            # Teleport both hands to the per-env reference frame, then step so the
            # RecorderManager captures cameras/state and the VOC advances the object.
            fi = torch.clamp(frame_idx, max=traj.num_frames - 1)
            with torch.inference_mode():
                # Teleport writes to sim-state buffers (inference tensors), so they must
                # happen inside inference_mode, same as the step.
                write_dual_hand_frame_per_env(
                    right_robot,
                    left_robot,
                    traj,
                    fi,
                    env_origins,
                    right_reorder,
                    left_reorder,
                    right_names,
                    left_names,
                    device,
                )
                _, _, terminated, truncated, _ = base_env.step(zero_action)
            dones = terminated | truncated
            frame_idx = frame_idx + 1
            frame_idx[dones] = 0  # restart replay where the env auto-resets
            num_done = int(dones.sum().item())
            if num_done:
                if dr is not None:
                    dr.randomize(dones)
                completed += num_done
                print(
                    f"[INFO] recorded {completed}/{args_cli.num_episodes} episodes",
                    flush=True,
                )
        print(f"[INFO] Done. Exported episodes -> {output_file}")
        env.close()
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Load policy.
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        base_env = env.unwrapped
        print(
            f"[INFO] Recording up to {args_cli.num_episodes} episodes "
            f"({base_env.num_envs} envs) -> {output_file}"
        )

        dr = (
            SceneMaterialRandomizer(base_env, scene_config)
            if args_cli.domain_randomization and scene_config is not None
            else None
        )

        # Camera poses (intrinsics + look-at orientation) are baked into the env cfg offsets,
        # so they hold across resets — no runtime re-aim needed.
        obs = env.get_observations()

        completed = 0
        while simulation_app.is_running() and completed < args_cli.num_episodes:
            with torch.inference_mode():
                actions = policy(obs)
                # Stepping triggers the RecorderManager hooks; done envs auto-reset
                # and their episodes are exported to HDF5 inside step().
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
            num_done = int(dones.sum().item())
            if num_done:
                if dr is not None:
                    dr.randomize(dones)
                completed += num_done
                print(
                    f"[INFO] recorded {completed}/{args_cli.num_episodes} episodes",
                    flush=True,
                )

        print(f"[INFO] Done. Exported episodes -> {output_file}")
        env.close()

    # From-frame-0 completion metrics: written per-episode into the HDF5 + aggregate printed.
    # (Runs before any LeRobot conversion so the intermediate HDF5 still exists.)
    if completion_horizon > 0 and os.path.exists(output_file):
        _report_completion_metrics(
            output_file, completion_horizon, warmup=args_cli.voc_decay_steps
        )

    # ------------------------------------------------------------------
    # Post-process: convert HDF5 to LeRobot v3 format (default).
    # ------------------------------------------------------------------
    if args_cli.output_format == "lerobot":
        # Derive the LeRobot output directory from the HDF5 path
        # e.g. datasets/foo.hdf5  ->  datasets/foo/
        lerobot_dir = output_file
        if lerobot_dir.endswith(".hdf5"):
            lerobot_dir = lerobot_dir[:-5]

        # Add export_lerobot.py's directory to path so it's importable here
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)

        from export_lerobot import hdf5_to_lerobot

        print(f"[INFO] Converting HDF5 -> LeRobot v3 @ {lerobot_dir}")
        hdf5_to_lerobot(
            hdf5_path=output_file,
            output_dir=lerobot_dir,
            fps=args_cli.fps,
            task_name=args_cli.task_name,
        )

        if not args_cli.keep_hdf5:
            os.remove(output_file)
            print(f"[INFO] Removed intermediate HDF5 (pass --keep_hdf5 to retain).")

        print(f"[INFO] LeRobot dataset ready at: {lerobot_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
