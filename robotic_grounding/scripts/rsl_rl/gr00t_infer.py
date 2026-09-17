# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a GR00T policy as a closed-loop controller in an IsaacLab environment."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Closed-loop GR00T inference in IsaacLab.")
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--motion_file", default=None)
parser.add_argument("--require_partitioned_motion", action="store_true")
parser.add_argument("--require_support_surface", action="store_true")
parser.add_argument("--expected_sequence_id", default="")
parser.add_argument("--expected_robot_name", default="")
parser.add_argument("--checkpoint_path", default="")
parser.add_argument("--checkpoint_manifest_sha256", default="")
parser.add_argument("--model_seed", type=int, default=0)
parser.add_argument("--gr00t_revision", default="unknown")
parser.add_argument("--gr00t_dirty", choices=("true", "false"), default="false")
parser.add_argument("--contract", required=True)
parser.add_argument("--task_profile", required=True)
parser.add_argument("--gr00t_host", default="localhost")
parser.add_argument("--gr00t_port", type=int, default=5555)
parser.add_argument("--execution_length", type=int, default=4)
parser.add_argument("--num_steps", type=int, default=390)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=0,
    help=("If positive, evaluate this many completed episodes using the task profile."),
)
parser.add_argument(
    "--eval_results",
    default="",
    help="JSON output path for --eval_episodes results.",
)
parser.add_argument(
    "--eval_episode_horizon",
    type=int,
    default=0,
    help=(
        "If positive, replace the trajectory-end timeout with a wall-clock timeout "
        "at this many control steps. The command remains clamped to its final frame."
    ),
)
parser.add_argument(
    "--visual_mode",
    choices=("training", "off"),
    default="training",
    help=(
        "Evaluation visuals: 'training' keeps the recording texture/light pools and "
        "draws one condition per episode; 'off' disables all visual event terms."
    ),
)
parser.add_argument("--visual_warmup_steps", type=int, default=60)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=390)
parser.add_argument("--save_traj", default="out/gr00t_infer/trajectory.npz")
parser.add_argument("--save_cam_video", default="out/gr00t_infer/three_camera.mp4")
parser.add_argument(
    "--save_success_videos_dir",
    default="",
    help=(
        "If set during --eval_episodes, save only task-successful episode videos to this directory."
    ),
)
parser.add_argument(
    "--success_video_camera",
    default="",
    help="Contract video key to retain; defaults to the contract's first camera.",
)
parser.add_argument("--max_success_videos", type=int, default=3)
parser.add_argument("--use_primitive_urdfs", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim launches."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import isaaclab.envs.mdp as il_mdp
import isaaclab_tasks  # noqa: F401
import numpy as np
import robotic_grounding.tasks  # noqa: F401
import torch
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from robotic_grounding.rendering.dr.controller import (
    disable_visual_event_terms,
    set_visual_event_mode,
)
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from groot_finetune.closed_loop.action_adapter import Gr00tActionAdapter  # noqa: E402
from groot_finetune.closed_loop.embodiment import get_profile  # noqa: E402
from groot_finetune.closed_loop.evaluation import (  # noqa: E402
    LiftHoldResult,
    LiftHoldTracker,
)
from groot_finetune.closed_loop.policy_client import PolicyClient  # noqa: E402
from groot_finetune.contracts import load_embodiment_contract  # noqa: E402
from groot_finetune.source_rollout import termination_reasons_for_env  # noqa: E402
from groot_finetune.task_profile import load_task_profile  # noqa: E402

_TRACKING_COMMAND = "dual_hands_object_tracking_command"


def _tracking_command_cfg(env_cfg):
    """Return the floating-hand or whole-body tracking command config."""
    commands = getattr(env_cfg, "commands", None)
    if commands is None:
        return None
    return getattr(commands, _TRACKING_COMMAND, None) or getattr(
        commands, "motion", None
    )


def _configure_eval(env_cfg) -> None:
    """Use frame-zero, finger-open, unassisted resets and explicit visual behavior."""
    if args_cli.eval_episode_horizon < 0:
        raise ValueError("--eval_episode_horizon must be non-negative")
    if args_cli.eval_episode_horizon > 0:
        control_dt = float(env_cfg.decimation) * float(env_cfg.sim.dt)
        env_cfg.episode_length_s = args_cli.eval_episode_horizon * control_dt
        env_cfg.terminations.timeout = DoneTerm(func=il_mdp.time_out, time_out=True)
        print(
            "[INFO] Evaluation timeout override: "
            f"wall_clock_steps={args_cli.eval_episode_horizon}, "
            f"episode_length_s={env_cfg.episode_length_s:.6f}; "
            "the tracking command will hold its terminal reference frame"
        )
    cmd = _tracking_command_cfg(env_cfg)
    if cmd is not None:
        if hasattr(cmd, "always_reset_to_first_frame"):
            cmd.always_reset_to_first_frame = True
        if hasattr(cmd, "initial_virtual_object_control_curriculum_scale"):
            cmd.initial_virtual_object_control_curriculum_scale = 0.0
        if hasattr(cmd, "debug_vis"):
            cmd.debug_vis = False
        if hasattr(cmd, "reset_freeze_steps"):
            cmd.reset_freeze_steps = 0
        if hasattr(cmd, "reset_finger_openness"):
            cmd.reset_finger_openness = 0.0
        if hasattr(cmd, "virtual_object_control_decay_steps"):
            cmd.virtual_object_control_decay_steps = 0
        if hasattr(cmd, "reset_to_first_frame_prob"):
            cmd.reset_to_first_frame_prob = 1.0
        if hasattr(cmd, "voc_reset_scale"):
            cmd.voc_reset_scale = 0.0
        if hasattr(cmd, "voc_decay_steps"):
            cmd.voc_decay_steps = 0
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = None
    events = getattr(env_cfg, "events", None)
    if args_cli.visual_mode == "off":
        changed = disable_visual_event_terms(events)
        print(f"[INFO] Disabled visual event terms: {changed}")
    else:
        changed = set_visual_event_mode(events, "reset", exclude_suffixes=("_startup",))
        print(f"[INFO] Training-matched per-episode visual event terms: {changed}")
    env_cfg.num_rerenders_on_reset = max(
        1, int(getattr(env_cfg, "num_rerenders_on_reset", 0))
    )


def _eval_control_contract(raw_env) -> dict[str, float | int | bool | str | None]:
    """Validate and report that closed-loop evaluation has no hidden assistance."""
    command = None
    for name in (_TRACKING_COMMAND, "motion"):
        try:
            command = raw_env.command_manager.get_term(name)
        except (KeyError, ValueError):
            continue
        break
    if command is None:
        raise RuntimeError("Closed-loop evaluation requires a tracking command")

    cfg = command.cfg
    contract = {
        "command_debug_vis": bool(getattr(cfg, "debug_vis", False)),
        "reset_freeze_steps": getattr(cfg, "reset_freeze_steps", None),
        "voc_initial_scale": getattr(
            cfg, "initial_virtual_object_control_curriculum_scale", None
        ),
        "voc_reset_scale": getattr(cfg, "voc_reset_scale", None),
        "voc_decay_steps": getattr(
            cfg,
            "voc_decay_steps",
            getattr(cfg, "virtual_object_control_decay_steps", None),
        ),
        "curriculum_enabled": raw_env.cfg.curriculum is not None,
    }
    for key, attribute in (
        ("voc_runtime_target_scale", "virtual_object_controller_scale_factor"),
        ("voc_runtime_env_scale", "virtual_object_controller_scale_factor_per_env"),
    ):
        value = getattr(command, attribute, None)
        contract[key] = (
            float(value.detach().abs().max().cpu().item())
            if value is not None
            else None
        )
    nonzero = {
        key: value
        for key, value in contract.items()
        if key != "curriculum_enabled" and value not in (None, False, 0, 0.0)
    }
    if nonzero or contract["curriculum_enabled"]:
        raise RuntimeError(
            f"Closed-loop evaluation must be marker-free, freeze-free, and VOC-free; resolved contract={contract}"
        )
    return contract


def _warmup(
    env,
    raw_env,
    obs: dict[str, dict[str, torch.Tensor]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, int | str]]:
    """Warm only the renderer, leaving simulation and command state untouched."""
    if not args_cli.visual_warmup_steps:
        return obs, {"mode": "disabled", "render_ticks": 0}

    print(f"[INFO] Renderer warmup: {args_cli.visual_warmup_steps} ticks")
    with torch.inference_mode():
        for _ in range(args_cli.visual_warmup_steps):
            raw_env.sim.render()
        obs = env.get_observations()

    contract = {"mode": "render_only", "render_ticks": args_cli.visual_warmup_steps}
    print(
        f"[INFO] Warmup initialization contract: {json.dumps(contract, sort_keys=True)}"
    )
    return obs, contract


def _fingerprint_path(path: str | Path) -> dict[str, str | int | list[str]]:
    """Return a deterministic SHA-256 fingerprint for one file or directory tree."""
    resolved = Path(path).resolve()
    if resolved.is_file():
        files = [resolved]
        root = resolved.parent
    elif resolved.is_dir():
        files = sorted(
            candidate for candidate in resolved.rglob("*") if candidate.is_file()
        )
        root = resolved
    else:
        raise FileNotFoundError(
            f"Cannot fingerprint missing evaluation artifact: {resolved}"
        )
    if not files:
        raise RuntimeError(f"Evaluation artifact contains no files: {resolved}")

    digest = hashlib.sha256()
    total_bytes = 0
    relative_names = []
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        relative_names.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                total_bytes += len(chunk)
                digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": total_bytes,
        "files": relative_names,
    }


def _validate_camera_observations(
    record_obs: dict[str, torch.Tensor],
    profile,
) -> dict[str, float]:
    """Reject an all-black first policy observation and report camera means."""
    means: dict[str, float] = {}
    black = []
    for field in profile.video_fields:
        name = field.source_term
        if name not in record_obs:
            raise RuntimeError(f"Required camera observation is missing: {name!r}")
        value = record_obs[name]
        means[name] = float(value.float().mean().item())
        if not bool(torch.any(value != 0)):
            black.append(name)
    if black:
        raise RuntimeError(
            f"First post-warmup policy observation is all black for cameras {black}"
        )
    print(f"[INFO] First policy camera means: {means}")
    return means


def _object_z(record_obs: dict[str, torch.Tensor], object_index: int) -> np.ndarray:
    """Return one selected object's base-relative Z coordinate per environment."""
    value = record_obs.get("object_position_e")
    column = object_index * 3 + 2
    if value is None or value.ndim != 2 or value.shape[1] <= column:
        raise RuntimeError(
            "Lift-and-hold evaluation requires record.object_position_e to contain "
            f"object index {object_index}; got "
            f"{None if value is None else tuple(value.shape)}"
        )
    return value[:, column].detach().cpu().numpy().astype(np.float64, copy=True)


def _uint8_image(value: torch.Tensor) -> np.ndarray:
    arr = value.detach().cpu().numpy()[0, ..., :3]
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def _uint8_image_batch(value: torch.Tensor) -> np.ndarray:
    """Convert a batched camera observation to RGB uint8 without dropping environments."""
    arr = value.detach().cpu().numpy()[..., :3]
    if arr.dtype != np.uint8:
        arr = (arr.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return arr


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    """Connect to the server and execute a closed-loop rollout."""
    contract = load_embodiment_contract(args_cli.contract)
    task_profile = load_task_profile(args_cli.task_profile)
    profile = get_profile(contract.contract_id)
    success_video_camera = args_cli.success_video_camera or contract.cameras[0].key
    if args_cli.task != contract.inference_task:
        raise ValueError(
            f"closed-loop task does not match the embodiment contract: {args_cli.task!r} != {contract.inference_task!r}"
        )
    profile_state_layout = tuple(
        (field.key, field.source_term, field.start, field.end)
        for field in profile.state_fields
    )
    contract_state_layout = tuple(
        (field.key, field.source_term, field.start, field.end)
        for field in contract.state_fields
    )
    if profile_state_layout != contract_state_layout:
        raise ValueError("closed-loop state profile does not match the contract")
    profile_action_layout = tuple(
        (field.key, field.source_term, field.start, field.end)
        for field in profile.action_fields
    )
    contract_action_layout = tuple(
        (field.key, field.source_term, field.start, field.end)
        for field in contract.action_fields
    )
    if profile_action_layout != contract_action_layout:
        raise ValueError("closed-loop action profile does not match the contract")
    profile_video_layout = tuple(
        (field.key, field.source_term) for field in profile.video_fields
    )
    contract_video_layout = tuple(
        (camera.key, camera.observation_term) for camera in contract.cameras
    )
    if profile_video_layout != contract_video_layout:
        raise ValueError("closed-loop camera profile does not match the contract")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.viewer.env_index = 0
    if args_cli.motion_file is not None:
        env_cfg.motion_file = args_cli.motion_file
    scene_provenance: dict[str, object] = {}
    target_object_index: int | None = None
    target_object_name: str | None = None
    if getattr(env_cfg, "motion_file", None) is not None:
        scene_cfg = SceneConfig.from_motion_file(env_cfg.motion_file)
        object_names = tuple(obj.name for obj in scene_cfg.scene_objects)
        target_object_index = task_profile.target_object.resolve(object_names)
        target_object_name = object_names[target_object_index]
        if args_cli.require_partitioned_motion and (
            scene_cfg.sequence_id is None or scene_cfg.robot_name is None
        ):
            raise RuntimeError(
                "Closed-loop evaluation requires an explicit "
                "sequence_id=<id>/robot_name=<robot> motion partition; "
                f"resolved motion={scene_cfg.motion_file}"
            )
        if (
            args_cli.expected_sequence_id
            and scene_cfg.sequence_id != args_cli.expected_sequence_id
        ):
            raise RuntimeError(
                "Resolved motion sequence does not match the evaluation contract: "
                f"expected={args_cli.expected_sequence_id!r}, "
                f"resolved={scene_cfg.sequence_id!r}"
            )
        if (
            args_cli.expected_robot_name
            and scene_cfg.robot_name != args_cli.expected_robot_name
        ):
            raise RuntimeError(
                "Resolved motion robot does not match the evaluation contract: "
                f"expected={args_cli.expected_robot_name!r}, "
                f"resolved={scene_cfg.robot_name!r}"
            )
        support_surfaces = [
            obj for obj in scene_cfg.fixed_objects if obj.name == "support_surface"
        ]
        if args_cli.require_support_surface and len(support_surfaces) != 1:
            raise RuntimeError(
                "Closed-loop evaluation requires exactly one discovered support surface; "
                f"resolved={len(support_surfaces)}, motion={scene_cfg.motion_file}"
            )
        scene_provenance = {
            "sequence_id": scene_cfg.sequence_id,
            "robot_name": scene_cfg.robot_name,
            "motion": _fingerprint_path(scene_cfg.motion_file),
            "support_surface": (
                _fingerprint_path(support_surfaces[0].usd_path)
                if support_surfaces
                else None
            ),
        }
        print(
            f"[INFO] Resolved evaluation scene: {json.dumps(scene_provenance, sort_keys=True)}"
        )
        apply_scene_config(
            env_cfg,
            scene_cfg,
            use_primitive_urdfs=args_cli.use_primitive_urdfs,
        )
    _configure_eval(env_cfg)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    if args_cli.eval_episodes > 0 and target_object_index is None:
        raise RuntimeError("task evaluation requires a resolved scene object")
    client = PolicyClient(host=args_cli.gr00t_host, port=args_cli.gr00t_port)
    if not client.ping():
        client.close()
        raise RuntimeError(
            f"No GR00T server at {args_cli.gr00t_host}:{args_cli.gr00t_port}."
        )
    adapter = Gr00tActionAdapter(
        client,
        profile,
        execution_length=args_cli.execution_length,
        task=task_profile.instruction,
    )
    if adapter.action_chunk_size != contract.action_horizon:
        raise ValueError(
            "served action horizon does not match the embodiment contract: "
            f"{adapter.action_chunk_size} != {contract.action_horizon}"
        )
    print(
        f"[INFO] GR00T server ready: action_keys={adapter.action_keys}, "
        f"chunk={adapter.action_chunk_size}, profile={profile.name}"
    )

    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.video:
        kwargs = {
            "video_folder": "out/gr00t_infer/videos",
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    raw_env = env.unwrapped
    eval_control_contract = _eval_control_contract(raw_env)
    resolved_terminations = set(raw_env.termination_manager.active_terms)
    expected_terminations = set(contract.evaluation_terminations)
    if resolved_terminations != expected_terminations:
        raise RuntimeError(
            "closed-loop environment terminations do not match the embodiment contract: "
            f"resolved={sorted(resolved_terminations)}, expected={sorted(expected_terminations)}"
        )
    resolved_episode_horizon = int(raw_env.max_episode_length)
    if (
        args_cli.eval_episode_horizon > 0
        and resolved_episode_horizon != args_cli.eval_episode_horizon
    ):
        raise RuntimeError(
            "Resolved wall-clock episode horizon does not match the request: "
            f"requested={args_cli.eval_episode_horizon}, "
            f"resolved={resolved_episode_horizon}"
        )
    eval_control_contract.update(
        {
            "episode_timeout_mode": (
                "wall_clock" if args_cli.eval_episode_horizon > 0 else "trajectory_end"
            ),
            "episode_horizon_steps": resolved_episode_horizon,
        }
    )
    print(
        f"[INFO] Evaluation control contract: {json.dumps(eval_control_contract, sort_keys=True)}"
    )

    device = raw_env.device
    obs = env.get_observations()
    obs, initialization_contract = _warmup(env, raw_env, obs)
    first_camera_means = _validate_camera_observations(obs["record"], profile)

    trajectory: dict[str, list[np.ndarray]] = {}
    camera_frames: list[np.ndarray] = []
    dones = None
    eval_completed = 0
    eval_successes = 0
    eval_lengths: list[int] = []
    eval_metric_samples: list[int] = []
    eval_max_lifts_m: list[float] = []
    eval_max_hold_steps: list[int] = []
    current_lengths = np.zeros(args_cli.num_envs, dtype=np.int64)
    termination_counts: Counter[str] = Counter()
    success_video_frames: list[list[np.ndarray]] | None = None
    successful_video_rollouts: list[
        tuple[int, int, LiftHoldResult, list[np.ndarray]]
    ] = []
    success_video_paths: list[str] = []
    lift_hold_tracker = None
    success_config = None
    if args_cli.eval_episodes > 0:
        term_mgr = raw_env.termination_manager
        success_config = task_profile.evaluator
        assert target_object_index is not None
        lift_hold_tracker = LiftHoldTracker(
            _object_z(obs["record"], target_object_index), config=success_config
        )
        print(
            "[INFO] Evaluation reset: frame=0, reset_finger_openness=0.0, "
            f"evaluator={json.dumps(success_config.as_dict(), sort_keys=True)}, "
            f"terminations={term_mgr.active_terms}"
        )
        if args_cli.save_success_videos_dir:
            if args_cli.max_success_videos <= 0:
                raise ValueError("--max_success_videos must be positive")
            camera_terms = {
                field.key: field.source_term for field in profile.video_fields
            }
            if success_video_camera not in camera_terms:
                raise ValueError(
                    f"unknown success-video camera {success_video_camera!r}; available={sorted(camera_terms)}"
                )
            camera_term = camera_terms[success_video_camera]
            if camera_term not in obs["record"]:
                raise RuntimeError(
                    f"Success video camera {camera_term!r} is not in the record observations"
                )
            success_video_frames = [[] for _ in range(args_cli.num_envs)]
            print(
                f"[INFO] Capturing up to {args_cli.max_success_videos} successful {success_video_camera} videos"
            )
    elif args_cli.save_success_videos_dir:
        raise ValueError("--save_success_videos_dir requires --eval_episodes")
    try:
        for step in range(args_cli.num_steps):
            with torch.inference_mode():
                if args_cli.eval_episodes > 0:
                    assert lift_hold_tracker is not None
                    assert target_object_index is not None
                    # Metrics are sampled from the policy observation before its action.
                    # Automatic-reset observations returned by the previous step are the
                    # baseline for a new episode, never a terminal sample for the old one.
                    lift_hold_tracker.update(
                        _object_z(obs["record"], target_object_index)
                    )
                action_np = adapter.act(obs["record"], dones=dones)
                if args_cli.save_traj:
                    for name, value in obs["record"].items():
                        if value.ndim <= 2:
                            trajectory.setdefault(name, []).append(
                                value.detach().cpu().numpy().copy()
                            )
                    trajectory.setdefault("actions", []).append(action_np.copy())
                camera_source_terms = tuple(
                    field.source_term for field in profile.video_fields
                )
                if args_cli.save_cam_video and all(
                    name in obs["record"] for name in camera_source_terms
                ):
                    camera_frames.append(
                        np.concatenate(
                            [
                                _uint8_image(obs["record"][name])
                                for name in camera_source_terms
                            ],
                            axis=1,
                        )
                    )
                if success_video_frames is not None:
                    camera_term = next(
                        field.source_term
                        for field in profile.video_fields
                        if field.key == success_video_camera
                    )
                    image_batch = _uint8_image_batch(obs["record"][camera_term])
                    for env_idx, image in enumerate(image_batch):
                        success_video_frames[env_idx].append(image.copy())
                actions = torch.as_tensor(action_np, dtype=torch.float32, device=device)
                obs, _, dones, _ = env.step(actions)
                if args_cli.eval_episodes > 0:
                    assert lift_hold_tracker is not None
                    current_lengths += 1
                    done_indices = (
                        torch.nonzero(dones, as_tuple=False).flatten().cpu().tolist()
                    )
                    for env_idx in done_indices:
                        if eval_completed >= args_cli.eval_episodes:
                            break
                        episode_index = eval_completed
                        assert target_object_index is not None
                        result = lift_hold_tracker.complete(
                            env_idx,
                            next_initial_z=float(
                                _object_z(obs["record"], target_object_index)[env_idx]
                            ),
                        )
                        eval_completed += 1
                        eval_successes += int(result.success)
                        eval_lengths.append(int(current_lengths[env_idx]))
                        eval_metric_samples.append(result.sample_count)
                        eval_max_lifts_m.append(result.max_lift_m)
                        eval_max_hold_steps.append(result.max_hold_steps)
                        termination_counts.update(
                            termination_reasons_for_env(
                                raw_env.termination_manager, env_idx
                            )
                        )
                        current_lengths[env_idx] = 0
                        if success_video_frames is not None:
                            frames = success_video_frames[env_idx]
                            if (
                                result.success
                                and len(successful_video_rollouts)
                                < args_cli.max_success_videos
                            ):
                                successful_video_rollouts.append(
                                    (
                                        episode_index,
                                        env_idx,
                                        result,
                                        frames,
                                    )
                                )
                            success_video_frames[env_idx] = []
            if (step + 1) % 20 == 0:
                if args_cli.eval_episodes > 0:
                    print(
                        f"[INFO] Evaluation step {step + 1}/{args_cli.num_steps}: "
                        f"completed={eval_completed}/{args_cli.eval_episodes}, "
                        f"successes={eval_successes}"
                    )
                else:
                    print(f"[INFO] Closed-loop step {step + 1}/{args_cli.num_steps}")
            if args_cli.eval_episodes > 0 and eval_completed >= args_cli.eval_episodes:
                break
    finally:
        client.close()
        env.close()

    if args_cli.eval_episodes > 0:
        if eval_completed != args_cli.eval_episodes:
            raise RuntimeError(
                f"Only {eval_completed}/{args_cli.eval_episodes} evaluation episodes "
                f"completed within {args_cli.num_steps} environment steps."
            )
        assert success_config is not None
        eval_result = {
            "schema_version": 1,
            "task": args_cli.task,
            "task_profile": task_profile.task_id,
            "task_profile_sha256": task_profile.sha256,
            "embodiment_contract": contract.contract_id,
            "embodiment_contract_sha256": contract.sha256,
            "target_object": target_object_name,
            "seed": args_cli.seed,
            "model_seed": args_cli.model_seed,
            "checkpoint_path": args_cli.checkpoint_path,
            "checkpoint_manifest_sha256": args_cli.checkpoint_manifest_sha256,
            "gr00t_revision": args_cli.gr00t_revision,
            "gr00t_dirty": args_cli.gr00t_dirty == "true",
            "action_keys": list(adapter.action_keys),
            "model_action_horizon": int(adapter.action_chunk_size),
            "scene": scene_provenance,
            "initialization": initialization_contract,
            "num_envs": args_cli.num_envs,
            "requested_episodes": args_cli.eval_episodes,
            "completed_episodes": eval_completed,
            "successful_episodes": eval_successes,
            "unsuccessful_episodes": eval_completed - eval_successes,
            "success_rate": eval_successes / eval_completed,
            "success_evaluator": success_config.evaluator_id,
            "metric_sample_timing": "pre_action_excluding_terminal_post_action_state",
            "success_evaluator_config": success_config.as_dict(),
            "termination_reasons": dict(termination_counts),
            "episode_lengths": eval_lengths,
            "metric_sample_counts": eval_metric_samples,
            "max_object_lift_m": eval_max_lifts_m,
            "max_hold_steps": eval_max_hold_steps,
            "reset_frame": 0,
            "reset_finger_openness": 0.0,
            **eval_control_contract,
            "visual_mode": args_cli.visual_mode,
            "visual_warmup_steps": args_cli.visual_warmup_steps,
            "first_policy_camera_means": first_camera_means,
            "execution_length": args_cli.execution_length,
        }
        if args_cli.save_success_videos_dir:
            video_dir = Path(args_cli.save_success_videos_dir)
            video_dir.mkdir(parents=True, exist_ok=True)
            for episode_index, env_idx, result, frames in successful_video_rollouts:
                path = video_dir / (
                    f"episode_{episode_index:03d}_env_{env_idx:02d}_"
                    f"{success_video_camera}_lift_hold_"
                    f"lift_{result.max_lift_m:.3f}m.mp4"
                )
                imageio.mimsave(path, frames, fps=contract.fps, codec="libx264")
                success_video_paths.append(str(path))
                print(
                    f"[INFO] Saved successful episode video ({len(frames)} frames): {path}"
                )
            eval_result["successful_episode_videos"] = success_video_paths
        print(
            f"[SUMMARY] completed={eval_completed}, successes={eval_successes}, "
            f"failures={eval_completed - eval_successes}, "
            f"success_rate={eval_result['success_rate']:.2%}, "
            f"reasons={dict(termination_counts)}"
        )
        if args_cli.eval_results:
            path = Path(args_cli.eval_results)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(eval_result, indent=2, sort_keys=True) + "\n")
            print(f"[INFO] Saved evaluation results: {path}")

    if args_cli.save_traj:
        path = Path(args_cli.save_traj)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.stack(v) for k, v in trajectory.items()})
        print(f"[INFO] Saved trajectory: {path}")
    if args_cli.save_cam_video and camera_frames:
        path = Path(args_cli.save_cam_video)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(path, camera_frames, fps=contract.fps, codec="libx264")
        print(f"[INFO] Saved three-camera video: {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
