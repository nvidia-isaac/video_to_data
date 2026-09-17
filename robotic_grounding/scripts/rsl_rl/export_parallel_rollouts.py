# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export camera-free expert rollouts for GR00T post-training."""

import argparse
import sys
from typing import Any

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--motion_file", required=True)
parser.add_argument("--export_dir", required=True)
parser.add_argument(
    "--contract",
    required=True,
    help="Embodiment contract ID or JSON file used to validate exported tensors.",
)
parser.add_argument("--wandb_run", default="")
parser.add_argument(
    "--reset_arm_noise_rad",
    type=float,
    default=0.0,
    help="Uniform +/- frame-zero perturbation for each arm joint (default: off).",
)
parser.add_argument(
    "--reset_finger_noise_rad",
    type=float,
    default=0.0,
    help="Uniform +/- frame-zero perturbation for each finger joint (default: off).",
)
parser.add_argument(
    "--reset_object_xy_noise_m",
    type=float,
    default=0.0,
    help="Uniform +/- frame-zero XY perturbation for tracked scene objects.",
)
parser.add_argument(
    "--reset_object_yaw_noise_rad",
    type=float,
    default=0.0,
    help="Uniform +/- world-yaw perturbation for tracked scene objects.",
)
parser.add_argument("--use_primitive_urdfs", action="store_true")
parser.add_argument(
    "--replace_export",
    action="store_true",
    help="Remove this export's prior episode files and manifest before writing.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The simulator must be running before these imports.
import hashlib  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
import numpy as np  # noqa: E402
import robotic_grounding.tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from robotic_grounding.tasks.scene_utils import (  # noqa: E402
    SceneConfig,
    apply_scene_config,
)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from groot_finetune.contracts import (  # noqa: E402
    EmbodimentContract,
    load_embodiment_contract,
)
from groot_finetune.source_rollout import (  # noqa: E402
    configure_timeout_only_terminations,
    snapshot_applied_action_target,
    source_success_from_terminations,
    summarize_trajectory_diversity,
    termination_reasons_for_env,
    validate_terminal_action_target,
)
from groot_finetune.source_policy import OnnxSourcePolicy  # noqa: E402


def _apply_eval_defaults(env_cfg: Any) -> None:
    """Use frame-zero resets with freeze, markers, VOC, and curriculum disabled."""
    cmd = env_cfg.commands.motion
    if hasattr(cmd, "always_reset_to_first_frame"):
        cmd.always_reset_to_first_frame = True
    if hasattr(cmd, "reset_to_first_frame_prob"):
        cmd.reset_to_first_frame_prob = 1.0
    if hasattr(cmd, "reset_finger_openness"):
        cmd.reset_finger_openness = 0.0
    if hasattr(cmd, "reset_freeze_steps"):
        cmd.reset_freeze_steps = 0
    if hasattr(cmd, "debug_vis"):
        cmd.debug_vis = False
    cmd.initial_virtual_object_control_curriculum_scale = 0.0
    if hasattr(cmd, "voc_reset_scale"):
        cmd.voc_reset_scale = 0.0
    if hasattr(cmd, "voc_decay_steps"):
        cmd.voc_decay_steps = 0
    env_cfg.curriculum = None


def _configure_reset_noise(
    env_cfg: Any, contract: EmbodimentContract
) -> dict[str, float]:
    """Set collection-only event ranges before environment construction."""
    for name in (
        "reset_arm_noise_rad",
        "reset_finger_noise_rad",
        "reset_object_xy_noise_m",
        "reset_object_yaw_noise_rad",
    ):
        if getattr(args_cli, name) < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    params = env_cfg.events.reset_to_trajectory_frame.params
    groups = params["joint_position_noise_groups"]
    configured_groups = {
        name: tuple(group["joint_names"]) for name, group in groups.items()
    }
    expected_groups = {
        name: tuple(joint_names)
        for name, joint_names in contract.reset_joint_groups.items()
    }
    if configured_groups != expected_groups:
        raise ValueError(
            "environment reset joint groups do not match the embodiment contract: "
            f"configured={configured_groups}, expected={expected_groups}"
        )
    if set(groups) != {"arm", "finger"}:
        raise ValueError(
            f"this exporter requires 'arm' and 'finger' reset groups; got {sorted(groups)}"
        )
    groups["arm"]["range"] = (
        -args_cli.reset_arm_noise_rad,
        args_cli.reset_arm_noise_rad,
    )
    groups["finger"]["range"] = (
        -args_cli.reset_finger_noise_rad,
        args_cli.reset_finger_noise_rad,
    )
    params["object_xy_noise_range"] = (
        -args_cli.reset_object_xy_noise_m,
        args_cli.reset_object_xy_noise_m,
    )
    params["object_yaw_noise_range"] = (
        -args_cli.reset_object_yaw_noise_rad,
        args_cli.reset_object_yaw_noise_rad,
    )
    return {
        "arm_rad": args_cli.reset_arm_noise_rad,
        "finger_rad": args_cli.reset_finger_noise_rad,
        "object_xy_m": args_cli.reset_object_xy_noise_m,
        "object_yaw_rad": args_cli.reset_object_yaw_noise_rad,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: Any, agent_cfg: Any) -> None:
    """Run the RL expert once in every environment and export compact NPZ traces."""
    contract = load_embodiment_contract(args_cli.contract)
    if args_cli.task != contract.source_task:
        raise ValueError(
            f"source task does not match the embodiment contract: {args_cli.task!r} != {contract.source_task!r}"
        )
    if not contract.joint_names or set(contract.reset_joint_groups) != {
        "arm",
        "finger",
    }:
        raise ValueError(
            "parallel joint-rollout export requires named joints and arm/finger reset groups in the embodiment contract"
        )
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 6.0
    env_cfg.viewer.env_index = min(env_cfg.viewer.env_index, args_cli.num_envs - 1)
    env_cfg.motion_file = args_cli.motion_file
    scene_config = SceneConfig.from_motion_file(args_cli.motion_file)
    apply_scene_config(
        env_cfg,
        scene_config,
        use_primitive_urdfs=args_cli.use_primitive_urdfs,
    )
    _apply_eval_defaults(env_cfg)
    configured_timeout_name = configure_timeout_only_terminations(env_cfg.terminations)
    reset_noise = _configure_reset_noise(env_cfg, contract)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed

    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required")
    checkpoint_path = Path(retrieve_file_path(args_cli.checkpoint))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    print(f"[INFO] loading checkpoint: {checkpoint_path}")
    print("[INFO] source eligibility: timeout termination")
    print(f"[INFO] reset noise: {json.dumps(reset_noise, sort_keys=True)}")
    if checkpoint_path.suffix.lower() == ".onnx":
        policy_nn = OnnxSourcePolicy.from_checkpoint(
            checkpoint_path, device=raw_env.device
        )
        policy = policy_nn
        checkpoint_format = "onnx"
        policy_runtime = policy_nn.runtime
    else:
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(checkpoint_path))
        policy = runner.get_inference_policy(device=raw_env.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        checkpoint_format = "rsl_rl"
        policy_runtime = "torch"
    print(f"[INFO] source policy runtime: {policy_runtime}")

    robot = raw_env.scene["robot"]
    sim_joint_names = list(robot.joint_names)
    joint_ids = torch.tensor(
        [sim_joint_names.index(name) for name in contract.joint_names],
        device=raw_env.device,
        dtype=torch.long,
    )
    if len(contract.source_action_terms) != 1:
        raise ValueError(
            f"joint-rollout export requires exactly one source action term; got {contract.source_action_terms}"
        )
    action_term_name = contract.source_action_terms[0]
    action_term = raw_env.action_manager.get_term(action_term_name)
    action_names = list(action_term.IO_descriptor.joint_names)
    if not action_names:
        raise RuntimeError(
            f"action term {type(action_term).__name__} does not publish joint names"
        )
    action_ids = torch.tensor(
        [action_names.index(name) for name in contract.joint_names],
        device=raw_env.device,
        dtype=torch.long,
    )
    # Fail before an expensive rollout if this task cannot preserve applied targets
    # across ManagerBasedEnv's terminal auto-reset.
    if not hasattr(action_term, "last_applied_actions"):
        raise RuntimeError(
            f"action term {type(action_term).__name__} has no last_applied_actions buffer"
        )

    object_names = [scene_object.name for scene_object in scene_config.scene_objects]
    if not object_names:
        raise RuntimeError("rollout export requires at least one rigid scene object")
    unsupported = [
        name for name in object_names if name not in raw_env.scene.rigid_objects
    ]
    if unsupported:
        raise NotImplementedError(
            f"rollout export supports rigid scene objects only; unsupported objects={unsupported}"
        )
    objects = [raw_env.scene.rigid_objects[name] for name in object_names]

    term_mgr = raw_env.termination_manager
    timeout_terms = [
        name
        for name in term_mgr.active_terms
        if bool(getattr(term_mgr.get_term_cfg(name), "time_out", False))
    ]
    if len(timeout_terms) != 1:
        raise RuntimeError(f"expected exactly one timeout term, got {timeout_terms}")
    timeout_name = timeout_terms[0]
    if timeout_name != configured_timeout_name:
        raise RuntimeError(
            "constructed source timeout does not match the configured timeout: "
            f"{timeout_name!r} != {configured_timeout_name!r}"
        )
    resolved_terminations = set(term_mgr.active_terms)
    expected_terminations = set(contract.source_terminations)
    if resolved_terminations != expected_terminations:
        raise RuntimeError(
            "source environment terminations do not match the embodiment contract: "
            f"resolved={sorted(resolved_terminations)}, expected={sorted(expected_terminations)}"
        )

    output_dir = Path(args_cli.export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args_cli.replace_export:
        for previous_episode in output_dir.glob("episode_*.npz"):
            previous_episode.unlink()
        (output_dir / "manifest.json").unlink(missing_ok=True)
    buffers: list[dict[str, list[np.ndarray]]] = [
        {"joint_pos": [], "action_target": [], "object_pose": []}
        for _ in range(args_cli.num_envs)
    ]
    active = torch.ones(args_cli.num_envs, dtype=torch.bool, device=raw_env.device)
    source_success_count = 0
    completed_count = 0
    reason_counts: Counter[str] = Counter()
    episode_lengths: list[int] = [0] * args_cli.num_envs
    episode_source_successes: list[bool] = [False] * args_cli.num_envs

    obs = env.get_observations()
    step_count = 0
    while (
        simulation_app.is_running()
        and step_count < args_cli.num_steps
        and bool(active.any())
    ):
        with torch.inference_mode():
            joint_snapshot = (
                robot.data.joint_pos[:, joint_ids].detach().cpu().numpy().copy()
            )
            object_snapshot = np.stack(
                [
                    obj.data.root_state_w[:, :7].detach().cpu().numpy().copy()
                    for obj in objects
                ],
                axis=1,
            )
            object_snapshot[:, :, :3] -= (
                raw_env.scene.env_origins.detach().cpu().numpy()[:, None, :]
            )
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            # This is the absolute target issued to the PD controller, not the raw
            # residual policy action and not reset-cleared processed_actions.
            action_snapshot = snapshot_applied_action_target(action_term, action_ids)

            active_indices = (
                torch.nonzero(active, as_tuple=False).flatten().cpu().tolist()
            )
            for env_idx in active_indices:
                buffer = buffers[env_idx]
                buffer["joint_pos"].append(joint_snapshot[env_idx])
                buffer["action_target"].append(action_snapshot[env_idx])
                buffer["object_pose"].append(object_snapshot[env_idx])

            dones_bool = dones.to(dtype=torch.bool)
            completed = dones_bool & active
            completed_indices = (
                torch.nonzero(completed, as_tuple=False).flatten().cpu().tolist()
            )
            if completed_indices:
                for env_idx in completed_indices:
                    termination_reasons = termination_reasons_for_env(term_mgr, env_idx)
                    terminated = np.asarray(
                        [name in termination_reasons for name in term_mgr.active_terms],
                        dtype=np.bool_,
                    )
                    source_success = source_success_from_terminations(
                        term_mgr.active_terms,
                        terminated,
                        timeout_name,
                    )
                    episode_source_successes[env_idx] = source_success
                    buffer = buffers[env_idx]
                    length = len(buffer["joint_pos"])
                    episode_lengths[env_idx] = length
                    action_targets = np.stack(buffer["action_target"]).astype(
                        np.float32
                    )
                    terminal_action_step_l2 = (
                        validate_terminal_action_target(action_targets)
                        if len(action_targets) >= 2
                        else float("nan")
                    )
                    np.savez_compressed(
                        output_dir / f"episode_{env_idx:06d}.npz",
                        joint_pos=np.stack(buffer["joint_pos"]).astype(np.float32),
                        action_target=action_targets,
                        object_pose=np.stack(buffer["object_pose"]).astype(np.float32),
                        source_success=np.asarray(source_success, dtype=np.bool_),
                        termination_reasons=np.asarray(termination_reasons),
                        terminal_action_step_l2=np.asarray(
                            terminal_action_step_l2, dtype=np.float32
                        ),
                        env_idx=np.asarray(env_idx, dtype=np.int64),
                    )
                    source_success_count += int(source_success)
                    completed_count += 1
                    reason_counts.update(termination_reasons)
                active[completed] = False
                print(
                    f"[TERM] step={step_count} completed_now={len(completed_indices)} "
                    f"total={completed_count}/{args_cli.num_envs} "
                    f"source_successes={source_success_count}"
                )
            policy_nn.reset(dones_bool)

        step_count += 1
        if step_count % 100 == 0:
            print(
                f"[INFO] step={step_count}/{args_cli.num_steps} "
                f"completed={completed_count} source_successes={source_success_count}"
            )

    env.close()
    if completed_count != args_cli.num_envs:
        raise RuntimeError(
            f"only {completed_count}/{args_cli.num_envs} environments completed within {args_cli.num_steps} steps"
        )

    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    successful_indices = [
        env_idx
        for env_idx, source_success in enumerate(episode_source_successes)
        if source_success
    ]
    if successful_indices:
        successful_diversity = summarize_trajectory_diversity(
            np.stack(
                [
                    np.stack(buffers[env_idx]["joint_pos"])
                    for env_idx in successful_indices
                ]
            ),
            np.stack(
                [
                    np.stack(buffers[env_idx]["action_target"])
                    for env_idx in successful_indices
                ]
            ),
            np.stack(
                [
                    np.stack(buffers[env_idx]["object_pose"])
                    for env_idx in successful_indices
                ]
            ),
        )
    else:
        successful_diversity = {"episode_count": 0}
    manifest = {
        "format": "joint_rollout",
        "schema_version": 1,
        "embodiment_contract": contract.contract_id,
        "embodiment_contract_sha256": contract.sha256,
        "fps": float(contract.fps),
        "joint_names": list(contract.joint_names),
        "object_names": object_names,
        "source_task": args_cli.task,
        "motion_file": args_cli.motion_file,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "env_spacing": 6.0,
        "reset_finger_openness": 0.0,
        "reset_freeze_steps": 0,
        "command_debug_vis": False,
        "episode_count": completed_count,
        "source_successful_episode_count": source_success_count,
        "source_unsuccessful_episode_count": completed_count - source_success_count,
        "termination_reasons": dict(reason_counts),
        "timeout_termination": timeout_name,
        "episode_lengths": episode_lengths,
        "source_success_criterion": f"{timeout_name}.time_out=True",
        "action_target_source": (f"{action_term_name}.last_applied_actions"),
        "initial_condition_noise": {
            "distribution": "independent_uniform_symmetric",
            "seed": args_cli.seed,
            **reset_noise,
            "reference_trajectory_perturbed": False,
            "policy_action_noise_added": False,
        },
        "successful_trajectory_diversity": successful_diversity,
        "wandb_run": args_cli.wandb_run,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_format": checkpoint_format,
        "policy_runtime": policy_runtime,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[SUMMARY] completed={completed_count} source_successful={source_success_count} "
        f"source_unsuccessful={completed_count - source_success_count} "
        f"reasons={dict(reason_counts)}"
    )
    print(f"[DIVERSITY] successful={json.dumps(successful_diversity, sort_keys=True)}")
    print(f"[INFO] wrote rollout export to {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
