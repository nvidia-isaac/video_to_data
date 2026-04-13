# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Play an environment without loading a policy checkpoint.

Supports Hydra overrides for env_cfg fields, e.g.:
    python scripts/rsl_rl/play.py --task Sharpa-V2P-v0-Play \
        env_cfg.motion_file=arctic_processed/arctic_s01_ketchup_use_01/sharpa_wave

Usage:
    # Play mode with sinusoidal actions
    python scripts/rsl_rl/play.py --task Sharpa-V2P-v0-Play --num_envs 4

    # View mode: zero VOC, zero actions
    python scripts/rsl_rl/play.py --task Sharpa-V2P-v0-Play --num_envs 1 --view

    # Override motion file
    python scripts/rsl_rl/play.py --task Sharpa-V2P-v0-Play \
        env_cfg.motion_file=arctic_processed/arctic_s01_ketchup_use_01/sharpa_wave
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play an IsaacLab environment without a policy checkpoint."
)
parser.add_argument("--task", type=str, required=True, help="Gym task ID to load.")
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record a video."
)
parser.add_argument(
    "--video_length", type=int, default=400, help="Video length (steps)."
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric."
)
parser.add_argument(
    "--real-time", action="store_true", default=False, help="Run close to real-time."
)
parser.add_argument(
    "--zero-actions",
    action="store_true",
    default=False,
    help="Use zero actions instead of sinusoidal (useful for GUI-controlled envs).",
)
parser.add_argument(
    "--view",
    action="store_true",
    default=False,
    help="View mode: zero VOC, zero actions.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import robotic_grounding.tasks  # noqa: F401, E402
from robotic_grounding.tasks.scene_utils import (
    SceneConfig,
    apply_scene_config,
)  # noqa: E402


def prepare_env_for_playing(
    env_cfg: ManagerBasedRLEnvCfg, view: bool = False
) -> ManagerBasedRLEnvCfg:
    """Prepare environment for interactive playing."""
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        env_cfg.curriculum = None

    if view:
        if hasattr(env_cfg, "actions") and hasattr(
            env_cfg.actions, "virtual_rigid_object_control"
        ):
            env_cfg.actions.virtual_rigid_object_control = None
        env_cfg.terminations = None

    return env_cfg


def generate_sinusoidal_actions(
    timestep: int, num_envs: int, action_dim: int, dt: float
) -> np.ndarray:
    time_elapsed = timestep * dt
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)
    for i in range(action_dim):
        frequency = 0.3 + (i % 5) * 0.1
        phase_offset = i * np.pi / action_dim
        actions[:, i] = 0.5 * np.sin(
            2 * np.pi * frequency * time_elapsed + phase_offset
        )
    return actions


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg=None):
    is_debug_env = "Debug" in args_cli.task

    # Apply scene config from motion_file (after Hydra overrides are merged)
    if hasattr(env_cfg, "motion_file"):
        scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
        apply_scene_config(env_cfg, scene_config)

    # Apply non-hydra CLI overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    if hasattr(env_cfg, "eval"):
        env_cfg.eval()

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg = prepare_env_for_playing(env_cfg, view=args_cli.view)

    render_mode = "rgb_array" if args_cli.video else None
    env = ManagerBasedRLEnv(env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_dir = os.path.abspath(os.path.join("logs", "videos", "play"))
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.reset()

    dt = env.unwrapped.step_dt
    timestep = 0
    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.action_manager.total_action_dim

    use_zero_actions = args_cli.zero_actions or is_debug_env or args_cli.view

    print(f"[INFO] Environment loaded: {args_cli.task}")
    print(f"[INFO] Number of environments: {num_envs}")
    print(f"[INFO] Action dimension: {action_dim}")
    print(f"[INFO] Control timestep: {dt:.4f}s ({1.0 / dt:.1f} Hz)")

    while simulation_app.is_running():
        start = time.time()

        if use_zero_actions:
            actions = torch.zeros(
                num_envs, action_dim, dtype=torch.float32, device=env.unwrapped.device
            )
        else:
            actions_np = generate_sinusoidal_actions(timestep, num_envs, action_dim, dt)
            actions = torch.as_tensor(
                actions_np, dtype=torch.float32, device=env.unwrapped.device
            )

        with torch.inference_mode():
            obs, _, _, _, _ = env.step(actions)

        timestep += 1

        if args_cli.video and timestep == args_cli.video_length:
            break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
