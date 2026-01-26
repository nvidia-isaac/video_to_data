# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Play an environment without loading a policy checkpoint.

Supports two modes:
- Debug mode (--task Sharpa-V2P-Debug-v0): GUI controls are used, actions are ignored
- Play mode (--task Sharpa-V2P-v0-Play): Sinusoidal actions for environment validation

Usage:
    # Debug mode with GUI control
    python scripts/rsl_rl/play.py --task Sharpa-V2P-Debug-v0

    # Play mode with sinusoidal actions
    python scripts/rsl_rl/play.py --task Sharpa-V2P-v0-Play --num_envs 4

    # Record video
    python scripts/rsl_rl/play.py --task Sharpa-V2P-v0-Play --video --video_length 500
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#

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

# Isaac Sim app args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras for video
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------#
# Launch app
# -----------------------------------------------------------------------------#

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Omniverse/Isaac imports must happen AFTER SimulationApp instantiation
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Import task modules to register gym environments
import robotic_grounding.tasks  # noqa: F401, E402


# -----------------------------------------------------------------------------#
# Helper Functions
# -----------------------------------------------------------------------------#


def prepare_env_for_playing(env_cfg: ManagerBasedRLEnvCfg) -> ManagerBasedRLEnvCfg:
    """Prepare environment for interactive playing (remove training-specific components)."""
    # Remove curriculum if present
    if hasattr(env_cfg, "curriculum") and env_cfg.curriculum is not None:
        env_cfg.curriculum = None

    return env_cfg


def generate_sinusoidal_actions(
    timestep: int, num_envs: int, action_dim: int, dt: float
) -> np.ndarray:
    """Generate smooth sinusoidal trajectory for actions.

    Args:
        timestep: Current timestep counter
        num_envs: Number of parallel environments
        action_dim: Dimension of action space
        dt: Time step in seconds

    Returns:
        Array of shape (num_envs, action_dim) with sinusoidal actions in range [-0.5, 0.5]
    """
    time_elapsed = timestep * dt
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)

    # Generate sinusoidal motion with different frequencies for each joint
    for i in range(action_dim):
        # Use different frequencies for different joints (0.3 Hz to 0.8 Hz)
        frequency = 0.3 + (i % 5) * 0.1
        # Use different phase offsets to avoid all joints moving in sync
        phase_offset = i * np.pi / action_dim
        # Generate sinusoidal values in range [-0.5, 0.5]
        actions[:, i] = 0.5 * np.sin(
            2 * np.pi * frequency * time_elapsed + phase_offset
        )

    return actions


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#


def main() -> None:
    # Detect if this is a debug environment (GUI-controlled)
    is_debug_env = "Debug" in args_cli.task

    # Parse env cfg from registry
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Apply eval mode if available
    if hasattr(env_cfg, "eval"):
        env_cfg.eval()

    # Cleanup env for playing (remove training-specific components)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg = prepare_env_for_playing(env_cfg)

    # Create environment directly from cfg (no RL wrappers)
    render_mode = "rgb_array" if args_cli.video else None
    env = ManagerBasedRLEnv(env_cfg, render_mode=render_mode)

    # Optional video recording
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

    # Reset environment
    obs, _ = env.reset()

    # Get timing and action info
    dt = env.unwrapped.step_dt
    timestep = 0
    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.action_manager.total_action_dim

    # Determine action mode
    use_zero_actions = args_cli.zero_actions or is_debug_env

    print(f"[INFO] Environment loaded: {args_cli.task}")
    print(f"[INFO] Number of environments: {num_envs}")
    print(f"[INFO] Action dimension: {action_dim}")
    print(f"[INFO] Control timestep: {dt:.4f}s ({1.0 / dt:.1f} Hz)")

    if is_debug_env:
        print("[INFO] Debug environment detected - GUI controls are active.")
        print(
            "[INFO] Actions will be ignored; use the GUI sliders to control the robot."
        )
    elif use_zero_actions:
        print("[INFO] Using zero actions (environment will maintain default poses).")
    else:
        print("[INFO] Generating sinusoidal actions for environment validation...")

    while simulation_app.is_running():
        start = time.time()

        # Generate actions
        if use_zero_actions:
            actions = torch.zeros(
                num_envs, action_dim, dtype=torch.float32, device=env.unwrapped.device
            )
        else:
            actions_np = generate_sinusoidal_actions(timestep, num_envs, action_dim, dt)
            actions = torch.as_tensor(
                actions_np, dtype=torch.float32, device=env.unwrapped.device
            )

        # Step environment
        with torch.inference_mode():
            obs, _, _, _, _ = env.step(actions)

        timestep += 1

        # Stop video recording after specified length
        if args_cli.video and timestep == args_cli.video_length:
            break

        # Sleep to approximate real-time, if requested
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
