# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Script to run an environment with dummy action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=15, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--action_mode",
    type=str,
    default="zero",
    choices=["zero"],
    help="Action mode.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Motion file to load (e.g. arctic_processed/arctic_s01_mixer_use_01/sharpa_wave).",
)
parser.add_argument(
    "--initial_virtual_object_control_curriculum_scale",
    type=float,
    default=1.0,
    help="Initial VOC curriculum scale for dual-hands object tracking command.",
)
parser.add_argument(
    "--no-collision",
    action="store_true",
    default=False,
    help="Load the *_no_collision.urdf for the object (tiny dummy collision geometry) for stage-1 collision-free warmup.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

###################################
# Debug
###################################
args_cli.task = "Sharpa-V2P-v0-Play"
args_cli.headless = True
args_cli.no_collision = False

# Multiple rigid objects
# args_cli.motion_file = (
#     "taco/taco_processed/taco_empty__kettle__plate_20231031_060/sharpa_wave"
# )

# Single rigid object
args_cli.motion_file = (
    "arctic/arctic_processed/arctic_s01_rigid_capsulemachine_grab_01/sharpa_wave"
)

# Single articulated object
# args_cli.motion_file = "arctic/arctic_processed/arctic_s01_capsulemachine_use_01/sharpa_wave"
###################################

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All Omniverse/Isaac Sim imports after SimulationApp is created."""

import gymnasium as gym
import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from robotic_grounding.tasks import *
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # Apply scene config from --motion_file
    scene_config = None
    if args_cli.motion_file is not None:
        env_cfg.motion_file = args_cli.motion_file
    if hasattr(env_cfg, "motion_file") and env_cfg.motion_file is not None:
        scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
        apply_scene_config(env_cfg, scene_config)

    # clamp viewer env_index to valid range (env_cfg may default to a higher index)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg) and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.env_index = min(
            env_cfg.viewer.env_index, env_cfg.scene.num_envs - 1
        )

    # --no-collision: replace the primary scene object's URDF with *_no_collision.urdf.
    if (
        args_cli.no_collision
        and scene_config is not None
        and scene_config.scene_objects
    ):
        primary = scene_config.scene_objects[0]
        scene_obj = getattr(env_cfg.scene, primary.name, None)
        if (
            scene_obj is not None
            and hasattr(scene_obj, "spawn")
            and hasattr(scene_obj.spawn, "asset_path")
        ):
            no_coll_path = scene_obj.spawn.asset_path.replace(
                "_art.urdf", "_no_collision.urdf"
            )
            scene_obj.spawn = scene_obj.spawn.replace(asset_path=no_coll_path)

    env_cfg.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale = (
        1.0
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    # Use trajectory frame rate for real-time motion playback.
    # step_dt (policy rate) is 2x slower than target_fps, so using step_dt would
    # play the motion at 0.5x speed. Use 1/target_fps instead.
    dt = env.unwrapped.step_dt
    cmd_manager = env.unwrapped.command_manager
    for term_name in [
        "dual_hands_object_tracking_command",
        "dual_hands_tracking_command",
    ]:
        try:
            cmd_term = cmd_manager.get_term(term_name)
            if getattr(cmd_term.cfg, "target_fps", None) is not None:
                dt = 1.0 / cmd_term.cfg.target_fps
            break
        except Exception:
            continue

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():

            # actions = torch.zeros(*env.action_space.shape, device=env.unwrapped.device)
            # actions = torch.randn(*env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
