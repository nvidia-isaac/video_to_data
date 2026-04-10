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
    help="Motion file to load.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

###################################
# Debug
###################################
args_cli.task = "Sharpa-V2P-v0-Play"
args_cli.headless = True

# Multiple rigid objects
# args_cli.motion_file = (
#     "taco/taco_processed/taco_empty__kettle__plate_20231031_060/sharpa_wave"
# )

# Single rigid object
# args_cli.motion_file = (
#     "arctic/arctic_processed/arctic_s01_rigid_mixer_grab_01/sharpa_wave"
# )

# Single articulated object
args_cli.motion_file = "arctic/arctic_processed/arctic_s01_mixer_use_01/sharpa_wave"

###################################

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from robotic_grounding.tasks import *
from robotic_grounding.tasks.scene_utils import (
    SceneConfig,
    apply_scene_config,
)  # noqa: E402


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env_cfg.motion_file = args_cli.motion_file
    if hasattr(env_cfg, "motion_file"):
        scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
        apply_scene_config(env_cfg, scene_config)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    while simulation_app.is_running():
        with torch.inference_mode():

            # actions = torch.zeros(*env.action_space.shape, device=env.unwrapped.device)
            # actions = torch.randn(*env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
