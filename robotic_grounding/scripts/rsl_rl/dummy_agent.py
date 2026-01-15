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
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--action_mode",
    type=str,
    default="zero",
    choices=["zero", "tracking_target"],
    help="Action mode.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

###################################
# Debug
###################################
args_cli.task = "Sharpa-V2P-v0-Play"
args_cli.num_envs = 4
args_cli.headless = True
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
from robotic_grounding.tasks import *
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG


def random_sample_trajectory(
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    current_joint_pos: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    target_joint_pos = (
        torch.rand_like(current_joint_pos, device=device)
        * (upper_limits - lower_limits)
        + lower_limits
    )
    alphas = torch.linspace(0, 1, num_steps, device=device)
    trajectory = (
        current_joint_pos.unsqueeze(-1) * (1 - alphas)
        + target_joint_pos.unsqueeze(-1) * alphas
    )
    return trajectory


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # env_cfg.actions.joint_pos.scale = 1.0

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    robot = env.unwrapped.scene["robot"]
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    if args_cli.action_mode == "tracking_target":
        target_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Target")
        target_marker_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
        target_marker = VisualizationMarkers(target_marker_cfg)

        count = 0

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            if args_cli.action_mode == "tracking_target":
                if count % 200 == 0:
                    trajectory = random_sample_trajectory(
                        robot.data.joint_pos_limits[..., 0],
                        robot.data.joint_pos_limits[..., 1],
                        actions.clone(),
                        200,
                        env.unwrapped.device,
                    )
                    count = 0

                actions = trajectory[..., count]
                target_marker.visualize(
                    actions[:, [0, 2, 4]]
                    + robot.data.default_joint_pos[:, [0, 2, 4]]
                    + env.unwrapped.scene.env_origins,
                    torch.tensor(
                        [[1, 0, 0, 0]] * len(actions), device=env.unwrapped.device
                    ),
                )

                count += 1

            elif args_cli.action_mode == "zero":
                actions = torch.zeros(
                    *env.action_space.shape, device=env.unwrapped.device
                )

            else:
                raise ValueError(f"Invalid action mode: {args_cli.action_mode}")

            env.step(actions)

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
