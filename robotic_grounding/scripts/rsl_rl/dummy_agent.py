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
import sys
import time

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

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
    help="Motion file to load (e.g. arctic/arctic_processed/dataset_s01_mixer_use_01/sharpa_wave).",
)
parser.add_argument(
    "--initial_virtual_object_control_curriculum_scale",
    type=float,
    default=1.0,
    help="Initial VOC curriculum scale for dual-hands object tracking command.",
)
parser.add_argument(
    "--disable_robot_to_object_collisions",
    action="store_true",
    default=False,
    help="Disable robot-to-object collisions.",
)
parser.add_argument(
    "--disable_robot_to_fixed_object_collisions",
    action="store_true",
    default=False,
    help="Disable robot-to-fixed-object collisions.",
)
parser.add_argument(
    "--use_primitive_urdfs",
    action="store_true",
    default=False,
    help="Use primitive URDFs for the robot.",
)
parser.add_argument(
    "--debug_vis",
    action="store_true",
    default=False,
    help="Enable command debug-vis markers (target/current poses, contacts, fingertips).",
)
parser.add_argument(
    "--record_video",
    action="store_true",
    default=False,
    help="Record an MP4 of the scene replay.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to save recorded video (required with --record_video).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=300,
    help="Number of simulation steps to record (default: 300).",
)
parser.add_argument(
    "--success_marker",
    type=str,
    default=None,
    help=(
        "Optional path to a sentinel file. Written ONLY after the recording "
        "loop completes cleanly. CUDA assert, exception, or timeout-kill all "
        "prevent the write from happening, so a present sentinel proves the "
        "sequence played end-to-end. Used by workflow/retarget.yaml Stage 6 "
        "to split processed parquets into renderable vs. invalid buckets."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras when recording video (must be set before AppLauncher)
if args_cli.record_video:
    args_cli.enable_cameras = True
    if not args_cli.output_dir:
        raise ValueError("--output_dir is required when using --record_video")

###################################
# Debug — uncomment to override CLI args
###################################
# args_cli.task = "Sharpa-V2D-v0-Play"
# args_cli.headless = True
# args_cli.disable_robot_to_object_collisions = False
# args_cli.disable_robot_to_fixed_object_collisions = True
# args_cli.motion_file = "arctic/arctic_processed/dataset_s01_mixer_use_01/sharpa_wave"
# args_cli.motion_file = "arctic/arctic_processed/dataset_s07_box_grab_01/sharpa_wave"
# args_cli.motion_file = "hot3d/hot3d_processed/P0002_59a84a3a_seg025/sharpa_wave"
###################################

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""All Omniverse/Isaac Sim imports after SimulationApp is created."""

import os

import gymnasium as gym
import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from robotic_grounding.tasks import *
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from isaaclab_tasks.utils import parse_env_cfg
from viewer_utils import autoframe_viewer


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
    if args_cli.motion_file is not None:
        env_cfg.motion_file = args_cli.motion_file
    if hasattr(env_cfg, "motion_file") and env_cfg.motion_file is not None:
        scene_config = SceneConfig.from_motion_file(env_cfg.motion_file)
        apply_scene_config(
            env_cfg, scene_config, use_primitive_urdfs=args_cli.use_primitive_urdfs
        )
        # Auto-frame the viewer on the actual motion bounding box.  The default
        # eye/lookat in v2d_hand_env_cfg targets standing-human scenes (lookat
        # z=1.2); tabletop datasets like h2o / dexycb are at z<0.3 and end up
        # off-screen otherwise.
        autoframe_viewer(env_cfg, scene_config.motion_file)

    # clamp viewer env_index to valid range (env_cfg may default to a higher index)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg) and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.env_index = min(
            env_cfg.viewer.env_index, env_cfg.scene.num_envs - 1
        )

    # V2D dual-hands specific config (skip for whole-body envs)
    if hasattr(env_cfg, "commands") and hasattr(
        env_cfg.commands, "dual_hands_object_tracking_command"
    ):
        env_cfg.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale = float(
            args_cli.initial_virtual_object_control_curriculum_scale
        )
    if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "setup_collision_groups"):
        env_cfg.events.setup_collision_groups.params[
            "disable_robot_to_object_collisions"
        ] = args_cli.disable_robot_to_object_collisions
        env_cfg.events.setup_collision_groups.params[
            "disable_robot_to_fixed_object_collisions"
        ] = args_cli.disable_robot_to_fixed_object_collisions

    # Enable debug-vis markers on any command term that supports it
    if args_cli.debug_vis and hasattr(env_cfg, "commands"):
        for term_name in vars(env_cfg.commands):
            term = getattr(env_cfg.commands, term_name)
            if hasattr(term, "debug_vis"):
                term.debug_vis = True

    # create environment (with RecordVideo wrapper if requested)
    if args_cli.record_video:
        os.makedirs(args_cli.output_dir, exist_ok=True)
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.output_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    else:
        env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    env.reset()

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    if args_cli.record_video:
        print(
            f"[INFO] Recording {args_cli.video_length} steps to {args_cli.output_dir}"
        )
        for step in range(args_cli.video_length):
            with torch.inference_mode():
                env.step(actions)
            if (step + 1) % 100 == 0:
                print(f"[INFO] Step: {step + 1}/{args_cli.video_length}")
        print(f"[INFO] Recording loop done; closing env to flush MP4")
    else:
        while simulation_app.is_running():
            with torch.inference_mode():
                env.step(actions)

    env.close()

    if args_cli.record_video and args_cli.success_marker:
        from pathlib import Path as _Path

        expected_mp4 = _Path(args_cli.output_dir) / "rl-video-step-0.mp4"
        if expected_mp4.is_file() and expected_mp4.stat().st_size > 0:
            marker = _Path(args_cli.success_marker)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            print(f"[INFO] Wrote success marker {marker}")
        else:
            print(
                f"[WARN] No MP4 at {expected_mp4}; withholding success marker "
                f"{args_cli.success_marker}"
            )


if __name__ == "__main__":
    main()
    simulation_app.close()
