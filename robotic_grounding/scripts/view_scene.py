r"""View a scene: spawns object, optional support surface, and robot from Parquet (no RL).

Robot articulation is added when the motion path includes a ``robot_name=...``
partition and that name exists in ``assets/robot_registry.py`` (e.g. ``g1``).

Usage (Sharpa / Arctic-style Parquet under ``human_motion_data``):
    python scripts/view_scene.py \
        --motion_file source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic_processed/sequence_id=arctic_s01_ketchup_use_01/robot_name=sharpa_wave

SOMA→G1 retargeted data (after ``soma_to_g1.py --save``):
    # Save motion first (writes under HUMAN_MOTION_DATA_DIR/whole_body/soma/...)
    python scripts/retarget/soma_to_g1.py <sequence_dir> --save

    # Point at the partition directory (sequence_id=.../robot_name=g1) or use cwd-relative path
    python scripts/view_scene.py \
        --motion_file source/robotic_grounding/robotic_grounding/assets/human_motion_data/whole_body/soma/sequence_id=<your_sequence>/robot_name=g1

    # Headless (validate without GUI)
    python scripts/view_scene.py --motion_file <path> --headless

    # Record MP4 video
    python scripts/view_scene.py --motion_file <path> --headless \\
        --record_video --output_dir /tmp/videos --video_length 300
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a scene from a motion file.")
parser.add_argument(
    "--motion_file", type=str, required=True, help="Path to parquet motion dir."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--record_video",
    action="store_true",
    default=False,
    help="Record the scene replay as MP4 video.",
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras when recording video (must be set before AppLauncher)
if args_cli.record_video:
    args_cli.enable_cameras = True
    if not args_cli.output_dir:
        raise ValueError("--output_dir is required when using --record_video")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from robotic_grounding.tasks.scene_utils.scene_viewer_env_cfg import (  # noqa: E402
    SceneViewerEnvCfg,
)

# Register for gym.make() + RecordVideo support
gym.register(
    id="SceneViewer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
)


def main() -> None:
    """View a scene from a motion file, optionally recording video."""
    cfg = SceneViewerEnvCfg(motion_file=args_cli.motion_file)
    cfg.scene.num_envs = args_cli.num_envs

    if args_cli.record_video:
        os.makedirs(args_cli.output_dir, exist_ok=True)

        # Use gym.make so RecordVideo can wrap it
        env = gym.make(
            "SceneViewer-v0",
            cfg=cfg,
            render_mode="rgb_array",
        )
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.output_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        env.reset()

        actions = torch.zeros(env.unwrapped.num_envs, 0, device=env.unwrapped.device)

        print(f"[INFO] Scene loaded from: {args_cli.motion_file}")
        print(
            f"[INFO] Recording {args_cli.video_length} steps to {args_cli.output_dir}"
        )
        for step in range(args_cli.video_length):
            env.step(actions)
            if (step + 1) % 100 == 0:
                print(f"[INFO] Step: {step + 1}/{args_cli.video_length}")
        print(f"[INFO] Video saved to {args_cli.output_dir}")

    else:
        # Interactive / headless mode — no gym wrapper needed
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset()

        actions = torch.zeros(env.num_envs, 0, device=env.device)

        print(f"[INFO] Scene loaded from: {args_cli.motion_file}")
        print("[INFO] Close the window to exit.")

        step = 0
        while simulation_app.is_running():
            env.step(actions)
            step += 1
            if step % 100 == 0:
                print(f"[INFO] Step: {step}")
                env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
