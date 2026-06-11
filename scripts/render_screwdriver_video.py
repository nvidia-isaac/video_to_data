"""Render an mp4 of the screwdriver scene that RESETS every N steps (re-randomizing the
layout each time), for K resets total, from the demo (normal, no-zoom) camera.

Run: ./isaaclab.sh -p <this> --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps_per_reset", type=int, default=30)
parser.add_argument("--num_resets", type=int, default=5)
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--render_warmup", type=int, default=80, help="render ticks before recording (stream textures)")
parser.add_argument("--out", type=str, default="/home/cning/simtoolreal_isaaclab/videos/screwdriver_resets.mp4")
# demo (normal) camera, env-local — same as deploy_pretrained.py
parser.add_argument("--cam_eye", type=str, default="-0.55,-0.45,0.80")
parser.add_argument("--cam_lookat", type=str, default="0.0,0.20,0.62")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg  # noqa: E402

RESULT_FILE = "/home/cning/simtoolreal_isaaclab/render_video_result.txt"


def main():
    rf = open(RESULT_FILE, "w")
    emit = lambda m: (rf.write(m + "\n"), rf.flush(), os.fsync(rf.fileno()))
    try:
        cfg = ScrewdriverEnvCfg()
        cfg.scene.num_envs = 1
        cfg.domain_randomization = False
        cfg.randomize_layout = True  # re-randomize on every reset
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.resolution = (1280, 720)
        cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
        cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))

        env = gym.make("Isaac-SimToolReal-Screwdriver-Direct-v0", cfg=cfg, render_mode="rgb_array")
        base = env.unwrapped
        zero = torch.zeros((base.num_envs, base.cfg.action_space), device=base.device)

        env.reset()
        for _ in range(args_cli.render_warmup):  # let RTX stream textures in before recording
            base.render()

        frames = []
        for seg in range(args_cli.num_resets):
            env.reset()  # new randomized layout
            for _ in range(args_cli.steps_per_reset):
                env.step(zero)
                img = np.asarray(base.render())
                frames.append(img[..., :3].astype(np.uint8))
            emit(f"segment {seg}: tool={[round(x,3) for x in (base.object.data.root_pos_w[0]-base.scene.env_origins[0]).tolist()]} "
                 f"screw|v|={base.screw.data.root_lin_vel_w[0].norm().item():.3f}")

        os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
        imageio.mimwrite(args_cli.out, frames, fps=args_cli.fps)
        emit(f"WROTE {args_cli.out}  frames={len(frames)} fps={args_cli.fps} "
             f"({args_cli.num_resets} resets x {args_cli.steps_per_reset} steps)")
        emit("VIDEO_OK")
        env.close()
    except Exception:
        emit("EXCEPTION:\n" + traceback.format_exc())
    finally:
        rf.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
