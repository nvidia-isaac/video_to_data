"""Render a still image of the GR1-Screwdriver scene/scaffold env (GR-1 + screwdriver objects).

  cd IsaacLab && ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/render_gr1_scene.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default="/home/cning/simtoolreal_isaaclab/videos/gr1_screwdriver_scene.png")
parser.add_argument("--cam_eye", type=str, default="1.35,2.0,1.5", help="camera position x,y,z (env-local)")
parser.add_argument("--cam_lookat", type=str, default="0.05,0.45,0.9", help="camera target x,y,z (env-local)")
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
parser.add_argument("--settle", type=int, default=40, help="zero-action steps to settle before rendering")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app = AppLauncher(args_cli).app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.gr1_screwdriver.gr1_screwdriver_env_cfg import GR1ScrewdriverEnvCfg  # noqa: E402


def main():
    cfg = GR1ScrewdriverEnvCfg()
    cfg.scene.num_envs = 1
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.resolution = (args_cli.width, args_cli.height)
    cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
    cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))

    env = gym.make("Isaac-GR1-Screwdriver-Direct-v0", cfg=cfg, render_mode="rgb_array")
    base = env.unwrapped
    env.reset()
    act = torch.zeros((1, cfg.action_space), device=base.device)
    for _ in range(args_cli.settle):     # settle (zero action holds the manipulation pose)
        env.step(act)
    frame = None
    for _ in range(30):                  # warm up the headless RTX viewport (first frames are black)
        frame = env.render()
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    imageio.imwrite(args_cli.out, frame)
    print(f"[render] wrote {args_cli.out}  {frame.shape}  nonzero_px={int((frame > 5).sum())}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
