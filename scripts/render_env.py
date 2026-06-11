"""Render a still image of a screwdriver env (for checking asset placement). Uses the env's
built-in per-env camera; steps a few zero-action frames to settle, then saves one RTX frame."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-SimToolReal-Screwdriver043-Direct-v0")
parser.add_argument("--out", type=str, default="/home/cning/simtoolreal_isaaclab/videos/env043_check.png")
parser.add_argument("--cam_eye", type=str, default="-0.28,-0.62,0.95")
parser.add_argument("--cam_lookat", type=str, default="0.12,0.05,0.60")
parser.add_argument("--steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import os  # noqa: E402
import sys  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.screwdriver043.screwdriver043_env_cfg import Screwdriver043EnvCfg  # noqa: E402


def main():
    cfg = Screwdriver043EnvCfg()
    cfg.scene.num_envs = 1
    cfg.randomize_layout = False
    cfg.demo_mode = False
    cfg.use_fixed_goal_trajectory = False
    cfg.domain_randomization = False
    cfg.physical_screw = False
    cfg.responsive_goals = False
    cfg.reset_position_noise_x = cfg.reset_position_noise_y = cfg.reset_position_noise_z = 0.0
    cfg.sim.physx.gpu_collision_stack_size = 2 ** 28
    # per-env camera framed on the table region
    cfg.per_env_camera = True
    cfg.cam_width, cfg.cam_height = 1600, 1000
    cfg.cam_eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
    cfg.cam_lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))

    env = gym.make(args_cli.task, cfg=cfg, render_mode=None)
    base = env.unwrapped
    env.reset()
    cam = base.scene.sensors["per_env_cam"]
    zero = torch.zeros((1, base.cfg.action_space), device=base.device)
    for _ in range(args_cli.steps):
        env.step(zero)
    rgb = cam.data.output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    imageio.imwrite(args_cli.out, rgb.astype("uint8"))
    print(f"[render_env] saved {args_cli.out}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
