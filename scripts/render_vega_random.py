"""Render an mp4 of the Vega + Sharpa robot doing RANDOM actions in two environments (side by side).

Uses the per-env TiledCamera (one view per sub-env) so each of the `num_envs` environments is shown
in its own tile, concatenated horizontally into one frame. Random 29-DOF actions are sampled in
[-1,1] and held for `action_hold` control steps (the env's EMA/integration smooths them into sweeping
arm + finger motion). Also dumps frame 0 + a mid frame as PNGs so the camera framing can be checked.

Run: source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && \
     OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/render_vega_random.py \
        --headless --task hammer --num_envs 2 --steps 150 --fps 30
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["hammer", "screwdriver"], default="hammer")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--action_hold", type=int, default=6, help="resample the random action every N steps")
parser.add_argument("--action_scale", type=float, default=1.0, help="scale on the [-1,1] random action")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--warmup", type=int, default=40, help="steps before recording (stream textures + settle)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="")
# per-env camera (env-local). Default: front-3/4 elevated view framing the standing Vega robot + table.
parser.add_argument("--cam_eye", type=str, default="0.95,-1.05,1.05")
parser.add_argument("--cam_lookat", type=str, default="0.0,-0.20,0.66")
parser.add_argument("--cam_z_far", type=float, default=3.2)
parser.add_argument("--cam_focal", type=float, default=24.0, help="per-env camera focal length (mm); larger = zoom in")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import simtoolreal_lab.tasks  # noqa: E402,F401


def main():
    if args.task == "hammer":
        from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv
        from simtoolreal_lab.tasks.vega_hammer.vega_hammer_env_cfg import VegaHammerEnvCfg
        cfg, EnvCls = VegaHammerEnvCfg(), HammerEnv
    else:
        from simtoolreal_lab.tasks.screwdriver.screwdriver_env import ScrewdriverEnv
        from simtoolreal_lab.tasks.vega_screwdriver.vega_screwdriver_env_cfg import VegaScrewdriverEnvCfg
        cfg, EnvCls = VegaScrewdriverEnvCfg(), ScrewdriverEnv

    cfg.scene.num_envs = args.num_envs
    cfg.domain_randomization = False           # clean visual: the ACTIONS are the only randomness
    cfg.per_env_camera = True
    cfg.cam_eye = tuple(float(v) for v in args.cam_eye.split(","))
    cfg.cam_lookat = tuple(float(v) for v in args.cam_lookat.split(","))
    cfg.cam_z_far = args.cam_z_far
    cfg.cam_focal = args.cam_focal

    env = EnvCls(cfg, render_mode=None)
    cam = env.scene.sensors["per_env_cam"]
    N, A = env.num_envs, cfg.action_space
    gen = torch.Generator(device=env.device).manual_seed(args.seed)

    def grab():
        rgb = cam.data.output["rgb"][..., :3]            # (N,H,W,3) uint8
        tiles = [rgb[i].cpu().numpy().astype(np.uint8) for i in range(N)]
        sep = np.full((tiles[0].shape[0], 4, 3), 30, np.uint8)  # thin divider between envs
        out = tiles[0]
        for t in tiles[1:]:
            out = np.concatenate([out, sep, t], axis=1)
        return out

    env.reset()
    zero = torch.zeros((N, A), device=env.device)
    for _ in range(args.warmup):
        env.step(zero)

    frames = []
    action = torch.zeros((N, A), device=env.device)
    for t in range(args.steps):
        if t % args.action_hold == 0:
            action = (2.0 * torch.rand((N, A), generator=gen, device=env.device) - 1.0) * args.action_scale
        env.step(action)
        frames.append(grab())

    out = args.out or f"/home/cning/simtoolreal_isaaclab/videos/vega_{args.task}_random_{N}env.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimwrite(out, frames, fps=args.fps, quality=8)
    # sample PNGs for a quick framing check
    imageio.imwrite(out.replace(".mp4", "_frame000.png"), frames[0])
    imageio.imwrite(out.replace(".mp4", "_frameMID.png"), frames[len(frames) // 2])
    print(f"WROTE {out}  frames={len(frames)} size={frames[0].shape} fps={args.fps}", flush=True)
    print("RENDER_OK", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
