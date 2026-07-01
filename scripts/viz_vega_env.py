"""Render a 10 s mp4 of the Vega-right SimToolReal TRAINING env to eyeball the setup:
the robot, the 4 OBJECT keypoints (GREEN box on the hammer -- verifies the keypoint-size fix),
and the 4 GOAL keypoints (BLUE box at the elevated target -- verifies goal placement).

Driven by gentle random actions (the policy isn't trained); the point is to SEE the geometry, not a
grasp. Runs on a chosen CUDA device so it won't disturb a training run on another GPU:
  CUDA_VISIBLE_DEVICES=1 ... isaaclab.sh -p scripts/viz_vega_env.py --headless
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=600, help="control steps (60 Hz); 600 = 10 s")
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--warmup", type=int, default=40, help="zero-action settle steps before recording")
parser.add_argument("--action_scale", type=float, default=0.5)
parser.add_argument("--action_hold", type=int, default=12, help="resample random action every N steps")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cam_eye", type=str, default="0.95,-1.75,1.45")
parser.add_argument("--cam_lookat", type=str, default="0.0,-0.50,1.00")
parser.add_argument("--cam_focal", type=float, default=22.0)
parser.add_argument("--out", type=str, default="/home/cning/simtoolreal_isaaclab/videos/vega_right_env_check.mp4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import os, sys  # noqa: E402
sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

import simtoolreal_lab.tasks  # noqa: E402,F401
from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv  # noqa: E402
from simtoolreal_lab.tasks.vega_hammer_right.vega_hammer_right_env_cfg import VegaHammerRightEnvCfg  # noqa: E402


def sphere(path, color, radius=0.014):
    return VisualizationMarkers(VisualizationMarkersCfg(
        prim_path=path, markers={"s": sim_utils.SphereCfg(
            radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))}))


def main():
    cfg = VegaHammerRightEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.domain_randomization = False          # clean: random ACTIONS are the only motion
    cfg.per_env_camera = True
    cfg.cam_eye = tuple(float(v) for v in args.cam_eye.split(","))
    cfg.cam_lookat = tuple(float(v) for v in args.cam_lookat.split(","))
    cfg.cam_focal = args.cam_focal

    env = HammerEnv(cfg, render_mode=None)
    cam = env.scene.sensors["per_env_cam"]
    N, A = env.num_envs, cfg.action_space

    obj_kp = sphere("/Visuals/object_kp", (0.10, 0.95, 0.10))   # GREEN  = object bbox keypoints
    goal_kp = sphere("/Visuals/goal_kp", (0.20, 0.45, 1.00))    # BLUE   = goal bbox keypoints
    goal_ctr = sphere("/Visuals/goal_ctr", (1.00, 0.25, 0.20), radius=0.020)  # RED = goal center

    def draw():
        eo = env.scene.env_origins                                   # (N,3) env-local -> world
        obj_w = (env.object_keypoints + eo.unsqueeze(1)).reshape(-1, 3)
        goal_w = (env.goal_keypoints + eo.unsqueeze(1)).reshape(-1, 3)
        gc_w = (env.goal_pos + eo)
        obj_kp.visualize(translations=obj_w)
        goal_kp.visualize(translations=goal_w)
        goal_ctr.visualize(translations=gc_w)

    def grab():
        return cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)

    gen = torch.Generator(device=env.device).manual_seed(args.seed)
    env.reset()
    zero = torch.zeros((N, A), device=env.device)
    for _ in range(args.warmup):
        env.step(zero); draw()

    # log the keypoint box extent so the fix is verifiable from the console too
    kp = env.object_keypoints[0]                                     # (4,3) env-local
    ext = (kp.max(0).values - kp.min(0).values).tolist()
    print(f"[viz] object keypoint bbox extent (m): "
          f"x={ext[0]:.3f} y={ext[1]:.3f} z={ext[2]:.3f}  (fix: ~0.15/0.034/0.022; bug was ~0.008)", flush=True)

    frames = []
    action = torch.zeros((N, A), device=env.device)
    for t in range(args.steps):
        if t % args.action_hold == 0:
            action = (2.0 * torch.rand((N, A), generator=gen, device=env.device) - 1.0) * args.action_scale
        env.step(action)
        draw()
        env.sim.render()                                             # re-render so markers appear this frame
        frames.append(grab())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=args.fps, quality=8)
    imageio.imwrite(args.out.replace(".mp4", "_frame000.png"), frames[0])
    imageio.imwrite(args.out.replace(".mp4", "_frameMID.png"), frames[len(frames) // 2])
    print(f"WROTE {args.out}  frames={len(frames)} size={frames[0].shape} fps={args.fps}", flush=True)
    print("VIZ_OK", flush=True)
    env.close(); app.close()


if __name__ == "__main__":
    main()
