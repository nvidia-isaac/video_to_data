"""Render a single RGB image of the screwdriver scene (robot + screwdriver + thread_test +
screw resting on the fixture) from the SAME camera as the deploy_pretrained.py demo.

Run: ./isaaclab.sh -p <this> --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--settle_steps", type=int, default=90, help="zero-action steps to let the screw settle on the fixture")
parser.add_argument("--render_warmup", type=int, default=80, help="render ticks before capture (lets RTX stream textures in)")
parser.add_argument("--out", type=str, default="/home/cning/simtoolreal_isaaclab/videos/check_frames/screwdriver_scene.png")
# demo camera (deploy_pretrained.py defaults), env-local
parser.add_argument("--cam_eye", type=str, default="-0.55,-0.45,0.80")
parser.add_argument("--cam_lookat", type=str, default="0.0,0.20,0.62")
# visual changes (match the hammer BC look): dark-grey floor+backdrop, light-blue screw, bigger env gap
parser.add_argument("--ground_rgb", type=str, default="0.12,0.12,0.12", help="solid floor+backdrop color 'r,g,b' ('' = default grid)")
parser.add_argument("--screw_rgb", type=str, default="0.55,0.72,0.82", help="recolor the screw 'r,g,b' ('' = asset color)")
parser.add_argument("--env_spacing", type=float, default=8.0, help="bigger environment gap (neighbor sub-envs pushed away)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True  # offscreen RTX rendering

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import simtoolreal_lab.tasks  # noqa: E402, F401  (registers the task)
from simtoolreal_lab.tasks.screwdriver.screwdriver_env_cfg import ScrewdriverEnvCfg  # noqa: E402

RESULT_FILE = "/home/cning/simtoolreal_isaaclab/render_screwdriver_result.txt"


def main():
    rf = open(RESULT_FILE, "w")

    def emit(msg):
        rf.write(msg + "\n")
        rf.flush()
        os.fsync(rf.fileno())

    try:
        cfg = ScrewdriverEnvCfg()
        cfg.scene.num_envs = 1
        cfg.domain_randomization = False  # clean, deterministic scene
        # --- visual changes (same look as the hammer BC frames) ---
        cfg.scene.env_spacing = args_cli.env_spacing            # bigger environment gap
        if args_cli.ground_rgb:                                 # dark-grey floor + backdrop
            cfg.ground_color = tuple(float(v) for v in args_cli.ground_rgb.split(","))
        if args_cli.screw_rgb:                                  # light-blue screw
            cfg.screw_color = tuple(float(v) for v in args_cli.screw_rgb.split(","))
        # demo camera (env-local), exactly like deploy_pretrained.py --video
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = 0
        cfg.viewer.resolution = (1280, 720)
        cfg.viewer.eye = tuple(float(v) for v in args_cli.cam_eye.split(","))
        cfg.viewer.lookat = tuple(float(v) for v in args_cli.cam_lookat.split(","))

        env = gym.make("Isaac-SimToolReal-Screwdriver-Direct-v0", cfg=cfg, render_mode="rgb_array")
        base = env.unwrapped
        env.reset()

        # settle: zero actions so the screw drops from its spawn onto the thread_test fixture
        zero = torch.zeros((base.num_envs, base.cfg.action_space), device=base.device)
        speeds = []
        for k in range(args_cli.settle_steps):
            env.step(zero)
            v = base.object.data.root_lin_vel_w[0].norm().item()      # screwdriver resting on the table
            w = base.object.data.root_ang_vel_w[0].norm().item()
            speeds.append(v)
            if k % 20 == 0 or k == args_cli.settle_steps - 1:
                emit(f"settle step {k:3d}: screwdriver |v|={v:.5f} m/s |w|={w:.4f} rad/s")
        emit(f"SCREWDRIVER settle: last-30 max |v|={max(speeds[-30:]):.6f} m/s  "
             f"(near 0 + steady = asleep/settled, NOT jittering)")

        base._compute_intermediate_values()
        r = lambda t: [round(x, 3) for x in t]
        eo = base.scene.env_origins[0]
        emit("LOCAL tool=" + str(r((base.object.data.root_pos_w[0] - eo).tolist())))

        # render the viewport from the demo camera. RTX streams textures (e.g. the
        # screwdriver's PNG) in ASYNCHRONOUSLY over many frames, so warm up the renderer
        # before capturing or the asset shows its un-streamed (dark) base material.
        img = None
        for _ in range(args_cli.render_warmup):
            img = base.render()
        if img is None:
            emit("RENDER_RETURNED_NONE")
        else:
            img = np.asarray(img)
            emit(f"IMG shape={img.shape} dtype={img.dtype} min={int(img.min())} max={int(img.max())}")
            os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
            try:
                from PIL import Image

                Image.fromarray(img[..., :3].astype(np.uint8)).save(args_cli.out)
            except Exception:
                import imageio.v2 as imageio

                imageio.imwrite(args_cli.out, img[..., :3].astype(np.uint8))
            emit(f"SAVED {args_cli.out}")
        emit("RENDER_OK")
        env.close()
    except Exception:
        emit("EXCEPTION:\n" + traceback.format_exc())
    finally:
        rf.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
