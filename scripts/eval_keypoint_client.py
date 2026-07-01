"""Env CLIENT for evaluating the STATE-BASED keypoint policy (run through isaaclab.sh).

Runs the Isaac Lab hammer env (no camera needed — keypoints come from physics) and drives the
robot with the keypoint policy served by eval_keypoint_server.py. Receding-horizon: every --replan
steps it computes the object-centric keypoints (tool + dynamic screw, via keypoint_utils — the SAME
function the collector used) + the joint state, sends them, gets a delta-joint chunk, and applies
cur_targets = clamp(joint_pos + delta). Reports the screw-seating success rate.

Run (server first in GR00T venv), then:
  source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/eval_keypoint_client.py \
      --headless --num_envs 25 --episodes 100 --replan 1 --port 5601
"""

import argparse
import math
import pickle
import socket
import struct
import time

from isaaclab.app import AppLauncher

REPO = "/home/cning/simtoolreal_isaaclab"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--max_ep_steps", type=int, default=800)
parser.add_argument("--replan", type=int, default=1, help="re-query the policy every K control steps")
parser.add_argument("--success_joint", type=float, default=-0.006)
parser.add_argument("--video", action="store_true", help="record a tiled grid of env cameras to videos/eval_keypoint_hammer.mp4")
parser.add_argument("--video_envs", type=int, default=16, help="number of envs to tile in the recorded grid")
parser.add_argument("--video_max_frames", type=int, default=1500)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5601)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = bool(args.video)  # keypoint policy needs no images; cameras only for the optional rollout video

app = AppLauncher(args).app

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, REPO)
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402
from simtoolreal_lab.tasks.simtoolreal.keypoint_utils import (  # noqa: E402
    compute_keypoints, find_screw_body, screw_offsets,
)


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def send_msg(sock, obj):
    sock.sendall(struct.pack(">I", len(pickle.dumps(obj, protocol=4))) + pickle.dumps(obj, protocol=4))


def recv_msg(sock):
    hdr = recvall(sock, 4)
    return None if hdr is None else pickle.loads(recvall(sock, struct.unpack(">I", hdr)[0]))


def make_grid(frames, cols):
    """Tile [G,h,w,3] uint8 env frames into a single (rows*h, cols*w, 3) grid image."""
    G, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    rows = (G + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(G):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frames[i]
    return grid


class BCEvalHammerEnv(HammerEnv):
    """Drive with BC delta joint targets (cur_targets = clamp(joint_pos + delta))."""

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions
        cur = torch.clamp(self.joint_pos + actions, self.dof_lower, self.dof_upper)
        self.cur_targets = cur
        self.prev_targets = cur


def make_cfg():
    cfg = HammerEnvCfg()                       # deploy/eval defaults baked into __post_init__
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.per_env_camera = bool(args.video)      # keypoint policy uses keypoints; camera only for the rollout video
    cfg.episode_length_s = args.max_ep_steps / 60.0
    cfg.terminate_on_nail_driven = args.success_joint
    return cfg


def main():
    torch.manual_seed(args.seed)
    sock = None
    for _ in range(120):  # server may still be loading
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((args.host, args.port))
            send_msg(sock, {"cmd": "ping"})
            if recv_msg(sock).get("ok"):
                break
        except (ConnectionRefusedError, OSError):
            if sock:
                sock.close()
            sock = None
            time.sleep(1)
    else:
        raise RuntimeError(f"could not reach keypoint server at {args.host}:{args.port}")
    print(f"[client] connected to keypoint server {args.host}:{args.port}", flush=True)

    env = BCEvalHammerEnv(make_cfg())
    base = env
    screw_off = screw_offsets(base)
    screw_body_idx = find_screw_body(base)
    H = base.cfg.action_horizon if hasattr(base.cfg, "action_horizon") else 40

    def query():
        kp = compute_keypoints(base, screw_off, screw_body_idx).detach().cpu().numpy().astype(np.float32)  # [N,8,3]
        js = base.joint_pos.detach().cpu().numpy().astype(np.float32)                                       # [N,29]
        send_msg(sock, {"cmd": "act", "keypoints": kp, "state": js})
        return recv_msg(sock)["action"]   # [N, horizon, 29] denormalized deltas

    # optional rollout video: tile the per-env cameras into a grid (policy still runs on keypoints)
    writer = pe_cam = None
    cols = G = 0
    if args.video:
        import os
        pe_cam = base.scene.sensors["per_env_cam"]
        G = min(args.num_envs, args.video_envs)
        cols = int(math.ceil(math.sqrt(G)))
        import imageio.v2 as imageio
        os.makedirs(f"{REPO}/videos", exist_ok=True)
        writer = imageio.get_writer(f"{REPO}/videos/eval_keypoint_hammer.mp4", fps=30,
                                    macro_block_size=None, codec="libx264")

    env.reset()
    for _ in range(40 if args.video else 1):
        base.sim.render()
    base.scene.update(base.physics_dt)
    plan, age, force = None, 0, True
    successes = attempts = steps = vid_n = 0
    K = args.replan
    print(f"[client] keypoint rollout: N={args.num_envs} envs, replan {K}, target {args.episodes} episodes"
          + (f" | recording {G}-env grid ({cols} cols)" if args.video else ""), flush=True)
    while attempts < args.episodes and steps < args.max_steps and not (args.video and vid_n >= args.video_max_frames):
        if force or age >= K or plan is None:
            plan = query(); age = 0; force = False
        if writer is not None and vid_n < args.video_max_frames:
            fr = pe_cam.data.output["rgb"][:G, ..., :3].to(torch.uint8).cpu().numpy()[:, ::2, ::2]  # [G,240,320,3]
            writer.append_data(make_grid(fr, cols)); vid_n += 1
        a = torch.from_numpy(plan[:, min(age, plan.shape[1] - 1)]).to(env.device)
        _, _, terminated, truncated, _ = env.step(a)
        done = terminated | truncated
        ndone = torch.nonzero(done).flatten().tolist()
        if ndone:
            attempts += len(ndone)
            successes += int(env.nail_driven[done].sum().item())
            force = True
        age += 1; steps += 1
        if steps % 100 == 0:
            print(f"[client] step={steps} episodes={attempts} success={successes} "
                  f"rate={successes/max(1,attempts):.0%}", flush=True)

    print(f"\n[client] DONE: {successes}/{attempts} episodes succeeded "
          f"(success_rate {successes/max(1,attempts):.1%}) over {steps} control steps", flush=True)
    if writer is not None:
        writer.close()
        print(f"[client] wrote {vid_n} grid frames -> {REPO}/videos/eval_keypoint_hammer.mp4", flush=True)
    try:
        send_msg(sock, {"cmd": "close"})
    except Exception:
        pass
    sock.close(); env.close(); app.close()


if __name__ == "__main__":
    main()
