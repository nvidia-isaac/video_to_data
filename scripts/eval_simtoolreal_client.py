"""Env CLIENT for evaluating the SimToolReal SPECIALIST (state-based). Run through isaaclab.sh.

Runs the Isaac Lab hammer env and drives the robot with the SimToolReal specialist served by
eval_simtoolreal_server.py. Each --replan steps it computes the expert-faithful state via
keypoint_utils.compute_simtoolreal_obs (the SAME function the collector used) -- PALM-RELATIVE
keypoints + the 109-dim proprio (joint_pos, joint_vel, prev_targets, palm pose, fingertips) --
sends them, gets a delta-joint chunk, and applies cur_targets = clamp(joint_pos + delta).

Run (server first in GR00T venv), then:
  source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/eval_simtoolreal_client.py \
      --headless --num_envs 25 --episodes 100 --replan 1 --port 5602
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
parser.add_argument("--with_goal", action="store_true", help="append keypoints_rel_goal(12) to proprio (match a --with_goal-trained checkpoint); the env provides the goal each step")
parser.add_argument("--no_joint_vel", action="store_true", help="DROP joint_vel [29:58] from proprio (109->80) to match a --no_joint_vel-trained checkpoint")
parser.add_argument("--table_dist", type=float, default=0.15, help="move the work-table this far (m) further from the robot (match the collection's --table_dist). DEFAULT 0.15 (the datasets' value); pass 0 only to eval against an old table_dist=0 dataset")
parser.add_argument("--video", action="store_true", help="record a tiled grid of env cameras to videos/eval_simtoolreal_hammer.mp4")
parser.add_argument("--video_envs", type=int, default=16)
parser.add_argument("--video_max_frames", type=int, default=1500)
parser.add_argument("--success_videos", type=int, default=0, help="save standalone mp4s of the first N episodes that END in success (nail driven), one file each")
parser.add_argument("--success_max_len", type=int, default=1500, help="max frames buffered per episode for --success_videos")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5602)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log_goal_idx", action="store_true", help="log base.successes (goal/trajectory index) distribution over the rollout -> does the goal advance when met during eval?")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = bool(args.video or args.success_videos)  # state policy needs no images; cameras only for video

app = AppLauncher(args).app

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, REPO)
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402
from simtoolreal_lab.tasks.simtoolreal.keypoint_utils import (  # noqa: E402
    compute_simtoolreal_obs, compute_goal_rel, find_screw_body, screw_offsets,
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
    cfg.per_env_camera = bool(args.video or args.success_videos)  # camera only for the optional video
    cfg.episode_length_s = args.max_ep_steps / 60.0
    cfg.terminate_on_nail_driven = args.success_joint
    cfg.table_dist = args.table_dist               # match the collection's table distance (sim-to-real eval condition)
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
        raise RuntimeError(f"could not reach simtoolreal server at {args.host}:{args.port}")
    print(f"[client] connected to simtoolreal server {args.host}:{args.port}", flush=True)

    env = BCEvalHammerEnv(make_cfg())
    base = env
    screw_off = screw_offsets(base)
    screw_body_idx = find_screw_body(base)

    def query():
        kp_rel, proprio = compute_simtoolreal_obs(base, screw_off, screw_body_idx)
        if args.no_joint_vel:   # drop joint_vel [29:58] -> 80, matching a --no_joint_vel checkpoint
            proprio = torch.cat([proprio[:, :29], proprio[:, 58:]], dim=-1)
        if args.with_goal:   # append keypoints_rel_goal(12) -> 121, matching a --with_goal checkpoint
            proprio = torch.cat([proprio, compute_goal_rel(base)], dim=-1)
        kp = kp_rel.detach().cpu().numpy().astype(np.float32)        # [N,8,3] palm-relative
        pr = proprio.detach().cpu().numpy().astype(np.float32)       # [N,109] or [N,121] with goal
        send_msg(sock, {"cmd": "act", "keypoints": kp, "proprio": pr})
        return recv_msg(sock)["action"]   # [N, horizon, 29] denormalized deltas

    # optional rollout video (policy still runs on state; camera only for viewing)
    import os
    writer = pe_cam = None
    cols = G = 0
    SV = args.success_videos
    benv = None; saved_succ = 0
    if args.video or SV:
        pe_cam = base.scene.sensors["per_env_cam"]
        os.makedirs(f"{REPO}/videos", exist_ok=True)
        import imageio.v2 as imageio
    if args.video:
        G = min(args.num_envs, args.video_envs)
        cols = int(math.ceil(math.sqrt(G)))
        writer = imageio.get_writer(f"{REPO}/videos/eval_simtoolreal_hammer.mp4", fps=30,
                                    macro_block_size=None, codec="libx264")
    if SV:
        benv = [[] for _ in range(args.num_envs)]   # per-env episode frame buffers (downscaled)

    def _save_succ(frames, k):
        out = f"{REPO}/videos/eval_simtoolreal_success_{k}.mp4"
        w = imageio.get_writer(out, fps=30, macro_block_size=None, codec="libx264")
        for fr in frames:
            w.append_data(fr)
        w.close()
        return out

    env.reset()
    for _ in range(40 if (args.video or SV) else 1):
        base.sim.render()
    base.scene.update(base.physics_dt)
    plan, age, force = None, 0, True
    successes = attempts = steps = vid_n = 0
    goal_idx_max = 0   # max goal/trajectory index reached across the run (does the goal advance when met?)
    K = args.replan
    print(f"[client] simtoolreal rollout: N={args.num_envs} envs, replan {K}, target {args.episodes} episodes"
          + (f" | recording {G}-env grid ({cols} cols)" if args.video else "")
          + (f" | saving {SV} success episode videos" if SV else ""), flush=True)
    while (attempts < args.episodes and steps < args.max_steps
           and not (args.video and vid_n >= args.video_max_frames)
           and not (SV and saved_succ >= SV)):
        if force or age >= K or plan is None:
            plan = query(); age = 0; force = False
        if writer is not None and vid_n < args.video_max_frames:
            fr = pe_cam.data.output["rgb"][:G, ..., :3].to(torch.uint8).cpu().numpy()[:, ::2, ::2]
            writer.append_data(make_grid(fr, cols)); vid_n += 1
        if SV:   # buffer each env's current frame (downscaled 2x) for per-episode success videos
            fr = pe_cam.data.output["rgb"][:, ::2, ::2, :3].to(torch.uint8).cpu().numpy()  # [N,h,w,3]
            for i in range(args.num_envs):
                if len(benv[i]) < args.success_max_len:
                    benv[i].append(fr[i])
        a = torch.from_numpy(plan[:, min(age, plan.shape[1] - 1)]).to(env.device)
        _, _, terminated, truncated, _ = env.step(a)
        done = terminated | truncated
        nd = env.nail_driven
        ndone = torch.nonzero(done).flatten().tolist()
        if ndone:
            attempts += len(ndone)
            successes += int(nd[done].sum().item())
            force = True
            if SV:
                for i in ndone:
                    if bool(nd[i]) and saved_succ < SV and len(benv[i]) > 5:
                        out = _save_succ(benv[i], saved_succ); saved_succ += 1
                        print(f"[client] success video {saved_succ}/{SV}: env {i}, {len(benv[i])} frames -> {out}", flush=True)
                    benv[i] = []   # clear (success saved or failure discarded)
                base.sim.render(); base.scene.update(base.physics_dt)  # frame-alignment after reset
        age += 1; steps += 1
        if args.log_goal_idx:
            goal_idx_max = max(goal_idx_max, int(base.successes.max().item()))
        if steps % 100 == 0:
            msg = (f"[client] step={steps} episodes={attempts} success={successes} "
                   f"rate={successes/max(1,attempts):.0%}" + (f" succ_vids={saved_succ}/{SV}" if SV else ""))
            if args.log_goal_idx:
                gi = base.successes
                msg += (f" | goal_idx now[min={int(gi.min())} mean={float(gi.mean()):.2f} max={int(gi.max())}]"
                        f" run_max={goal_idx_max}/{base._traj_T}")
            print(msg, flush=True)

    print(f"\n[client] DONE: {successes}/{attempts} episodes succeeded "
          f"(success_rate {successes/max(1,attempts):.1%}) over {steps} control steps"
          + (f" | max goal index reached this run = {goal_idx_max}/{base._traj_T} "
             f"(goal {'ADVANCES' if goal_idx_max > 0 else 'STUCK at 0'} during eval)" if args.log_goal_idx else ""), flush=True)
    if writer is not None:
        writer.close()
        print(f"[client] wrote {vid_n} grid frames -> {REPO}/videos/eval_simtoolreal_hammer.mp4", flush=True)
    if SV:
        print(f"[client] saved {saved_succ} success episode videos -> {REPO}/videos/eval_simtoolreal_success_*.mp4", flush=True)
    try:
        send_msg(sock, {"cmd": "close"})
    except Exception:
        pass
    sock.close(); env.close(); app.close()


if __name__ == "__main__":
    main()
