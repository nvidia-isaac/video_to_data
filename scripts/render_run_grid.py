"""Render a 5x5 grid video of 25 envs, each running ONE episode, that FREEZES + marks each tile
SUCCESS (green) / FAIL (red) the moment that env's episode ends. Ends when all 25 have finished.

Client for eval_simtoolreal_server.py (state specialist). Run through isaaclab.sh; start the server
first (GR00T venv) with the chosen checkpoint.
  ./isaaclab.sh -p scripts/render_run_grid.py --headless --num_envs 25 --no_joint_vel --table_dist 0.15 \
      --port 5602 --out videos/run_grid_best.mp4
"""
import argparse, math, pickle, socket, struct, time
from isaaclab.app import AppLauncher

REPO = "/home/cning/simtoolreal_isaaclab"
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--max_steps", type=int, default=1100, help="hard cap (episodes truncate at max_ep_steps, so all finish well under this)")
parser.add_argument("--max_ep_steps", type=int, default=800)
parser.add_argument("--replan", type=int, default=1)
parser.add_argument("--success_joint", type=float, default=-0.006)
parser.add_argument("--no_joint_vel", action="store_true")
parser.add_argument("--table_dist", type=float, default=0.15)
parser.add_argument("--downscale", type=int, default=2, help="per-env frame downscale before tiling")
parser.add_argument("--hold_frames", type=int, default=75, help="frames to hold the final marked grid")
parser.add_argument("--out", default=f"{REPO}/videos/run_grid_best.mp4")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5602)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import sys  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import cv2  # noqa: E402

sys.path.insert(0, REPO)
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402
from simtoolreal_lab.tasks.simtoolreal.keypoint_utils import (  # noqa: E402
    compute_simtoolreal_obs, find_screw_body, screw_offsets,
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


def mark_tile(img, ok):
    """Draw a SUCCESS (green) / FAIL (red) border + label on a tile (in place-ish)."""
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    color = (40, 200, 40) if ok else (220, 40, 40)   # RGB
    t = max(4, h // 28)
    img[:t] = color; img[-t:] = color; img[:, :t] = color; img[:, -t:] = color
    label = "SUCCESS" if ok else "FAIL"
    cv2.putText(img, label, (t + 4, h - t - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


class BCEvalHammerEnv(HammerEnv):
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions
        cur = torch.clamp(self.joint_pos + actions, self.dof_lower, self.dof_upper)
        self.cur_targets = cur
        self.prev_targets = cur


def make_cfg():
    cfg = HammerEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.per_env_camera = True
    cfg.episode_length_s = args.max_ep_steps / 60.0
    cfg.terminate_on_nail_driven = args.success_joint
    cfg.table_dist = args.table_dist
    return cfg


def main():
    torch.manual_seed(args.seed)
    sock = None
    for _ in range(120):
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
        raise RuntimeError(f"could not reach server at {args.host}:{args.port}")
    print(f"[grid] connected to server {args.host}:{args.port}", flush=True)

    env = BCEvalHammerEnv(make_cfg())
    base = env
    screw_off = screw_offsets(base)
    screw_body_idx = find_screw_body(base)
    N = args.num_envs
    cols = int(math.ceil(math.sqrt(N)))
    DS = args.downscale

    def query():
        kp_rel, proprio = compute_simtoolreal_obs(base, screw_off, screw_body_idx)
        if args.no_joint_vel:
            proprio = torch.cat([proprio[:, :29], proprio[:, 58:]], dim=-1)
        kp = kp_rel.detach().cpu().numpy().astype(np.float32)
        pr = proprio.detach().cpu().numpy().astype(np.float32)
        send_msg(sock, {"cmd": "act", "keypoints": kp, "proprio": pr})
        return recv_msg(sock)["action"]

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import imageio.v2 as imageio
    writer = imageio.get_writer(args.out, fps=30, macro_block_size=None, codec="libx264")
    pe_cam = base.scene.sensors["per_env_cam"]

    env.reset()
    for _ in range(40):
        base.sim.render()
    base.scene.update(base.physics_dt)

    done = np.zeros(N, dtype=bool)       # episode-1 finished?
    success = np.zeros(N, dtype=bool)
    frozen = [None] * N                  # marked frozen tile per finished env
    plan, age = None, 0
    K = args.replan
    step = 0
    print(f"[grid] {N} envs, one episode each, replan {K} -> {args.out}", flush=True)
    while not done.all() and step < args.max_steps:
        if plan is None or age % K == 0:
            plan = query(); age = 0
        # capture frames at the START of the step (pre auto-reset terminal-ish state)
        fr = pe_cam.data.output["rgb"][:N, ..., :3].to(torch.uint8).cpu().numpy()[:, ::DS, ::DS]  # [N,h,w,3]
        a = torch.from_numpy(plan[:, min(age, plan.shape[1] - 1)]).to(env.device)
        _, _, terminated, truncated, _ = env.step(a)
        ep_done = (terminated | truncated).cpu().numpy()
        nd = base.nail_driven.cpu().numpy()          # success = nail driven (read pre-reset, as in eval client)
        tiles = []
        for i in range(N):
            if done[i]:
                tiles.append(frozen[i])
            elif ep_done[i]:                          # newly finished -> freeze + mark this frame
                done[i] = True; success[i] = bool(nd[i])
                frozen[i] = mark_tile(fr[i], success[i])
                tiles.append(frozen[i])
            else:
                tiles.append(fr[i])
        writer.append_data(make_grid(np.stack(tiles), cols))
        age += 1; step += 1
        if step % 100 == 0:
            print(f"[grid] step={step} finished={int(done.sum())}/{N} success={int(success[done].sum())}", flush=True)

    # any env that never finished (shouldn't happen): mark FAIL on its last frame
    for i in range(N):
        if frozen[i] is None:
            frozen[i] = mark_tile(fr[i], False)
    final = make_grid(np.stack(frozen), cols)
    for _ in range(args.hold_frames):
        writer.append_data(final)
    writer.close()
    nsucc = int(success.sum())
    print(f"[grid] DONE: {nsucc}/{N} envs succeeded over {step} steps -> {args.out}", flush=True)
    try:
        send_msg(sock, {"cmd": "close"})
    except Exception:
        pass
    sock.close(); env.close(); app.close()


if __name__ == "__main__":
    main()
