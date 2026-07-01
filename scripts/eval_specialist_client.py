"""Env CLIENT for evaluating the GR00T DINOv3 specialist (run through isaaclab.sh).

Runs the Isaac Lab hammer env (same scene/visuals/physics as collect_bc_data.py, so the policy
sees in-distribution observations) and drives the robot with the specialist policy served by
eval_specialist_server.py (which must be running in the GR00T venv). Receding-horizon control:
every --replan steps it sends the current camera image + joint state to the server, gets a chunk
of delta joint targets, and applies them as cur_targets = joint_pos + delta. Reports the success
rate (screw seated -> nail_driven) over --episodes attempts.

The two halves run in separate processes/venvs because the model needs the GR00T venv and the
sim needs the isaaclab venv.

Run (server first, in GR00T venv), then:
  source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/eval_specialist_client.py \
      --headless --num_envs 25 --episodes 100 --replan 8 --port 5599
"""

import argparse
import math
import pickle
import socket
import struct

from isaaclab.app import AppLauncher

REPO = "/home/cning/simtoolreal_isaaclab"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--episodes", type=int, default=100, help="stop after this many episode attempts")
parser.add_argument("--max_steps", type=int, default=20000, help="hard cap on control steps")
parser.add_argument("--max_ep_steps", type=int, default=800, help="per-episode budget (time_out)")
parser.add_argument("--replan", type=int, default=8, help="re-query the policy every K control steps (receding horizon)")
parser.add_argument("--success_joint", type=float, default=-0.006)
parser.add_argument("--warmup", type=int, default=40)
parser.add_argument("--video", action="store_true", help="record a tiled grid of env cameras to videos/eval_image_hammer.mp4")
parser.add_argument("--video_envs", type=int, default=16, help="number of envs to tile in the recorded grid")
parser.add_argument("--video_max_frames", type=int, default=1500)
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5599)
parser.add_argument("--wrist", action="store_true", help="2nd palm-facing wrist view (match a wrist-trained policy)")
parser.add_argument("--wrist_eye", type=str, default="0.08,-0.02,0.08")
parser.add_argument("--wrist_lookat", type=str, default="-0.02,-0.015,0.18")
parser.add_argument("--wrist_up", type=str, default="Y")
parser.add_argument("--wrist_focal", type=float, default=14.0)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, REPO)
import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402


# ---- socket helpers (length-prefixed pickle) ----
def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def send_msg(sock, obj):
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_msg(sock):
    hdr = recvall(sock, 4)
    if hdr is None:
        return None
    return pickle.loads(recvall(sock, struct.unpack(">I", hdr)[0]))


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
    """Drive the robot with BC delta joint targets instead of the SimToolReal action mapping.

    The policy outputs delta = (commanded joint target - current joint pos), matching how
    collect_bc_data.py recorded actions. Deployment: cur_targets = clamp(joint_pos + delta).
    Reuses all of HammerEnv (camera, dones, nail_driven, auto-reset).
    """

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions  # consumed by obs (unused by the BC controller, kept for safety)
        cur = torch.clamp(self.joint_pos + actions, self.dof_lower, self.dof_upper)
        self.cur_targets = cur
        self.prev_targets = cur


def make_cfg():
    """Hammer env configured exactly like collect_bc_data.py (in-distribution for the policy)."""
    cfg = HammerEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 4.0
    cfg.pretrained_compat = True          # per-shape friction (grasp) + original conventions
    cfg.domain_randomization = False
    cfg.use_tolerance_curriculum = False
    cfg.success_steps = 1
    cfg.success_tolerance = 0.01
    cfg.max_consecutive_successes = 0
    cfg.episode_length_s = args.max_ep_steps / 60.0
    cfg.reset_dof_pos_noise_arm = 0.0
    cfg.reset_dof_pos_noise_fingers = 0.0
    cfg.reset_position_noise_x = cfg.reset_position_noise_y = cfg.reset_position_noise_z = 0.0
    cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_2"] = 1.571 - math.radians(10)
    cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_4"] = 1.376 + math.radians(10)
    if args.wrist:                       # 2nd view: palm-facing wrist camera (match the trained policy)
        cfg.wrist_camera = True
        cfg.wrist_cam_eye = tuple(float(v) for v in args.wrist_eye.split(","))
        cfg.wrist_cam_lookat = tuple(float(v) for v in args.wrist_lookat.split(","))
        cfg.wrist_cam_up = args.wrist_up
        cfg.wrist_cam_focal = args.wrist_focal
    # recording camera (matches training images: 640x480, same eye/lookat/visuals)
    cfg.per_env_camera = True
    cfg.cam_width, cfg.cam_height = 640, 480
    cfg.cam_eye = (0.0, -0.65, 0.85)
    cfg.cam_lookat = (0.0, 0.30, 0.55)
    cfg.cam_z_far = 2.5
    cfg.ground_color = (0.12, 0.12, 0.12)
    cfg.screw_color = (0.55, 0.72, 0.82)
    # hammer task (tighten goals, physical screw, terminate when seated)
    cfg.use_fixed_goal_trajectory = False
    cfg.use_tighten_goals = True
    cfg.randomize_layout = True
    cfg.physical_screw = True
    cfg.screw_contact_clearance = -0.04
    cfg.terminate_on_nail_driven = args.success_joint
    cfg.sim.physx.gpu_collision_stack_size = 2 ** 30
    return cfg


def main():
    import time
    torch.manual_seed(args.seed)
    sock = None
    for _ in range(90):  # server may still be loading the checkpoint; retry
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
        raise RuntimeError(f"could not reach policy server at {args.host}:{args.port}")
    print(f"[client] connected to policy server {args.host}:{args.port}", flush=True)

    env = BCEvalHammerEnv(make_cfg())
    N = args.num_envs
    pe_cam = env.scene.sensors["per_env_cam"]
    wrist_cam = env.scene.sensors["wrist_cam"] if args.wrist else None

    def query(joints):
        front = pe_cam.data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()   # [N,H,W,3]
        req = {"cmd": "act", "image": front, "state": joints}
        if wrist_cam is not None:
            req["image_wrist"] = wrist_cam.data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()
        send_msg(sock, req)
        return recv_msg(sock)["action"]  # [N, horizon, 29] denormalized deltas

    env.reset()
    for _ in range(max(1, args.warmup)):
        env.sim.render()
    env.scene.update(env.physics_dt)

    writer = None
    G = min(N, args.video_envs)
    cols = int(math.ceil(math.sqrt(G)))
    if args.video:
        import imageio.v2 as imageio, os
        os.makedirs(f"{REPO}/videos", exist_ok=True)
        writer = imageio.get_writer(f"{REPO}/videos/eval_image_hammer.mp4", fps=30,
                                    macro_block_size=None, codec="libx264")
    vid_n = 0
    plan = None
    age = 0
    force = True
    successes = attempts = steps = 0
    K = args.replan
    print(f"[client] rollout: N={N} envs, replan every {K} steps, target {args.episodes} episodes"
          + (f" | recording {G}-env grid ({cols} cols)" if args.video else ""), flush=True)
    while attempts < args.episodes and steps < args.max_steps and not (args.video and vid_n >= args.video_max_frames):
        if force or age >= K or plan is None:
            js = env.joint_pos.detach().cpu().numpy().astype(np.float32)             # [N,29]
            plan = query(js)
            age = 0; force = False
        if writer is not None and vid_n < args.video_max_frames:
            fr = pe_cam.data.output["rgb"][:G, ..., :3].to(torch.uint8).cpu().numpy()[:, ::2, ::2]
            writer.append_data(make_grid(fr, cols)); vid_n += 1
        a = torch.from_numpy(plan[:, min(age, plan.shape[1] - 1)]).to(env.device)
        _, _, terminated, truncated, _ = env.step(a)
        done = (terminated | truncated)
        nd = env.nail_driven
        ndone = torch.nonzero(done).flatten().tolist()
        if ndone:
            attempts += len(ndone)
            successes += int(nd[done].sum().item())
            force = True  # reset envs invalidate the plan -> re-query next step
        age += 1; steps += 1
        if steps % 100 == 0:
            print(f"[client] step={steps} episodes={attempts} success={successes} "
                  f"rate={successes/max(1,attempts):.0%}", flush=True)

    print(f"\n[client] DONE: {successes}/{attempts} episodes succeeded "
          f"(success_rate {successes/max(1,attempts):.1%}) over {steps} control steps", flush=True)
    if writer is not None:
        writer.close()
        print(f"[client] wrote {vid_n} grid frames -> {REPO}/videos/eval_image_hammer.mp4", flush=True)
    try:
        send_msg(sock, {"cmd": "close"})
    except Exception:
        pass
    sock.close()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
