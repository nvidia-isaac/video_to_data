"""Eval the SimToolReal-style specialist trained on VEGA-RETARGET data (state-only, 29-dim delta-joint),
on the Vega bimanual scene. Mirrors eval_simtoolreal_client.py but:
  - the env is the Vega retarget scene; the RIGHT arm+hand is driven DIRECTLY by the BC delta
    (cur_targets = clamp(joint_pos + delta)) -- NO shadow/IK at eval (the BC policy IS the controller);
  - the LEFT arm holds the thread_tester via the pretrained SAPG policy (the same 2nd-instance retarget
    used at collection), for scene fidelity -- skip with --no_left_hold (left then parks);
  - the state (palm-relative keypoints + 109/80 proprio) is the SAME compute_simtoolreal_obs used by the
    collector, so train/eval inputs match. Success = nail_driven.

Run: server (GR00T venv, port 5602) then this client (isaaclab.sh) with --no_joint_vel --replan 1.
"""
import argparse
import math
import os
import socket
import struct
import pickle
import time

from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"
DEFAULT_CONFIG = f"{ORIG_REPO}/pretrained_policy/config.yaml"
DEFAULT_CKPT = f"{ORIG_REPO}/pretrained_policy/model.pth"
REPO = "/home/cning/simtoolreal_isaaclab"
LOG_DIR = f"{REPO}/logs/simtoolreal"

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=5602)
parser.add_argument("--num_envs", type=int, default=25)
parser.add_argument("--episodes", type=int, default=400)
parser.add_argument("--replan", type=int, default=1, help="re-query the specialist every N control steps (1 = closed-loop, best)")
parser.add_argument("--no_joint_vel", action="store_true", help="drop joint_vel [29:58] (109->80) to match a --no_joint_vel checkpoint")
parser.add_argument("--with_goal", action="store_true")
parser.add_argument("--max_ep_steps", type=int, default=1200)
parser.add_argument("--max_steps", type=int, default=200000)
parser.add_argument("--success_joint", type=float, default=-0.006)
parser.add_argument("--success_tolerance", type=float, default=0.03)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no_left_hold", action="store_true", help="park the left arm instead of holding the thread_tester (skips building the pretrained player)")
parser.add_argument("--orig_config", default=DEFAULT_CONFIG); parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
parser.add_argument("--video", action="store_true"); parser.add_argument("--video_envs", type=int, default=9)
parser.add_argument("--video_max_frames", type=int, default=1200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = bool(args.video)
app = AppLauncher(args).app

import sys  # noqa: E402
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import simtoolreal_lab.tasks  # noqa: E402, F401
from simtoolreal_lab.tasks.vega_hammer_retarget.vega_hammer_retarget_env import VegaHammerRetargetEnv  # noqa: E402
from simtoolreal_lab.tasks.vega_hammer_retarget.vega_hammer_retarget_env_cfg import VegaHammerRetargetEnvCfg  # noqa: E402
from simtoolreal_lab.tasks.simtoolreal.keypoint_utils import (  # noqa: E402
    compute_simtoolreal_obs, compute_goal_rel, find_screw_body, screw_offsets,
)
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        ch = sock.recv(n - len(buf))
        if not ch:
            raise ConnectionError("server closed")
        buf += ch
    return buf


def send_msg(sock, obj):
    b = pickle.dumps(obj); sock.sendall(struct.pack("!I", len(b)) + b)


def recv_msg(sock):
    n = struct.unpack("!I", recvall(sock, 4))[0]
    return pickle.loads(recvall(sock, n))


class BCEvalVegaRetargetEnv(VegaHammerRetargetEnv):
    """RIGHT arm/hand driven by the BC delta (cur_targets = clamp(joint_pos+delta)); LEFT arm holds the
    thread_tester via the 2nd pretrained policy (if attached). No shadow/IK for the right at eval."""

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        cur = torch.clamp(self.joint_pos + actions, self.dof_lower, self.dof_upper)   # right canonical
        self.cur_targets = cur
        self.prev_targets = cur
        self.expert_targets = cur.clone()
        if getattr(self.cfg, "left_hold", False) and self._left_player is not None:
            self._step_left_instance()


def build_agent_cfg(num_envs):
    with open(args.orig_config) as f:
        params = yaml.safe_load(f)["train"]["params"]
    params["seed"] = args.seed; params["load_checkpoint"] = True; params["load_path"] = args.checkpoint
    c = params["config"]
    c["name"] = "00_vega_eval"; c["device_name"] = c["device"] = "cuda:0"; c["multi_gpu"] = False
    c["num_actors"] = num_envs; c["clip_actions"] = False; c["max_epochs"] = 1; c["max_frames"] = 100
    c["expl_coef_block_size"] = max(1, num_envs // 6); c["minibatch_size"] = num_envs
    if "central_value_config" in c:
        c["central_value_config"]["minibatch_size"] = num_envs
    c["train_dir"] = LOG_DIR
    c["defer_summaries_sec"] = 5; c["summaries_interval_sec_min"] = 5; c["summaries_interval_sec_max"] = 300
    c.setdefault("player", {}); c["player"]["games_num"] = 128; c["player"]["deterministic"] = True
    return {"params": params}


def main():
    torch.manual_seed(args.seed)
    sock = None
    for _ in range(120):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); sock.connect((args.host, args.port))
            send_msg(sock, {"cmd": "ping"})
            if recv_msg(sock).get("ok"):
                break
        except (ConnectionRefusedError, OSError):
            if sock:
                sock.close()
            sock = None; time.sleep(1)
    else:
        raise RuntimeError(f"could not reach specialist server at {args.host}:{args.port}")
    print(f"[client] connected to specialist server {args.host}:{args.port}", flush=True)

    cfg = VegaHammerRetargetEnvCfg()
    cfg.seed = args.seed; cfg.scene.num_envs = args.num_envs
    cfg.per_env_camera = bool(args.video)
    cfg.episode_length_s = args.max_ep_steps / 60.0
    cfg.terminate_on_nail_driven = args.success_joint
    cfg.success_tolerance = args.success_tolerance
    if args.no_left_hold:
        cfg.left_hold = False

    env = BCEvalVegaRetargetEnv(cfg, render_mode="rgb_array" if args.video else None)
    base = env
    # LEFT-hold pretrained policy (2nd instance), via rl_games -- same as deploy/collect
    if not args.no_left_hold:
        wrapped = RlGamesVecEnvWrapper(env, "cuda:0", math.inf, math.inf)
        vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
        runner = Runner(IsaacAlgoObserver()); runner.load(build_agent_cfg(args.num_envs)); runner.reset()
        pl = runner.create_player(); pl.restore(args.checkpoint); pl.has_batch_dimension = True
        if pl.is_rnn:
            pl.init_rnn()
        base.attach_left_policy(pl)
        print("[client] LEFT-hand thread_tester-hold policy attached", flush=True)

    screw_off = screw_offsets(base); screw_body_idx = find_screw_body(base)

    def query():
        kp_rel, proprio = compute_simtoolreal_obs(base, screw_off, screw_body_idx)
        if args.no_joint_vel:
            proprio = torch.cat([proprio[:, :29], proprio[:, 58:]], dim=-1)
        if args.with_goal:
            proprio = torch.cat([proprio, compute_goal_rel(base)], dim=-1)
        send_msg(sock, {"cmd": "act", "keypoints": kp_rel.detach().cpu().numpy().astype(np.float32),
                        "proprio": proprio.detach().cpu().numpy().astype(np.float32)})
        return recv_msg(sock)["action"]   # [N, horizon, 29]

    # --- optional video: buffer the per-env camera frames, save ONE labelled mp4 per finished episode
    # (GREEN border/SUCCESS = nail driven; RED/FAILURE otherwise). Buffers only the first `video_envs`
    # envs; episode count is then driven by saved videos so we get exactly --episodes clips. -----------
    VID = bool(args.video)
    nve = 0
    if VID:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        vid_dir = f"{REPO}/videos/vega_eval_specialist"
        os.makedirs(vid_dir, exist_ok=True)
        pe_cam = base.scene.sensors["per_env_cam"]
        nve = min(args.video_envs, args.num_envs)
        bufs = [[] for _ in range(nve)]
        FPS = 30
        _font = [None]

        def _label(fr, text, color):
            if _font[0] is None:
                try:
                    _font[0] = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, fr.shape[0] // 18))
                except Exception:
                    _font[0] = ImageFont.load_default()
            im = Image.fromarray(np.ascontiguousarray(fr)); d = ImageDraw.Draw(im)
            H, W = fr.shape[:2]; bw = max(3, H // 80)
            d.rectangle([0, 0, W - 1, H - 1], outline=color, width=bw)
            x, y = max(2, W // 40), max(2, H // 30)
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                d.text((x + dx, y + dy), text, font=_font[0], fill=(0, 0, 0))
            d.text((x, y), text, font=_font[0], fill=color)
            return np.asarray(im)

        def _save_ep(frames, ok, idx):
            color = (60, 220, 60) if ok else (235, 60, 60)
            tag = "SUCCESS" if ok else "FAILURE"
            path = f"{vid_dir}/ep_{idx:03d}_{'success' if ok else 'fail'}.mp4"
            w = imageio.get_writer(path, fps=FPS, macro_block_size=None, codec="libx264")
            for fr in frames:
                w.append_data(_label(fr, f"ep {idx}  {tag}", color))
            w.close()
            print(f"[client]  video {idx:03d} -> {path}  ({len(frames)} frames, {tag})", flush=True)

    env.reset()
    base.scene.update(base.physics_dt)
    plan, age, force = None, 0, True
    successes = attempts = steps = vids_saved = 0
    K = args.replan
    target = args.episodes
    print(f"[client] vega-retarget rollout: N={args.num_envs} envs, replan {K}, target {target} eps, "
          f"tol {args.success_tolerance}, max_ep {args.max_ep_steps}"
          + (f", VIDEO {nve} envs -> {vid_dir}" if VID else ""), flush=True)
    while (vids_saved if VID else attempts) < target and steps < args.max_steps:
        if force or age >= K or plan is None:
            plan = query(); age = 0; force = False
        a = torch.from_numpy(plan[:, min(age, plan.shape[1] - 1)]).to(env.device)
        _, _, terminated, truncated, _ = env.step(a)
        done = terminated | truncated
        nd = env.nail_driven
        if VID:
            rgb = pe_cam.data.output["rgb"][:nve, ..., :3].to(torch.uint8).cpu().numpy()
            for i in range(nve):
                if len(bufs[i]) < args.video_max_frames:
                    bufs[i].append(rgb[i])
        ndone = torch.nonzero(done).flatten().tolist()
        if ndone:
            attempts += len(ndone); successes += int(nd[done].sum().item()); force = True
            if VID:
                for i in ndone:
                    if i < nve:
                        if vids_saved < target:
                            _save_ep(bufs[i], bool(nd[i].item()), vids_saved); vids_saved += 1
                        bufs[i] = []
        age += 1; steps += 1
        if steps % 100 == 0:
            print(f"[client] step={steps} eps={attempts} success={successes} "
                  f"rate={successes/max(1,attempts):.0%}"
                  + (f" vids={vids_saved}/{target}" if VID else ""), flush=True)
    print(f"\n[client] DONE: {successes}/{attempts} episodes succeeded "
          f"(success_rate {successes/max(1,attempts):.1%}) over {steps} steps"
          + (f"; saved {vids_saved} videos -> {vid_dir}" if VID else ""), flush=True)
    try:
        send_msg(sock, {"cmd": "close"})
    except Exception:
        pass
    sock.close(); env.close(); app.close()


if __name__ == "__main__":
    main()
