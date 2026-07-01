"""DAgger CLIENT for the SimToolReal specialist (run through isaaclab.sh).

The "expert supervises the specialist on its own online rollout" loop (DAgger / dataset aggregation).
Per iteration i, with mixing coefficient beta_i (1 -> pure expert at iter 0, decaying toward 0):
  1. Roll out the hammer env. At each step, query the LEARNER (dagger_server.act) for its action AND
     the SAPG EXPERT (rl_games player) for its action; each env is driven by the expert w.p. beta_i,
     else by the learner -> the state distribution shifts toward the LEARNER's as beta decays.
  2. RELABEL every visited state with the EXPERT's would-be delta-joint target
     (env._compute_targets(a_expert, prev_targets) - joint_pos -- the same delta space the specialist
     outputs / the BC collector recorded), computed WITHOUT disturbing the rollout.
  3. Aggregate the relabelled episodes into the server's growing buffer (D <- D u D_i) and retrain.
This fixes BC's compounding-error failure: the expert teaches the learner how to recover from the
off-distribution states the learner actually drifts into.

Run (server first, in GR00T venv), then:
  source ~/isaaclab/env_isaaclab/bin/activate && cd ~/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -p ~/simtoolreal_isaaclab/scripts/dagger_client.py \
      --headless --num_envs 60 --iters 6 --episodes_per_iter 60 --train_steps 2000 --port 5603
"""

import argparse
import math
import pickle
import socket
import struct
import time

from isaaclab.app import AppLauncher

ORIG_REPO = "/home/cning/simtoolreal"
REPO = "/home/cning/simtoolreal_isaaclab"
DEFAULT_CONFIG = f"{ORIG_REPO}/pretrained_policy/config.yaml"
DEFAULT_CKPT = f"{ORIG_REPO}/pretrained_policy/model.pth"
LOG_DIR = f"{REPO}/logs/simtoolreal"

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=60, help="parallel rollouts (multiple of 6 for the SAPG net)")
parser.add_argument("--iters", type=int, default=6, help="DAgger iterations (aggregate + retrain)")
parser.add_argument("--episodes_per_iter", type=int, default=60, help="completed episodes to collect each iter")
parser.add_argument("--rollout_step_cap", type=int, default=4000, help="safety cap on control steps per iter")
parser.add_argument("--beta_start", type=float, default=1.0, help="mixing coef at iter 0 (1=pure expert -> BC)")
parser.add_argument("--beta_decay", type=float, default=0.5, help="beta_i = max(beta_min, beta_start*decay**i)")
parser.add_argument("--beta_step", type=float, default=0.0, help="if >0, use LINEAR beta decay instead: beta_i = max(beta_min, beta_start - i*beta_step) (e.g. start 1.0, step 0.1, iters 10 -> 1.0,0.9,...,0.1). 0 = exponential")
parser.add_argument("--beta_min", type=float, default=0.0)
parser.add_argument("--train_steps", type=int, default=2000, help="gradient steps per iter (on the server)")
parser.add_argument("--train_batch", type=int, default=128)
parser.add_argument("--train_lr", type=float, default=1e-4)
parser.add_argument("--max_ep_steps", type=int, default=800)
parser.add_argument("--success_joint", type=float, default=-0.006)
parser.add_argument("--orig_config", type=str, default=DEFAULT_CONFIG)
parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
parser.add_argument("--name", type=str, default="hammer_dagger")
parser.add_argument("--no_joint_vel", action="store_true", help="drop joint_vel [29:58] from proprio (109->80) to match a --no_joint_vel-trained learner")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5603)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False   # state-based: no images -> no rendering

app = AppLauncher(args_cli).app

import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, REPO)
import gymnasium as gym  # noqa: E402
import simtoolreal_lab.tasks  # noqa: E402, F401
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env import HammerEnv  # noqa: E402
from simtoolreal_lab.tasks.hammer.hammer_env_cfg import HammerEnvCfg  # noqa: E402
from simtoolreal_lab.tasks.simtoolreal.keypoint_utils import (  # noqa: E402
    compute_simtoolreal_obs, find_screw_body, screw_offsets,
)

GYM_ID = "Isaac-SimToolReal-Hammer-Direct-v0"


# ---- socket helpers ----
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
    return None if hdr is None else pickle.loads(recvall(sock, struct.unpack(">I", hdr)[0]))


def build_agent_cfg(num_envs):
    """Rebuild the SAPG net from the policy config.yaml so the expert checkpoint loads bit-for-bit."""
    with open(args_cli.orig_config) as f:
        params = yaml.safe_load(f)["train"]["params"]
    params["seed"] = args_cli.seed
    params["load_checkpoint"] = True
    params["load_path"] = args_cli.checkpoint
    c = params["config"]
    c["name"] = "00_dagger"
    c["device_name"] = c["device"] = "cuda:0"
    c["multi_gpu"] = False
    c["num_actors"] = num_envs
    c["clip_actions"] = False
    c["max_epochs"] = 1
    c["max_frames"] = 100
    c["expl_coef_block_size"] = max(1, num_envs // 6)
    c["minibatch_size"] = num_envs
    if "central_value_config" in c:
        c["central_value_config"]["minibatch_size"] = num_envs
    c["train_dir"] = LOG_DIR
    c.setdefault("player", {})
    c["player"]["games_num"] = 128
    c["player"]["deterministic"] = True
    return {"params": params}


def make_cfg():
    """Hammer env in pretrained-deploy mode (expert obs convention), state-based (no camera)."""
    cfg = HammerEnvCfg()                       # deploy defaults baked into __post_init__
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.pretrained_compat = True
    cfg.eval_append_expl_coef = True           # SAPG player expects the exploit coef appended at idx 140
    cfg.domain_randomization = False
    cfg.use_tolerance_curriculum = False
    cfg.success_steps = 1
    cfg.success_tolerance = 0.01
    cfg.max_consecutive_successes = 0
    cfg.episode_length_s = args_cli.max_ep_steps / 60.0
    cfg.reset_dof_pos_noise_arm = 0.0
    cfg.reset_dof_pos_noise_fingers = 0.0
    cfg.reset_position_noise_x = cfg.reset_position_noise_y = cfg.reset_position_noise_z = 0.0
    cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_2"] = 1.571 - math.radians(10)
    cfg.robot_cfg.init_state.joint_pos["iiwa14_joint_4"] = 1.376 + math.radians(10)
    cfg.per_env_camera = False                 # DAgger is state-based -> skip rendering
    cfg.use_fixed_goal_trajectory = False
    cfg.use_tighten_goals = True
    cfg.randomize_layout = True
    cfg.physical_screw = True
    cfg.screw_contact_clearance = -0.04
    cfg.terminate_on_nail_driven = args_cli.success_joint
    cfg.terminate_on_screw_rotated = None
    cfg.sim.physx.gpu_collision_stack_size = 2 ** 30
    return cfg


class DaggerHammerEnv(HammerEnv):
    """The DAgger loop computes the chosen joint targets per env; the env just applies them."""

    def _pre_physics_step(self, chosen_targets: torch.Tensor) -> None:
        self.actions = chosen_targets
        self.cur_targets = torch.clamp(chosen_targets, self.dof_lower, self.dof_upper)
        self.prev_targets = self.cur_targets


def main():
    torch.manual_seed(args_cli.seed)
    N = args_cli.num_envs

    # ---- connect to the learner/training server ----
    sock = None
    for _ in range(180):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((args_cli.host, args_cli.port))
            send_msg(sock, {"cmd": "ping"})
            if recv_msg(sock).get("ok"):
                break
        except (ConnectionRefusedError, OSError):
            if sock:
                sock.close()
            sock = None
            time.sleep(1)
    else:
        raise RuntimeError(f"could not reach dagger server at {args_cli.host}:{args_cli.port}")
    print(f"[dagger] connected to server {args_cli.host}:{args_cli.port}", flush=True)

    # ---- env + SAPG expert player ----
    agent_cfg = build_agent_cfg(N)
    env = gym.make(GYM_ID, cfg=make_cfg(), render_mode=None)
    env.unwrapped.__class__ = DaggerHammerEnv     # use the externally-driven _pre_physics_step
    env = RlGamesVecEnvWrapper(env, "cuda:0", math.inf, math.inf)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    runner = Runner(IsaacAlgoObserver()); runner.load(agent_cfg); runner.reset()
    player = runner.create_player(); player.restore(args_cli.checkpoint); player.has_batch_dimension = True
    base = env.unwrapped
    dev = base.device
    screw_off = screw_offsets(base)
    sbi = find_screw_body(base)
    ckpt_dir = f"{REPO}/logs/gr00t_specialist/{args_cli.name}"; os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/{args_cli.name}.pt"

    def learner_action():
        kp_rel, proprio = compute_simtoolreal_obs(base, screw_off, sbi)
        if args_cli.no_joint_vel:   # drop joint_vel [29:58] -> 80, matching the learner (used for BOTH act + the recorded buffer proprio)
            proprio = torch.cat([proprio[:, :29], proprio[:, 58:]], dim=-1)
        send_msg(sock, {"cmd": "act", "keypoints": kp_rel.detach().cpu().numpy().astype(np.float32),
                        "proprio": proprio.detach().cpu().numpy().astype(np.float32)})
        chunk = recv_msg(sock)["action"]                              # [N,40,29] denorm deltas
        return kp_rel, proprio, torch.from_numpy(chunk[:, 0]).to(dev) # closed-loop: first action

    obs = env.reset(); o = obs["obs"]
    if player.is_rnn:
        player.init_rnn()

    for it in range(args_cli.iters):
        if args_cli.beta_step > 0:   # LINEAR schedule: 1.0, 0.9, 0.8, ... (gradual distribution shift)
            beta = max(args_cli.beta_min, args_cli.beta_start - it * args_cli.beta_step)
        else:                        # exponential (default)
            beta = max(args_cli.beta_min, args_cli.beta_start * (args_cli.beta_decay ** it))
        bkp = [[] for _ in range(N)]; bpro = [[] for _ in range(N)]; blab = [[] for _ in range(N)]
        episodes = []
        ep_done = ep_succ = steps = 0
        while ep_done < args_cli.episodes_per_iter and steps < args_cli.rollout_step_cap:
            js = base.joint_pos.clone()
            a_exp = player.get_action(o, is_deterministic=True).clamp(-1.0, 1.0)   # expert raw action
            expert_cur = base._compute_targets(a_exp, base.prev_targets)           # faithful expert target
            expert_label = expert_cur - js                                         # delta label (all envs)
            kp_rel, proprio, learner_delta = learner_action()
            learner_cur = torch.clamp(js + learner_delta, base.dof_lower, base.dof_upper)
            drive_expert = (torch.rand(N, device=dev) < beta).unsqueeze(1)
            chosen = torch.where(drive_expert, expert_cur, learner_cur)
            kp_np = kp_rel.detach().cpu().numpy(); pr_np = proprio.detach().cpu().numpy()
            lab_np = expert_label.detach().cpu().numpy()
            for i in range(N):
                bkp[i].append(kp_np[i]); bpro[i].append(pr_np[i]); blab[i].append(lab_np[i])
            obs, _, done, _ = env.step(chosen); o = obs["obs"]
            done_b = done.bool()
            if player.is_rnn and bool(done_b.any()):
                for s in player.states:
                    s[:, done_b, :] = 0.0
            for i in torch.nonzero(done_b).flatten().tolist():
                episodes.append({"kp": np.stack(bkp[i]), "proprio": np.stack(bpro[i]), "label": np.stack(blab[i])})
                ep_done += 1
                ep_succ += int(bool(base.nail_driven[i]))
                bkp[i] = []; bpro[i] = []; blab[i] = []
            steps += 1
        # flush partial (on-policy) episodes too -- all visited states are useful DAgger data
        for i in range(N):
            if len(bkp[i]) >= 2:
                episodes.append({"kp": np.stack(bkp[i]), "proprio": np.stack(bpro[i]), "label": np.stack(blab[i])})
        send_msg(sock, {"cmd": "add", "episodes": episodes})
        agg = recv_msg(sock)
        send_msg(sock, {"cmd": "train", "steps": args_cli.train_steps,
                        "batch_size": args_cli.train_batch, "lr": args_cli.train_lr})
        ml = recv_msg(sock)["mean_loss"]
        for sp in (f"{ckpt_dir}/{args_cli.name}_iter{it}.pt", ckpt_path):   # per-iter + latest
            send_msg(sock, {"cmd": "save", "path": sp, "step": (it + 1) * args_cli.train_steps}); recv_msg(sock)
        print(f"[dagger] iter {it} beta={beta:.3f} | rollout {ep_succ}/{ep_done} success "
              f"({ep_succ/max(1,ep_done):.0%}) | +{len(episodes)} eps -> buffer {agg['n_eps']} eps/"
              f"{agg['n_frames']} fr | train_loss={ml:.4f} -> {ckpt_path}", flush=True)

    print(f"[dagger] DONE: {args_cli.iters} iterations -> {ckpt_path}", flush=True)
    try:
        send_msg(sock, {"cmd": "close"})
    except Exception:
        pass
    sock.close(); env.close(); app.close()


if __name__ == "__main__":
    main()
