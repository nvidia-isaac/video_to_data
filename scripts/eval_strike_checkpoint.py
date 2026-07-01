"""Eval ONE strike-training checkpoint on the nail-driving DEPLOY env: 100 episodes, report the
nail_driven success rate, append a row to a CSV. --perturbation enables the full collection-recipe
perturbation set (tool/joint teleport + random-action bursts + force + goal-diversify + action noise).
No video -> fast. Used by scripts/monitor_strike_eval.sh (runs on GPU 1 while training uses GPU 0).
"""
import argparse, math
from isaaclab.app import AppLauncher

REPO = "/home/cning/simtoolreal_isaaclab"
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--agent_cfg", default="rl_games_sapg_cfg.yaml")
p.add_argument("--num_envs", type=int, default=25)
p.add_argument("--episodes", type=int, default=100)
p.add_argument("--max_steps", type=int, default=40000)
p.add_argument("--perturbation", action="store_true")
p.add_argument("--action_noise", type=float, default=0.1, help="player-action noise (only with --perturbation)")
p.add_argument("--step", type=int, default=0, help="training epoch of this checkpoint (x axis)")
p.add_argument("--csv", default=f"{REPO}/logs/strike_eval/strike_eval.csv")
p.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.enable_cameras = False
app = AppLauncher(args).app

import os, sys  # noqa: E402
sys.path.insert(0, REPO)
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import simtoolreal_lab.tasks  # noqa: E402,F401
from simtoolreal_lab.tasks.vega_hammer_right.vega_hammer_right_deploy_env_cfg import VegaHammerRightDeployEnvCfg  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

TASK = "Isaac-SimToolReal-Vega-Hammer-Right-Direct-v0"   # gym id (HammerEnv); cfg overridden below
AGENTS_DIR = f"{REPO}/simtoolreal_lab/tasks/simtoolreal/agents"


def main():
    torch.manual_seed(args.seed)
    cfg = VegaHammerRightDeployEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.per_env_camera = False                 # no video -> fast eval
    if args.perturbation:                      # full collection-recipe perturbation set
        cfg.domain_randomization = True
        cfg.force_perturbation = True
        cfg.tool_displacement = True
        cfg.tool_displace_pregrasp = True
        cfg.joint_displacement = True
        cfg.random_action = True
        cfg.random_action_prob = 0.007
        cfg.random_action_steps_std = 27.0
        cfg.goal_diversify = True

    with open(os.path.join(AGENTS_DIR, args.agent_cfg)) as f:
        agent_cfg = yaml.safe_load(f)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(TASK, cfg=cfg, render_mode=None)
    base = env.unwrapped
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["config"]["num_actors"] = base.num_envs
    agent_cfg["params"]["config"].setdefault("player", {})
    agent_cfg["params"]["config"]["player"]["deterministic"] = True
    runner = Runner(IsaacAlgoObserver()); runner.load(agent_cfg); runner.reset()
    player = runner.create_player(); player.restore(args.checkpoint); player.has_batch_dimension = True

    obs = env.reset(); o = obs["obs"]
    if player.is_rnn:
        player.init_rnn()
    N = base.num_envs
    successes = attempts = steps = 0
    an = args.action_noise if args.perturbation else 0.0
    print(f"[eval] step={args.step} perturb={int(args.perturbation)} N={N} target={args.episodes} eps", flush=True)
    while attempts < args.episodes and steps < args.max_steps:
        a = player.get_action(o, is_deterministic=True)
        if an > 0:
            a = (a + an * torch.randn_like(a)).clamp(-1.0, 1.0)
        obs, _, done, _ = env.step(a); o = obs["obs"]
        nd = base.nail_driven
        d = done.bool()
        if player.is_rnn and bool(d.any()):
            for s in player.states:
                s[:, d, :] = 0.0
        if bool(d.any()):
            attempts += int(d.sum().item()); successes += int(nd[d].sum().item())
        steps += 1
        if steps % 300 == 0:
            print(f"[eval] step={steps} eps={attempts} succ={successes} rate={successes/max(1,attempts):.0%}", flush=True)
    rate = successes / max(1, attempts)
    print(f"[eval] step={args.step} perturb={int(args.perturbation)} DONE {successes}/{attempts} = {rate:.1%}", flush=True)
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    newf = not os.path.exists(args.csv)
    with open(args.csv, "a") as f:
        if newf:
            f.write("step,perturbation,successes,attempts,rate\n")
        f.write(f"{args.step},{int(args.perturbation)},{successes},{attempts},{rate:.4f}\n")
    env.close(); app.close()


if __name__ == "__main__":
    main()
