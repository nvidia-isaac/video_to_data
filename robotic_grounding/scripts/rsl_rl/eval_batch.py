"""Batch eval: one Isaac Sim session, multiple checkpoints.

Iterates over all model_*.pt files in a checkpoint directory (sorted by iteration),
runs --eval_episodes episodes for each, and prints a summary table.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Batch eval across multiple checkpoints.")
parser.add_argument(
    "--checkpoints_dir",
    type=str,
    required=True,
    help="Directory containing model_*.pt files to evaluate in order.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=100,
    help="Number of completed episodes to collect per checkpoint.",
)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--motion_file", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import os
import re
import torch

import gymnasium as gym

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import robotic_grounding.tasks  # noqa: F401
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from viewer_utils import autoframe_viewer

from isaaclab_tasks.utils.hydra import hydra_task_config


def _itr(path):
    m = re.search(r"model_(\d+)\.pt", os.path.basename(path))
    return int(m.group(1)) if m else -1


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    # Find and sort checkpoints
    ckpts = sorted(
        glob.glob(os.path.join(args_cli.checkpoints_dir, "model_*.pt")), key=_itr
    )
    if not ckpts:
        print(f"[eval_batch] No model_*.pt found in {args_cli.checkpoints_dir}")
        return

    print(
        f"[eval_batch] Found {len(ckpts)} checkpoints: "
        f"{os.path.basename(ckpts[0])} → {os.path.basename(ckpts[-1])}"
    )

    # Configure env
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.motion_file = args_cli.motion_file
    if env_cfg.motion_file is not None:
        scene_cfg = SceneConfig.from_motion_file(env_cfg.motion_file)
        apply_scene_config(env_cfg, scene_cfg)
        autoframe_viewer(env_cfg, scene_cfg.motion_file)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Build runner (loads first checkpoint to initialise architecture)
    agent_cfg.load_checkpoint = ckpts[0]
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )

    _cmd = env.unwrapped.command_manager.get_term("dual_hands_object_tracking_command")
    _warmup = getattr(_cmd.cfg, "virtual_object_control_decay_steps", 20)
    traj_len = _cmd.retargeted_horizon
    num_envs = env.unwrapped.num_envs

    results = []  # list of (iter, mean_ratio, std_ratio, n_full, n_total)

    for ckpt_path in ckpts:
        iteration = _itr(ckpt_path)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        # Reset tracking state
        obs = env.get_observations()
        my_ep_len = torch.zeros(
            num_envs, dtype=torch.float32, device=env.unwrapped.device
        )
        my_ep_tracking_len = _cmd.tracking_lengths.clone().float().squeeze(-1)
        completed: list[tuple[float, float]] = []

        while len(completed) < args_cli.eval_episodes:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)
            my_ep_len += 1
            done_mask = dones.bool()
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
                for i in done_ids:
                    ep_len = my_ep_len[i].item()
                    track_len = my_ep_tracking_len[i].item()
                    completed.append((ep_len, track_len))
                    my_ep_len[i] = 0.0
                    my_ep_tracking_len[i] = _cmd.tracking_lengths[i].float()

        data = completed[: args_cli.eval_episodes]
        _usable = max(traj_len - _warmup, 1)
        ratios = [min(max(e[0] - _warmup, 0), _usable) / _usable for e in data]
        n_full = sum(1 for r in ratios if r >= 0.99)
        mean_r = sum(ratios) / len(ratios)
        std_r = (
            sum((x - mean_r) ** 2 for x in ratios) / max(len(ratios) - 1, 1)
        ) ** 0.5
        results.append((iteration, mean_r, std_r, n_full, len(data)))
        print(
            f"  step={iteration:6d}  ratio={mean_r:.3f}±{std_r:.3f}  "
            f"full={n_full}/{len(data)} ({100*n_full/len(data):.0f}%)",
            flush=True,
        )

    print("\n[eval_batch] ===== Full Table =====")
    print(
        f"  Trajectory: {traj_len} steps, warmup: {_warmup}, episodes/ckpt: {args_cli.eval_episodes}"
    )
    print(f"  {'Step':>7}  {'Ratio mean':>11}  {'Ratio std':>10}  {'Full%':>6}")
    print(f"  {'-'*7}  {'-'*11}  {'-'*10}  {'-'*6}")
    for itr, mr, sr, nf, nt in results:
        print(f"  {itr:>7d}  {mr:>11.3f}  {sr:>10.3f}  {100*nf/nt:>5.1f}%")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
