"""Instrumented policy rollout: what does the trained policy actually do, per hand?"""
import argparse, os, sys
from isaaclab.app import AppLauncher
import cli_args  # noqa: E402  (scripts/rsl_rl on path)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--task", type=str, default="Sharpa-V2D-v0-Play")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--steps", type=int, default=520)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--random_start", action="store_true")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym, numpy as np, torch
import isaaclab_tasks  # noqa: F401
from robotic_grounding.tasks import *  # noqa: F401,F403
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.motion_file = args_cli.motion_file
    apply_scene_config(env_cfg, SceneConfig.from_motion_file(env_cfg.motion_file))
    c = env_cfg.commands.dual_hands_object_tracking_command
    c.always_reset_to_first_frame = not args_cli.random_start
    c.reset_to_first_frame_prob = 0.0
    c.initial_virtual_object_control_curriculum_scale = 0.0
    c.virtual_object_control_decay_steps = 0
    env_cfg.curriculum = None
    env_cfg.viewer.env_index = 0
    env_cfg.seed = agent_cfg.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    cmd = u.command_manager.get_term("dual_hands_object_tracking_command")
    obs, _ = env.reset()
    rec = {k: [] for k in ("oz", "ozc", "rwe", "lwe", "rfc", "lfc", "rd", "ld", "ract", "lact", "dz")}
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        if dones.any():
            print(f"[DIAG] terminated at step {len(rec['oz'])}: {int(dones.sum())} envs", flush=True)
        f = lambda t: float(t.float().mean().item())
        rec["oz"].append(f(cmd.object_position_e[:, 0, 2]))
        rec["ozc"].append(f(cmd.object_body_position_command_e[:, 0, 2]))
        rec["dz"].append(f(cmd.object_position_e[:, 0, 2] - cmd.object_body_position_command_e[:, 0, 2]))
        rec["rwe"].append(f(torch.norm(cmd.right_hand_wrist_position_e - cmd.right_hand_wrist_pose_command_e[:, :3], dim=-1)))
        rec["lwe"].append(f(torch.norm(cmd.left_hand_wrist_position_e - cmd.left_hand_wrist_pose_command_e[:, :3], dim=-1)))
        rec["rd"].append(f(torch.norm(cmd.right_hand_wrist_position_e - cmd.object_position_e[:, 0], dim=-1)))
        rec["ld"].append(f(torch.norm(cmd.left_hand_wrist_position_e - cmd.object_position_e[:, 0], dim=-1)))
        rec["rfc"].append(f((cmd.right_hand_contact_wrench_supports.amax(dim=1) > 0.01).float().mean(-1)))
        rec["lfc"].append(f((cmd.left_hand_contact_wrench_supports.amax(dim=1) > 0.01).float().mean(-1)))
        rec["ract"].append(f(cmd.right_hand_contact_active_command > 0.5))
        rec["lact"].append(f(cmd.left_hand_contact_active_command > 0.5))
    A = {k: np.array(v) for k, v in rec.items()}
    n = len(A["oz"])
    print(f"\n[DIAG] === {n} steps, {args_cli.num_envs} envs, VOC=0, start=frame 0 ===")
    print(f"[DIAG] object z   actual  min {A['oz'].min():.4f} max {A['oz'].max():.4f} range {A['oz'].ptp():.4f} m")
    print(f"[DIAG] object z   REFERENCE min {A['ozc'].min():.4f} max {A['ozc'].max():.4f} range {A['ozc'].ptp():.4f} m")
    print(f"[DIAG] lift ratio (actual range / reference range) = {A['oz'].ptp()/max(A['ozc'].ptp(),1e-9):.3f}")
    print(f"[DIAG] z(actual)-z(reference)  mean {A['dz'].mean()*100:+.2f} cm  p5 {np.percentile(A['dz'],5)*100:+.2f}  min {A['dz'].min()*100:+.2f} cm")
    print(f"[DIAG] steps more than 3cm BELOW reference: {(A['dz'] < -0.03).mean()*100:.1f}%")
    for nm, k in (("RIGHT wrist err", "rwe"), ("LEFT  wrist err", "lwe")):
        print(f"[DIAG] {nm}  mean {A[k].mean():.4f}  p95 {np.percentile(A[k],95):.4f}  max {A[k].max():.4f} m  (term @0.20)")
    for nm, k in (("RIGHT wrist->obj", "rd"), ("LEFT  wrist->obj", "ld")):
        print(f"[DIAG] {nm}  mean {A[k].mean():.4f}  min {A[k].min():.4f}  max {A[k].max():.4f} m  (ref 0.209/0.210)")
    for nm, k, a in (("RIGHT", "rfc", "ract"), ("LEFT ", "lfc", "lact")):
        print(f"[DIAG] {nm} wrench-support fraction  mean {A[k].mean():.4f}   ref contact-active fraction {A[a].mean():.3f}")
    np.savez("out/diag_policy.npz", **A)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
