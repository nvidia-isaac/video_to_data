"""Instrumented zero-action rollout: which termination fires at VOC=1.0, and why."""
import argparse, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--task", type=str, default="Sharpa-V2D-v0")
p.add_argument("--motion_file", type=str, required=True)
p.add_argument("--voc", type=float, default=1.0)
p.add_argument("--steps", type=int, default=700)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import torch, gymnasium as gym
import isaaclab_tasks  # noqa
from robotic_grounding.tasks import *  # noqa
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab.utils.math as math_utils

cfg = parse_env_cfg(a.task, device=a.device, num_envs=a.num_envs, use_fabric=True)
cfg.motion_file = a.motion_file
apply_scene_config(cfg, SceneConfig.from_motion_file(cfg.motion_file))
cfg.viewer.env_index = 0
cfg.commands.dual_hands_object_tracking_command.initial_virtual_object_control_curriculum_scale = a.voc
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
cmd = env.command_manager.get_term("dual_hands_object_tracking_command")
print(f"[DIAG] retargeted_horizon = {cmd.retargeted_horizon}", flush=True)
print(f"[DIAG] object mass = {env.scene['tissue_box_refined'].root_physx_view.get_masses()[0].tolist()}", flush=True)

act = torch.zeros(env.action_space.shape, device=env.device)
tm = env.termination_manager
n_pos = n_ori = n_both = 0
counts = {k: 0 for k in tm.active_terms}
pos_hi = []; ori_hi = []
for step in range(a.steps):
    with torch.inference_mode():
        env.step(act)
    dpos = torch.norm(cmd.object_body_position_command_e - cmd.object_position_e, dim=-1).max(dim=-1).values
    dori = math_utils.quat_error_magnitude(cmd.object_orientation_e, cmd.object_body_wxyz_command_e)
    if dori.dim() > 1: dori = dori.max(dim=-1).values
    pos_hi.append(dpos.max().item()); ori_hi.append(dori.max().item())
    for k in tm.active_terms:
        counts[k] += int(tm.get_term(k).sum().item())
    oa = tm.get_term("object_away_from_trajectory")
    if oa.any():
        m = oa
        pv = (dpos[m] > 0.2); ov = (dori[m] > 0.7)
        n_pos += int((pv & ~ov).sum()); n_ori += int((ov & ~pv).sum()); n_both += int((pv & ov).sum())
    if (step + 1) % 100 == 0:
        print(f"[DIAG] step {step+1}  counts={counts}  objAway(pos/ori/both)={n_pos}/{n_ori}/{n_both}"
              f"  max dpos={max(pos_hi):.3f}m  max dori={max(ori_hi):.3f}rad", flush=True)
print("[DIAG] FINAL", counts, "objAway pos-only/ori-only/both =", n_pos, n_ori, n_both, flush=True)
import numpy as np
print(f"[DIAG] dpos p50/p95/p99/max = {np.percentile(pos_hi,50):.4f}/{np.percentile(pos_hi,95):.4f}/{np.percentile(pos_hi,99):.4f}/{max(pos_hi):.4f} m", flush=True)
print(f"[DIAG] dori p50/p95/p99/max = {np.percentile(ori_hi,50):.4f}/{np.percentile(ori_hi,95):.4f}/{np.percentile(ori_hi,99):.4f}/{max(ori_hi):.4f} rad", flush=True)
env.close()
app.close()
