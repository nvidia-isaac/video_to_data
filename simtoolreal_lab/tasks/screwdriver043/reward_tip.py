"""Modular reward augmentation for the cross-slot tighten task (loaded via cfg.reward_module).

WHY: the base success metric is the max distance between the object's bounding-box KEYPOINTS and the
goal keypoints. The screwdriver tip sits ~7.5 cm from the box center, so even the tightest keypoint
tolerance (kp_tol 0.015 m) allows the TIP to be 1.5-3 cm off -> the policy "hovers" without seating in
a mm-scale slot. This module adds a TIP-engagement term so precision is enforced exactly where it
matters (insertion + rotate), without over-constraining the easy lift/reorient phases.

The env (SimToolRealEnv._get_rewards) calls `augment_reward(env)` each step and:
  - ADDS the returned `reward_add` to the (pre-shaper) reward,
  - AND-s the returned `success_gate` into the keypoint near-goal test (so a goal only counts as
    reached when the tip is ALSO seated, on the contact/rotate phases).

It reads env state only (object pose, tip offset, screw head, current goal phase, curriculum progress)
-- no stored state -- so it is safe to hot-swap. Tip tolerance follows the env's curriculum progress
(0 loose -> 1 tight), annealing TIP_TOL_START -> TIP_TOL_TARGET in lockstep with the keypoint tolerance.
"""
import torch
from isaaclab.utils.math import quat_apply

TIP_REW_SCALE = 50.0       # peak per-step tip-proximity bonus (cf. base keypoint_rew_scale 200, delta-based)
TIP_REW_SIGMA = 0.012      # m  -- proximity-bonus width (bonus = exp(-(d/sigma)^2)*scale)
TIP_TOL_START = 0.010      # m  -- loose tip success tolerance (curriculum start; reachable zero-shot)
TIP_TOL_TARGET = 0.002     # m  -- tight tip success tolerance (curriculum end; precise seating)


def _contact_tail(env):
    """# of trailing goals that are the insertion(lower)+rotate phases -> the precision phases."""
    gg = env._goal_gen
    return int(getattr(gg, "N_LOWER", 10)) + int(getattr(gg, "N_TURN", 24))


def augment_reward(env):
    """Returns (reward_add (N,) float, success_gate (N,) bool). Gate is True off the contact phases."""
    N = env.num_envs
    tip = env.object_pos + quat_apply(env.object_quat, env._tip_local.unsqueeze(0).expand(N, 3))
    dist = torch.norm(tip - env.screw_head_world, dim=-1)                 # (N,) tip-to-slot
    phase = env.successes.long() % env._traj_T                           # current goal index
    contact = phase >= (env._traj_T - _contact_tail(env))                # insertion + rotate phases
    # tip tolerance: the env anneals it on its OWN success-gated curriculum (TIP_TOL_START->TARGET),
    # independent of the keypoint tolerance. Fallback to the tight target if the env doesn't set it.
    tip_tol = float(getattr(env, "tip_tol", TIP_TOL_TARGET))
    # dense proximity bonus, only credited on the contact/rotate phases (where seating matters)
    reward_add = torch.exp(-(dist / TIP_REW_SIGMA) ** 2) * TIP_REW_SCALE * contact.float()
    # success gate: tip within tip_tol on the contact phases; no tip requirement elsewhere
    success_gate = (dist <= tip_tol) | (~contact)
    log = env.extras.setdefault("log", {})
    log["tip_dist_contact"] = dist[contact].mean() if bool(contact.any()) else torch.zeros((), device=env.device)
    log["tip_tol"] = tip_tol
    return reward_add, success_gate
