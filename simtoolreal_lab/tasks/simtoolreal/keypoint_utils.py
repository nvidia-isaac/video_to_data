"""Object-centric keypoint extraction (tool + dynamic screw), shared by data collection and eval.

Both `scripts/collect_bc_data.py` (recording `obs/keypoints`) and the keypoint-policy eval client
MUST compute the keypoints identically, so the definition lives here.

8 env-local keypoints per env:
  - 4 TOOL keypoints  = the env's `object_keypoints` (the manipulated tool's bbox corners).
  - 4 SCREW keypoints = corners at the ACTUAL (dynamic) screw pose, read from the physical
    `screw_asm` body (rotates for the screwdriver, sinks for the hammer); falls back to the
    env-driven kinematic `screw`, the nominal pose, or the goal keypoints.
Import these AFTER the Isaac Lab app is launched.
"""

import torch
from isaaclab.utils.math import quat_apply

KP_CORNERS = [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]  # unit cube corners
SCREW_HALF = (0.008, 0.008, 0.015)                              # screw keypoint half-extents (m)


def screw_offsets(base):
    """Per-corner offsets (4,3) for the screw keypoints (unit corners * half-extents), on device."""
    c = torch.tensor(KP_CORNERS, device=base.device, dtype=torch.float)
    return c * torch.tensor(SCREW_HALF, device=base.device, dtype=torch.float)


def find_screw_body(base):
    """Body index of the driven screw/nail in screw_asm (the body that rotates/sinks), or None."""
    asm = getattr(base, "screw_asm", None)
    if asm is None:
        return None
    names = list(asm.body_names)
    cand = [i for i, n in enumerate(names) if "screw" in n.lower() or "nail" in n.lower()]
    return cand[-1] if cand else (len(names) - 1)


def compute_keypoints(base, screw_off, screw_body_idx):
    """Env-local object-centric keypoints -> (N, 8, 3): 4 TOOL + 4 SCREW (actual dynamic pose)."""
    tool_kp = base.object_keypoints                                  # (N,4,3) env-local
    eo = base.scene.env_origins
    asm = getattr(base, "screw_asm", None)
    if asm is not None and screw_body_idx is not None:               # physical: live rotating/sinking body
        spos = asm.data.body_pos_w[:, screw_body_idx] - eo
        squat = asm.data.body_quat_w[:, screw_body_idx]
    elif getattr(base, "screw", None) is not None:                   # kinematic: env-written driven pose
        spos = base.screw.data.root_pos_w - eo
        squat = base.screw.data.root_quat_w
    elif getattr(base, "screw_nom_pos", None) is not None:           # fallback: nominal (static)
        spos = base.screw_nom_pos - eo
        squat = base.screw_nom_quat
    else:
        return torch.cat([tool_kp, base.goal_keypoints], dim=1)      # no screw (base claw_hammer)
    screw_kp = spos.unsqueeze(1) + quat_apply(
        squat.unsqueeze(1).expand(-1, 4, -1), screw_off.unsqueeze(0).expand(base.num_envs, -1, -1))
    return torch.cat([tool_kp, screw_kp], dim=1)                     # (N,8,3)


# ---- SimToolReal specialist obs: mirrors the RL expert's 140-dim actor input (minus the goal) ----
# proprio = joint_pos(29) + joint_vel(29) + prev_targets(29) + palm_pos(3) + palm_rot(4)
#           + fingertip_pos_rel_palm(5*3=15)  -> 109 dims (Sharpa hand has 5 fingertips).
SIMTOOLREAL_PROPRIO_DIM = 109


def compute_simtoolreal_obs(base, screw_off, screw_body_idx):
    """State obs for the SimToolReal specialist, faithful to the expert's actor observation.

    Returns:
      kp_rel_palm : (N, 8, 3)  the 8 keypoints (4 tool + 4 dynamic screw) made PALM-RELATIVE by
                    subtracting palm_center -- exactly the expert's keypoints_rel_palm convention.
      proprio     : (N, 109)   concat[joint_pos(29), joint_vel(29), prev_targets(29),
                    palm_center(3), palm_quat(4), fingertip_pos_rel_palm(15)] -- the privileged
                    proprio the keypoint policy lacked (velocity, action history, hand pose, grasp).

    Must be computed at the SAME instant the action is recorded (pre-step), so collection and eval
    share this one function.
    """
    kp = compute_keypoints(base, screw_off, screw_body_idx)          # (N,8,3) absolute env-local
    palm = base.palm_center                                          # (N,3) palm reference point
    kp_rel = kp - palm.unsqueeze(1)                                  # (N,8,3) palm-relative
    ft_rel = (base.fingertip_pos - palm.unsqueeze(1)).reshape(base.num_envs, -1)  # (N,15)
    proprio = torch.cat([
        base.joint_pos,                                             # 29  current joint positions
        base.joint_vel,                                             # 29  joint velocities
        base.prev_targets,                                          # 29  previous commanded targets
        palm,                                                       # 3   palm position (palm_center)
        base.palm_quat,                                             # 4   palm orientation (wxyz)
        ft_rel,                                                     # 15  5 fingertip positions rel palm
    ], dim=-1)                                                      # (N,109)
    return kp_rel, proprio


def compute_goal_rel(base):
    """4 TOOL keypoints relative to the GOAL pose: (object_keypoints - goal_keypoints) -> (N, 12).

    This is exactly the SAPG teacher actor's `keypoints_rel_goal` term -- the only genuinely-missing
    input for the current best state specialist. The hammer env provides `goal_keypoints` (the current
    sub-goal of the tighten trajectory) every step, so it is available at BOTH collection and eval."""
    return (base.object_keypoints - base.goal_keypoints).reshape(base.num_envs, -1)
