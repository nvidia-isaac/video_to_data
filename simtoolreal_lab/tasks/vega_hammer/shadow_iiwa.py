"""Shadow IIWA14 arm forward kinematics (vectorized torch) for the pretrained-policy retarget.

The pretrained SimToolReal policy was trained on the IIWA14 + left-Sharpa robot in JOINT space. To
deploy it on the Vega arm we let it drive a virtual ("shadow") IIWA arm in its trained joint space,
then take the shadow arm's PALM end-effector pose and IK it onto the Vega arm. This module is the
shadow arm's FK: 7 arm joints -> palm pose, in the IIWA link_0 (robot-root) frame.

All IIWA arm joints rotate about their own local +z, so each link transform is T_origin(xyz,rpy) @
Rz(q). Validated against the measured startArmHigher palm pose (see __main__).
"""
from __future__ import annotations

import math

import torch

# iiwa14_joint_1..7 (xyz, rpy) in the parent-link frame; axis is +z for every joint.
_IIWA_ORIGINS = [
    ((0.0, 0.0, 0.1575), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.2025), (math.pi / 2, 0.0, math.pi)),
    ((0.0, 0.2045, 0.0), (math.pi / 2, 0.0, math.pi)),
    ((0.0, 0.0, 0.2155), (math.pi / 2, 0.0, 0.0)),
    ((0.0, 0.1845, 0.0), (-math.pi / 2, math.pi, 0.0)),
    ((0.0, 0.0, 0.2155), (math.pi / 2, 0.0, 0.0)),
    ((0.0, 0.081, 0.0), (-math.pi / 2, math.pi, 0.0)),
]
# palm-center offset in the iiwa14_link_7 frame (observation_action_utils_sharpa.PALM_OFFSET)
_PALM_OFFSET = (-0.0, -0.02, 0.16)
# IIWA arm joint limits (Q_LOWER/UPPER arm block) for clamping the shadow integration
IIWA_ARM_LOWER = torch.tensor([-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543])
IIWA_ARM_UPPER = torch.tensor([2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543])


def _rpy_to_R(r, p, y, device):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    Rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], device=device)
    Ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], device=device)
    Rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], device=device)
    return Rz @ Ry @ Rx


def _origin_T(xyz, rpy, device):
    T = torch.eye(4, device=device)
    T[:3, :3] = _rpy_to_R(*rpy, device=device)
    T[:3, 3] = torch.tensor(xyz, device=device)
    return T


def _matrix_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    # R: (N,3,3) -> (N,4) wxyz
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros((m.shape[0], 4), device=m.device, dtype=m.dtype)
    s = torch.sqrt(torch.clamp(t + 1.0, min=1e-8)) * 2
    q[:, 0] = 0.25 * s
    q[:, 1] = (m[:, 2, 1] - m[:, 1, 2]) / s
    q[:, 2] = (m[:, 0, 2] - m[:, 2, 0]) / s
    q[:, 3] = (m[:, 1, 0] - m[:, 0, 1]) / s
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def iiwa_palm_fk(q_arm: torch.Tensor):
    """q_arm: (N,7) IIWA arm joint angles (rad). Returns (palm_pos (N,3), palm_quat (N,4 wxyz)) in
    the IIWA link_0 (robot-root) frame."""
    N, dev = q_arm.shape[0], q_arm.device
    T = torch.eye(4, device=dev).expand(N, 4, 4).contiguous()
    for i in range(7):
        To = _origin_T(_IIWA_ORIGINS[i][0], _IIWA_ORIGINS[i][1], dev).expand(N, 4, 4)
        q = q_arm[:, i]
        cz, sz = torch.cos(q), torch.sin(q)
        Rz = torch.zeros((N, 4, 4), device=dev); Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz
        Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz; Rz[:, 2, 2] = 1.0; Rz[:, 3, 3] = 1.0
        T = torch.bmm(torch.bmm(T, To), Rz)
    R7 = T[:, :3, :3]
    p7 = T[:, :3, 3]
    off = torch.tensor(_PALM_OFFSET, device=dev).expand(N, 3)
    palm_pos = p7 + torch.bmm(R7, off.unsqueeze(-1)).squeeze(-1)
    return palm_pos, _matrix_to_quat_wxyz(R7)


# --- mirror across the robot sagittal plane x=0 (present the Vega RIGHT hand as a LEFT hand) ---------
# hand joint mirror signs (left<->right Sharpa), canonical 22-DOF order. DERIVED from URDF FK
# (scripts/derive_hand_mirror_sign.py): the right hand's link frames are mirror-constructed (rest-pose
# palm reflection = diag(1,-1,1)), so the SAME joint value produces the mirror grasp -> uniform +1
# (identity passthrough). q_left_equiv = +q_right. (An earlier -1 guess inverted the grasp -> hand opened.)
SIGN_HAND = torch.tensor([1.0] * 22)
# arm Q limits (for unscaling the shadow arm joint_pos obs, matching the original)
_QL = IIWA_ARM_LOWER
_QU = IIWA_ARM_UPPER


def mirror_vec_x(v: torch.Tensor) -> torch.Tensor:
    """Reflect a position/relative vector across the x=0 plane: negate x. v: (...,3)."""
    out = v.clone()
    out[..., 0] = -out[..., 0]
    return out


def mirror_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Reflect a rotation across x=0: R' = M R M, M=diag(-1,1,1). For a quat (w,x,y,z) this is
    (w, x, -y, -z) up to sign. q: (N,4) wxyz -> (N,4) wxyz."""
    out = q.clone()
    out[..., 2] = -out[..., 2]
    out[..., 3] = -out[..., 3]
    return out


def shadow_arm_step(prev_targets, action_arm, dof_speed_scale, control_dt, arm_ma):
    """Integrate the shadow IIWA arm targets exactly like SimToolRealEnv._compute_targets (arm branch):
    arm = prev + speed*dt*a (clamped to IIWA limits); cur = ma*arm + (1-ma)*prev."""
    a = action_arm.clamp(-1.0, 1.0)
    lo = _QL.to(prev_targets.device); hi = _QU.to(prev_targets.device)
    arm = (prev_targets + dof_speed_scale * control_dt * a).clamp(lo, hi)
    return arm_ma * arm + (1.0 - arm_ma) * prev_targets


if __name__ == "__main__":
    # validate against the measured startArmHigher palm (env-local (-0.0053,0.2091,0.8666);
    # base (0,0.8,0) -> link_0 frame (-0.0053,-0.5909,0.8666)).
    sah = torch.tensor([[-1.571, 1.571 - math.radians(10), 0.0, 1.376 + math.radians(10), 0.0, 1.485, 1.308]])
    pos, quat = iiwa_palm_fk(sah)
    print("shadow palm (link_0 frame):", [round(v, 4) for v in pos[0].tolist()])
    print("expected (measured)       :", [-0.0053, -0.5909, 0.8666])
    print("error (m):", round((pos[0] - torch.tensor([-0.0053, -0.5909, 0.8666])).norm().item(), 4))
    print("palm quat wxyz:", [round(v, 4) for v in quat[0].tolist()])
