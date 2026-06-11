"""Per-env 'tighten the screw' goal-pose trajectories for the (aligned) 044 flat screwdriver.

Vectorized over envs: given each env's screwdriver start pose, screw head, slot direction, and
screw axis, builds the 5-phase trajectory (lift -> reorient tip-down + blade-to-slot -> over
screw -> lower to contact -> rotate CW in place) and returns (n, T, 7) goal poses (xyz + xyzw).
Mirrors dextoolbench/generate_tighten_trajectory.py; used by ScrewdriverEnv at each reset so the
goals match each env's randomized screw. Quaternions here are XYZW (SimToolReal trajectory format).

Geometry constants are for the ALIGNED 044 mesh (tool axis -> local +x, blade wide -> local +z).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# aligned-044 local geometry
TOOL = np.array([1.0, 0.0, 0.0])   # origin -> tip
BLADE = np.array([0.0, 0.0, 1.0])  # flat blade WIDE axis (fits the slot)
TIP = np.array([0.134, 0.0, 0.0])  # body-origin -> tip

N_LIFT, N_REORIENT, N_OVER, N_LOWER, N_TURN = 10, 18, 12, 10, 24
T = N_LIFT + N_REORIENT + N_OVER + N_LOWER + N_TURN  # 74


def _u(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def align_frames(la, wa, lb, wb):
    la, wa = _u(la), _u(wa)
    lb = _u(np.asarray(lb, float) - np.dot(lb, la) * la)
    wb = _u(np.asarray(wb, float) - np.dot(wb, wa) * wa)
    Lc = np.column_stack([la, lb, np.cross(la, lb)])
    Wc = np.column_stack([wa, wb, np.cross(wa, wb)])
    return R.from_matrix(Wc @ Lc.T)


def _interp(p0, r0, p1, r1, n):
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    return [[*((1 - k / n) * p0 + (k / n) * p1).tolist(), *slerp([k / n])[0].as_quat().tolist()]
            for k in range(1, n + 1)]


def _rotate(pos, r_base, axis, total_rad, sign, n):
    return [[*pos.tolist(), *(R.from_rotvec(sign * total_rad * k / n * axis) * r_base).as_quat().tolist()]
            for k in range(1, n + 1)]


def _one(sd_pos, sd_quat_xyzw, head, slot, axis, lift_h, approach, clearance, turn_deg, cw):
    sd_rot = R.from_quat(sd_quat_xyzw)
    axis = _u(axis)
    R0 = align_frames(TOOL, -axis, BLADE, slot)        # tip-down + blade in slot
    p_lift = sd_pos + np.array([0.0, 0.0, lift_h])
    p_over = head + approach * axis - R0.apply(TIP)
    p_contact = head + clearance * axis - R0.apply(TIP)
    sign = -1.0 if cw else 1.0
    g = []
    g += _interp(sd_pos, sd_rot, p_lift, sd_rot, N_LIFT)
    g += _interp(p_lift, sd_rot, p_lift, R0, N_REORIENT)
    g += _interp(p_lift, R0, p_over, R0, N_OVER)
    g += _interp(p_over, R0, p_contact, R0, N_LOWER)
    g += _rotate(p_contact, R0, axis, np.deg2rad(turn_deg), sign, N_TURN)
    return np.asarray(g, dtype=np.float32)  # (T,7)


def compute_goals_batch(sd_pos, sd_quat_xyzw, screw_head, slot_dir, screw_axis,
                        lift_height=0.15, approach_height=0.08, contact_clearance=0.004,
                        turn_degrees=180.0, clockwise=True):
    """All inputs are (n,3)/(n,4) numpy arrays (world/env-local frame). Returns (n, T, 7) xyzw."""
    n = len(sd_pos)
    out = np.zeros((n, T, 7), dtype=np.float32)
    for i in range(n):
        out[i] = _one(np.asarray(sd_pos[i], float), np.asarray(sd_quat_xyzw[i], float),
                      np.asarray(screw_head[i], float), np.asarray(slot_dir[i], float),
                      np.asarray(screw_axis[i], float), lift_height, approach_height,
                      contact_clearance, turn_degrees, clockwise)
    return out
