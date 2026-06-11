"""Per-env 'tighten the screw' goal-pose trajectories for the (aligned) 043 PHILLIPS screwdriver.

Same 5-phase logic as the flat-screwdriver generator (lift -> reorient tip-down -> over screw ->
lower to contact -> rotate in place), but the tip/slot are a CROSS (+) instead of a single line.
A Phillips tip + cross slot are 4-fold symmetric, so the tip seats at ANY 90 deg roll. The reorient
therefore uses the MINIMAL rotation to bring the tool axis tip-down (preserving the grasped roll),
then snaps the cross arm onto the NEAREST of the slot's 4 arms (mod 90 deg) -- the smallest
reorientation that still aligns tip & slot. (The flat generator instead forced the blade onto the
one slot line, mod 180 deg.)

The aligned-043 mesh matches the aligned-044 convention exactly (tool -> local +x, tip at +x=0.134),
so TOOL/BLADE/TIP are identical; only the alignment rule differs. Quaternions are XYZW.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# aligned-043 local geometry (identical to the aligned-044 convention)
TOOL = np.array([1.0, 0.0, 0.0])   # origin -> tip
BLADE = np.array([0.0, 0.0, 1.0])  # one tip cross-arm (the other is +90deg; 4-fold symmetric)
TIP = np.array([0.134, 0.0, 0.0])  # body-origin -> tip

N_LIFT, N_REORIENT, N_OVER, N_LOWER, N_TURN = 10, 18, 12, 10, 24
T = N_LIFT + N_REORIENT + N_OVER + N_LOWER + N_TURN  # 74 (matches the flat generator)


def _u(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _azimuth(v, axis, e1, e2):
    """Signed angle of v's component perpendicular to `axis`, in the (e1,e2) reference frame."""
    vp = v - np.dot(v, axis) * axis
    return np.arctan2(np.dot(vp, e2), np.dot(vp, e1))


def _interp(p0, r0, p1, r1, n):
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    return [[*((1 - k / n) * p0 + (k / n) * p1).tolist(), *slerp([k / n])[0].as_quat().tolist()]
            for k in range(1, n + 1)]


def _rotate(pos, r_base, axis, total_rad, sign, n):
    return [[*pos.tolist(), *(R.from_rotvec(sign * total_rad * k / n * axis) * r_base).as_quat().tolist()]
            for k in range(1, n + 1)]


def _tipdown_cross(sd_rot, axis, slot):
    """Rotation that points the tool tip-down (TOOL -> -axis) with the MINIMAL rotation from the
    grasped pose, then snaps the cross arm onto the nearest of the slot's 4 arms (mod 90 deg)."""
    axis = _u(axis)
    t_cur = sd_rot.apply(TOOL)                     # current tool dir (world)
    t_tgt = -axis                                  # tip-down target
    v = np.cross(t_cur, t_tgt); s = np.linalg.norm(v); c = float(np.clip(np.dot(t_cur, t_tgt), -1, 1))
    if s < 1e-8:                                    # already (anti)parallel
        r_min = R.identity() if c > 0 else R.from_rotvec(np.pi * _u(np.cross(t_cur, [0.0, 0.0, 1.0])))
    else:
        r_min = R.from_rotvec(np.arccos(c) * (v / s))
    R0 = r_min * sd_rot                             # minimal tip-down, roll preserved
    # reference frame perpendicular to the screw axis, to measure rolls
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _u(ref - np.dot(ref, axis) * axis); e2 = np.cross(axis, e1)
    th_cur = _azimuth(R0.apply(BLADE), axis, e1, e2)   # current cross-arm azimuth
    th_slot = _azimuth(slot, axis, e1, e2)             # one slot-arm azimuth
    d = th_cur - th_slot
    resid = d - (np.pi / 2) * np.round(d / (np.pi / 2))   # in [-45,45] deg: roll to nearest arm
    return R.from_rotvec(-resid * axis) * R0


def _one(sd_pos, sd_quat_xyzw, head, slot, axis, lift_h, approach, clearance, turn_deg, cw):
    sd_rot = R.from_quat(sd_quat_xyzw)
    axis = _u(axis)
    R0 = _tipdown_cross(sd_rot, axis, _u(slot))    # tip-down + cross-arm onto nearest slot arm
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
