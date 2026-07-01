"""WRONG-way 'tighten' trajectory for the 043 cross slot: the screwdriver contacts the OUTER RIM of
the screw head (off the central cross slot) and spins about the screw axis, so the tip RIDES THE RIM
instead of seating in the slot.

This is the negative control for the screw-drive physics-validity test. The correct trajectory
(tighten_traj043) seats the cross tip in the slot and rotates -> form closure drives the screw. This
one keeps the tip on the flat rim and rotates -> the only thing that could turn the screw is rim
friction. A correctly resisted screw should NOT spin here; if it does, the revolute-joint friction/
damping is too low (the "screw rotates from outer contact" failure the policy was exploiting).

Same 5 phases / phase counts / tip-down + cross-arm alignment as tighten_traj043 (so the two replays
are directly comparable) -- only the contact point (rim, not slot) and the final motion (orbit the
rim, not spin in the slot) differ. Quaternions are XYZW.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# aligned-043 local geometry (identical to tighten_traj043)
TOOL = np.array([1.0, 0.0, 0.0])
BLADE = np.array([0.0, 0.0, 1.0])
TIP = np.array([0.1608, 0.0, 0.0])

N_LIFT, N_REORIENT, N_OVER, N_LOWER, N_TURN = 10, 18, 12, 10, 24
T = N_LIFT + N_REORIENT + N_OVER + N_LOWER + N_TURN  # 74 (matches tighten_traj043)

# rim-contact geometry (for the +80% screw: head radius ~23mm, central cross slot ~r9mm).
RIM_R = 0.012        # radial offset from the screw axis to the rim contact (m): clearly on the flat
                     # rim, outside the central cross slot
RIM_AXIAL = 0.0028   # tip height above `head` (the slot level): the head TOP is ~2.8mm above the slot
                     # @1.8x, so this rests the tip on the rim surface (light contact, not in the slot)


def _u(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _azimuth(v, axis, e1, e2):
    vp = v - np.dot(v, axis) * axis
    return np.arctan2(np.dot(vp, e2), np.dot(vp, e1))


def _interp(p0, r0, p1, r1, n):
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    return [[*((1 - k / n) * p0 + (k / n) * p1).tolist(), *slerp([k / n])[0].as_quat().tolist()]
            for k in range(1, n + 1)]


def _tipdown_cross(sd_rot, axis, slot):
    """Identical to tighten_traj043: minimal-rotation tip-down, then snap the cross arm onto the
    nearest of the slot's 4 arms (mod 90 deg)."""
    axis = _u(axis)
    t_cur = sd_rot.apply(TOOL)
    t_tgt = -axis
    v = np.cross(t_cur, t_tgt); s = np.linalg.norm(v); c = float(np.clip(np.dot(t_cur, t_tgt), -1, 1))
    if s < 1e-8:
        r_min = R.identity() if c > 0 else R.from_rotvec(np.pi * _u(np.cross(t_cur, [0.0, 0.0, 1.0])))
    else:
        r_min = R.from_rotvec(np.arccos(c) * (v / s))
    R0 = r_min * sd_rot
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _u(ref - np.dot(ref, axis) * axis); e2 = np.cross(axis, e1)
    th_cur = _azimuth(R0.apply(BLADE), axis, e1, e2)
    th_slot = _azimuth(slot, axis, e1, e2)
    d = th_cur - th_slot
    resid = d - (np.pi / 2) * np.round(d / (np.pi / 2))
    return R.from_rotvec(-resid * axis) * R0


def _orbit(center, r_base, axis, radial0, total_rad, sign, n):
    """Spin the screwdriver rigidly about the screw axis through `center`, tip riding the rim at
    radius RIM_R: the whole driver rotates (Rz * r_base) so its off-axis tip orbits the rim."""
    out = []
    for k in range(1, n + 1):
        Rz = R.from_rotvec(sign * total_rad * k / n * axis)
        tip_w = center + RIM_R * Rz.apply(radial0)
        body_rot = Rz * r_base
        body_pos = tip_w - body_rot.apply(TIP)
        out.append([*body_pos.tolist(), *body_rot.as_quat().tolist()])
    return out


def _one(sd_pos, sd_quat_xyzw, head, slot, axis, lift_h, approach, clearance, turn_deg, cw):
    sd_rot = R.from_quat(sd_quat_xyzw)
    axis = _u(axis)
    R0 = _tipdown_cross(sd_rot, axis, _u(slot))    # same tip-down + cross-arm alignment as correct
    radial = _u(slot)                              # rim-contact direction (a horizontal slot-arm dir)
    center = head + RIM_AXIAL * axis               # screw-axis point at the rim-contact height
    p_lift = sd_pos + np.array([0.0, 0.0, lift_h])
    p_over = center + approach * axis + RIM_R * radial - R0.apply(TIP)
    p_contact = center + RIM_R * radial - R0.apply(TIP)   # tip on the rim, off the slot
    sign = -1.0 if cw else 1.0
    g = []
    g += _interp(sd_pos, sd_rot, p_lift, sd_rot, N_LIFT)
    g += _interp(p_lift, sd_rot, p_lift, R0, N_REORIENT)
    g += _interp(p_lift, R0, p_over, R0, N_OVER)
    g += _interp(p_over, R0, p_contact, R0, N_LOWER)
    g += _orbit(center, R0, axis, radial, np.deg2rad(turn_deg), sign, N_TURN)
    return np.asarray(g, dtype=np.float32)  # (T,7)


def compute_goals_batch(sd_pos, sd_quat_xyzw, screw_head, slot_dir, screw_axis,
                        lift_height=0.15, approach_height=0.08, contact_clearance=0.004,
                        turn_degrees=180.0, clockwise=True):
    """Same signature as tighten_traj043.compute_goals_batch. Returns (n, T, 7) xyzw."""
    n = len(sd_pos)
    out = np.zeros((n, T, 7), dtype=np.float32)
    for i in range(n):
        out[i] = _one(np.asarray(sd_pos[i], float), np.asarray(sd_quat_xyzw[i], float),
                      np.asarray(screw_head[i], float), np.asarray(slot_dir[i], float),
                      np.asarray(screw_axis[i], float), lift_height, approach_height,
                      contact_clearance, turn_degrees, clockwise)
    return out
