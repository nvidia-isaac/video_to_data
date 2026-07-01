"""Per-env 'hammer the nail' goal-pose trajectories for the claw_hammer.

Phases: lift -> reorient (face down) -> over the nail -> lower to contact -> SWING strikes. The
final phase is a SWING: the hammer rotates about its grip (root) so the head arcs DOWN onto the
nail and back up, repeatedly -- a hammer blow, not a vertical translation.

claw_hammer local geometry (native USD frame, identity spawn; from the mesh):
  handle runs along +x (grip ~ at the root/origin); the HEAD is at the +x end (x~0.12), a flat
  plate whose long axis is y -- the CLAW is the +y end (tapered/forked), the flat striking FACE is
  the -y end. So:
    TOOL  = -y  (striking-face normal; reorient points this -> -screw_axis = face down on the nail)
    TIP   = (0.125, -0.038, 0)  striking-face center (the point that contacts the nail head)
    BLADE = +x  (handle direction; secondary axis / roll reference)
Quaternions are XYZW (matches the env's goal interface).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

TOOL = np.array([0.0, -1.0, 0.0])       # striking-face normal (-y); points down at the nail when striking
BLADE = np.array([1.0, 0.0, 0.0])       # handle direction (secondary axis -> roll reference)
TIP = np.array([0.125, -0.038, 0.0])    # body-origin -> striking-face center (head +x end, -y face)
GRIP = np.array([0.0, 0.0, 0.0])        # swing pivot in the tool frame (~ the hand grip, at the root)

# phase lengths: lift -> reorient (to the RAISED pre-hit orientation) -> move to the pre-hit pose
# (over the nail, raised) -> HIT (swing down to the nail and back up, repeatedly). NO separate
# lowering phase (N_LOWER kept = 0 for the goal-noise-schedule interface). N_TURN = the hit phase.
N_LIFT, N_REORIENT, N_OVER, N_LOWER, N_TURN = 10, 18, 12, 0, 28
T = N_LIFT + N_REORIENT + N_OVER + N_LOWER + N_TURN  # 68

N_STRIKES = 3            # hammer blows within the hit phase
SWING_ANGLE = 0.9        # swing amplitude (rad, ~51 deg): the head is raised this far between blows (bigger -> more pronounced strike)


def _u(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def _interp(p0, r0, p1, r1, n):
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    return [[*((1 - k / n) * p0 + (k / n) * p1).tolist(), *slerp([k / n])[0].as_quat().tolist()]
            for k in range(1, n + 1)]


def _facedown(sd_rot, axis):
    """Minimal rotation pointing TOOL (striking-face normal) -> -axis (face down on the nail),
    preserving the grasped roll (the nail is axisymmetric, so no slot/roll snap is needed)."""
    axis = _u(axis)
    t_cur = sd_rot.apply(TOOL)
    t_tgt = -axis
    v = np.cross(t_cur, t_tgt); s = np.linalg.norm(v); c = float(np.clip(np.dot(t_cur, t_tgt), -1, 1))
    if s < 1e-8:
        r_min = R.identity() if c > 0 else R.from_rotvec(np.pi * _u(np.cross(t_cur, [1.0, 0.0, 0.0])))
    else:
        r_min = R.from_rotvec(np.arccos(c) * (v / s))
    return r_min * sd_rot


def _swing_axis(R0):
    """Horizontal axis (perpendicular to the handle) about which the hammer swings; +angle raises
    the head. v = face position relative to the grip (world)."""
    v = R0.apply(TIP - GRIP)
    return _u(np.cross(v, [0.0, 0.0, 1.0]))


def _hit(root_c, R0, a, swing_angle, n_strikes, n):
    """HIT phase: rotate the hammer about its grip (= root_c) so the head/face arcs DOWN onto the
    nail and back up, n_strikes times. Starts RAISED (phi=swing_angle, the pre-hit pose) and the
    first downstroke hits the nail (phi=0 -> R0, face on the nail). Pivot = GRIP at the root, so
    the root stays at root_c and only the orientation swings."""
    out = []
    for k in range(1, n + 1):
        phi = swing_angle * 0.5 * (1.0 + np.cos(2.0 * np.pi * n_strikes * k / n))  # raised -> hit -> raised
        newR = R.from_rotvec(phi * a) * R0
        out.append([*root_c.tolist(), *newR.as_quat().tolist()])
    return out


def _one(sd_pos, sd_quat_xyzw, head, axis, lift_h, approach, clearance, swing_angle, n_strikes):
    sd_rot = R.from_quat(sd_quat_xyzw)
    axis = _u(axis)
    R0 = _facedown(sd_rot, axis)                       # contact orientation: striking face down on the nail
    a = _swing_axis(R0)
    R_pre = R.from_rotvec(swing_angle * a) * R0        # RAISED pre-hit orientation (head up, ready to swing)
    p_lift = sd_pos + np.array([0.0, 0.0, lift_h])
    # contact root = grip pose with the face on the nail. The grip is at the root, so the hit phase
    # swings about THIS point; the pre-hit pose shares this root (just raised orientation).
    root_c = head + clearance * axis - R0.apply(TIP)
    g = []
    g += _interp(sd_pos, sd_rot, p_lift, sd_rot, N_LIFT)        # lift
    g += _interp(p_lift, sd_rot, p_lift, R_pre, N_REORIENT)     # reorient to the RAISED pre-hit orientation
    g += _interp(p_lift, R_pre, root_c, R_pre, N_OVER)          # move to the pre-hit pose (over the nail, raised)
    g += _hit(root_c, R0, a, swing_angle, n_strikes, N_TURN)    # HIT, HIT (swing down to the nail, repeat)
    return np.asarray(g, dtype=np.float32)  # (T,7)


def _geti(v, i):
    """Per-env param accessor: scalar -> itself; array-like (len n) -> i-th element."""
    return v[i] if np.ndim(v) > 0 else v


def sample_diversify_params(n, scale=1.0):
    """Per-env random GENERATION parameters for trajectory diversity (training only). Returns a dict of
    length-n arrays consumable by compute_goals_batch. lift_height / swing_angle vary CONTINUOUSLY (range
    x scale); n_strikes varies over a small discrete set (with probability ~scale, else the base). scale=0
    -> base values (no diversity). Coherent -> each env gets a different but smooth trajectory shape."""
    lift = 0.15 + np.random.uniform(-0.05, 0.06, n) * scale
    swing = SWING_ANGLE + np.random.uniform(-0.25, 0.25, n) * scale
    vary = np.random.random(n) < min(1.0, abs(scale))               # which envs get a non-base strike count
    strikes = np.where(vary, np.random.randint(2, 5, n), N_STRIKES)  # {2,3,4} vs base 3
    return {"lift_height": np.clip(lift, 0.08, 0.25).astype(np.float32),
            "swing_angle": np.clip(swing, 0.5, 1.3).astype(np.float32),
            "n_strikes": strikes.astype(np.int64)}


def compute_goals_batch(sd_pos, sd_quat_xyzw, screw_head, slot_dir, screw_axis,
                        lift_height=0.15, approach_height=0.08, contact_clearance=0.0,
                        swing_angle=SWING_ANGLE, n_strikes=N_STRIKES, **_ignore):
    """All inputs (n,3)/(n,4) numpy (world/env-local). `slot_dir` unused (the nail has no slot). The
    shape params (lift_height/approach_height/contact_clearance/swing_angle/n_strikes) may be SCALARS
    (whole batch) or per-env arrays (len n) -- see sample_diversify_params. Extra kwargs (turn_degrees,
    clockwise) accepted+ignored for interface compatibility. Returns (n, T, 7) xyzw."""
    n = len(sd_pos)
    out = np.zeros((n, T, 7), dtype=np.float32)
    for i in range(n):
        out[i] = _one(np.asarray(sd_pos[i], float), np.asarray(sd_quat_xyzw[i], float),
                      np.asarray(screw_head[i], float), np.asarray(screw_axis[i], float),
                      _geti(lift_height, i), _geti(approach_height, i), _geti(contact_clearance, i),
                      _geti(swing_angle, i), int(_geti(n_strikes, i)))
    return out
