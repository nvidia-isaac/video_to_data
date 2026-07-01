"""IK the Vega arms so the RIGHT hand matches the ORIGINAL hammer hand POSE (position + orientation)
relative to the table, and the LEFT hand is its mirror across the robot center plane (x=0).

Hand pose is defined HAND-INTRINSICALLY from the 5 fingertips (robust across robots / wrists, fingers
at the default open pose): position = fingertip-centroid; orientation = frame built from
across(pinky-index) + thumb direction + palm normal. Full 6-DOF coordinate-descent IK. Pure NumPy.
"""
import sys
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import scripts.urdf_fk as u

BASE = np.array([0.0, -0.531, 0.0])                  # centered fixed-base root
TABLE_C = np.array([0.0, -0.65, 0.68])               # down 0.2 + front 0.4 from current (0,-0.25,0.88)

# ORIGINAL hammer LEFT-hand fingertips at reset (env-local), order [index, middle, ring, thumb, pinky]
ORIG_FT = np.array([
    [-0.0304, 0.0874, 0.9132],
    [-0.0101, 0.0848, 0.9148],
    [ 0.0102, 0.0902, 0.9119],
    [-0.1168, 0.1701, 0.8793],
    [ 0.0310, 0.0957, 0.9090],
])
TABLE_ORIG = np.array([0.0, -0.15, 0.38])            # original table cuboid center


def hand_frame(ft):
    """ft: (5,3) [index,middle,ring,thumb,pinky] -> (centroid (3,), R (3,3) columns=[x,y,z])."""
    c = ft.mean(0)
    across = ft[4] - ft[0]                            # pinky - index (lateral)
    thumb_dir = ft[3] - c                             # thumb out from centroid
    z = np.cross(across, thumb_dir); z = z / (np.linalg.norm(z) + 1e-9)   # palm normal
    x = across / (np.linalg.norm(across) + 1e-9)
    x = x - (x @ z) * z; x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    return c, np.stack([x, y, z], axis=1)


C_orig, R_orig = hand_frame(ORIG_FT)
O_ftc = C_orig - TABLE_ORIG                           # fingertip-centroid offset vs table center
SPREAD = np.array([-0.10, 0.0, 0.0])                  # move right hand 0.1 toward the robot's RIGHT (-x)
T_R = TABLE_C + O_ftc + SPREAD                        # right-hand target (spread out so hands don't collide)
M = np.diag([-1.0, 1.0, 1.0])                          # reflection across x=0 (robot center plane)
T_L = M @ T_R                                          # left hand mirrors -> moves 0.1 toward the LEFT (+x)
R_R = R_orig
R_L = M @ R_orig @ M                                  # mirrored orientation (still a proper rotation)

joints, root = u.load_joints()
urdf = ET.parse(u.URDF).getroot()
LARM = [f"L_arm_j{i}" for i in range(1, 8)]
RARM = [f"R_arm_j{i}" for i in range(1, 8)]
LFT = ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]
RFT = [f.replace("left_", "right_") for f in LFT]


def limits(names):
    lim = {}
    for j in urdf.iter("joint"):
        if j.get("name") in names:
            l = j.find("limit"); lim[j.get("name")] = (float(l.get("lower")), float(l.get("upper")))
    return [lim[n] for n in names]


def ik(arm, fts, tgt_pos, tgt_R, w_rot=0.05, restarts=45, iters=160, margin=0.05, seed=0):
    L = [(lo + margin, hi - margin) for lo, hi in limits(arm)]
    rng = np.random.default_rng(seed)

    def frame(q):
        Tw = u.fk(joints, root, {arm[i]: q[i] for i in range(7)})
        ft = np.array([Tw[f][:3, 3] + BASE for f in fts])
        return hand_frame(ft)

    def cost(q):
        c, R = frame(q)
        return np.sum((c - tgt_pos) ** 2) + w_rot * np.sum((R - tgt_R) ** 2)

    best, bc = None, 1e18
    for r in range(restarts):
        q = np.array([rng.uniform(*L[k]) for k in range(7)]); step = 0.5; cc = cost(q)
        for _ in range(iters):
            improved = False
            for k in range(7):
                for d in (step, -step):
                    q2 = q.copy(); q2[k] = np.clip(q2[k] + d, *L[k]); c2 = cost(q2)
                    if c2 < cc - 1e-9:
                        q, cc, improved = q2, c2, True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        if cc < bc:
            bc, best = cc, q
    c, R = frame(best)
    pos_err = np.linalg.norm(c - tgt_pos)
    rot_err = np.degrees(np.arccos(np.clip((np.trace(tgt_R.T @ R) - 1) / 2, -1, 1)))
    return best, pos_err, rot_err, c


print(f"T_R={np.round(T_R,4).tolist()}  T_L={np.round(T_L,4).tolist()}")
print("R_orig=\n", np.round(R_orig, 3))
qr, pr, rr, cr = ik(RARM, RFT, T_R, R_R, seed=1)
ql, pl, rl, cl = ik(LARM, LFT, T_L, R_L, seed=2)
print(f"RIGHT pos_err={pr*1000:.1f}mm rot_err={rr:.1f}deg hand={np.round(cr,3).tolist()}")
print("  R_arm =", ",".join(f"{v:.4f}" for v in qr))
print(f"LEFT  pos_err={pl*1000:.1f}mm rot_err={rl:.1f}deg hand={np.round(cl,3).tolist()}")
print("  L_arm =", ",".join(f"{v:.4f}" for v in ql))
