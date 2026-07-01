"""Right arm IK to a PALM-DOWN target at the kept (spread) hand position; LEFT arm = EXACT joint
mirror of the right (sign pattern from the URDF L/R axis analysis). Verifies palm direction, the
right-hand position, and how close the joint-mirrored left hand is to the true mirror. Pure NumPy."""
import sys
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import scripts.urdf_fk as u

BASE = np.array([0.0, -0.531, 0.0])
TABLE_C = np.array([0.0, -0.65, 0.68])
ORIG_FT = np.array([[-0.0304, 0.0874, 0.9132], [-0.0101, 0.0848, 0.9148], [0.0102, 0.0902, 0.9119],
                    [-0.1168, 0.1701, 0.8793], [0.0310, 0.0957, 0.9090]])
TABLE_ORIG = np.array([0.0, -0.15, 0.38])
SPREAD = np.array([-0.10, 0.0, 0.0])
# joint-mirror sign pattern q_L = SIGN * q_R (from L/R axis comparison: only j4 keeps sign)
SIGN = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
# PALM-DOWN target frame (cols x=across,y,z=palm-normal): palm faces -z (down), fingers along -y.
R_R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def hand_frame(ft):
    c = ft.mean(0); across = ft[4] - ft[0]; thumb = ft[3] - c
    z = np.cross(across, thumb); z /= (np.linalg.norm(z) + 1e-9)
    x = across / (np.linalg.norm(across) + 1e-9); x = x - (x @ z) * z; x /= (np.linalg.norm(x) + 1e-9)
    return c, np.stack([x, np.cross(z, x), z], axis=1)


C_orig, _ = hand_frame(ORIG_FT)
O_ftc = C_orig - TABLE_ORIG
T_R = TABLE_C + O_ftc + SPREAD
joints, root = u.load_joints(); urdf = ET.parse(u.URDF).getroot()
RARM = [f"R_arm_j{i}" for i in range(1, 8)]; LARM = [f"L_arm_j{i}" for i in range(1, 8)]
RFT = [f"right_{n}_DP" for n in ["index", "middle", "ring", "thumb", "pinky"]]
LFT = [f"left_{n}_DP" for n in ["index", "middle", "ring", "thumb", "pinky"]]
LIM = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
       for j in urdf.iter("joint") if j.get("name") in RARM + LARM}


def frame_of(arm, fts, q):
    Tw = u.fk(joints, root, {arm[i]: q[i] for i in range(7)})
    return hand_frame(np.array([Tw[f][:3, 3] + BASE for f in fts]))


PALM_DOWN = np.array([0.0, 0.0, -1.0])      # want the palm-normal (frame z) to point straight down


def ik(arm, fts, tgt_pos, tgt_R, w_rot=0.30, restarts=60, iters=200, margin=0.05, seed=1):
    L = [(LIM[n][0] + margin, LIM[n][1] - margin) for n in arm]
    rng = np.random.default_rng(seed)

    def cost(q):
        c, R = frame_of(arm, fts, q)
        # constrain ONLY the palm-normal to point down (yaw free -> reachable); position kept
        return np.sum((c - tgt_pos) ** 2) + w_rot * np.sum((R[:, 2] - PALM_DOWN) ** 2)

    best, bc = None, 1e18
    for r in range(restarts):
        q = np.array([rng.uniform(*L[k]) for k in range(7)]); s = 0.5; cc = cost(q)
        for _ in range(iters):
            imp = False
            for k in range(7):
                for d in (s, -s):
                    q2 = q.copy(); q2[k] = np.clip(q2[k] + d, *L[k]); c2 = cost(q2)
                    if c2 < cc - 1e-9:
                        q, cc, imp = q2, c2, True
            if not imp:
                s *= 0.5
                if s < 1e-3:
                    break
        if cc < bc:
            bc, best = cc, q
    return best


qr = ik(RARM, RFT, T_R, R_R)
ql = SIGN * qr
cr, Rr = frame_of(RARM, RFT, qr)
cl, Rl = frame_of(LARM, LFT, ql)
M = np.diag([-1.0, 1.0, 1.0])
pos_err_R = np.linalg.norm(cr - T_R) * 1000
palm_R = Rr[:, 2]                                   # right palm-facing dir (want ~ -z)
mir_pos = np.linalg.norm(cl - M @ cr) * 1000        # left hand vs mirror of right hand
mir_rot = np.degrees(np.arccos(np.clip((np.trace((M @ Rr @ M).T @ Rl) - 1) / 2, -1, 1)))
lim_ok = all(LIM[LARM[k]][0] <= ql[k] <= LIM[LARM[k]][1] for k in range(7))
print(f"RIGHT pos_err={pos_err_R:.1f}mm  palm_dir(z)={np.round(palm_R,3).tolist()} (want ~[0,0,-1])")
print(f"  R_arm =", ",".join(f"{v:.4f}" for v in qr))
print(f"LEFT (joint-mirror) within L-limits={lim_ok}  hand-vs-true-mirror: pos={mir_pos:.1f}mm rot={mir_rot:.1f}deg")
print(f"  L_arm =", ",".join(f"{v:.4f}" for v in ql))
