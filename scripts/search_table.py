"""Find the max BACK-shift (+y) of the table that keeps the RIGHT hand able to reach the original
hand->table pose (pos + orientation). Up is fixed at +0.2 from the current table (z 0.68 -> 0.88).
Pure NumPy. Prints right-arm IK error per candidate so we pick the farthest-back reachable spot."""
import sys
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import scripts.urdf_fk as u

BASE = np.array([0.0, -0.531, 0.0])
CUR_TABLE = np.array([0.0, -0.45, 0.68])             # current table center (env-local)
UP = 0.20                                            # +z, fixed (relative to current)
ORIG_FT = np.array([
    [-0.0304, 0.0874, 0.9132], [-0.0101, 0.0848, 0.9148], [0.0102, 0.0902, 0.9119],
    [-0.1168, 0.1701, 0.8793], [0.0310, 0.0957, 0.9090]])
TABLE_ORIG = np.array([0.0, -0.15, 0.38])


def hand_frame(ft):
    c = ft.mean(0); across = ft[4] - ft[0]; thumb = ft[3] - c
    z = np.cross(across, thumb); z /= (np.linalg.norm(z) + 1e-9)
    x = across / (np.linalg.norm(across) + 1e-9); x = x - (x @ z) * z; x /= (np.linalg.norm(x) + 1e-9)
    return c, np.stack([x, np.cross(z, x), z], axis=1)


C_orig, R_orig = hand_frame(ORIG_FT)
O_ftc = C_orig - TABLE_ORIG
joints, root = u.load_joints(); urdf = ET.parse(u.URDF).getroot()
RARM = [f"R_arm_j{i}" for i in range(1, 8)]
RFT = [f"right_{n}_DP" for n in ["index", "middle", "ring", "thumb", "pinky"]]
lim = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
       for j in urdf.iter("joint") if j.get("name") in RARM}
L = [(lim[n][0] + 0.05, lim[n][1] - 0.05) for n in RARM]


def ik(tgt_pos, tgt_R, restarts=30, iters=150, seed=1):
    rng = np.random.default_rng(seed)

    def frame(q):
        Tw = u.fk(joints, root, {RARM[i]: q[i] for i in range(7)})
        return hand_frame(np.array([Tw[f][:3, 3] + BASE for f in RFT]))

    def cost(q):
        c, R = frame(q)
        return np.sum((c - tgt_pos) ** 2) + 0.05 * np.sum((R - tgt_R) ** 2)

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
    c, R = frame(best)
    pe = np.linalg.norm(c - tgt_pos) * 1000
    re = np.degrees(np.arccos(np.clip((np.trace(tgt_R.T @ R) - 1) / 2, -1, 1)))
    return pe, re


print(f"O_ftc={np.round(O_ftc,4).tolist()}  up=+{UP} (table z {CUR_TABLE[2]}->{CUR_TABLE[2]+UP})")
print("back(+y)  table_y   hand_y    pos_err   rot_err")
for back in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    tc = CUR_TABLE + np.array([0.0, back, UP])
    tgt = tc + O_ftc
    pe, re = ik(tgt, R_orig)
    flag = "  OK" if (pe < 15 and re < 6) else ("  marginal" if pe < 35 else "  UNREACH")
    print(f"  {back:.2f}     {tc[1]:+.3f}   {tgt[1]:+.3f}   {pe:6.1f}mm  {re:5.1f}deg{flag}")
