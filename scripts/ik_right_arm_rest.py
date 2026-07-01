"""One-off: IK a tucked REST pose for the Vega RIGHT arm (held, not task-controlled) so its hand
hangs down by the robot's side instead of floating out at shoulder height. Pure NumPy (no Isaac)."""
import sys
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab")
import scripts.urdf_fk as u

joints, root = u.load_joints()
BASE = np.array([-0.169, -0.531, 0.0])
RARM = [f"R_arm_j{i}" for i in range(1, 8)]
RFT = [f.replace("left_", "right_") for f in u.FINGERTIPS]
TARGET = np.array([-0.45, -0.30, 0.55])      # down by the robot's right side, clear of the table

t = ET.parse(u.URDF).getroot()
lim = {}
for j in t.iter("joint"):
    if j.get("name") in RARM:
        l = j.find("limit")
        lim[j.get("name")] = (float(l.get("lower")), float(l.get("upper")))
M = 0.05
RLm = [(lim[n][0] + M, lim[n][1] - M) for n in RARM]


def ftc(q):
    cfg = {RARM[i]: q[i] for i in range(7)}
    Tw = u.fk(joints, root, cfg)
    ft = np.mean([Tw[f][:3, 3] for f in RFT], axis=0) + BASE
    return ft, Tw["R_arm_l7"][:3, 3] + BASE


def cost(q):
    ft, wr = ftc(q)
    return np.sum((ft - TARGET) ** 2) + 0.3 * max(0.0, (ft[2] - wr[2]) + 0.04) ** 2


rng = np.random.default_rng(1)
best, bc = None, 1e18
for r in range(50):
    q = np.array([rng.uniform(*RLm[k]) for k in range(7)])
    step, c = 0.5, cost(q)
    for _ in range(150):
        improved = False
        for k in range(7):
            for d in (step, -step):
                q2 = q.copy(); q2[k] = np.clip(q2[k] + d, *RLm[k])
                c2 = cost(q2)
                if c2 < c - 1e-9:
                    q, c, improved = q2, c2, True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    if c < bc:
        bc, best = c, q

ft, wr = ftc(best)
print("residual", round(float(np.sqrt(bc)), 4), "right fingertip ->", np.round(ft, 3).tolist(), flush=True)
print("R_arm_rest =", ",".join(f"{v:.4f}" for v in best), flush=True)
