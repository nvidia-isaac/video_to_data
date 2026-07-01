"""Pure-NumPy forward kinematics for the Vega URDF (no Isaac/GPU).

Computes world poses of the LEFT-arm chain / hand / fingertips at a given joint config, relative to
the robot root. Used to plan the SimToolReal robot-swap placement without spinning up Isaac Sim:
  - pick a base offset + LEFT-arm pose that hovers the left hand over the table workspace (z~0.55),
  - derive the palm-center offset in the L_arm_l7 frame (cfg.palm_offset).

Usage:
  python scripts/urdf_fk.py                         # zero pose
  python scripts/urdf_fk.py --arm 0,0.6,0,-1.2,0,0.8,0   # preview an L_arm_j1..7 pose (rad)
  python scripts/urdf_fk.py --target 0,-0.05,0.58   # solve a base offset to put the hand there (zero arm)
"""

import argparse
import xml.etree.ElementTree as ET

import numpy as np

URDF = "/home/cning/simtoolreal_isaaclab/assets/urdf/vega_sharpa/vega_sharpa_reduced.urdf"
ARM = [f"L_arm_j{i}" for i in range(1, 8)]
FINGERTIPS = ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]
REPORT = [*[f"L_arm_l{i}" for i in range(1, 8)], "left_hand_C_MC", *FINGERTIPS]


def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_R(axis, q):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    x, y, z = a
    c, s, C = np.cos(q), np.sin(q), 1 - np.cos(q)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t; return M


def load_joints():
    root = ET.parse(URDF).getroot()
    joints = {}
    children = set()
    for j in root.iter("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0").split())] if o is not None else [0, 0, 0]
        ax = j.find("axis")
        axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1, 0, 0]
        child = j.find("child").get("link")
        joints[child] = dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                             xyz=xyz, rpy=rpy, axis=axis)
        children.add(child)
    all_links = {j.find("parent").get("link") for j in root.iter("joint")} | children
    root_link = (all_links - children).pop()
    return joints, root_link


def fk(joints, root_link, cfg):
    """world 4x4 transform per link (root at identity); cfg: joint_name -> angle."""
    Tw = {root_link: np.eye(4)}
    changed = True
    while changed:
        changed = False
        for child, j in joints.items():
            if child in Tw or j["parent"] not in Tw:
                continue
            Torigin = T(rpy_to_R(*j["rpy"]), j["xyz"])
            q = cfg.get(j["name"], 0.0) if j["type"] in ("revolute", "continuous", "prismatic") else 0.0
            Tmot = T(axis_R(j["axis"], q), [0, 0, 0]) if j["type"] != "prismatic" else T(np.eye(3), np.array(j["axis"]) * q)
            Tw[child] = Tw[j["parent"]] @ Torigin @ Tmot
            changed = True
    return Tw


# L_arm_j1..7 limits (from the URDF) for the IK search
ARM_LIMITS = [(-3.071, 3.071), (-0.453, 1.553), (-3.071, 3.071), (-3.071, 0.244),
              (-3.071, 3.071), (-1.396, 1.396), (-1.378, 1.117)]


def _hand_and_wrist(joints, root_link, q, base):
    cfg = {ARM[i]: q[i] for i in range(7)}
    Tw = fk(joints, root_link, cfg)
    ft = np.mean([Tw[f][:3, 3] for f in FINGERTIPS], axis=0) + base
    wrist = Tw["L_arm_l7"][:3, 3] + base
    return ft, wrist


def ik_reach_down(joints, root_link, base, target, restarts=60, iters=200, seed=0, margin=0.05):
    """Coordinate-descent IK: hand-centroid -> target, palm reaching DOWN (fingertips below wrist).

    Limits are shrunk by `margin` (rad) so no solved joint sits exactly on a bound (IsaacLab's
    articulation cfg validation rejects a default position at/over the limit)."""
    rng = np.random.default_rng(seed)
    limits = [(lo + margin, hi - margin) for lo, hi in ARM_LIMITS]

    def cost(q):
        ft, wrist = _hand_and_wrist(joints, root_link, q, base)
        c = np.sum((ft - target) ** 2)
        c += 0.5 * max(0.0, (ft[2] - wrist[2]) + 0.04) ** 2   # want fingertips BELOW the wrist
        return c

    best_q, best_c = None, 1e18
    for r in range(restarts):
        q = np.array([rng.uniform(lo, hi) for lo, hi in limits])
        step = 0.5
        c = cost(q)
        for _ in range(iters):
            improved = False
            for k in range(7):
                for d in (step, -step):
                    q2 = q.copy(); q2[k] = np.clip(q2[k] + d, *limits[k])
                    c2 = cost(q2)
                    if c2 < c - 1e-9:
                        q, c = q2, c2; improved = True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        if c < best_c:
            best_q, best_c = q, c
    return best_q, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=str, default="")
    ap.add_argument("--base", type=str, default="0,0,0")
    ap.add_argument("--target", type=str, default="")
    ap.add_argument("--ik", type=str, default="", help="world target 'x,y,z' to IK the hand to (uses --base)")
    args = ap.parse_args()

    if args.ik:
        joints, root_link = load_joints()
        base = np.array([float(v) for v in args.base.split(",")])
        tgt = np.array([float(v) for v in args.ik.split(",")])
        q, c = ik_reach_down(joints, root_link, base, tgt)
        ft, wrist = _hand_and_wrist(joints, root_link, q, base)
        print(f"IK base={base.tolist()} target={tgt.tolist()} residual={np.sqrt(c):.4f}")
        print("  arm =", ",".join(f"{v:.4f}" for v in q))
        print(f"  hand-centroid ({ft[0]:+.3f},{ft[1]:+.3f},{ft[2]:+.3f})  wrist z={wrist[2]:+.3f}")
        return

    joints, root_link = load_joints()
    cfg = {}
    if args.arm:
        cfg = {ARM[i]: float(v) for i, v in enumerate(args.arm.split(","))}
    base = np.array([float(v) for v in args.base.split(",")])

    Tw = fk(joints, root_link, cfg)
    print(f"root_link={root_link}  base={base.tolist()}  arm={args.arm or 'zero'}")
    pos = {k: Tw[k][:3, 3] + base for k in REPORT if k in Tw}
    for k in REPORT:
        if k in pos:
            p = pos[k]
            print(f"  {k:16s} ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
    ft = np.mean([pos[f] for f in FINGERTIPS if f in pos], axis=0)
    print(f"  fingertip-centroid ({ft[0]:+.3f},{ft[1]:+.3f},{ft[2]:+.3f})")
    # palm-center offset in the L_arm_l7 LOCAL frame (cfg.palm_offset): R_l7^T @ (centroid - p_l7)
    l7 = Tw["L_arm_l7"]
    off_local = l7[:3, :3].T @ ((ft - base) - l7[:3, 3])
    print(f"  PALM_OFFSET(in L_arm_l7 frame) = ({off_local[0]:+.4f},{off_local[1]:+.4f},{off_local[2]:+.4f})")

    if args.target:
        tgt = np.array([float(v) for v in args.target.split(",")])
        # base offset so the fingertip-centroid lands at target (for the CURRENT arm cfg)
        need = tgt - (ft - base)
        print(f"  -> base to put hand-centroid at {tgt.tolist()}: ({need[0]:+.3f},{need[1]:+.3f},{need[2]:+.3f})")


if __name__ == "__main__":
    main()
