"""Derive the per-joint LEFT<->RIGHT Sharpa-hand mirror sign for the pretrained-policy retarget.

The Vega right hand is the geometric mirror of the left hand, but corresponding joints share the
same local +z axis (the mirror lives in the link-frame rpy). So q_right = s_i * q_left for some
per-joint sign s_i in {+1,-1}. We find s_i purely from URDF FK (no Isaac):

  1. fit the rest-pose reflection R_mir (one of diag(+/-1,...), det=-1) mapping LEFT fingertips in the
     L_arm_l7 palm frame -> RIGHT fingertips in the R_arm_l7 palm frame (both at q=0),
  2. for each hand joint, actuate only that joint by +delta on the left and by +/-delta on the right;
     the correct s_i makes the right fingertip displacement (in the right palm frame) match
     R_mir @ (left fingertip displacement in the left palm frame).

Prints the 22-vector in canonical order for SIGN_HAND.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/cning/simtoolreal_isaaclab/scripts")
from urdf_fk import fk, load_joints  # noqa: E402

CANON = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
]
FT = ["index_DP", "middle_DP", "ring_DP", "thumb_DP", "pinky_DP"]
DELTA = 0.3


def ft_in_palm(joints, root, cfg, side):
    """fingertip positions (5,3) expressed in the side's palm (l7) frame."""
    Tw = fk(joints, root, cfg)
    palm = Tw[f"{'L' if side=='left' else 'R'}_arm_l7"]
    Tinv = np.linalg.inv(palm)
    out = []
    for f in FT:
        p = Tw[f"{side}_{f}"][:3, 3]
        out.append((Tinv @ np.array([*p, 1.0]))[:3])
    return np.array(out)  # (5,3)


def main():
    joints, root = load_joints()
    base_left = ft_in_palm(joints, root, {}, "left")    # (5,3) rest
    base_right = ft_in_palm(joints, root, {}, "right")

    # 1. fit rest-pose reflection (axis-aligned, det=-1) palm-left -> palm-right
    best_M, best_e = None, 1e9
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                if sx * sy * sz != -1:
                    continue  # reflection has det -1
                M = np.diag([sx, sy, sz]).astype(float)
                e = np.linalg.norm(base_right - base_left @ M.T)
                if e < best_e:
                    best_e, best_M = e, M
    M = best_M
    print(f"rest-pose reflection (palm frame) = diag({np.diag(M).astype(int).tolist()})  fit_resid={best_e:.4f} m")

    # 2. per-joint sign
    signs = []
    for jn in CANON:
        lj, rj = f"left_{jn}", f"right_{jn}"
        dL = ft_in_palm(joints, root, {lj: DELTA}, "left") - base_left      # (5,3)
        dRp = ft_in_palm(joints, root, {rj: DELTA}, "right") - base_right
        dRm = ft_in_palm(joints, root, {rj: -DELTA}, "right") - base_right
        target = dL @ M.T                                                   # mirror of the left motion
        e_plus = np.linalg.norm(dRp - target)
        e_minus = np.linalg.norm(dRm - target)
        s = 1 if e_plus <= e_minus else -1
        signs.append(s)
        print(f"  {jn:16s} s={s:+d}   |+err|={e_plus:.4f} |-err|={e_minus:.4f}  (motion_mag={np.linalg.norm(dL):.4f})")

    print("\nSIGN_HAND (canonical 22 order):")
    print(signs)
    print("torch.tensor(" + str([float(s) for s in signs]) + ")")


if __name__ == "__main__":
    main()
