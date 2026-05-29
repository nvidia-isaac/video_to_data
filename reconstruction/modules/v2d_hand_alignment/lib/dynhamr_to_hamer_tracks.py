# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Convert a DynHaMR ``world_results.npz`` to per-frame v2d_hamer-style tracks.

Each frame's hand state is re-expressed in DynHaMR's *per-frame camera*
coordinates, so the output mirrors what a real HaMeR detection looks like
on disk. This lets ``v2d_hamer.lib.align_hands`` consume the tracks as-is
— no world / camera-pose interaction, no global SLAM, just per-frame
cam-frame hand detections that the proven HaMeR alignment can refine.

Conceptually: we drop everything DynHaMR estimated about the world (and
about ViPE's noisy intrinsics-vs-pose entanglement) and keep only the
per-frame hand pose in cam frame. Whatever ViPE got right is preserved
(image-stability — DynHaMR's joint optimization keeps the hand anchored to
its pixels under ViPE intrins). Whatever ViPE got wrong (focal length,
camera-pose noise) is *fully absorbed* by the per-frame cam-frame
representation, which has no notion of world or inter-frame extrinsics.

Schema produced (matches HaMeR pre-alignment, what ``v2d_hamer.align_hands``
expects):

    <output_dir>/<track_id>/<frame:06d>.json
    {
      "track_id":  int,                 # e.g. 2=left, 3=right
      "is_right":  bool,
      "frame_idx": int,
      "image_size": [W, H],
      "camera": {
        "scaled_focal_length": float,    # = fx_vipe; align_hands rescales to fx_real
        "pred_cam_t_full":     [x,y,z]   # cam-frame translation in DynHaMR units
      },
      "mano": {
        "betas":         [10],
        "global_orient": [3],            # axis-angle, in DynHaMR cam frame
        "hand_pose":     [45]
      }
    }

Math
----
Per (hand h, frame t):
  Let M = diag(-1, 1, 1) for left, I for right (DynHaMR convention).

  R_root_cam  = M · cam_R · M · R(root_orient_world)        # = cam_R·R for right
  global_aa   = axis_angle(R_root_cam)
  cam_t_cam   = cam_R · (M · trans) + cam_t                 # = cam_R·trans+cam_t for right

Verification (per-vertex projection equality):
  v_world_dynhamr = M · (verts_local_mano + trans)          # mirror after add for left
  v_cam_dynhamr   = cam_R · v_world_dynhamr + cam_t
                  = cam_R · M · verts_local + cam_R · M · trans + cam_t
                  = (cam_R·M·verts_local) + cam_t_cam

  v_cam_hamer     = M · verts_local_hamer + cam_t_hamer    # v2d_hamer left convention
                  = M · R(global_aa) · root_relative + cam_t_hamer
                  = M · (M·cam_R·M·R_root_world) · root_relative + cam_t_cam
                  = cam_R · M · R_root_world · root_relative + cam_t_cam
                  = cam_R · M · verts_local + cam_t_cam                ✓

So projecting v_cam_hamer through the same intrinsics gives the same pixel
the unaligned DynHaMR render produces — image-stable by construction.

Usage
-----
    python -m v2d.hand_alignment.lib.dynhamr_to_hamer_tracks \\
        --input_npz       /data/.../world_results.npz \\
        --output_dir      /data/.../dynhamr_as_hamer_tracks \\
        [--intrinsics_path /data/.../intrinsics_stable.json]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Rotation helpers (axis-angle ↔ rotmat)
# ---------------------------------------------------------------------------

def _axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → (3, 3) rotation matrix via Rodrigues' formula.

    Handles the small-angle case with a Taylor expansion of (1 − cos θ)/θ²
    so the result is smooth as ||aa|| → 0.
    """
    aa = np.asarray(aa, dtype=np.float64)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2],  k[1]],
                  [k[2],  0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    s, c = np.sin(theta), np.cos(theta)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """(3, 3) → (3,) axis-angle. Canonical |θ| ≤ π."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        # Near identity: axis is undefined; small-angle approximation gives
        # aa ≈ (R − R.T)/2 vee-mapped.
        return np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]], dtype=np.float64) * 0.5
    if theta > np.pi - 1e-6:
        # Near π: (R − R.T) → 0, so recover axis from diagonal of (R + I)/2.
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]])
        i = int(np.argmax(diag))
        axis = np.sqrt(np.clip((diag + 1.0) / 2.0, 0.0, None))
        # Fix signs from off-diagonal: axis[j] = (R[i,j] + R[j,i]) / (4 axis[i])
        if axis[i] > 1e-8:
            for j in range(3):
                if j == i:
                    continue
                axis[j] = (R[i, j] + R[j, i]) / (4.0 * axis[i])
                if axis[j] < 0:
                    axis[j] = -axis[j]
        # Sign of axis[i] from R[i, i].
        # (Both ±axis represent the same rotation when θ = π, so pick + arbitrarily.)
        return axis * theta
    sin_theta = float(np.sin(theta))
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]], dtype=np.float64) / (2.0 * sin_theta)
    return axis * theta


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

_M_MIRROR = np.diag([-1.0, 1.0, 1.0])


def convert_dynhamr_to_hamer_tracks(
    input_npz: str,
    output_dir: str,
    intrinsics_path: Optional[str] = None,
    image_size: Optional[tuple[int, int]] = None,
    left_id: int = 2,
    right_id: int = 3,
) -> None:
    """Write per-frame v2d_hamer-style JSON records into ``output_dir``.

    ``intrinsics_path`` is only used to override (W, H); intrinsic
    conversion happens later inside ``align_hands`` (which reads its own
    ``--intrinsics_path``). When ``intrinsics_path`` is omitted, (W, H) is
    inferred from ViPE's ``cx, cy`` assuming a center-aligned principal
    point.
    """
    wr = np.load(input_npz, allow_pickle=True)

    trans       = wr["trans"].astype(np.float64)             # (B, T, 3)
    root_orient = wr["root_orient"].astype(np.float64)       # (B, T, 3)
    pose_body   = wr["pose_body"].astype(np.float64)         # (B, T, 15, 3)
    betas       = wr["betas"].astype(np.float64)             # (B, 10)
    cam_R       = wr["cam_R"].astype(np.float64)             # (B, T, 3, 3)
    cam_t       = wr["cam_t"].astype(np.float64)             # (B, T, 3)
    is_right    = wr["is_right"]                              # (B, T)
    intrins_v   = wr["intrins"].astype(np.float64)           # (4,)

    B, T = trans.shape[:2]
    is_right_track = is_right.mean(axis=1) > 0.5             # (B,)

    fx_vipe = float(intrins_v[0])
    if image_size is None:
        if intrinsics_path is not None:
            with open(intrinsics_path) as f:
                k = json.load(f)
            image_size = (int(k["width"]), int(k["height"]))
        else:
            # Fall back to 2·cx, 2·cy from ViPE (assumes center-aligned).
            image_size = (int(round(2 * float(intrins_v[2]))),
                          int(round(2 * float(intrins_v[3]))))
    W, H = image_size

    print(f"Converting DynHaMR npz → HaMeR-style tracks:")
    print(f"  hands: {B}  frames: {T}  image: {W}×{H}")
    print(f"  scaled_focal_length = fx_vipe = {fx_vipe:.1f}")
    print(f"  hand IDs: left={left_id}  right={right_id}")

    os.makedirs(output_dir, exist_ok=True)
    n_written = 0
    for h in range(B):
        is_r = bool(is_right_track[h])
        track_id = right_id if is_r else left_id
        track_dir = os.path.join(output_dir, str(track_id))
        os.makedirs(track_dir, exist_ok=True)

        M = np.eye(3) if is_r else _M_MIRROR
        b_list = betas[h].tolist()

        for f in range(T):
            R_root_world = _axis_angle_to_rotmat(root_orient[h, f])
            # Right: cam_R · R_world. Left: M · cam_R · M · R_world.
            R_root_cam = M @ cam_R[h, f] @ M @ R_root_world
            aa_cam = _rotmat_to_axis_angle(R_root_cam)

            # cam_t in DynHaMR cam frame.
            #   right: cam_R · trans + cam_t
            #   left:  cam_R · (M · trans) + cam_t
            cam_t_cam = cam_R[h, f] @ (M @ trans[h, f]) + cam_t[h, f]

            rec = {
                "track_id":   int(track_id),
                "is_right":   is_r,
                "frame_idx":  int(f),
                "image_size": [int(W), int(H)],
                "camera": {
                    "scaled_focal_length": fx_vipe,
                    "pred_cam_t_full":     cam_t_cam.tolist(),
                },
                "mano": {
                    "betas":         list(b_list),
                    "global_orient": aa_cam.tolist(),
                    "hand_pose":     pose_body[h, f].reshape(-1).tolist(),
                },
            }
            out_path = os.path.join(track_dir, f"{f:06d}.json")
            with open(out_path, "w") as fp:
                json.dump(rec, fp, indent=2)
            n_written += 1

        side = "right" if is_r else "left"
        print(f"  track {track_id} ({side}): wrote {T} frames → {track_dir}")

    print(f"Total {n_written} records → {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_npz",       required=True,
                   help="Path to DynHaMR world_results.npz")
    p.add_argument("--output_dir",      required=True,
                   help="Output root; per-track subdirs are created inside")
    p.add_argument("--intrinsics_path", default=None,
                   help="JSON with width/height — only used to set image_size. "
                        "Per-frame intrinsics rescaling happens later in "
                        "align_hands. If omitted, (W, H) is inferred from "
                        "ViPE's (cx, cy).")
    p.add_argument("--left_id",  type=int, default=2,
                   help="Track-id subdir for the left hand (default 2, "
                        "matching SAM2 mask convention).")
    p.add_argument("--right_id", type=int, default=3,
                   help="Track-id subdir for the right hand (default 3).")
    args = p.parse_args()
    convert_dynhamr_to_hamer_tracks(
        input_npz       = args.input_npz,
        output_dir      = args.output_dir,
        intrinsics_path = args.intrinsics_path,
        left_id         = args.left_id,
        right_id        = args.right_id,
    )


if __name__ == "__main__":
    main()
