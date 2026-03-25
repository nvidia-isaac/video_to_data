"""
Reproject hand mesh vertices from world frame + hand intrinsics to per-frame
camera space using target intrinsics.

Two transforms are applied in sequence:

  1. World → camera space (per-frame)
     DynHaMR outputs verts in a fixed world frame. To bring verts into frame
     t's camera space:
         v_cam = R @ v_world + t   (w2c convention)

     Two pose sources are supported:
       --world_results  DynHaMR world_results.npz with cam_R (n_hands,n_frames,3,3)
                        and cam_t (n_hands,n_frames,3) — per-track, per-frame.
       --pose           Legacy: NPZ with 'data' key (n_frames,4,4) w2c matrices
                        applied identically across all hand tracks.

  2. Intrinsics re-projection
     The hand estimator ran with intrinsics A; we want verts consistent with
     intrinsics B (e.g. MoGe). Pixel observations are ground truth; only x/y
     lateral positions change (z is preserved):
         u  = x * fx_A / z + cx_A
         x' = (u - cx_B) * z / fx_B   (same for y)

Usage:
    # Recommended: use world_results.npz with per-track poses
    python scripts/reproject_hand_mesh.py \
        --input             data/.../hand/hand_mesh/airplane_hand_mesh_traj_000300.npz \
        --world_results     data/.../hand/airplane_000300_world_results.npz \
        --hand_intrinsics   data/.../hand/intrinsics/intrinsics.json \
        --target_intrinsics data/.../outputs/intrinsics_moge_stable.json \
        --output            data/.../hand/hand_mesh/airplane_hand_mesh_traj_000300_aligned.npz

    # Legacy: single pose file (same w2c for all hand tracks)
    python scripts/reproject_hand_mesh.py \
        --input ... --pose ... --hand_intrinsics ... --target_intrinsics ... --output ...

    # Intrinsics-only (no pose, e.g. static camera):
    python scripts/reproject_hand_mesh.py \
        --input ... --hand_intrinsics ... --target_intrinsics ... --output ...
"""

import argparse
import json

import numpy as np


def load_intrinsics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_w2c(verts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transform (..., 3) world-frame verts into camera space.
        v_cam = R @ v_world + t
    Matches DynHaMR's cam_util.reproject convention.
    R: (3,3), t: (3,)
    """
    shape = verts.shape
    v = verts.reshape(-1, 3).astype(np.float64)
    v_cam = (R @ v.T).T + t
    return v_cam.reshape(shape).astype(np.float32)


def reproject_intrinsics(
    verts: np.ndarray,
    src: dict,
    dst: dict,
    src_res: tuple[int, int] | None = None,
    dst_res: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Reproject (..., 3) verts from src intrinsics to dst intrinsics.
    Depth (z) is preserved; only x/y change.
    src_res/dst_res: (width, height) — used to rescale pixel coords if
    the estimator ran on a different resolution than the intrinsics describe.
    """
    shape = verts.shape
    v = verts.reshape(-1, 3).astype(np.float64)

    z = v[:, 2]
    safe_z = np.where(z > 1e-6, z, 1e-6)

    u  = v[:, 0] * src['fx'] / safe_z + src['cx']
    vp = v[:, 1] * src['fy'] / safe_z + src['cy']

    if src_res is not None and dst_res is not None:
        u  = u  * dst_res[0] / src_res[0]
        vp = vp * dst_res[1] / src_res[1]

    x_new = (u  - dst['cx']) * safe_z / dst['fx']
    y_new = (vp - dst['cy']) * safe_z / dst['fy']

    return np.stack([x_new, y_new, z], axis=-1).reshape(shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Reproject hand mesh to per-frame camera space + new intrinsics")
    parser.add_argument('--input',             required=True,  help='Input NPZ hand mesh (world frame)')
    parser.add_argument('--world_results',     default=None,   help='DynHaMR world_results.npz with cam_R/cam_t (n_hands,n_frames,...)')
    parser.add_argument('--pose',              default=None,   help='Legacy pose NPZ with w2c matrices (data: N,4,4)')
    parser.add_argument('--hand_intrinsics',   required=True,  help='Intrinsics used by hand estimator (JSON)')
    parser.add_argument('--target_intrinsics', required=True,  help='Target intrinsics to align to (JSON)')
    parser.add_argument('--output',            required=True,  help='Output NPZ path')
    parser.add_argument('--hand_width',  type=int, default=None,
                        help='Image width hand estimator ran on (if different from hand_intrinsics["width"])')
    parser.add_argument('--hand_height', type=int, default=None,
                        help='Image height hand estimator ran on (if different from hand_intrinsics["height"])')
    args = parser.parse_args()

    hand_intr   = load_intrinsics(args.hand_intrinsics)
    target_intr = load_intrinsics(args.target_intrinsics)

    src_res = (args.hand_width, args.hand_height) if (args.hand_width and args.hand_height) else \
              (hand_intr.get('width'), hand_intr.get('height'))
    dst_res = (target_intr.get('width'), target_intr.get('height'))
    if None in (src_res or [None]): src_res = None
    if None in (dst_res or [None]): dst_res = None

    print(f"Hand intrinsics:   fx={hand_intr['fx']:.2f}  fy={hand_intr['fy']:.2f}  "
          f"cx={hand_intr['cx']:.2f}  cy={hand_intr['cy']:.2f}"
          + (f"  res={src_res}" if src_res else ""))
    print(f"Target intrinsics: fx={target_intr['fx']:.2f}  fy={target_intr['fy']:.2f}  "
          f"cx={target_intr['cx']:.2f}  cy={target_intr['cy']:.2f}"
          + (f"  res={dst_res}" if dst_res else ""))

    mesh_data = np.load(args.input, allow_pickle=True)
    out = dict(mesh_data)

    verts  = mesh_data['verts'].copy()   # (n_hands, n_frames, n_verts,  3)
    joints = mesh_data['joints'].copy()  # (n_hands, n_frames, n_joints, 3)
    n_hands, n_frames = verts.shape[:2]

    # --- Step 1: world → per-frame camera space ---
    if args.world_results:
        wr = np.load(args.world_results, allow_pickle=True)
        cam_R = wr['cam_R'].astype(np.float64)  # (n_hands, n_frames, 3, 3)
        cam_t = wr['cam_t'].astype(np.float64)  # (n_hands, n_frames, 3)
        print(f"Applying per-track per-frame w2c from world_results (cam_R/cam_t) across {n_hands} hands × {n_frames} frames...")
        for h in range(n_hands):
            for f in range(n_frames):
                R, t = cam_R[h, f], cam_t[h, f]
                verts[h, f]  = apply_w2c(verts[h, f],  R, t)
                joints[h, f] = apply_w2c(joints[h, f], R, t)
    elif args.pose:
        pose_data = np.load(args.pose, allow_pickle=True)
        w2c_matrices = pose_data['data'].astype(np.float64)  # (n_frames, 4, 4)
        print(f"Applying per-frame w2c from pose NPZ across {n_frames} frames...")
        for f in range(n_frames):
            R = w2c_matrices[f, :3, :3]
            t = w2c_matrices[f, :3, 3]
            for h in range(n_hands):
                verts[h, f]  = apply_w2c(verts[h, f],  R, t)
                joints[h, f] = apply_w2c(joints[h, f], R, t)
    else:
        print("No pose file provided — skipping world→camera transform.")

    # --- Step 2: re-project from hand intrinsics to target intrinsics ---
    print("Re-projecting from hand intrinsics to target intrinsics...")
    out['verts']  = reproject_intrinsics(verts,  hand_intr, target_intr, src_res, dst_res)
    out['joints'] = reproject_intrinsics(joints, hand_intr, target_intr, src_res, dst_res)

    np.savez(args.output, **out)
    print(f"Saved → {args.output}")

    h, f = 0, 0
    before = mesh_data['verts'][h, f, 0]
    after  = out['verts'][h, f, 0]
    print(f"\nSample vert [hand=0, frame=0, vert=0]:")
    print(f"  before: x={before[0]:.4f}  y={before[1]:.4f}  z={before[2]:.4f}")
    print(f"  after:  x={after[0]:.4f}  y={after[1]:.4f}  z={after[2]:.4f}")


if __name__ == '__main__':
    main()
