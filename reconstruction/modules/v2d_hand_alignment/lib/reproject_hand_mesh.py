"""
Reproject hand mesh vertices from world frame + hand intrinsics to per-frame
camera space using target intrinsics.

Two transforms are applied in sequence:

  0. World scale (optional)
     DynHaMR solves with a monocular scale ambiguity. world_results.npz may
     contain a scalar 'world_scale' that converts internal units to metric.
     When present it is applied to verts/joints before the w2c transform:
         v_metric = v_world * world_scale

  1. World → camera space (per-frame)
     DynHaMR outputs verts in a fixed world frame. To bring verts into frame
     t's camera space:
         v_cam = R @ v_world + t   (w2c convention)

     Two pose sources are supported:
       --world_results  DynHaMR world_results.npz with cam_R (n_hands,n_frames,3,3)
                        and cam_t (n_hands,n_frames,3) — per-track, per-frame.
                        If the file also contains 'world_scale', it is applied
                        automatically before the transform.
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
    python -m v2d.hand_alignment.lib.reproject_hand_mesh \
        --input             data/.../hand/hand_mesh/airplane_hand_mesh_traj_000300.npz \
        --world_results     data/.../hand/airplane_000300_world_results.npz \
        --hand_intrinsics   data/.../hand/intrinsics/intrinsics.json \
        --target_intrinsics data/.../outputs/intrinsics_moge_stable.json \
        --output            data/.../hand/hand_mesh/airplane_hand_mesh_traj_000300_aligned.npz

    # Legacy: single pose file (same w2c for all hand tracks)
    python -m v2d.hand_alignment.lib.reproject_hand_mesh \
        --input ... --pose ... --hand_intrinsics ... --target_intrinsics ... --output ...

    # Intrinsics-only (no pose, e.g. static camera):
    python -m v2d.hand_alignment.lib.reproject_hand_mesh \
        --input ... --hand_intrinsics ... --target_intrinsics ... --output ...

    # No hand intrinsics (skip lateral reprojection, world→camera only):
    python -m v2d.hand_alignment.lib.reproject_hand_mesh \
        --input ... --world_results ... --target_intrinsics ... --output ...
"""

import argparse
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp


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

    For each vertex:
      1. Project to pixel using src intrinsics (pixel position is ground truth)
      2. Compute metric depth: centroid z is scaled by (dst_fx/src_fx) to correct
         the focal-length inflation; each vertex's depth relative to the centroid
         is preserved unchanged (MANO shape is already in metric).
      3. Unproject from pixel using dst intrinsics at the new depth.

    This gives:
      - Exact pixel alignment for every vertex (no drift at fingertips/wrist)
      - Correct absolute depth (centroid at dst-camera metric scale)
      - Correct 3D hand shape (relative depths not squished)

    src_res/dst_res: (width, height) — used to rescale pixel coords if
    the estimator ran on a different resolution than the intrinsics describe.
    """
    shape = verts.shape
    v = verts.reshape(-1, 3).astype(np.float64)

    z = v[:, 2]
    safe_z = np.where(z > 1e-6, z, 1e-6)

    # Step 1: project to pixels with src intrinsics
    u  = v[:, 0] * src['fx'] / safe_z + src['cx']
    vp = v[:, 1] * src['fy'] / safe_z + src['cy']

    if src_res is not None and dst_res is not None:
        u  = u  * dst_res[0] / src_res[0]
        vp = vp * dst_res[1] / src_res[1]

    # Step 2: compute per-vertex metric depth
    #   centroid z is scaled by the focal-length ratio (corrects depth inflation)
    #   each vertex's depth relative to the centroid is kept as-is (already metric)
    z_centroid = safe_z.mean()
    scale = dst['fx'] / src['fx']
    z_new = z_centroid * scale + (safe_z - z_centroid)
    z_new = np.where(z_new > 1e-6, z_new, 1e-6)

    # Step 3: unproject with dst intrinsics at the corrected depth
    x_new = (u  - dst['cx']) * z_new / dst['fx']
    y_new = (vp - dst['cy']) * z_new / dst['fy']

    return np.stack([x_new, y_new, z_new], axis=-1).reshape(shape).astype(np.float32)


def reproject_hand_mesh(
    input_path: str,
    target_intrinsics_path: str,
    output_path: str,
    world_results_path: str | None = None,
    pose_path: str | None = None,
    hand_intrinsics_path: str | None = None,
    apply_world_scale: bool = False,
    smooth_poses_sigma: float = 0.0,
    hand_width: int | None = None,
    hand_height: int | None = None,
) -> None:
    """Reproject hand mesh from world frame to per-frame camera space with target intrinsics."""
    hand_intr   = load_intrinsics(hand_intrinsics_path) if hand_intrinsics_path else None
    target_intr = load_intrinsics(target_intrinsics_path)

    if hand_intr is not None:
        src_res = (hand_width, hand_height) if (hand_width and hand_height) else \
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
    else:
        src_res = dst_res = None
        print("No hand intrinsics provided — skipping lateral reprojection (z preserved)")

    mesh_data = np.load(input_path, allow_pickle=True)
    out = dict(mesh_data)

    verts  = mesh_data['verts'].copy()   # (n_hands, n_frames, n_verts,  3)
    joints = mesh_data['joints'].copy()  # (n_hands, n_frames, n_joints, 3)
    n_hands, n_frames = verts.shape[:2]

    # --- Step 1: world → per-frame camera space ---
    if world_results_path:
        wr = np.load(world_results_path, allow_pickle=True)
        cam_R = wr['cam_R'].astype(np.float64)  # (n_hands, n_frames, 3, 3)
        cam_t = wr['cam_t'].astype(np.float64)  # (n_hands, n_frames, 3)

        if apply_world_scale and 'world_scale' in wr:
            world_scale = float(wr['world_scale'].flat[0])
            print(f"Applying world_scale={world_scale:.4f} to convert DynHaMR internal units to metric...")
            verts  *= world_scale
            joints *= world_scale
        elif 'world_scale' in wr and not apply_world_scale:
            print(f"world_scale={float(wr['world_scale'].flat[0]):.4f} found but not applied (use apply_world_scale=True to enable)")

        if smooth_poses_sigma > 0:
            sigma = smooth_poses_sigma
            print(f"Smoothing cam_R/cam_t with sigma={sigma} frames...")
            for h in range(n_hands):
                # Smooth translation with Gaussian
                cam_t[h] = gaussian_filter1d(cam_t[h], sigma=sigma, axis=0)
                # Smooth rotation via SLERP on the Rotation object
                rots = Rotation.from_matrix(cam_R[h])         # (n_frames,)
                rotvecs = rots.as_rotvec()                     # (n_frames, 3)
                smoothed_rv = gaussian_filter1d(rotvecs, sigma=sigma, axis=0)
                cam_R[h] = Rotation.from_rotvec(smoothed_rv).as_matrix()

        print(f"Applying per-track per-frame w2c from world_results (cam_R/cam_t) across {n_hands} hands × {n_frames} frames...")
        for h in range(n_hands):
            for f in range(n_frames):
                R, t = cam_R[h, f], cam_t[h, f]
                verts[h, f]  = apply_w2c(verts[h, f],  R, t)
                joints[h, f] = apply_w2c(joints[h, f], R, t)
    elif pose_path:
        pose_data = np.load(pose_path, allow_pickle=True)
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
    if hand_intr is not None:
        print("Re-projecting from hand intrinsics to target intrinsics...")
        out['verts']  = reproject_intrinsics(verts,  hand_intr, target_intr, src_res, dst_res)
        out['joints'] = reproject_intrinsics(joints, hand_intr, target_intr, src_res, dst_res)
    else:
        print("Skipping intrinsics reprojection.")
        out['verts']  = verts
        out['joints'] = joints

    np.savez(output_path, **out)
    print(f"Saved → {output_path}")

    h, f = 0, 0
    before = mesh_data['verts'][h, f, 0]
    after  = out['verts'][h, f, 0]
    print(f"\nSample vert [hand=0, frame=0, vert=0]:")
    print(f"  before: x={before[0]:.4f}  y={before[1]:.4f}  z={before[2]:.4f}")
    print(f"  after:  x={after[0]:.4f}  y={after[1]:.4f}  z={after[2]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Reproject hand mesh to per-frame camera space + new intrinsics")
    parser.add_argument('--input_path',             required=True,  help='Input NPZ hand mesh (world frame)')
    parser.add_argument('--world_results_path',     default=None,   help='DynHaMR world_results.npz with cam_R/cam_t (n_hands,n_frames,...)')
    parser.add_argument('--pose_path',              default=None,   help='Legacy pose NPZ with w2c matrices (data: N,4,4)')
    parser.add_argument('--hand_intrinsics_path',   default=None,   help='Intrinsics used by hand estimator (JSON); if omitted, skips lateral reprojection')
    parser.add_argument('--target_intrinsics_path', required=True,  help='Target intrinsics to align to (JSON)')
    parser.add_argument('--output_path',            required=True,  help='Output NPZ path')
    parser.add_argument('--apply_world_scale', action='store_true',
                        help='Apply world_scale from world_results.npz (DynHaMR monocular scale factor) before w2c transform')
    parser.add_argument('--smooth_poses_sigma', type=float, default=0.0,
                        help='Gaussian sigma (frames) for temporal smoothing of cam_R/cam_t before w2c. '
                             'Reduces DynHaMR camera-trajectory noise that causes per-frame image misalignment.')
    parser.add_argument('--hand_width',  type=int, default=None,
                        help='Image width hand estimator ran on (if different from hand_intrinsics["width"])')
    parser.add_argument('--hand_height', type=int, default=None,
                        help='Image height hand estimator ran on (if different from hand_intrinsics["height"])')
    args = parser.parse_args()

    reproject_hand_mesh(
        input_path=args.input_path,
        target_intrinsics_path=args.target_intrinsics_path,
        output_path=args.output_path,
        world_results_path=args.world_results_path,
        pose_path=args.pose_path,
        hand_intrinsics_path=args.hand_intrinsics_path,
        apply_world_scale=args.apply_world_scale,
        smooth_poses_sigma=args.smooth_poses_sigma,
        hand_width=args.hand_width,
        hand_height=args.hand_height,
    )


if __name__ == '__main__':
    main()
