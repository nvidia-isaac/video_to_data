"""
Recover MANO parameters in aligned camera space from the alignment pipeline outputs.

The alignment pipeline moves world-frame DynHaMR verts/joints into a target camera
space (MoGe or DA3), applies depth alignment, and smooths centroids.  The MANO shape
params (betas) and finger-joint poses (pose_body) are unchanged by these transforms;
only the global orientation and translation need to be re-solved.

Algorithm (per hand, per frame):
  1. Load world-frame joints from the original hand_mesh_traj NPZ.
  2. Scale by world_scale to convert DynHaMR internal units → metric.
  3. Procrustes solve between centred world joints and centred aligned joints
     to recover the rotation R (global_orient) and translation t (transl).
  4. Pass betas and pose_body straight through from world_results.

Inputs:
  aligned_path        Final aligned NPZ (output of smooth_hand_mesh).
                      Contains joints (B, T, 21, 3) in target camera space.
  world_results_path  DynHaMR world_results.npz.
                      Contains betas (B,10), pose_body (B,T,15,3),
                      root_orient (B,T,3), trans (B,T,3), world_scale (1,1).
  hand_mesh_traj_path Original hand_mesh_traj NPZ (save_hand_mesh_trajectory output).
                      Contains joints (B, T, 21, 3) in DynHaMR world frame.
  output_path         Output NPZ path.

Output NPZ schema:
  betas              (B, 10)       shape params — passed through from world_results
  global_orient      (B, T, 3)     axis-angle in aligned camera space
  transl             (B, T, 3)     metric, aligned camera space
  hand_pose          (B, T, 45)    finger joints axis-angle — from world_results pose_body
  is_right           (B, T)        passed through from aligned NPZ
  procrustes_rmsd    (B, T)        mean per-joint error after fit (metres); diagnostic
  flat_hand_mean     scalar bool   MANO convention flag (always False for DynHaMR)
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation


def _procrustes_rt(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find R, t (no scale) such that R @ src[i] + t ≈ dst[i] for all i.
    src, dst: (N, 3)
    Returns R as (3, 3) ndarray, t as (3,) ndarray.
    """
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    A = src - c_src
    B = dst - c_dst
    U, _, Vt = np.linalg.svd(B.T @ A)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0, d])
    R = (U @ D @ Vt).astype(np.float32)
    t = (c_dst - R @ c_src).astype(np.float32)
    return R, t


def recover_mano_params(
    aligned_path: str,
    world_results_path: str,
    hand_mesh_traj_path: str,
    output_path: str,
) -> None:
    aligned = np.load(aligned_path, allow_pickle=True)
    wr      = np.load(world_results_path, allow_pickle=True)
    traj    = np.load(hand_mesh_traj_path, allow_pickle=True)

    aligned_joints = aligned["joints"].astype(np.float64)   # (B, T, 21, 3)
    world_joints   = traj["joints"].astype(np.float64)       # (B, T, 21, 3)
    world_scale    = float(wr["world_scale"].flat[0])

    B, T = aligned_joints.shape[:2]

    # pose_body is (B, T, 15, 3) axis-angle per finger joint → flatten to (B, T, 45)
    pose_body = wr["pose_body"].astype(np.float32).reshape(B, T, 45)

    global_orient   = np.zeros((B, T, 3),  dtype=np.float32)
    transl          = np.zeros((B, T, 3),  dtype=np.float32)
    procrustes_rmsd = np.zeros((B, T),     dtype=np.float32)

    world_joints_metric = world_joints * world_scale

    for h in range(B):
        for f in range(T):
            src = world_joints_metric[h, f]   # (21, 3) world frame metric
            dst = aligned_joints[h, f]        # (21, 3) aligned camera space

            R, t = _procrustes_rt(src, dst)

            # RMSD after fit
            residuals = (R @ src.T).T + t - dst
            procrustes_rmsd[h, f] = float(np.sqrt((residuals ** 2).sum(axis=-1).mean()))

            global_orient[h, f] = Rotation.from_matrix(R).as_rotvec().astype(np.float32)
            transl[h, f]        = t

    np.savez_compressed(
        output_path,
        betas           = wr["betas"].astype(np.float32),   # (B, 10)
        global_orient   = global_orient,                     # (B, T, 3)
        transl          = transl,                            # (B, T, 3)
        hand_pose       = pose_body,                         # (B, T, 45)
        is_right        = aligned["is_right"],               # (B, T)
        procrustes_rmsd = procrustes_rmsd,                   # (B, T)
        flat_hand_mean  = np.bool_(False),
    )

    mean_rmsd = procrustes_rmsd.mean()
    max_rmsd  = procrustes_rmsd.max()
    print(f"Procrustes RMSD — mean: {mean_rmsd*1000:.2f} mm  max: {max_rmsd*1000:.2f} mm")
    print(f"Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover MANO params from aligned hand mesh")
    parser.add_argument("--aligned_path",        required=True)
    parser.add_argument("--world_results_path",  required=True)
    parser.add_argument("--hand_mesh_traj_path", required=True)
    parser.add_argument("--output_path",         required=True)
    args = parser.parse_args()

    recover_mano_params(
        aligned_path        = args.aligned_path,
        world_results_path  = args.world_results_path,
        hand_mesh_traj_path = args.hand_mesh_traj_path,
        output_path         = args.output_path,
    )


if __name__ == "__main__":
    main()
