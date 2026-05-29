# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Recover MANO parameters in aligned camera space from the alignment pipeline outputs.

Strategy
--------
Use DynHaMR's cam_R as the world→camera rotation, then compute translation
analytically so the centroid of the FK mesh matches the aligned mesh centroid.

For each hand track and each frame:
  1. R_w2c = cam_R[h, f]  (DynHaMR's world→camera rotation matrix).
  2. global_orient = R_w2c @ R_root_eff  (R_root_eff is the DynHaMR root
     orientation, converted to left-hand space for left tracks).
  3. transl is computed from MANO FK identities:
       LBS(R, pose, beta, t=0) = R @ (v_I - J0) + J0
     where v_I = LBS(I, pose, beta, t=0) and J0 is the rest-pose root joint.
     Equating the centroid of LBS + transl with mean(aligned_verts) gives:
       transl = mean(aligned) - R_eff_full @ (mean(v_I) - J0) - J0

Note: Procrustes on the full mesh was tried but produces distorted rotations
because the alignment pipeline shifts depth by ~50 cm; this depth offset
changes the centered shape of the point cloud and confuses the Procrustes fit.

DynHaMR/manotorch left-hand convention
---------------------------------------
DynHaMR: runs MANO_RIGHT for ALL tracks, then negates x-coords for left tracks.
  v_world_left = M_flip @ MANO_RIGHT_FK(root_orient, trans, pose_body, betas)
  where M_flip = diag(-1, 1, 1).
manotorch: uses MANO_LEFT (= MANO_RIGHT with x-mirrored template/weights).
  v = MANO_LEFT_FK(global_orient_l, transl_l, hand_pose_l, betas)
  ≡ M_flip @ MANO_RIGHT_FK(M_flip @ R_l @ M_flip, M_flip @ t_l, r_l * [1,-1,-1], betas)
Equating these gives the axis-angle transform described above.

Inputs
------
  aligned_path        Final aligned NPZ (smooth_hand_mesh output).
                      Uses verts (B, T, 778, 3) in target camera space.
  world_results_path  DynHaMR world_results.npz.
                      Contains cam_R (B,T,3,3), cam_t (B,T,3),
                      root_orient (B,T,3), trans (B,T,3), world_scale,
                      betas (B,10), pose_body (B,T,15,3).
  hand_mesh_traj_path Original hand_mesh_traj NPZ.
                      Contains verts (B, T, 778, 3) in DynHaMR world frame.
  mano_model_dir      Directory containing MANO_RIGHT.pkl and MANO_LEFT.pkl.
  output_path         Output NPZ path.

Output NPZ schema
-----------------
  betas          (B, 10)    shape params — passed through
  global_orient  (B, T, 3)  axis-angle in camera space (manotorch convention)
  transl         (B, T, 3)  metric camera space
  hand_pose      (B, T, 45) finger joints (manotorch convention, left converted)
  is_right       (B, T)     passed through from aligned NPZ
  vertex_rmsd    (B, T)     per-vertex RMS between FK mesh and aligned mesh (diagnostic)
  flat_hand_mean scalar     always True — DynHaMR pose_body is full axis-angle, hands_mean must NOT be re-added
"""

import argparse
import os
import pickle
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation

# Flip matrix: negates x-axis.  M_flip @ R(r) @ M_flip = R(FLIP_AA @ r)
M_FLIP = np.diag([-1.0, 1.0, 1.0])
# Axis-angle conjugation by M_flip: (rx, ry, rz) → (rx, -ry, -rz)
FLIP_AA = np.array([1.0, -1.0, -1.0])


# ---------------------------------------------------------------------------
# Minimal numpy MANO helpers
# ---------------------------------------------------------------------------

def _rodrigues(r: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → (3,3) rotation matrix."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3)
    n = r / theta
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _load_mano_model(pkl_path: str) -> dict:
    """Load MANO pkl, returning arrays needed for FK."""
    if 'chumpy' not in sys.modules:
        class _Ch:
            def __new__(cls, *args, **kwargs):
                return object.__new__(cls)
            def __init__(self, x=None, *args, **kwargs):
                self._state = {'x': np.asarray(x)} if x is not None else {}
            def __setstate__(self, state):
                self._state = state if isinstance(state, dict) else {'x': state}
            def _resolve(self) -> np.ndarray:
                state = self._state
                if 'x' in state:
                    v = state['x']
                    return v._resolve() if isinstance(v, _Ch) else np.asarray(v)
                if 'a' in state and 'idxs' in state:
                    a = state['a']
                    a = a._resolve() if isinstance(a, _Ch) else np.asarray(a)
                    idxs = np.asarray(state['idxs'])
                    result = a.flatten()[idxs]
                    ps = state.get('preferred_shape')
                    return result.reshape(ps) if ps is not None else result
                return np.array([])
            def __array__(self, dtype=None):
                return np.asarray(self._resolve(), dtype=dtype)
            @property
            def r(self):
                return self._resolve()

        class _ChumMod(types.ModuleType):
            def __getattr__(self, name: str):
                return _Ch

        stub = _ChumMod('chumpy')
        stub.__path__ = []
        stub.Ch = _Ch
        sys.modules['chumpy'] = stub
        for _sub in ['reordering', 'utils', 'ch', 'logic']:
            _m = _ChumMod(f'chumpy.{_sub}')
            sys.modules[f'chumpy.{_sub}'] = _m
            setattr(stub, _sub, _m)

    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')

    model = {
        'v_template':  np.array(raw['v_template'],  dtype=np.float64),
        'shapedirs':   np.array(raw['shapedirs'],   dtype=np.float64),
        'posedirs':    np.array(raw['posedirs'],    dtype=np.float64),
        'weights':     np.array(raw['weights'],     dtype=np.float64),
        'hands_mean':  np.array(raw['hands_mean'],  dtype=np.float64),
        'parents':     np.array(raw['kintree_table'][0], dtype=np.int32),
    }
    jr = raw['J_regressor']
    model['J_regressor'] = np.array(jr.todense(), dtype=np.float64)
    return model


def _mano_fk_zero(model: dict, hand_pose: np.ndarray, betas: np.ndarray) -> tuple:
    """MANO FK with identity global_orient and zero transl.

    hand_pose: (45,) without hands_mean (raw DynHaMR convention).
    Returns (v_out (778,3), J0 (3,)).
    """
    full_pose = np.concatenate([np.zeros(3), hand_pose])
    v_shaped = model['v_template'] + np.einsum('ijk,k->ij', model['shapedirs'], betas)
    J = model['J_regressor'] @ v_shaped
    R = np.stack([_rodrigues(full_pose[3*i:3*i+3]) for i in range(16)])
    pose_feature = (R[1:] - np.eye(3)).reshape(-1)
    v_posed = v_shaped + np.einsum('ijk,k->ij', model['posedirs'], pose_feature)
    parents = model['parents']
    G = np.zeros((16, 4, 4))
    for k in range(16):
        local = np.eye(4)
        local[:3, :3] = R[k]
        local[:3, 3] = J[k] if k == 0 else J[k] - J[parents[k]]
        G[k] = G[parents[k]] @ local if k > 0 else local
    G_final = np.zeros((16, 4, 4))
    for k in range(16):
        offset = np.eye(4)
        offset[:3, 3] = -J[k]
        G_final[k] = G[k] @ offset
    T = np.einsum('vk,kij->vij', model['weights'], G_final)
    v_homo = np.concatenate([v_posed, np.ones((len(v_posed), 1))], axis=1)
    v_out = np.einsum('vij,vj->vi', T, v_homo)[:, :3]
    return v_out.astype(np.float64), J[0]


def _to_left_hand(R_root: np.ndarray, pose_body_45: np.ndarray) -> tuple:
    """Convert DynHaMR right-hand-space root rotation and finger pose to manotorch left-hand-space.

    DynHaMR stores ALL tracks (including left) in right-hand MANO space and
    negates x at geometry level. manotorch LEFT expects params in the mirrored
    coordinate system. The conversion is a conjugation by M_flip = diag(-1,1,1):
      R_left = M_flip @ R_right @ M_flip
      r_left = FLIP_AA * r_right  (negate y,z of each axis-angle)
    """
    R_root_left = M_FLIP @ R_root @ M_FLIP
    pose_left = (pose_body_45.reshape(15, 3) * FLIP_AA).reshape(45).astype(np.float32)
    return R_root_left, pose_left


def recover_mano_params(
    aligned_path: str,
    world_results_path: str,
    hand_mesh_traj_path: str,
    mano_model_dir: str,
    output_path: str,
) -> None:
    aligned = np.load(aligned_path,       allow_pickle=True)
    wr      = np.load(world_results_path, allow_pickle=True)

    aligned_verts = aligned["verts"].astype(np.float64)   # (B, T, 778, 3) camera space

    cam_R = wr["cam_R"].astype(np.float64)                      # (B, T, 3, 3) world→cam rotation
    root_orient_world = wr["root_orient"].astype(np.float64)    # (B, T, 3) axis-angle
    pose_body = wr["pose_body"].astype(np.float32).reshape(*aligned_verts.shape[:2], 45)
    betas_all = wr["betas"].astype(np.float64)                  # (B, 10)
    is_right  = aligned["is_right"]                             # (B, T)

    B, T = aligned_verts.shape[:2]

    global_orient = np.zeros((B, T, 3), dtype=np.float32)
    transl        = np.zeros((B, T, 3), dtype=np.float32)
    hand_pose_out = pose_body.copy()
    vertex_rmsd   = np.zeros((B, T),   dtype=np.float32)

    # Per-track left/right flag (majority vote across frames)
    is_right_track = is_right.mean(axis=1) > 0.5  # (B,)

    models = {
        'right': _load_mano_model(os.path.join(mano_model_dir, 'MANO_RIGHT.pkl')),
        'left':  _load_mano_model(os.path.join(mano_model_dir, 'MANO_LEFT.pkl')),
    }

    for h in range(B):
        left_hand = not is_right_track[h]
        side = 'left' if left_hand else 'right'
        betas_h = betas_all[h]

        for f in range(T):
            # Use DynHaMR's cam_R as the world→camera rotation
            R_w2c = cam_R[h, f]

            # Root orientation in DynHaMR world space
            R_root = Rotation.from_rotvec(root_orient_world[h, f]).as_matrix()

            if left_hand:
                R_root_eff, hand_pose_f = _to_left_hand(R_root, pose_body[h, f])
            else:
                R_root_eff = R_root
                hand_pose_f = pose_body[h, f]

            # Full camera-space rotation: world→cam composed with root orient
            R_eff_full = R_w2c @ R_root_eff
            global_orient[h, f] = Rotation.from_matrix(R_eff_full).as_rotvec().astype(np.float32)

            # Transl from MANO FK identity: LBS(R,...,t=0) = R @ (v_I - J0) + J0
            # => t = mean(aligned) - R_eff_full @ (mean(v_I) - J0) - J0
            v_I, J0 = _mano_fk_zero(models[side], hand_pose_f, betas_h)
            mean_aligned = aligned_verts[h, f].mean(0)
            t = mean_aligned - R_eff_full @ (v_I.mean(0) - J0) - J0
            transl[h, f] = t.astype(np.float32)

            # Diagnostic: per-vertex RMSD between FK mesh and aligned mesh
            v_fk = (R_eff_full @ (v_I - J0).T).T + J0 + t
            vertex_rmsd[h, f] = float(np.sqrt(((v_fk - aligned_verts[h, f]) ** 2).sum(-1).mean()))

        # Convert finger pose for left hands (done per-track over all frames at once)
        if left_hand:
            hand_pose_out[h] = (pose_body[h].reshape(T, 15, 3) * FLIP_AA).reshape(T, 45)

    np.savez_compressed(
        output_path,
        betas         = wr["betas"].astype(np.float32),
        global_orient = global_orient,
        transl        = transl,
        hand_pose     = hand_pose_out,
        is_right      = is_right,
        vertex_rmsd   = vertex_rmsd,
        flat_hand_mean = np.bool_(True),
    )

    print(f"Vertex RMSD — mean: {vertex_rmsd.mean()*1000:.2f} mm  "
          f"max: {vertex_rmsd.max()*1000:.2f} mm")
    print(f"Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover MANO params from aligned hand mesh")
    parser.add_argument("--aligned_path",        required=True)
    parser.add_argument("--world_results_path",  required=True)
    parser.add_argument("--hand_mesh_traj_path", required=True)
    parser.add_argument("--mano_model_dir",      required=True)
    parser.add_argument("--output_path",         required=True)
    args = parser.parse_args()

    recover_mano_params(
        aligned_path        = args.aligned_path,
        world_results_path  = args.world_results_path,
        hand_mesh_traj_path = args.hand_mesh_traj_path,
        mano_model_dir      = args.mano_model_dir,
        output_path         = args.output_path,
    )


if __name__ == "__main__":
    main()
