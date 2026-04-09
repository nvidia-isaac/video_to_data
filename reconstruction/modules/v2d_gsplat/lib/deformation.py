"""
Deformation models for each entity type:
  - SmplDeformer: SMPL LBS (body/hand entities)
  - Se3Deformer: rigid SE(3) transform (object entities)
  - Learnable pose parameters for body (BodyPoseParams) and objects (ObjectPoseParams)
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def rotation_6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    r6d: (..., 6) - first two columns of the rotation matrix
    Returns: (..., 3, 3)
    """
    a1 = r6d[..., :3]
    a2 = r6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


class SmplDeformer:
    """
    Wraps the smplx SMPL model to provide:
      - Rest-pose (T-pose) vertices for Gaussian initialisation
      - Per-frame posed vertices for deformation
      - Per-joint transforms for full LBS with custom skinning weights
    """

    def __init__(
        self,
        smpl_model_dir: str,
        gender: str = 'neutral',
        model_type: str = 'smpl',
        device: str = 'cuda',
    ):
        import smplx
        self.device = device
        self.model_type = model_type

        self.body_model = smplx.create(
            smpl_model_dir,
            model_type=model_type,
            gender=gender,
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        ).to(device).eval()

        self._lbs_weights = self.body_model.lbs_weights.to(device)  # (V, J)
        self.num_joints = self._lbs_weights.shape[1]
        self.num_vertices = self._lbs_weights.shape[0]

    @property
    def lbs_weights(self) -> torch.Tensor:
        return self._lbs_weights

    @torch.no_grad()
    def get_rest_vertices(self, betas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return T-pose vertices for given shape (no pose applied). Returns (V, 3)."""
        B = 1
        if betas is None:
            betas = torch.zeros(B, 10, device=self.device)
        elif betas.ndim == 1:
            betas = betas.unsqueeze(0)
        zeros_pose = torch.zeros(B, self.body_model.NUM_BODY_JOINTS * 3, device=self.device)
        zeros_orient = torch.zeros(B, 3, device=self.device)
        zeros_transl = torch.zeros(B, 3, device=self.device)
        out = self.body_model(
            betas=betas,
            body_pose=zeros_pose,
            global_orient=zeros_orient,
            transl=zeros_transl,
        )
        return out.vertices.squeeze(0)  # (V, 3)

    def get_posed_vertices(
        self,
        global_orient: torch.Tensor,  # (1, 3)
        body_pose: torch.Tensor,       # (1, J_body*3)
        betas: torch.Tensor,           # (10,) or (1, 10)
        transl: torch.Tensor,          # (1, 3)
    ) -> torch.Tensor:
        """Return world-space vertices for given pose. Returns (1, V, 3) — keeps grad."""
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)
        out = self.body_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )
        return out.vertices  # (1, V, 3)

    def get_joint_transforms(
        self,
        global_orient: torch.Tensor,  # (1, 3)
        body_pose: torch.Tensor,       # (1, J_body*3)
        betas: torch.Tensor,           # (10,) or (1, 10)
        transl: torch.Tensor,          # (1, 3)
    ) -> torch.Tensor:
        """
        Compute per-joint world transform matrices A for full LBS.
        Returns (1, J, 4, 4).
        """
        from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform

        B = 1
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)

        # Shape blend
        v_shaped = self.body_model.v_template + blend_shapes(betas, self.body_model.shapedirs)

        # Joints from shaped mesh
        J = vertices2joints(self.body_model.J_regressor, v_shaped)

        # Pose rotations
        pose = torch.cat([global_orient, body_pose], dim=1)  # (1, J*3)
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(B, -1, 3, 3)  # (1, J, 3, 3)

        # Global joint transforms (no translation yet)
        _, A = batch_rigid_transform(rot_mats, J, self.body_model.parents, dtype=pose.dtype)
        # A: (1, J, 4, 4)

        # Add global translation into the root transform
        transl_mat = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
        transl_mat = transl_mat.expand(1, A.shape[1], -1, -1).clone()
        transl_mat[:, 0, :3, 3] = transl  # apply only to root column

        # Broadcast translation into all joints via root
        # Simpler: just offset by transl after computing world positions
        return A  # (1, J, 4, 4)


def apply_lbs(
    canonical_positions: torch.Tensor,  # (N, 3)
    skinning_weights: torch.Tensor,     # (N, J) sum-to-one
    joint_transforms: torch.Tensor,     # (J, 4, 4)
    transl: torch.Tensor,               # (3,)
) -> torch.Tensor:
    """Apply LBS to canonical positions. Returns (N, 3) world positions."""
    # Weighted blend of joint transforms: T_n = sum_j w_j A_j
    T = torch.einsum('nj,jab->nab', skinning_weights, joint_transforms)  # (N, 4, 4)

    # Homogeneous transform
    ones = torch.ones(canonical_positions.shape[0], 1, device=canonical_positions.device)
    p_hom = torch.cat([canonical_positions, ones], dim=-1)  # (N, 4)
    world_pos = torch.einsum('nab,nb->na', T, p_hom)[:, :3]  # (N, 3)

    return world_pos + transl.unsqueeze(0)


class BodyPoseParams(nn.Module):
    """
    Learnable per-frame SMPL pose parameters and shared body shape.
    Initialised from NlfResult data if provided.
    """

    def __init__(
        self,
        num_frames: int,
        num_body_joints: int = 23,
        num_betas: int = 10,
        device: str = 'cuda',
    ):
        super().__init__()
        self.global_orient = nn.Parameter(torch.zeros(num_frames, 3, device=device))
        self.body_pose = nn.Parameter(torch.zeros(num_frames, num_body_joints * 3, device=device))
        self.betas = nn.Parameter(torch.zeros(num_betas, device=device))
        self.transl = nn.Parameter(torch.zeros(num_frames, 3, device=device))

    def frame(self, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (global_orient, body_pose, betas, transl) for frame t."""
        return (
            self.global_orient[t:t+1],   # (1, 3)
            self.body_pose[t:t+1],        # (1, J*3)
            self.betas,                   # (10,)
            self.transl[t:t+1],           # (1, 3)
        )

    @torch.no_grad()
    def load_from_npz(self, path: str) -> None:
        """Initialise parameters from a depth-aligned NlfResult file (NPZ or HDF5)."""
        import h5py
        if h5py.is_hdf5(path):
            with h5py.File(path, 'r') as f:
                poses     = torch.tensor(f['poses'][:],  dtype=torch.float32)
                betas_arr = torch.tensor(f['betas'][:],  dtype=torch.float32)
                transls   = torch.tensor(f['transls'][:], dtype=torch.float32)
        else:
            data      = np.load(path, allow_pickle=True)
            poses     = torch.tensor(data['poses'],   dtype=torch.float32)
            betas_arr = torch.tensor(data['betas'],   dtype=torch.float32)
            transls   = torch.tensor(data['transls'], dtype=torch.float32)

        T_data = poses.shape[0]
        T_param = self.global_orient.shape[0]
        T = min(T_data, T_param)

        self.global_orient.data[:T] = poses[:T, :3]
        self.body_pose.data[:T] = poses[:T, 3:3 + self.body_pose.shape[1]]
        self.betas.data = betas_arr[:T].mean(0)
        self.transl.data[:T] = transls[:T]


class ObjectPoseParams(nn.Module):
    """
    Learnable per-frame SE(3) transforms for each rigid object.
    Rotation stored in 6D form for smooth gradient flow.
    """

    def __init__(self, num_frames: int, num_objects: int, device: str = 'cuda'):
        super().__init__()
        # 6D rotation: columns 0 and 1 of the rotation matrix
        r6d = torch.zeros(num_frames, num_objects, 6, device=device)
        r6d[:, :, 0] = 1.0  # identity: first col = (1, 0, 0)
        r6d[:, :, 4] = 1.0  # identity: second col = (0, 1, 0)
        self.rotations_6d = nn.Parameter(r6d)
        self.translations = nn.Parameter(torch.zeros(num_frames, num_objects, 3, device=device))

    def get_transform(self, t: int, obj_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (R: 3×3, t: 3) for frame t, object obj_id."""
        r6d = self.rotations_6d[t, obj_id]          # (6,)
        R = rotation_6d_to_matrix(r6d.unsqueeze(0)).squeeze(0)  # (3, 3)
        t_vec = self.translations[t, obj_id]          # (3,)
        return R, t_vec

    def load_from_fp_poses_dir(
        self,
        poses_dir: str,
        obj_id: int,
        frame0_transform_path: str,
    ) -> None:
        """
        Initialise per-frame SE(3) params from FoundationPose per-frame pose JSONs.

        FP saves object-to-camera transforms {frame:06d}.json.  The canonical
        Gaussian positions live in camera-space of frame 0 (already baked via
        frame0_transform_path).  The relative transform from canonical to frame t is:
            R_rel = R_t @ R0^T
            t_rel = t_t − R_t @ R0^T @ t0
        """
        from scipy.spatial.transform import Rotation as SciRot
        from v2d.common.datatypes import Transform3d

        # Load frame-0 reference transform
        t3d0 = Transform3d.load(frame0_transform_path)
        w0, x0, y0, z0 = t3d0.rotation
        R0 = SciRot.from_quat([x0, y0, z0, w0]).as_matrix()   # (3,3)
        t0 = np.array(t3d0.translation, dtype=np.float64)       # (3,)

        loaded = 0
        for json_file in sorted(os.listdir(poses_dir)):
            if not json_file.endswith('.json'):
                continue
            frame_idx = int(os.path.splitext(json_file)[0])
            if frame_idx >= self.rotations_6d.shape[0]:
                continue

            t3d = Transform3d.load(os.path.join(poses_dir, json_file))
            w, x, y, z = t3d.rotation
            Rt = SciRot.from_quat([x, y, z, w]).as_matrix()    # (3,3)
            tt = np.array(t3d.translation, dtype=np.float64)    # (3,)

            R_rel = Rt @ R0.T
            t_rel = tt - R_rel @ t0

            # Encode R_rel as 6D (first two columns)
            r6d = np.concatenate([R_rel[:, 0], R_rel[:, 1]]).astype(np.float32)

            with torch.no_grad():
                self.rotations_6d[frame_idx, obj_id] = torch.from_numpy(r6d).to(self.rotations_6d.device)
                self.translations[frame_idx, obj_id] = torch.from_numpy(t_rel.astype(np.float32)).to(self.translations.device)
            loaded += 1

        print(f"[gsplat] Loaded FP poses for object {obj_id}: {loaded} frames from {poses_dir}")
