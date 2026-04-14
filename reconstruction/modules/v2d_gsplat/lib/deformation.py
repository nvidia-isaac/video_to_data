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


def _rotation_matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle.
    R: (..., 3, 3) → (..., 3)
    Used only for SMPL compatibility (no skinning-weight path).
    Has known numerical issues near θ = π; prefer the 6D → matrix path
    for gradient-sensitive operations.
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))  # (...)
    skew = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)  # (..., 3)
    axis = skew / (2.0 * theta.sin().clamp(min=1e-7))[..., None]
    aa = axis * theta[..., None]
    near_zero = (theta < 1e-6).unsqueeze(-1).expand_as(aa)
    return torch.where(near_zero, torch.zeros_like(aa), aa)


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
        global_orient: torch.Tensor,                     # (1, 3) axis-angle — ignored if global_orient_R is given
        body_pose: torch.Tensor,                         # (1, J_body*3)
        betas: torch.Tensor,                             # (10,) or (1, 10)
        transl: torch.Tensor,                            # (1, 3)
        global_orient_R: Optional[torch.Tensor] = None, # (1, 3, 3) rotation matrix — preferred, avoids θ=π singularity
    ) -> torch.Tensor:
        """
        Compute per-joint world transform matrices A for full LBS.
        Returns (1, J, 4, 4).

        When global_orient_R is provided it is used directly as the root joint
        rotation matrix, bypassing batch_rodrigues for global_orient.  This is
        important when global_orient is stored in 6D representation — the
        rotation matrix can be derived without going through axis-angle, keeping
        the gradient path free of the θ=π discontinuity.
        """
        from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform

        B = 1
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)

        # Shape blend
        v_shaped = self.body_model.v_template + blend_shapes(betas, self.body_model.shapedirs)

        # Joints from shaped mesh
        J = vertices2joints(self.body_model.J_regressor, v_shaped)

        # Body joint rotation matrices (axis-angle → matrix)
        body_rot_mats = batch_rodrigues(body_pose.view(-1, 3)).view(B, -1, 3, 3)  # (1, J_body, 3, 3)

        if global_orient_R is not None:
            # Use provided rotation matrix directly — no axis-angle conversion,
            # no θ=π singularity in the gradient path.
            go_mat = global_orient_R  # (1, 3, 3)
            if go_mat.ndim == 3:
                go_mat = go_mat.unsqueeze(1)  # (1, 1, 3, 3)
            rot_mats = torch.cat([go_mat, body_rot_mats], dim=1)  # (1, J, 3, 3)
        else:
            go_rot = batch_rodrigues(global_orient.view(-1, 3)).view(B, 1, 3, 3)
            rot_mats = torch.cat([go_rot, body_rot_mats], dim=1)  # (1, J, 3, 3)

        # Global joint transforms
        _, A = batch_rigid_transform(rot_mats, J, self.body_model.parents, dtype=body_pose.dtype)
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

    global_orient is stored as 6D rotation (continuous, no θ=π singularity)
    to prevent the optimizer from flipping the body through a discontinuity.
    body_pose (joint angles) remains in axis-angle since joints rarely reach
    rotations near π and skinning deformation provides a natural barrier.
    """

    def __init__(
        self,
        num_frames: int,
        num_body_joints: int = 23,
        num_betas: int = 10,
        device: str = 'cuda',
    ):
        super().__init__()
        # 6D rotation: first two columns of the rotation matrix.
        # Identity = [[1,0,0],[0,1,0],[0,0,1]] → cols 0 and 1 = [1,0,0,0,1,0]
        r6d = torch.zeros(num_frames, 6, device=device)
        r6d[:, 0] = 1.0  # first column x-component
        r6d[:, 4] = 1.0  # second column y-component
        self.global_orient = nn.Parameter(r6d)          # (T, 6)
        self.body_pose = nn.Parameter(torch.zeros(num_frames, num_body_joints * 3, device=device))
        self.betas = nn.Parameter(torch.zeros(num_betas, device=device))
        self.transl = nn.Parameter(torch.zeros(num_frames, 3, device=device))

    def global_orient_matrix(self, t: slice = slice(None)) -> torch.Tensor:
        """Return (N, 3, 3) rotation matrices for frame slice t."""
        return rotation_6d_to_matrix(self.global_orient[t])  # (N, 3, 3)

    def frame(self, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (global_orient_aa, body_pose, betas, transl) for frame t.
        global_orient_aa is (1, 3) axis-angle converted from 6D for SMPL compatibility.
        Use frame_matrix() for gradient-sensitive paths to avoid θ=π singularity.
        """
        go_R = rotation_6d_to_matrix(self.global_orient[t:t+1])  # (1, 3, 3)
        go_aa = _rotation_matrix_to_axis_angle(go_R)              # (1, 3)
        return (
            go_aa,                    # (1, 3)
            self.body_pose[t:t+1],    # (1, J*3)
            self.betas,               # (10,)
            self.transl[t:t+1],       # (1, 3)
        )

    def frame_matrix(self, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (global_orient_R, body_pose, betas, transl) for frame t.
        global_orient_R is (1, 3, 3) rotation matrix — no singularity in gradient path.
        Pass to SmplDeformer.get_joint_transforms(global_orient_R=...) for clean gradients.
        """
        go_R = rotation_6d_to_matrix(self.global_orient[t:t+1])  # (1, 3, 3)
        return (
            go_R,                     # (1, 3, 3)
            self.body_pose[t:t+1],    # (1, J*3)
            self.betas,               # (10,)
            self.transl[t:t+1],       # (1, 3)
        )

    @torch.no_grad()
    def load_from_npz(self, path: str) -> None:
        """Initialise parameters from a depth-aligned NlfResult file (NPZ or HDF5)."""
        import h5py
        from scipy.spatial.transform import Rotation as SciRot
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

        # Convert axis-angle → rotation matrix → 6D (first two columns)
        aa_np = poses[:T, :3].numpy()  # (T, 3)
        R_np = SciRot.from_rotvec(aa_np).as_matrix()  # (T, 3, 3)
        r6d_np = np.concatenate([R_np[:, :, 0], R_np[:, :, 1]], axis=1).astype(np.float32)  # (T, 6)
        self.global_orient.data[:T] = torch.from_numpy(r6d_np).to(self.global_orient.device)

        self.body_pose.data[:T] = poses[:T, 3:3 + self.body_pose.shape[1]]
        self.betas.data = betas_arr[:T].mean(0)
        self.transl.data[:T] = transls[:T]


class ObjectPoseParams(nn.Module):
    """
    Learnable per-frame SE(3) transforms + global scale for each rigid object.
    Rotation stored in 6D form for smooth gradient flow.
    Scale is time-independent (one scalar per object) stored in log-space for
    positivity; initialized to 0 (scale = 1.0).  Used to correct SAM3D mesh
    scale errors that persist after FoundationPose depth alignment.
    """

    def __init__(self, num_frames: int, num_objects: int, device: str = 'cuda'):
        super().__init__()
        # 6D rotation: columns 0 and 1 of the rotation matrix
        r6d = torch.zeros(num_frames, num_objects, 6, device=device)
        r6d[:, :, 0] = 1.0  # identity: first col = (1, 0, 0)
        r6d[:, :, 4] = 1.0  # identity: second col = (0, 1, 0)
        self.rotations_6d = nn.Parameter(r6d)
        self.translations = nn.Parameter(torch.zeros(num_frames, num_objects, 3, device=device))
        # Per-object global scale correction (log-space; 0 = no correction)
        self.log_scales = nn.Parameter(torch.zeros(num_objects, device=device))

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


class ExposureParams(nn.Module):
    """
    Per-frame learned global exposure correction.

    Applies a scalar multiplier exp(log_exposure[t]) to rendered RGB before
    computing the RGB loss, so Gaussians learn appearance at neutral exposure
    and the camera's auto-exposure variation is absorbed here instead.

    log_exposure is initialised to 0 (neutral, multiplier=1.0).
    L2 regularisation on log_exposure keeps values near 0 and prevents the
    exposure from absorbing real scene colour variation.
    """

    def __init__(self, n_frames: int, device: str = 'cuda'):
        super().__init__()
        self.log_exposure = nn.Parameter(torch.zeros(n_frames, device=device))

    def get(self, frame_t: int) -> torch.Tensor:
        """Return scalar exposure multiplier for frame t (always positive)."""
        return torch.exp(self.log_exposure[frame_t])

    def to(self, device):
        super().to(device)
        return self
