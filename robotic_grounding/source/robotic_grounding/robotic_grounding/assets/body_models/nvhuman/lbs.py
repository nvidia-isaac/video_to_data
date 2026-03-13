"""Linear Blend Skinning utilities for NVHuman model."""

from typing import Tuple

import torch
import torch.nn.functional as F


def blend_shapes(betas: torch.Tensor, shape_disps: torch.Tensor) -> torch.Tensor:
    """Calculate the per vertex displacement due to the blend shapes.

    Parameters
    ----------
    betas : torch.Tensor
        Blend shape coefficients, shape Bx(num_betas).
    shape_disps : torch.Tensor
        Blend shapes, shape Vx3x(num_betas).

    Returns:
    -------
    torch.Tensor
        The per-vertex displacement due to shape deformation, shape BxVx3.
    """
    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum("bl,lmk->bmk", [betas, shape_disps])
    return blend_shape


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Calculate the 3D joint locations from the vertices.

    Parameters
    ----------
    J_regressor : torch.Tensor
        The regressor array that is used to calculate the joints from the
        position of the vertices, shape JxV.
    vertices : torch.Tensor
        The tensor of mesh vertices, shape BxVx3.

    Returns:
    -------
    torch.Tensor
        The location of the joints, shape BxJx3.
    """
    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: Quaternion tensor, size = [B, 4], format (w, x, y, z).

    Returns:
        Rotation matrix corresponding to the quaternion, size = [B, 3, 3].
    """
    norm_quat = quat
    norm_quat = norm_quat / (norm_quat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def batch_rodrigues(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-8,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Calculate the rotation matrices for a batch of rotation vectors.

    Parameters
    ----------
    rot_vecs : torch.Tensor
        Array of N axis-angle vectors, shape Nx3.
    epsilon : float
        Small value for numerical stability.
    dtype : torch.dtype
        Data type for tensors.

    Returns:
    -------
    torch.Tensor
        The rotation matrices for the given axis-angle parameters, shape Nx3x3.
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(
        (batch_size, 3, 3)
    )

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Create a batch of transformation matrices.

    Args:
        R: Bx3x3 array of a batch of rotation matrices.
        t: Bx3x1 array of a batch of translation vectors.

    Returns:
        Bx4x4 Transformation matrix.
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a batch of rigid transformations to the joints.

    Parameters
    ----------
    rot_mats : torch.Tensor
        Tensor of rotation matrices, shape BxNx3x3.
    joints : torch.Tensor
        Locations of joints (Template Pose), shape BxNx3.
    parents : torch.Tensor
        The kinematic tree of each object, shape BxN.
    dtype : torch.dtype
        The data type of the created tensors, the default is torch.float32.

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - posed_joints: The locations of the joints after applying rotations, BxNx3.
        - rel_transforms: The relative rigid transformations for all joints, BxNx4x4.
        - global_rot_mats: The global rotation matrices for each joint, BxNx3x3.
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # (B, K + 1, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        # (B, 4, 4) x (B, 4, 4)
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    # (B, K + 1, 4, 4)
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # Extract global rotation matrices from the transforms (top-left 3x3)
    global_rot_mats = transforms[:, :, :3, :3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms, global_rot_mats


def lbs(
    betas: torch.Tensor,
    pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
    pose2rot: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform Linear Blend Skinning with the given shape and pose parameters.

    Parameters
    ----------
    betas : torch.Tensor
        The tensor of shape parameters, shape BxNB.
    pose : torch.Tensor
        The pose parameters in axis-angle format, shape Bx(J + 1) * 3.
    v_template : torch.Tensor
        The template mesh that will be deformed, shape BxVx3.
    shapedirs : torch.Tensor
        The tensor of PCA shape displacements, shape 1xNB.
    J_regressor : torch.Tensor
        The regressor array that is used to calculate the joints from
        the position of the vertices, shape JxV.
    parents : torch.Tensor
        The array that describes the kinematic tree for the model, shape J.
    lbs_weights : torch.Tensor
        The linear blend skinning weights that represent how much the
        rotation matrix of each part affects each vertex, shape N x V x (J + 1).
    pose2rot : bool
        Flag on whether to convert the input pose tensor to rotation
        matrices. The default value is True. If False, then the pose tensor
        should already contain rotation matrices and have a size of Bx(J + 1)x9.
    dtype : torch.dtype
        The data type of tensors.

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - verts: The vertices of the mesh after applying deformations, BxVx3.
        - joints: The joints of the model, BxJx3.
        - rot_mats: The rotation matrices of each joint, BxJx3x3.
    """
    batch_size = pose.shape[0]
    device = pose.device

    # Add shape contribution
    if betas is not None:
        v_shaped = v_template + blend_shapes(betas, shapedirs)
    else:
        v_shaped = v_template

    # Get the joints
    # NxJx3 array
    num_joints = J_regressor.shape[0]
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    if pose2rot:
        if pose.numel() == batch_size * num_joints * 4:
            rot_mats = quat_to_rotmat(pose.reshape(batch_size * num_joints, 4)).reshape(
                batch_size, num_joints, 3, 3
            )
        else:
            rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view(
                [batch_size, -1, 3, 3]
            )

    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    if v_shaped.ndim == 2:
        v_posed = v_shaped.unsqueeze(dim=0).expand([batch_size, -1, -1])
    else:
        v_posed = v_shaped

    # 4. Get the global joint location
    J_transformed, A, global_rot_mats = batch_rigid_transform(
        rot_mats, J, parents, dtype=dtype
    )

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones(
        [batch_size, v_posed.shape[1], 1], dtype=dtype, device=device
    )
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, global_rot_mats
