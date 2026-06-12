# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rotation and motion representation utilities for the planner pipeline."""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import numpy as np
import torch

# -- Quaternion helpers (numpy, wxyz convention) ----------------------------------


def quaternion_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions to 3x3 rotation matrices.

    Args:
        q: (..., 4) array with (w, x, y, z) ordering.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    mat = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1 - (tyy + tzz)
    mat[..., 0, 1] = txy - twz
    mat[..., 0, 2] = txz + twy
    mat[..., 1, 0] = txy + twz
    mat[..., 1, 1] = 1 - (txx + tzz)
    mat[..., 1, 2] = tyz - twx
    mat[..., 2, 0] = txz - twy
    mat[..., 2, 1] = tyz + twx
    mat[..., 2, 2] = 1 - (txx + tyy)
    return mat


def quaternion_to_cont6d_np(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions to 6D continuous rotation representation.

    Takes the first two columns of the rotation matrix.

    Args:
        q: (N, 4) wxyz quaternions.

    Returns:
        (N, 6) continuous 6D representation [col0(3), col1(3)].
    """
    mat = quaternion_to_matrix_np(q)  # (N, 3, 3)
    # 6D = [col0(3), col1(3)] — concatenate first two columns
    col0 = mat[..., :, 0]  # (N, 3)
    col1 = mat[..., :, 1]  # (N, 3)
    return np.concatenate([col0, col1], axis=-1).astype(np.float32)


# -- Torch versions (for model I/O) -----------------------------------------------


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to rotation matrices (torch).

    Args:
        quaternions: (..., 4) tensor with (w, x, y, z).

    Returns:
        (..., 3, 3) rotation matrices.
    """
    w, x, y, z = torch.unbind(quaternions, dim=-1)
    tx, ty, tz = 2 * x, 2 * y, 2 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z

    mat = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))
    return mat


def cont6d_to_matrix(cont6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D continuous representation to rotation matrices.

    Args:
        cont6d: (..., 6) tensor.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    a1 = cont6d[..., :3]
    a2 = cont6d[..., 3:6]

    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to wxyz quaternions.

    Args:
        matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions in (w, x, y, z) order.
    """
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(m.shape[0], 4, device=matrix.device, dtype=matrix.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # 4w
    mask = trace > 0
    q[mask, 0] = 0.25 * s[mask]
    q[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    q[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    q[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]

    # Case 2: m00 is largest diagonal
    mask2 = (~mask) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s2 = (
        torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-10))
        * 2
    )
    q[mask2, 0] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s2[mask2]
    q[mask2, 1] = 0.25 * s2[mask2]
    q[mask2, 2] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s2[mask2]
    q[mask2, 3] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s2[mask2]

    # Case 3: m11 is largest diagonal
    mask3 = (~mask) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    s3 = (
        torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-10))
        * 2
    )
    q[mask3, 0] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s3[mask3]
    q[mask3, 1] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s3[mask3]
    q[mask3, 2] = 0.25 * s3[mask3]
    q[mask3, 3] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s3[mask3]

    # Case 4: m22 is largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = (
        torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-10))
        * 2
    )
    q[mask4, 0] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s4[mask4]
    q[mask4, 1] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s4[mask4]
    q[mask4, 2] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s4[mask4]
    q[mask4, 3] = 0.25 * s4[mask4]

    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
    return q.reshape(batch_shape + (4,))


# -- Quaternion format conversion ---------------------------------------------------


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert (w,x,y,z) to (x,y,z,w)."""
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert (x,y,z,w) to (w,x,y,z)."""
    return q[..., [3, 0, 1, 2]]
