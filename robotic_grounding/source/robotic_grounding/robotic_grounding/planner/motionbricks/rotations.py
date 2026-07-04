# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rotation conversions shared across the planner (numpy + scipy).

The model uses ``wxyz`` quaternions and 6D rotations (first two matrix columns);
scipy uses ``xyzw``. These helpers bridge the two and are sign-invariant in the
quaternion representation, so results are stable regardless of quaternion sign.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Index maps between wxyz (model) and xyzw (scipy) quaternion order.
_W2X = [1, 2, 3, 0]
_X2W = [3, 0, 1, 2]


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., _W2X]


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[..., _X2W]


def _rot6d_first_two_cols_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation (first two matrix columns) -> rotation matrices."""
    cols = np.asarray(rot6d, dtype=np.float64).reshape(*rot6d.shape[:-1], 3, 2)
    col0 = cols[..., :, 0]
    col1 = cols[..., :, 1]
    col0 = col0 / np.clip(np.linalg.norm(col0, axis=-1, keepdims=True), 1e-8, None)
    col1 = col1 - (col0 * col1).sum(axis=-1, keepdims=True) * col0
    col1 = col1 / np.clip(np.linalg.norm(col1, axis=-1, keepdims=True), 1e-8, None)
    col2 = np.cross(col0, col1, axis=-1)
    return np.stack([col0, col1, col2], axis=-1)


def _matrix_to_rot6d_first_two_cols(mat: np.ndarray) -> np.ndarray:
    """Rotation matrices -> 6D rotation (first two columns, flattened)."""
    return mat[..., :2].reshape(*mat.shape[:-2], 6)


def _heading_rotation_from_wxyz(quat_wxyz: np.ndarray) -> Rotation:
    """Yaw-only heading rotation in a Z-up frame."""
    heading = np.asarray(quat_wxyz, dtype=np.float64).copy()
    heading[..., 1:3] = 0.0
    norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    heading = np.divide(heading, np.clip(norm, 1e-8, None))
    return Rotation.from_quat(_wxyz_to_xyzw(heading))


def _quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(_wxyz_to_xyzw(np.asarray(quat_wxyz, dtype=np.float64)))
