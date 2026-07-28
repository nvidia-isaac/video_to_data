# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame canonicalization for planner inputs and chunk-local re-framing.

The model conditions on transforms expressed in a first-frame, heading-aligned
global frame: positions are relative to the first frame's ground-projected root,
and orientations have the first frame's yaw removed. These helpers build that
representation for body and end-effector inputs, and rotate packed
``pos(3) + rot6d(6)`` transforms between the shared frame and a per-chunk
heading-local frame during autoregressive inference.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from robotic_grounding.planner.motionbricks.rotations import (
    _heading_rotation_from_wxyz,
    _matrix_to_rot6d_first_two_cols,
    _quat_wxyz_to_rotation,
    _rot6d_first_two_cols_to_matrix,
    _xyzw_to_wxyz,
)


def _heading_matrix_from_pose_frame(pose_frame: np.ndarray) -> np.ndarray:
    """Yaw-only chunk heading from the pelvis rot6d of a single pose frame."""
    pose_9d = np.asarray(pose_frame, dtype=np.float64).reshape(-1, 9)
    pelvis_mat = _rot6d_first_two_cols_to_matrix(pose_9d[0, 3:9])
    pelvis_quat_wxyz = _xyzw_to_wxyz(Rotation.from_matrix(pelvis_mat).as_quat())
    return _heading_rotation_from_wxyz(pelvis_quat_wxyz).as_matrix()


def _yaw_from_heading_matrix(heading_mat: np.ndarray) -> float:
    mat = np.asarray(heading_mat, dtype=np.float64)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def _root_xy_shared_to_local(
    root_xy: np.ndarray, origin_xy: np.ndarray, heading_mat: np.ndarray
) -> np.ndarray:
    root_xy = np.asarray(root_xy, dtype=np.float32)
    origin_xy = np.asarray(origin_xy, dtype=np.float32).reshape(2)
    rel = np.zeros((*root_xy.shape[:-1], 3), dtype=np.float64)
    rel[..., :2] = root_xy[..., :2] - origin_xy
    local = np.einsum("ij,...j->...i", np.asarray(heading_mat, dtype=np.float64).T, rel)
    return local[..., :2].astype(np.float32)


def _root_xy_local_to_shared(
    root_xy: np.ndarray, origin_xy: np.ndarray, heading_mat: np.ndarray
) -> np.ndarray:
    root_xy = np.asarray(root_xy, dtype=np.float32)
    origin_xy = np.asarray(origin_xy, dtype=np.float32).reshape(2)
    local = np.zeros((*root_xy.shape[:-1], 3), dtype=np.float64)
    local[..., :2] = root_xy[..., :2]
    shared = np.einsum(
        "ij,...j->...i", np.asarray(heading_mat, dtype=np.float64), local
    )
    shared[..., :2] += origin_xy
    return shared[..., :2].astype(np.float32)


def _packed_transforms_change_frame(
    transforms: np.ndarray,
    heading_mat: np.ndarray,
    *,
    origin_xy: np.ndarray | None = None,
    inverse_heading: bool,
    positions_are_root_relative: bool,
) -> np.ndarray:
    """Rotate packed pos3+rot6d transforms between shared and chunk frames."""
    src = np.asarray(transforms, dtype=np.float32)
    if src.ndim != 2 or src.shape[-1] % 9 != 0:
        raise ValueError(f"Expected [F, bodies*9] transforms, got {src.shape}")
    F = src.shape[0]
    B = src.shape[-1] // 9
    packed = src.reshape(F, B, 9).astype(np.float64, copy=True)
    frame_mat = (
        np.asarray(heading_mat, dtype=np.float64).T
        if inverse_heading
        else np.asarray(heading_mat, dtype=np.float64)
    )
    origin = (
        np.asarray(origin_xy, dtype=np.float64).reshape(2)
        if origin_xy is not None
        else None
    )
    pos = packed[..., :3]
    if origin is not None and inverse_heading and not positions_are_root_relative:
        pos[..., 0] -= origin[0]
        pos[..., 1] -= origin[1]
    pos = np.einsum("ij,fbj->fbi", frame_mat, pos)
    if origin is not None and not inverse_heading and not positions_are_root_relative:
        pos[..., 0] += origin[0]
        pos[..., 1] += origin[1]
    rot_mat = _rot6d_first_two_cols_to_matrix(packed[..., 3:9])
    rot_mat = np.einsum("ij,fbjk->fbik", frame_mat, rot_mat)
    packed[..., :3] = pos
    packed[..., 3:9] = _matrix_to_rot6d_first_two_cols(rot_mat)
    return packed.reshape(src.shape).astype(np.float32)


def _derive_root_relative_ee_from_target(
    root_target_ee: np.ndarray, root_xy: np.ndarray
) -> np.ndarray:
    """Convert non-root-relative EE targets to the root-relative EE condition."""
    F = min(root_target_ee.shape[0], root_xy.shape[0])
    ee = np.asarray(root_target_ee[:F], dtype=np.float32).reshape(F, 2, 9).copy()
    root_pad = np.zeros((F, 1, 3), dtype=np.float32)
    root_pad[:, 0, :2] = np.asarray(root_xy[:F], dtype=np.float32)
    ee[..., :3] -= root_pad
    return ee.reshape(F, 18)


def _canonicalize_heading_transforms(
    body_pos_w: np.ndarray,
    body_wxyz_w: np.ndarray,
    root_pos_w: np.ndarray,
    root_wxyz_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Body world poses -> first-frame heading-aligned global transforms + root XY."""
    first_heading_inv = _heading_rotation_from_wxyz(root_wxyz_w[0]).inv()
    first_root_projected = root_pos_w[0].astype(np.float64).copy()
    first_root_projected[2] = 0.0
    T, B = body_pos_w.shape[:2]
    pos_rel = body_pos_w.astype(np.float64) - first_root_projected.reshape(1, 1, 3)
    pos_heading = first_heading_inv.apply(pos_rel.reshape(-1, 3)).reshape(T, B, 3)
    root_rel = root_pos_w.astype(np.float64) - first_root_projected.reshape(1, 3)
    root_heading = first_heading_inv.apply(root_rel)
    per_frame_root_xy = root_heading.copy()
    per_frame_root_xy[:, 2] = 0.0
    pos_canon = pos_heading - per_frame_root_xy[:, None, :]
    body_rot = first_heading_inv * _quat_wxyz_to_rotation(body_wxyz_w.reshape(-1, 4))
    rot6d = _matrix_to_rot6d_first_two_cols(body_rot.as_matrix()).reshape(T, B, 6)
    transforms = np.concatenate([pos_canon, rot6d], axis=-1).astype(np.float32)
    return transforms.reshape(T, B * 9), root_heading[:, :2].astype(np.float32)


def _canonicalize_ee_targets(
    left_pos_w: np.ndarray,
    left_wxyz_w: np.ndarray,
    right_pos_w: np.ndarray,
    right_wxyz_w: np.ndarray,
    root_pos_w: np.ndarray,
    root_wxyz_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build root EE conditioning, pose EE conditioning, and marker positions."""
    first_heading_inv = _heading_rotation_from_wxyz(root_wxyz_w[0]).inv()
    first_root_projected = root_pos_w[0].astype(np.float64).copy()
    first_root_projected[2] = 0.0
    ee_pos_w = np.stack([left_pos_w, right_pos_w], axis=1).astype(np.float64)
    ee_wxyz_w = np.stack([left_wxyz_w, right_wxyz_w], axis=1).astype(np.float64)
    T = ee_pos_w.shape[0]
    ee_rel = ee_pos_w - first_root_projected.reshape(1, 1, 3)
    ee_heading = first_heading_inv.apply(ee_rel.reshape(-1, 3)).reshape(T, 2, 3)
    root_rel = root_pos_w.astype(np.float64) - first_root_projected.reshape(1, 3)
    root_heading = first_heading_inv.apply(root_rel)
    per_frame_root_xy = root_heading.copy()
    per_frame_root_xy[:, 2] = 0.0
    root_ee_pos = ee_heading
    pose_ee_pos = ee_heading - per_frame_root_xy[:, None, :]
    ee_rot = first_heading_inv * _quat_wxyz_to_rotation(ee_wxyz_w.reshape(-1, 4))
    ee_rot6d = _matrix_to_rot6d_first_two_cols(ee_rot.as_matrix()).reshape(T, 2, 6)
    root_ee_transforms = np.concatenate([root_ee_pos, ee_rot6d], axis=-1).reshape(T, 18)
    pose_ee_transforms = np.concatenate([pose_ee_pos, ee_rot6d], axis=-1).reshape(T, 18)
    ee_marker_pos = root_ee_pos
    return (
        root_ee_transforms.astype(np.float32),
        pose_ee_transforms.astype(np.float32),
        ee_marker_pos.astype(np.float32),
    )
