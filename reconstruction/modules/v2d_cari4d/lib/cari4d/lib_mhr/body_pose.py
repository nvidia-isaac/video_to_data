from __future__ import annotations

from typing import Any

import numpy as np


BODY_3DOF_IDXS = np.array([(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)], dtype=np.int64)
BODY_1DOF_ROT_IDXS = np.array([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123], dtype=np.int64)
BODY_1DOF_TRANS_IDXS = np.array([124, 125, 126, 127, 128, 129], dtype=np.int64)
BODY_CONT_ROTATION_DIM = int(len(BODY_3DOF_IDXS) * 6 + len(BODY_1DOF_ROT_IDXS) * 2)
BODY_CONT_INTERNAL_TRANSLATION_DIM = int(len(BODY_1DOF_TRANS_IDXS))
BODY_CONT_INTERNAL_TRANSLATION_SLICE = slice(BODY_CONT_ROTATION_DIM, BODY_CONT_ROTATION_DIM + BODY_CONT_INTERNAL_TRANSLATION_DIM)
HAND_DOFS_IN_ORDER = np.array([3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1], dtype=np.int64)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _batch6d_from_xyz(euler_xyz: np.ndarray) -> np.ndarray:
    rc = np.cos(euler_xyz)
    rs = np.sin(euler_xyz)
    cx, cy, cz = rc[..., 0], rc[..., 1], rc[..., 2]
    sx, sy, sz = rs[..., 0], rs[..., 1], rs[..., 2]
    result = np.empty((*euler_xyz.shape[:-1], 3, 3), dtype=euler_xyz.dtype)
    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cx * sz + sx * sy * cz
    result[..., 0, 2] = sx * sz + cx * sy * cz
    result[..., 1, 0] = cy * sz
    result[..., 1, 1] = cx * cz + sx * sy * sz
    result[..., 1, 2] = -sx * cz + cx * sy * sz
    result[..., 2, 0] = -sy
    result[..., 2, 1] = sx * cy
    result[..., 2, 2] = cx * cy
    return np.concatenate([result[..., :, 0], result[..., :, 1]], axis=-1)


def compact_model_params_to_cont_body_np(body_pose_params: Any) -> np.ndarray:
    body_pose_params = _to_numpy(body_pose_params)
    if body_pose_params.shape[-1] != 133:
        raise ValueError(f"body_pose_params expected final dimension 133, got {body_pose_params.shape}")
    body_params_3dofs = body_pose_params[..., BODY_3DOF_IDXS.reshape(-1)]
    body_params_1dofs = body_pose_params[..., BODY_1DOF_ROT_IDXS]
    body_params_trans = body_pose_params[..., BODY_1DOF_TRANS_IDXS]
    body_cont_3dofs = _batch6d_from_xyz(body_params_3dofs.reshape(*body_params_3dofs.shape[:-1], -1, 3))
    body_cont_1dofs = np.stack([np.sin(body_params_1dofs), np.cos(body_params_1dofs)], axis=-1)
    return np.concatenate([body_cont_3dofs.reshape(*body_cont_3dofs.shape[:-2], -1), body_cont_1dofs.reshape(*body_cont_1dofs.shape[:-2], -1), body_params_trans], axis=-1).astype(np.float32, copy=False)


def compact_model_params_to_cont_hand_np(hand_model_params: Any) -> np.ndarray:
    hand_model_params = _to_numpy(hand_model_params)
    if hand_model_params.shape[-1] != 27:
        raise ValueError(f"hand_model_params expected final dimension 27, got {hand_model_params.shape}")
    if int(HAND_DOFS_IN_ORDER.sum()) != 27:
        raise RuntimeError("MHR hand degree-of-freedom layout must contain 27 model parameters")
    model_3dof_mask = np.repeat(HAND_DOFS_IN_ORDER == 3, HAND_DOFS_IN_ORDER)
    model_1dof_mask = np.repeat(np.isin(HAND_DOFS_IN_ORDER, (1, 2)), HAND_DOFS_IN_ORDER)
    cont_3dof_mask = np.repeat(HAND_DOFS_IN_ORDER == 3, 2 * HAND_DOFS_IN_ORDER)
    cont_1dof_mask = np.repeat(np.isin(HAND_DOFS_IN_ORDER, (1, 2)), 2 * HAND_DOFS_IN_ORDER)
    model_3dof = hand_model_params[..., model_3dof_mask].reshape(*hand_model_params.shape[:-1], -1, 3)
    model_1dof = hand_model_params[..., model_1dof_mask]
    cont = np.zeros((*hand_model_params.shape[:-1], 54), dtype=np.float32)
    cont[..., cont_3dof_mask] = _batch6d_from_xyz(model_3dof).reshape(*hand_model_params.shape[:-1], -1)
    cont[..., cont_1dof_mask] = np.stack([np.sin(model_1dof), np.cos(model_1dof)], axis=-1).reshape(*hand_model_params.shape[:-1], -1)
    return cont
