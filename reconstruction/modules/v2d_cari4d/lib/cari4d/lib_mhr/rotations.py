from __future__ import annotations

from typing import Any

import numpy as np


def is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def rot6d_to_rotmat_np(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    orig_shape = value.shape[:-1]
    x = value.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = _normalize_np(a1)
    b2 = _normalize_np(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1).reshape(*orig_shape, 3, 3)


def rotmat_to_6d_np(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value[..., :, :2].reshape(*value.shape[:-2], 6)


def _normalize_np(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, eps)


def rot6d_to_rotmat(value: Any) -> Any:
    if is_torch_tensor(value):
        import torch
        import torch.nn.functional as F

        orig_shape = value.shape[:-1]
        x = value.reshape(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1).reshape(*orig_shape, 3, 3)
    return rot6d_to_rotmat_np(value)


def rotmat_to_6d(value: Any) -> Any:
    if is_torch_tensor(value):
        return value[..., :, :2].reshape(*value.shape[:-2], 6)
    return rotmat_to_6d_np(value)


def rotation_geodesic_distance_radians(rotation_pred: Any, rotation_target: Any) -> Any:
    if not is_torch_tensor(rotation_pred) or not is_torch_tensor(rotation_target):
        raise TypeError("rotation geodesic distance requires torch tensors")
    if rotation_pred.shape != rotation_target.shape or rotation_pred.shape[-2:] != (3, 3):
        raise ValueError(f"rotations must have matching [...,3,3] shapes, got {tuple(rotation_pred.shape)} and {tuple(rotation_target.shape)}")
    import torch

    relative = rotation_pred @ rotation_target.transpose(-1, -2)
    skew_vector = torch.stack((relative[..., 2, 1] - relative[..., 1, 2], relative[..., 0, 2] - relative[..., 2, 0], relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
    sine_twice = torch.linalg.vector_norm(skew_vector, dim=-1)
    cosine_twice = torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1) - 1.0
    return torch.atan2(sine_twice, cosine_twice)
