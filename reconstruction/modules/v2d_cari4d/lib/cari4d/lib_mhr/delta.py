from __future__ import annotations

from typing import Any, Mapping

from .rotations import is_torch_tensor, rot6d_to_rotmat, rotmat_to_6d
from .schema import MHR_PARAM_DIMS


def _zeros_like_block(reference: Any, dim: int) -> Any:
    if is_torch_tensor(reference):
        import torch

        return torch.zeros((*reference.shape[:-1], dim), device=reference.device, dtype=reference.dtype)
    import numpy as np

    return np.zeros((*reference.shape[:-1], dim), dtype=getattr(reference, "dtype", None) or np.float32)


def _identity_rot6d_like(reference: Any) -> Any:
    if is_torch_tensor(reference):
        import torch

        ident = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=reference.device, dtype=reference.dtype)
        return ident.reshape(*((1,) * (reference.ndim - 1)), 6).expand_as(reference)
    import numpy as np

    ident = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=getattr(reference, "dtype", None) or np.float32)
    return np.broadcast_to(ident, reference.shape).copy()


def rotation_residual_6d(init_rot6d: Any, gt_rot6d: Any) -> Any:
    """Return a 6D residual whose numeric zero is the identity rotation."""

    init_rot = rot6d_to_rotmat(init_rot6d)
    gt_rot = rot6d_to_rotmat(gt_rot6d)
    if is_torch_tensor(init_rot):
        rel = gt_rot @ init_rot.transpose(-1, -2)
    else:
        rel = gt_rot @ init_rot.swapaxes(-1, -2)
    return rotmat_to_6d(rel) - _identity_rot6d_like(init_rot6d)


def compute_mhr_delta(init: Mapping[str, Any], gt: Mapping[str, Any]) -> dict[str, Any]:
    """Compute blockwise MHR residuals from init to target parameters."""

    delta: dict[str, Any] = {}
    delta["delta_mhr_global_rot6d"] = rotation_residual_6d(
        init["mhr_global_rot6d"],
        gt["mhr_global_rot6d"],
    )

    for key in MHR_PARAM_DIMS:
        if key == "mhr_global_rot6d":
            continue
        init_value = init.get(key)
        gt_value = gt.get(key)
        if init_value is None or gt_value is None:
            continue
        delta[f"delta_{key}"] = gt_value - init_value
    return delta


def compose_mhr_delta(init: Mapping[str, Any], delta: Mapping[str, Any]) -> dict[str, Any]:
    """Compose MHR init parameters and residuals into predicted parameters."""

    pred = dict(init)
    if "delta_mhr_global_rot6d" in delta:
        delta_rot6d = delta["delta_mhr_global_rot6d"] + _identity_rot6d_like(init["mhr_global_rot6d"])
        delta_rot = rot6d_to_rotmat(delta_rot6d)
        init_rot = rot6d_to_rotmat(init["mhr_global_rot6d"])
        pred["mhr_global_rot6d"] = rotmat_to_6d(delta_rot @ init_rot)

    for key, dim in MHR_PARAM_DIMS.items():
        if key == "mhr_global_rot6d":
            continue
        delta_key = f"delta_{key}"
        if key in init:
            pred[key] = init[key] + delta.get(delta_key, _zeros_like_block(init[key], dim))
    return pred
