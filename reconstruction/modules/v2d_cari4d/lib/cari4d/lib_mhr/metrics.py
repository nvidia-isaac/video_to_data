from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .rotations import is_torch_tensor, rot6d_to_rotmat


def _point_error(pred: Any, target: Any) -> Any:
    if is_torch_tensor(pred):
        import torch

        return torch.linalg.norm(pred - target, dim=-1).mean()
    return np.linalg.norm(np.asarray(pred) - np.asarray(target), axis=-1).mean()


def _root_rot_error(pred_rot6d: Any, target_rot6d: Any) -> Any:
    pred = rot6d_to_rotmat(pred_rot6d)
    target = rot6d_to_rotmat(target_rot6d)
    if is_torch_tensor(pred):
        import torch

        rel = pred.transpose(-1, -2) @ target
        trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
        cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        return torch.acos(cos).mean()
    rel = pred.swapaxes(-1, -2) @ target
    trace = np.trace(rel, axis1=-1, axis2=-2)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos).mean()


def mhr_metrics(pred: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    """Compute native MHR metrics for matching topology predictions."""

    out: dict[str, Any] = {}
    if "mhr_vertices" in pred and "mhr_vertices" in target:
        out["mhr_v2v"] = _point_error(pred["mhr_vertices"], target["mhr_vertices"])
    if "mhr_joints" in pred and "mhr_joints" in target:
        out["mhr_mpjpe"] = _point_error(pred["mhr_joints"], target["mhr_joints"])
    elif "mhr_keypoints" in pred and "mhr_keypoints" in target:
        out["mhr_keypoint_error"] = _point_error(pred["mhr_keypoints"], target["mhr_keypoints"])
    if "mhr_trans" in pred and "mhr_trans" in target:
        out["mhr_trans_error"] = _point_error(pred["mhr_trans"], target["mhr_trans"])
    if "mhr_global_rot6d" in pred and "mhr_global_rot6d" in target:
        out["mhr_root_rot_error"] = _root_rot_error(pred["mhr_global_rot6d"], target["mhr_global_rot6d"])
    return out

