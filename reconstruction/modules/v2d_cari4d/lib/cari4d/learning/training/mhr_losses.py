from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from lib_mhr import compose_mhr_delta, mhr70_to_coco17
from lib_mhr.mhr_layer import MHRLayerOutput
from lib_mhr.rotations import is_torch_tensor
from lib_mhr.schema import MHR_PARAM_DIMS
from learning.datasets.mhr_input_materialization import denormalize_mhr_translation, normalize_mhr_translation
from learning.training.cuda_event_timing import cuda_timing_section


MHR_LOSS_BLOCKS = (
    ("delta_mhr_global_rot6d", "w_mhr_root_rot"),
    ("delta_mhr_trans", "w_mhr_trans"),
    ("delta_mhr_body_pose_cont", "w_mhr_body_pose"),
    ("delta_mhr_hand", "w_mhr_hand"),
    ("delta_mhr_shape", "w_mhr_shape"),
    ("delta_mhr_scale", "w_mhr_scale"),
    ("delta_mhr_face", "w_mhr_face"),
)
MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY = "body12_pose_only"
MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES = "body12_pose_only_all_losses"
MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES = "body12_freeze_hand_face_all_losses"
MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS = "legacy_coco17_all_blocks"
MHR_JOINT_SUPERVISION_MODES = frozenset({MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY, MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES, MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES, MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS})
MHR_BODY12_COCO17_INDICES = tuple(range(5, 17))
MHR_JOINT_POSE_DELTA_KEYS = ("delta_mhr_global_rot6d", "delta_mhr_trans", "delta_mhr_body_pose_cont")
MHR_JOINT_DETACHED_PARAM_KEYS = ("mhr_hand", "mhr_shape", "mhr_scale", "mhr_face")
MHR_ALL_LOSS_FROZEN_DELTA_KEYS = ("delta_mhr_hand", "delta_mhr_shape", "delta_mhr_scale", "delta_mhr_face")
MHR_HAND_FACE_FROZEN_DELTA_KEYS = ("delta_mhr_hand", "delta_mhr_face")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_mhr_joint_supervision_mode(cfg: Any) -> str:
    mode = _cfg_get(cfg, "mhr_joint_supervision_mode", MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS)
    if mode not in MHR_JOINT_SUPERVISION_MODES:
        raise ValueError(f"unsupported MHR joint supervision mode: {mode}")
    return str(mode)


def mhr_frozen_delta_keys_for_mode(mode: str) -> tuple[str, ...]:
    if mode == MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES:
        return MHR_ALL_LOSS_FROZEN_DELTA_KEYS
    if mode == MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES:
        return MHR_HAND_FACE_FROZEN_DELTA_KEYS
    if mode not in MHR_JOINT_SUPERVISION_MODES:
        raise ValueError(f"unsupported MHR joint supervision mode: {mode}")
    return ()


def mhr_all_loss_frozen_delta_keys(cfg: Any) -> tuple[str, ...]:
    return mhr_frozen_delta_keys_for_mode(resolve_mhr_joint_supervision_mode(cfg))


def mhr_training_metrics_for_logging(loss_dict: Mapping[str, Any], cfg: Any) -> dict[str, Any]:
    metrics = dict(loss_dict)
    for loss_name, weight_name, metric_name in (("loss_mhr_v2v", "w_mhr_v2v", "mhr_v2v_m"), ("loss_mhr_joints", "w_mhr_joints", "mhr_mpjpe_m")):
        if loss_name not in metrics:
            continue
        weight = float(_cfg_get(cfg, weight_name, 0.0))
        if weight == 0:
            raise ValueError(f"{loss_name} cannot be logged without a nonzero {weight_name}")
        unweighted = metrics.pop(loss_name) / weight
        metrics.setdefault(metric_name, unweighted)
    return metrics


def _zero_like(value: Any) -> Any:
    if is_torch_tensor(value):
        return value.sum() * 0
    return 0.0


def _l1(pred: Any, target: Any) -> Any:
    if is_torch_tensor(pred):
        import torch

        if not is_torch_tensor(target):
            target = torch.as_tensor(target, device=pred.device, dtype=pred.dtype)
        else:
            target = target.to(device=pred.device, dtype=pred.dtype)
        return (pred - target).abs().mean()
    return float(np.abs(np.asarray(pred) - np.asarray(target)).mean())


def _masked_l1(pred: Any, target: Any, frame_mask: Any | None) -> Any:
    if frame_mask is None:
        return _l1(pred, target)
    if is_torch_tensor(pred):
        import torch

        if not is_torch_tensor(target):
            target = torch.as_tensor(target, device=pred.device, dtype=pred.dtype)
        else:
            target = target.to(device=pred.device, dtype=pred.dtype)
        mask = frame_mask if is_torch_tensor(frame_mask) else torch.as_tensor(frame_mask, device=pred.device)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        errors = (pred - target).abs()
        if errors.ndim < mask.ndim:
            raise ValueError(f"parameter error shape {tuple(errors.shape)} is incompatible with frame_mask {tuple(mask.shape)}")
        reduction_dims = tuple(range(mask.ndim, errors.ndim))
        per_frame = errors.mean(dim=reduction_dims) if reduction_dims else errors
        if tuple(per_frame.shape) != tuple(mask.shape):
            raise ValueError(f"parameter error frame shape {tuple(per_frame.shape)} does not match frame_mask {tuple(mask.shape)}")
        valid_count = mask.sum()
        return (per_frame * mask).sum() / valid_count
    target_array = np.asarray(target)
    mask = np.asarray(frame_mask, dtype=np.float32)
    errors = np.abs(np.asarray(pred) - target_array)
    if errors.ndim < mask.ndim:
        raise ValueError(f"parameter error shape {errors.shape} is incompatible with frame_mask {mask.shape}")
    reduction_dims = tuple(range(mask.ndim, errors.ndim))
    per_frame = errors.mean(axis=reduction_dims) if reduction_dims else errors
    if per_frame.shape != mask.shape:
        raise ValueError(f"parameter error frame shape {per_frame.shape} does not match frame_mask {mask.shape}")
    valid_count = float(mask.sum())
    if valid_count <= 0:
        raise ValueError("MHR parameter loss requires at least one valid frame")
    return float((per_frame * mask).sum() / valid_count)


def _point_l2(pred: Any, target: Any) -> Any:
    if is_torch_tensor(pred):
        import torch

        if not is_torch_tensor(target):
            target = torch.as_tensor(target, device=pred.device, dtype=pred.dtype)
        else:
            target = target.to(device=pred.device, dtype=pred.dtype)
        return torch.linalg.norm(pred - target, dim=-1).mean()
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(target), axis=-1).mean())


def _masked_point_l2(pred: Any, target: Any, frame_mask: Any | None) -> Any:
    if frame_mask is None:
        return _point_l2(pred, target)
    if is_torch_tensor(pred):
        import torch

        if not is_torch_tensor(target):
            target = torch.as_tensor(target, device=pred.device, dtype=pred.dtype)
        else:
            target = target.to(device=pred.device, dtype=pred.dtype)
        mask = frame_mask if is_torch_tensor(frame_mask) else torch.as_tensor(frame_mask, device=pred.device)
        mask = mask.to(device=pred.device, dtype=pred.dtype)
        errors = torch.linalg.norm(pred - target, dim=-1)
        if errors.ndim < mask.ndim:
            raise ValueError(f"point error shape {tuple(errors.shape)} is incompatible with frame_mask {tuple(mask.shape)}")
        reduction_dims = tuple(range(mask.ndim, errors.ndim))
        per_frame = errors.mean(dim=reduction_dims) if reduction_dims else errors
        if tuple(per_frame.shape) != tuple(mask.shape):
            raise ValueError(f"point error frame shape {tuple(per_frame.shape)} does not match frame_mask {tuple(mask.shape)}")
        valid_count = mask.sum()
        return (per_frame * mask).sum() / valid_count
    target_array = np.asarray(target)
    mask = np.asarray(frame_mask, dtype=np.float32)
    errors = np.linalg.norm(np.asarray(pred) - target_array, axis=-1)
    if errors.ndim < mask.ndim:
        raise ValueError(f"point error shape {errors.shape} is incompatible with frame_mask {mask.shape}")
    reduction_dims = tuple(range(mask.ndim, errors.ndim))
    per_frame = errors.mean(axis=reduction_dims) if reduction_dims else errors
    if per_frame.shape != mask.shape:
        raise ValueError(f"point error frame shape {per_frame.shape} does not match frame_mask {mask.shape}")
    valid_count = float(mask.sum())
    if valid_count <= 0:
        raise ValueError("MHR geometry loss requires at least one valid frame")
    return float((per_frame * mask).sum() / valid_count)


def _first_tensor(mapping: Mapping[str, Any]) -> Any:
    for value in mapping.values():
        if hasattr(value, "shape"):
            return value
    raise ValueError("no tensor-like values found")


def _init_params_from_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    params = {}
    for key in MHR_PARAM_DIMS:
        init_key = f"{key}_init"
        if init_key in batch:
            params[key] = batch[init_key]
    return params


def gt_mhr_params_from_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    params = {}
    for key in MHR_PARAM_DIMS:
        gt_key = f"{key}_gt"
        if gt_key in batch:
            params[key] = batch[gt_key]
    return params


def _pred_delta_from_output(output: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
    delta = {}
    for key in MHR_LOSS_BLOCKS:
        delta_key = key[0]
        if delta_key in output:
            delta[delta_key] = output[delta_key]
        elif delta_key in batch:
            target = batch[delta_key]
            delta[delta_key] = target * 0 if is_torch_tensor(target) else np.zeros_like(target)
    return delta


def compose_mhr_output(output: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
    init = _init_params_from_batch(batch)
    delta = _pred_delta_from_output(output, batch)
    if "delta_mhr_trans" in delta:
        delta["delta_mhr_trans"] = denormalize_mhr_translation(delta["delta_mhr_trans"], batch)
    return compose_mhr_delta(init, delta)


def _require_coco17(layer_output: Any, label: str) -> Any:
    if isinstance(layer_output, MHRLayerOutput):
        coco17 = layer_output.coco17
    elif isinstance(layer_output, Mapping):
        coco17 = layer_output.get("mhr_coco17")
        if coco17 is None:
            coco17 = layer_output.get("coco17")
    else:
        coco17 = getattr(layer_output, "coco17", None)
    if coco17 is None:
        raise RuntimeError(f"MHR layer output for {label} did not contain COCO17 keypoints")
    return coco17


def _require_vertices(layer_output: Any, label: str) -> Any:
    if isinstance(layer_output, MHRLayerOutput):
        vertices = layer_output.vertices
    elif isinstance(layer_output, Mapping):
        vertices = layer_output.get("mhr_vertices")
        if vertices is None:
            vertices = layer_output.get("vertices")
    else:
        vertices = getattr(layer_output, "vertices", None)
    if vertices is None:
        raise RuntimeError(f"MHR layer output for {label} did not contain vertices")
    return vertices


def _mhr_forward(mhr_layer: Any, params: Mapping[str, Any]) -> Any:
    forward = getattr(mhr_layer, "mhr_forward", None)
    if callable(forward):
        return forward(params)
    return mhr_layer(params)


def _cached_gt_coco17_from_batch(batch: Mapping[str, Any]) -> Any | None:
    if "mhr_coco17_gt" in batch:
        return batch["mhr_coco17_gt"]
    if "mhr_keypoints_gt" in batch:
        return mhr70_to_coco17(batch["mhr_keypoints_gt"])
    return None


def _joint_supervision_params(pred_params: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS:
        return dict(pred_params)
    if mode not in {MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY, MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES, MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES}:
        raise ValueError(f"unsupported MHR joint supervision mode: {mode}")
    params = dict(pred_params)
    detached_keys = ("mhr_hand", "mhr_face") if mode == MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES else MHR_JOINT_DETACHED_PARAM_KEYS
    for key in detached_keys:
        if key in params and is_torch_tensor(params[key]):
            params[key] = params[key].detach()
    return params


def _joint_supervision_points(coco17: Any, mode: str) -> Any:
    if mode == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS:
        return coco17
    if mode not in {MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY, MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES, MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES}:
        raise ValueError(f"unsupported MHR joint supervision mode: {mode}")
    return coco17[..., list(MHR_BODY12_COCO17_INDICES), :]


def _zero_frozen_mhr_deltas(output: Mapping[str, Any], frozen_delta_keys: Sequence[str]) -> dict[str, Any]:
    filtered = dict(output)
    for key in frozen_delta_keys:
        if key not in filtered:
            continue
        value = filtered[key]
        if is_torch_tensor(value):
            import torch

            filtered[key] = torch.zeros_like(value, requires_grad=False)
        else:
            filtered[key] = np.zeros_like(value)
    if frozen_delta_keys:
        filtered.pop("mhr_vertices", None)
    return filtered


def _mhr_shape_loss(pred_params: Mapping[str, Any], batch: Mapping[str, Any], cfg: Any, frame_mask: Any | None) -> Any | None:
    weight = float(_cfg_get(cfg, "w_mhr_shape", 0.0))
    if weight == 0 or "mhr_shape" not in pred_params or "mhr_shape_gt" not in batch:
        return None
    return _masked_l1(pred_params["mhr_shape"], batch["mhr_shape_gt"], frame_mask) * weight


def _detached_visualization_vertices(vertices: Any, batch_index: int, frame_indices: Sequence[int]) -> Any:
    indices = [int(index) for index in frame_indices]
    if not indices:
        raise ValueError("MHR visualization requires at least one frame index")
    if len(vertices.shape) != 4 or vertices.shape[-1] != 3:
        raise ValueError(f"MHR visualization vertices must have shape [B,T,V,3], got {tuple(vertices.shape)}")
    if batch_index < 0 or batch_index >= vertices.shape[0] or min(indices) < 0 or max(indices) >= vertices.shape[1]:
        raise IndexError(f"MHR visualization selection batch={batch_index}, frames={indices} exceeds vertices {tuple(vertices.shape)}")
    selected = vertices[batch_index, indices]
    if is_torch_tensor(selected):
        return selected.detach().clone()
    return np.asarray(selected).copy()


def _selected_mhr_params(pred_params: Mapping[str, Any], batch_index: int, frame_indices: Sequence[int]) -> dict[str, Any]:
    indices = [int(index) for index in frame_indices]
    selected = {}
    for key, value in pred_params.items():
        if len(value.shape) < 3:
            raise ValueError(f"MHR parameter {key} must have shape [B,T,D], got {tuple(value.shape)}")
        selected[key] = value[batch_index:batch_index + 1, indices]
    return selected


def _visualization_vertices(output: Mapping[str, Any], pred_params: Mapping[str, Any], pred_body: Any | None, mhr_layer: Any | None, batch_index: int, frame_indices: Sequence[int]) -> Any:
    if pred_body is not None:
        return _detached_visualization_vertices(_require_vertices(pred_body, "prediction visualization"), batch_index, frame_indices)
    if "mhr_vertices" in output:
        return _detached_visualization_vertices(output["mhr_vertices"], batch_index, frame_indices)
    if mhr_layer is None:
        raise RuntimeError("MHR prediction visualization requires a differentiable MHRLayer")
    selected_params = _selected_mhr_params(pred_params, batch_index, frame_indices)
    if any(is_torch_tensor(value) for value in selected_params.values()):
        import torch

        with torch.no_grad():
            selected_body = _mhr_forward(mhr_layer, selected_params)
    else:
        selected_body = _mhr_forward(mhr_layer, selected_params)
    return _detached_visualization_vertices(_require_vertices(selected_body, "prediction visualization"), 0, range(len(frame_indices)))


def _compute_mhr_training_loss(output: Mapping[str, Any], batch: Mapping[str, Any], cfg: Any, mhr_layer: Any | None = None, include_geometry_metrics: bool = False, cuda_event_timer: Any | None = None, visualization_batch_index: int | None = None, visualization_frame_indices: Sequence[int] | None = None) -> tuple[Any, dict[str, Any], Any | None]:
    """Compute MHR-native parameter and geometry losses."""

    loss_dict: dict[str, Any] = {}
    first = _first_tensor(batch)
    total = _zero_like(first)
    frame_mask = batch.get("human_frame_mask", batch.get("frame_mask"))
    mode = resolve_mhr_joint_supervision_mode(cfg)
    frozen_delta_keys = frozenset(mhr_all_loss_frozen_delta_keys(cfg))
    effective_output = _zero_frozen_mhr_deltas(output, frozen_delta_keys)

    for delta_key, weight_key in MHR_LOSS_BLOCKS:
        if delta_key == "delta_mhr_shape" or delta_key in frozen_delta_keys:
            continue
        if delta_key not in output or delta_key not in batch:
            continue
        weight = float(_cfg_get(cfg, weight_key, 0.0))
        if weight == 0:
            continue
        target = normalize_mhr_translation(batch[delta_key], batch) if delta_key == "delta_mhr_trans" else batch[delta_key]
        loss_block = _masked_l1(output[delta_key], target, frame_mask) * weight
        loss_dict[f"loss_{delta_key}"] = loss_block
        total = total + loss_block

    pred_params = compose_mhr_output(effective_output, batch)
    loss_shape = None if "delta_mhr_shape" in frozen_delta_keys else _mhr_shape_loss(pred_params, batch, cfg, frame_mask)
    if loss_shape is not None:
        loss_dict["loss_mhr_shape"] = loss_shape
        loss_dict["loss_delta_mhr_shape"] = loss_shape
        total = total + loss_shape

    pred_body = None
    weight_v2v = float(_cfg_get(cfg, "w_mhr_v2v", 0.0))
    need_v2v = weight_v2v != 0 or include_geometry_metrics
    if need_v2v:
        if mhr_layer is None:
            raise RuntimeError("MHR vertex evaluation requires a differentiable MHRLayer")
        if "mhr_vertices" in effective_output:
            pred_vertices = effective_output["mhr_vertices"]
        else:
            with cuda_timing_section(cuda_event_timer, "mhr_decode"):
                pred_body = _mhr_forward(mhr_layer, pred_params)
            pred_vertices = _require_vertices(pred_body, "prediction")
        gt_params = gt_mhr_params_from_batch(batch)
        missing_gt = sorted(set(MHR_PARAM_DIMS) - set(gt_params))
        if missing_gt:
            raise RuntimeError(f"MHR vertex evaluation requires complete GT parameters; missing {missing_gt}")
        with cuda_timing_section(cuda_event_timer, "mhr_decode_gt"):
            gt_vertices = _require_vertices(_mhr_forward(mhr_layer, gt_params), "ground truth")
        v2v_m = _masked_point_l2(pred_vertices, gt_vertices, frame_mask)
        if include_geometry_metrics:
            loss_dict["mhr_v2v_m"] = v2v_m
        if weight_v2v:
            loss_v = v2v_m * weight_v2v
            loss_dict["loss_mhr_v2v"] = loss_v
            total = total + loss_v

    weight_joints = float(_cfg_get(cfg, "w_mhr_joints", 0.0))
    gt_coco17 = _cached_gt_coco17_from_batch(batch)
    if weight_joints and gt_coco17 is None:
        raise RuntimeError("w_mhr_joints requires cached mhr_coco17_gt or mhr_keypoints_gt in the batch")
    need_joints = gt_coco17 is not None and (weight_joints != 0 or include_geometry_metrics)
    if need_joints:
        if mhr_layer is None:
            if weight_joints:
                raise RuntimeError("w_mhr_joints requires a differentiable MHRLayer")
            raise RuntimeError("MHR joint evaluation requires a differentiable MHRLayer")
        if pred_body is None and (include_geometry_metrics or mode == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS):
            with cuda_timing_section(cuda_event_timer, "mhr_decode"):
                pred_body = _mhr_forward(mhr_layer, pred_params)
        if include_geometry_metrics:
            mpjpe_m = _masked_point_l2(_require_coco17(pred_body, "prediction"), gt_coco17, frame_mask)
            loss_dict["mhr_mpjpe_m"] = mpjpe_m
        if weight_joints:
            if mode in {MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS, MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES, MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES} and pred_body is not None:
                joint_body = pred_body
            else:
                with cuda_timing_section(cuda_event_timer, "mhr_decode"):
                    joint_body = _mhr_forward(mhr_layer, _joint_supervision_params(pred_params, mode))
                if pred_body is None:
                    pred_body = joint_body
            joint_error_m = _masked_point_l2(_joint_supervision_points(_require_coco17(joint_body, "joint supervision"), mode), _joint_supervision_points(gt_coco17, mode), frame_mask)
            loss_j = joint_error_m * weight_joints
            loss_dict["loss_mhr_joints"] = loss_j
            loss_dict["loss_mhr_coco17" if mode == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS else "loss_mhr_body12"] = loss_j
            total = total + loss_j

    if not loss_dict:
        raise ValueError("No MHR losses were active; check MHR output keys and loss weights")

    visualization_vertices = None
    if (visualization_batch_index is None) != (visualization_frame_indices is None):
        raise ValueError("visualization_batch_index and visualization_frame_indices must be provided together")
    if visualization_frame_indices is not None:
        visualization_vertices = _visualization_vertices(effective_output, pred_params, pred_body, mhr_layer, int(visualization_batch_index), visualization_frame_indices)
    loss_dict["loss_mhr_total"] = total
    return total, loss_dict, visualization_vertices


def compute_mhr_training_loss(output: Mapping[str, Any], batch: Mapping[str, Any], cfg: Any, mhr_layer: Any | None = None, include_geometry_metrics: bool = False, cuda_event_timer: Any | None = None) -> tuple[Any, dict[str, Any]]:
    total, loss_dict, _ = _compute_mhr_training_loss(output, batch, cfg, mhr_layer=mhr_layer, include_geometry_metrics=include_geometry_metrics, cuda_event_timer=cuda_event_timer)
    return total, loss_dict


def compute_mhr_training_loss_with_geometry(output: Mapping[str, Any], batch: Mapping[str, Any], cfg: Any, *, mhr_layer: Any, visualization_batch_index: int, visualization_frame_indices: Sequence[int], include_geometry_metrics: bool = False, cuda_event_timer: Any | None = None) -> tuple[Any, dict[str, Any], Any]:
    total, loss_dict, visualization_vertices = _compute_mhr_training_loss(output, batch, cfg, mhr_layer=mhr_layer, include_geometry_metrics=include_geometry_metrics, cuda_event_timer=cuda_event_timer, visualization_batch_index=visualization_batch_index, visualization_frame_indices=visualization_frame_indices)
    if visualization_vertices is None:
        raise RuntimeError("MHR prediction visualization did not produce selected vertices")
    return total, loss_dict, visualization_vertices
