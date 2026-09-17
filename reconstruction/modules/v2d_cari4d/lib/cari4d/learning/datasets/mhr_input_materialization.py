from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
import torch

from lib_mhr.camera_conventions import MHR_ROOT_JOINT_INDEX
from learning.datasets.mhr_augmentation import MHR_INPUT_AUGMENTATION_KEEP_ALIGNED_DEPTH


MHR_INPUT_MATERIALIZATION_CPU = "cpu_float32_v1"
MHR_INPUT_MATERIALIZATION_GPU = "gpu_compact_v1"
MHR_INPUT_MATERIALIZATION_DEFAULT = MHR_INPUT_MATERIALIZATION_GPU
MHR_INPUT_MATERIALIZATION_MODES = frozenset({MHR_INPUT_MATERIALIZATION_CPU, MHR_INPUT_MATERIALIZATION_GPU})
MHR_INPUT_MATERIALIZATION_GPU_UPGRADE_REVISION = "cari4d.mhr_gpu_compact_keep_unidepth.v1"
MHR_ENCODED_RGBS_KEY = "_mhr_encoded_rgbs"
MHR_ENCODED_DEPTHS_KEY = "_mhr_encoded_depths"
MHR_ENCODED_MASKS_KEY = "_mhr_encoded_masks"
MHR_ENCODED_RENDER_KEYS = (MHR_ENCODED_RGBS_KEY, MHR_ENCODED_DEPTHS_KEY, MHR_ENCODED_MASKS_KEY)
MHR_XYZ_ANCHOR_CONTRACT_SCHEMA = "cari4d.mhr_xyz_anchor_contract.v1"
MHR_XYZ_ANCHOR_BODY_WORLD = "body_world_translation"
MHR_XYZ_ANCHOR_ROOT_JOINT = "root_joint_1"
MHR_XYZ_ANCHOR_DEFAULT = MHR_XYZ_ANCHOR_ROOT_JOINT
MHR_XYZ_ANCHOR_TYPES = frozenset({MHR_XYZ_ANCHOR_BODY_WORLD, MHR_XYZ_ANCHOR_ROOT_JOINT})
MHR_XYZ_ANCHOR_KEY = "mhr_xyz_anchor_init"
MHR_SPATIAL_NORMALIZATION_CONTRACT_SCHEMA = "cari4d.mhr_spatial_normalization_contract.v1"
MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT = "human_height_2m"
MHR_SPATIAL_NORMALIZATION_DEFAULT = MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT
MHR_SPATIAL_NORMALIZATION_TYPES = frozenset({MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT})
MHR_SPATIAL_TARGET_HEIGHT_DEFAULT = 2.0
MHR_SPATIAL_HEIGHT_SOURCE = "neutral_initialized_mhr_vertex_y_extent"
MHR_SPATIAL_SCALE_KEY = "mhr_spatial_scale"
MHR_NEUTRAL_HEIGHT_KEY = "mhr_neutral_height_init"
MHR_SPATIAL_NORMALIZATION_APPLIED_KEY = "_mhr_spatial_normalization_applied"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, MutableMapping):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def resolve_mhr_xyz_anchor_type(cfg: Any, default: str = MHR_XYZ_ANCHOR_DEFAULT) -> str:
    anchor_type = str(_cfg_get(cfg, "mhr_xyz_anchor_type", default))
    if anchor_type not in MHR_XYZ_ANCHOR_TYPES:
        raise ValueError(f"unsupported MHR XYZ anchor type: {anchor_type!r}; expected one of {sorted(MHR_XYZ_ANCHOR_TYPES)}")
    return anchor_type


def build_mhr_xyz_anchor_contract(cfg: Any, default: str = MHR_XYZ_ANCHOR_DEFAULT) -> dict[str, Any]:
    anchor_type = resolve_mhr_xyz_anchor_type(cfg, default)
    return {"schema": MHR_XYZ_ANCHOR_CONTRACT_SCHEMA, "anchorType": anchor_type, "rootJointIndex": MHR_ROOT_JOINT_INDEX if anchor_type == MHR_XYZ_ANCHOR_ROOT_JOINT else None}


def validate_mhr_xyz_anchor_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or contract.get("schema") != MHR_XYZ_ANCHOR_CONTRACT_SCHEMA:
        raise ValueError(f"unsupported MHR XYZ anchor contract schema: {None if not isinstance(contract, Mapping) else contract.get('schema')!r}")
    rebuilt = build_mhr_xyz_anchor_contract({"mhr_xyz_anchor_type": contract.get("anchorType")})
    if dict(contract) != rebuilt:
        raise ValueError(f"MHR XYZ anchor contract is not canonical: expected {rebuilt}, got {dict(contract)}")
    return rebuilt


def restore_mhr_xyz_anchor_contract(checkpoint: Mapping[str, Any], runtime_cfg: Any) -> dict[str, Any]:
    embedded_contract = checkpoint.get("mhr_xyz_anchor_contract")
    checkpoint_cfg = checkpoint.get("cfg")
    if embedded_contract is None and checkpoint_cfg is None:
        raise ValueError("resumed MHR checkpoint has neither an XYZ anchor contract nor stored training configuration")
    config_contract = None
    if checkpoint_cfg is not None:
        config_contract = build_mhr_xyz_anchor_contract(checkpoint_cfg, default=MHR_XYZ_ANCHOR_BODY_WORLD)
    checkpoint_contract = validate_mhr_xyz_anchor_contract(embedded_contract) if embedded_contract is not None else config_contract
    if embedded_contract is not None and checkpoint_cfg is not None and _cfg_get(checkpoint_cfg, "mhr_xyz_anchor_type") is not None and checkpoint_contract != config_contract:
        raise ValueError("resumed checkpoint MHR XYZ anchor contract disagrees with its stored training configuration")
    _cfg_set(runtime_cfg, "mhr_xyz_anchor_type", checkpoint_contract["anchorType"])
    restored = build_mhr_xyz_anchor_contract(runtime_cfg)
    if checkpoint_contract != restored:
        raise ValueError("resumed checkpoint MHR XYZ anchor contract disagrees with the active configuration")
    return restored


def select_mhr_xyz_anchor(mhr_trans: Any, root_joint: Any | None, cfg: Any) -> Any:
    anchor_type = resolve_mhr_xyz_anchor_type(cfg)
    if anchor_type == MHR_XYZ_ANCHOR_ROOT_JOINT and root_joint is None:
        raise KeyError(f"{MHR_XYZ_ANCHOR_ROOT_JOINT} requires decoded MHR joint {MHR_ROOT_JOINT_INDEX}")
    anchor = mhr_trans if anchor_type == MHR_XYZ_ANCHOR_BODY_WORLD else root_joint
    if tuple(anchor.shape) != tuple(mhr_trans.shape):
        raise ValueError(f"MHR XYZ anchor shape {tuple(anchor.shape)} differs from mhr_trans shape {tuple(mhr_trans.shape)}")
    finite = bool(torch.isfinite(anchor).all()) if torch.is_tensor(anchor) else bool(np.isfinite(np.asarray(anchor)).all())
    if not finite:
        raise ValueError("MHR XYZ anchor must be finite")
    return anchor


def resolve_mhr_spatial_normalization_type(cfg: Any, default: str = MHR_SPATIAL_NORMALIZATION_DEFAULT) -> str:
    normalization_type = str(_cfg_get(cfg, "mhr_spatial_normalization_type", default))
    if normalization_type not in MHR_SPATIAL_NORMALIZATION_TYPES:
        raise ValueError(f"unsupported MHR spatial normalization type: {normalization_type!r}; expected one of {sorted(MHR_SPATIAL_NORMALIZATION_TYPES)}")
    return normalization_type


def resolve_mhr_spatial_target_height(cfg: Any) -> float:
    target_height = float(_cfg_get(cfg, "mhr_spatial_target_height", MHR_SPATIAL_TARGET_HEIGHT_DEFAULT))
    if not np.isfinite(target_height) or target_height <= 0:
        raise ValueError(f"MHR spatial target height must be finite and positive, got {target_height}")
    return target_height


def build_mhr_spatial_normalization_contract(cfg: Any) -> dict[str, Any]:
    normalization_type = resolve_mhr_spatial_normalization_type(cfg)
    if normalization_type == MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT and resolve_mhr_xyz_anchor_type(cfg) != MHR_XYZ_ANCHOR_ROOT_JOINT:
        raise ValueError(f"{MHR_SPATIAL_NORMALIZATION_HUMAN_HEIGHT} requires {MHR_XYZ_ANCHOR_ROOT_JOINT} XYZ anchoring")
    return {"schema": MHR_SPATIAL_NORMALIZATION_CONTRACT_SCHEMA, "normalizationType": normalization_type, "targetHumanHeight": resolve_mhr_spatial_target_height(cfg), "heightSource": MHR_SPATIAL_HEIGHT_SOURCE, "appliesTo": ["xyz", "human_translation_residual", "object_translation_residual", "object_pose_condition_translation"]}


def validate_mhr_spatial_normalization_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or contract.get("schema") != MHR_SPATIAL_NORMALIZATION_CONTRACT_SCHEMA:
        raise ValueError(f"unsupported MHR spatial normalization contract schema: {None if not isinstance(contract, Mapping) else contract.get('schema')!r}")
    rebuilt = build_mhr_spatial_normalization_contract({"mhr_spatial_normalization_type": contract.get("normalizationType"), "mhr_spatial_target_height": contract.get("targetHumanHeight"), "mhr_xyz_anchor_type": MHR_XYZ_ANCHOR_ROOT_JOINT})
    if dict(contract) != rebuilt:
        raise ValueError(f"MHR spatial normalization contract is not canonical: expected {rebuilt}, got {dict(contract)}")
    return rebuilt


def restore_mhr_spatial_normalization_contract(checkpoint: Mapping[str, Any], runtime_cfg: Any) -> dict[str, Any]:
    embedded_contract = checkpoint.get("mhr_spatial_normalization_contract")
    if embedded_contract is None:
        raise ValueError("MHR checkpoint predates the human-height spatial normalization contract and cannot be resumed or inferred under the changed objective")
    checkpoint_contract = validate_mhr_spatial_normalization_contract(embedded_contract)
    _cfg_set(runtime_cfg, "mhr_spatial_normalization_type", checkpoint_contract["normalizationType"])
    _cfg_set(runtime_cfg, "mhr_spatial_target_height", checkpoint_contract["targetHumanHeight"])
    restored = build_mhr_spatial_normalization_contract(runtime_cfg)
    if checkpoint_contract != restored:
        raise ValueError("resumed checkpoint MHR spatial normalization contract disagrees with the active configuration")
    return restored


def mhr_spatial_scale_from_height(height: Any, cfg: Any) -> Any:
    target_height = resolve_mhr_spatial_target_height(cfg)
    finite = bool(torch.isfinite(height).all()) if torch.is_tensor(height) else bool(np.isfinite(np.asarray(height)).all())
    positive = bool((height > 0).all()) if torch.is_tensor(height) else bool((np.asarray(height) > 0).all())
    if not finite or not positive:
        raise ValueError("initialized MHR neutral height must be finite and positive")
    return target_height / height


def _broadcast_mhr_spatial_scale(scale: Any, value: Any) -> Any:
    if scale.ndim != 2 or value.ndim < 2 or tuple(scale.shape) != tuple(value.shape[:2]):
        raise ValueError(f"MHR spatial scale {tuple(scale.shape)} cannot broadcast over {tuple(value.shape)}")
    return scale.reshape(*scale.shape, *((1,) * (value.ndim - 2)))


def require_mhr_spatial_scale(batch: Mapping[str, Any], reference: Any | None = None) -> Any:
    scale = batch.get(MHR_SPATIAL_SCALE_KEY)
    if scale is None:
        raise KeyError(f"MHR batch is missing {MHR_SPATIAL_SCALE_KEY}")
    if reference is not None:
        if torch.is_tensor(reference) and not torch.is_tensor(scale):
            scale = torch.as_tensor(scale, device=reference.device, dtype=reference.dtype)
        elif torch.is_tensor(scale):
            scale = scale.to(device=reference.device, dtype=reference.dtype)
        else:
            scale = np.asarray(scale, dtype=np.asarray(reference).dtype)
        _broadcast_mhr_spatial_scale(scale, reference)
    finite = bool(torch.isfinite(scale).all()) if torch.is_tensor(scale) else bool(np.isfinite(np.asarray(scale)).all())
    positive = bool((scale > 0).all()) if torch.is_tensor(scale) else bool((np.asarray(scale) > 0).all())
    if not finite or not positive:
        raise ValueError("MHR spatial scale must be finite and positive")
    return scale


def normalize_mhr_translation(value: Any, batch: Mapping[str, Any]) -> Any:
    scale = require_mhr_spatial_scale(batch, value)
    return value * _broadcast_mhr_spatial_scale(scale, value)


def denormalize_mhr_translation(value: Any, batch: Mapping[str, Any]) -> Any:
    scale = require_mhr_spatial_scale(batch, value)
    return value / _broadcast_mhr_spatial_scale(scale, value)


def prepare_mhr_spatial_batch(batch: MutableMapping[str, Any], cfg: Any, mhr_layer: Any | None) -> MutableMapping[str, Any]:
    build_mhr_spatial_normalization_contract(cfg)
    if batch.get(MHR_SPATIAL_NORMALIZATION_APPLIED_KEY, False):
        require_mhr_spatial_scale(batch, batch.get("mhr_trans_init"))
        return batch
    if MHR_SPATIAL_SCALE_KEY not in batch:
        height = batch.get(MHR_NEUTRAL_HEIGHT_KEY)
        if height is None:
            if mhr_layer is None:
                raise RuntimeError("human-height spatial normalization requires a packed neutral height or an MHR decoder")
            missing = [key for key in ("mhr_shape_init", "mhr_scale_init") if key not in batch]
            if missing:
                raise KeyError(f"human-height spatial normalization requires initialized MHR identity fields: {missing}")
            with torch.no_grad():
                height = mhr_layer.neutral_height({"mhr_shape": batch["mhr_shape_init"], "mhr_scale": batch["mhr_scale_init"]})
            batch[MHR_NEUTRAL_HEIGHT_KEY] = height
        batch[MHR_SPATIAL_SCALE_KEY] = mhr_spatial_scale_from_height(height, cfg)
    scale = require_mhr_spatial_scale(batch, batch.get("mhr_trans_init"))
    for key in ("input_xyz", "render_xyz"):
        if key not in batch:
            continue
        value = batch[key].clone()
        if value.ndim != 5 or value.shape[2] < 3:
            raise ValueError(f"{key} must have shape [B,T,C>=3,H,W], got {tuple(value.shape)}")
        value[:, :, :3] *= scale[:, :, None, None, None].to(device=value.device, dtype=value.dtype)
        batch[key] = value
    if "pose_perturbed" in batch:
        pose_normalized = batch["pose_perturbed"].clone()
        if pose_normalized.ndim != 4 or pose_normalized.shape[-2:] != (4, 4) or tuple(pose_normalized.shape[:2]) != tuple(scale.shape):
            raise ValueError(f"pose_perturbed must have shape [B,T,4,4], got {tuple(pose_normalized.shape)} for scale {tuple(scale.shape)}")
        pose_normalized[:, :, :3, 3] *= scale[:, :, None].to(device=pose_normalized.device, dtype=pose_normalized.dtype)
        batch["poseA_norm"] = pose_normalized
    batch[MHR_SPATIAL_NORMALIZATION_APPLIED_KEY] = True
    return batch


def resolve_mhr_input_materialization_mode(cfg: Any) -> str:
    mode = str(_cfg_get(cfg, "mhr_rank_local_materialization_mode", MHR_INPUT_MATERIALIZATION_DEFAULT))
    if mode not in MHR_INPUT_MATERIALIZATION_MODES:
        raise ValueError(f"unsupported MHR input materialization mode: {mode!r}; expected one of {sorted(MHR_INPUT_MATERIALIZATION_MODES)}")
    return mode


def restore_mhr_input_materialization_mode(checkpoint: Mapping[str, Any], runtime_cfg: Any) -> str:
    checkpoint_cfg = checkpoint.get("cfg")
    if checkpoint_cfg is None:
        raise ValueError("resumed MHR checkpoint has no stored training configuration for input materialization")
    checkpoint_mode = str(_cfg_get(checkpoint_cfg, "mhr_rank_local_materialization_mode", MHR_INPUT_MATERIALIZATION_CPU))
    if checkpoint_mode not in MHR_INPUT_MATERIALIZATION_MODES:
        raise ValueError(f"checkpoint has unsupported MHR input materialization mode: {checkpoint_mode!r}")
    checkpoint_val_mode = _cfg_get(checkpoint_cfg, "val_mhr_rank_local_materialization_mode")
    if checkpoint_val_mode is not None and str(checkpoint_val_mode) not in MHR_INPUT_MATERIALIZATION_MODES:
        raise ValueError(f"checkpoint has unsupported validation MHR input materialization mode: {checkpoint_val_mode!r}")
    checkpoint_revision = _cfg_get(checkpoint_cfg, "mhr_input_materialization_upgrade_revision")
    runtime_revision = _cfg_get(runtime_cfg, "mhr_input_materialization_upgrade_revision")
    if checkpoint_revision is not None and runtime_revision is not None and str(checkpoint_revision) != str(runtime_revision):
        raise ValueError(f"MHR input materialization upgrade revision mismatch: checkpoint={checkpoint_revision!r} runtime={runtime_revision!r}")
    upgrade_revision = runtime_revision if runtime_revision is not None else checkpoint_revision
    if upgrade_revision is None:
        mode = checkpoint_mode
        val_mode = checkpoint_val_mode
    else:
        if str(upgrade_revision) != MHR_INPUT_MATERIALIZATION_GPU_UPGRADE_REVISION:
            raise ValueError(f"unsupported MHR input materialization upgrade revision: {upgrade_revision!r}")
        augmentation_mode = str(_cfg_get(checkpoint_cfg, "mhr_input_augmentation_mode", "disabled_legacy"))
        if augmentation_mode != MHR_INPUT_AUGMENTATION_KEEP_ALIGNED_DEPTH:
            raise ValueError(f"{MHR_INPUT_MATERIALIZATION_GPU_UPGRADE_REVISION} requires checkpoint augmentation {MHR_INPUT_AUGMENTATION_KEEP_ALIGNED_DEPTH!r}, got {augmentation_mode!r}")
        mode = resolve_mhr_input_materialization_mode(runtime_cfg) if runtime_revision is not None else checkpoint_mode
        val_mode = _cfg_get(runtime_cfg, "val_mhr_rank_local_materialization_mode") if runtime_revision is not None else checkpoint_val_mode
        if mode != MHR_INPUT_MATERIALIZATION_GPU or str(val_mode) != MHR_INPUT_MATERIALIZATION_CPU:
            raise ValueError(f"{MHR_INPUT_MATERIALIZATION_GPU_UPGRADE_REVISION} requires GPU training and CPU validation materialization, got train={mode!r} val={val_mode!r}")
    _cfg_set(runtime_cfg, "mhr_rank_local_materialization_mode", mode)
    _cfg_set(runtime_cfg, "val_mhr_rank_local_materialization_mode", None if val_mode is None else str(val_mode))
    _cfg_set(runtime_cfg, "mhr_input_materialization_upgrade_revision", None if upgrade_revision is None else str(upgrade_revision))
    return mode


def validate_mhr_gpu_materialization_config(cfg: Any) -> None:
    expected = {
        "normalize_xyz": True,
        "subtract_transl": True,
        "add_ho_mask": True,
        "mask_encode_type": "hum-obj-fullobj",
        "mask_rgb_bkg": True,
        "crop_xyz_3d": False,
    }
    mismatches = {key: (_cfg_get(cfg, key), value) for key, value in expected.items() if _cfg_get(cfg, key) != value}
    if mismatches:
        raise ValueError(f"gpu_compact_v1 supports only the production MHR input contract; mismatches={mismatches}")
    build_mhr_spatial_normalization_contract(cfg)


def _require_tensor(batch: Mapping[str, Any], key: str, dtype: torch.dtype, ndim: int) -> torch.Tensor:
    value = batch.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"MHR compact batch field {key} must be a tensor, got {type(value).__name__}")
    if value.dtype != dtype or value.ndim != ndim:
        raise ValueError(f"MHR compact batch field {key} must be {dtype} with {ndim} dimensions, got {value.dtype} {tuple(value.shape)}")
    return value


def depth_to_xyzmap_torch(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    if depth.ndim != 4:
        raise ValueError(f"depth must have shape [B,T,H,W], got {tuple(depth.shape)}")
    if intrinsics.shape != depth.shape[:2] + (3, 3):
        raise ValueError(f"intrinsics must have shape [B,T,3,3], got {tuple(intrinsics.shape)} for depth {tuple(depth.shape)}")
    z = depth.float()
    height, width = depth.shape[-2:]
    u = torch.arange(width, dtype=torch.float32, device=depth.device).view(1, 1, 1, width)
    v = torch.arange(height, dtype=torch.float32, device=depth.device).view(1, 1, height, 1)
    fx = intrinsics[..., 0, 0].float().unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[..., 1, 1].float().unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[..., 0, 2].float().unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[..., 1, 2].float().unsqueeze(-1).unsqueeze(-1)
    xyz = torch.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), dim=2)
    xyz.masked_fill_((z < 0.001).unsqueeze(2), 0.0)
    return xyz


def materialize_mhr_gpu_inputs(batch: MutableMapping[str, Any], cfg: Any) -> MutableMapping[str, Any]:
    validate_mhr_gpu_materialization_config(cfg)
    encoded_rgbs = _require_tensor(batch, MHR_ENCODED_RGBS_KEY, torch.uint8, 5)
    encoded_depths = _require_tensor(batch, MHR_ENCODED_DEPTHS_KEY, torch.float16, 5)
    encoded_masks = _require_tensor(batch, MHR_ENCODED_MASKS_KEY, torch.uint8, 5)
    if encoded_rgbs.shape[2] != 6 or encoded_depths.shape[2] != 2 or encoded_masks.shape[2] != 5:
        raise ValueError(f"MHR compact channel contract mismatch: rgbs={tuple(encoded_rgbs.shape)} depths={tuple(encoded_depths.shape)} masks={tuple(encoded_masks.shape)}")
    if encoded_rgbs.shape[:2] + encoded_rgbs.shape[-2:] != encoded_depths.shape[:2] + encoded_depths.shape[-2:] or encoded_rgbs.shape[:2] + encoded_rgbs.shape[-2:] != encoded_masks.shape[:2] + encoded_masks.shape[-2:]:
        raise ValueError(f"MHR compact spatial contract mismatch: rgbs={tuple(encoded_rgbs.shape)} depths={tuple(encoded_depths.shape)} masks={tuple(encoded_masks.shape)}")
    intrinsics = _require_tensor(batch, "K_rois", torch.float32, 4)
    xyz_anchor = _require_tensor(batch, MHR_XYZ_ANCHOR_KEY, torch.float32, 3)
    if xyz_anchor.shape != encoded_rgbs.shape[:2] + (3,):
        raise ValueError(f"MHR compact normalization contract mismatch: anchor={tuple(xyz_anchor.shape)} rgbs={tuple(encoded_rgbs.shape)}")
    input_rgbs = encoded_rgbs[:, :, :3].float().div_(255.0)
    render_rgbs = encoded_rgbs[:, :, 3:].float().div_(255.0)
    input_xyz = depth_to_xyzmap_torch(encoded_depths[:, :, 0], intrinsics)
    render_xyz = depth_to_xyzmap_torch(encoded_depths[:, :, 1], intrinsics)
    masks = encoded_masks.float().div_(255.0)
    foreground = (masks[:, :, 0:1] > 0.5) | (masks[:, :, 1:2] > 0.5)
    input_rgbs.masked_fill_(~foreground.expand_as(input_rgbs), 0.0)
    input_xyz.masked_fill_(~foreground.expand_as(input_xyz), 0.0)
    input_invalid = input_xyz[:, :, 2:3] < 0.01
    render_invalid = render_xyz[:, :, 2:3] < 0.01
    anchor = xyz_anchor.unsqueeze(-1).unsqueeze(-1)
    input_xyz.sub_(anchor)
    render_xyz.sub_(anchor)
    input_xyz.masked_fill_(input_invalid.expand_as(input_xyz), 0.0)
    render_xyz.masked_fill_(render_invalid.expand_as(render_xyz), 0.0)
    batch["input_rgbs"] = input_rgbs
    batch["render_rgbs"] = render_rgbs
    batch["input_xyz"] = torch.cat((input_xyz, masks[:, :, 0:2], masks[:, :, 4:5]), dim=2)
    batch["render_xyz"] = torch.cat((render_xyz, masks[:, :, 2:4], masks[:, :, 4:5]), dim=2)
    for key in MHR_ENCODED_RENDER_KEYS:
        batch.pop(key)
    return batch
