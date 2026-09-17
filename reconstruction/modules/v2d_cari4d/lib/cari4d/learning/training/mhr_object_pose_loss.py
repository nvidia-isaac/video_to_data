from __future__ import annotations

from typing import Any, Mapping

import torch

from lib_mhr.rotations import rotation_geodesic_distance_radians
from learning.training.mhr_supervision import canonical_sha256


MHR_OBJECT_POSE_LOSS_CONTRACT_SCHEMA = "cari4d.mhr_object_pose_loss_contract.v1"
MHR_OBJECT_POSE_LOSS_GEODESIC_RADIAN_EQUAL_WEIGHT = "geodesic_radian_vector_mean_equal_weight_v2"
MHR_OBJECT_POSE_LOSS_VECTOR_MEAN_EQUAL_WEIGHT = "vector_mean_equal_weight_v1"
MHR_OBJECT_POSE_LOSS_LEGACY = "legacy_symmetry_translation_sum_config_weights_v1"
MHR_OBJECT_POSE_LOSS_MODES = frozenset({MHR_OBJECT_POSE_LOSS_GEODESIC_RADIAN_EQUAL_WEIGHT, MHR_OBJECT_POSE_LOSS_VECTOR_MEAN_EQUAL_WEIGHT, MHR_OBJECT_POSE_LOSS_LEGACY})
MHR_OBJECT_POSE_LOSS_EQUAL_WEIGHT_MODES = frozenset({MHR_OBJECT_POSE_LOSS_GEODESIC_RADIAN_EQUAL_WEIGHT, MHR_OBJECT_POSE_LOSS_VECTOR_MEAN_EQUAL_WEIGHT})


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, Mapping):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def build_mhr_object_pose_loss_contract(cfg: Any, default_mode: str = MHR_OBJECT_POSE_LOSS_GEODESIC_RADIAN_EQUAL_WEIGHT) -> dict[str, Any]:
    mode = str(_cfg_get(cfg, "mhr_object_pose_loss_mode", default_mode))
    if mode not in MHR_OBJECT_POSE_LOSS_MODES:
        raise ValueError(f"unsupported MHR object-pose loss mode: {mode}")
    rotation_weight = float(_cfg_get(cfg, "w_rot"))
    translation_weight = float(_cfg_get(cfg, "w_transl"))
    if mode in MHR_OBJECT_POSE_LOSS_EQUAL_WEIGHT_MODES and (rotation_weight != 1.0 or translation_weight != 1.0):
        raise ValueError(f"{mode} requires w_rot=1.0 and w_transl=1.0, got {rotation_weight} and {translation_weight}")
    contract = {"schema": MHR_OBJECT_POSE_LOSS_CONTRACT_SCHEMA, "mode": mode, "rotationWeight": rotation_weight, "translationWeight": translation_weight}
    contract["contractSha256"] = canonical_sha256(contract)
    return contract


def validate_mhr_object_pose_loss_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or contract.get("schema") != MHR_OBJECT_POSE_LOSS_CONTRACT_SCHEMA:
        raise ValueError(f"unsupported MHR object-pose loss contract schema: {None if not isinstance(contract, Mapping) else contract.get('schema')}")
    normalized = dict(contract)
    contract_hash = normalized.pop("contractSha256", None)
    if contract_hash != canonical_sha256(normalized):
        raise ValueError("MHR object-pose loss contract hash mismatch")
    rebuilt = build_mhr_object_pose_loss_contract({"mhr_object_pose_loss_mode": normalized.get("mode"), "w_rot": normalized.get("rotationWeight"), "w_transl": normalized.get("translationWeight")})
    if rebuilt != dict(contract):
        raise ValueError("MHR object-pose loss contract does not match its configuration")
    return rebuilt


def restore_mhr_object_pose_loss_contract(checkpoint: Mapping[str, Any], runtime_cfg: Any) -> dict[str, Any]:
    embedded_contract = checkpoint.get("mhr_object_pose_loss_contract")
    checkpoint_cfg = checkpoint.get("cfg")
    if embedded_contract is None and checkpoint_cfg is None:
        raise ValueError("resumed MHR checkpoint has neither an object-pose loss contract nor stored training configuration")
    config_contract = None if checkpoint_cfg is None else build_mhr_object_pose_loss_contract(checkpoint_cfg, default_mode=MHR_OBJECT_POSE_LOSS_LEGACY)
    checkpoint_contract = validate_mhr_object_pose_loss_contract(embedded_contract) if embedded_contract is not None else config_contract
    if config_contract is not None and checkpoint_contract != config_contract:
        raise ValueError("resumed checkpoint MHR object-pose loss contract disagrees with its stored training configuration")
    _cfg_set(runtime_cfg, "mhr_object_pose_loss_mode", checkpoint_contract["mode"])
    _cfg_set(runtime_cfg, "w_rot", checkpoint_contract["rotationWeight"])
    _cfg_set(runtime_cfg, "w_transl", checkpoint_contract["translationWeight"])
    restored = build_mhr_object_pose_loss_contract(runtime_cfg)
    if checkpoint_contract != restored:
        raise ValueError("resumed checkpoint MHR object-pose loss contract disagrees with the active training configuration")
    return restored


def reduce_mhr_object_translation_loss(loss_values: Any, object_pose_loss_mode: str) -> Any:
    if loss_values.shape[-1] != 3:
        raise ValueError(f"object translation loss must end in xyz, got {tuple(loss_values.shape)}")
    if object_pose_loss_mode in MHR_OBJECT_POSE_LOSS_EQUAL_WEIGHT_MODES:
        return loss_values.mean(-1)
    if object_pose_loss_mode == MHR_OBJECT_POSE_LOSS_LEGACY:
        return loss_values.sum(-1)
    raise ValueError(f"unsupported MHR object-pose loss mode: {object_pose_loss_mode}")


def reduce_mhr_object_rotation_loss(rotation_pred: torch.Tensor, rotation_target: torch.Tensor, loss_func: Any, object_pose_loss_mode: str) -> torch.Tensor:
    if object_pose_loss_mode == MHR_OBJECT_POSE_LOSS_GEODESIC_RADIAN_EQUAL_WEIGHT:
        return rotation_geodesic_distance_radians(rotation_pred, rotation_target)
    if object_pose_loss_mode in {MHR_OBJECT_POSE_LOSS_VECTOR_MEAN_EQUAL_WEIGHT, MHR_OBJECT_POSE_LOSS_LEGACY}:
        pred_rot6d = rotation_pred[..., :3, 0:2].reshape(*rotation_pred.shape[:-2], 6)
        target_rot6d = rotation_target[..., :3, 0:2].reshape(*rotation_target.shape[:-2], 6)
        return loss_func(pred_rot6d, target_rot6d, reduction="none").mean(-1)
    raise ValueError(f"unsupported MHR object-pose loss mode: {object_pose_loss_mode}")
