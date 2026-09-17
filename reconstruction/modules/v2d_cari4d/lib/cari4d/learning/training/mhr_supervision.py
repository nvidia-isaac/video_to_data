from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from learning.datasets.mhr_augmentation import MHR_INPUT_AUGMENTATION_DISABLED
from learning.training.mhr_losses import MHR_JOINT_POSE_DELTA_KEYS, MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES, MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY, MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES, MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS, MHR_LOSS_BLOCKS, mhr_frozen_delta_keys_for_mode, resolve_mhr_joint_supervision_mode
from lib_mhr.rotations import is_torch_tensor


MHR_SUPERVISION_CONTRACT_SCHEMA = "cari4d.mhr_supervision_contract.v2"
MHR_SUPERVISION_CONTRACT_LEGACY_SCHEMA = "cari4d.mhr_supervision_contract.v1"
MHR_INFERENCE_PROVENANCE_SCHEMA = "cari4d.mhr_inference_provenance.v1"
MHR_ALL_DELTA_KEYS = tuple(delta_key for delta_key, _ in MHR_LOSS_BLOCKS)
MHR_GEOMETRY_WEIGHT_KEYS = ("w_mhr_v2v", "w_mhr_joints")
CHECKPOINT_STEP_RE = re.compile(r"^step(\d+)\.pth$")
WANDB_LAUNCH_METADATA_KEYS = frozenset({"launch_command", "submit_command", "slurm_job_id", "submit_script"})


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            plain = OmegaConf.to_container(value, resolve=True)
            if not isinstance(plain, Mapping):
                raise TypeError(f"resolved OmegaConf must be a mapping, got {type(plain).__name__}")
            return dict(plain)
    except ImportError:
        pass
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        plain = as_dict()
        if not isinstance(plain, Mapping):
            raise TypeError(f"configuration as_dict() must return a mapping, got {type(plain).__name__}")
        return dict(plain)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"configuration must be mapping-like, got {type(value).__name__}")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"__cari4d_nonfinite_float__": "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_wandb_training_config(value: Any) -> dict[str, Any]:
    return {key: item for key, item in _plain_mapping(value).items() if key not in WANDB_LAUNCH_METADATA_KEYS}


def _normalize_wandb_special_float_strings(value: Any, checkpoint_value: Any) -> Any:
    if isinstance(value, Mapping) and isinstance(checkpoint_value, Mapping):
        return {key: _normalize_wandb_special_float_strings(item, checkpoint_value[key]) if key in checkpoint_value else item for key, item in value.items()}
    if isinstance(value, list) and isinstance(checkpoint_value, (list, tuple)) and len(value) == len(checkpoint_value):
        return [_normalize_wandb_special_float_strings(item, checkpoint_item) for item, checkpoint_item in zip(value, checkpoint_value)]
    if isinstance(value, str) and isinstance(checkpoint_value, (float, np.floating)) and not math.isfinite(float(checkpoint_value)):
        expected = "NaN" if math.isnan(float(checkpoint_value)) else ("Infinity" if float(checkpoint_value) > 0 else "-Infinity")
        return checkpoint_value if value == expected else value
    return value


def _wandb_config_hash_matches_checkpoint(wandb_config: Mapping[str, Any], checkpoint_hash: str, contract: Mapping[str, Any]) -> bool:
    if checkpoint_hash == canonical_sha256(wandb_config):
        return True
    compatible = dict(wandb_config)
    if contract["jointSupervisionMode"] == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS and compatible.get("mhr_joint_supervision_mode") == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS:
        del compatible["mhr_joint_supervision_mode"]
    if compatible.get("mhr_input_augmentation_mode") == MHR_INPUT_AUGMENTATION_DISABLED:
        del compatible["mhr_input_augmentation_mode"]
    return checkpoint_hash == canonical_sha256(compatible)


def mhr_supervision_weights(cfg: Any) -> dict[str, float]:
    keys = [weight_key for _, weight_key in MHR_LOSS_BLOCKS] + list(MHR_GEOMETRY_WEIGHT_KEYS)
    weights = {}
    for key in keys:
        value = float(_cfg_get(cfg, key, 0.0))
        if not np.isfinite(value):
            raise ValueError(f"MHR supervision weight {key} must be finite, got {value}")
        weights[key] = value
    return weights


def _geometry_loss_blocks(mode: str) -> dict[str, tuple[str, ...]]:
    if mode not in {MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY, MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES, MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES, MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS}:
        raise ValueError(f"unsupported MHR joint supervision mode: {mode}")
    if mode == MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES:
        active_blocks = tuple(key for key in MHR_ALL_DELTA_KEYS if key not in mhr_frozen_delta_keys_for_mode(mode))
        return {"w_mhr_v2v": active_blocks, "w_mhr_joints": active_blocks}
    v2v_blocks = MHR_JOINT_POSE_DELTA_KEYS if mode == MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES else MHR_ALL_DELTA_KEYS
    joint_blocks = MHR_ALL_DELTA_KEYS if mode == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS else MHR_JOINT_POSE_DELTA_KEYS
    return {"w_mhr_v2v": v2v_blocks, "w_mhr_joints": joint_blocks}


def _build_contract(weights: Mapping[str, float], mode: str, schema: str) -> dict[str, Any]:
    geometry_loss_blocks = _geometry_loss_blocks(mode)
    globally_frozen = frozenset(mhr_frozen_delta_keys_for_mode(mode))
    blocks = {}
    for delta_key, direct_weight_key in MHR_LOSS_BLOCKS:
        reasons = []
        if delta_key not in globally_frozen and weights[direct_weight_key] != 0:
            reasons.append(f"direct:{direct_weight_key}={weights[direct_weight_key]:g}")
        for geometry_weight_key, affected_blocks in geometry_loss_blocks.items():
            if weights[geometry_weight_key] != 0 and delta_key in affected_blocks:
                if geometry_weight_key == "w_mhr_v2v":
                    geometry_name = "decoded MHR vertices with pose-only gradients" if mode == MHR_JOINT_SUPERVISION_BODY12_POSE_ONLY_ALL_LOSSES else ("decoded MHR vertices with hand/face frozen" if mode == MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES else "all decoded MHR vertices")
                else:
                    geometry_name = "decoded MHR COCO17 keypoints" if mode == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS else ("decoded MHR body-12 keypoints with hand/face frozen" if mode == MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES else "decoded MHR body-12 keypoints with pose-only gradients")
                reasons.append(f"indirect:{geometry_weight_key}={weights[geometry_weight_key]:g} through {geometry_name}")
        blocks[delta_key] = {"supervised": bool(reasons), "reasons": reasons}
    supervised = [key for key, value in blocks.items() if value["supervised"]]
    frozen = [key for key, value in blocks.items() if not value["supervised"]]
    contract = {
        "schema": schema,
        "weights": dict(weights),
        "blocks": blocks,
        "supervisedDeltaKeys": supervised,
        "frozenDeltaKeys": frozen,
    }
    if schema == MHR_SUPERVISION_CONTRACT_SCHEMA:
        contract["jointSupervisionMode"] = mode
        contract["configSha256"] = canonical_sha256({"weights": dict(weights), "jointSupervisionMode": mode})
    else:
        contract["configSha256"] = canonical_sha256(weights)
    contract["contractSha256"] = canonical_sha256(contract)
    return contract


def build_mhr_supervision_contract(cfg: Any) -> dict[str, Any]:
    return _build_contract(mhr_supervision_weights(cfg), resolve_mhr_joint_supervision_mode(cfg), MHR_SUPERVISION_CONTRACT_SCHEMA)


def _build_legacy_mhr_supervision_contract(cfg: Any) -> dict[str, Any]:
    return _build_contract(mhr_supervision_weights(cfg), MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS, MHR_SUPERVISION_CONTRACT_LEGACY_SCHEMA)


def validate_mhr_supervision_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or contract.get("schema") not in {MHR_SUPERVISION_CONTRACT_SCHEMA, MHR_SUPERVISION_CONTRACT_LEGACY_SCHEMA}:
        raise ValueError(f"unsupported MHR supervision contract schema: {None if not isinstance(contract, Mapping) else contract.get('schema')}")
    normalized = dict(contract)
    contract_hash = normalized.pop("contractSha256", None)
    if contract_hash != canonical_sha256(normalized):
        raise ValueError("MHR supervision contract hash mismatch")
    legacy = contract.get("schema") == MHR_SUPERVISION_CONTRACT_LEGACY_SCHEMA
    rebuilt = _build_legacy_mhr_supervision_contract(normalized.get("weights", {})) if legacy else build_mhr_supervision_contract({**normalized.get("weights", {}), "mhr_joint_supervision_mode": normalized.get("jointSupervisionMode")})
    if rebuilt != dict(contract):
        raise ValueError("MHR supervision contract does not match its configured loss graph")
    return build_mhr_supervision_contract({**normalized.get("weights", {}), "mhr_joint_supervision_mode": MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS}) if legacy else rebuilt


def restore_mhr_training_supervision_contract(checkpoint: Mapping[str, Any], runtime_cfg: Any) -> dict[str, Any]:
    embedded_contract = checkpoint.get("mhr_supervision_contract")
    checkpoint_cfg = checkpoint.get("cfg")
    if embedded_contract is None and checkpoint_cfg is None:
        raise ValueError("resumed MHR checkpoint has neither a supervision contract nor stored training configuration")
    checkpoint_contract = validate_mhr_supervision_contract(embedded_contract) if embedded_contract is not None else build_mhr_supervision_contract(checkpoint_cfg)
    if checkpoint_contract["weights"] != mhr_supervision_weights(runtime_cfg):
        raise ValueError("resumed checkpoint MHR supervision weights disagree with the active training configuration")
    if isinstance(runtime_cfg, Mapping):
        runtime_cfg["mhr_joint_supervision_mode"] = checkpoint_contract["jointSupervisionMode"]
    else:
        setattr(runtime_cfg, "mhr_joint_supervision_mode", checkpoint_contract["jointSupervisionMode"])
    restored = build_mhr_supervision_contract(runtime_cfg)
    if checkpoint_contract != restored:
        raise ValueError("resumed checkpoint MHR supervision contract disagrees with the active training configuration")
    return restored


def apply_mhr_supervision_contract(output: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_mhr_supervision_contract(contract)
    filtered = dict(output)
    for key in contract["frozenDeltaKeys"]:
        if key not in filtered:
            continue
        value = filtered[key]
        if is_torch_tensor(value):
            import torch

            filtered[key] = torch.zeros_like(value)
        else:
            filtered[key] = np.zeros_like(value)
    return filtered


def configure_mhr_training_heads(model: Any, contract: Mapping[str, Any]) -> dict[str, list[str]]:
    contract = validate_mhr_supervision_contract(contract)
    globally_frozen = frozenset(mhr_frozen_delta_keys_for_mode(contract["jointSupervisionMode"]))
    frozen_delta_keys = frozenset(contract["frozenDeltaKeys"] if globally_frozen else ())
    head_names = getattr(model, "mhr_head_names", None)
    if globally_frozen and head_names is None:
        raise AttributeError("globally frozen MHR supervision requires model.mhr_head_names")
    frozen_heads = []
    trainable_heads = []
    for output_key, module_name in head_names or ():
        module = getattr(model, module_name, None)
        if module is None:
            raise AttributeError(f"MHR prediction head {output_key} references missing module {module_name}")
        trainable = output_key not in frozen_delta_keys
        module.requires_grad_(trainable)
        (trainable_heads if trainable else frozen_heads).append(output_key)
    return {"frozenHeadKeys": frozen_heads, "trainableHeadKeys": trainable_heads}


def _checkpoint_step(checkpoint: Mapping[str, Any], checkpoint_path: str | Path) -> int:
    value = checkpoint.get("step", checkpoint.get("best_step", checkpoint.get("source_checkpoint_step")))
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"checkpoint has no valid training step: {checkpoint_path}")
    step = int(value)
    match = CHECKPOINT_STEP_RE.match(Path(checkpoint_path).name)
    if match is not None and int(match.group(1)) != step:
        raise ValueError(f"checkpoint payload step {step} does not match filename {checkpoint_path}")
    return step


def _identity_value(source: Any, key: str) -> str | None:
    value = _cfg_get(source, key)
    return None if value in (None, "") else str(value)


def _wandb_identity_from_run(run: Any) -> dict[str, str]:
    identity = {"entity": str(run.entity), "project": str(run.project), "runId": str(run.id)}
    if any(not value for value in identity.values()):
        raise ValueError(f"W&B run returned an incomplete identity: {identity}")
    return identity


def _run_path(identity: Mapping[str, Any]) -> str:
    return f"{identity['entity']}/{identity['project']}/{identity['runId']}"


def _resolve_wandb_identity(checkpoint: Mapping[str, Any], runtime_cfg: Any, wandb_run_path: str | None, wandb_api: Any) -> dict[str, str]:
    checkpoint_cfg = checkpoint.get("cfg")
    embedded = checkpoint.get("wandb_identity") or {}
    if wandb_run_path:
        parts = str(wandb_run_path).strip("/").split("/")
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError("W&B run path must be entity/project/run_id")
        explicit = {"entity": parts[0], "project": parts[1], "runId": parts[2]}
    else:
        entity = _identity_value(embedded, "entity") or _identity_value(checkpoint_cfg, "wandb_entity") or _identity_value(runtime_cfg, "wandb_entity") or os.environ.get("WANDB_ENTITY")
        default_entity = getattr(wandb_api, "default_entity", None)
        if callable(default_entity):
            default_entity = default_entity()
        entity = entity or (None if default_entity in (None, "") else str(default_entity))
        project = _identity_value(embedded, "project") or _identity_value(checkpoint_cfg, "wandb_project") or _identity_value(runtime_cfg, "wandb_project")
        run_id = _identity_value(embedded, "runId") or _identity_value(checkpoint_cfg, "run_id") or _identity_value(runtime_cfg, "run_id")
        if not entity or not project or not run_id:
            raise ValueError("legacy MHR checkpoint requires an identifiable W&B entity, project, and run ID")
        explicit = {"entity": entity, "project": project, "runId": run_id}
    for key, cfg_key in (("entity", "wandb_entity"), ("project", "wandb_project"), ("runId", "run_id")):
        for source_name, source in (("checkpoint", checkpoint_cfg), ("runtime", runtime_cfg)):
            source_value = _identity_value(source, cfg_key)
            if source_value is not None and source_value != explicit[key]:
                raise ValueError(f"{source_name} {cfg_key}={source_value} disagrees with W&B identity {explicit[key]}")
        embedded_value = _identity_value(embedded, key)
        if embedded_value is not None and embedded_value != explicit[key]:
            raise ValueError(f"checkpoint W&B identity {key}={embedded_value} disagrees with requested run {explicit[key]}")
    return explicit


def resolve_mhr_inference_provenance(checkpoint: Mapping[str, Any], checkpoint_path: str | Path, runtime_cfg: Any = None, *, wandb_run_path: str | None = None, wandb_api: Any = None, offline: bool = False) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("MHR checkpoint must be a mapping")
    checkpoint_path = str(Path(checkpoint_path).resolve())
    checkpoint_step = _checkpoint_step(checkpoint, checkpoint_path)
    embedded_contract = checkpoint.get("mhr_supervision_contract")
    if offline:
        if embedded_contract is None:
            raise ValueError("offline MHR inference requires a checkpoint-embedded supervision contract")
        contract = validate_mhr_supervision_contract(embedded_contract)
        identity = checkpoint.get("wandb_identity")
        if not isinstance(identity, Mapping) or any(_identity_value(identity, key) is None for key in ("entity", "project", "runId")):
            raise ValueError("offline MHR inference requires checkpoint-embedded W&B identity")
        wandb_config_sha256 = checkpoint.get("wandb_config_sha256")
        if not isinstance(wandb_config_sha256, str) or not wandb_config_sha256:
            raise ValueError("offline MHR inference requires a checkpoint-embedded W&B configuration hash")
        return {"schema": MHR_INFERENCE_PROVENANCE_SCHEMA, "checkpoint": {"path": checkpoint_path, "step": checkpoint_step}, "wandbRun": dict(identity), "wandbConfigSha256": wandb_config_sha256, "supervisionContract": contract, "configSource": "checkpoint_embedded"}
    if wandb_api is None:
        import wandb

        wandb_api = wandb.Api()
    requested_identity = _resolve_wandb_identity(checkpoint, runtime_cfg, wandb_run_path, wandb_api)
    run = wandb_api.run(_run_path(requested_identity))
    observed_identity = _wandb_identity_from_run(run)
    if observed_identity != requested_identity:
        raise ValueError(f"W&B returned {observed_identity}, expected {requested_identity}")
    checkpoint_cfg = checkpoint.get("cfg")
    if checkpoint_cfg is None:
        raise ValueError("legacy MHR checkpoint has no stored training configuration")
    wandb_config = _normalize_wandb_special_float_strings(stable_wandb_training_config(run.config), _plain_mapping(checkpoint_cfg))
    contract = build_mhr_supervision_contract(wandb_config)
    checkpoint_contract = build_mhr_supervision_contract(checkpoint_cfg)
    if checkpoint_contract["weights"] != contract["weights"]:
        raise ValueError("checkpoint MHR supervision weights disagree with the source W&B run")
    if checkpoint_contract["jointSupervisionMode"] != contract["jointSupervisionMode"]:
        raise ValueError("checkpoint MHR joint supervision mode disagrees with the source W&B run")
    if runtime_cfg is not None and mhr_supervision_weights(runtime_cfg) != contract["weights"]:
        raise ValueError("runtime MHR supervision weights disagree with the source W&B run")
    if embedded_contract is not None and validate_mhr_supervision_contract(embedded_contract) != contract:
        raise ValueError("checkpoint-embedded MHR supervision contract disagrees with the source W&B run")
    wandb_config_sha256 = canonical_sha256(wandb_config)
    embedded_config_hash = checkpoint.get("wandb_config_sha256")
    if embedded_config_hash not in (None, "") and not _wandb_config_hash_matches_checkpoint(wandb_config, str(embedded_config_hash), contract):
        raise ValueError("checkpoint W&B configuration hash disagrees with the source W&B run")
    return {"schema": MHR_INFERENCE_PROVENANCE_SCHEMA, "checkpoint": {"path": checkpoint_path, "step": checkpoint_step}, "wandbRun": observed_identity, "wandbConfigSha256": wandb_config_sha256, "supervisionContract": contract, "configSource": "wandb"}


def training_wandb_identity(run: Any | None) -> dict[str, str] | None:
    return None if run is None else _wandb_identity_from_run(run)


def training_wandb_config_sha256(run: Any | None) -> str | None:
    return None if run is None else canonical_sha256(stable_wandb_training_config(run.config))
