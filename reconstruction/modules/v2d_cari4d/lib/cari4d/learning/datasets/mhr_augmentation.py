from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MHR_INPUT_AUGMENTATION_DISABLED = "disabled_legacy"
MHR_INPUT_AUGMENTATION_SMPLH = "smplh_rgb_depth_v1"
MHR_INPUT_AUGMENTATION_KEEP_ALIGNED_DEPTH = "smplh_rgb_depth_keep50_v2"
MHR_INPUT_AUGMENTATION_DEFAULT = MHR_INPUT_AUGMENTATION_KEEP_ALIGNED_DEPTH
MHR_INPUT_AUGMENTATION_MODES = frozenset({MHR_INPUT_AUGMENTATION_DISABLED, MHR_INPUT_AUGMENTATION_SMPLH, MHR_INPUT_AUGMENTATION_KEEP_ALIGNED_DEPTH})


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, Mapping):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def resolve_mhr_input_augmentation_mode(cfg: Any) -> str:
    mode = str(_cfg_get(cfg, "mhr_input_augmentation_mode", MHR_INPUT_AUGMENTATION_DEFAULT))
    if mode not in MHR_INPUT_AUGMENTATION_MODES:
        raise ValueError(f"unsupported MHR input augmentation mode: {mode!r}; expected one of {sorted(MHR_INPUT_AUGMENTATION_MODES)}")
    return mode


def restore_mhr_input_augmentation_mode(checkpoint: Mapping[str, Any], runtime_cfg: Any) -> str:
    checkpoint_cfg = checkpoint.get("cfg")
    if checkpoint_cfg is None:
        raise ValueError("resumed MHR checkpoint has no stored training configuration for input augmentation")
    mode = str(_cfg_get(checkpoint_cfg, "mhr_input_augmentation_mode", MHR_INPUT_AUGMENTATION_DISABLED))
    if mode not in MHR_INPUT_AUGMENTATION_MODES:
        raise ValueError(f"checkpoint has unsupported MHR input augmentation mode: {mode!r}")
    _cfg_set(runtime_cfg, "mhr_input_augmentation_mode", mode)
    return mode
