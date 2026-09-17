from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from prep.mhr_preprocess_manifest import contiguous_frame_starts


MHR_WINDOW_SAMPLING_CONTRACT_SCHEMA = "cari4d.mhr_window_sampling_contract.v1"
MHR_WINDOW_SAMPLING_LEGACY = "contiguous_unit_stride_v1"
MHR_WINDOW_SAMPLING_STRIDE_DEDUP = "strides_1_10_max_sampled_overlap_v1"
MHR_WINDOW_SAMPLING_DEFAULT_STRIDES = tuple(range(1, 11))
MHR_WINDOW_SAMPLING_DEFAULT_MAX_OVERLAP = 10
MHR_CANONICAL_OBJECT_NONEMPTY_FRAME_FRACTION_DEFAULT = 0.10
MHR_CANONICAL_OBJECT_NONEMPTY_WINDOW_REVISION = "all-sampled-cameras-minimum-canonical-object-nonempty-fraction-v1"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_set(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, Mapping):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _validated_strides(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = [item for item in value.replace("[", "").replace("]", "").split(",") if item.strip()]
    strides = tuple(int(item) for item in value)
    if not strides or any(stride <= 0 for stride in strides) or tuple(sorted(set(strides))) != strides:
        raise ValueError(f"mhr_window_temporal_strides must be unique, increasing positive integers, got {strides}")
    return strides


def build_mhr_window_sampling_contract(cfg: Any, split: str = "train", *, force_legacy: bool = False) -> dict[str, Any]:
    clip_len = int(_cfg_get(cfg, "clip_len", 96))
    if clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")
    mode = MHR_WINDOW_SAMPLING_LEGACY if force_legacy or split != "train" else str(_cfg_get(cfg, "mhr_window_sampling_mode", MHR_WINDOW_SAMPLING_LEGACY))
    if mode == MHR_WINDOW_SAMPLING_LEGACY:
        window = int(_cfg_get(cfg, "window", 1))
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        return {"schema": MHR_WINDOW_SAMPLING_CONTRACT_SCHEMA, "mode": mode, "clipLength": clip_len, "temporalStrides": [1], "maximumSampledFrameOverlap": None, "windowStartStep": window, "includeTerminal": True}
    if mode != MHR_WINDOW_SAMPLING_STRIDE_DEDUP:
        raise ValueError(f"Unsupported MHR window sampling mode {mode!r}")
    strides = _validated_strides(_cfg_get(cfg, "mhr_window_temporal_strides", MHR_WINDOW_SAMPLING_DEFAULT_STRIDES))
    max_overlap = int(_cfg_get(cfg, "mhr_window_max_overlap_frames", MHR_WINDOW_SAMPLING_DEFAULT_MAX_OVERLAP))
    if not 0 <= max_overlap < clip_len:
        raise ValueError(f"mhr_window_max_overlap_frames must be in [0, {clip_len}), got {max_overlap}")
    return {"schema": MHR_WINDOW_SAMPLING_CONTRACT_SCHEMA, "mode": mode, "clipLength": clip_len, "temporalStrides": list(strides), "maximumSampledFrameOverlap": max_overlap, "windowStartStep": None, "includeTerminal": False}


def validate_mhr_window_sampling_contract(contract: Any) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or contract.get("schema") != MHR_WINDOW_SAMPLING_CONTRACT_SCHEMA:
        raise ValueError(f"Invalid MHR window sampling contract schema: {None if not isinstance(contract, Mapping) else contract.get('schema')!r}")
    cfg = {"clip_len": contract.get("clipLength"), "window": contract.get("windowStartStep"), "mhr_window_sampling_mode": contract.get("mode"), "mhr_window_temporal_strides": contract.get("temporalStrides"), "mhr_window_max_overlap_frames": contract.get("maximumSampledFrameOverlap")}
    normalized = build_mhr_window_sampling_contract(cfg, force_legacy=contract.get("mode") == MHR_WINDOW_SAMPLING_LEGACY)
    if dict(contract) != normalized:
        raise ValueError(f"MHR window sampling contract is not canonical: expected {normalized}, got {dict(contract)}")
    return normalized


def apply_mhr_window_sampling_contract(cfg: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    contract = validate_mhr_window_sampling_contract(contract)
    configured_clip_len = int(_cfg_get(cfg, "clip_len", contract["clipLength"]))
    if configured_clip_len != contract["clipLength"]:
        raise ValueError(f"MHR window sampling contract clip length {contract['clipLength']} differs from configured clip_len={configured_clip_len}")
    _cfg_set(cfg, "mhr_window_sampling_mode", contract["mode"])
    _cfg_set(cfg, "mhr_window_temporal_strides", list(contract["temporalStrides"]))
    if contract["maximumSampledFrameOverlap"] is not None:
        _cfg_set(cfg, "mhr_window_max_overlap_frames", contract["maximumSampledFrameOverlap"])
    if contract["mode"] == MHR_WINDOW_SAMPLING_LEGACY:
        _cfg_set(cfg, "window", int(contract["windowStartStep"]))
    return contract


def load_mhr_window_sampling_contract(path: str | Path) -> dict[str, Any]:
    return validate_mhr_window_sampling_contract(json.loads(Path(path).read_text()))


def apply_mhr_window_sampling_contract_path(cfg: Any) -> dict[str, Any] | None:
    path = _cfg_get(cfg, "mhr_window_sampling_contract_path")
    return None if not path else apply_mhr_window_sampling_contract(cfg, load_mhr_window_sampling_contract(path))


def restore_mhr_window_sampling_contract(checkpoint: Mapping[str, Any], cfg: Any) -> dict[str, Any]:
    checkpoint_contract = checkpoint.get("mhr_window_sampling_contract")
    contract = build_mhr_window_sampling_contract(cfg, force_legacy=True) if checkpoint_contract is None else validate_mhr_window_sampling_contract(checkpoint_contract)
    path = _cfg_get(cfg, "mhr_window_sampling_contract_path")
    if path:
        sidecar_contract = load_mhr_window_sampling_contract(path)
        if sidecar_contract != contract:
            raise ValueError(f"Checkpoint and run-sidecar MHR window sampling contracts differ: checkpoint={contract} sidecar={sidecar_contract}")
    return apply_mhr_window_sampling_contract(cfg, contract)


def resolve_run_mhr_window_sampling_contract(cfg: Any, experiment_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.is_file():
        return apply_mhr_window_sampling_contract(cfg, load_mhr_window_sampling_contract(output_path))
    experiment_dir = Path(experiment_dir)
    has_historical_checkpoint = any(experiment_dir.glob("step*.pth"))
    contract = build_mhr_window_sampling_contract(cfg, force_legacy=has_historical_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, output_path)
    return apply_mhr_window_sampling_contract(cfg, contract)


def _contiguous_chunks(frames: Sequence[str]) -> list[tuple[int, int]]:
    values = [Path(str(frame)).name for frame in frames]
    if not values:
        return []
    if not all(value.isdigit() for value in values):
        return [(0, len(values))]
    numeric = [int(value) for value in values]
    chunks = []
    start = 0
    for index in range(1, len(numeric) + 1):
        if index == len(numeric) or numeric[index] != numeric[index - 1] + 1:
            chunks.append((start, index))
            start = index
    return chunks


def _valid_window_starts(valid_mask: np.ndarray, chunk_start: int, chunk_end: int, clip_len: int, stride: int, min_valid_count: int) -> np.ndarray:
    chunk = valid_mask[chunk_start:chunk_end].astype(np.int64, copy=False)
    starts = []
    for offset in range(stride):
        series = chunk[offset::stride]
        if len(series) < clip_len:
            continue
        cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(series, dtype=np.int64)))
        window_sums = cumulative[clip_len:] - cumulative[:-clip_len]
        valid_indices = np.flatnonzero(window_sums >= min_valid_count)
        if valid_indices.size:
            starts.append(chunk_start + offset + valid_indices * stride)
    return np.empty(0, dtype=np.int64) if not starts else np.sort(np.concatenate(starts))


def _greedily_spaced_starts(starts: np.ndarray, minimum_start_gap: int) -> np.ndarray:
    if starts.size == 0:
        return starts.astype(np.int64, copy=False)
    retained = [int(starts[0])]
    for value in starts[1:]:
        current = int(value)
        if current - retained[-1] >= minimum_start_gap:
            retained.append(current)
    return np.asarray(retained, dtype=np.int64)


def validate_mhr_interaction_trim(value: Any, context: str = "interaction trim") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping, got {type(value).__name__}")
    required = ("export_source_start_frame", "export_source_end_frame", "export_frame_count")
    missing = [key for key in required if key not in value]
    if missing:
        raise KeyError(f"{context} is missing {missing}")
    start = int(value["export_source_start_frame"])
    end = int(value["export_source_end_frame"])
    export_frame_count = int(value["export_frame_count"])
    source_frame_count = int(value["source_frame_count"]) if value.get("source_frame_count") is not None else None
    if start < 0 or end < start or end - start != export_frame_count:
        raise ValueError(f"{context} has invalid half-open source range [{start},{end}) for export_frame_count={export_frame_count}")
    if source_frame_count is not None and end > source_frame_count:
        raise ValueError(f"{context} source end {end} exceeds source_frame_count={source_frame_count}")
    normalized = {"export_source_start_frame": start, "export_source_end_frame": end, "export_frame_count": export_frame_count}
    if source_frame_count is not None:
        normalized["source_frame_count"] = source_frame_count
    if value.get("schema") is not None:
        normalized["schema"] = str(value["schema"])
    exclusions = value.get("excluded_source_frame_ranges", [])
    if not isinstance(exclusions, Sequence) or isinstance(exclusions, (str, bytes)):
        raise TypeError(f"{context} excluded_source_frame_ranges must be a sequence")
    normalized_exclusions = []
    previous_end = start
    for index, exclusion in enumerate(exclusions):
        if not isinstance(exclusion, Mapping) or "start_frame" not in exclusion or "end_frame" not in exclusion:
            raise ValueError(f"{context} excluded_source_frame_ranges[{index}] must contain start_frame and end_frame")
        exclusion_start = int(exclusion["start_frame"])
        exclusion_end = int(exclusion["end_frame"])
        if exclusion_start < start or exclusion_end <= exclusion_start or exclusion_end > end:
            raise ValueError(f"{context} excluded source range [{exclusion_start},{exclusion_end}) must be non-empty and contained in [{start},{end})")
        if exclusion_start < previous_end:
            raise ValueError(f"{context} excluded source ranges must be sorted and non-overlapping")
        normalized_exclusion = {"start_frame": exclusion_start, "end_frame": exclusion_end}
        if exclusion.get("reason") is not None:
            reason = str(exclusion["reason"]).strip()
            if not reason:
                raise ValueError(f"{context} excluded_source_frame_ranges[{index}] has an empty reason")
            normalized_exclusion["reason"] = reason
        normalized_exclusions.append(normalized_exclusion)
        previous_end = exclusion_end
    if normalized_exclusions:
        normalized["excluded_source_frame_ranges"] = normalized_exclusions
    return normalized


def load_mhr_interaction_trim(root: str | Path, sequence: str) -> dict[str, Any]:
    path = Path(root) / str(sequence) / "interaction_trim.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing interaction trim for {sequence}: {path}")
    return validate_mhr_interaction_trim(json.loads(path.read_text()), f"interaction trim for {sequence}")


def _window_trim_mask(frames: Sequence[str], starts: np.ndarray, strides: np.ndarray, clip_len: int, interaction_trim: Mapping[str, Any]) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    strides = np.asarray(strides, dtype=np.int64)
    if starts.shape != strides.shape:
        raise ValueError(f"Window starts and strides must have identical shapes, got {starts.shape} and {strides.shape}")
    if starts.size == 0:
        return np.empty(0, dtype=bool)
    frame_names = [Path(str(frame)).name for frame in frames]
    if not all(name.isdigit() for name in frame_names):
        raise ValueError("Interaction-trim-aware sampling requires numeric source-frame names")
    positions = starts[:, None] + np.arange(clip_len, dtype=np.int64)[None] * strides[:, None]
    if np.any(starts < 0) or np.any(strides <= 0) or np.any(positions >= len(frame_names)):
        raise ValueError("Window starts or strides address frames outside the packed timeline")
    source_frames = np.asarray([int(name) for name in frame_names], dtype=np.int64)[positions]
    trim = validate_mhr_interaction_trim(interaction_trim)
    keep = np.all((source_frames >= trim["export_source_start_frame"]) & (source_frames < trim["export_source_end_frame"]), axis=1)
    for exclusion in trim.get("excluded_source_frame_ranges", []):
        keep &= ~np.any((source_frames >= exclusion["start_frame"]) & (source_frames < exclusion["end_frame"]), axis=1)
    return keep


def _required_frame_window_mask(required_frame_mask: np.ndarray, starts: np.ndarray, strides: np.ndarray, clip_len: int) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    strides = np.asarray(strides, dtype=np.int64)
    if starts.shape != strides.shape:
        raise ValueError(f"Window starts and strides must have identical shapes, got {starts.shape} and {strides.shape}")
    if starts.size == 0:
        return np.empty(0, dtype=bool)
    positions = starts[:, None] + np.arange(int(clip_len), dtype=np.int64)[None] * strides[:, None]
    if np.any(starts < 0) or np.any(strides <= 0) or np.any(positions >= len(required_frame_mask)):
        raise ValueError("Window starts or strides address frames outside the required-frame mask")
    return np.all(required_frame_mask[positions], axis=1)


def minimum_true_frame_fraction_window_mask(frame_mask_by_camera: Any, starts: Any, strides: Any, clip_len: int, minimum_fraction: float) -> np.ndarray:
    frame_mask_by_camera = np.asarray(frame_mask_by_camera, dtype=bool)
    starts = np.asarray(starts, dtype=np.int64)
    strides = np.asarray(strides, dtype=np.int64)
    clip_len = int(clip_len)
    minimum_fraction = float(minimum_fraction)
    if frame_mask_by_camera.ndim != 2 or frame_mask_by_camera.shape[0] == 0:
        raise ValueError(f"Per-camera frame mask must have shape [C,F] with at least one camera, got {frame_mask_by_camera.shape}")
    if starts.shape != strides.shape:
        raise ValueError(f"Window starts and strides must have identical shapes, got {starts.shape} and {strides.shape}")
    if clip_len <= 0 or not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError(f"clip_len must be positive and minimum_fraction must be in [0,1], got {clip_len} and {minimum_fraction}")
    if starts.size == 0:
        return np.empty(0, dtype=bool)
    positions = starts[:, None] + np.arange(clip_len, dtype=np.int64)[None] * strides[:, None]
    if np.any(starts < 0) or np.any(strides <= 0) or np.any(positions >= frame_mask_by_camera.shape[1]):
        raise ValueError("Window starts or strides address frames outside the per-camera frame mask")
    minimum_count = int(np.ceil(clip_len * minimum_fraction))
    counts = np.count_nonzero(frame_mask_by_camera[:, positions], axis=2)
    return np.all(counts >= minimum_count, axis=0)


def validate_mhr_windows_within_interaction_trim(frames: Sequence[str], starts: Any, strides: Any, clip_len: int, interaction_trim: Mapping[str, Any], context: str) -> None:
    mask = _window_trim_mask(frames, np.asarray(starts), np.asarray(strides), int(clip_len), interaction_trim)
    if not np.all(mask):
        raise ValueError(f"{context} contains {int((~mask).sum())} windows outside its interaction trim or intersecting excluded source frames")


def validate_mhr_windows_against_required_frame_mask(required_frame_mask: Any, starts: Any, strides: Any, clip_len: int, context: str) -> None:
    required_frame_mask = np.asarray(required_frame_mask, dtype=bool)
    mask = _required_frame_window_mask(required_frame_mask, np.asarray(starts), np.asarray(strides), int(clip_len))
    if not np.all(mask):
        raise ValueError(f"{context} contains {int((~mask).sum())} windows intersecting unusable canonical-mask frames")


def validate_mhr_windows_against_minimum_true_frame_fraction(frame_mask_by_camera: Any, starts: Any, strides: Any, clip_len: int, minimum_fraction: float, context: str) -> None:
    mask = minimum_true_frame_fraction_window_mask(frame_mask_by_camera, starts, strides, clip_len, minimum_fraction)
    if not np.all(mask):
        raise ValueError(f"{context} contains {int((~mask).sum())} windows below the minimum per-camera canonical object-mask visibility fraction {minimum_fraction}")


def mhr_window_starts_and_strides(frames: Sequence[str], valid_mask: Any, cfg: Any, split: str, interaction_trim: Mapping[str, Any] | None = None, required_frame_mask: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != (len(frames),):
        raise ValueError(f"frame_valid_mask must have shape {(len(frames),)}, got {valid_mask.shape}")
    if required_frame_mask is not None:
        required_frame_mask = np.asarray(required_frame_mask, dtype=bool)
        if required_frame_mask.shape != (len(frames),):
            raise ValueError(f"required_frame_mask must have shape {(len(frames),)}, got {required_frame_mask.shape}")
    contract = build_mhr_window_sampling_contract(cfg, split)
    clip_len = int(contract["clipLength"])
    min_valid_count = int(np.floor(clip_len * float(_cfg_get(cfg, "min_valid_frame_fraction", 0.0)))) + 1
    if contract["mode"] == MHR_WINDOW_SAMPLING_LEGACY:
        starts = contiguous_frame_starts(frames, clip_len, int(contract["windowStartStep"]), include_terminal=True)
        starts = [start for start in starts if int(valid_mask[start:start + clip_len].sum()) >= min_valid_count]
        starts = np.asarray(starts, dtype=np.int64)
        strides = np.ones(len(starts), dtype=np.int64)
        if required_frame_mask is not None:
            keep = _required_frame_window_mask(required_frame_mask, starts, strides, clip_len)
            starts, strides = starts[keep], strides[keep]
        if interaction_trim is not None:
            starts = starts[_window_trim_mask(frames, starts, strides, clip_len, interaction_trim)]
            strides = np.ones(len(starts), dtype=np.int64)
        return starts.astype(np.int32, copy=False), strides.astype(np.int16, copy=False)
    all_starts = []
    all_strides = []
    max_overlap = int(contract["maximumSampledFrameOverlap"])
    for chunk_start, chunk_end in _contiguous_chunks(frames):
        for stride in contract["temporalStrides"]:
            starts = _valid_window_starts(valid_mask, chunk_start, chunk_end, clip_len, int(stride), min_valid_count)
            if required_frame_mask is not None:
                starts = starts[_required_frame_window_mask(required_frame_mask, starts, np.full(len(starts), int(stride), dtype=np.int64), clip_len)]
            if interaction_trim is not None:
                starts = starts[_window_trim_mask(frames, starts, np.full(len(starts), int(stride), dtype=np.int64), clip_len, interaction_trim)]
            starts = _greedily_spaced_starts(starts, (clip_len - max_overlap) * int(stride))
            all_starts.extend(starts.tolist())
            all_strides.extend([int(stride)] * len(starts))
    return np.asarray(all_starts, dtype=np.int32), np.asarray(all_strides, dtype=np.int16)
