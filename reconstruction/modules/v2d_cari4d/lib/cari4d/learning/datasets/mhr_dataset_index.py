from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from learning.datasets.mhr_window_sampling import MHR_CANONICAL_OBJECT_NONEMPTY_FRAME_FRACTION_DEFAULT, MHR_CANONICAL_OBJECT_NONEMPTY_WINDOW_REVISION, MHR_WINDOW_SAMPLING_LEGACY, build_mhr_window_sampling_contract, validate_mhr_interaction_trim, validate_mhr_windows_against_minimum_true_frame_fraction, validate_mhr_windows_against_required_frame_mask, validate_mhr_windows_within_interaction_trim
from prep.mhr_effective_masks import EFFECTIVE_MASK_REVISION, EFFECTIVE_MASK_WINDOW_VALIDITY_REVISION
from prep.mhr_export_utils import MHR_CAMERA_NAMES


MHR_DATASET_INDEX_SCHEMA = "cari4d.mhr_dataset_index.v2"
MHR_DATASET_INDEX_LEGACY_SCHEMA = "cari4d.mhr_dataset_index.v1"
MHR_DATASET_INDEX_CONFIG_KEYS = (
    "body_model",
    "packed_format",
    "packed_root",
    "mhr_gt_root",
    "render_root",
    "clip_len",
    "window",
    "min_valid_frame_fraction",
    "mhr_sample_kids",
    "random_flip",
    "mhr_rank_local_preprocess",
    "mhr_foundationpose_tier_sampling",
    "mhr_require_foundationpose_training_tier_schema",
    "mhr_required_contact_revision",
    "mhr_object_pose_frame",
)


class DatasetIndexStaleError(ValueError):
    pass


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Unsupported dataset-index configuration value {type(value).__name__}")


def dataset_index_config(cfg: Any, seqs: Sequence[str], split: str) -> dict[str, Any]:
    config = {
        "split": str(split),
        "sequences": [str(seq) for seq in seqs],
        "settings": {key: _json_value(_cfg_get(cfg, key)) for key in MHR_DATASET_INDEX_CONFIG_KEYS},
    }
    interaction_trim_root = _cfg_get(cfg, "mhr_interaction_trim_root")
    if interaction_trim_root:
        config["settings"]["mhr_interaction_trim_root"] = _json_value(interaction_trim_root)
    effective_mask_root = _cfg_get(cfg, "mhr_effective_mask_root")
    if effective_mask_root:
        config["settings"]["mhr_effective_mask_root"] = _json_value(effective_mask_root)
        config["settings"]["mhr_effective_mask_revision"] = EFFECTIVE_MASK_REVISION
        config["settings"]["mhr_effective_mask_window_validity_revision"] = EFFECTIVE_MASK_WINDOW_VALIDITY_REVISION
        if split == "train":
            config["settings"]["mhr_min_canonical_object_nonempty_frame_fraction"] = float(_cfg_get(cfg, "mhr_min_canonical_object_nonempty_frame_fraction", MHR_CANONICAL_OBJECT_NONEMPTY_FRAME_FRACTION_DEFAULT))
            config["settings"]["mhr_canonical_object_nonempty_window_revision"] = MHR_CANONICAL_OBJECT_NONEMPTY_WINDOW_REVISION
    window_sampling_contract = build_mhr_window_sampling_contract(cfg, split)
    if window_sampling_contract["mode"] != MHR_WINDOW_SAMPLING_LEGACY:
        config["window_sampling_contract"] = window_sampling_contract
    return config


def _json_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_index_config_digest(cfg: Any, seqs: Sequence[str], split: str) -> str:
    return _json_digest(dataset_index_config(cfg, seqs, split))


def dataset_source_identity(cfg: Any, seqs: Sequence[str]) -> dict[str, Any]:
    if str(_cfg_get(cfg, "packed_format", "pickle")).lower() != "h5":
        raise ValueError("Persistent MHR dataset indexes require packed_format=h5")
    packed_root = _cfg_get(cfg, "mhr_gt_root") or _cfg_get(cfg, "packed_root")
    render_root = _cfg_get(cfg, "render_root")
    interaction_trim_root = _cfg_get(cfg, "mhr_interaction_trim_root")
    effective_mask_root = _cfg_get(cfg, "mhr_effective_mask_root")
    if not packed_root:
        raise ValueError("Persistent MHR dataset indexes require mhr_gt_root or packed_root")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    latest_mtime_ns = 0
    for seq in seqs:
        paths = (("packed", Path(str(packed_root)) / f"{seq}_MHR-packed.h5"),)
        if render_root:
            paths += (("render", Path(str(render_root)) / f"{seq}_render.h5"),)
        if interaction_trim_root:
            paths += (("interaction_trim", Path(str(interaction_trim_root)) / str(seq) / "interaction_trim.json"),)
        if effective_mask_root:
            sidecar_root = Path(str(effective_mask_root)) / str(seq)
            sidecars = tuple((f"effective_mask_k{camera_id}", sidecar_root / f"{camera_name}.h5") for camera_id, camera_name in enumerate(MHR_CAMERA_NAMES) if (sidecar_root / f"{camera_name}.h5").is_file())
            if not sidecars:
                raise FileNotFoundError(f"No canonical effective-mask sidecars found for {seq}: {sidecar_root}")
            paths += sidecars
        for kind, path in paths:
            stat = path.stat()
            record = f"{kind}\0{seq}\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8")
            digest.update(record)
            file_count += 1
            total_bytes += int(stat.st_size)
            latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
    return {"digest": digest.hexdigest(), "file_count": file_count, "total_bytes": total_bytes, "latest_mtime_ns": latest_mtime_ns}


def dataset_index_payload(dataset: Any, cfg: Any, seqs: Sequence[str], split: str, source_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": MHR_DATASET_INDEX_SCHEMA,
        "split": str(split),
        "sequences": [str(seq) for seq in seqs],
        "config": dataset_index_config(cfg, seqs, split),
        "config_digest": dataset_index_config_digest(cfg, seqs, split),
        "source_identity": dict(source_identity),
        "sequence_data": dataset.sequence_data,
        "render_metadata": dataset.render_metadata,
        "sample_offsets": np.asarray(dataset.sample_offsets, dtype=np.int64),
        "sample_count": int(dataset._sample_count),
    }


def _validate_payload(payload: Any, cfg: Any, seqs: Sequence[str], split: str, source_identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"MHR dataset index must be a dictionary, got {type(payload).__name__}")
    schema = payload.get("schema")
    expected_window_contract = build_mhr_window_sampling_contract(cfg, split)
    if schema not in (MHR_DATASET_INDEX_SCHEMA, MHR_DATASET_INDEX_LEGACY_SCHEMA):
        raise DatasetIndexStaleError(f"Unsupported MHR dataset index schema {schema!r}")
    if schema == MHR_DATASET_INDEX_LEGACY_SCHEMA and expected_window_contract["mode"] != MHR_WINDOW_SAMPLING_LEGACY:
        raise DatasetIndexStaleError("Legacy MHR dataset indexes do not contain temporal-stride metadata")
    expected_sequences = [str(seq) for seq in seqs]
    if payload.get("split") != split or payload.get("sequences") != expected_sequences:
        raise DatasetIndexStaleError("MHR dataset index split or sequence order changed")
    expected_config_digest = dataset_index_config_digest(cfg, seqs, split)
    if payload.get("config_digest") != expected_config_digest:
        raise DatasetIndexStaleError("MHR dataset index configuration changed")
    if source_identity is not None and payload.get("source_identity", {}).get("digest") != source_identity.get("digest"):
        raise DatasetIndexStaleError("MHR dataset index source files changed")
    sequence_data = payload.get("sequence_data")
    render_metadata = payload.get("render_metadata")
    sample_offsets = np.asarray(payload.get("sample_offsets"))
    sample_count = payload.get("sample_count")
    if not isinstance(sequence_data, list) or len(sequence_data) != len(expected_sequences):
        raise ValueError("MHR dataset index sequence_data has an invalid length")
    if not isinstance(render_metadata, dict):
        raise TypeError("MHR dataset index render_metadata must be a dictionary")
    if sample_offsets.shape != (len(expected_sequences) + 1,) or not np.issubdtype(sample_offsets.dtype, np.integer):
        raise ValueError(f"MHR dataset index sample_offsets has invalid shape or dtype: {sample_offsets.shape} {sample_offsets.dtype}")
    if sample_offsets[0] != 0 or np.any(np.diff(sample_offsets) < 0) or int(sample_offsets[-1]) != int(sample_count):
        raise ValueError("MHR dataset index sample offsets are inconsistent")
    for sequence_index, (expected_seq, data) in enumerate(zip(expected_sequences, sequence_data)):
        if not isinstance(data, dict) or data.get("seq") != expected_seq:
            raise ValueError(f"MHR dataset index sequence identity mismatch for {expected_seq}")
        frames = data.get("frames")
        kids = data.get("kids")
        starts = np.asarray(data.get("sample_starts"))
        strides = np.ones_like(starts, dtype=np.int16) if schema == MHR_DATASET_INDEX_LEGACY_SCHEMA else np.asarray(data.get("sample_strides"))
        sample_kids = data.get("sample_kids")
        if not isinstance(frames, list) or not frames or not isinstance(kids, list) or not kids:
            raise ValueError(f"MHR dataset index has invalid frames or camera IDs for {expected_seq}")
        if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer) or strides.shape != starts.shape or not np.issubdtype(strides.dtype, np.integer) or np.any(strides <= 0) or sample_kids is None:
            raise ValueError(f"MHR dataset index has invalid sample metadata for {expected_seq}")
        data["sample_starts"] = starts.astype(np.int32, copy=False)
        data["sample_strides"] = strides.astype(np.int16, copy=False)
        expected_samples = len(starts) * len(sample_kids)
        if int(sample_offsets[sequence_index + 1] - sample_offsets[sequence_index]) != expected_samples:
            raise ValueError(f"MHR dataset index sample offset for {expected_seq} does not match {len(starts)} windows and {len(sample_kids)} cameras")
        for key in ("human_pose_valid_mask", "object_pose_valid_mask", "frame_valid_mask"):
            if np.asarray(data.get(key)).shape != (len(frames),):
                raise ValueError(f"MHR dataset index has invalid {key} for {expected_seq}")
        if _cfg_get(cfg, "mhr_effective_mask_root"):
            canonical_mask_frame_valid = np.asarray(data.get("canonical_mask_frame_valid_mask"))
            if canonical_mask_frame_valid.shape != (len(frames),) or not np.isin(canonical_mask_frame_valid, (0, 1)).all():
                raise ValueError(f"MHR dataset index has invalid canonical_mask_frame_valid_mask for {expected_seq}")
            if data.get("canonical_mask_window_validity_revision") != EFFECTIVE_MASK_WINDOW_VALIDITY_REVISION:
                raise DatasetIndexStaleError(f"MHR dataset index has stale canonical-mask window validity for {expected_seq}")
            data["canonical_mask_frame_valid_mask"] = canonical_mask_frame_valid.astype(bool, copy=False)
            validate_mhr_windows_against_required_frame_mask(data["canonical_mask_frame_valid_mask"], starts, strides, int(expected_window_contract["clipLength"]), f"dataset index for {expected_seq}")
            if split == "train":
                minimum_fraction = float(_cfg_get(cfg, "mhr_min_canonical_object_nonempty_frame_fraction", MHR_CANONICAL_OBJECT_NONEMPTY_FRAME_FRACTION_DEFAULT))
                object_nonempty_by_camera = np.asarray(data.get("canonical_object_nonempty_by_camera_mask"))
                if object_nonempty_by_camera.shape != (len(sample_kids), len(frames)) or not np.isin(object_nonempty_by_camera, (0, 1)).all():
                    raise ValueError(f"MHR dataset index has invalid canonical_object_nonempty_by_camera_mask for {expected_seq}")
                if data.get("canonical_object_nonempty_window_revision") != MHR_CANONICAL_OBJECT_NONEMPTY_WINDOW_REVISION:
                    raise DatasetIndexStaleError(f"MHR dataset index has stale canonical object-mask visibility window policy for {expected_seq}")
                data["canonical_object_nonempty_by_camera_mask"] = object_nonempty_by_camera.astype(bool, copy=False)
                validate_mhr_windows_against_minimum_true_frame_fraction(data["canonical_object_nonempty_by_camera_mask"], starts, strides, int(expected_window_contract["clipLength"]), minimum_fraction, f"dataset index for {expected_seq}")
        if _cfg_get(cfg, "mhr_interaction_trim_root"):
            interaction_trim = data.get("interaction_trim")
            if interaction_trim is None:
                raise DatasetIndexStaleError(f"MHR dataset index lacks interaction-trim metadata for {expected_seq}")
            data["interaction_trim"] = validate_mhr_interaction_trim(interaction_trim, f"dataset-index interaction trim for {expected_seq}")
            validate_mhr_windows_within_interaction_trim(frames, starts, strides, int(expected_window_contract["clipLength"]), data["interaction_trim"], f"dataset index for {expected_seq}")
    payload["sample_offsets"] = sample_offsets.astype(np.int64, copy=False)
    payload["sample_count"] = int(sample_count)
    return payload


def load_dataset_index(path: str | Path, cfg: Any, seqs: Sequence[str], split: str, source_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return _validate_payload(payload, cfg, seqs, split, source_identity)


def dataset_index_is_current(path: str | Path, cfg: Any, seqs: Sequence[str], split: str, source_identity: Mapping[str, Any]) -> bool:
    if not Path(path).is_file():
        return False
    try:
        load_dataset_index(path, cfg, seqs, split, source_identity)
    except DatasetIndexStaleError:
        return False
    return True


def write_dataset_index(path: str | Path, dataset: Any, cfg: Any, seqs: Sequence[str], split: str, source_identity: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataset_index_payload(dataset, cfg, seqs, split, source_identity)
    temporary_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return load_dataset_index(path, cfg, seqs, split, source_identity)
