from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from lib_mhr.schema import MHR_PARAM_DIMS


MHR_REFIT_CACHE_SCHEMA = "mhr-sam3d-refit-cache-v2"
MHR_REFIT_CACHE_LEGACY_SCHEMAS = frozenset(("mhr-sam3d-refit-cache-v1",))
MHR_REFIT_ALIGNMENT_KEYS = (
    "matrix_cam",
    "scales",
    "translations_cam",
    "image_corrections_cam",
    "image_corrections_px",
    "source_points",
    "target_points",
    "fallback",
)


@dataclass(frozen=True)
class MHRRefitCache:
    frames: tuple[str, ...]
    camera_id: int
    params: dict[str, np.ndarray]
    alignment: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _validate_cache(cache: MHRRefitCache) -> None:
    frame_count = len(cache.frames)
    if frame_count == 0:
        raise ValueError("MHR refit cache must contain at least one frame")
    if len(set(cache.frames)) != frame_count:
        raise ValueError("MHR refit cache frame names must be unique")
    if cache.camera_id < 0:
        raise ValueError(f"MHR refit cache camera_id must be nonnegative, got {cache.camera_id}")
    if set(cache.params) != set(MHR_PARAM_DIMS):
        raise ValueError(f"MHR refit cache parameter keys do not match canonical MHR keys: {sorted(cache.params)}")
    for key, dim in MHR_PARAM_DIMS.items():
        value = np.asarray(cache.params[key])
        if value.shape != (frame_count, dim):
            raise ValueError(f"MHR refit cache {key} has shape {value.shape}, expected {(frame_count, dim)}")
        if not np.issubdtype(value.dtype, np.floating) or not np.all(np.isfinite(value)):
            raise ValueError(f"MHR refit cache {key} must contain finite floating-point values")
    if set(cache.alignment) != set(MHR_REFIT_ALIGNMENT_KEYS):
        raise ValueError(f"MHR refit cache alignment keys are invalid: {sorted(cache.alignment)}")
    expected_shapes = {
        "matrix_cam": (frame_count, 4, 4),
        "scales": (frame_count,),
        "translations_cam": (frame_count, 3),
        "image_corrections_cam": (frame_count, 3),
        "image_corrections_px": (frame_count, 2),
        "source_points": (frame_count,),
        "target_points": (frame_count,),
        "fallback": (frame_count,),
    }
    for key, shape in expected_shapes.items():
        value = np.asarray(cache.alignment[key])
        if value.shape != shape:
            raise ValueError(f"MHR refit cache alignment/{key} has shape {value.shape}, expected {shape}")
        if key not in ("fallback", "source_points", "target_points") and not np.all(np.isfinite(value)):
            raise ValueError(f"MHR refit cache alignment/{key} must contain finite values")


def _array_options(value: np.ndarray) -> dict[str, Any]:
    if value.ndim == 0 or value.size == 0:
        return {}
    return {"compression": "lzf", "shuffle": value.dtype.itemsize > 1, "chunks": True}


def save_mhr_refit_cache(path: str | Path, cache: MHRRefitCache, *, overwrite: bool = False) -> Path:
    _validate_cache(cache)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"MHR refit cache already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = MHR_REFIT_CACHE_SCHEMA
            handle.attrs["camera_id"] = cache.camera_id
            handle.attrs["metadata_json"] = json.dumps(cache.metadata, sort_keys=True, separators=(",", ":"))
            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("frames", data=np.asarray(cache.frames, dtype=object), dtype=string_dtype)
            params_group = handle.create_group("params")
            for key in MHR_PARAM_DIMS:
                value = np.asarray(cache.params[key], dtype=np.float32)
                params_group.create_dataset(key, data=value, **_array_options(value))
            alignment_group = handle.create_group("alignment")
            for key in MHR_REFIT_ALIGNMENT_KEYS:
                value = np.asarray(cache.alignment[key])
                if key == "fallback":
                    alignment_group.create_dataset(key, data=value.astype(object), dtype=string_dtype)
                else:
                    alignment_group.create_dataset(key, data=value, **_array_options(value))
            handle.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_mhr_refit_cache(path: str | Path, *, expected_frames: Sequence[str] | None = None, expected_camera_id: int | None = None, expected_metadata: Mapping[str, Any] | None = None) -> MHRRefitCache:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        if schema != MHR_REFIT_CACHE_SCHEMA and schema not in MHR_REFIT_CACHE_LEGACY_SCHEMAS:
            raise ValueError(f"Unsupported MHR refit cache schema {schema!r} in {path}")
        frames = tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in handle["frames"][:])
        camera_id = int(handle.attrs["camera_id"])
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        params = {key: np.asarray(handle[f"params/{key}"][:], dtype=np.float32) for key in MHR_PARAM_DIMS}
        alignment = {}
        for key in MHR_REFIT_ALIGNMENT_KEYS:
            value = handle[f"alignment/{key}"][:]
            alignment[key] = np.asarray([item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value], dtype=object) if key == "fallback" else np.asarray(value)
    cache = MHRRefitCache(frames=frames, camera_id=camera_id, params=params, alignment=alignment, metadata=metadata)
    _validate_cache(cache)
    if expected_frames is not None and tuple(expected_frames) != cache.frames:
        raise ValueError(f"MHR refit cache frame list does not match requested frames in {path}")
    if expected_camera_id is not None and expected_camera_id != cache.camera_id:
        raise ValueError(f"MHR refit cache camera {cache.camera_id} does not match requested camera {expected_camera_id}")
    if expected_metadata is not None:
        mismatched = {key: (metadata.get(key), expected) for key, expected in expected_metadata.items() if metadata.get(key) != expected}
        if mismatched:
            raise ValueError(f"MHR refit cache metadata mismatch in {path}: {mismatched}")
    return cache


def reconstruct_mhr_refit_target_vertices(mhr_layer: Any, cache: MHRRefitCache, *, batch_size: int = 64) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError(f"MHR refit target decode batch_size must be positive, got {batch_size}")
    frame_count = len(cache.frames)
    outputs = []
    for start in range(0, frame_count, batch_size):
        stop = min(start + batch_size, frame_count)
        value = mhr_layer.mhr_forward_vertices({key: array[start:stop] for key, array in cache.params.items()})
        if value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor":
            value = value.detach().float().cpu().numpy()
        outputs.append(np.asarray(value, dtype=np.float32))
    decoded = np.concatenate(outputs, axis=0)
    translations = np.asarray(cache.params["mhr_trans"], dtype=np.float32)[:, None, :]
    scales = np.asarray(cache.alignment["scales"], dtype=np.float32)[:, None, None]
    return ((decoded - translations) * scales + translations).astype(np.float32, copy=False)
