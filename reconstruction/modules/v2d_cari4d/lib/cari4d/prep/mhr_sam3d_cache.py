from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from lib_mhr.body_pose import compact_model_params_to_cont_body_np
from lib_mhr.camera_conventions import sam3d_root_camera_to_world_rot6d
from lib_mhr.schema import MHR_PARAM_DIMS


MHR_SAM3D_DIRECT_CACHE_SCHEMA = "mhr-sam3d-direct-cache-v2"
MHR_SAM3D_DIRECT_CACHE_LEGACY_SCHEMAS = frozenset(("mhr-sam3d-direct-cache-v1",))
MHR_SAM3D_DIRECT_PARAM_DIMS = {
    "global_rot": 3,
    "pred_cam_t": 3,
    "body_pose_params": 133,
    "hand_pose_params": MHR_PARAM_DIMS["mhr_hand"],
    "shape_params": MHR_PARAM_DIMS["mhr_shape"],
    "scale_params": MHR_PARAM_DIMS["mhr_scale"],
    "expr_params": MHR_PARAM_DIMS["mhr_face"],
}
MHR_SAM3D_DIRECT_RUNTIME_GEOMETRY_KEYS = (
    "pred_vertices",
    "pred_joint_coords",
    "pred_keypoints_3d",
)


@dataclass(frozen=True)
class MHRSAM3DDirectCache:
    frames: tuple[str, ...]
    camera_id: int
    predictions: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _validate_cache(cache: MHRSAM3DDirectCache) -> None:
    frame_count = len(cache.frames)
    if frame_count == 0:
        raise ValueError("SAM 3D Body direct cache must contain at least one frame")
    if len(set(cache.frames)) != frame_count:
        raise ValueError("SAM 3D Body direct cache frame names must be unique")
    if cache.camera_id < 0:
        raise ValueError(f"SAM 3D Body direct cache camera_id must be nonnegative, got {cache.camera_id}")
    required = set(MHR_SAM3D_DIRECT_PARAM_DIMS)
    if set(cache.predictions) != required:
        raise ValueError(f"SAM 3D Body direct cache prediction keys are invalid: {sorted(cache.predictions)}")
    for key, dim in MHR_SAM3D_DIRECT_PARAM_DIMS.items():
        value = np.asarray(cache.predictions[key])
        if value.shape != (frame_count, dim):
            raise ValueError(f"SAM 3D Body direct cache {key} has shape {value.shape}, expected {(frame_count, dim)}")
        if not np.issubdtype(value.dtype, np.floating) or not np.all(np.isfinite(value)):
            raise ValueError(f"SAM 3D Body direct cache {key} must contain finite floating-point values")


def _array_options(value: np.ndarray) -> dict[str, Any]:
    if value.ndim == 0 or value.size == 0:
        return {}
    return {"compression": "lzf", "shuffle": value.dtype.itemsize > 1, "chunks": True}


def save_mhr_sam3d_direct_cache(path: str | Path, cache: MHRSAM3DDirectCache, *, overwrite: bool = False) -> Path:
    _validate_cache(cache)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"SAM 3D Body direct cache already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = MHR_SAM3D_DIRECT_CACHE_SCHEMA
            handle.attrs["camera_id"] = cache.camera_id
            handle.attrs["metadata_json"] = json.dumps(cache.metadata, sort_keys=True, separators=(",", ":"))
            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("frames", data=np.asarray(cache.frames, dtype=object), dtype=string_dtype)
            predictions = handle.create_group("predictions")
            for key in MHR_SAM3D_DIRECT_PARAM_DIMS:
                value = np.asarray(cache.predictions[key], dtype=np.float32)
                predictions.create_dataset(key, data=value, **_array_options(value))
            handle.flush()
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_mhr_sam3d_direct_cache(path: str | Path, *, expected_frames: Sequence[str] | None = None, expected_camera_id: int | None = None, expected_metadata: Mapping[str, Any] | None = None) -> MHRSAM3DDirectCache:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        if schema != MHR_SAM3D_DIRECT_CACHE_SCHEMA and schema not in MHR_SAM3D_DIRECT_CACHE_LEGACY_SCHEMAS:
            raise ValueError(f"Unsupported SAM 3D Body direct cache schema {schema!r} in {path}")
        frames = tuple(value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in handle["frames"][:])
        camera_id = int(handle.attrs["camera_id"])
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        predictions = {key: np.asarray(handle[f"predictions/{key}"][:], dtype=np.float32) for key in MHR_SAM3D_DIRECT_PARAM_DIMS}
    cache = MHRSAM3DDirectCache(frames=frames, camera_id=camera_id, predictions=predictions, metadata=metadata)
    _validate_cache(cache)
    if expected_frames is not None and tuple(expected_frames) != cache.frames:
        raise ValueError(f"SAM 3D Body direct cache frame list does not match requested frames in {path}")
    if expected_camera_id is not None and expected_camera_id != cache.camera_id:
        raise ValueError(f"SAM 3D Body direct cache camera {cache.camera_id} does not match requested camera {expected_camera_id}")
    if expected_metadata is not None:
        mismatched = {key: (metadata.get(key), expected) for key, expected in expected_metadata.items() if metadata.get(key) != expected}
        if mismatched:
            raise ValueError(f"SAM 3D Body direct cache metadata mismatch in {path}: {mismatched}")
    return cache


def mhr_sam3d_direct_body_params(predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    frame_count = len(np.asarray(predictions["global_rot"]))
    return {
        "mhr_global_rot6d": sam3d_root_camera_to_world_rot6d(predictions["global_rot"], np.eye(4, dtype=np.float32)),
        "mhr_trans": np.zeros((frame_count, 3), dtype=np.float32),
        "mhr_body_pose_cont": compact_model_params_to_cont_body_np(predictions["body_pose_params"]),
        "mhr_hand": np.asarray(predictions["hand_pose_params"], dtype=np.float32),
        "mhr_shape": np.asarray(predictions["shape_params"], dtype=np.float32),
        "mhr_scale": np.asarray(predictions["scale_params"], dtype=np.float32),
        "mhr_face": np.asarray(predictions["expr_params"], dtype=np.float32),
    }


def decode_mhr_sam3d_direct_geometry(mhr_layer: Any, predictions: Mapping[str, np.ndarray], *, batch_size: int = 64) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError(f"SAM 3D Body direct decode batch_size must be positive, got {batch_size}")
    params = mhr_sam3d_direct_body_params(predictions)
    frame_count = len(params["mhr_trans"])
    decoded = {key: [] for key in MHR_SAM3D_DIRECT_RUNTIME_GEOMETRY_KEYS}
    for start in range(0, frame_count, batch_size):
        output = mhr_layer.mhr_forward({key: value[start:start + batch_size] for key, value in params.items()})
        values = (output.vertices, output.joints, output.keypoints)
        for key, value in zip(MHR_SAM3D_DIRECT_RUNTIME_GEOMETRY_KEYS, values):
            if value is None:
                raise RuntimeError(f"MHR parameter decoder did not return {key}")
            if value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor":
                value = value.detach().float().cpu().numpy()
            decoded[key].append(np.asarray(value, dtype=np.float32))
    return {key: np.concatenate(values, axis=0) for key, values in decoded.items()}
