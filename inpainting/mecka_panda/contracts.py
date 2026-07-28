"""Strict on-disk contracts shared by retargeting and rendering stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

TRACKING_SCHEMA = "v2d.inpainting.tracking/v1"
PARALLEL_JAW_SCHEMA = "v2d.inpainting.parallel-jaw-target/v1"
ROBOT_RENDER_SCHEMA = "v2d.inpainting.robot-render/v1"
SIDES = ("left", "right")


class ContractError(ValueError):
    """Raised when an artifact would require an implicit convention."""


def scalar_text(value: object, key: str) -> str:
    """Return a non-empty scalar string."""
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ContractError(f"{key} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    text = str(item)
    if not text:
        raise ContractError(f"{key} must not be empty")
    return text


def sha256(path: str | Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: str | Path) -> dict[str, Any]:
    """Return a stable identity record for one file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256(source),
    }


def _validate_frame_indices(value: np.ndarray) -> int:
    indices = np.asarray(value)
    if indices.ndim != 1 or indices.size == 0:
        raise ContractError("frame_indices must be a non-empty vector")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ContractError("frame_indices must use an integer dtype")
    expected = np.arange(indices.size, dtype=indices.dtype)
    if not np.array_equal(indices, expected):
        raise ContractError("frame_indices must be contiguous source indices 0..N-1")
    return int(indices.size)


def _validate_valid(value: np.ndarray, side: str, frame_count: int) -> np.ndarray:
    valid = np.asarray(value)
    if valid.shape != (frame_count,) or valid.dtype != np.bool_:
        raise ContractError(f"{side}_valid must be bool with shape ({frame_count},)")
    return valid


def _validate_float_rows(
    value: np.ndarray,
    *,
    key: str,
    shape: tuple[int, ...],
    valid: np.ndarray,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise ContractError(f"{key} must be floating with shape {shape}")
    if valid.any() and not np.isfinite(array[valid]).all():
        raise ContractError(f"{key} contains non-finite values in valid rows")
    return array


def validate_tracking_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    """Validate the common camera/world hand tracking archive."""
    required = {
        "schema_version",
        "tracker",
        "coordinate_frame",
        "frame_indices",
        "left_valid",
        "right_valid",
        "left_wrist_position",
        "right_wrist_position",
        "left_wrist_wxyz",
        "right_wrist_wxyz",
        "left_joints_3d",
        "right_joints_3d",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ContractError(f"tracking archive is missing keys: {missing}")
    if scalar_text(arrays["schema_version"], "schema_version") != TRACKING_SCHEMA:
        raise ContractError(f"schema_version must be {TRACKING_SCHEMA!r}")
    scalar_text(arrays["tracker"], "tracker")
    coordinate_frame = scalar_text(arrays["coordinate_frame"], "coordinate_frame")
    if coordinate_frame not in {"camera", "world"}:
        raise ContractError("coordinate_frame must be 'camera' or 'world'")
    frame_count = _validate_frame_indices(arrays["frame_indices"])
    for side in SIDES:
        valid = _validate_valid(arrays[f"{side}_valid"], side, frame_count)
        _validate_float_rows(
            arrays[f"{side}_wrist_position"],
            key=f"{side}_wrist_position",
            shape=(frame_count, 3),
            valid=valid,
        )
        quaternion = _validate_float_rows(
            arrays[f"{side}_wrist_wxyz"],
            key=f"{side}_wrist_wxyz",
            shape=(frame_count, 4),
            valid=valid,
        )
        _validate_float_rows(
            arrays[f"{side}_joints_3d"],
            key=f"{side}_joints_3d",
            shape=(frame_count, 21, 3),
            valid=valid,
        )
        if valid.any() and not np.allclose(
            np.linalg.norm(quaternion[valid], axis=1), 1.0, atol=1e-5
        ):
            raise ContractError(f"{side}_wrist_wxyz contains non-unit quaternions")
    return frame_count


PARALLEL_JAW_KEYS = {
    "schema_version",
    "tracker",
    "coordinate_frame",
    "frame_indices",
    "left_valid",
    "right_valid",
    "left_position",
    "right_position",
    "left_wxyz",
    "right_wxyz",
    "left_aperture_m",
    "right_aperture_m",
}


def validate_parallel_jaw_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    """Validate the embodiment-neutral parallel-jaw target archive."""
    missing = sorted(PARALLEL_JAW_KEYS - set(arrays))
    extra = sorted(set(arrays) - PARALLEL_JAW_KEYS)
    if missing or extra:
        raise ContractError(f"parallel-jaw keys differ: missing={missing}, extra={extra}")
    if scalar_text(arrays["schema_version"], "schema_version") != PARALLEL_JAW_SCHEMA:
        raise ContractError(f"schema_version must be {PARALLEL_JAW_SCHEMA!r}")
    scalar_text(arrays["tracker"], "tracker")
    if scalar_text(arrays["coordinate_frame"], "coordinate_frame") != "camera":
        raise ContractError("MECKA Panda targets must be in the camera frame")
    frame_count = _validate_frame_indices(arrays["frame_indices"])
    for side in SIDES:
        valid = _validate_valid(arrays[f"{side}_valid"], side, frame_count)
        _validate_float_rows(
            arrays[f"{side}_position"],
            key=f"{side}_position",
            shape=(frame_count, 3),
            valid=valid,
        )
        quaternion = _validate_float_rows(
            arrays[f"{side}_wxyz"],
            key=f"{side}_wxyz",
            shape=(frame_count, 4),
            valid=valid,
        )
        aperture = _validate_float_rows(
            arrays[f"{side}_aperture_m"],
            key=f"{side}_aperture_m",
            shape=(frame_count,),
            valid=valid,
        )
        if valid.any() and not np.allclose(
            np.linalg.norm(quaternion[valid], axis=1), 1.0, atol=1e-5
        ):
            raise ContractError(f"{side}_wxyz contains non-unit quaternions")
        if valid.any() and np.any(aperture[valid] < 0.0):
            raise ContractError(f"{side}_aperture_m contains negative widths")
    return frame_count


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ without permitting object deserialization."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def write_npz_atomic(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Atomically install a compressed NPZ in an existing or new directory."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.stem}-", suffix=".partial.npz"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_npy_atomic(path: str | Path, array: np.ndarray) -> Path:
    """Atomically install one NPY array."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.stem}-", suffix=".partial.npy"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, np.asarray(array), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically install strict JSON."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.stem}-", suffix=".partial.json"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

