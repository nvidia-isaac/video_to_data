"""Generic, provenance-strict metric occluder-depth bundle contract.

The contract deliberately describes only the rasterized occlusion seam.  A
producer may estimate depth directly from RGB or render an estimated mesh and
pose, but every consumer receives the same boolean mask and positive metric
OpenCV camera-z depth.  Producer-specific inference remains outside this
module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .contracts import (
    ContractError,
    VideoGeometry,
    validate_depth_file,
    validate_mask_file,
)


OCCLUDER_DEPTH_SCHEMA = "v2d.inpainting.occluder-depth/v1"
OCCLUDER_DEPTH_PROVENANCE_SCHEMA = "v2d.inpainting.occluder-depth-provenance/v1"
OCCLUDER_ARTIFACT_NAMES = {
    "mask": "occluder_mask.npy",
    "depth": "occluder_depth.npy",
}
OCCLUDER_METADATA_NAME = "occluder_depth_metadata.json"
OCCLUDER_SCOPES = frozenset(("tool_and_target", "all_scene_surfaces"))
DEPTH_SEMANTICS = {
    "quantity": "camera_z",
    "units": "metres",
    "coordinate_frame": "opencv_camera",
    "positive_direction": "+z_forward",
    "invalid_value": "+inf",
}

_SAFE_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PRODUCER_KEYS = frozenset(("name", "method", "version"))
_FILE_RECORD_KEYS = frozenset(("path", "bytes", "sha256"))


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, int | str]:
    before = _stat_signature(path)
    digest = _sha256(path)
    if _stat_signature(path) != before:
        raise ContractError(f"Input changed while being fingerprinted: {path}")
    return {"bytes": before[2], "sha256": digest}


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read occluder metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError("Occluder metadata must be a JSON object")
    return value


def _geometry(value: object) -> VideoGeometry:
    if not isinstance(value, dict) or set(value) != {
        "frame_count",
        "width",
        "height",
        "fps",
    }:
        raise ContractError(
            "Occluder geometry must contain exactly frame_count, width, height, and fps"
        )
    integers: list[int] = []
    for key in ("frame_count", "width", "height"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ContractError(f"Occluder geometry {key} must be a positive integer")
        integers.append(item)
    fps = value["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ContractError("Occluder geometry fps must be numeric")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ContractError("Occluder geometry fps must be positive and finite")
    return VideoGeometry(
        frame_count=integers[0], width=integers[1], height=integers[2], fps=fps
    )


def _same_geometry(left: VideoGeometry, right: VideoGeometry) -> bool:
    return (
        left.frame_count == right.frame_count
        and left.width == right.width
        and left.height == right.height
        and abs(left.fps - right.fps) <= 1e-3
    )


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value


def _safe_identifier(value: object, *, label: str) -> str:
    value = _nonempty_text(value, label=label)
    if _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise ContractError(
            f"{label} must be a lowercase identifier containing only letters, "
            "digits, and underscores"
        )
    return value


def _validate_producer(value: object) -> None:
    if not isinstance(value, dict):
        raise ContractError("Occluder producer must be an object")
    missing = _PRODUCER_KEYS - set(value)
    if missing:
        raise ContractError(
            f"Occluder producer is missing required keys: {sorted(missing)}"
        )
    _safe_identifier(value["name"], label="Occluder producer name")
    _nonempty_text(value["method"], label="Occluder producer method")
    _nonempty_text(value["version"], label="Occluder producer version")


def _validate_source_modalities(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ContractError("source_modalities must be a non-empty list")
    modalities = [_safe_identifier(item, label="source modality") for item in value]
    if len(set(modalities)) != len(modalities):
        raise ContractError("source_modalities must not contain duplicates")


def _validate_record(record: object, *, label: str) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != _FILE_RECORD_KEYS:
        raise ContractError(f"{label} must contain exactly path, bytes, and sha256")
    path = record["path"]
    if not isinstance(path, str) or not path:
        raise ContractError(f"{label} path must be a non-empty string")
    path_value = Path(path)
    if not path_value.is_absolute() and (
        path_value == Path(".") or ".." in path_value.parts
    ):
        raise ContractError(
            f"{label} relative path must not escape its provenance root"
        )
    byte_count = record["bytes"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise ContractError(f"{label} bytes must be a non-negative integer")
    if (
        not isinstance(record["sha256"], str)
        or _SHA256_PATTERN.fullmatch(record["sha256"]) is None
    ):
        raise ContractError(f"{label} sha256 must be a lowercase 64-character hash")
    return record


def _validate_provenance(
    value: object,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise ContractError("Occluder provenance must be an object")
    if value.get("schema_version") != OCCLUDER_DEPTH_PROVENANCE_SCHEMA:
        raise ContractError(
            f"Occluder provenance schema must be {OCCLUDER_DEPTH_PROVENANCE_SCHEMA!r}"
        )
    if value.get("hash_algorithm") != "sha256":
        raise ContractError("Occluder provenance hash_algorithm must be 'sha256'")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ContractError("Occluder provenance inputs must be a non-empty object")
    checked_inputs: dict[str, dict[str, Any]] = {}
    for name, record in inputs.items():
        name = _safe_identifier(name, label="Occluder provenance input name")
        checked_inputs[name] = _validate_record(
            record, label=f"Occluder provenance input {name}"
        )

    implementation = value.get("implementation_sources")
    if not isinstance(implementation, list) or not implementation:
        raise ContractError(
            "Occluder provenance implementation_sources must be a non-empty list"
        )
    checked_implementation = [
        _validate_record(record, label=f"Occluder implementation source {index}")
        for index, record in enumerate(implementation)
    ]
    implementation_paths = [record["path"] for record in checked_implementation]
    if len(set(implementation_paths)) != len(implementation_paths):
        raise ContractError(
            "Occluder provenance implementation source paths must be unique"
        )
    return checked_inputs, checked_implementation


def _record_path(record: dict[str, Any], provenance_root: Path | None) -> Path:
    recorded = Path(record["path"])
    if recorded.is_absolute():
        return recorded.resolve()
    if provenance_root is None:
        raise ContractError(
            "A provenance_root is required to verify relative provenance paths"
        )
    root = provenance_root.resolve()
    candidate = (root / recorded).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractError(
            f"Provenance path escapes the selected root: {recorded}"
        ) from exc
    return candidate


def _verify_record(
    record: dict[str, Any], *, label: str, provenance_root: Path | None
) -> None:
    path = _record_path(record, provenance_root)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = _fingerprint(path)
    if actual["bytes"] != record["bytes"]:
        raise ContractError(
            f"{label} byte mismatch: metadata {record['bytes']}, file {actual['bytes']}"
        )
    if actual["sha256"] != record["sha256"]:
        raise ContractError(f"{label} SHA-256 mismatch")


def _resolve_artifacts(
    metadata: dict[str, Any], bundle_root: Path
) -> tuple[dict[str, Path], dict[str, tuple[int, int, int, int]]]:
    artifacts = metadata.get("artifacts")
    sizes = metadata.get("artifact_bytes")
    hashes = metadata.get("artifact_sha256")
    expected_keys = set(OCCLUDER_ARTIFACT_NAMES)
    if not isinstance(artifacts, dict) or set(artifacts) != expected_keys:
        raise ContractError("Occluder artifacts must declare exactly mask and depth")
    if not isinstance(sizes, dict) or set(sizes) != expected_keys:
        raise ContractError(
            "Occluder artifact_bytes must declare exactly mask and depth"
        )
    if not isinstance(hashes, dict) or set(hashes) != expected_keys:
        raise ContractError(
            "Occluder artifact_sha256 must declare exactly mask and depth"
        )

    resolved: dict[str, Path] = {}
    signatures: dict[str, tuple[int, int, int, int]] = {}
    for key, expected_name in OCCLUDER_ARTIFACT_NAMES.items():
        declared = artifacts[key]
        if (
            not isinstance(declared, str)
            or ".." in Path(declared).parts
            or Path(declared).name != expected_name
        ):
            raise ContractError(
                f"Occluder artifact {key!r} must declare basename {expected_name!r}"
            )
        path = (bundle_root / expected_name).resolve()
        try:
            path.relative_to(bundle_root)
        except ValueError as exc:
            raise ContractError(
                f"Occluder artifact {key!r} escapes its bundle"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(f"Occluder artifact {key}: {path}")
        expected_bytes = sizes[key]
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise ContractError(
                f"Occluder artifact_bytes[{key!r}] must be a positive integer"
            )
        expected_hash = hashes[key]
        if (
            not isinstance(expected_hash, str)
            or _SHA256_PATTERN.fullmatch(expected_hash) is None
        ):
            raise ContractError(
                f"Occluder artifact_sha256[{key!r}] must be a lowercase 64-character hash"
            )
        signature = _stat_signature(path)
        actual = _fingerprint(path)
        if _stat_signature(path) != signature:
            raise ContractError(f"Occluder artifact changed during validation: {path}")
        if actual["bytes"] != expected_bytes:
            raise ContractError(
                f"Occluder artifact size mismatch for {key}: metadata "
                f"{expected_bytes}, file {actual['bytes']}"
            )
        if actual["sha256"] != expected_hash:
            raise ContractError(f"Occluder artifact SHA-256 mismatch for {key}")
        resolved[key] = path
        signatures[key] = signature
    return resolved, signatures


def validate_occluder_depth_bundle(
    metadata_path: str | Path,
    expected_geometry: VideoGeometry,
    *,
    verify_provenance_files: bool = False,
    provenance_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate one committed generic occluder bundle and return exact artifacts.

    Artifact bytes and hashes are always checked.  Upstream input and
    implementation records are structurally validated by default; producers
    and batch resume can additionally rehash those files by setting
    ``verify_provenance_files``.  Relative provenance paths then resolve below
    the explicit ``provenance_root``.
    """

    if not isinstance(expected_geometry, VideoGeometry):
        raise TypeError("expected_geometry must be a VideoGeometry")
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata_fingerprint = _fingerprint(metadata_path)
    metadata = _load_metadata(metadata_path)
    if metadata.get("schema_version") != OCCLUDER_DEPTH_SCHEMA:
        raise ContractError(
            f"Occluder schema must be {OCCLUDER_DEPTH_SCHEMA!r}, got "
            f"{metadata.get('schema_version')!r}"
        )
    if metadata.get("state") != "complete":
        raise ContractError("Occluder bundle state must be 'complete'")
    _nonempty_text(metadata.get("run_id"), label="Occluder run_id")
    sequence_id = _nonempty_text(
        metadata.get("sequence_id"), label="Occluder sequence_id"
    )
    if sequence_id in {".", ".."} or Path(sequence_id).name != sequence_id:
        raise ContractError("Occluder sequence_id must be one safe path segment")

    geometry = _geometry(metadata.get("geometry"))
    if not _same_geometry(geometry, expected_geometry):
        raise ContractError(
            f"Occluder geometry {geometry} does not match expected {expected_geometry}"
        )
    scope = metadata.get("occluder_scope")
    if scope not in OCCLUDER_SCOPES:
        raise ContractError(
            f"occluder_scope must be one of {sorted(OCCLUDER_SCOPES)}, got {scope!r}"
        )
    _validate_source_modalities(metadata.get("source_modalities"))
    _validate_producer(metadata.get("producer"))
    if metadata.get("depth_semantics") != DEPTH_SEMANTICS:
        raise ContractError(
            "Occluder depth_semantics must declare positive metric OpenCV camera-z "
            "with +inf invalid values"
        )

    bundle_root = metadata_path.parent.resolve()
    host_output_dir = metadata.get("host_output_dir")
    if (
        not isinstance(host_output_dir, str)
        or not Path(host_output_dir).is_absolute()
        or Path(host_output_dir).resolve() != bundle_root
    ):
        raise ContractError(
            "Occluder host_output_dir must resolve to its metadata directory"
        )
    inputs, implementation = _validate_provenance(metadata.get("provenance"))
    artifacts, artifact_signatures = _resolve_artifacts(metadata, bundle_root)

    mask = validate_mask_file(artifacts["mask"], geometry)
    validate_depth_file(artifacts["depth"], mask, geometry, name="Occluder depth")
    if verify_provenance_files:
        root = (
            Path(provenance_root).expanduser().resolve()
            if provenance_root is not None
            else None
        )
        for name, record in inputs.items():
            _verify_record(
                record,
                label=f"Occluder provenance input {name}",
                provenance_root=root,
            )
        for index, record in enumerate(implementation):
            _verify_record(
                record,
                label=f"Occluder implementation source {index}",
                provenance_root=root,
            )

    # Detect replacement or mutation after the first metadata read and after
    # artifact validation. Composite integration can call this validator again
    # before committing its output, matching the existing robot/TACO pattern.
    if _fingerprint(metadata_path) != metadata_fingerprint:
        raise ContractError(
            f"Occluder metadata changed during validation: {metadata_path}"
        )
    for key, path in artifacts.items():
        if _stat_signature(path) != artifact_signatures[key]:
            raise ContractError(f"Occluder artifact changed during validation: {path}")
    return metadata, artifacts
