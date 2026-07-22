"""Deterministic SHA-256 records for robot-render inputs and implementation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any


PROVENANCE_SCHEMA = "v2d.inpainting.robot-render-provenance/v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

# The enrichment CLI verifies old bundles but does not participate in a render.
# Every other top-level module in this package is part of the host/container
# renderer implementation and is captured automatically in sorted path order.
_NON_RENDER_SOURCE_FILES = frozenset({"enrich_metadata.py"})


class ProvenanceError(ValueError):
    """Raised when a provenance record is invalid or no longer matches a file."""


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the lowercase SHA-256 of one regular file without loading it at once."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, *, recorded_path: str | None = None) -> dict[str, Any]:
    """Build a deterministic portable record for one existing file."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source) if recorded_path is None else recorded_path,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def renderer_source_records(
    package_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Hash all production Python files in this renderer package.

    Paths are recorded relative to the repository root so host and container
    captures are byte-for-byte comparable even though their mount points differ.
    """

    package = (
        Path(package_root).resolve()
        if package_root is not None
        else Path(__file__).resolve().parent
    )
    if not package.is_dir():
        raise FileNotFoundError(package)
    repository = package.parents[1]
    sources = sorted(
        path.resolve()
        for path in package.glob("*.py")
        if path.name not in _NON_RENDER_SOURCE_FILES
    )
    if not sources:
        raise ProvenanceError(f"renderer package contains no Python sources: {package}")
    return [
        file_record(
            source,
            recorded_path=source.relative_to(repository).as_posix(),
        )
        for source in sources
    ]


def build_provenance(
    *,
    trajectory: str | Path,
    intrinsic: str | Path,
    world_to_camera: str | Path,
    capture_mode: str,
    recorded_paths: dict[str, str] | None = None,
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the input/source portion of a render provenance sidecar."""

    if not capture_mode:
        raise ProvenanceError("capture_mode must not be empty")
    labels = recorded_paths or {}
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "hash_algorithm": "sha256",
        "capture_mode": capture_mode,
        "inputs": {
            "trajectory": file_record(
                trajectory, recorded_path=labels.get("trajectory")
            ),
            "intrinsic": file_record(
                intrinsic, recorded_path=labels.get("intrinsic")
            ),
            "world_to_camera": file_record(
                world_to_camera, recorded_path=labels.get("world_to_camera")
            ),
        },
        "renderer_source_files": renderer_source_records(package_root),
    }


def validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ProvenanceError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def verify_file_record(
    path: str | Path,
    record: object,
    *,
    label: str,
    compare_path: bool = False,
) -> None:
    """Refuse a malformed or stale file record."""

    source = Path(path).resolve()
    if not isinstance(record, dict):
        raise ProvenanceError(f"{label} provenance must be an object")
    expected_bytes = record.get("bytes")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise ProvenanceError(f"{label} provenance bytes must be an integer")
    if source.stat().st_size != expected_bytes:
        raise ProvenanceError(
            f"{label} provenance byte mismatch: recorded {expected_bytes}, "
            f"actual {source.stat().st_size}"
        )
    expected_hash = validate_sha256(record.get("sha256"), label=f"{label} sha256")
    actual_hash = sha256_file(source)
    if actual_hash != expected_hash:
        raise ProvenanceError(
            f"{label} provenance SHA-256 mismatch: recorded {expected_hash}, "
            f"actual {actual_hash}"
        )
    if compare_path and record.get("path") != str(source):
        raise ProvenanceError(
            f"{label} provenance path {record.get('path')!r} != {str(source)!r}"
        )
