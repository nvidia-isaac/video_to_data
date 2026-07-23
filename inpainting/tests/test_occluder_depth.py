from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from inpainting.contracts import ContractError, VideoGeometry
from inpainting.occluder_depth import (
    DEPTH_SEMANTICS,
    OCCLUDER_DEPTH_PROVENANCE_SCHEMA,
    OCCLUDER_DEPTH_SCHEMA,
    OCCLUDER_METADATA_NAME,
    validate_occluder_depth_bundle,
)


GEOMETRY = VideoGeometry(frame_count=3, width=5, height=4, fps=30.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, recorded_path: str | None = None) -> dict[str, object]:
    return {
        "path": recorded_path or str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_bundle(root: Path) -> tuple[Path, dict[str, object]]:
    root.mkdir()
    mask_path = root / "occluder_mask.npy"
    depth_path = root / "occluder_depth.npy"
    metadata_path = root / OCCLUDER_METADATA_NAME
    source = root.parent / "source.mp4"
    implementation = root.parent / "producer.py"
    source.write_bytes(b"rgb source")
    implementation.write_bytes(b"implementation source")

    mask = np.zeros(
        (GEOMETRY.frame_count, GEOMETRY.height, GEOMETRY.width), dtype=np.bool_
    )
    mask[:, 1:3, 2:4] = True
    depth = np.full(mask.shape, np.inf, dtype=np.float32)
    depth[mask] = 0.75
    np.save(mask_path, mask)
    np.save(depth_path, depth)
    metadata: dict[str, object] = {
        "schema_version": OCCLUDER_DEPTH_SCHEMA,
        "state": "complete",
        "run_id": "test-run",
        "sequence_id": "taco_cut__knife__plate_20231013_105",
        "host_output_dir": str(root.resolve()),
        "geometry": GEOMETRY.as_dict(),
        "occluder_scope": "tool_and_target",
        "source_modalities": ["rgb"],
        "producer": {
            "name": "rgb_estimated_depth",
            "method": "synthetic test producer",
            "version": "test-v1",
        },
        "depth_semantics": DEPTH_SEMANTICS,
        "artifacts": {
            "mask": "/output/occluder_mask.npy",
            "depth": "/output/occluder_depth.npy",
        },
        "artifact_bytes": {
            "mask": mask_path.stat().st_size,
            "depth": depth_path.stat().st_size,
        },
        "artifact_sha256": {
            "mask": _sha256(mask_path),
            "depth": _sha256(depth_path),
        },
        "provenance": {
            "schema_version": OCCLUDER_DEPTH_PROVENANCE_SCHEMA,
            "hash_algorithm": "sha256",
            "inputs": {"source_video": _record(source)},
            "implementation_sources": [_record(implementation)],
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path, metadata


def _rewrite(metadata_path: Path, metadata: dict[str, object]) -> None:
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def _refresh_artifact(metadata: dict[str, object], root: Path, key: str) -> None:
    path = root / f"occluder_{key}.npy"
    metadata["artifact_bytes"][key] = path.stat().st_size  # type: ignore[index]
    metadata["artifact_sha256"][key] = _sha256(path)  # type: ignore[index]


def test_valid_bundle_returns_exact_committed_artifacts(tmp_path: Path) -> None:
    metadata_path, metadata = _write_bundle(tmp_path / "bundle")
    loaded, artifacts = validate_occluder_depth_bundle(metadata_path, GEOMETRY)

    assert loaded == metadata
    assert artifacts == {
        "mask": (metadata_path.parent / "occluder_mask.npy").resolve(),
        "depth": (metadata_path.parent / "occluder_depth.npy").resolve(),
    }


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("schema_version", "wrong", "Occluder schema"),
        ("state", "running", "state must be 'complete'"),
        ("occluder_scope", "objects", "occluder_scope"),
        ("source_modalities", ["rgb", "rgb"], "must not contain duplicates"),
        ("producer", {"name": "rgb_depth"}, "missing required keys"),
        (
            "depth_semantics",
            {**DEPTH_SEMANTICS, "quantity": "ray_range"},
            "metric OpenCV camera-z",
        ),
    ],
)
def test_metadata_semantics_are_strict(
    tmp_path: Path, key: str, value: object, match: str
) -> None:
    metadata_path, metadata = _write_bundle(tmp_path / "bundle")
    metadata[key] = value
    _rewrite(metadata_path, metadata)
    with pytest.raises(ContractError, match=match):
        validate_occluder_depth_bundle(metadata_path, GEOMETRY)


def test_geometry_and_host_output_directory_must_match_consumer(
    tmp_path: Path,
) -> None:
    metadata_path, metadata = _write_bundle(tmp_path / "bundle")
    with pytest.raises(ContractError, match="does not match expected"):
        validate_occluder_depth_bundle(
            metadata_path,
            VideoGeometry(frame_count=3, width=6, height=4, fps=30.0),
        )

    metadata["host_output_dir"] = str(tmp_path)
    _rewrite(metadata_path, metadata)
    with pytest.raises(ContractError, match="host_output_dir"):
        validate_occluder_depth_bundle(metadata_path, GEOMETRY)


@pytest.mark.parametrize(
    "failure", ["mask_dtype", "depth_dtype", "finite_off_mask", "invalid_on_mask"]
)
def test_mask_and_metric_camera_z_depth_contract(tmp_path: Path, failure: str) -> None:
    metadata_path, metadata = _write_bundle(tmp_path / "bundle")
    root = metadata_path.parent
    mask_path = root / "occluder_mask.npy"
    depth_path = root / "occluder_depth.npy"
    mask = np.load(mask_path)
    depth = np.load(depth_path)
    if failure == "mask_dtype":
        np.save(mask_path, mask.astype(np.uint8))
        _refresh_artifact(metadata, root, "mask")
        match = "boolean dtype"
    elif failure == "depth_dtype":
        np.save(depth_path, depth.astype(np.float64))
        _refresh_artifact(metadata, root, "depth")
        match = "float32 dtype"
    elif failure == "finite_off_mask":
        depth[0, 0, 0] = 1.0
        np.save(depth_path, depth)
        _refresh_artifact(metadata, root, "depth")
        match = r"\+inf outside"
    else:
        depth[0, 1, 2] = 0.0
        np.save(depth_path, depth)
        _refresh_artifact(metadata, root, "depth")
        match = "non-finite or non-positive"
    _rewrite(metadata_path, metadata)

    with pytest.raises(ContractError, match=match):
        validate_occluder_depth_bundle(metadata_path, GEOMETRY)


def test_artifact_hash_detects_same_size_tampering(tmp_path: Path) -> None:
    metadata_path, _ = _write_bundle(tmp_path / "bundle")
    mask_path = metadata_path.parent / "occluder_mask.npy"
    payload = bytearray(mask_path.read_bytes())
    payload[-1] ^= 1
    mask_path.write_bytes(payload)

    with pytest.raises(ContractError, match="SHA-256 mismatch for mask"):
        validate_occluder_depth_bundle(metadata_path, GEOMETRY)


def test_artifact_paths_cannot_escape_bundle(tmp_path: Path) -> None:
    metadata_path, metadata = _write_bundle(tmp_path / "bundle")
    metadata["artifacts"]["mask"] = "../occluder_mask.npy"  # type: ignore[index]
    _rewrite(metadata_path, metadata)
    with pytest.raises(ContractError, match="must declare basename"):
        validate_occluder_depth_bundle(metadata_path, GEOMETRY)


def test_provenance_is_structural_by_default_and_optionally_rehashed(
    tmp_path: Path,
) -> None:
    metadata_path, metadata = _write_bundle(tmp_path / "bundle")
    source_record = metadata["provenance"]["inputs"]["source_video"]  # type: ignore[index]
    source = Path(source_record["path"])
    source.write_bytes(b"changed!!!")

    # Composite consumption need not make upstream source files live inputs.
    validate_occluder_depth_bundle(metadata_path, GEOMETRY)
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        validate_occluder_depth_bundle(
            metadata_path, GEOMETRY, verify_provenance_files=True
        )


def test_relative_implementation_provenance_uses_explicit_root(
    tmp_path: Path,
) -> None:
    metadata_path, metadata = _write_bundle(tmp_path / "bundle")
    implementation = tmp_path / "producer.py"
    metadata["provenance"]["implementation_sources"] = [  # type: ignore[index]
        _record(implementation, recorded_path="producer.py")
    ]
    _rewrite(metadata_path, metadata)

    with pytest.raises(ContractError, match="provenance_root is required"):
        validate_occluder_depth_bundle(
            metadata_path, GEOMETRY, verify_provenance_files=True
        )
    validate_occluder_depth_bundle(
        metadata_path,
        GEOMETRY,
        verify_provenance_files=True,
        provenance_root=tmp_path,
    )
