# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stabilize per-frame camera intrinsics produced by monocular depth models.

Monocular models (MoGe, UniDepth, etc.) estimate focal length independently
for every frame. Small per-frame errors create apparent scale jitter and, in
particular, cause close-up objects to be placed at the wrong depth when the
estimated focal length deviates from the true value.

The legacy :func:`stabilize_intrinsics` API remains available for callers that
do not yet have an upstream generation record. New experiment code should use
:func:`stabilize_intrinsics_with_provenance`: it accepts a *validated* MoGe
generation, binds the exact per-frame inputs, and writes a sidecar commit that
can be independently revalidated.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from v2d.common.datatypes import CameraIntrinsics


STABLE_INTRINSICS_GENERATION_SCHEMA = (
    "v2d.depth.stable-intrinsics-generation/v1"
)
STABILIZATION_ALGORITHM = "coordinate_wise_temporal_median/v1"
PRINCIPAL_POINT_ESTIMATED = "temporal_median"
PRINCIPAL_POINT_FIXED = "image_center"
_INTRINSICS_KEYS = {"fx", "fy", "cx", "cy", "width", "height"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, recorded_path: str | None = None) -> dict[str, Any]:
    if path.is_symlink():
        raise FileNotFoundError(f"Expected a regular file, got symlink: {path}")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Expected a regular file, got: {path}")
    return {
        "path": recorded_path if recorded_path is not None else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _artifact_identity(value: dict[str, Any]) -> dict[str, Any]:
    try:
        size_bytes = value["size_bytes"]
        sha256 = value["sha256"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Incomplete artifact identity: {exc}") from exc
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RuntimeError("Invalid artifact size or SHA-256 identity")
    return {"size_bytes": size_bytes, "sha256": sha256}


def _intrinsics_files(directory: Path) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(
            f"Intrinsics input must be a regular directory: {directory}"
        )
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    if not entries:
        raise RuntimeError(f"No intrinsics JSON files found in {directory}")
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise RuntimeError(
            f"Intrinsics input contains a non-regular file: {directory}"
        )
    expected = [f"{index:06d}.json" for index in range(len(entries))]
    if [path.name for path in entries] != expected:
        raise RuntimeError(
            "Intrinsics input must contain exactly contiguous six-digit JSON "
            f"frames: {directory}"
        )
    return entries


def _directory_identity(directory: Path, files: list[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    identities: dict[str, dict[str, Any]] = {}
    total_size = 0
    for path in files:
        identity = _artifact_identity(_artifact(path))
        name = path.name.encode("utf-8")
        size = identity["size_bytes"]
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        identities[path.name] = identity
        total_size += size
    return {
        "directory": str(directory.resolve()),
        "file_count": len(files),
        "size_bytes": total_size,
        "aggregate_sha256": digest.hexdigest(),
        "files": identities,
    }


def _directory_identity_without_path(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Invalid intrinsics-directory identity")
    return {key: item for key, item in value.items() if key != "directory"}


def _load_intrinsics(files: list[Path]) -> list[CameraIntrinsics]:
    intrinsics: list[CameraIntrinsics] = []
    for path in files:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid intrinsics JSON {path}: {exc}") from exc
        if not isinstance(value, dict) or set(value) != _INTRINSICS_KEYS:
            raise RuntimeError(
                f"Intrinsics JSON has an invalid field set: {path}"
            )
        width = value["width"]
        height = value["height"]
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise RuntimeError(f"Intrinsics image dimensions are invalid: {path}")
        numeric_values: dict[str, float] = {}
        for name in ("fx", "fy", "cx", "cy"):
            raw = value[name]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise RuntimeError(f"Intrinsics {name} is not numeric: {path}")
            numeric = float(raw)
            if not math.isfinite(numeric):
                raise RuntimeError(f"Intrinsics {name} is not finite: {path}")
            numeric_values[name] = numeric
        if numeric_values["fx"] <= 0 or numeric_values["fy"] <= 0:
            raise RuntimeError(f"Intrinsics focal length is not positive: {path}")
        intrinsics.append(
            CameraIntrinsics(
                **numeric_values,
                width=width,
                height=height,
            )
        )
    return intrinsics


def _compute_stable_intrinsics(
    intrinsics: list[CameraIntrinsics],
    *,
    fix_principal_point: bool,
    require_constant_dimensions: bool,
) -> CameraIntrinsics:
    if not intrinsics:
        raise RuntimeError("Cannot stabilize an empty intrinsics sequence")
    if not isinstance(fix_principal_point, bool):
        raise TypeError("fix_principal_point must be a bool")

    widths = [value.width for value in intrinsics]
    heights = [value.height for value in intrinsics]
    if require_constant_dimensions and (
        len(set(widths)) != 1 or len(set(heights)) != 1
    ):
        raise RuntimeError(
            "Every frame in a committed intrinsics sequence must have the same "
            "image dimensions"
        )

    width = int(np.median(widths))
    height = int(np.median(heights))
    return CameraIntrinsics(
        fx=float(np.median([value.fx for value in intrinsics])),
        fy=float(np.median([value.fy for value in intrinsics])),
        cx=(
            width / 2.0
            if fix_principal_point
            else float(np.median([value.cx for value in intrinsics]))
        ),
        cy=(
            height / 2.0
            if fix_principal_point
            else float(np.median([value.cy for value in intrinsics]))
        ),
        width=width,
        height=height,
    )


def _stage_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_staged(temporary: Path, destination: Path) -> None:
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = _stage_json(path, value)
    try:
        _publish_staged(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_moge_generation_manifest(path: Path) -> dict[str, Any]:
    try:
        from v2d.moge.docker.run_video_to_depth import (
            validate_generation_manifest,
        )
    except ImportError as exc:  # pragma: no cover - installation error path
        raise RuntimeError(
            "v2d-moge-docker is required to validate MoGe provenance"
        ) from exc
    try:
        return validate_generation_manifest(path)
    except Exception as exc:
        raise RuntimeError(f"MoGe generation validation failed: {exc}") from exc


def _validated_moge_intrinsics(
    manifest_path: Path,
    intrinsics_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    moge = _validate_moge_generation_manifest(manifest_path)
    try:
        recorded = moge["outputs"]["intrinsics"]
        recorded_directory = Path(recorded["directory"])
        generation_id = moge["generation_id"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"MoGe generation has no committed intrinsics output: {exc}"
        ) from exc
    if not isinstance(generation_id, str) or not generation_id.startswith("sha256:"):
        raise RuntimeError("MoGe generation ID is invalid")
    if recorded_directory.resolve() != intrinsics_directory.resolve():
        raise RuntimeError(
            "The supplied intrinsics directory is not the exact directory "
            "committed by the MoGe generation"
        )

    files = _intrinsics_files(intrinsics_directory)
    current = _directory_identity(intrinsics_directory, files)
    if _directory_identity_without_path(current) != _directory_identity_without_path(
        recorded
    ):
        raise RuntimeError(
            "Intrinsics bytes do not match the verified MoGe generation"
        )
    return moge, current


def _parameters(fix_principal_point: bool) -> dict[str, Any]:
    return {
        "algorithm": STABILIZATION_ALGORITHM,
        "fix_principal_point": fix_principal_point,
        "principal_point_policy": (
            PRINCIPAL_POINT_FIXED
            if fix_principal_point
            else PRINCIPAL_POINT_ESTIMATED
        ),
        "dimension_policy": "require_constant_across_frames",
        "frame_order": "contiguous_zero_based_six_digit_filenames",
    }


def _static_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        sources = manifest["sources"]
        implementation = manifest["implementation_sources"]
        return {
            "moge_generation_manifest": _artifact_identity(
                sources["moge_generation_manifest"]
            ),
            "moge_generation_id": sources["moge_generation_id"],
            "moge_schema_version": sources["moge_schema_version"],
            "intrinsics": _directory_identity_without_path(
                sources["intrinsics"]
            ),
            "implementation_sources": {
                name: _artifact_identity(artifact)
                for name, artifact in sorted(implementation.items())
            },
            "parameters": manifest["parameters"],
        }
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"Incomplete stable-intrinsics provenance: {exc}") from exc


def _generation_id(
    static_identity: dict[str, Any], output: dict[str, Any]
) -> str:
    value = {
        "static_identity": static_identity,
        "output": {
            "stable_intrinsics": _artifact_identity(output["stable_intrinsics"]),
            "values": output["values"],
        },
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def default_provenance_manifest_path(output_path: str | Path) -> Path:
    """Return the default sidecar path for a stable-intrinsics JSON file."""

    output = Path(output_path)
    return output.with_suffix(".generation.json")


def stabilize_intrinsics(
    intrinsics_folder: str,
    output_path: str,
    fix_principal_point: bool = False,
) -> CameraIntrinsics:
    """Compute temporal-median intrinsics and atomically write one JSON file.

    This is the backward-compatible API. It does not claim auditable lineage;
    use :func:`stabilize_intrinsics_with_provenance` for experiment artifacts.
    """

    # Preserve the original API's permissive input behavior: it considers
    # only JSON entries and delegates parsing to CameraIntrinsics. The strict
    # generation API below deliberately has a narrower committed contract.
    files = sorted(
        path
        for path in Path(intrinsics_folder).iterdir()
        if path.name.endswith(".json")
    )
    if not files:
        raise RuntimeError(
            f"No intrinsics JSON files found in {intrinsics_folder}"
        )
    stable = _compute_stable_intrinsics(
        [CameraIntrinsics.load(str(path)) for path in files],
        fix_principal_point=fix_principal_point,
        require_constant_dimensions=False,
    )
    _atomic_json(Path(output_path).resolve(), stable.to_dict())
    return stable


def stabilize_intrinsics_with_provenance(
    intrinsics_folder: str | Path,
    output_path: str | Path,
    moge_generation_manifest_path: str | Path,
    fix_principal_point: bool = False,
    *,
    provenance_manifest_path: str | Path | None = None,
) -> CameraIntrinsics:
    """Write stable intrinsics and a strict, auditable sidecar generation.

    The sidecar is the commit marker and is published only after the stable-K
    JSON has been durably installed. Both destinations must be fresh, which
    prevents accidental mixing with an older or uncommitted generation.
    """

    if not isinstance(fix_principal_point, bool):
        raise TypeError("fix_principal_point must be a bool")
    directory = Path(intrinsics_folder).resolve()
    output = Path(output_path).resolve()
    moge_manifest_path = Path(moge_generation_manifest_path).resolve()
    provenance = Path(
        provenance_manifest_path
        if provenance_manifest_path is not None
        else default_provenance_manifest_path(output)
    ).resolve()
    if output == provenance:
        raise ValueError("Stable intrinsics and provenance must use distinct paths")
    for destination in (output, provenance):
        if os.path.lexists(destination):
            raise FileExistsError(
                f"Refusing to overwrite stable-intrinsics generation: {destination}"
            )

    moge, input_identity = _validated_moge_intrinsics(
        moge_manifest_path,
        directory,
    )
    files = _intrinsics_files(directory)
    stable = _compute_stable_intrinsics(
        _load_intrinsics(files),
        fix_principal_point=fix_principal_point,
        require_constant_dimensions=True,
    )

    # Close the read/compute time-of-check gap before committing the derivative.
    final_moge, final_input_identity = _validated_moge_intrinsics(
        moge_manifest_path,
        directory,
    )
    if (
        final_moge["generation_id"] != moge["generation_id"]
        or final_input_identity != input_identity
    ):
        raise RuntimeError(
            "MoGe generation or intrinsics inputs changed during stabilization"
        )

    output_temporary: Path | None = None
    manifest_temporary: Path | None = None
    output_published = False
    manifest_published = False
    try:
        output_temporary = _stage_json(output, stable.to_dict())
        output_artifact = _artifact(
            output_temporary,
            recorded_path=str(output),
        )
        source_name = "v2d/depth/lib/stabilize_intrinsics.py"
        manifest: dict[str, Any] = {
            "schema_version": STABLE_INTRINSICS_GENERATION_SCHEMA,
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "sources": {
                "moge_generation_manifest": _artifact(moge_manifest_path),
                "moge_generation_id": moge["generation_id"],
                "moge_schema_version": moge["schema_version"],
                "intrinsics": input_identity,
            },
            "implementation_sources": {
                source_name: _artifact(Path(__file__), recorded_path=source_name)
            },
            "parameters": _parameters(fix_principal_point),
            "output": {
                "stable_intrinsics": output_artifact,
                "values": stable.to_dict(),
            },
        }
        static_identity = _static_identity(manifest)
        manifest["static_identity"] = static_identity
        manifest["generation_id"] = _generation_id(
            static_identity,
            manifest["output"],
        )
        manifest_temporary = _stage_json(provenance, manifest)

        _publish_staged(output_temporary, output)
        output_temporary = None
        output_published = True
        _publish_staged(manifest_temporary, provenance)
        manifest_temporary = None
        manifest_published = True
        validate_stable_intrinsics_manifest(provenance)
    except Exception:
        if manifest_published:
            provenance.unlink(missing_ok=True)
        if output_published:
            output.unlink(missing_ok=True)
        raise
    finally:
        if output_temporary is not None:
            output_temporary.unlink(missing_ok=True)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)
    return stable


def validate_stable_intrinsics_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Strictly validate a stable-K sidecar and all reachable current bytes."""

    path = Path(manifest_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(
            f"Stable-intrinsics manifest is not a regular file: {path}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid stable-intrinsics manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version")
        != STABLE_INTRINSICS_GENERATION_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise RuntimeError(
            "Stable-intrinsics manifest is not a complete v1 commit"
        )

    expected_top_level_keys = {
        "schema_version",
        "state",
        "completed_at",
        "sources",
        "implementation_sources",
        "parameters",
        "output",
        "static_identity",
        "generation_id",
    }
    if set(manifest) != expected_top_level_keys:
        raise RuntimeError("Stable-intrinsics manifest has an invalid field set")

    expected_parameter_keys = {
        "algorithm",
        "fix_principal_point",
        "principal_point_policy",
        "dimension_policy",
        "frame_order",
    }
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != expected_parameter_keys:
        raise RuntimeError("Stable-intrinsics parameters are incomplete")
    fix_principal_point = parameters.get("fix_principal_point")
    if (
        not isinstance(fix_principal_point, bool)
        or parameters != _parameters(fix_principal_point)
    ):
        raise RuntimeError("Stable-intrinsics parameter policy is invalid")

    try:
        sources = manifest["sources"]
        if set(sources) != {
            "moge_generation_manifest",
            "moge_generation_id",
            "moge_schema_version",
            "intrinsics",
        }:
            raise RuntimeError("Stable-intrinsics sources have an invalid field set")
        implementation_sources = manifest["implementation_sources"]
        if set(implementation_sources) != {
            "v2d/depth/lib/stabilize_intrinsics.py"
        }:
            raise RuntimeError(
                "Stable-intrinsics implementation sources are invalid"
            )
        moge_artifact = sources["moge_generation_manifest"]
        moge_path = Path(moge_artifact["path"])
        intrinsics_identity = sources["intrinsics"]
        intrinsics_directory = Path(intrinsics_identity["directory"])
        output_record = manifest["output"]
        if set(output_record) != {"stable_intrinsics", "values"}:
            raise RuntimeError("Stable-intrinsics output has an invalid field set")
        output_artifact = output_record["stable_intrinsics"]
        output_path = Path(output_artifact["path"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Stable-intrinsics manifest is incomplete: {exc}"
        ) from exc

    if _artifact(moge_path) != moge_artifact:
        raise RuntimeError(
            "MoGe generation manifest bytes no longer match stable-K provenance"
        )
    moge, current_input_identity = _validated_moge_intrinsics(
        moge_path,
        intrinsics_directory,
    )
    if current_input_identity != intrinsics_identity:
        raise RuntimeError(
            "Intrinsics input bytes no longer match stable-K provenance"
        )
    if (
        sources.get("moge_generation_id") != moge.get("generation_id")
        or sources.get("moge_schema_version") != moge.get("schema_version")
    ):
        raise RuntimeError("Stable-K provenance references the wrong MoGe generation")

    files = _intrinsics_files(intrinsics_directory)
    expected = _compute_stable_intrinsics(
        _load_intrinsics(files),
        fix_principal_point=fix_principal_point,
        require_constant_dimensions=True,
    )
    if output_record.get("values") != expected.to_dict():
        raise RuntimeError(
            "Stable intrinsics values do not match the declared inputs and policy"
        )
    if _artifact(output_path) != output_artifact:
        raise RuntimeError(
            "Stable intrinsics output bytes no longer match its provenance"
        )
    try:
        current_output_value = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Stable intrinsics output is invalid: {exc}") from exc
    if current_output_value != output_record["values"]:
        raise RuntimeError(
            "Stable intrinsics output JSON contradicts its recorded values"
        )

    static_identity = _static_identity(manifest)
    if manifest.get("static_identity") != static_identity:
        raise RuntimeError(
            "Stable-intrinsics top-level provenance contradicts static_identity"
        )
    if manifest.get("generation_id") != _generation_id(
        static_identity,
        output_record,
    ):
        raise RuntimeError("Stable-intrinsics generation ID is invalid")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics_folder", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--moge_generation_manifest_path", required=True)
    parser.add_argument("--provenance_manifest_path", default=None)
    parser.add_argument("--fix_principal_point", action="store_true")
    arguments = parser.parse_args()
    stabilize_intrinsics_with_provenance(
        intrinsics_folder=arguments.intrinsics_folder,
        output_path=arguments.output_path,
        moge_generation_manifest_path=arguments.moge_generation_manifest_path,
        fix_principal_point=arguments.fix_principal_point,
        provenance_manifest_path=arguments.provenance_manifest_path,
    )
