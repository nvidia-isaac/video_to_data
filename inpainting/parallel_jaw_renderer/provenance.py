"""Deterministic provenance for target, calibration, bundle, assets, and code."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .bundle import RobotBundle


PROVENANCE_SCHEMA = "v2d.inpainting.robot-render-provenance/v1"


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def file_record(
    path: str | Path, *, recorded_path: str | None = None
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source) if recorded_path is None else recorded_path,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def renderer_source_records() -> list[dict[str, Any]]:
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    sources = sorted(
        path.resolve()
        for path in package.glob("*.py")
        if path.name not in {"container_runner.py"}
    )
    return [
        file_record(
            source,
            recorded_path=source.relative_to(repository).as_posix(),
        )
        for source in sources
    ]


def build_provenance(
    *,
    target: str | Path,
    intrinsic: str | Path,
    world_to_camera: str | Path,
    world_hub: str | Path | None,
    bundle: RobotBundle,
    arm_ik_source: str | Path,
    capture_mode: str,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "target": file_record(target),
        "intrinsic": file_record(intrinsic),
        "world_to_camera": file_record(world_to_camera),
        "robot_bundle": file_record(bundle.source_path),
    }
    if world_hub is not None:
        inputs["world_hub"] = file_record(world_hub)
    visual_assets = [
        file_record(path) for path in bundle.render_inspection.visual_mesh_paths
    ]
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "hash_algorithm": "sha256",
        "capture_mode": capture_mode,
        "inputs": inputs,
        "robot_assets": {
            "render_urdf": file_record(bundle.render_urdf),
            "ik_urdf": file_record(bundle.ik_urdf),
            "visual_meshes": visual_assets,
            "declared_asset_provenance": dict(bundle.asset_provenance),
            "fixed_root_posture_provenance": dict(bundle.fixed_root_provenance),
        },
        "external_ik_source": file_record(arm_ik_source),
        "renderer_source_files": renderer_source_records(),
    }
