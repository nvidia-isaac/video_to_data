# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset path resolution for objects + support surfaces.

Parquet-stored mesh/URDF paths are absolute paths from the data-generation environment. At load time
they are re-rooted onto the repo's vendored assets by anchoring on a known asset-root boundary, so
the source prefix is irrelevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flash_chord.assets import ASSETS_DIR
from flash_chord.data.reference import (
    ObjectArticulationSpec,
    ObjectAssetSpec,
    ObjectBodySpec,
    ObjectJointDriveSpec,
    ObjectJointPhysicsSpec,
)

if TYPE_CHECKING:
    from flash_chord.data.reference import Reference

HUMAN_MOTION_DATA_DIR = ASSETS_DIR / "human_motion_data"
ARCTIC_URDF_DIR = ASSETS_DIR / "urdfs" / "arctic"
_ASSETS_MARKER = "/assets/"
_OBJECT_ASSETS_MARKER = "/data/object_assets/"


def arctic_art_urdf(object_name: str) -> Path | None:
    """Articulated-object URDF for an arctic object (``<object_name>_art.urdf``), or None.

    Arctic objects leave per-body URDFs empty in the parquet; the articulated URDF (which
    defines the part hinge) is resolved by ``object_name``."""
    path = ARCTIC_URDF_DIR / f"{object_name}_art.urdf"
    return path if path.exists() else None


def resolve_asset_path(path: str, *, reference_path: str | Path | None = None) -> str:
    """Re-root a parquet-stored asset path onto the vendored repo assets.

    Repository-generated paths anchor on their final ``assets/`` boundary. Dataset-container paths may
    anchor on ``/data/object_assets/``. Paths with neither marker are returned unchanged. When the original
    producer prefix is unavailable, ``reference_path`` lets a materialized V2D dataset supply its own sibling
    asset closure before falling back to FlashCHORD's vendored assets.
    """
    if Path(path).is_file():
        return path
    idx = path.rfind(_ASSETS_MARKER)
    if idx != -1:
        suffix = path[idx + len(_ASSETS_MARKER) :]
        reference_assets = _reference_assets_root(reference_path)
        if reference_assets is not None:
            candidate = reference_assets / suffix
            if candidate.is_file():
                return str(candidate)
        return str(ASSETS_DIR / suffix)
    idx = path.rfind(_OBJECT_ASSETS_MARKER)
    if idx != -1:
        return str(ASSETS_DIR / path[idx + len(_OBJECT_ASSETS_MARKER) :])
    return path


def _reference_assets_root(reference_path: str | Path | None) -> Path | None:
    if reference_path is None:
        return None
    resolved = Path(reference_path).expanduser().resolve()
    parts = resolved.parts
    indices = [index for index, part in enumerate(parts) if part == "assets"]
    return None if not indices else Path(*parts[: indices[-1] + 1])


def _v2d_articulated_urdf(
    *,
    source_dataset: str,
    object_name: str,
    mesh_paths: list[str],
    reference_path: str | Path | None,
) -> Path | None:
    """Resolve V2D's ``object_assets/{meshes,urdfs}`` articulated-object convention."""
    if not mesh_paths:
        return None
    parts = Path(mesh_paths[0]).parts
    try:
        marker = parts.index("object_assets")
    except ValueError:
        return None
    tail = parts[marker + 1 :]
    if len(tail) < 4 or tail[0] != "meshes" or tail[1] != source_dataset or tail[2] != object_name:
        return None

    relative = Path("object_assets") / "urdfs" / source_dataset / f"{object_name}.urdf"
    candidates = []
    reference_assets = _reference_assets_root(reference_path)
    if reference_assets is not None:
        candidates.append(reference_assets / "human_motion_data" / source_dataset / relative)
    if marker:
        candidates.append(Path(*parts[:marker]) / relative)
    elif Path(mesh_paths[0]).is_absolute():
        candidates.append(Path("/") / relative)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def support_usda_path(dataset: str, sequence_id: str) -> Path:
    """Support-surface USDA for a sequence:
    ``<assets>/human_motion_data/<dataset>/reconstructed_stage/<sequence_id>_support.usda``."""
    return HUMAN_MOTION_DATA_DIR / dataset / "reconstructed_stage" / f"{sequence_id}_support.usda"


def support_usda_for_parquet(parquet_path: str, *, robot_name: str | None = None) -> Path | None:
    """Resolve the support USDA from a parquet path
    (``.../human_motion_data/<dataset>/<stage>/sequence_id=<seq>/robot_name=...``), or None."""
    path = Path(parquet_path).expanduser().resolve()
    parents = (path, *path.parents)
    sequence_dir = next((parent for parent in parents if parent.name.startswith("sequence_id=")), None)
    path_robot_name = next(
        (parent.name.split("=", 1)[1] for parent in parents if parent.name.startswith("robot_name=")),
        None,
    )
    if sequence_dir is not None:
        sequence_id = sequence_dir.name.split("=", 1)[1]
        stage_dir = sequence_dir.parent.parent / "reconstructed_stage"
        resolved_robot_name = robot_name or path_robot_name
        candidates = []
        if resolved_robot_name:
            candidates.append(stage_dir / f"{sequence_id}_{resolved_robot_name}_support.usda")
        candidates.append(stage_dir / f"{sequence_id}_support.usda")
        match = next((candidate for candidate in candidates if candidate.is_file()), None)
        if match is not None:
            return match

    parts = path.parts
    if "human_motion_data" not in parts:
        return None
    dataset = parts[parts.index("human_motion_data") + 1]
    seq = next((p[len("sequence_id=") :] for p in parts if p.startswith("sequence_id=")), None)
    if seq is None:
        return None
    path = support_usda_path(dataset, seq)
    return path if path.exists() else None


def support_usda_for_reference(reference: Reference) -> Path | None:
    """Resolve colocated support, with source-dataset metadata as a fallback."""
    metadata = reference.metadata
    path = support_usda_for_parquet(metadata.source_path, robot_name=metadata.robot_name)
    if path is not None:
        return path
    if metadata.source_dataset is not None and metadata.sequence_id is not None:
        stage_dir = HUMAN_MOTION_DATA_DIR / metadata.source_dataset / "reconstructed_stage"
        candidates = []
        if metadata.robot_name:
            candidates.append(stage_dir / f"{metadata.sequence_id}_{metadata.robot_name}_support.usda")
        candidates.append(stage_dir / f"{metadata.sequence_id}_support.usda")
        return next((candidate for candidate in candidates if candidate.is_file()), None)
    return None


def resolve_object_asset_specs(
    *,
    source_dataset: str | None,
    object_name: str,
    body_names: list[str],
    mesh_paths: list[str],
    urdf_paths: list[str],
    num_articulations: int,
    reference_path: str | Path | None = None,
) -> tuple[ObjectAssetSpec, ...]:
    """Adapt stored asset metadata into declarative object import specifications."""
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("object reference body names must be nonempty and unique")
    if num_articulations < 0:
        raise ValueError("object articulation count must be nonnegative")

    if urdf_paths:
        if len(urdf_paths) != len(body_names):
            raise ValueError(
                f"rigid object URDF count {len(urdf_paths)} does not match {len(body_names)} reference bodies"
            )
        if num_articulations:
            raise ValueError("independent rigid object assets cannot bind articulation columns")
        return tuple(
            ObjectAssetSpec(
                name=body_name,
                urdf_path=resolve_asset_path(urdf_path, reference_path=reference_path),
                bodies=(ObjectBodySpec(reference_name=body_name, simulation_name=None),),
                root_reference_name=body_name,
            )
            for body_name, urdf_path in zip(body_names, urdf_paths, strict=True)
        )

    if source_dataset not in ("arctic", "synthbox"):
        raise ValueError(
            "references without per-body URDFs require an explicit supported asset adapter; "
            f"source_dataset={source_dataset!r}"
        )
    articulated_urdf = _v2d_articulated_urdf(
        source_dataset=source_dataset,
        object_name=object_name,
        mesh_paths=mesh_paths,
        reference_path=reference_path,
    )
    if articulated_urdf is None and source_dataset == "arctic":
        articulated_urdf = arctic_art_urdf(object_name)
    if articulated_urdf is None:
        raise ValueError(
            f"no {source_dataset} articulated URDF exists for object {object_name!r}; "
            "expected object_assets/urdfs/<dataset>/<object>.urdf beside the reference assets"
        )
    if "bottom" not in body_names:
        raise ValueError(f"{source_dataset} articulated object reference must contain the root body 'bottom'")
    if num_articulations != 1:
        raise ValueError(
            f"{source_dataset} articulated object {object_name!r} requires exactly one articulation column, "
            f"got {num_articulations}"
        )
    return (
        ObjectAssetSpec(
            name=object_name,
            urdf_path=str(articulated_urdf),
            bodies=tuple(
                ObjectBodySpec(reference_name=body_name, simulation_name=body_name) for body_name in body_names
            ),
            root_reference_name="bottom",
            articulations=(
                ObjectArticulationSpec(
                    simulation_joint_name="rotation",
                    reference_index=0,
                    physics=ObjectJointPhysicsSpec(
                        armature=0.01,
                        friction=0.1,
                    ),
                    drive=ObjectJointDriveSpec(kp=50.0, kd=2.0, effort_limit=50.0),
                ),
            ),
        ),
    )
