# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic explicit collider generation and rigid mass properties."""

from __future__ import annotations

import hashlib
from importlib.metadata import version as package_version
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import trimesh

from geometry import matrix_to_quaternion_wxyz


MIN_COLLIDER_VOLUME_M3 = 1e-15


@dataclass(frozen=True)
class ColliderBuildConfig:
    coacd_threshold: float = 0.05
    max_convex_hulls: int = 16
    coacd_resolution: int = 2000
    max_decomposition_source_faces: int = 20_000
    simplify_decomposition_source: bool = True
    coacd_mcts_nodes: int = 20
    coacd_mcts_iterations: int = 150
    coacd_mcts_max_depth: int = 3
    max_hull_vertices: int = 64
    coacd_seed: int = 0
    canonical_decimals: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.coacd_threshold <= 1:
            raise ValueError("coacd_threshold must be in (0, 1]")
        integer_values = (
            self.max_convex_hulls,
            self.coacd_resolution,
            self.max_decomposition_source_faces,
            self.coacd_mcts_nodes,
            self.coacd_mcts_iterations,
            self.coacd_mcts_max_depth,
            self.max_hull_vertices,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("collider generation limits must be positive")


@dataclass(frozen=True)
class ColliderPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    volume_m3: float
    center_of_mass_m: tuple[float, float, float]
    sha256: str


@dataclass(frozen=True)
class DiscardedColliderPart:
    source_index: int
    reason: str
    vertex_count: int
    face_count: int
    volume_m3: float
    extents_m: tuple[float, float, float]


@dataclass(frozen=True)
class ExplicitColliderSet:
    collider_type: str
    source_mesh_sha256: str
    generator: str
    generator_version: str
    config: ColliderBuildConfig
    parts: tuple[ColliderPart, ...]
    discarded_parts: tuple[DiscardedColliderPart, ...] = ()


@dataclass(frozen=True)
class RigidMassProperties:
    mass_kg: float
    volume_m3: float
    density_kg_m3: float
    center_of_mass_m: tuple[float, float, float]
    diagonal_inertia_kg_m2: tuple[float, float, float]
    principal_axes_wxyz: tuple[float, float, float, float]
    inertia_tensor_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


def _as_repaired_mesh(vertices: np.ndarray, faces: np.ndarray):
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError("Collider mesh must contain at least four vertices and faces")
    if not mesh.is_watertight or not mesh.is_volume:
        mesh = mesh.convex_hull
    if float(mesh.volume) < 0:
        mesh.invert()
    if not mesh.is_watertight:
        raise ValueError("Collider mesh must be watertight")
    return mesh


def _as_mesh(vertices: np.ndarray, faces: np.ndarray):
    mesh = _as_repaired_mesh(vertices, faces)
    if (
        not math.isfinite(float(mesh.volume))
        or float(mesh.volume) <= MIN_COLLIDER_VOLUME_M3
    ):
        raise ValueError("Collider mesh must be a positive watertight volume")
    return mesh


def _as_source_mesh(vertices: np.ndarray, faces: np.ndarray):
    """Load source triangles without replacing open geometry by its convex hull."""

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=False,
    )
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError("Source mesh must contain at least four vertices and faces")
    return mesh


def _array_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(vertices, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(faces, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _canonical_arrays(
    vertices: np.ndarray, faces: np.ndarray, *, decimals: int
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.round(np.asarray(vertices, dtype=np.float64), decimals=decimals)
    faces = np.asarray(faces, dtype=np.int64)
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    vertices = vertices[order]
    faces = inverse[faces]

    canonical_faces = []
    for face in faces:
        offset = int(np.argmin(face))
        canonical_faces.append(np.roll(face, -offset))
    faces = np.asarray(canonical_faces, dtype=np.int64)
    face_order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    return vertices, faces[face_order]


def _canonical_part(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    name: str,
    decimals: int,
) -> ColliderPart:
    mesh = _as_mesh(vertices, faces)
    vertices, faces = _canonical_arrays(mesh.vertices, mesh.faces, decimals=decimals)
    mesh = _as_mesh(vertices, faces)
    properties = mesh.mass_properties
    return ColliderPart(
        name=name,
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        volume_m3=float(properties.volume),
        center_of_mass_m=tuple(float(value) for value in properties.center_mass),
        sha256=_array_hash(mesh.vertices, mesh.faces),
    )


def _source_hash(vertices: np.ndarray, faces: np.ndarray, decimals: int) -> str:
    canonical_vertices, canonical_faces = _canonical_arrays(
        vertices, faces, decimals=decimals
    )
    return _array_hash(canonical_vertices, canonical_faces)


def build_explicit_colliders(
    vertices: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray,
    collider_type: str,
    config: ColliderBuildConfig = ColliderBuildConfig(),
) -> ExplicitColliderSet:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    source = _as_source_mesh(vertices, faces)
    if collider_type != "convexDecomposition":
        raise ValueError("Only convexDecomposition is supported")

    import coacd

    generator = "coacd"
    decomposition_source = source
    if (
        config.simplify_decomposition_source
        and len(source.faces) > config.max_decomposition_source_faces
    ):
        decomposition_source = source.simplify_quadric_decimation(
            face_count=config.max_decomposition_source_faces, aggression=7
        )
        decomposition_source = _as_source_mesh(
            decomposition_source.vertices, decomposition_source.faces
        )
    decomposition = coacd.run_coacd(
        coacd.Mesh(
            np.asarray(decomposition_source.vertices, dtype=np.float64),
            np.asarray(decomposition_source.faces, dtype=np.int32),
        ),
        threshold=config.coacd_threshold,
        max_convex_hull=config.max_convex_hulls,
        resolution=config.coacd_resolution,
        mcts_nodes=config.coacd_mcts_nodes,
        mcts_iterations=config.coacd_mcts_iterations,
        mcts_max_depth=config.coacd_mcts_max_depth,
        max_ch_vertex=config.max_hull_vertices,
        seed=config.coacd_seed,
        merge=True,
        decimate=True,
    )
    raw_parts = [
        (np.asarray(part_vertices), np.asarray(part_faces))
        for part_vertices, part_faces in decomposition
    ]
    if not raw_parts:
        raise RuntimeError("CoACD returned no collider parts")

    retained_parts = []
    discarded_parts = []
    for source_index, (part_vertices, part_faces) in enumerate(raw_parts):
        mesh = _as_repaired_mesh(part_vertices, part_faces)
        volume_m3 = float(mesh.volume)
        if (
            math.isfinite(volume_m3)
            and 0.0 < volume_m3 <= MIN_COLLIDER_VOLUME_M3
        ):
            discarded_parts.append(
                DiscardedColliderPart(
                    source_index=source_index,
                    reason="volume_at_or_below_minimum",
                    vertex_count=len(mesh.vertices),
                    face_count=len(mesh.faces),
                    volume_m3=volume_m3,
                    extents_m=tuple(float(value) for value in mesh.extents),
                )
            )
            continue
        retained_parts.append((mesh.vertices, mesh.faces))

    if not retained_parts:
        raise RuntimeError(
            "CoACD returned no collider parts above the minimum volume "
            f"({MIN_COLLIDER_VOLUME_M3:.3e} m^3)"
        )

    provisional = [
        _canonical_part(
            part_vertices,
            part_faces,
            name="pending",
            decimals=config.canonical_decimals,
        )
        for part_vertices, part_faces in retained_parts
    ]
    provisional.sort(
        key=lambda part: (
            -round(part.volume_m3, 12),
            *tuple(round(value, 10) for value in part.center_of_mass_m),
            part.sha256,
        )
    )
    parts = tuple(
        ColliderPart(
            name=f"Hull_{index:03d}",
            vertices=part.vertices,
            faces=part.faces,
            volume_m3=part.volume_m3,
            center_of_mass_m=part.center_of_mass_m,
            sha256=part.sha256,
        )
        for index, part in enumerate(provisional)
    )

    return ExplicitColliderSet(
        collider_type=collider_type,
        source_mesh_sha256=_source_hash(
            source.vertices, source.faces, config.canonical_decimals
        ),
        generator=generator,
        generator_version=(
            f"coacd={package_version('coacd')};"
            f"fast_simplification={package_version('fast-simplification')};"
            f"trimesh={package_version('trimesh')}"
        ),
        config=config,
        parts=parts,
        discarded_parts=tuple(discarded_parts),
    )


def calculate_mass_properties(
    colliders: ExplicitColliderSet, mass_kg: float
) -> RigidMassProperties:
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass_kg must be positive and finite")
    meshes = [_as_mesh(part.vertices, part.faces) for part in colliders.parts]
    properties = [mesh.mass_properties for mesh in meshes]
    volumes = np.asarray([float(value.volume) for value in properties])
    total_volume = float(volumes.sum())
    if total_volume <= 1e-15:
        raise ValueError("Collider volume must be positive")
    centers = np.asarray([value.center_mass for value in properties], dtype=np.float64)
    center = (volumes[:, None] * centers).sum(axis=0) / total_volume

    inertia_unit_density = np.zeros((3, 3), dtype=np.float64)
    identity = np.eye(3)
    for volume, part_center, value in zip(volumes, centers, properties, strict=True):
        offset = part_center - center
        inertia_unit_density += np.asarray(value.inertia, dtype=np.float64)
        inertia_unit_density += volume * (
            float(np.dot(offset, offset)) * identity - np.outer(offset, offset)
        )

    density = mass_kg / total_volume
    inertia = density * inertia_unit_density
    inertia = 0.5 * (inertia + inertia.T)
    diagonal, axes = np.linalg.eigh(inertia)
    order = np.argsort(diagonal)
    diagonal = diagonal[order]
    axes = axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        raise ValueError("Calculated inertia must be positive and finite")

    return RigidMassProperties(
        mass_kg=float(mass_kg),
        volume_m3=total_volume,
        density_kg_m3=float(density),
        center_of_mass_m=tuple(float(value) for value in center),
        diagonal_inertia_kg_m2=tuple(float(value) for value in diagonal),
        principal_axes_wxyz=matrix_to_quaternion_wxyz(axes),
        inertia_tensor_kg_m2=tuple(
            tuple(float(value) for value in row) for row in inertia
        ),
    )


def collider_set_to_report(colliders: ExplicitColliderSet) -> dict:
    return {
        "collider_type": colliders.collider_type,
        "source_mesh_sha256": colliders.source_mesh_sha256,
        "generator": colliders.generator,
        "generator_version": colliders.generator_version,
        "config": asdict(colliders.config),
        "part_filter": {
            "minimum_retained_volume_m3": MIN_COLLIDER_VOLUME_M3,
            "input_part_count": len(colliders.parts) + len(colliders.discarded_parts),
            "retained_part_count": len(colliders.parts),
            "discarded_part_count": len(colliders.discarded_parts),
            "discarded_parts": [
                asdict(part) for part in colliders.discarded_parts
            ],
        },
        "parts": [
            {
                "name": part.name,
                "vertex_count": len(part.vertices),
                "face_count": len(part.faces),
                "volume_m3": part.volume_m3,
                "center_of_mass_m": part.center_of_mass_m,
                "sha256": part.sha256,
            }
            for part in colliders.parts
        ],
    }
