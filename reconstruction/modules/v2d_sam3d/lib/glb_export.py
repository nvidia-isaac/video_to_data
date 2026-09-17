#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for exporting SAM3D meshes with outward-facing surfaces."""

from __future__ import annotations

from collections import OrderedDict
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from trimesh.exchange.gltf import export_glb


def _drop_fully_opaque_vertex_alpha(
    buffer_items: OrderedDict,
    tree: dict[str, Any],
) -> None:
    """Encode fully opaque vertex colors as RGB instead of redundant RGBA.

    Trimesh normally writes ``COLOR_0`` as RGBA even when every alpha value is
    255.  Some viewers, including view3dscene, then show angle-dependent
    transparency artifacts for vertex-color-only GLBs.  A VEC3 color accessor
    is the glTF-native representation for an opaque vertex-colored primitive;
    readers synthesize alpha=1.0.  Preserve VEC4 whenever real transparency is
    present.
    """

    accessor_map = tree.get("accessors", {})
    if not accessor_map:
        return
    accessors = list(accessor_map.values())
    buffer_keys = list(buffer_items.keys())

    color_accessor_indices = {
        primitive.get("attributes", {}).get("COLOR_0")
        for mesh in tree.get("meshes", [])
        for primitive in mesh.get("primitives", [])
    } - {None}

    accessors_by_view: dict[int, set[int]] = {}
    for index, accessor in enumerate(accessors):
        view_index = accessor.get("bufferView")
        if view_index is not None:
            accessors_by_view.setdefault(int(view_index), set()).add(index)

    converted_views: dict[int, int] = {}
    for accessor_index in sorted(color_accessor_indices):
        accessor = accessors[int(accessor_index)]
        if (
            accessor.get("componentType") != 5121
            or accessor.get("type") != "VEC4"
            or int(accessor.get("byteOffset", 0)) != 0
        ):
            continue

        count = int(accessor["count"])
        source_view = int(accessor["bufferView"])
        source_key = buffer_keys[source_view]
        source = buffer_items[source_key]
        rgba = np.frombuffer(source, dtype=np.uint8, count=count * 4).reshape(
            count, 4
        )
        if not np.all(rgba[:, 3] == 255):
            continue

        rgb = rgba[:, :3].copy()
        packed = rgb.tobytes()
        packed += b"\x00" * ((-len(packed)) % 4)

        # Trimesh normally gives this accessor an independent buffer view.  If
        # another accessor happens to share it, append a new view rather than
        # changing unrelated data in place.
        if accessors_by_view[source_view] == {int(accessor_index)}:
            buffer_items[source_key] = packed
            target_view = source_view
        elif source_view in converted_views:
            target_view = converted_views[source_view]
        else:
            target_view = len(buffer_items)
            unique_key = ("opaque_rgb", int(accessor_index), target_view)
            buffer_items[unique_key] = packed
            converted_views[source_view] = target_view

        accessor["bufferView"] = target_view
        accessor["type"] = "VEC3"
        accessor["min"] = rgb.min(axis=0).tolist()
        accessor["max"] = rgb.max(axis=0).tolist()


def _needs_normal_repair(mesh: trimesh.Trimesh) -> bool:
    """Return whether a mesh has inconsistent or globally inverted triangles.

    A closed mesh may contain a positive outer shell and negative nested cavity
    shells.  Those relative signs are valid, so orientation is evaluated from
    the signed volume of the complete closed geometry rather than by requiring
    every connected shell to have positive volume.
    """

    if not mesh.is_winding_consistent:
        return True
    if not mesh.is_watertight:
        return False
    try:
        signed_volume = float(mesh.volume)
    except (TypeError, ValueError):
        return False
    return math.isfinite(signed_volume) and signed_volume < 0.0


def _orient_mesh_outward(mesh: trimesh.Trimesh) -> bool:
    """Repair winding and outward direction only when validation finds a defect."""

    if not _needs_normal_repair(mesh):
        return False
    # The default whole-mesh inversion preserves the relative orientation of
    # nested shells: an outer boundary stays opposite to an inner cavity.  The
    # multibody mode would incorrectly turn every valid cavity volume positive.
    # Open geometry receives only local winding repair because signed volume is
    # not a reliable indication of its outside.
    trimesh.repair.fix_normals(mesh, multibody=False)
    return True


def normalize_outward_winding(mesh_scene: Any) -> Any:
    """Return a copy whose closed outer boundaries use outward face winding.

    SAM3D normally returns a :class:`trimesh.Scene`, but accepting an individual
    ``Trimesh`` keeps the exporter useful for direct callers and unit tests.
    Valid negative-volume cavity shells are preserved.  Open geometry is never
    globally inverted from signed volume because it does not reliably identify
    its outside; only inconsistent local winding is repaired.
    """

    normalized = mesh_scene.copy()
    if isinstance(normalized, trimesh.Trimesh):
        _orient_mesh_outward(normalized)
        return normalized
    if isinstance(normalized, trimesh.Scene):
        for geometry in normalized.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                _orient_mesh_outward(geometry)
    return normalized


def export_mesh(mesh_scene: Any, mesh_path: str | Path) -> None:
    """Export a SAM3D mesh after repairing a globally inverted closed mesh."""

    output = Path(mesh_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_outward_winding(mesh_scene)

    if output.suffix.lower() == ".glb":
        glb = export_glb(
            normalized,
            buffer_postprocessor=_drop_fully_opaque_vertex_alpha,
        )
        if not isinstance(glb, bytes):
            raise TypeError("trimesh GLB export did not return bytes")
        output.write_bytes(glb)
        return

    # Passing a path lets trimesh write the main OBJ and any companion files
    # (MTL and textures) together with valid relative references.
    normalized.export(str(output))
