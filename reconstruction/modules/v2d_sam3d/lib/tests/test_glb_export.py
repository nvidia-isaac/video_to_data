# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct

import numpy as np
import trimesh

from v2d.sam3d.lib.glb_export import (
    export_mesh,
    normalize_outward_winding,
)


def _single_mesh(path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="scene", process=False)
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) == 1
    return next(iter(scene.geometry.values()))


def _colored_icosphere(radius: float = 0.2) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    colors = np.empty((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, 0] = np.linspace(10, 240, len(mesh.vertices), dtype=np.uint8)
    colors[:, 1] = 80
    colors[:, 2] = 170
    colors[:, 3] = 255
    mesh.visual.vertex_colors = colors
    return mesh


def _glb_tree(path) -> dict:
    data = path.read_bytes()
    json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    assert chunk_type == b"JSON"
    return json.loads(data[20 : 20 + json_length].decode("utf-8"))


def test_export_corrects_inward_winding_and_preserves_vertex_colors(tmp_path) -> None:
    mesh = _colored_icosphere()
    expected_colors = np.asarray(mesh.visual.vertex_colors).copy()
    mesh.invert()
    assert mesh.is_watertight
    assert mesh.volume < 0.0
    source_scene = trimesh.Scene(mesh)

    output = tmp_path / "mesh.glb"
    export_mesh(source_scene, output)

    exported = _single_mesh(output)
    assert exported.is_watertight
    assert exported.volume > 0.0
    np.testing.assert_array_equal(exported.visual.vertex_colors, expected_colors)
    # Export works on a copy rather than mutating the inference result.
    assert next(iter(source_scene.geometry.values())).volume < 0.0


def test_export_encodes_fully_opaque_vertex_colors_as_rgb(tmp_path) -> None:
    source = _colored_icosphere()
    expected_colors = np.asarray(source.visual.vertex_colors).copy()
    output = tmp_path / "mesh.glb"
    export_mesh(trimesh.Scene(source), output)

    tree = _glb_tree(output)
    primitive = tree["meshes"][0]["primitives"][0]
    color_accessor = tree["accessors"][primitive["attributes"]["COLOR_0"]]
    exported = _single_mesh(output)

    assert "COLOR_0" in primitive["attributes"]
    assert color_accessor["type"] == "VEC3"
    assert "materials" not in tree
    np.testing.assert_array_equal(exported.visual.vertex_colors, expected_colors)


def test_export_preserves_nonopaque_vertex_alpha(tmp_path) -> None:
    source = _colored_icosphere()
    colors = np.asarray(source.visual.vertex_colors).copy()
    colors[0, 3] = 128
    source.visual.vertex_colors = colors
    output = tmp_path / "mesh.glb"

    export_mesh(trimesh.Scene(source), output)

    tree = _glb_tree(output)
    primitive = tree["meshes"][0]["primitives"][0]
    color_accessor = tree["accessors"][primitive["attributes"]["COLOR_0"]]
    exported = _single_mesh(output)

    assert color_accessor["type"] == "VEC4"
    np.testing.assert_array_equal(exported.visual.vertex_colors, colors)


def test_repair_preserves_valid_nested_cavity_shell() -> None:
    outer = _colored_icosphere(radius=0.4)
    cavity = _colored_icosphere(radius=0.2)
    cavity.invert()
    mesh = trimesh.util.concatenate([outer, cavity])
    expected_colors = np.asarray(mesh.visual.vertex_colors).copy()
    original_faces = mesh.faces.copy()
    original_volumes = [body.volume for body in mesh.split(only_watertight=False)]

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0.0
    assert original_volumes[0] > 0.0
    assert original_volumes[1] < 0.0

    repaired = normalize_outward_winding(mesh)
    repaired_volumes = [body.volume for body in repaired.split(only_watertight=False)]

    np.testing.assert_allclose(repaired_volumes, original_volumes)
    np.testing.assert_array_equal(repaired.faces, original_faces)
    np.testing.assert_array_equal(repaired.visual.vertex_colors, expected_colors)


def test_repair_globally_inverts_closed_mesh_and_preserves_cavity() -> None:
    outer = _colored_icosphere(radius=0.4)
    cavity = _colored_icosphere(radius=0.2)
    cavity.invert()
    mesh = trimesh.util.concatenate([outer, cavity])
    mesh.invert()
    expected_colors = np.asarray(mesh.visual.vertex_colors).copy()
    original_faces = mesh.faces.copy()

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume < 0.0

    repaired = normalize_outward_winding(mesh)
    repaired_volumes = [body.volume for body in repaired.split(only_watertight=False)]

    assert repaired.volume > 0.0
    assert repaired_volumes[0] > 0.0
    assert repaired_volumes[1] < 0.0
    np.testing.assert_array_equal(repaired.visual.vertex_colors, expected_colors)
    np.testing.assert_array_equal(mesh.faces, original_faces)


def test_outward_mesh_is_not_rewritten() -> None:
    mesh = _colored_icosphere()
    expected_faces = mesh.faces.copy()

    repaired = normalize_outward_winding(mesh)

    np.testing.assert_array_equal(repaired.faces, expected_faces)


def test_open_mesh_is_not_globally_inverted() -> None:
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    assert not mesh.is_watertight
    expected_faces = mesh.faces.copy()

    repaired = normalize_outward_winding(mesh)

    np.testing.assert_array_equal(repaired.faces, expected_faces)
