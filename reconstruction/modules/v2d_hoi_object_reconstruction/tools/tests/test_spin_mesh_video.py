# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import trimesh

pytest.importorskip("pyrender")

from reconstruction.modules.v2d_hoi_object_reconstruction.tools.spin_mesh_video import (
    build_render_mesh,
    load_meshes,
    repair_inverted_winding,
    should_use_flat_shading,
    uses_vertex_colors,
)


def _colored_icosphere(radius: float = 0.2) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    colors = np.empty((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, 0] = np.linspace(20, 230, len(mesh.vertices), dtype=np.uint8)
    colors[:, 1] = 80
    colors[:, 2] = 170
    colors[:, 3] = 255
    mesh.visual.vertex_colors = colors
    return mesh


def test_load_meshes_applies_scene_transform_and_preserves_colors(tmp_path) -> None:
    mesh = _colored_icosphere()
    expected_colors = np.asarray(mesh.visual.vertex_colors).copy()
    transform = np.eye(4)
    transform[:3, 3] = [1.5, -0.4, 0.7]
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="translated", transform=transform)
    path = tmp_path / "transformed.glb"
    path.write_bytes(scene.export(file_type="glb"))

    loaded = load_meshes(path)

    assert len(loaded) == 1
    np.testing.assert_allclose(loaded[0].bounds.mean(axis=0), transform[:3, 3])
    np.testing.assert_array_equal(loaded[0].visual.vertex_colors, expected_colors)


def test_repair_inverted_winding_preserves_vertex_colors() -> None:
    mesh = _colored_icosphere()
    expected_colors = np.asarray(mesh.visual.vertex_colors).copy()
    mesh.invert()
    assert mesh.is_watertight
    assert mesh.volume < 0.0

    repaired_count = repair_inverted_winding(mesh)

    assert repaired_count == 1
    assert mesh.volume > 0.0
    np.testing.assert_array_equal(mesh.visual.vertex_colors, expected_colors)


def test_repair_preserves_valid_nested_cavity_shell() -> None:
    outer = _colored_icosphere(radius=0.4)
    cavity = _colored_icosphere(radius=0.2)
    cavity.invert()
    mesh = trimesh.util.concatenate([outer, cavity])
    original_faces = mesh.faces.copy()
    original_volumes = [body.volume for body in mesh.split(only_watertight=False)]

    assert mesh.is_watertight
    assert mesh.volume > 0.0
    assert original_volumes[0] > 0.0
    assert original_volumes[1] < 0.0

    assert repair_inverted_winding(mesh) == 0
    np.testing.assert_array_equal(mesh.faces, original_faces)


def test_repair_globally_inverts_mesh_and_preserves_cavity() -> None:
    outer = _colored_icosphere(radius=0.4)
    cavity = _colored_icosphere(radius=0.2)
    cavity.invert()
    mesh = trimesh.util.concatenate([outer, cavity])
    mesh.invert()
    expected_colors = np.asarray(mesh.visual.vertex_colors).copy()

    assert mesh.is_watertight
    assert mesh.volume < 0.0

    assert repair_inverted_winding(mesh) == 1
    repaired_volumes = [body.volume for body in mesh.split(only_watertight=False)]
    assert mesh.volume > 0.0
    assert repaired_volumes[0] > 0.0
    assert repaired_volumes[1] < 0.0
    np.testing.assert_array_equal(mesh.visual.vertex_colors, expected_colors)


def test_repair_does_not_guess_orientation_for_open_mesh() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 2, 1]]),
        process=False,
    )
    faces_before = mesh.faces.copy()

    assert not repair_inverted_winding(mesh)
    np.testing.assert_array_equal(mesh.faces, faces_before)


def test_repair_skips_mesh_with_mixed_open_and_closed_components() -> None:
    closed = trimesh.creation.icosphere(subdivisions=1, radius=0.2)
    closed.invert()
    open_surface = trimesh.Trimesh(
        vertices=np.array([[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.0, 0.2, 0.0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    mesh = trimesh.util.concatenate([closed, open_surface])
    assert not mesh.is_watertight

    assert repair_inverted_winding(mesh) == 0
    closed_components = [
        component
        for component in mesh.split(only_watertight=False)
        if component.is_watertight
    ]
    assert len(closed_components) == 1
    assert closed_components[0].volume < 0.0


def test_vertex_colors_select_flat_opaque_rendering() -> None:
    mesh = _colored_icosphere()

    assert uses_vertex_colors(mesh)
    assert should_use_flat_shading([mesh], "auto")
    assert not should_use_flat_shading([mesh], "lit")
    assert not should_use_flat_shading([trimesh.creation.box()], "auto")

    render_mesh = build_render_mesh(
        mesh,
        force_opaque_vertex_colors=True,
    )
    primitive = render_mesh.primitives[0]
    assert primitive.color_0 is not None
    assert primitive.material.alphaMode == "OPAQUE"
    np.testing.assert_allclose(primitive.material.baseColorFactor, np.ones(4))
    assert primitive.material.metallicFactor == pytest.approx(0.0)
    assert primitive.material.roughnessFactor == pytest.approx(0.9)
