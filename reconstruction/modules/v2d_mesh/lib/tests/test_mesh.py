# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
import os
import numpy as np
import trimesh
from PIL import Image as PILImage

from v2d.common.datatypes import BoundingBox3d, Image
from v2d.mesh.lib.mesh import Mesh


# ---------------------------------------------------------------------------
# Mesh round-trip: to_trimesh / from_trimesh
# ---------------------------------------------------------------------------

def test_from_trimesh_preserves_vertices(box_mesh):
    tm = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh = Mesh.from_trimesh(tm)
    np.testing.assert_allclose(mesh.vertices, tm.vertices)


def test_from_trimesh_preserves_faces(box_mesh):
    tm = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh = Mesh.from_trimesh(tm)
    np.testing.assert_array_equal(mesh.faces, tm.faces)


def test_to_trimesh_round_trip():
    tm_orig = trimesh.creation.icosphere(subdivisions=2)
    mesh = Mesh.from_trimesh(tm_orig)
    tm_back = mesh.to_trimesh()
    np.testing.assert_allclose(tm_back.vertices, tm_orig.vertices, atol=1e-9)
    np.testing.assert_array_equal(tm_back.faces, tm_orig.faces)


def test_from_trimesh_captures_vertex_colors():
    tm = trimesh.creation.box()
    n = len(tm.vertices)
    colors = np.tile([200, 100, 50, 255], (n, 1)).astype(np.uint8)
    tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=colors)
    mesh = Mesh.from_trimesh(tm)
    assert mesh.vertex_colors is not None
    assert mesh.vertex_colors.shape == (n, 4)


def test_to_trimesh_passes_vertex_colors():
    tm_orig = trimesh.creation.box()
    n = len(tm_orig.vertices)
    colors = np.tile([10, 20, 30, 255], (n, 1)).astype(np.uint8)
    tm_orig.visual = trimesh.visual.ColorVisuals(mesh=tm_orig, vertex_colors=colors)
    mesh = Mesh.from_trimesh(tm_orig)
    tm_back = mesh.to_trimesh()
    np.testing.assert_array_equal(np.array(tm_back.visual.vertex_colors)[:, :4], colors)


def test_no_vertex_colors_by_default():
    tm = trimesh.creation.box()
    # Create a mesh without explicit vertex colors
    mesh = Mesh(vertices=np.array(tm.vertices), faces=np.array(tm.faces))
    assert mesh.vertex_colors is None
    tm_back = mesh.to_trimesh()
    assert tm_back is not None


def test_from_trimesh_captures_pbr_base_color_texture():
    tm = trimesh.creation.box()
    uv = np.zeros((len(tm.vertices), 2), dtype=np.float64)
    image = PILImage.new('RGBA', (2, 2), (255, 0, 0, 255))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image)
    tm.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    mesh = Mesh.from_trimesh(tm)

    assert mesh.uv is not None
    assert mesh.texture is not None
    assert mesh.texture.shape == (2, 2, 4)


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_glb(box_mesh):
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
        path = f.name
    try:
        box_mesh.save(path)
        loaded = Mesh.load(path)
        assert len(loaded.vertices) == len(box_mesh.vertices)
        assert len(loaded.faces) == len(box_mesh.faces)
    finally:
        os.unlink(path)


def test_save_and_load_obj():
    tm = trimesh.creation.box()
    mesh = Mesh.from_trimesh(tm)
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        path = f.name
    try:
        mesh.save(path)
        loaded = Mesh.load(path)
        assert len(loaded.vertices) > 0
        assert len(loaded.faces) > 0
    finally:
        os.unlink(path)


def test_load_merges_multi_geometry_scene():
    # Build a scene with two geometries and save it as GLB.
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(), geom_name='box1')
    scene.add_geometry(trimesh.creation.icosphere(subdivisions=1), geom_name='sphere')
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
        path = f.name
    try:
        scene.export(path)
        loaded = Mesh.load(path)
        assert len(loaded.faces) > 0
    finally:
        os.unlink(path)


def test_load_applies_scene_graph_transform():
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    scene.add_geometry(trimesh.creation.box(), geom_name='box', transform=transform)
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
        path = f.name
    try:
        scene.export(path)
        loaded = Mesh.load(path)
        np.testing.assert_allclose(loaded.vertices.mean(axis=0), [1.0, 2.0, 3.0])
    finally:
        os.unlink(path)


def test_load_force_mesh_applies_scene_graph_transform():
    scene = trimesh.Scene()
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    scene.add_geometry(trimesh.creation.box(), geom_name='box', transform=transform)
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f:
        path = f.name
    try:
        scene.export(path)
        loaded = Mesh.load(path, force_mesh=True)
        np.testing.assert_allclose(loaded.vertices.mean(axis=0), [1.0, 2.0, 3.0])
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# BoundingBox3d
# ---------------------------------------------------------------------------

def test_bounding_box3d_dict_round_trip():
    bb = BoundingBox3d(x0=-1.0, y0=-2.0, z0=-3.0, x1=1.0, y1=2.0, z1=3.0)
    assert BoundingBox3d.from_dict(bb.to_dict()) == bb


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def test_rgb_image_pil_round_trip():
    data = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    img = Image(data=data)
    pil = img.to_pil_image()
    back = Image.from_pil_image(pil)
    np.testing.assert_array_equal(back.data, data)


def test_rgb_image_dimensions():
    data = np.zeros((48, 64, 3), dtype=np.uint8)
    img = Image(data=data)
    assert img.width == 64
    assert img.height == 48
