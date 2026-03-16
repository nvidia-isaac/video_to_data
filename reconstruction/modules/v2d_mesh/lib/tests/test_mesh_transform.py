import numpy as np
import pytest

from v2d.common.datatypes import Transform3d
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_transform import mesh_transform


def test_identity_transform_leaves_vertices_unchanged(box_mesh, identity_transform):
    result = mesh_transform(box_mesh, identity_transform)
    np.testing.assert_allclose(result.vertices, box_mesh.vertices, atol=1e-9)


def test_translation_shifts_vertices(box_mesh, translation_transform):
    result = mesh_transform(box_mesh, translation_transform)
    expected = box_mesh.vertices + np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.vertices, expected, atol=1e-9)


def test_scale_multiplies_vertices(box_mesh, scale_transform):
    result = mesh_transform(box_mesh, scale_transform)
    np.testing.assert_allclose(result.vertices, box_mesh.vertices * 2.0, atol=1e-9)


def test_90_degree_rotation_around_z():
    # 90° rotation around Z: [w=cos45°, x=0, y=0, z=sin45°]
    angle = np.pi / 2
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    t = Transform3d(rotation=[c, 0.0, 0.0, s], translation=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0])
    # A vertex at (1, 0, 0) should rotate to (0, 1, 0).
    mesh = Mesh(
        vertices=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    result = mesh_transform(mesh, t)
    np.testing.assert_allclose(result.vertices[0], [0.0, 1.0, 0.0], atol=1e-9)


def test_faces_are_preserved(box_mesh, translation_transform):
    result = mesh_transform(box_mesh, translation_transform)
    np.testing.assert_array_equal(result.faces, box_mesh.faces)


def test_vertex_colors_are_preserved(colored_box_mesh, translation_transform):
    result = mesh_transform(colored_box_mesh, translation_transform)
    np.testing.assert_array_equal(result.vertex_colors, colored_box_mesh.vertex_colors)


def test_transform_does_not_mutate_input(box_mesh, translation_transform):
    original_vertices = box_mesh.vertices.copy()
    mesh_transform(box_mesh, translation_transform)
    np.testing.assert_array_equal(box_mesh.vertices, original_vertices)


def test_anisotropic_scale():
    t = Transform3d(rotation=[1.0, 0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0], scale=[1.0, 2.0, 3.0])
    mesh = Mesh(
        vertices=np.array([[1.0, 1.0, 1.0]], dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    result = mesh_transform(mesh, t)
    np.testing.assert_allclose(result.vertices[0], [1.0, 2.0, 3.0], atol=1e-9)
