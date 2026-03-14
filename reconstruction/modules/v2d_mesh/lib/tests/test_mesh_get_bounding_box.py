import numpy as np
import pytest

from v2d.common.datatypes import BoundingBox3d
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_get_bounding_box import mesh_get_bounding_box


def test_returns_bounding_box3d(box_mesh):
    result = mesh_get_bounding_box(box_mesh)
    assert isinstance(result, BoundingBox3d)


def test_box_mesh_bounds(box_mesh):
    # box_mesh: 0.5 m cube centered at (0, 0, 2)
    # Expected: X in [-0.25, 0.25], Y in [-0.25, 0.25], Z in [1.75, 2.25]
    bb = mesh_get_bounding_box(box_mesh)
    assert abs(bb.x0 - (-0.25)) < 1e-6
    assert abs(bb.x1 - 0.25) < 1e-6
    assert abs(bb.y0 - (-0.25)) < 1e-6
    assert abs(bb.y1 - 0.25) < 1e-6
    assert abs(bb.z0 - 1.75) < 1e-6
    assert abs(bb.z1 - 2.25) < 1e-6


def test_min_less_than_max():
    mesh = Mesh(
        vertices=np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    bb = mesh_get_bounding_box(mesh)
    assert bb.x0 < bb.x1
    assert bb.y0 < bb.y1
    assert bb.z0 < bb.z1


def test_exact_values_from_known_vertices():
    mesh = Mesh(
        vertices=np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ], dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    bb = mesh_get_bounding_box(mesh)
    assert bb.x0 == 0.0 and bb.x1 == 3.0
    assert bb.y0 == 0.0 and bb.y1 == 4.0
    assert bb.z0 == 0.0 and bb.z1 == 5.0


def test_single_vertex_mesh():
    mesh = Mesh(
        vertices=np.array([[7.0, -3.0, 1.5]], dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    bb = mesh_get_bounding_box(mesh)
    assert bb.x0 == bb.x1 == 7.0
    assert bb.y0 == bb.y1 == -3.0
    assert bb.z0 == bb.z1 == 1.5


def test_negative_coordinates():
    mesh = Mesh(
        vertices=np.array([[-5.0, -10.0, -2.0], [-1.0, -3.0, -0.5]], dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
    )
    bb = mesh_get_bounding_box(mesh)
    assert bb.x0 == -5.0 and bb.x1 == -1.0
    assert bb.y0 == -10.0 and bb.y1 == -3.0
    assert bb.z0 == -2.0 and bb.z1 == -0.5
