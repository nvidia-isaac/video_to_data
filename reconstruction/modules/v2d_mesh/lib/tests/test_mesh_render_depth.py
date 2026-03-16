import numpy as np
import pytest

pyrender = pytest.importorskip('pyrender', reason='pyrender not installed', exc_type=ImportError)

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_depth import mesh_render_depth


def test_returns_depth_image(box_mesh, camera):
    result = mesh_render_depth(box_mesh, camera)
    assert isinstance(result, DepthImage)


def test_output_shape_matches_intrinsics(box_mesh, camera):
    result = mesh_render_depth(box_mesh, camera)
    assert result.depth.shape == (camera.height, camera.width)


def test_depth_dtype_is_float32(box_mesh, camera):
    result = mesh_render_depth(box_mesh, camera)
    assert result.depth.dtype == np.float32


def test_background_pixels_are_zero(box_mesh, camera):
    result = mesh_render_depth(box_mesh, camera)
    # Corner pixel (0, 0) should be background for the small centered box.
    assert result.depth[0, 0] == 0.0


def test_center_pixel_depth_is_correct(box_mesh, camera):
    # box_mesh: 0.5 m cube centered at (0, 0, 2); front face at Z = 1.75
    result = mesh_render_depth(box_mesh, camera)
    center_depth = result.depth[camera.height // 2, camera.width // 2]
    assert abs(center_depth - 1.75) < 0.05, f"Expected ~1.75, got {center_depth}"


def test_depth_positive_where_mesh_visible(box_mesh, camera):
    result = mesh_render_depth(box_mesh, camera)
    assert np.any(result.depth > 0)


def test_mesh_further_away_has_greater_depth(camera):
    import trimesh
    near = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    near.apply_translation([0.0, 0.0, 1.0])
    far = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    far.apply_translation([0.0, 0.0, 3.0])

    near_depth = mesh_render_depth(Mesh.from_trimesh(near), camera)
    far_depth = mesh_render_depth(Mesh.from_trimesh(far), camera)

    cy, cx = camera.height // 2, camera.width // 2
    assert near_depth.depth[cy, cx] < far_depth.depth[cy, cx]
