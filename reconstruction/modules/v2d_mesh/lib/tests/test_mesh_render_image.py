# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

pyrender = pytest.importorskip('pyrender', reason='pyrender not installed', exc_type=ImportError)

from v2d.common.datatypes import CameraIntrinsics
from v2d.common.datatypes import Image
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_image import mesh_render_image


def test_returns_rgb_image(colored_box_mesh, camera):
    result = mesh_render_image(colored_box_mesh, camera)
    assert isinstance(result, Image)


def test_output_shape_matches_intrinsics(colored_box_mesh, camera):
    result = mesh_render_image(colored_box_mesh, camera)
    assert result.data.shape == (camera.height, camera.width, 3)


def test_output_dtype_is_uint8(colored_box_mesh, camera):
    result = mesh_render_image(colored_box_mesh, camera)
    assert result.data.dtype == np.uint8


def test_has_nonzero_pixels(colored_box_mesh, camera):
    result = mesh_render_image(colored_box_mesh, camera)
    assert np.any(result.data > 0)


def test_background_is_black(colored_box_mesh, camera):
    result = mesh_render_image(colored_box_mesh, camera)
    # Corner should be background (black).
    assert np.all(result.data[0, 0] == 0)


def test_width_and_height_properties(colored_box_mesh, camera):
    result = mesh_render_image(colored_box_mesh, camera)
    assert result.width == camera.width
    assert result.height == camera.height


def test_pil_round_trip(colored_box_mesh, camera):
    result = mesh_render_image(colored_box_mesh, camera)
    pil_img = result.to_pil_image()
    back = Image.from_pil_image(pil_img)
    np.testing.assert_array_equal(back.data, result.data)


# --- background compositing ---

def test_background_returns_image(colored_box_mesh, camera, test_image):
    result = mesh_render_image(colored_box_mesh, camera, background=test_image)
    assert isinstance(result, Image)


def test_background_shape_matches_intrinsics(colored_box_mesh, camera, test_image):
    result = mesh_render_image(colored_box_mesh, camera, background=test_image)
    assert result.data.shape == (camera.height, camera.width, 3)


def test_background_visible_at_corner(colored_box_mesh, camera, test_image):
    """Corner pixel has no mesh coverage — background should show through exactly."""
    result = mesh_render_image(colored_box_mesh, camera, background=test_image)
    np.testing.assert_array_equal(result.data[0, 0], test_image.data[0, 0])


def test_background_modified_by_mesh(colored_box_mesh, camera, test_image):
    """Result should differ from the background where the mesh is rendered."""
    result = mesh_render_image(colored_box_mesh, camera, background=test_image)
    assert not np.array_equal(result.data, test_image.data)


def test_no_background_corner_is_black(colored_box_mesh, camera, test_image):
    """Without background, corner pixel stays black; with background it takes the bg value."""
    no_bg = mesh_render_image(colored_box_mesh, camera)
    with_bg = mesh_render_image(colored_box_mesh, camera, background=test_image)
    assert np.all(no_bg.data[0, 0] == 0)
    np.testing.assert_array_equal(with_bg.data[0, 0], test_image.data[0, 0])
