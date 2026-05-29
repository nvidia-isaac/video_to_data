# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import numpy as np
import pytest

pyrender = pytest.importorskip('pyrender', reason='pyrender not installed', exc_type=ImportError)

from v2d.common.datatypes import CameraIntrinsics, Mask
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_mask import mesh_render_mask


def test_returns_mask(box_mesh, camera):
    result = mesh_render_mask(box_mesh, camera)
    assert isinstance(result, Mask)


def test_output_shape_matches_intrinsics(box_mesh, camera):
    result = mesh_render_mask(box_mesh, camera)
    assert result.mask.shape == (camera.height, camera.width)


def test_mask_values_are_binary(box_mesh, camera):
    result = mesh_render_mask(box_mesh, camera)
    unique = np.unique(result.mask)
    assert set(unique).issubset({0.0, 1.0})


def test_center_pixel_is_foreground(box_mesh, camera):
    result = mesh_render_mask(box_mesh, camera)
    assert result.mask[camera.height // 2, camera.width // 2] == 1.0


def test_corner_pixel_is_background(box_mesh, camera):
    result = mesh_render_mask(box_mesh, camera)
    assert result.mask[0, 0] == 0.0


def test_mask_has_foreground_pixels(box_mesh, camera):
    result = mesh_render_mask(box_mesh, camera)
    assert np.sum(result.mask) > 0


def test_larger_mesh_produces_larger_mask(camera):
    import trimesh

    small = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    small.apply_translation([0.0, 0.0, 2.0])
    large = trimesh.creation.box(extents=[0.8, 0.8, 0.1])
    large.apply_translation([0.0, 0.0, 2.0])

    small_mask = mesh_render_mask(Mesh.from_trimesh(small), camera)
    large_mask = mesh_render_mask(Mesh.from_trimesh(large), camera)

    assert np.sum(large_mask.mask) > np.sum(small_mask.mask)
