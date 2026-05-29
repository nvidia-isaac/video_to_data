# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
# Must be set before any pyrender/OpenGL/pyglet import.
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import pyglet
pyglet.options['headless'] = True

import pathlib
import numpy as np
import pytest
import trimesh

from v2d.common.datatypes import CameraIntrinsics, Image, Transform3d
from v2d.mesh.lib.mesh import Mesh

ASSETS_DIR = pathlib.Path(__file__).parent.parent.parent / "assets"


@pytest.fixture
def camera() -> CameraIntrinsics:
    """640x480 pinhole camera with 500px focal length."""
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


@pytest.fixture
def box_mesh() -> Mesh:
    """
    Unit cube (0.5 m extents) centered at (0, 0, 2).

    In the OpenCV camera convention the cube is directly in front of the camera:
      - front face at Z = 1.75
      - back  face at Z = 2.25
      - XY footprint: [-0.25, 0.25] × [-0.25, 0.25]
    """
    tm = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    tm.apply_translation([0.0, 0.0, 2.0])
    return Mesh.from_trimesh(tm)


@pytest.fixture
def colored_box_mesh() -> Mesh:
    """Unit cube at (0, 0, 2) with solid red vertex colors."""
    tm = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    tm.apply_translation([0.0, 0.0, 2.0])
    n = len(tm.vertices)
    red = np.tile([255, 0, 0, 255], (n, 1)).astype(np.uint8)
    tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=red)
    return Mesh.from_trimesh(tm)


@pytest.fixture
def sphere_mesh() -> Mesh:
    """Medium-density icosphere at the origin — useful for simplification tests."""
    tm = trimesh.creation.icosphere(subdivisions=3)
    tm.apply_translation([0.0, 0.0, 2.0])
    return Mesh.from_trimesh(tm)


@pytest.fixture
def identity_transform() -> Transform3d:
    return Transform3d(rotation=[1.0, 0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0])


@pytest.fixture
def translation_transform() -> Transform3d:
    """Pure translation of (1, 2, 3)."""
    return Transform3d(rotation=[1.0, 0.0, 0.0, 0.0], translation=[1.0, 2.0, 3.0], scale=[1.0, 1.0, 1.0])


@pytest.fixture
def scale_transform() -> Transform3d:
    """Uniform scale of 2×."""
    return Transform3d(rotation=[1.0, 0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0], scale=[2.0, 2.0, 2.0])


@pytest.fixture
def test_image(camera) -> Image:
    """Assets test image resized to match the camera dimensions."""
    from PIL import Image as PILImage
    pil = PILImage.open(ASSETS_DIR / "test_image.jpg").convert("RGB").resize((camera.width, camera.height))
    return Image.from_pil_image(pil)
