# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import trimesh

from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_simplify import mesh_simplify


def test_simplify_reduces_face_count(sphere_mesh):
    original_faces = len(sphere_mesh.faces)
    result = mesh_simplify(sphere_mesh, factor=0.25)
    assert len(result.faces) < original_faces


def test_simplify_face_count_arg(sphere_mesh):
    target = 50
    result = mesh_simplify(sphere_mesh, face_count=target)
    # Quadric decimation may not hit exactly target, but should be in the ballpark.
    assert len(result.faces) <= target * 2


def test_simplify_factor_arg(sphere_mesh):
    original = len(sphere_mesh.faces)
    result = mesh_simplify(sphere_mesh, factor=0.1)
    assert len(result.faces) < original * 0.5


def test_simplify_default_is_ten_percent(sphere_mesh):
    original = len(sphere_mesh.faces)
    result = mesh_simplify(sphere_mesh)
    assert len(result.faces) < original * 0.5


def test_simplify_face_count_takes_priority(sphere_mesh):
    target = 40
    result = mesh_simplify(sphere_mesh, face_count=target, factor=0.9)
    assert len(result.faces) <= target * 2


def test_simplify_returns_mesh_type(sphere_mesh):
    result = mesh_simplify(sphere_mesh, factor=0.5)
    assert isinstance(result, Mesh)


def test_simplify_preserves_vertex_colors():
    tm = trimesh.creation.icosphere(subdivisions=3)
    tm.apply_translation([0.0, 0.0, 2.0])
    n = len(tm.vertices)
    colors = np.tile([255, 128, 0, 255], (n, 1)).astype(np.uint8)
    tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=colors)
    mesh = Mesh.from_trimesh(tm)

    result = mesh_simplify(mesh, factor=0.2)
    assert result.vertex_colors is not None
    assert result.vertex_colors.shape[0] == len(result.vertices)


def test_simplify_does_not_mutate_input(sphere_mesh):
    original_face_count = len(sphere_mesh.faces)
    mesh_simplify(sphere_mesh, factor=0.1)
    assert len(sphere_mesh.faces) == original_face_count
