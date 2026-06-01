# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

from v2d.common.datatypes import Transform3d
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.run_mesh_transform import run_mesh_transform


def _save_mesh(mesh: Mesh, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mesh.save(path)


def _save_transform(transform: Transform3d, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    transform.save(path)


def test_single_mesh_single_transform(tmp_path, box_mesh, translation_transform):
    mesh_p = str(tmp_path / "mesh.glb")
    transform_p = str(tmp_path / "transform.json")
    out_p = str(tmp_path / "out.glb")
    _save_mesh(box_mesh, mesh_p)
    _save_transform(translation_transform, transform_p)

    run_mesh_transform(mesh_p, transform_p, out_p)

    result = Mesh.load(out_p)
    np.testing.assert_allclose(result.vertices, box_mesh.vertices + [1, 2, 3], atol=1e-6)


def test_broadcast_1_mesh_n_transforms(tmp_path, box_mesh, translation_transform, scale_transform):
    _save_mesh(box_mesh, str(tmp_path / "mesh.glb"))
    _save_transform(translation_transform, str(tmp_path / "transforms/000000.json"))
    _save_transform(scale_transform,       str(tmp_path / "transforms/000001.json"))

    run_mesh_transform(
        str(tmp_path / "mesh.glb"),
        str(tmp_path / "transforms/*.json"),
        str(tmp_path / "outputs/*.glb"),
    )

    assert os.path.isfile(str(tmp_path / "outputs/000000.glb"))
    assert os.path.isfile(str(tmp_path / "outputs/000001.glb"))
    translated = Mesh.load(str(tmp_path / "outputs/000000.glb"))
    np.testing.assert_allclose(translated.vertices, box_mesh.vertices + [1, 2, 3], atol=1e-6)
    scaled = Mesh.load(str(tmp_path / "outputs/000001.glb"))
    np.testing.assert_allclose(scaled.vertices, box_mesh.vertices * 2.0, atol=1e-6)


def test_broadcast_1_mesh_n_transforms_loads_mesh_once(tmp_path, box_mesh, translation_transform, scale_transform):
    """Mesh file should be loaded only once even when N transforms are applied."""
    _save_mesh(box_mesh, str(tmp_path / "mesh.glb"))
    _save_transform(translation_transform, str(tmp_path / "transforms/000000.json"))
    _save_transform(scale_transform,       str(tmp_path / "transforms/000001.json"))

    load_count = 0
    original_load = Mesh.load

    def counting_load(path):
        nonlocal load_count
        load_count += 1
        return original_load(path)

    Mesh.load = staticmethod(counting_load)
    try:
        run_mesh_transform(
            str(tmp_path / "mesh.glb"),
            str(tmp_path / "transforms/*.json"),
            str(tmp_path / "outputs/*.glb"),
        )
    finally:
        Mesh.load = staticmethod(original_load)

    assert load_count == 1


def test_broadcast_n_meshes_1_transform(tmp_path, box_mesh, translation_transform):
    _save_mesh(box_mesh, str(tmp_path / "meshes/000000.glb"))
    _save_mesh(box_mesh, str(tmp_path / "meshes/000001.glb"))
    _save_transform(translation_transform, str(tmp_path / "transform.json"))

    run_mesh_transform(
        str(tmp_path / "meshes/*.glb"),
        str(tmp_path / "transform.json"),
        str(tmp_path / "outputs/*.glb"),
    )

    assert os.path.isfile(str(tmp_path / "outputs/000000.glb"))
    assert os.path.isfile(str(tmp_path / "outputs/000001.glb"))


def test_broadcast_n_meshes_n_transforms_zip(tmp_path, box_mesh, translation_transform, scale_transform):
    _save_mesh(box_mesh, str(tmp_path / "meshes/000000.glb"))
    _save_mesh(box_mesh, str(tmp_path / "meshes/000001.glb"))
    _save_transform(translation_transform, str(tmp_path / "transforms/000000.json"))
    _save_transform(scale_transform,       str(tmp_path / "transforms/000001.json"))

    run_mesh_transform(
        str(tmp_path / "meshes/*.glb"),
        str(tmp_path / "transforms/*.json"),
        str(tmp_path / "outputs/*.glb"),
    )

    assert os.path.isfile(str(tmp_path / "outputs/000000.glb"))
    assert os.path.isfile(str(tmp_path / "outputs/000001.glb"))


def test_broadcast_zip_mismatched_lengths_raises(tmp_path, box_mesh, translation_transform, scale_transform):
    # 2 meshes, 3 transforms → N:M mismatch → should raise
    _save_mesh(box_mesh, str(tmp_path / "meshes/000000.glb"))
    _save_mesh(box_mesh, str(tmp_path / "meshes/000001.glb"))
    _save_transform(translation_transform, str(tmp_path / "transforms/000000.json"))
    _save_transform(scale_transform,       str(tmp_path / "transforms/000001.json"))
    _save_transform(translation_transform, str(tmp_path / "transforms/000002.json"))

    with pytest.raises(ValueError, match="equal lengths"):
        run_mesh_transform(
            str(tmp_path / "meshes/*.glb"),
            str(tmp_path / "transforms/*.json"),
            str(tmp_path / "outputs/*.glb"),
        )
