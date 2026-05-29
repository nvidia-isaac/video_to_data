# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import shutil

from v2d.mesh.docker.run_mesh_transform import run_mesh_transform

from .conftest import is_glb


def test_single(output_dir, mesh, transform):
    out = str(output_dir / "mesh_transformed.glb")
    run_mesh_transform(mesh, transform, out)
    assert is_glb(out)


def test_broadcast_1_mesh_n_transforms(output_dir, mesh, transforms_glob):
    run_mesh_transform(mesh, transforms_glob, str(output_dir / "*.glb"))
    assert is_glb(str(output_dir / "0.glb"))
    assert is_glb(str(output_dir / "1.glb"))


def test_broadcast_n_meshes_1_transform(output_dir, tmp_path, mesh, transform):
    meshes_dir = tmp_path / "meshes"
    meshes_dir.mkdir()
    shutil.copy(mesh, meshes_dir / "a.glb")
    shutil.copy(mesh, meshes_dir / "b.glb")

    run_mesh_transform(str(meshes_dir / "*.glb"), transform, str(output_dir / "*.glb"))

    assert is_glb(str(output_dir / "a.glb"))
    assert is_glb(str(output_dir / "b.glb"))
