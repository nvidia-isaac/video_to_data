# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import json
import shutil

from v2d.mesh.docker.run_mesh_get_bounding_box import run_mesh_get_bounding_box


def test_bounding_box(output_dir, mesh):
    out = str(output_dir / "bbox.json")
    run_mesh_get_bounding_box(mesh, out)
    data = json.loads(open(out).read())
    assert {"x0", "y0", "z0", "x1", "y1", "z1"}.issubset(data.keys())


def test_bounding_box_broadcast(output_dir, tmp_path, mesh):
    meshes_dir = tmp_path / "meshes"
    meshes_dir.mkdir()
    shutil.copy(mesh, meshes_dir / "a.glb")
    shutil.copy(mesh, meshes_dir / "b.glb")

    run_mesh_get_bounding_box(str(meshes_dir / "*.glb"), str(output_dir / "*.json"))

    for name in ("a.json", "b.json"):
        data = json.loads(open(str(output_dir / name)).read())
        assert {"x0", "y0", "z0", "x1", "y1", "z1"}.issubset(data.keys())
