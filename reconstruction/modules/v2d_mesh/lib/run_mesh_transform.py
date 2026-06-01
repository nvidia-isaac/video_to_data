# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os

from v2d.common.datatypes import Transform3d
from v2d.common.broadcast import broadcast_pairs, resolve_glob, resolve_output
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_transform import mesh_transform


def run_mesh_transform(
    input_mesh: str,
    transform: str,
    output_mesh: str,
) -> None:
    mesh_paths = resolve_glob(input_mesh)
    transform_paths = resolve_glob(transform)
    mesh_cache: dict[str, Mesh] = {}
    for mesh_p, transform_p in broadcast_pairs(mesh_paths, transform_paths):
        if mesh_p not in mesh_cache:
            mesh_cache[mesh_p] = Mesh.load(mesh_p)
        t = Transform3d.load(transform_p)
        result = mesh_transform(mesh_cache[mesh_p], t)
        out_p = resolve_output(output_mesh, [(mesh_p, mesh_paths), (transform_p, transform_paths)])
        os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
        result.save(out_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a Transform3d to a mesh")
    parser.add_argument("--input_mesh", required=True, help="Mesh file or glob (e.g. meshes/*.glb)")
    parser.add_argument("--transform", required=True, help="Transform JSON file or glob (e.g. transforms/*.json)")
    parser.add_argument("--output_mesh", required=True, help="Output file or pattern (e.g. outputs/*.glb)")
    args = parser.parse_args()
    run_mesh_transform(args.input_mesh, args.transform, args.output_mesh)
