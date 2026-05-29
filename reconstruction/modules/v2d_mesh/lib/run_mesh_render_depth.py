# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import argparse
import os

from v2d.common.datatypes import CameraIntrinsics, Transform3d
from v2d.common.broadcast import broadcast_zip, resolve_glob, resolve_output
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_depth import mesh_render_depth
from v2d.mesh.lib.mesh_transform import mesh_transform


def run_mesh_render_depth(
    mesh: str,
    intrinsics: str,
    output_depth: str,
    transform: str | None = None,
) -> None:
    mesh_paths = resolve_glob(mesh)
    intrinsics_paths = resolve_glob(intrinsics)
    transform_paths = resolve_glob(transform) if transform is not None else None

    if transform_paths is not None:
        tuples = broadcast_zip(mesh_paths, intrinsics_paths, transform_paths)
    else:
        tuples = [(m, i, None) for m, i in broadcast_zip(mesh_paths, intrinsics_paths)]

    mesh_cache: dict[str, Mesh] = {}
    for mesh_p, intrinsics_p, transform_p in tuples:
        if mesh_p not in mesh_cache:
            mesh_cache[mesh_p] = Mesh.load(mesh_p)
        m = mesh_cache[mesh_p]

        if transform_p is not None:
            m = mesh_transform(m, Transform3d.load(transform_p))

        cam = CameraIntrinsics.load(intrinsics_p)

        path_sources = [(mesh_p, mesh_paths), (intrinsics_p, intrinsics_paths)]
        if transform_p is not None:
            path_sources.append((transform_p, transform_paths))
        out_p = resolve_output(output_depth, path_sources)
        os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
        mesh_render_depth(m, cam).save(out_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a depth image of a mesh")
    parser.add_argument("--mesh", required=True, help="Mesh file or glob (e.g. meshes/*.glb)")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON file or glob")
    parser.add_argument("--output_depth", required=True, help="Output file or pattern (e.g. depth/*.png)")
    parser.add_argument("--transform", default=None, help="Optional transform JSON file or glob")
    args = parser.parse_args()
    run_mesh_render_depth(args.mesh, args.intrinsics, args.output_depth, transform=args.transform)
