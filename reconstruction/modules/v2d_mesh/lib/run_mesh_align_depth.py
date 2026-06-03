# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Transform3d
from v2d.common.broadcast import broadcast_zip, resolve_glob, resolve_output
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_transform import mesh_transform
from v2d.mesh.lib.mesh_align_depth import mesh_align_depth


def run_mesh_align_depth(
    mesh: str,
    depth: str,
    intrinsics: str,
    output_transform: str,
    transform: str | None = None,
) -> None:
    """Estimate a scale Transform3d to align mesh depth to real depth, for each input tuple.

    Outputs a scale-only Transform3d JSON per pair, suitable for feeding directly
    into run_mesh_transform to rescale the mesh.
    """
    mesh_paths = resolve_glob(mesh)
    depth_paths = resolve_glob(depth)
    intrinsics_paths = resolve_glob(intrinsics)
    transform_paths = resolve_glob(transform) if transform is not None else None

    if transform_paths is not None:
        tuples = broadcast_zip(mesh_paths, depth_paths, intrinsics_paths, transform_paths)
    else:
        tuples = [(m, d, i, None) for m, d, i in broadcast_zip(mesh_paths, depth_paths, intrinsics_paths)]

    mesh_cache: dict[str, Mesh] = {}
    for mesh_p, depth_p, intrinsics_p, transform_p in tuples:
        if mesh_p not in mesh_cache:
            mesh_cache[mesh_p] = Mesh.load(mesh_p)
        m = mesh_cache[mesh_p]

        input_transform = Transform3d.load(transform_p) if transform_p is not None else None
        if input_transform is not None:
            m = mesh_transform(m, input_transform)

        scale_transform = mesh_align_depth(m, DepthImage.load(depth_p), CameraIntrinsics.load(intrinsics_p))

        if input_transform is not None:
            # scale_transform is relative to the posed mesh, but we want it relative
            # to the input mesh (before the transform). Compose with the input transform scale.
            composed = [s * t for s, t in zip(scale_transform.scale, input_transform.scale)]
            scale_transform = Transform3d(rotation=scale_transform.rotation, translation=scale_transform.translation, scale=composed)

        path_sources = [(mesh_p, mesh_paths), (depth_p, depth_paths), (intrinsics_p, intrinsics_paths)]
        if transform_p is not None:
            path_sources.append((transform_p, transform_paths))
        out_p = resolve_output(output_transform, path_sources)
        os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
        scale_transform.save(out_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate a scale Transform3d to align mesh depth to real depth")
    parser.add_argument("--mesh", required=True, help="Mesh file or glob (e.g. meshes/*.glb)")
    parser.add_argument("--depth", required=True, help="Depth image file or glob (e.g. depth/*.png)")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON file or glob")
    parser.add_argument("--output_transform", required=True, help="Output Transform3d JSON file or pattern (e.g. scales/*.json)")
    parser.add_argument("--transform", default=None, help="Optional pose transform to apply to mesh before comparison")
    args = parser.parse_args()
    run_mesh_align_depth(args.mesh, args.depth, args.intrinsics, args.output_transform, transform=args.transform)
