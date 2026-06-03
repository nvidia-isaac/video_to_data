# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from pathlib import Path

from v2d.common.broadcast import apply_output_pattern, resolve_glob
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_get_bounding_box import mesh_get_bounding_box


def run_mesh_get_bounding_box(mesh: str, output: str) -> None:
    for mesh_p in resolve_glob(mesh):
        m = Mesh.load(mesh_p)
        bb = mesh_get_bounding_box(m)
        out_p = apply_output_pattern(output, Path(mesh_p).stem)
        os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
        bb.save(out_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the axis-aligned bounding box of a mesh")
    parser.add_argument("--mesh", required=True, help="Mesh file or glob (e.g. meshes/*.glb)")
    parser.add_argument("--output", required=True, help="Output JSON file or pattern (e.g. bboxes/*.json)")
    args = parser.parse_args()
    run_mesh_get_bounding_box(args.mesh, args.output)
