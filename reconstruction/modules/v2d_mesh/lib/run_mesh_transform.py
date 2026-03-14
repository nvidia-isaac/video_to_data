import argparse
import json
import os

from v2d.common.datatypes import Transform3d
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_transform import mesh_transform


def run_mesh_transform(input_mesh_path: str, transform_path: str, output_mesh_path: str) -> None:
    mesh = Mesh.load(input_mesh_path)
    with open(transform_path) as f:
        transform = Transform3d.from_dict(json.load(f))
    result = mesh_transform(mesh, transform)
    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
    result.save(output_mesh_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a Transform3d to a mesh")
    parser.add_argument("--input_mesh", required=True, help="Input mesh file (GLB, OBJ, ...)")
    parser.add_argument("--transform", required=True, help="Transform JSON file")
    parser.add_argument("--output_mesh", required=True, help="Output mesh file")
    args = parser.parse_args()
    run_mesh_transform(args.input_mesh, args.transform, args.output_mesh)
