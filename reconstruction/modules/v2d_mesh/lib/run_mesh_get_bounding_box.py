import argparse
import json
import os

from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_get_bounding_box import mesh_get_bounding_box


def run_mesh_get_bounding_box(mesh_path: str, output_path: str) -> None:
    mesh = Mesh.load(mesh_path)
    bb = mesh_get_bounding_box(mesh)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bb.to_dict(), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the axis-aligned bounding box of a mesh")
    parser.add_argument("--mesh", required=True, help="Input mesh file")
    parser.add_argument("--output", required=True, help="Output JSON file for BoundingBox3d")
    args = parser.parse_args()
    run_mesh_get_bounding_box(args.mesh, args.output)
