import argparse
import json
import os

from v2d.common.datatypes import CameraIntrinsics
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_depth import mesh_render_depth


def run_mesh_render_depth(mesh_path: str, intrinsics_path: str, output_depth_path: str) -> None:
    mesh = Mesh.load(mesh_path)
    with open(intrinsics_path) as f:
        intrinsics = CameraIntrinsics.from_dict(json.load(f))
    depth_image = mesh_render_depth(mesh, intrinsics)
    os.makedirs(os.path.dirname(os.path.abspath(output_depth_path)), exist_ok=True)
    depth_image.to_pil_image().save(output_depth_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a depth image of a mesh")
    parser.add_argument("--mesh", required=True, help="Input mesh file")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON")
    parser.add_argument("--output_depth", required=True, help="Output depth PNG (uint16 inverse-depth)")
    args = parser.parse_args()
    run_mesh_render_depth(args.mesh, args.intrinsics, args.output_depth)
