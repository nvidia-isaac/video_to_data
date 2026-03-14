import argparse
import json
import os

from v2d.common.datatypes import CameraIntrinsics
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_image import mesh_render_image


def run_mesh_render_image(mesh_path: str, intrinsics_path: str, output_image_path: str) -> None:
    mesh = Mesh.load(mesh_path)
    with open(intrinsics_path) as f:
        intrinsics = CameraIntrinsics.from_dict(json.load(f))
    image = mesh_render_image(mesh, intrinsics)
    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)
    image.to_pil_image().save(output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an RGB image of a mesh")
    parser.add_argument("--mesh", required=True, help="Input mesh file")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON")
    parser.add_argument("--output_image", required=True, help="Output image file (PNG, JPG, ...)")
    args = parser.parse_args()
    run_mesh_render_image(args.mesh, args.intrinsics, args.output_image)
