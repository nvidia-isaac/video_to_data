import argparse
import json
import os

from v2d.common.datatypes import CameraIntrinsics
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_mask import mesh_render_mask


def run_mesh_render_mask(mesh_path: str, intrinsics_path: str, output_mask_path: str) -> None:
    mesh = Mesh.load(mesh_path)
    with open(intrinsics_path) as f:
        intrinsics = CameraIntrinsics.from_dict(json.load(f))
    mask = mesh_render_mask(mesh, intrinsics)
    os.makedirs(os.path.dirname(os.path.abspath(output_mask_path)), exist_ok=True)
    mask.to_pil_image().save(output_mask_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a silhouette mask of a mesh")
    parser.add_argument("--mesh", required=True, help="Input mesh file")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON")
    parser.add_argument("--output_mask", required=True, help="Output mask PNG (grayscale)")
    args = parser.parse_args()
    run_mesh_render_mask(args.mesh, args.intrinsics, args.output_mask)
