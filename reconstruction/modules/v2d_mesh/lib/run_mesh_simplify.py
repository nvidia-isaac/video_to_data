import argparse
import os

from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_simplify import mesh_simplify


def run_mesh_simplify(
    input_mesh_path: str,
    output_mesh_path: str,
    face_count: int | None = None,
    factor: float | None = None,
) -> None:
    mesh = Mesh.load(input_mesh_path)
    result = mesh_simplify(mesh, face_count=face_count, factor=factor)
    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
    result.save(output_mesh_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify a mesh via quadric decimation")
    parser.add_argument("--input_mesh", required=True, help="Input mesh file")
    parser.add_argument("--output_mesh", required=True, help="Output mesh file")
    parser.add_argument("--face_count", type=int, default=None, help="Target face count")
    parser.add_argument("--factor", type=float, default=None, help="Reduction factor (0.0–1.0)")
    args = parser.parse_args()
    run_mesh_simplify(args.input_mesh, args.output_mesh, face_count=args.face_count, factor=args.factor)
