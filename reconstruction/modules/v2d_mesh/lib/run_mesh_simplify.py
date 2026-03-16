import argparse
import os
from pathlib import Path

from v2d.common.broadcast import apply_output_pattern, resolve_glob
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_simplify import mesh_simplify


def run_mesh_simplify(
    input_mesh: str,
    output_mesh: str,
    face_count: int | None = None,
    factor: float | None = None,
) -> None:
    for mesh_p in resolve_glob(input_mesh):
        mesh = Mesh.load(mesh_p)
        result = mesh_simplify(mesh, face_count=face_count, factor=factor)
        out_p = apply_output_pattern(output_mesh, Path(mesh_p).stem)
        os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
        result.save(out_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplify a mesh via quadric decimation")
    parser.add_argument("--input_mesh", required=True, help="Mesh file or glob (e.g. meshes/*.glb)")
    parser.add_argument("--output_mesh", required=True, help="Output file or pattern (e.g. simplified/*.glb)")
    parser.add_argument("--face_count", type=int, default=None, help="Target face count")
    parser.add_argument("--factor", type=float, default=None, help="Reduction factor (0.0–1.0)")
    args = parser.parse_args()
    run_mesh_simplify(args.input_mesh, args.output_mesh, face_count=args.face_count, factor=args.factor)
