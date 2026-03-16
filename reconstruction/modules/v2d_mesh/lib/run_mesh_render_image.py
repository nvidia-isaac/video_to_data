import argparse
import os

from v2d.common.datatypes import CameraIntrinsics, Image, Transform3d
from v2d.common.broadcast import broadcast_zip, resolve_glob, resolve_output
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_image import mesh_render_image
from v2d.mesh.lib.mesh_transform import mesh_transform


def run_mesh_render_image(
    mesh: str,
    intrinsics: str,
    output_image: str,
    transform: str | None = None,
    background: str | None = None,
) -> None:
    mesh_paths = resolve_glob(mesh)
    intrinsics_paths = resolve_glob(intrinsics)
    transform_paths = resolve_glob(transform) if transform is not None else None
    background_paths = resolve_glob(background) if background is not None else None

    varying = [p for p in [mesh_paths, intrinsics_paths, transform_paths, background_paths] if p is not None]
    base_tuples = broadcast_zip(*varying)

    mesh_cache: dict[str, Mesh] = {}
    for row in base_tuples:
        it = iter(row)
        mesh_p = next(it)
        intrinsics_p = next(it)
        transform_p = next(it) if transform_paths is not None else None
        background_p = next(it) if background_paths is not None else None

        if mesh_p not in mesh_cache:
            mesh_cache[mesh_p] = Mesh.load(mesh_p)
        m = mesh_cache[mesh_p]

        if transform_p is not None:
            m = mesh_transform(m, Transform3d.load(transform_p))

        cam = CameraIntrinsics.load(intrinsics_p)

        bg = Image.load(background_p) if background_p is not None else None

        path_sources = [(mesh_p, mesh_paths), (intrinsics_p, intrinsics_paths)]
        if transform_p is not None:
            path_sources.append((transform_p, transform_paths))
        if background_p is not None:
            path_sources.append((background_p, background_paths))
        out_p = resolve_output(output_image, path_sources)
        os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
        mesh_render_image(m, cam, background=bg).save(out_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an RGB image of a mesh")
    parser.add_argument("--mesh", required=True, help="Mesh file or glob (e.g. meshes/*.glb)")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON file or glob")
    parser.add_argument("--output_image", required=True, help="Output file or pattern (e.g. images/*.png)")
    parser.add_argument("--transform", default=None, help="Optional transform JSON file or glob")
    parser.add_argument("--background", default=None, help="Optional background image file or glob")
    args = parser.parse_args()
    run_mesh_render_image(args.mesh, args.intrinsics, args.output_image, transform=args.transform, background=args.background)
