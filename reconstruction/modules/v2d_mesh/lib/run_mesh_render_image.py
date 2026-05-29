# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import argparse
import os

import numpy as np

from v2d.common.datatypes import CameraIntrinsics, Image, Transform3d
from v2d.common.broadcast import broadcast_zip, resolve_glob, resolve_output
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_depth import _CV_TO_GL


def run_mesh_render_image(
    mesh: str,
    intrinsics: str,
    output_image: str,
    transform: str | None = None,
    background: str | None = None,
) -> None:
    """
    Render mesh overlays for all (mesh, intrinsics, transform, background) tuples.

    Optimised for the common case of a fixed mesh rendered from many poses:
    the mesh geometry is uploaded to the GPU once (as a pyrender.Mesh) and only
    the scene-node pose matrix is updated per frame, avoiding the repeated
    mesh-geometry re-uploads that dominate runtime in the naive approach.
    """
    import pyrender

    mesh_paths = resolve_glob(mesh)
    intrinsics_paths = resolve_glob(intrinsics)
    transform_paths = resolve_glob(transform) if transform is not None else None
    background_paths = resolve_glob(background) if background is not None else None

    varying = [p for p in [mesh_paths, intrinsics_paths, transform_paths, background_paths] if p is not None]
    base_tuples = broadcast_zip(*varying)

    # Caches — avoid redundant I/O and GPU uploads
    mesh_cache: dict[str, Mesh] = {}
    intrinsics_cache: dict[str, CameraIntrinsics] = {}
    py_mesh_cache: dict[str, object] = {}  # mesh_path -> pyrender.Mesh (GPU-resident)

    scene = None
    mesh_node = None
    scene_key = None  # (mesh_p, intrinsics_p) — rebuilt only when either changes
    renderer = None

    try:
        for row in base_tuples:
            it = iter(row)
            mesh_p       = next(it)
            intrinsics_p = next(it)
            transform_p  = next(it) if transform_paths  is not None else None
            background_p = next(it) if background_paths is not None else None

            if mesh_p not in mesh_cache:
                mesh_cache[mesh_p] = Mesh.load(mesh_p)
            m = mesh_cache[mesh_p]

            if intrinsics_p not in intrinsics_cache:
                intrinsics_cache[intrinsics_p] = CameraIntrinsics.load(intrinsics_p)
            cam = intrinsics_cache[intrinsics_p]

            # Rebuild scene only when mesh or camera changes (once per video in typical use)
            key = (mesh_p, intrinsics_p)
            if key != scene_key:
                if mesh_p not in py_mesh_cache:
                    py_mesh_cache[mesh_p] = pyrender.Mesh.from_trimesh(
                        m.to_trimesh(), smooth=False
                    )
                py_mesh = py_mesh_cache[mesh_p]

                scene = pyrender.Scene(
                    bg_color=[0.0, 0.0, 0.0, 0.0],
                    ambient_light=[0.3, 0.3, 0.3],
                )
                mesh_node = scene.add(py_mesh, name="mesh", pose=np.eye(4))
                light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
                scene.add(light, pose=_CV_TO_GL)
                camera_obj = pyrender.IntrinsicsCamera(
                    fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
                    znear=0.001, zfar=10000.0,
                )
                scene.add(camera_obj, pose=_CV_TO_GL)
                scene_key = key

                if (renderer is None
                        or renderer.viewport_width  != cam.width
                        or renderer.viewport_height != cam.height):
                    if renderer is not None:
                        renderer.delete()
                    renderer = pyrender.OffscreenRenderer(cam.width, cam.height)

            # Update only the 4×4 pose matrix — no geometry re-upload
            M = Transform3d.load(transform_p).to_matrix() if transform_p is not None else np.eye(4)
            scene.set_pose(mesh_node, M)

            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

            bg = Image.load(background_p) if background_p is not None else None
            if bg is None:
                result = Image(data=color[:, :, :3].astype(np.uint8))
            else:
                alpha      = color[:, :, 3:].astype(np.float32) / 255.0
                rendered   = color[:, :, :3].astype(np.float32)
                bg_data    = bg.data.astype(np.float32)
                composited = (rendered * alpha + bg_data * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
                result     = Image(data=composited)

            path_sources = [(mesh_p, mesh_paths), (intrinsics_p, intrinsics_paths)]
            if transform_p  is not None: path_sources.append((transform_p,  transform_paths))
            if background_p is not None: path_sources.append((background_p, background_paths))
            out_p = resolve_output(output_image, path_sources)
            os.makedirs(os.path.dirname(os.path.abspath(out_p)), exist_ok=True)
            result.save(out_p)
    finally:
        if renderer is not None:
            renderer.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an RGB image of a mesh")
    parser.add_argument("--mesh",         required=True, help="Mesh file or glob (e.g. meshes/*.glb)")
    parser.add_argument("--intrinsics",   required=True, help="Camera intrinsics JSON file or glob")
    parser.add_argument("--output_image", required=True, help="Output file or pattern (e.g. images/*.png)")
    parser.add_argument("--transform",    default=None,  help="Optional transform JSON file or glob")
    parser.add_argument("--background",   default=None,  help="Optional background image file or glob")
    args = parser.parse_args()
    run_mesh_render_image(
        args.mesh, args.intrinsics, args.output_image,
        transform=args.transform, background=args.background,
    )
