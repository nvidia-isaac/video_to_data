import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import pyglet
pyglet.options['headless'] = True

import numpy as np

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.mesh.lib.mesh import Mesh

# Camera pose: OpenCV convention (looking along +Z, Y down) → OpenGL (looking along -Z, Y up).
# This 4x4 matrix flips Y and Z so the OpenGL camera sees the same scene as the OpenCV camera.
_CV_TO_GL = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1],
], dtype=np.float64)


def mesh_render_depth(mesh: Mesh, intrinsics: CameraIntrinsics) -> DepthImage:
    """
    Render a depth image of the mesh from a camera at the origin.

    The camera is assumed to be at the world origin looking along +Z (OpenCV convention).
    Mesh vertices must have Z > 0 to be visible.

    Returns a DepthImage where depth[y, x] is the distance in meters to the closest
    mesh surface along that ray. Background pixels have depth == 0.

    Requires pyrender and an offscreen OpenGL backend (set PYOPENGL_PLATFORM=osmesa
    for software rendering or PYOPENGL_PLATFORM=egl for GPU-based offscreen rendering).
    """
    import pyrender

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.0, 0.0, 0.0])

    py_mesh = pyrender.Mesh.from_trimesh(mesh.to_trimesh(), smooth=False)
    scene.add(py_mesh)

    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        znear=0.001,
        zfar=10000.0,
    )
    scene.add(camera, pose=_CV_TO_GL)

    r = pyrender.OffscreenRenderer(intrinsics.width, intrinsics.height)
    try:
        depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    finally:
        r.delete()

    return DepthImage(depth=depth.astype(np.float32))
