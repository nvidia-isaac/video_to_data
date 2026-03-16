import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import pyglet
pyglet.options['headless'] = True

import numpy as np

from v2d.common.datatypes import CameraIntrinsics, Image
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_depth import _CV_TO_GL


def mesh_render_image(
    mesh: Mesh,
    intrinsics: CameraIntrinsics,
    background: Image | None = None,
) -> Image:
    """
    Render an RGB image of the mesh from a camera at the origin.

    The camera convention matches mesh_render_depth (OpenCV: looking along +Z, Y down).
    A directional light is placed at the camera position so front-facing surfaces are lit.

    If background is provided, the rendered mesh is alpha-composited over it.
    Otherwise background pixels are black (0, 0, 0).

    Requires pyrender and an offscreen OpenGL backend.
    """
    import pyrender

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])

    py_mesh = pyrender.Mesh.from_trimesh(mesh.to_trimesh(), smooth=False)
    scene.add(py_mesh)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=_CV_TO_GL)

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
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        r.delete()

    if background is None:
        return Image(data=color[:, :, :3].astype(np.uint8))

    alpha = color[:, :, 3:].astype(np.float32) / 255.0
    rendered = color[:, :, :3].astype(np.float32)
    bg = background.data.astype(np.float32)
    composited = (rendered * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return Image(data=composited)
