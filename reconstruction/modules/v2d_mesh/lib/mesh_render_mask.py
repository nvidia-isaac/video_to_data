import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import pyglet
pyglet.options['headless'] = True

from v2d.common.datatypes import CameraIntrinsics, Mask
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_depth import mesh_render_depth
import numpy as np


def mesh_render_mask(mesh: Mesh, intrinsics: CameraIntrinsics) -> Mask:
    """
    Render a binary silhouette mask of the mesh from a camera at the origin.

    The camera convention matches mesh_render_depth (OpenCV: looking along +Z, Y down).
    Mask pixels are 1.0 where the mesh is visible, 0.0 for background.
    """
    depth_img = mesh_render_depth(mesh, intrinsics)
    return Mask(mask=(depth_img.depth > 0).astype(np.float32))
