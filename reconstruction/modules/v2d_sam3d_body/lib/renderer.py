# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyglet
pyglet.options['headless'] = True

import numpy as np
import pyrender


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex unit normals. Returns (V, 3)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vertex_normals = np.zeros_like(verts)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals /= np.maximum(norms, 1e-8)
    return vertex_normals


def _create_raymond_lights() -> list[pyrender.Node]:
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=0.6),
                matrix=matrix,
            )
        )
    return nodes


_T_OPENGL_FROM_CV = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


class Renderer:
    """Offscreen mesh renderer backed by pyrender.

    The EGL context, scene, camera, and lights are persistent.
    Only the mesh and camera pose are updated per render call.
    """

    def __init__(self, image_size: tuple[int, int]):
        """
        Args:
            image_size: (width, height) of the render target in pixels.
        """
        W, H = image_size
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=W, viewport_height=H,
        )
        self._scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0.0], ambient_light=(0.5, 0.5, 0.5),
        )
        self._camera_node = None
        self._cached_K: np.ndarray | None = None
        self._mesh_node = None

    def _ensure_camera(self, K: np.ndarray) -> None:
        if self._cached_K is not None and np.array_equal(K, self._cached_K):
            return
        if self._camera_node is not None:
            self._scene.remove_node(self._camera_node)
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], zfar=100,
        )
        self._camera_node = self._scene.add(camera, pose=np.eye(4))
        for node in _create_raymond_lights():
            self._scene.add_node(node, parent_node=self._camera_node)
        self._cached_K = K.copy()

    def _set_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        mesh_color: tuple[float, float, float] = (0.4, 0.9, 0.7),
    ) -> None:
        self._ensure_camera(K)

        positions = vertices.astype(np.float32)
        normals = vertex_normals(vertices, faces).astype(np.float32)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(*mesh_color, 1.0),
        )
        primitive = pyrender.Primitive(
            positions=positions,
            normals=normals,
            indices=faces.astype(np.uint32),
            material=material,
        )
        if self._mesh_node is not None:
            self._scene.remove_node(self._mesh_node)
        self._mesh_node = self._scene.add(pyrender.Mesh(primitives=[primitive]))

        self._scene.set_pose(self._camera_node, T @ _T_OPENGL_FROM_CV)

    def render_overlay(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        image: np.ndarray,
        mesh_color: tuple[float, float, float] = (0.4, 0.9, 0.7),
    ) -> np.ndarray:
        """Render a Phong-shaded mesh composited over a background image.

        Args:
            vertices: (V, 3) mesh vertices.
            faces: (F, 3) mesh face indices.
            K: (3, 3) camera intrinsics matrix.
            T: (4, 4) world-from-camera extrinsic in CV convention.
            image: (H, W, 3) uint8 background image.
            mesh_color: Base RGB color for the mesh, in [0, 1].

        Returns:
            (H, W, 3) float32 composited image in [0, 1].
        """
        self._set_mesh(vertices, faces, K, T, mesh_color)

        color, _ = self._renderer.render(self._scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        image = image.astype(np.float32) / 255.0
        valid_mask = color[:, :, -1:]
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        return output_img.astype(np.float32)

    def render_depth(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """Render the mesh z-buffer (depth only).

        Args:
            vertices: (V, 3) mesh vertices.
            faces: (F, 3) mesh face indices.
            K: (3, 3) camera intrinsics matrix.
            T: (4, 4) world-from-camera extrinsic in CV convention.

        Returns:
            (H, W) float32 depth in scene units. 0 = background.
        """
        self._set_mesh(vertices, faces, K, T)
        return self._renderer.render(
            self._scene, flags=pyrender.RenderFlags.DEPTH_ONLY,
        )

    def close(self):
        """Release the OpenGL context and GPU resources."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
