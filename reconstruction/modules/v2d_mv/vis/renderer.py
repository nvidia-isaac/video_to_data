# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import pyglet
pyglet.options['headless'] = True

import numpy as np
import pyrender
import trimesh


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
    """Offscreen multi-mesh renderer backed by pyrender.

    The EGL context, scene, camera, and lights are persistent.
    Meshes and camera pose are updated per render call.
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
        self._mesh_nodes: list[pyrender.Node] = []

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

    def _set_meshes(
        self,
        meshes: list[trimesh.Trimesh],
        K: np.ndarray,
        T: np.ndarray,
    ) -> None:
        self._ensure_camera(K)

        for node in self._mesh_nodes:
            self._scene.remove_node(node)
        self._mesh_nodes.clear()

        for mesh in meshes:
            pr_mesh = pyrender.Mesh.from_trimesh(mesh)
            node = self._scene.add(pr_mesh)
            self._mesh_nodes.append(node)

        self._scene.set_pose(self._camera_node, T @ _T_OPENGL_FROM_CV)

    def render_overlay(
        self,
        meshes: list[trimesh.Trimesh],
        K: np.ndarray,
        T: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        """Render Phong-shaded meshes composited over a background image.

        Args:
            meshes: List of trimesh meshes to render. Each mesh may carry its
                own vertex_colors; otherwise pyrender applies a default material.
            K: (3, 3) camera intrinsics matrix.
            T: (4, 4) world-from-camera extrinsic in CV convention.
            image: (H, W, 3) uint8 background image.

        Returns:
            (H, W, 3) float32 composited image in [0, 1].
        """
        self._set_meshes(meshes, K, T)

        color, _ = self._renderer.render(self._scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        image = image.astype(np.float32) / 255.0
        valid_mask = color[:, :, -1:]
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        return output_img.astype(np.float32)

    def render_depth(
        self,
        meshes: list[trimesh.Trimesh],
        K: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """Render the combined z-buffer of all meshes.

        Args:
            meshes: List of trimesh meshes to render.
            K: (3, 3) camera intrinsics matrix.
            T: (4, 4) world-from-camera extrinsic in CV convention.

        Returns:
            (H, W) float32 depth in scene units. 0 = background.
        """
        self._set_meshes(meshes, K, T)
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
