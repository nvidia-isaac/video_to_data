# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
from typing import List, Optional

import numpy as np
import pyrender
import torch
import trimesh


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
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


T_OPENGL_FROM_CV = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float64)


class Renderer:

    def __init__(
        self,
        K: np.ndarray,
        viewport_size: tuple[int, int],
        num_vertices: int = 0,
        faces: Optional[np.ndarray] = None,
        mesh_base_rgb=(0.4, 0.9, 0.7),
        scene_bg_rgb=(0, 0, 0),
    ):
        """
        Persistent mesh renderer backed by pyrender.

        Scene, camera, lights, and material are created once and reused.
        Only the mesh vertices and camera pose are updated per frame.
        Call close() when done to free GPU resources.

        Args:
            K: (3, 3) camera intrinsics matrix.
            viewport_size: (width, height) of the render target in pixels.
            num_vertices: Number of vertices in the mesh. If provided, the mesh
                is created with the given number of vertices and faces.
            faces: Optional (F, 3) mesh face indices. Can be provided here
                to reuse across frames, or passed per-call.
            mesh_base_rgb: Base color of the mesh material.
            scene_bg_rgb: Background color of the scene.
        """
        self.K = K
        w, h = viewport_size
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=w,
            viewport_height=h,
        )

        self._cached_trimesh = None
        if num_vertices > 0 and faces is not None:
            self._cached_trimesh = trimesh.Trimesh(
                vertices=np.zeros((num_vertices, 3)), faces=faces, process=False,
            )

        self._material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(*mesh_base_rgb, 1.0),
        )

        self._scene = pyrender.Scene(
            bg_color=[*scene_bg_rgb, 0.0], ambient_light=(0.5, 0.5, 0.5),
        )

        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], zfar=1e12,
        )
        self._camera_node = self._scene.add(camera, pose=np.eye(4))

        for node in create_raymond_lights():
            self._scene.add_node(node, parent_node=self._camera_node)

        self._mesh_node = None

    def __call__(
        self,
        vertices: np.ndarray,
        camera_pose: np.ndarray,
        faces: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render a mesh, reusing the cached scene/lights/camera/material.

        Args:
            vertices: (V, 3) mesh vertices.
            camera_pose: (4, 4) world-from-camera in CV convention
                (Y-down, Z-forward). Converted to GL convention internally
                via T_OPENGL_FROM_CV.
            faces: Optional (F, 3) face indices. Overrides the faces set at
                init time. At least one of the two must be provided.
            image: Optional (H, W, 3) uint8 background image. If provided,
                the mesh is composited over it and (H, W, 3) float32 is
                returned. If None, returns (H, W, 4) RGBA float32 rendered
                against scene_bg_rgb.
        """
        if faces is not None:
            tm = trimesh.Trimesh(vertices, faces, process=False)
        elif self._cached_trimesh is not None:
            self._cached_trimesh.vertices = vertices
            tm = self._cached_trimesh
        else:
            raise ValueError(
                "Either num_vertices and faces must be provided at init, "
                "or faces must be provided at call time"
            )

        mesh = pyrender.Mesh.from_trimesh(tm, material=self._material)

        if self._mesh_node is not None:
            self._scene.remove_node(self._mesh_node)
        self._mesh_node = self._scene.add(mesh, "mesh")

        gl_camera_pose = camera_pose @ T_OPENGL_FROM_CV
        self._scene.set_pose(self._camera_node, gl_camera_pose)

        color, _ = self._renderer.render(self._scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if image is None:
            return color

        image = image.astype(np.float32) / 255.0
        valid_mask = color[:, :, -1:]
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        return output_img.astype(np.float32)

    def close(self):
        """Release the OpenGL context and GPU resources."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    def __del__(self):
        self.close()
