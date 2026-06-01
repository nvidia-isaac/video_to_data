# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


class GPURenderer:
    """GPU-accelerated batched mesh renderer backed by PyTorch3D.

    Only image_size and device are persistent. Faces and vertices
    are provided per render call.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        device: str | torch.device = "cuda",
    ):
        """
        Args:
            image_size: (width, height) of the render target in pixels.
            device: torch device.
        """
        self.device = torch.device(device)
        self.W, self.H = image_size
        self._overlay_renderer: MeshRenderer | None = None

    def _build_cameras(self, K: torch.Tensor, T: torch.Tensor, N: int):
        T_cam = torch.inverse(T.float().to(self.device))
        R = T_cam[:3, :3].unsqueeze(0).expand(N, -1, -1).contiguous()
        tvec = T_cam[:3, 3].unsqueeze(0).expand(N, -1).contiguous()
        camera_matrix = K.float().to(self.device).unsqueeze(0).expand(N, -1, -1).contiguous()
        image_size = torch.tensor(
            [[self.H, self.W]], device=self.device, dtype=torch.float32,
        ).expand(N, -1)
        return cameras_from_opencv_projection(
            R=R, tvec=tvec, camera_matrix=camera_matrix, image_size=image_size,
        )

    def _build_meshes(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        textures: TexturesVertex | None = None,
    ) -> Meshes:
        N = verts.shape[0]
        verts_dev = verts.float().to(self.device)
        faces_dev = faces.to(self.device, dtype=torch.int64)
        faces_batch = faces_dev.unsqueeze(0).expand(N, -1, -1)
        return Meshes(verts=verts_dev, faces=faces_batch, textures=textures)

    def render_depth(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Render single-layer z-buffer.

        Args:
            verts: (N, V, 3) mesh vertices.
            faces: (F, 3) int64 face indices.
            K: (3, 3) camera intrinsics.
            T: (4, 4) world-from-camera extrinsic.

        Returns:
            (N, H, W) float32 z-buffer. 0 = background.
        """
        zbuf = self.render_depth_layers(verts, faces, K, T, faces_per_pixel=1)
        zbuf = zbuf.squeeze(-1)
        zbuf = zbuf.clamp(min=0)
        return zbuf

    def render_depth_layers(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        faces_per_pixel: int = 4,
    ) -> torch.Tensor:
        """Render multi-layer z-buffer (depth peeling).

        Args:
            verts: (N, V, 3) mesh vertices.
            faces: (F, 3) int64 face indices.
            K: (3, 3) camera intrinsics.
            T: (4, 4) world-from-camera extrinsic.
            faces_per_pixel: Number of depth layers per pixel.

        Returns:
            (N, H, W, faces_per_pixel) float32 z-buffer.
            Sorted front-to-back. Empty slots = -1.
        """
        N = verts.shape[0]
        cameras = self._build_cameras(K, T, N)
        meshes = self._build_meshes(verts, faces)

        raster_settings = RasterizationSettings(
            image_size=(self.H, self.W),
            blur_radius=0.0,
            faces_per_pixel=faces_per_pixel,
            max_faces_per_bin=40000,
        )
        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings,
        )

        with torch.no_grad():
            fragments = rasterizer(meshes)

        return fragments.zbuf

    def _get_overlay_renderer(self):
        """Lazily build and cache the overlay MeshRenderer."""
        if self._overlay_renderer is None:
            dummy_cameras = self._build_cameras(
                torch.eye(3, device=self.device),
                torch.eye(4, device=self.device),
                1,
            )
            lights = DirectionalLights(
                device=self.device,
                ambient_color=((0.5, 0.5, 0.5),),
                diffuse_color=((0.5, 0.5, 0.5),),
                specular_color=((0.1, 0.1, 0.1),),
                direction=((0, -1, 0),),
            )
            raster_settings = RasterizationSettings(
                image_size=(self.H, self.W),
                blur_radius=0.0,
                faces_per_pixel=1,
                max_faces_per_bin=20000,
            )
            self._overlay_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=dummy_cameras, raster_settings=raster_settings,
                ),
                shader=HardPhongShader(
                    device=self.device,
                    cameras=dummy_cameras,
                    lights=lights,
                    blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
                ),
            )
        return self._overlay_renderer

    def render_overlay(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        images: torch.Tensor,
        mesh_color: tuple[float, float, float] = (0.4, 0.9, 0.7),
    ) -> torch.Tensor:
        """Render Phong-shaded mesh composited over background images.

        Args:
            verts: (N, V, 3) mesh vertices.
            faces: (F, 3) int64 face indices.
            K: (3, 3) camera intrinsics.
            T: (4, 4) world-from-camera extrinsic.
            images: (N, H, W, 3) background images, float32 or uint8.
            mesh_color: Base RGB color for the mesh, in [0, 1].

        Returns:
            (N, H, W, 3) float32 composited images in [0, 1].
        """
        N, V = verts.shape[:2]

        images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()

        color = torch.tensor(
            mesh_color, device=self.device, dtype=torch.float32,
        )
        verts_rgb = color.view(1, 1, 3).expand(N, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        meshes = self._build_meshes(verts, faces, textures=textures)

        cameras = self._build_cameras(K, T, N)
        renderer = self._get_overlay_renderer()

        with torch.no_grad():
            rendered = renderer(meshes, cameras=cameras)  # (N, H, W, 4) RGBA

        alpha = rendered[..., 3:4]
        mesh_rgb = rendered[..., :3]
        return mesh_rgb * alpha + images * (1 - alpha)
