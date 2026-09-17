from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class NvdiffMeshRenderer:
    """Render textured MHR humans and rigid objects for inference visualization."""

    def __init__(self, device: str = "cuda") -> None:
        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError("nvdiffrast rendering requires a CUDA device")
        import nvdiffrast.torch as dr
        import Utils

        self.device = torch.device(device)
        self.glctx = dr.RasterizeCudaContext()
        self.utils = Utils
        self.textured_human_faces_hash = None
        self.textured_human_tensors = None
        self.constant_human_material_key = None
        self.constant_human_tensors = None
        self.textured_object_source = None
        self.textured_object_tensors = ()

    def _load_textured_object_tensors(self, object_mesh_source: str) -> tuple[dict[str, torch.Tensor], ...]:
        object_mesh_source = str(Path(object_mesh_source).expanduser())
        if object_mesh_source == self.textured_object_source:
            return self.textured_object_tensors
        from lib_mhr.object_texture import load_original_object_visual_tensors

        tensors = load_original_object_visual_tensors(object_mesh_source, self.device)
        self.textured_object_source = object_mesh_source
        self.textured_object_tensors = tensors
        return tensors

    def _load_textured_human_tensors(self, human_faces: np.ndarray) -> dict[str, torch.Tensor]:
        from lib_mhr.human_texture import make_mhr_part_texture_tensors, mhr_faces_sha256

        faces_hash = mhr_faces_sha256(human_faces)
        if faces_hash != self.textured_human_faces_hash:
            self.textured_human_tensors = make_mhr_part_texture_tensors(human_faces, self.device)
            self.textured_human_faces_hash = faces_hash
        return self.textured_human_tensors

    def _load_constant_human_tensors(self, human_faces: np.ndarray, vertex_count: int, albedo: np.ndarray) -> dict[str, torch.Tensor]:
        from lib_mhr.human_texture import mhr_faces_sha256

        material_key = (mhr_faces_sha256(human_faces), int(vertex_count), tuple(float(value) for value in albedo))
        if material_key != self.constant_human_material_key:
            self.constant_human_tensors = {
                "faces": torch.as_tensor(human_faces, device=self.device, dtype=torch.int),
                "vertex_color": torch.as_tensor(albedo, device=self.device, dtype=torch.float).reshape(1, 3).expand(vertex_count, -1).contiguous(),
            }
            self.constant_human_material_key = material_key
        return self.constant_human_tensors

    def render_front_batch_constant_human_textured_object(self, human_vertices: np.ndarray, human_faces: np.ndarray, human_vertex_normals: np.ndarray, object_mesh_source: str, object_poses: np.ndarray, intrinsics: np.ndarray, size: tuple[int, int], human_albedo: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        from lib_mhr.object_texture import render_visualization_object_parts

        human_vertices_np = np.asarray(human_vertices, dtype=np.float32)
        human_faces_np = np.asarray(human_faces, dtype=np.int32)
        human_vertex_normals_np = np.asarray(human_vertex_normals, dtype=np.float32)
        object_poses_np = np.asarray(object_poses, dtype=np.float32)
        intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
        human_albedo_np = np.asarray(human_albedo, dtype=np.float32)
        if human_vertices_np.ndim != 3 or human_vertices_np.shape[2] != 3 or not np.isfinite(human_vertices_np).all():
            raise ValueError(f"human_vertices must be finite [B,V,3], got {human_vertices_np.shape}")
        if human_faces_np.ndim != 2 or human_faces_np.shape[1] != 3 or human_faces_np.size == 0 or human_faces_np.min() < 0 or human_faces_np.max() >= human_vertices_np.shape[1]:
            raise ValueError(f"human_faces are incompatible with {human_vertices_np.shape[1]} vertices: {human_faces_np.shape}")
        if human_vertex_normals_np.shape != human_vertices_np.shape or not np.isfinite(human_vertex_normals_np).all():
            raise ValueError(f"human_vertex_normals must be finite and match human_vertices, got {human_vertex_normals_np.shape} and {human_vertices_np.shape}")
        if object_poses_np.shape != (human_vertices_np.shape[0], 4, 4) or not np.isfinite(object_poses_np).all():
            raise ValueError(f"object_poses must be finite [B,4,4], got {object_poses_np.shape}")
        if intrinsics_np.shape != (human_vertices_np.shape[0], 3, 3) or not np.isfinite(intrinsics_np).all():
            raise ValueError(f"intrinsics must be finite [B,3,3], got {intrinsics_np.shape}")
        if human_albedo_np.shape != (3,) or not np.isfinite(human_albedo_np).all() or np.any((human_albedo_np < 0.0) | (human_albedo_np > 1.0)):
            raise ValueError(f"human_albedo must be finite RGB in [0,1], got {human_albedo_np}")
        height, width = map(int, size)
        batch_size = human_vertices_np.shape[0]
        human_tensors = dict(self._load_constant_human_tensors(human_faces_np, human_vertices_np.shape[1], human_albedo_np))
        human_tensors["pos"] = torch.as_tensor(human_vertices_np, device=self.device, dtype=torch.float)
        human_tensors["vnormals"] = torch.as_tensor(human_vertex_normals_np, device=self.device, dtype=torch.float)
        identity_poses = torch.eye(4, device=self.device, dtype=torch.float).reshape(1, 4, 4).expand(batch_size, -1, -1).contiguous()
        object_poses_t = torch.as_tensor(object_poses_np, device=self.device, dtype=torch.float)
        with torch.inference_mode():
            rgb, human_depth, _ = self.utils.nvdiffrast_render(K=intrinsics_np, H=height, W=width, ob_in_cams=identity_poses, glctx=self.glctx, context="cuda", get_normal=False, mesh_tensors=human_tensors, output_size=(height, width), use_light=True, extra={})
            human_foreground = torch.isfinite(human_depth) & (human_depth > 0)
            depth = torch.where(human_foreground, human_depth, torch.full_like(human_depth, torch.inf))
            object_rgb, object_depth = render_visualization_object_parts(self._load_textured_object_tensors(object_mesh_source), object_poses_t, intrinsics_np, height, width)
            object_foreground = torch.isfinite(object_depth) & (object_depth > 0)
            object_front = object_foreground & (object_depth < depth)
            rgb = torch.where(object_front[..., None], object_rgb, rgb)
            foreground = human_foreground | object_front
        return rgb.detach().cpu().numpy(), foreground.detach().cpu().numpy().astype(bool, copy=False)

    def render_front_batch_textured_human_textured_object(self, human_vertices: np.ndarray, human_faces: np.ndarray, human_vertex_normals: np.ndarray, object_mesh_source: str, object_poses: np.ndarray, intrinsics: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        from lib_mhr.object_texture import render_visualization_object_parts

        human_vertices_np = np.asarray(human_vertices, dtype=np.float32)
        human_faces_np = np.asarray(human_faces, dtype=np.int32)
        human_vertex_normals_np = np.asarray(human_vertex_normals, dtype=np.float32)
        object_poses_np = np.asarray(object_poses, dtype=np.float32)
        intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
        if human_vertices_np.ndim != 3 or human_vertices_np.shape[2] != 3 or not np.isfinite(human_vertices_np).all():
            raise ValueError(f"human_vertices must be finite [B,V,3], got {human_vertices_np.shape}")
        if human_faces_np.ndim != 2 or human_faces_np.shape[1] != 3 or human_faces_np.size == 0 or human_faces_np.min() < 0 or human_faces_np.max() >= human_vertices_np.shape[1]:
            raise ValueError(f"human_faces are incompatible with {human_vertices_np.shape[1]} vertices: {human_faces_np.shape}")
        if human_vertex_normals_np.shape != human_vertices_np.shape or not np.isfinite(human_vertex_normals_np).all():
            raise ValueError(f"human_vertex_normals must be finite and match human_vertices, got {human_vertex_normals_np.shape} and {human_vertices_np.shape}")
        if object_poses_np.shape != (human_vertices_np.shape[0], 4, 4) or not np.isfinite(object_poses_np).all():
            raise ValueError(f"object_poses must be finite [B,4,4], got {object_poses_np.shape}")
        if intrinsics_np.shape != (human_vertices_np.shape[0], 3, 3) or not np.isfinite(intrinsics_np).all():
            raise ValueError(f"intrinsics must be finite [B,3,3], got {intrinsics_np.shape}")
        height, width = map(int, size)
        batch_size = human_vertices_np.shape[0]
        human_vertices_t = torch.as_tensor(human_vertices_np, device=self.device, dtype=torch.float)
        human_tensors = dict(self._load_textured_human_tensors(human_faces_np))
        human_tensors["pos"] = human_vertices_t
        human_tensors["vnormals"] = torch.as_tensor(human_vertex_normals_np, device=self.device, dtype=torch.float)
        identity_poses = torch.eye(4, device=self.device, dtype=torch.float).reshape(1, 4, 4).expand(batch_size, -1, -1).contiguous()
        object_poses_t = torch.as_tensor(object_poses_np, device=self.device, dtype=torch.float)
        with torch.inference_mode():
            rgb, human_depth, _ = self.utils.nvdiffrast_render(K=intrinsics_np, H=height, W=width, ob_in_cams=identity_poses, glctx=self.glctx, context="cuda", get_normal=False, mesh_tensors=human_tensors, output_size=(height, width), use_light=True, extra={})
            foreground = torch.isfinite(human_depth) & (human_depth > 0)
            depth = torch.where(foreground, human_depth, torch.full_like(human_depth, torch.inf))
            object_rgb, object_depth = render_visualization_object_parts(self._load_textured_object_tensors(object_mesh_source), object_poses_t, intrinsics_np, height, width)
            object_foreground = torch.isfinite(object_depth) & (object_depth > 0)
            object_front = object_foreground & (object_depth < depth)
            rgb = torch.where(object_front[..., None], object_rgb, rgb)
            foreground |= object_foreground
        return rgb.detach().cpu().numpy(), foreground.detach().cpu().numpy().astype(bool, copy=False)
