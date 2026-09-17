from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


MHR_OBJECT_RENDER_MATERIAL_REVISION = "mhr-object-original-uv-normal-light-v1"
MHR_OBJECT_VISUALIZATION_RASTERIZER_REVISION = "mhr-object-pytorch3d-perspective-face-bin-safe-v2"


def _texture_rgb(texture_image: Any) -> np.ndarray:
    from PIL import Image

    image = texture_image.convert("RGB") if isinstance(texture_image, Image.Image) else Image.fromarray(np.asarray(texture_image)).convert("RGB")
    texture = np.asarray(image, dtype=np.float32)[..., :3] / 255.0
    if texture.ndim != 3 or texture.shape[2] != 3 or not np.isfinite(texture).all():
        raise ValueError(f"object texture must be finite RGB, got {texture.shape}")
    return texture


def load_object_mesh_parts(mesh_source: str | Path) -> tuple[Any, ...]:
    import trimesh

    mesh_source = Path(mesh_source).expanduser()
    mesh_or_scene = trimesh.load(mesh_source, force="scene", process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = tuple(mesh for mesh in mesh_or_scene.dump(concatenate=False) if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0 and len(mesh.faces) > 0)
    elif isinstance(mesh_or_scene, trimesh.Trimesh):
        meshes = (mesh_or_scene,)
    else:
        raise TypeError(f"unsupported object mesh type from {mesh_source}: {type(mesh_or_scene).__name__}")
    if not meshes:
        raise ValueError(f"object mesh contains no triangle geometry: {mesh_source}")
    return meshes


def concatenate_object_mesh_parts(meshes: tuple[Any, ...]) -> Any:
    import trimesh

    if not meshes:
        raise ValueError("cannot concatenate an empty object mesh")
    return trimesh.util.concatenate(meshes)


def make_original_object_visual_tensors(mesh: Any, device: Any) -> dict[str, Any]:
    import torch
    import trimesh

    visual = mesh.visual
    material = getattr(visual, "material", None)
    texture_image = None if material is None else getattr(material, "image", None)
    if texture_image is None and material is not None:
        texture_image = getattr(material, "baseColorTexture", None)
    has_uv_texture = isinstance(visual, trimesh.visual.texture.TextureVisuals) and texture_image is not None and getattr(visual, "uv", None) is not None
    if has_uv_texture:
        texture = _texture_rgb(texture_image)
        base_color_factor = getattr(material, "baseColorFactor", None)
        if base_color_factor is not None:
            factor = np.asarray(base_color_factor, dtype=np.float32).reshape(-1)[:3]
            if float(factor.max()) > 1.0:
                factor /= 255.0
            texture *= np.clip(factor, 0.0, 1.0).reshape(1, 1, 3)
        uv = np.asarray(visual.uv, dtype=np.float32).copy()
        if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
            raise ValueError(f"object UV coordinates must be finite [V,2] aligned with {len(mesh.vertices)} vertices, got {uv.shape}")
        uv[:, 1] = 1.0 - uv[:, 1]
        tensors = {
            "tex": torch.as_tensor(texture, device=device, dtype=torch.float)[None],
            "uv_idx": torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), device=device, dtype=torch.int),
            "uv": torch.as_tensor(uv, device=device, dtype=torch.float),
        }
    else:
        colors = getattr(visual, "vertex_colors", None)
        if colors is None or len(colors) != len(mesh.vertices):
            raise ValueError("object geometry has neither an original UV texture nor per-vertex colors")
        colors = np.asarray(colors, dtype=np.float32)[..., :3]
        if float(colors.max()) > 1.0:
            colors /= 255.0
        tensors = {"vertex_color": torch.as_tensor(np.clip(colors, 0.0, 1.0), device=device, dtype=torch.float)}
    tensors.update({
        "pos": torch.as_tensor(np.asarray(mesh.vertices, dtype=np.float32), device=device, dtype=torch.float),
        "faces": torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), device=device, dtype=torch.int),
        "vnormals": torch.as_tensor(np.asarray(mesh.vertex_normals, dtype=np.float32), device=device, dtype=torch.float),
    })
    return tensors


def load_original_object_visual_tensors(mesh_source: str | Path, device: Any) -> tuple[dict[str, Any], ...]:
    return tuple(make_original_object_visual_tensors(mesh, device) for mesh in load_object_mesh_parts(mesh_source))


def render_original_object_parts(mesh_tensors_parts: tuple[dict[str, Any], ...], object_poses: Any, K: Any, height: int, width: int, glctx: Any, output_size: tuple[int, int], bbox2d: Any = None) -> tuple[Any, Any]:
    import torch
    import Utils

    if not mesh_tensors_parts:
        raise ValueError("object rendering requires at least one material component")
    device = mesh_tensors_parts[0]["pos"].device
    object_poses = torch.as_tensor(object_poses, device=device, dtype=torch.float)
    batch_size = len(object_poses)
    rgb = torch.zeros((batch_size, int(height), int(width), 3), device=device, dtype=torch.float)
    depth = torch.full((batch_size, int(height), int(width)), torch.inf, device=device, dtype=torch.float)
    for mesh_tensors in mesh_tensors_parts:
        part_rgb, part_depth, _ = Utils.nvdiffrast_render(K=K, H=height, W=width, ob_in_cams=object_poses, context="cuda", get_normal=False, glctx=glctx, mesh_tensors=mesh_tensors, output_size=output_size, bbox2d=bbox2d, use_light=True, extra={})
        valid = torch.isfinite(part_depth) & (part_depth > 0)
        front = valid & (part_depth < depth)
        rgb = torch.where(front[..., None], part_rgb, rgb)
        depth = torch.where(front, part_depth, depth)
    return rgb, torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))


def render_visualization_object_parts(mesh_tensors_parts: tuple[dict[str, Any], ...], object_poses: Any, K: Any, height: int, width: int, use_light: bool = True) -> tuple[Any, Any]:
    """Render textured visualization objects without nvdiffrast 0.3 coplanar-face artifacts."""
    import torch
    import torch.nn.functional as F
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, TexturesUV
    from pytorch3d.renderer.mesh.textures import TexturesVertex
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    if not mesh_tensors_parts:
        raise ValueError("object visualization rendering requires at least one material component")
    device = mesh_tensors_parts[0]["pos"].device
    object_poses = torch.as_tensor(object_poses, device=device, dtype=torch.float)
    intrinsics = torch.as_tensor(K, device=device, dtype=torch.float)
    if object_poses.ndim != 3 or object_poses.shape[1:] != (4, 4) or not torch.isfinite(object_poses).all():
        raise ValueError(f"object poses must be finite [B,4,4], got {tuple(object_poses.shape)}")
    batch_size = len(object_poses)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None].expand(batch_size, -1, -1)
    if intrinsics.shape != (batch_size, 3, 3) or not torch.isfinite(intrinsics).all():
        raise ValueError(f"intrinsics must be finite [B,3,3], got {tuple(intrinsics.shape)}")
    image_size = torch.tensor([int(height), int(width)], device=device, dtype=torch.float).reshape(1, 2).expand(batch_size, -1)
    cameras = cameras_from_opencv_projection(torch.eye(3, device=device, dtype=torch.float)[None].expand(batch_size, -1, -1), torch.zeros((batch_size, 3), device=device, dtype=torch.float), intrinsics, image_size)
    max_faces_per_bin = max(int(mesh_tensors["faces"].shape[0]) for mesh_tensors in mesh_tensors_parts)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=(int(height), int(width)), blur_radius=0.0, faces_per_pixel=1, perspective_correct=True, cull_backfaces=False, max_faces_per_bin=max_faces_per_bin))
    rgb = torch.zeros((batch_size, int(height), int(width), 3), device=device, dtype=torch.float)
    depth = torch.full((batch_size, int(height), int(width)), torch.inf, device=device, dtype=torch.float)
    for mesh_tensors in mesh_tensors_parts:
        source_vertices = mesh_tensors["pos"]
        faces = mesh_tensors["faces"].to(device=device, dtype=torch.long)
        source_normals = mesh_tensors["vnormals"]
        vertices = torch.einsum("bij,vj->bvi", object_poses[:, :3, :3], source_vertices) + object_poses[:, None, :3, 3]
        normals = torch.einsum("bij,vj->bvi", object_poses[:, :3, :3], source_normals)
        faces_batch = faces[None].expand(batch_size, -1, -1)
        if "tex" in mesh_tensors:
            source_uv = mesh_tensors["uv"].clone()
            source_uv[:, 1] = 1.0 - source_uv[:, 1]
            textures = TexturesUV(maps=mesh_tensors["tex"].expand(batch_size, -1, -1, -1), faces_uvs=mesh_tensors["uv_idx"].to(device=device, dtype=torch.long)[None].expand(batch_size, -1, -1), verts_uvs=source_uv[None].expand(batch_size, -1, -1), padding_mode="border", align_corners=True, sampling_mode="bilinear")
        else:
            textures = TexturesVertex(verts_features=mesh_tensors["vertex_color"][None].expand(batch_size, -1, -1))
        meshes = Meshes(verts=vertices, faces=faces_batch, textures=textures)
        fragments = rasterizer(meshes)
        foreground = fragments.pix_to_face[..., 0] >= 0
        part_rgb = meshes.sample_textures(fragments)[..., 0, :3]
        if use_light:
            face_normals = normals[:, faces, :].reshape(-1, 3, 3)
            normal_map = F.normalize(interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_normals)[..., 0, :], dim=-1)
            diffuse = torch.clamp(-normal_map[..., 2:3], 0.0, 1.0)
            part_rgb = torch.clamp(part_rgb * (0.8 + 0.5 * diffuse), 0.0, 1.0)
        face_depths = vertices[:, faces, 2:3].reshape(-1, 3, 1)
        part_depth = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_depths)[..., 0, 0]
        foreground &= torch.isfinite(part_depth) & (part_depth > 0)
        part_depth = torch.where(foreground, part_depth, torch.full_like(part_depth, torch.inf))
        front = foreground & (part_depth < depth)
        rgb = torch.where(front[..., None], part_rgb, rgb)
        depth = torch.where(front, part_depth, depth)
    return rgb, torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
