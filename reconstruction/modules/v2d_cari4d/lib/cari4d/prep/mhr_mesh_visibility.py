from __future__ import annotations

import numpy as np


def visible_entity_masks_from_triangle_ids(triangle_ids: np.ndarray, human_face_count: int) -> tuple[np.ndarray, np.ndarray]:
    triangle_ids = np.asarray(triangle_ids)
    if triangle_ids.ndim != 3:
        raise ValueError(f"Triangle IDs must have shape [B,H,W], got {triangle_ids.shape}")
    human_face_count = int(human_face_count)
    if human_face_count <= 0:
        raise ValueError(f"human_face_count must be positive, got {human_face_count}")
    human = (triangle_ids >= 0) & (triangle_ids < human_face_count)
    object_mask = triangle_ids >= human_face_count
    if np.any(human & object_mask):
        raise RuntimeError("Joint raster assigned one pixel to both human and object")
    return human, object_mask


def _rasterize_triangle_ids(vertices_camera: np.ndarray, faces: np.ndarray, K: np.ndarray, image_shape: tuple[int, int], *, raster_context=None) -> np.ndarray:
    import nvdiffrast.torch as dr
    import torch
    import Utils

    vertices_camera = np.asarray(vertices_camera, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices_camera.ndim != 3 or vertices_camera.shape[-1] != 3 or vertices_camera.shape[0] == 0:
        raise ValueError(f"Camera vertices must have shape [B,V,3], got {vertices_camera.shape}")
    if not np.isfinite(vertices_camera).all():
        raise ValueError("Ground-truth render vertices contain nonfinite values")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0 or faces.max() >= vertices_camera.shape[1]:
        raise ValueError(f"Faces are incompatible with vertices: {faces.shape}/{vertices_camera.shape}")
    height, width = int(image_shape[0]), int(image_shape[1])
    device = torch.device("cuda")
    vertices = torch.as_tensor(vertices_camera, device=device)
    faces_tensor = torch.as_tensor(faces, dtype=torch.int32, device=device)
    homogeneous = torch.cat([vertices, torch.ones((*vertices.shape[:2], 1), dtype=vertices.dtype, device=device)], dim=-1)
    projection = torch.as_tensor(Utils.projection_matrix_from_intrinsics(np.asarray(K, dtype=np.float32), height=height, width=width, znear=0.001, zfar=100.0), dtype=torch.float32, device=device)
    cv_to_gl = torch.as_tensor(Utils.glcam_in_cvcam, dtype=torch.float32, device=device)
    clip = torch.matmul((projection @ cv_to_gl)[None, None], homogeneous[..., None])[..., 0]
    raster_context = dr.RasterizeCudaContext() if raster_context is None else raster_context
    raster, _ = dr.rasterize(raster_context, clip, faces_tensor, resolution=np.asarray([height, width]))
    return torch.flip(raster[..., 3].to(torch.int64) - 1, dims=[1]).cpu().numpy()


def render_joint_visible_masks(human_vertices_camera: np.ndarray, human_faces: np.ndarray, object_vertices_camera: np.ndarray, object_faces: np.ndarray, K: np.ndarray, image_shape: tuple[int, int], *, raster_context=None) -> tuple[np.ndarray, np.ndarray]:
    human_vertices_camera = np.asarray(human_vertices_camera, dtype=np.float32)
    object_vertices_camera = np.asarray(object_vertices_camera, dtype=np.float32)
    human_faces = np.asarray(human_faces, dtype=np.int32)
    object_faces = np.asarray(object_faces, dtype=np.int32)
    if human_vertices_camera.ndim != 3 or object_vertices_camera.ndim != 3 or human_vertices_camera.shape[0] != object_vertices_camera.shape[0] or human_vertices_camera.shape[-1] != 3 or object_vertices_camera.shape[-1] != 3:
        raise ValueError(f"Camera vertices must be matching [B,V,3] arrays, got {human_vertices_camera.shape} and {object_vertices_camera.shape}")
    vertices = np.concatenate([human_vertices_camera, object_vertices_camera], axis=1)
    faces = np.concatenate([human_faces, object_faces + human_vertices_camera.shape[1]], axis=0)
    triangle_ids = _rasterize_triangle_ids(vertices, faces, K, image_shape, raster_context=raster_context)
    return visible_entity_masks_from_triangle_ids(triangle_ids, len(human_faces))


def render_visible_mesh_mask(vertices_camera: np.ndarray, faces: np.ndarray, K: np.ndarray, image_shape: tuple[int, int], *, raster_context=None) -> np.ndarray:
    return _rasterize_triangle_ids(vertices_camera, faces, K, image_shape, raster_context=raster_context) >= 0

