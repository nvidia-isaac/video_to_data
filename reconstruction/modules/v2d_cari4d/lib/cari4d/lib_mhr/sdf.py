from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .rotations import is_torch_tensor


def validate_closed_triangle_mesh(faces: Any) -> None:
    faces_arr = np.asarray(faces, dtype=np.int64)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3 or len(faces_arr) == 0:
        raise ValueError(f"triangle faces must have shape [F, 3], got {faces_arr.shape}")
    if faces_arr.min() < 0:
        raise ValueError("triangle faces contain negative vertex indices")
    edges = np.concatenate((faces_arr[:, [0, 1]], faces_arr[:, [1, 2]], faces_arr[:, [2, 0]]), axis=0)
    edges = np.sort(edges, axis=1)
    _unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    invalid_edges = int(np.count_nonzero(edge_counts != 2))
    if invalid_edges:
        raise ValueError(f"human mesh must be watertight; {invalid_edges} edges do not have exactly two incident faces")


def object_inside_human_penetration_loss(human_vertices: Any, human_faces: Any, object_points_world: Any, *, weight: float = 1.0, frame_chunk_size: int = 8, validate_face_indices: bool = True, point_reduction: str = "mean", conservative_aabb_rejection: bool = False, aabb_margin_m: float = 1e-6) -> Any:
    """Object-surface penetration depth inside a watertight human mesh, averaged over frames."""

    if weight == 0:
        if is_torch_tensor(object_points_world):
            return object_points_world.sum() * 0
        return 0.0
    if not is_torch_tensor(human_vertices) or not is_torch_tensor(object_points_world):
        raise TypeError("object-inside-human penetration requires PyTorch tensors")
    if human_vertices.ndim != 3 or human_vertices.shape[-1] != 3:
        raise ValueError(f"human_vertices must have shape [T, V, 3], got {tuple(human_vertices.shape)}")
    if object_points_world.ndim != 3 or object_points_world.shape[-1] != 3:
        raise ValueError(f"object_points_world must have shape [T, P, 3], got {tuple(object_points_world.shape)}")
    if human_vertices.shape[0] != object_points_world.shape[0]:
        raise ValueError(f"human and object frame counts differ: {human_vertices.shape[0]} vs {object_points_world.shape[0]}")
    if human_vertices.shape[0] == 0 or object_points_world.shape[1] == 0:
        raise ValueError("object-inside-human penetration requires at least one frame and one object point")
    if frame_chunk_size <= 0:
        raise ValueError(f"frame_chunk_size must be positive, got {frame_chunk_size}")
    if point_reduction not in {"mean", "sum"}:
        raise ValueError(f"point_reduction must be 'mean' or 'sum', got {point_reduction!r}")
    if aabb_margin_m < 0:
        raise ValueError(f"aabb_margin_m must be nonnegative, got {aabb_margin_m}")
    import torch
    faces = torch.as_tensor(human_faces, device=human_vertices.device, dtype=torch.long)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.numel() == 0:
        raise ValueError(f"human_faces must have shape [F, 3], got {tuple(faces.shape)}")
    if validate_face_indices and (int(faces.min()) < 0 or int(faces.max()) >= human_vertices.shape[1]):
        raise ValueError(f"human_faces contain indices outside [0, {human_vertices.shape[1]})")
    original_frame_count = human_vertices.shape[0]
    depth_sum = (human_vertices.sum() + object_points_world.sum()) * 0
    if conservative_aabb_rejection:
        human_min = human_vertices.detach().amin(dim=1) - float(aabb_margin_m)
        human_max = human_vertices.detach().amax(dim=1) + float(aabb_margin_m)
        object_min = object_points_world.detach().amin(dim=1)
        object_max = object_points_world.detach().amax(dim=1)
        active = ((object_max >= human_min) & (object_min <= human_max)).all(dim=1)
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            denominator = original_frame_count * (object_points_world.shape[1] if point_reduction == "mean" else 1)
            return depth_sum / denominator * float(weight)
        human_vertices = human_vertices.index_select(0, active_indices)
        object_points_world = object_points_world.index_select(0, active_indices)
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    from kaolin.ops.mesh import check_sign

    for start in range(0, human_vertices.shape[0], int(frame_chunk_size)):
        stop = min(human_vertices.shape[0], start + int(frame_chunk_size))
        vertices_chunk = human_vertices[start:stop]
        points_chunk = object_points_world[start:stop]
        inside = check_sign(vertices_chunk.detach(), faces, points_chunk.detach())
        squared_distance, _face_index, _distance_type = point_to_mesh_distance(points_chunk, vertices_chunk[:, faces].contiguous())
        depth = torch.sqrt(torch.clamp(squared_distance, min=1e-12))
        depth_sum = depth_sum + (depth * inside.to(dtype=depth.dtype)).sum()
    denominator = original_frame_count * (object_points_world.shape[1] if point_reduction == "mean" else 1)
    return depth_sum / denominator * float(weight)


def object_sdf_penetration_loss(
    points_obj: Any,
    sdf_fn: Callable[[Any], Any],
    *,
    weight: float = 1.0,
) -> Any:
    """One-way object SDF penetration loss for human surface points.

    `points_obj` are already in object-local coordinates. Negative SDF values
    are inside the object and contribute positive loss.
    """

    if weight == 0:
        if is_torch_tensor(points_obj):
            return points_obj.sum() * 0
        return 0.0

    sdf = sdf_fn(points_obj)
    if is_torch_tensor(sdf):
        import torch

        return torch.relu(-sdf).mean() * weight
    return float(np.maximum(-np.asarray(sdf), 0.0).mean() * weight)
