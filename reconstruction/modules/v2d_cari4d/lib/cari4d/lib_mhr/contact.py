from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh


MHR_WRIST_KEYPOINTS: Mapping[str, int] = {
    "left_wrist": 62,
    "right_wrist": 41,
}
MHR_WRIST_NAMES = tuple(MHR_WRIST_KEYPOINTS.keys())
MHR_WRIST_INDICES = tuple(MHR_WRIST_KEYPOINTS[name] for name in MHR_WRIST_NAMES)
BEHAVE_OBJECT_MESH_POSE_FRAME_REVISION = "behave-registration-template-center-v1"
BEHAVE_CONTACT_REVISION = "behave-centered-object-mesh-contact-v1"
OBJECT_MESH_SCENE_LOADER_REVISION = "trimesh-scene-node-transforms-v1"


@dataclass(frozen=True)
class MHRContactDistances:
    distances: np.ndarray
    closest_points_world: np.ndarray
    wrist_points_world: np.ndarray
    closest_points_local: np.ndarray


def validate_object_mesh_to_pose_transform(value: Any, label: str = "object_mesh_to_pose_transform") -> np.ndarray:
    transform = np.asarray(value, dtype=np.float32)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix, got {transform.shape}")
    if not np.allclose(transform[3], np.array([0, 0, 0, 1], dtype=np.float32), rtol=0.0, atol=1e-6):
        raise ValueError(f"{label} must have homogeneous bottom row [0,0,0,1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3, dtype=np.float32), rtol=0.0, atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{label} must contain a proper rigid rotation")
    return transform


def transform_object_mesh_to_pose_frame(vertices: Any, object_mesh_to_pose_transform: Any) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    transform = validate_object_mesh_to_pose_transform(object_mesh_to_pose_transform)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"object mesh vertices must be finite [V,3], got {vertices.shape}")
    return (vertices @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32, copy=False)


def compose_object_mesh_poses(object_poses: Any, object_mesh_to_pose_transform: Any) -> np.ndarray:
    poses = np.asarray(object_poses, dtype=np.float32)
    transform = validate_object_mesh_to_pose_transform(object_mesh_to_pose_transform)
    if poses.ndim < 2 or poses.shape[-2:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"object poses must be finite [...,4,4], got {poses.shape}")
    return (poses @ transform).astype(np.float32, copy=False)


def load_object_mesh(path: str | Path) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(path, force="scene", process=False)
    if isinstance(mesh_or_scene, trimesh.Scene):
        geometries = [geom for geom in mesh_or_scene.dump(concatenate=False) if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0]
        if not geometries:
            raise ValueError(f"{path} has no mesh geometry")
        if len(geometries) == 1:
            return geometries[0]
        return trimesh.util.concatenate(geometries)
    return mesh_or_scene


def sample_mesh_surface_points(vertices: Any, faces: Any, sample_count: int, *, seed: int = 0) -> np.ndarray:
    vertices_arr = np.asarray(vertices, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int64)
    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3 or len(vertices_arr) == 0:
        raise ValueError(f"mesh vertices must have shape [V, 3], got {vertices_arr.shape}")
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3 or len(faces_arr) == 0:
        raise ValueError(f"mesh faces must have shape [F, 3], got {faces_arr.shape}")
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if faces_arr.min() < 0 or faces_arr.max() >= len(vertices_arr):
        raise ValueError(f"mesh faces contain indices outside [0, {len(vertices_arr)})")
    triangles = vertices_arr[faces_arr]
    double_areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1).astype(np.float64)
    total_area = float(double_areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("mesh has no finite non-degenerate surface area")
    rng = np.random.default_rng(seed)
    sampled_faces = rng.choice(len(faces_arr), size=int(sample_count), p=double_areas / total_area)
    selected = triangles[sampled_faces]
    barycentric_u = np.sqrt(rng.random(int(sample_count), dtype=np.float32))
    barycentric_v = rng.random(int(sample_count), dtype=np.float32)
    points = (1.0 - barycentric_u)[:, None] * selected[:, 0] + (barycentric_u * (1.0 - barycentric_v))[:, None] * selected[:, 1] + (barycentric_u * barycentric_v)[:, None] * selected[:, 2]
    return points.astype(np.float32, copy=False)


def mhr_wrist_points_from_keypoints(mhr_keypoints: Any) -> np.ndarray:
    keypoints = np.asarray(mhr_keypoints, dtype=np.float32)
    min_keypoints = max(MHR_WRIST_INDICES) + 1
    if keypoints.shape[-2] < min_keypoints:
        raise ValueError(f"MHR keypoints need at least {min_keypoints} points, got shape {keypoints.shape}")
    return keypoints[..., list(MHR_WRIST_INDICES), :].astype(np.float32, copy=False)


def object_points_world_to_local(points_world: Any, obj_rot: Any, obj_t: Any) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float32)
    obj_rot = np.asarray(obj_rot, dtype=np.float32)
    obj_t = np.asarray(obj_t, dtype=np.float32)
    return np.einsum("...pj,...jk->...pk", points_world - obj_t[..., None, :], obj_rot).astype(
        np.float32, copy=False
    )


def object_points_local_to_world(points_local: Any, obj_rot: Any, obj_t: Any) -> np.ndarray:
    points_local = np.asarray(points_local, dtype=np.float32)
    obj_rot = np.asarray(obj_rot, dtype=np.float32)
    obj_t = np.asarray(obj_t, dtype=np.float32)
    return (
        np.einsum("...pj,...kj->...pk", points_local, obj_rot).astype(np.float32, copy=False)
        + obj_t[..., None, :]
    )


def transform_object_vertices(vertices: Any, obj_rot: Any, obj_t: Any) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    obj_rot = np.asarray(obj_rot, dtype=np.float32)
    obj_t = np.asarray(obj_t, dtype=np.float32)
    return (vertices @ obj_rot.T + obj_t).astype(np.float32, copy=False)


def _closest_points_by_triangle_scan(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    *,
    chunk_size: int = 16384,
) -> tuple[np.ndarray, np.ndarray]:
    from trimesh.triangles import closest_point

    triangles = vertices[faces]
    closest_all = []
    dist_all = []
    for point in points:
        best_dist2 = np.inf
        best_point = None
        for start in range(0, len(triangles), chunk_size):
            tri_chunk = triangles[start : start + chunk_size]
            point_chunk = np.broadcast_to(point, (len(tri_chunk), 3))
            closest = closest_point(tri_chunk, point_chunk)
            dist2 = np.sum((closest - point) ** 2, axis=-1)
            best_idx = int(np.argmin(dist2))
            if float(dist2[best_idx]) < best_dist2:
                best_dist2 = float(dist2[best_idx])
                best_point = closest[best_idx]
        closest_all.append(best_point)
        dist_all.append(np.sqrt(best_dist2))
    return np.asarray(closest_all, dtype=np.float32), np.asarray(dist_all, dtype=np.float32)


def closest_points_on_object_mesh(
    points_local: Any,
    object_vertices: Any,
    object_faces: Any,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_local, dtype=np.float32).reshape(-1, 3)
    vertices = np.asarray(object_vertices, dtype=np.float32)
    faces = np.asarray(object_faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    try:
        closest, distances, _triangle_ids = trimesh.proximity.closest_point(mesh, points)
        return closest.astype(np.float32, copy=False), distances.astype(np.float32, copy=False)
    except Exception:
        return _closest_points_by_triangle_scan(vertices, faces, points)


def compute_mhr_wrist_object_distances(
    mhr_keypoints: Any,
    obj_rot: Any,
    obj_t: Any,
    object_vertices: Any,
    object_faces: Any,
) -> MHRContactDistances:
    wrists_world = mhr_wrist_points_from_keypoints(mhr_keypoints)
    obj_rot_arr = np.asarray(obj_rot, dtype=np.float32)
    obj_t_arr = np.asarray(obj_t, dtype=np.float32)
    leading_shape = wrists_world.shape[:-2]
    if obj_rot_arr.shape[:-2] != leading_shape:
        raise ValueError(f"obj_rot leading shape {obj_rot_arr.shape[:-2]} does not match wrists {leading_shape}")
    if obj_t_arr.shape[:-1] != leading_shape:
        raise ValueError(f"obj_t leading shape {obj_t_arr.shape[:-1]} does not match wrists {leading_shape}")

    wrists_local = object_points_world_to_local(wrists_world, obj_rot_arr, obj_t_arr)
    closest_flat, dist_flat = closest_points_on_object_mesh(wrists_local, object_vertices, object_faces)
    closest_local = closest_flat.reshape(*leading_shape, len(MHR_WRIST_INDICES), 3)
    distances = dist_flat.reshape(*leading_shape, len(MHR_WRIST_INDICES))
    closest_world = object_points_local_to_world(closest_local, obj_rot_arr, obj_t_arr)
    return MHRContactDistances(
        distances=distances.astype(np.float32, copy=False),
        closest_points_world=closest_world.astype(np.float32, copy=False),
        wrist_points_world=wrists_world.astype(np.float32, copy=False),
        closest_points_local=closest_local.astype(np.float32, copy=False),
    )
