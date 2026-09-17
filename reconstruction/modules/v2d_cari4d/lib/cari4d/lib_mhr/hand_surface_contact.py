from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh


MHR_HAND_ORDER = ("left_hand", "right_hand")
MHR_HAND_VERTEX_COUNT = 2318
MHR_HAND_SAMPLE_COUNT = 256
MHR_CONTACT_THRESHOLD_M = 0.015
MHR_HAND_SURFACE_CONTACT_REVISION = "mhr-hand-surface-contact-v1"
MHR_HAND_SURFACE_CONTACT_PENDING_REVISION = "mhr-hand-surface-contact-pending-v1"
OBJECT_TOPOLOGY_CLASSES = frozenset(("watertight_solid", "nonwatertight_proxy", "intentionally_open_or_thin", "uncertain"))


@dataclass(frozen=True)
class MHRHandSurfaceSpec:
    vertex_indices: np.ndarray
    sample_local_indices: np.ndarray
    sample_assignments: np.ndarray
    hand_faces_local: tuple[np.ndarray, np.ndarray]
    rest_coverage_radius_m: np.ndarray
    topology_sha256: str
    faces_sha256: str
    mhr_model_sha256: str


@dataclass(frozen=True)
class ObjectContactMesh:
    vertices: np.ndarray
    faces: np.ndarray
    topology_class: str
    containment_vertices: np.ndarray | None
    containment_faces: np.ndarray | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MHRHandSurfaceContactDistances:
    distances: np.ndarray
    closest_points_world: np.ndarray
    coverage_radii_m: np.ndarray
    methods: np.ndarray
    diagnostics: Mapping[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def deterministic_farthest_point_indices(points: Any, sample_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"points must be finite [N,3], got {points.shape}")
    if sample_count <= 0 or sample_count > len(points):
        raise ValueError(f"sample_count must be in [1,{len(points)}], got {sample_count}")
    centroid = points.mean(axis=0)
    first = int(np.argmax(np.sum((points - centroid) ** 2, axis=1)))
    selected = np.empty(sample_count, dtype=np.int32)
    selected[0] = first
    min_dist2 = np.sum((points - points[first]) ** 2, axis=1)
    for index in range(1, sample_count):
        selected[index] = int(np.argmax(min_dist2))
        candidate_dist2 = np.sum((points - points[selected[index]]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, candidate_dist2)
    if len(np.unique(selected)) != sample_count:
        raise RuntimeError("deterministic farthest-point sampling selected duplicate vertices")
    return selected


def nearest_sample_assignments(points: Any, sample_indices: Any) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    samples = points[sample_indices]
    distances2 = np.sum((points[:, None, :] - samples[None, :, :]) ** 2, axis=-1)
    assignments = np.argmin(distances2, axis=1).astype(np.int32)
    radius = float(np.sqrt(distances2[np.arange(len(points)), assignments].max()))
    return assignments, radius


def _hand_joint_ids(joint_names: list[str], side: str) -> set[int]:
    prefix = "l" if side == "left" else "r"
    finger_prefixes = tuple(f"{prefix}_{finger}" for finger in ("thumb", "index", "middle", "ring", "pinky"))
    ids = {index for index, name in enumerate(joint_names) if name == f"{prefix}_wrist" or name.startswith(finger_prefixes)}
    if len(ids) != 23:
        raise ValueError(f"expected 23 {side} wrist/finger joints, found {len(ids)}: {sorted(ids)}")
    return ids


def _neutral_mhr_vertices_m(model: Any) -> np.ndarray:
    import torch

    identity_count = int(model.get_num_identity_blendshapes())
    identity = torch.zeros((1, identity_count), dtype=torch.float32)
    model_parameter_count = len(model.get_parameter_names()) - identity_count
    if model_parameter_count <= 0:
        raise ValueError(f"invalid MHR model parameter count: total={len(model.get_parameter_names())}, identity={identity_count}")
    model_parameters = torch.zeros((1, model_parameter_count), dtype=torch.float32)
    expression = torch.zeros((1, int(model.get_num_face_expression_blendshapes())), dtype=torch.float32)
    output = model(identity, model_parameters, expression)
    vertices = output[0] if isinstance(output, tuple) else output
    vertices = np.asarray(vertices.detach().cpu(), dtype=np.float32)
    if vertices.shape != (1, 18439, 3) or not np.isfinite(vertices).all():
        raise ValueError(f"neutral MHR vertices must be finite [1,18439,3], got {vertices.shape}")
    extent = float(np.ptp(vertices, axis=1).max())
    if not 100.0 < extent < 300.0:
        raise ValueError(f"neutral MHR model extent {extent} does not look centimeter-valued")
    return vertices[0] / 100.0


def build_mhr_hand_surface_spec(mhr_model_path: str | Path, faces: Any, sample_count: int = MHR_HAND_SAMPLE_COUNT) -> MHRHandSurfaceSpec:
    import torch

    mhr_model_path = Path(mhr_model_path)
    model = torch.jit.load(str(mhr_model_path), map_location="cpu")
    joint_names = [str(name) for name in model.get_joint_names()]
    lbs_indices, lbs_weights = model.get_lbsw()
    lbs_indices = np.asarray(lbs_indices.to(torch.int64).cpu(), dtype=np.int64)
    lbs_weights = np.asarray(lbs_weights.cpu(), dtype=np.float32)
    if lbs_indices.shape != (18439, 8) or lbs_weights.shape != (18439, 8) or not np.isfinite(lbs_weights).all():
        raise ValueError(f"unexpected MHR skinning arrays: indices={lbs_indices.shape}, weights={lbs_weights.shape}")
    dominant_slots = np.argmax(lbs_weights, axis=1)
    dominant_joints = lbs_indices[np.arange(len(lbs_indices)), dominant_slots]
    rest_vertices = _neutral_mhr_vertices_m(model)
    faces = np.asarray(faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(rest_vertices):
        raise ValueError(f"invalid MHR faces: {faces.shape}")
    vertex_indices = []
    sample_indices = []
    assignments = []
    hand_faces = []
    radii = []
    for side in ("left", "right"):
        joint_ids = _hand_joint_ids(joint_names, side)
        selected = np.flatnonzero(np.isin(dominant_joints, list(joint_ids))).astype(np.int32)
        if len(selected) != MHR_HAND_VERTEX_COUNT:
            raise ValueError(f"expected {MHR_HAND_VERTEX_COUNT} dominant {side} hand vertices, found {len(selected)}")
        inverse = np.full(len(rest_vertices), -1, dtype=np.int32)
        inverse[selected] = np.arange(len(selected), dtype=np.int32)
        face_mask = np.all(inverse[faces] >= 0, axis=1)
        local_faces = inverse[faces[face_mask]]
        if len(local_faces) == 0:
            raise ValueError(f"{side} hand has no complete triangles")
        sampled = deterministic_farthest_point_indices(rest_vertices[selected], sample_count)
        assigned, radius = nearest_sample_assignments(rest_vertices[selected], sampled)
        vertex_indices.append(selected)
        sample_indices.append(sampled)
        assignments.append(assigned)
        hand_faces.append(local_faces.astype(np.int32, copy=False))
        radii.append(radius)
    vertex_indices_array = np.stack(vertex_indices).astype(np.int32, copy=False)
    sample_indices_array = np.stack(sample_indices).astype(np.int32, copy=False)
    assignments_array = np.stack(assignments).astype(np.int32, copy=False)
    topology_hash = array_sha256(faces, vertex_indices_array, sample_indices_array, assignments_array, *hand_faces)
    return MHRHandSurfaceSpec(vertex_indices=vertex_indices_array, sample_local_indices=sample_indices_array, sample_assignments=assignments_array, hand_faces_local=(hand_faces[0], hand_faces[1]), rest_coverage_radius_m=np.asarray(radii, dtype=np.float32), topology_sha256=topology_hash, faces_sha256=array_sha256(faces), mhr_model_sha256=file_sha256(mhr_model_path))


def save_mhr_hand_surface_spec(path: str | Path, spec: MHRHandSurfaceSpec) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, vertex_indices=spec.vertex_indices, sample_local_indices=spec.sample_local_indices, sample_assignments=spec.sample_assignments, hand_faces_left=spec.hand_faces_local[0], hand_faces_right=spec.hand_faces_local[1], rest_coverage_radius_m=spec.rest_coverage_radius_m, topology_sha256=np.asarray(spec.topology_sha256), faces_sha256=np.asarray(spec.faces_sha256), mhr_model_sha256=np.asarray(spec.mhr_model_sha256))
    temporary.replace(path)
    return path


def load_mhr_hand_surface_spec(path: str | Path, *, faces: Any | None = None) -> MHRHandSurfaceSpec:
    with np.load(path, allow_pickle=False) as payload:
        spec = MHRHandSurfaceSpec(vertex_indices=np.asarray(payload["vertex_indices"], dtype=np.int32), sample_local_indices=np.asarray(payload["sample_local_indices"], dtype=np.int32), sample_assignments=np.asarray(payload["sample_assignments"], dtype=np.int32), hand_faces_local=(np.asarray(payload["hand_faces_left"], dtype=np.int32), np.asarray(payload["hand_faces_right"], dtype=np.int32)), rest_coverage_radius_m=np.asarray(payload["rest_coverage_radius_m"], dtype=np.float32), topology_sha256=str(payload["topology_sha256"].item()), faces_sha256=str(payload["faces_sha256"].item()), mhr_model_sha256=str(payload["mhr_model_sha256"].item()))
    if spec.vertex_indices.shape != (2, MHR_HAND_VERTEX_COUNT) or spec.sample_local_indices.shape != (2, MHR_HAND_SAMPLE_COUNT) or spec.sample_assignments.shape != (2, MHR_HAND_VERTEX_COUNT):
        raise ValueError(f"invalid cached MHR hand-surface spec shapes: vertices={spec.vertex_indices.shape}, samples={spec.sample_local_indices.shape}, assignments={spec.sample_assignments.shape}")
    if faces is not None and array_sha256(np.asarray(faces, dtype=np.int32)) != spec.faces_sha256:
        raise ValueError("cached MHR hand-surface spec does not match packed MHR faces")
    expected_topology = array_sha256(np.asarray(faces, dtype=np.int32), spec.vertex_indices, spec.sample_local_indices, spec.sample_assignments, *spec.hand_faces_local) if faces is not None else spec.topology_sha256
    if expected_topology != spec.topology_sha256:
        raise ValueError("cached MHR hand-surface topology hash is invalid")
    return spec


def _validated_mesh_arrays(vertices: Any, faces: Any, label: str) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"{label} vertices must be finite [V,3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{label} faces are invalid: {faces.shape}")
    return vertices, faces


def _is_valid_watertight_solid(mesh: trimesh.Trimesh) -> bool:
    bbox_volume = float(np.prod(np.maximum(mesh.bounds[1] - mesh.bounds[0], 1e-9)))
    return bool(mesh.is_watertight and mesh.is_winding_consistent and np.isfinite(mesh.volume) and abs(float(mesh.volume)) > bbox_volume * 1e-8)


def classify_object_contact_mesh(vertices: Any, faces: Any, *, intentionally_open_or_thin: bool = False, proxy_max_added_area_fraction: float = 0.01, proxy_max_added_face_fraction: float = 0.01) -> ObjectContactMesh:
    vertices, faces = _validated_mesh_arrays(vertices, faces, "object")
    source_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    topology_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    edge_counts = np.bincount(topology_mesh.edges_unique_inverse, minlength=len(topology_mesh.edges_unique))
    boundary_edge_count = int(np.count_nonzero(edge_counts == 1))
    metadata = {"is_watertight": bool(topology_mesh.is_watertight), "is_winding_consistent": bool(topology_mesh.is_winding_consistent), "source_vertex_count": int(len(vertices)), "source_face_count": int(len(faces)), "topology_vertex_count": int(len(topology_mesh.vertices)), "topology_face_count": int(len(topology_mesh.faces)), "boundary_edge_count": boundary_edge_count, "surface_area_m2": float(source_mesh.area), "topology_processing": "trimesh process=True coordinate welding; source triangles retained for distance and intersection"}
    if intentionally_open_or_thin:
        metadata = {**metadata, "containment_mode": "disabled_for_intentionally_open_or_thin"}
        return ObjectContactMesh(vertices=vertices, faces=faces, topology_class="intentionally_open_or_thin", containment_vertices=None, containment_faces=None, metadata=metadata)
    if topology_mesh.is_watertight:
        containment = topology_mesh.copy()
        if not containment.is_winding_consistent:
            containment.fix_normals(multibody=True)
        if _is_valid_watertight_solid(containment):
            metadata = {**metadata, "containment_mode": "watertight_winding_number"}
            return ObjectContactMesh(vertices=vertices, faces=faces, topology_class="watertight_solid", containment_vertices=np.asarray(containment.vertices, dtype=np.float32), containment_faces=np.asarray(containment.faces, dtype=np.int32), metadata=metadata)
    repaired = topology_mesh.copy()
    original_face_count = len(repaired.faces)
    original_area = max(float(repaired.area), 1e-12)
    trimesh.repair.fill_holes(repaired)
    repaired.fix_normals(multibody=True)
    added_face_fraction = max(0, len(repaired.faces) - original_face_count) / max(1, original_face_count)
    added_area_fraction = max(0.0, float(repaired.area) - original_area) / original_area
    if _is_valid_watertight_solid(repaired) and added_face_fraction <= proxy_max_added_face_fraction and added_area_fraction <= proxy_max_added_area_fraction:
        metadata = {**metadata, "containment_mode": "validated_small_hole_proxy", "proxy_added_face_fraction": added_face_fraction, "proxy_added_area_fraction": added_area_fraction, "proxy_face_count": int(len(repaired.faces))}
        return ObjectContactMesh(vertices=vertices, faces=faces, topology_class="nonwatertight_proxy", containment_vertices=np.asarray(repaired.vertices, dtype=np.float32), containment_faces=np.asarray(repaired.faces, dtype=np.int32), metadata=metadata)
    metadata = {**metadata, "containment_mode": "winding_number_and_ray_vote_agreement", "proxy_added_face_fraction": added_face_fraction, "proxy_added_area_fraction": added_area_fraction}
    return ObjectContactMesh(vertices=vertices, faces=faces, topology_class="uncertain", containment_vertices=np.asarray(topology_mesh.vertices, dtype=np.float32), containment_faces=np.asarray(topology_mesh.faces, dtype=np.int32), metadata=metadata)


def hand_surface_contact_metadata(spec: MHRHandSurfaceSpec, object_mesh: ObjectContactMesh) -> dict[str, Any]:
    return {"mhr_contact_revision": MHR_HAND_SURFACE_CONTACT_REVISION, "mhr_contact_definition": "minimum posed MHR hand-vertex to object-surface distance with explicit surface-intersection and validated-containment zeroing", "mhr_contact_distance_units": "meters", "mhr_contact_hand_order": list(MHR_HAND_ORDER), "mhr_contact_hand_vertex_count": [int(len(indices)) for indices in spec.vertex_indices], "mhr_contact_sample_count": int(spec.sample_local_indices.shape[1]), "mhr_contact_threshold_m": MHR_CONTACT_THRESHOLD_M, "mhr_contact_sampling": "deterministic farthest-point vertices in neutral MHR geometry", "mhr_contact_coverage_radius": "per-frame posed radius using fixed neutral-geometry nearest-sample assignments", "mhr_contact_topology_sha256": spec.topology_sha256, "mhr_contact_mhr_model_sha256": spec.mhr_model_sha256, "mhr_contact_object_topology_class": object_mesh.topology_class, "mhr_contact_object_topology": dict(object_mesh.metadata), "mhr_contact_closest_points_space": "world", "mhr_contact_distance_value": "sampled upper bound outside the threshold ambiguity band; exhaustive 2318-vertex minimum inside the ambiguity band"}
