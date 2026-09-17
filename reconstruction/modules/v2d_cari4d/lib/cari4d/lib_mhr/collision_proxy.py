from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .human_texture import mhr_faces_sha256
from .sdf import validate_closed_triangle_mesh


MHR_COLLISION_PROXY_SCHEMA = "cari4d.mhr_collision_proxy.v1"
MHR_COLLISION_PROXY_REVISION = "mhr-collision-proxy-qem-4000v-v1"
MHR_COLLISION_PROXY_MODE_4000 = "proxy_4000"
MHR_COLLISION_PROXY_MODE_FULL = "full"
MHR_COLLISION_PROXY_MODES = (MHR_COLLISION_PROXY_MODE_4000, MHR_COLLISION_PROXY_MODE_FULL)
DEFAULT_MHR_COLLISION_PROXY_ASSET = Path(__file__).resolve().parent / "assets" / "mhr_collision_proxy_4000v.npz"


@dataclass(frozen=True)
class MHRCollisionProxy:
    source_vertex_count: int
    source_face_count: int
    source_faces_sha256: str
    source_vertex_indices: np.ndarray
    barycentric_weights: np.ndarray
    faces: np.ndarray
    reference_vertices: np.ndarray
    revision: str

    @property
    def vertex_count(self) -> int:
        return int(self.source_vertex_indices.shape[0])


@lru_cache(maxsize=4)
def _load_mhr_collision_proxy_asset(asset_path: str) -> MHRCollisionProxy:
    with np.load(asset_path, allow_pickle=False) as payload:
        schema = str(payload["schema"].item())
        revision = str(payload["revision"].item())
        source_vertex_count = int(payload["source_vertex_count"].item())
        source_face_count = int(payload["source_face_count"].item())
        source_faces_hash = str(payload["source_faces_sha256"].item())
        source_vertex_indices = np.asarray(payload["source_vertex_indices"], dtype=np.int32)
        barycentric_weights = np.asarray(payload["barycentric_weights"], dtype=np.float32)
        proxy_faces = np.asarray(payload["proxy_faces"], dtype=np.int32)
        reference_vertices = np.asarray(payload["reference_proxy_vertices"], dtype=np.float32)
    if schema != MHR_COLLISION_PROXY_SCHEMA or revision != MHR_COLLISION_PROXY_REVISION:
        raise ValueError(f"unsupported MHR collision proxy {schema!r}/{revision!r}")
    if source_vertex_count <= 0 or source_face_count <= 0:
        raise ValueError(f"invalid MHR collision-proxy source topology {source_vertex_count}/{source_face_count}")
    if source_vertex_indices.shape != (4000, 3) or source_vertex_indices.min() < 0 or source_vertex_indices.max() >= source_vertex_count:
        raise ValueError(f"invalid MHR collision-proxy source indices {source_vertex_indices.shape}")
    if barycentric_weights.shape != source_vertex_indices.shape or not np.isfinite(barycentric_weights).all() or not np.allclose(barycentric_weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"invalid MHR collision-proxy barycentric weights {barycentric_weights.shape}")
    if np.any(barycentric_weights < -1e-5) or np.any(barycentric_weights > 1.0 + 1e-5):
        raise ValueError("MHR collision-proxy barycentric weights leave their source triangles")
    if proxy_faces.shape != (7996, 3) or proxy_faces.min() < 0 or proxy_faces.max() >= len(source_vertex_indices):
        raise ValueError(f"invalid MHR collision-proxy faces {proxy_faces.shape}")
    if reference_vertices.shape != (len(source_vertex_indices), 3) or not np.isfinite(reference_vertices).all():
        raise ValueError(f"invalid MHR collision-proxy reference vertices {reference_vertices.shape}")
    validate_closed_triangle_mesh(proxy_faces)
    for value in (source_vertex_indices, barycentric_weights, proxy_faces, reference_vertices):
        value.setflags(write=False)
    return MHRCollisionProxy(source_vertex_count=source_vertex_count, source_face_count=source_face_count, source_faces_sha256=source_faces_hash, source_vertex_indices=source_vertex_indices, barycentric_weights=barycentric_weights, faces=proxy_faces, reference_vertices=reference_vertices, revision=revision)


def load_mhr_collision_proxy(source_faces: Any, asset_path: str | Path = DEFAULT_MHR_COLLISION_PROXY_ASSET) -> MHRCollisionProxy:
    source_faces = np.asarray(source_faces, dtype=np.int32)
    proxy = _load_mhr_collision_proxy_asset(str(Path(asset_path).resolve()))
    if source_faces.shape != (proxy.source_face_count, 3) or mhr_faces_sha256(source_faces) != proxy.source_faces_sha256:
        raise ValueError(f"MHR collision proxy does not match source topology {source_faces.shape}")
    return proxy


def apply_mhr_collision_proxy(vertices: Any, source_vertex_indices: Any, barycentric_weights: Any) -> Any:
    if vertices.ndim < 2 or vertices.shape[-1] != 3:
        raise ValueError(f"MHR collision-proxy source vertices must end in [V,3], got {vertices.shape}")
    triangles = vertices[..., source_vertex_indices, :]
    if triangles.__class__.__module__.startswith("torch"):
        return (triangles * barycentric_weights[..., None]).sum(dim=-2)
    return np.sum(triangles * np.asarray(barycentric_weights)[..., None], axis=-2)
