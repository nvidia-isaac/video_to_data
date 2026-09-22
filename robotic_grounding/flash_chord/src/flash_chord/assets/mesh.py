# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convex decomposition of collision meshes (CoACD), cached on disk by a geometry + params hash.

Newton/MuJoCo contacts a mesh as one convex hull, so concave objects collide as their solid hull;
decomposing into multiple convex hulls (one collision shape each) restores correct contact.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flash_chord.assets import ASSETS_DIR

# Bump when the decomposition logic or the cache file schema changes (invalidates stale caches).
_CACHE_VERSION = 1


@dataclass(frozen=True)
class ConvexHull:
    """One convex piece of a decomposition, in the source mesh's local frame."""

    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) int32


def default_cache_dir() -> Path:
    """On-disk cache location (override with ``FLASH_CHORD_CACHE_DIR``)."""
    return Path(os.environ.get("FLASH_CHORD_CACHE_DIR", str(ASSETS_DIR / ".cache"))) / "convex"


def _cache_key(vertices: np.ndarray, faces: np.ndarray, params: dict) -> str:
    h = hashlib.sha1()
    h.update(f"v{_CACHE_VERSION}".encode())
    h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    h.update(repr(sorted(params.items())).encode())
    return h.hexdigest()[:16]


def _save_cache(path: Path, hulls: list[ConvexHull]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs: dict[str, np.ndarray] = {"n": np.int64(len(hulls))}
    for i, hull in enumerate(hulls):
        arrs[f"v{i}"] = hull.vertices
        arrs[f"f{i}"] = hull.faces
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, path)  # atomic, so a crash mid-write can't leave a half-written cache file


def _load_cache(path: Path) -> list[ConvexHull]:
    with np.load(path) as d:
        return [ConvexHull(d[f"v{i}"], d[f"f{i}"]) for i in range(int(d["n"]))]


def convex_decompose(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    threshold: float = 0.05,
    max_convex_hull: int = -1,
    preprocess_mode: str = "auto",
    cache_dir: str | os.PathLike | None = None,
    **coacd_kwargs,
) -> list[ConvexHull]:
    """Decompose a triangle mesh into convex hulls via CoACD, cached on disk.

    ``threshold`` is CoACD's concavity tolerance (lower => more, tighter hulls); ``max_convex_hull``
    caps the piece count (-1 = unbounded). The result is cached by a hash of (geometry, params), so
    repeat calls (and repeat scene builds) are instant. Raises ``ImportError`` if ``coacd`` is missing.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1, 3)
    params = dict(threshold=threshold, max_convex_hull=max_convex_hull, preprocess_mode=preprocess_mode, **coacd_kwargs)

    cache_root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache_file = cache_root / f"{_cache_key(vertices, faces, params)}.npz"
    if cache_file.exists():
        return _load_cache(cache_file)

    import coacd  # lazy: importing this module must not require the decomposition backend

    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(vertices, faces), **params)
    hulls = [
        ConvexHull(np.ascontiguousarray(v, dtype=np.float32), np.ascontiguousarray(f, dtype=np.int32).reshape(-1, 3))
        for v, f in parts
    ]
    _save_cache(cache_file, hulls)
    return hulls


def decompose_mesh_file(path: str | os.PathLike, **kwargs) -> list[ConvexHull]:
    """Load a mesh file (via trimesh) and convex-decompose it. Convenience wrapper for offline use;
    the scene builder decomposes the geometry already loaded into Newton's builder instead."""
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    return convex_decompose(np.asarray(mesh.vertices), np.asarray(mesh.faces), **kwargs)
