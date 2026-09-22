#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-object signed distance fields (SDF) for the hand-object penetration check.

The penetration check must measure the robot hand against the object's true solid
(its convex decomposition), not its convex hull: the hull fills hollows, so a
finger inside a mug/bowl reads as deep penetration though it touches nothing.
Measuring the true distance per frame (``convex_decomposition`` + ``closest_point``)
is ~11x slower than the hull, so we precompute a per-object SDF once, offline, and
do O(1) trilinear lookups at runtime.

Build recipe (fast — a ``closest_point`` per voxel would be hours/object):

  VHACD convex decomposition -> solid voxelization of the parts -> Euclidean distance
  transform inside & outside -> signed grid, positive INSIDE, stored float16.

No erosion is applied at the 1 mm default pitch (see ``DEFAULT_ERODE_ITERS``); coarser
grids can strip a voxel shell to counter the conservative voxelization.

Query: vectorized trilinear interpolation, returning signed distance (positive
inside) in metres, matching ``trimesh.proximity.signed_distance``. Penetration depth
of a sphere of radius ``r`` at signed distance ``sd`` is ``max(0, r + sd)``.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

# Grid resolution and exterior margin. A 1 mm pitch keeps the discretization error
# small; 2 cm of exterior padding covers every point that can penetrate (depth is
# only positive for sd > -radius, i.e. within ~1 cm outside the surface). At 1 mm no
# erosion is applied by default; it is only useful at coarser pitch (see build_sdf).
DEFAULT_PITCH = 0.001  # metres
DEFAULT_PAD = 0.02  # metres
DEFAULT_ERODE_ITERS = 0
# ~200M voxels ≈ 400 MB (float16) / ~3 GB transient (EDT float64). A legit ≤58 cm
# object at 1 mm fits; anything larger is a mis-scaled mesh and is skipped.
MAX_GRID_VOXELS = 200_000_000


def load_object_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load an object collision mesh in metres, honoring the ``_cm.obj`` convention.

    Some datasets (e.g. taco) ship object meshes authored in centimetres, flagged by
    a ``_cm.obj`` suffix; those are scaled by 0.01 so the geometry matches the metres
    used by the object poses and robot frames. This is the same convention applied by
    the visualizer and support-surface reconstruction, kept in one place so the SDF
    builder and the penetration check load identical geometry.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh_path.endswith("_cm.obj"):
        mesh.apply_scale(0.01)
    return mesh


def _decompose(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """VHACD convex decomposition; fall back to coarser params, then the hull."""
    for kwargs in (
        dict(max_convex_hulls=48, resolution=200000),
        dict(maxNumVerticesPerCH=64),
        dict(),
    ):
        try:
            decomposed = mesh.convex_decomposition(**kwargs)
        except Exception:
            continue
        parts = [decomposed] if isinstance(decomposed, trimesh.Trimesh) else decomposed
        parts = [
            (
                p
                if isinstance(p, trimesh.Trimesh)
                else trimesh.Trimesh(vertices=p["vertices"], faces=p["faces"])
            )
            for p in parts
        ]
        if parts:
            return parts
    return [mesh.convex_hull]


def build_sdf(
    mesh_path: str,
    pitch: float = DEFAULT_PITCH,
    pad: float = DEFAULT_PAD,
    erode_iters: int = DEFAULT_ERODE_ITERS,
) -> dict:
    """Build the SDF grid for one mesh. Returns {sdf(float16), origin, pitch}.

    ``erode_iters`` strips that many voxel shells off the occupancy before the
    distance transform, compensating the conservative voxelization that otherwise
    inflates the solid outward (and reads penetration ~1 voxel too deep). Too much
    erosion thins thin-walled parts and under-reports, so keep it small.
    """
    mesh = load_object_mesh(mesh_path)
    parts = _decompose(mesh)

    lo = mesh.bounds[0] - pad
    hi = mesh.bounds[1] + pad
    dims = np.ceil((hi - lo) / pitch).astype(int) + 1

    # Guard against mis-scaled meshes whose bounds span many metres: such a grid
    # would allocate terabytes. Raise so the builder skips the object and the check
    # falls back to the convex hull, rather than failing with an out-of-memory error.
    if int(np.prod(dims.astype(np.int64))) > MAX_GRID_VOXELS:
        raise ValueError(
            f"SDF grid {tuple(int(x) for x in dims)} exceeds {MAX_GRID_VOXELS:,} voxels "
            f"(mesh extent {np.round((hi - lo) * 100, 1)} cm) — likely mis-scaled mesh"
        )

    # Occupancy: union of the (watertight, convex) parts voxelized solid.
    occ = np.zeros(tuple(dims), dtype=bool)
    for part in parts:
        try:
            voxels = part.voxelized(pitch=pitch).fill()
        except Exception:
            voxels = part.voxelized(pitch=pitch)
        idx = np.round((voxels.points - lo) / pitch).astype(int)
        inside = np.all((idx >= 0) & (idx < dims), axis=1)
        idx = idx[inside]
        if len(idx):
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    # Strip the conservative-voxelization shell so the surface sits at the true
    # boundary (otherwise every depth reads ~1 voxel too deep).
    if erode_iters > 0:
        occ = ndimage.binary_erosion(occ, iterations=erode_iters)

    dist_in = ndimage.distance_transform_edt(occ)
    dist_out = ndimage.distance_transform_edt(~occ)
    sdf = (dist_in - dist_out).astype(np.float32) * pitch  # +inside, metres

    return {
        "sdf": sdf.astype(np.float16),
        "origin": lo.astype(np.float32),
        "pitch": np.float32(pitch),
    }


def sdf_cache_path(mesh_path: str) -> str:
    """Co-located cache path for a mesh: ``foo.glb`` -> ``foo.glb.sdf.npz``.

    The extension is KEPT (appended, not replaced) so a collision mesh and a visual
    mesh sharing a stem but differing in extension (e.g. ``mug.obj`` + ``mug.stl``)
    map to distinct caches and can never overwrite one another. Both the builder
    (writer) and the penetration check (reader) call this, so they always agree.
    """
    return mesh_path + ".sdf.npz"


def save_sdf(cache_path: str, grid: dict) -> None:
    """Write an SDF grid to ``cache_path`` (compressed .npz)."""
    np.savez_compressed(
        cache_path, sdf=grid["sdf"], origin=grid["origin"], pitch=grid["pitch"]
    )


class ObjectSDF:
    """A precomputed signed distance field with a vectorized trilinear query."""

    def __init__(self, sdf: np.ndarray, origin: np.ndarray, pitch: float) -> None:
        """Wrap a signed-distance grid with its world origin and voxel pitch."""
        self.sdf = sdf
        self.origin = np.asarray(origin, dtype=np.float32)
        self.pitch = float(pitch)
        self.dims = np.array(sdf.shape)

    @classmethod
    def load(cls, cache_path: str) -> "ObjectSDF":
        """Load an SDF grid from a ``.sdf.npz`` cache file."""
        with np.load(cache_path) as data:
            return cls(data["sdf"], data["origin"], float(data["pitch"]))

    def query(self, pts_local: np.ndarray) -> np.ndarray:
        """Signed distance (positive inside, metres) at object-local points.

        Points outside the grid are clamped to the boundary; they read a large
        negative distance (well outside), which correctly yields zero penetration.
        """
        grid = self.sdf
        dims = self.dims
        coords = np.clip((pts_local - self.origin) / self.pitch, 0, dims - 1 - 1e-6)
        i0 = np.floor(coords).astype(np.intp)
        frac = coords - i0
        i1 = np.minimum(i0 + 1, dims - 1)

        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

        c00 = grid[x0, y0, z0] * (1 - fx) + grid[x1, y0, z0] * fx
        c10 = grid[x0, y1, z0] * (1 - fx) + grid[x1, y1, z0] * fx
        c01 = grid[x0, y0, z1] * (1 - fx) + grid[x1, y0, z1] * fx
        c11 = grid[x0, y1, z1] * (1 - fx) + grid[x1, y1, z1] * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return (c0 * (1 - fz) + c1 * fz).astype(np.float64)

    def outward_normal(self, pts_local: np.ndarray) -> np.ndarray:
        """Unit outward direction (object-local) at each query point.

        The field is positive inside, so signed distance decreases toward the
        surface and outward; the outward direction is therefore the normalized
        **negative gradient** of the field, estimated by central differences one
        voxel apart. Used by penetration-avoidance IK to push a penetrating
        fingertip target straight out of the solid (the SDF analogue of a mesh
        face normal). Degenerate (zero-gradient) points return a zero vector.
        """
        h = self.pitch
        grad = np.empty((len(pts_local), 3), dtype=np.float64)
        for ax in range(3):
            step = np.zeros(3, dtype=np.float64)
            step[ax] = h
            grad[:, ax] = (
                self.query(pts_local + step) - self.query(pts_local - step)
            ) / (2.0 * h)
        out = -grad
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        norm[norm < 1e-9] = 1.0
        return out / norm
