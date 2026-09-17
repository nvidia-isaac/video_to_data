# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Object collision proxies and surface projection for retargeting.

Reusable, dependency-light (NumPy + trimesh only) helpers to keep retargeted hand
keypoints out of the manipulated object. The monocular MANO reconstruction often
penetrates the object collider; feeding those penetrating keypoints to IK produces a
reference the sim contact solver fights. ``ObbSurfaceProjector`` pushes penetrating
keypoints back onto the object surface before IK, optionally preserving hand shape by
moving whole fingers (or the whole hand) rigidly.

The geometry is an oriented bounding box (OBB) per object body, which is exact for
box-like objects (e.g. ``wood_box``). The ``SurfaceProjector`` protocol leaves room for
a future signed-distance projector for non-convex objects without changing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh

from robotic_grounding.retarget.params import MANO_JOINTS_ORDER


def _mano_finger_groups() -> list[list[int]]:
    """Group the 21 MANO joint indices into [wrist] + one list per finger.

    Derived from :data:`MANO_JOINTS_ORDER` (``params.py``) by stripping the trailing
    joint number, so the grouping stays in sync with the canonical joint order rather
    than being hardcoded. Yields ``[[0], [1,2,3,4], ...]`` (wrist, thumb, index,
    middle, ring, pinky) for the standard 21-joint layout.
    """
    wrist: list[int] = []
    fingers: dict[str, list[int]] = {}
    for idx, name in enumerate(MANO_JOINTS_ORDER):
        if name == "wrist":
            wrist.append(idx)
            continue
        finger = name.rstrip("0123456789")
        fingers.setdefault(finger, []).append(idx)
    return [wrist, *fingers.values()]


#: ``[[0], [1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]``
#: (wrist, thumb, index, middle, ring, pinky) — derived from ``MANO_JOINTS_ORDER``.
MANO21_FINGER_GROUPS: list[list[int]] = _mano_finger_groups()


def quat_wxyz_to_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert a ``wxyz`` quaternion to a rotation matrix.

    Args:
        wxyz: Quaternion(s) with shape ``(..., 4)`` in ``(w, x, y, z)`` order.

    Returns:
        Rotation matrix/matrices with shape ``(..., 3, 3)``.
    """
    q = np.asarray(wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def resolve_groups(granularity: str, num_points: int) -> list[list[int]]:
    """Resolve a granularity string to index groups over ``num_points`` points.

    Args:
        granularity: ``"hand"`` (one rigid group), ``"finger"`` (per-finger rigid,
            requires 21 MANO joints), or ``"joint"`` (each point independent).
        num_points: Number of keypoints being projected.

    Returns:
        A list of index lists partitioning ``range(num_points)``.
    """
    if granularity == "hand":
        return [list(range(num_points))]
    if granularity == "joint":
        return [[i] for i in range(num_points)]
    if granularity == "finger":
        if num_points != len(MANO_JOINTS_ORDER):
            raise ValueError(
                f"finger granularity expects {len(MANO_JOINTS_ORDER)} MANO joints, "
                f"got {num_points}"
            )
        return MANO21_FINGER_GROUPS
    raise ValueError(f"unknown granularity {granularity!r}")


@dataclass(frozen=True)
class OrientedBox:
    """An oriented bounding box in an object's canonical (local) frame.

    Attributes:
        center: Box center in the local frame, shape ``(3,)``.
        axes: Orthonormal box axes as columns, shape ``(3, 3)``.
        half_extents: Half-extents along each axis, shape ``(3,)``.
    """

    center: np.ndarray
    axes: np.ndarray
    half_extents: np.ndarray

    @classmethod
    def from_points(cls, verts: np.ndarray) -> "OrientedBox":
        """Fit a minimum-volume oriented bounding box to a point cloud.

        Uses rotating calipers on the convex hull
        (:func:`trimesh.bounds.oriented_bounds`) rather than PCA. PCA axes are
        pulled toward dense vertex clusters — a box's end-caps outweigh its faces
        in the covariance — so for a thin, tilted slab PCA can miss the thin
        dimension entirely (on ``wood_box`` PCA reports a ~33 cm minimum side while
        the true thickness is ~13 cm, inflating penetration/offset estimates). The
        min-volume box recovers the real slab. Falls back to PCA on degenerate
        inputs (e.g. coplanar points where the hull is not 3D).
        """
        v = np.asarray(verts, dtype=np.float64)
        try:
            to_origin, extents = trimesh.bounds.oriented_bounds(v)
            from_origin = np.linalg.inv(to_origin)
            return cls(
                center=from_origin[:3, 3],
                axes=from_origin[:3, :3],  # columns = box axes in the input frame
                half_extents=np.asarray(extents, dtype=np.float64) / 2.0,
            )
        except Exception:
            c0 = v.mean(axis=0)
            vc = v - c0
            _evals, evecs = np.linalg.eigh(vc.T @ vc)  # columns are axes
            proj = vc @ evecs
            lo, hi = proj.min(axis=0), proj.max(axis=0)
            center = c0 + evecs @ ((lo + hi) / 2.0)
            half_extents = (hi - lo) / 2.0
            return cls(center=center, axes=evecs, half_extents=half_extents)

    def to_box_coords(self, pts_local: np.ndarray) -> np.ndarray:
        """Express local-frame points in box-axis coordinates: ``axes.T @ (p - center)``."""
        return (np.asarray(pts_local, dtype=np.float64) - self.center) @ self.axes

    def signed_distance(self, pts_local: np.ndarray) -> np.ndarray:
        """Per-point signed distance (m) to the box surface: >0 outside, <0 inside.

        Args:
            pts_local: Points in the box's local frame, shape ``(N, 3)``.

        Returns:
            Signed distance per point, shape ``(N,)``.
        """
        q = np.abs(self.to_box_coords(pts_local)) - self.half_extents
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside

    def closest_surface_point(self, pts_local: np.ndarray) -> np.ndarray:
        """Closest point ON the box surface per query point (local frame).

        Outside points clamp to the box; inside points snap to the nearest face.

        Args:
            pts_local: Points in the box's local frame, shape ``(N, 3)``.

        Returns:
            Surface points in the local frame, shape ``(N, 3)``.
        """
        b = self.to_box_coords(pts_local)
        c = np.clip(b, -self.half_extents, self.half_extents)
        inside = np.all(np.abs(b) < self.half_extents, axis=1)
        if inside.any():
            bi = b[inside]
            ci = c[inside]
            nearest_axis = (self.half_extents - np.abs(bi)).argmin(axis=1)
            rows = np.arange(len(bi))
            signs = np.where(bi[rows, nearest_axis] >= 0.0, 1.0, -1.0)
            ci[rows, nearest_axis] = signs * self.half_extents[nearest_axis]
            c[inside] = ci
        return self.center + c @ self.axes.T

    def penetration_depth(self, pts_local: np.ndarray) -> np.ndarray:
        """Per-point penetration depth (m): distance below the nearest face, else 0.

        Args:
            pts_local: Points in the box's local frame, shape ``(N, 3)``.

        Returns:
            Depth per point, shape ``(N,)``; ``0`` for points outside the box.
        """
        b = np.abs(self.to_box_coords(pts_local))
        inside = np.all(b < self.half_extents, axis=1)
        nearest_face = (self.half_extents - b).min(axis=1)
        return np.where(inside, nearest_face, 0.0)

    def push_out(
        self, pts_local: np.ndarray, margin: float, groups: list[list[int]]
    ) -> np.ndarray:
        """Rigidly push penetrating point-groups out to ``half_extent + margin``.

        For each group with at least one interior point, the group is translated (as a
        rigid unit, preserving intra-group geometry) along the *deepest* member's
        nearest-face normal by ``depth + margin``. Iterated up to 3x so a group whose
        points straddle two faces (e.g. a finger near a box edge) is fully cleared.
        Points/groups already outside are left untouched.

        Args:
            pts_local: Points in the box's local frame, shape ``(N, 3)``.
            margin: Extra clearance (m) added beyond the face.
            groups: Index lists; each is moved rigidly.

        Returns:
            Projected points, shape ``(N, 3)``.
        """
        out = np.array(pts_local, dtype=np.float64, copy=True)
        for _ in range(3):
            b = self.to_box_coords(out)
            absb = np.abs(b)
            slack = self.half_extents - absb  # dist to each face (neg => outside)
            inside = np.all(absb < self.half_extents, axis=1)
            if not inside.any():
                break
            moved = False
            for grp in groups:
                members = [i for i in grp if inside[i]]
                if not members:
                    continue
                face_slack = slack[members].min(axis=1)  # nearest-face dist per member
                deep = members[int(np.argmax(face_slack))]  # largest => deepest inside
                axis = int(np.argmin(slack[deep]))  # nearest face of the deepest point
                sign = 1.0 if b[deep, axis] >= 0 else -1.0
                shift = sign * (slack[deep, axis] + margin) * self.axes[:, axis]
                out[grp] += shift  # rigid: same shift for every member of the group
                moved = True
            if not moved:
                break
        return out


class SurfaceProjector(Protocol):
    """Pushes world-frame keypoints out of one or more posed object bodies."""

    def project_out(
        self,
        points_world: np.ndarray,
        obj_positions: np.ndarray,
        obj_wxyzs: np.ndarray,
        margin: float,
        granularity: str = "finger",
    ) -> np.ndarray:
        """Return ``points_world`` with penetrating points pushed onto the surface."""
        ...

    def penetration(
        self,
        points_world: np.ndarray,
        obj_positions: np.ndarray,
        obj_wxyzs: np.ndarray,
    ) -> np.ndarray:
        """Return per-point penetration depth (m), max over bodies."""


class ObbSurfaceProjector:
    """:class:`SurfaceProjector` backed by one :class:`OrientedBox` per object body."""

    def __init__(self, boxes: list[OrientedBox]) -> None:
        """Wrap one oriented box per object body, in body order."""
        self.boxes = boxes

    def _pose(
        self, obj_positions: np.ndarray, obj_wxyzs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize and validate per-body poses against the box list."""
        pos = np.asarray(obj_positions, dtype=np.float64).reshape(-1, 3)
        wxyz = np.asarray(obj_wxyzs, dtype=np.float64).reshape(-1, 4)
        if not (len(pos) == len(wxyz) == len(self.boxes)):
            raise ValueError(
                f"pose count mismatch: {len(pos)} positions, {len(wxyz)} quats, "
                f"{len(self.boxes)} boxes"
            )
        return pos, wxyz

    def project_out(
        self,
        points_world: np.ndarray,
        obj_positions: np.ndarray,
        obj_wxyzs: np.ndarray,
        margin: float,
        granularity: str = "finger",
    ) -> np.ndarray:
        """Push penetrating points out to the box surface, one body at a time."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        groups = resolve_groups(granularity, len(pts))
        pos, wxyz = self._pose(obj_positions, obj_wxyzs)
        for box, p, q in zip(self.boxes, pos, wxyz, strict=True):
            r = quat_wxyz_to_matrix(q)  # (3,3)
            local = (pts - p) @ r  # world -> body-local (r.T @ x via right-mul)
            local = box.push_out(local, margin, groups)
            pts = local @ r.T + p  # body-local -> world
        return pts

    def penetration(
        self,
        points_world: np.ndarray,
        obj_positions: np.ndarray,
        obj_wxyzs: np.ndarray,
    ) -> np.ndarray:
        """Return per-point penetration depth in metres, max over bodies."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        pos, wxyz = self._pose(obj_positions, obj_wxyzs)
        depth = np.zeros(len(pts))
        for box, p, q in zip(self.boxes, pos, wxyz, strict=True):
            r = quat_wxyz_to_matrix(q)
            local = (pts - p) @ r
            depth = np.maximum(depth, box.penetration_depth(local))
        return depth

    def signed_distance(
        self,
        points_world: np.ndarray,
        obj_positions: np.ndarray,
        obj_wxyzs: np.ndarray,
    ) -> np.ndarray:
        """Per-point, per-body signed surface distance (m): >0 outside, <0 inside.

        Returns:
            Signed distances, shape ``(N, num_bodies)``.
        """
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        pos, wxyz = self._pose(obj_positions, obj_wxyzs)
        out = np.empty((len(pts), len(self.boxes)))
        for body_idx, (box, p, q) in enumerate(zip(self.boxes, pos, wxyz, strict=True)):
            r = quat_wxyz_to_matrix(q)
            out[:, body_idx] = box.signed_distance((pts - p) @ r)
        return out

    def closest_surface_points(
        self,
        points_world: np.ndarray,
        obj_positions: np.ndarray,
        obj_wxyzs: np.ndarray,
        body_idx: int,
    ) -> np.ndarray:
        """Closest points ON one body's surface, world frame. Shape ``(N, 3)``."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        pos, wxyz = self._pose(obj_positions, obj_wxyzs)
        box, p, q = self.boxes[body_idx], pos[body_idx], wxyz[body_idx]
        r = quat_wxyz_to_matrix(q)
        local = box.closest_surface_point((pts - p) @ r)
        return local @ r.T + p


def _load_mesh_vertices(mesh_path: str, vertex_scale: float) -> np.ndarray:
    """Load a mesh and return scaled vertices, mirroring the retarget's trimesh use."""
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return np.asarray(mesh.vertices, dtype=np.float64) * float(vertex_scale)


def make_object_projector(
    mesh_paths: list[str], vertex_scale: float = 1.0, method: str = "obb"
) -> SurfaceProjector:
    """Build a :class:`SurfaceProjector` from per-body object mesh paths.

    Args:
        mesh_paths: One mesh path per object body (as stored in ``object_mesh_paths``).
        vertex_scale: Scale applied to mesh vertices (use the dataset's
            ``mesh_vertex_scale``; e.g. handles ``_cm.obj`` centimetre meshes).
        method: Collision proxy type. Only ``"obb"`` is implemented; ``"sdf"`` is
            reserved for a future signed-distance projector for non-convex objects.

    Returns:
        A projector with one proxy per body, in ``mesh_paths`` order.
    """
    if method != "obb":
        raise NotImplementedError(f"projector method {method!r} not implemented")
    boxes: list[OrientedBox] = []
    for path in mesh_paths:
        if not path or not Path(path).exists():
            raise FileNotFoundError(f"object mesh not found for projector: {path!r}")
        boxes.append(OrientedBox.from_points(_load_mesh_vertices(path, vertex_scale)))
    return ObbSurfaceProjector(boxes)
