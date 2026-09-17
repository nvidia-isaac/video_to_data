# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand-object contact labelling from FK keypoints + object mesh proximity.

Used by the whole-body retargeter (``ego_recon_to_dexmate_sharpa.py``). A probe (an FK
frame origin near a fingertip) is "in contact" for a frame when its distance to the object
surface is below ``threshold`` AND it belongs to a run of at least ``min_consecutive``
consecutive sub-threshold frames, optionally gated to the object-motion window so a
resting object does not register proximity false positives.

Standalone (numpy / scipy / trimesh only) so it imports cleanly inside the retarget
container without pulling Isaac Sim or pinocchio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import trimesh.proximity
from scipy.spatial.transform import Rotation

# Number of FK probes per hand side (task-frame origins near the fingertips + MPs).
PROBES_PER_SIDE = 11

# Object-motion gate (short-window displacement, not absolute-from-frame-0): the
# ego_recon object track drifts slowly during the approach, so an absolute threshold
# fires on tracking drift. Over a 10-frame window rest-phase drift stays < ~8.5 mm while
# a real lift exceeds 10 mm. The gate is padded to include the grasp-close lead-in and the
# release trail-out around the sustained motion bracket.
DISPLACEMENT_WINDOW_FRAMES = 10
DISPLACEMENT_THRESHOLD_M = 0.010
PRE_DISPLACEMENT_WINDOW_FRAMES = 30
POST_DISPLACEMENT_WINDOW_FRAMES = 30


def mesh_radius_from_obj(obj_path: str | Path) -> float:
    """Compute the max vertex norm (bounding radius in meters) from an OBJ's ``v`` lines."""
    max_sq = 0.0
    with open(obj_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                max_sq = max(max_sq, x * x + y * y + z * z)
    radius = float(np.sqrt(max_sq))
    if radius <= 0.0:
        raise ValueError(
            f"object mesh radius must be positive, got {radius} from {obj_path}"
        )
    return radius


def deflicker(active: np.ndarray, min_consecutive: int) -> np.ndarray:
    """Keep only runs of at least ``min_consecutive`` consecutive active frames.

    Args:
        active: Boolean activation (T,) for one slot.
        min_consecutive: Minimum run length to keep.

    Returns:
        De-flickered boolean activation (T,).
    """
    out = np.zeros_like(active)
    t = 0
    num_frames = active.shape[0]
    while t < num_frames:
        if active[t]:
            start = t
            while t < num_frames and active[t]:
                t += 1
            if t - start >= min_consecutive:
                out[start:t] = True
        else:
            t += 1
    return out


def manipulation_window(object_pos: np.ndarray) -> tuple[int, int] | None:
    """Inclusive ``[lo, hi]`` frame bracket of sustained object motion, padded.

    Finds the first and last frames the object is being moved (short-window displacement
    signal) and pads by the pre/post windows to include the grasp-close lead-in and release
    trail-out. Returns ``None`` if the object never moves (caller keeps contacts ungated).

    Args:
        object_pos: Object positions (T, 3).

    Returns:
        ``(lo, hi)`` inclusive frame indices, or ``None`` when no sustained motion is found.
    """
    window = DISPLACEMENT_WINDOW_FRAMES
    disp = np.linalg.norm(object_pos[window:] - object_pos[:-window], axis=-1)
    moved = np.nonzero(disp > DISPLACEMENT_THRESHOLD_M)[0]
    if moved.size == 0:
        return None
    lo = max(0, int(moved[0]) + window - PRE_DISPLACEMENT_WINDOW_FRAMES)
    hi = min(
        len(object_pos) - 1, int(moved[-1]) + window + POST_DISPLACEMENT_WINDOW_FRAMES
    )
    return lo, hi


def compute_contacts(
    obj_path: str | Path,
    probes_w: np.ndarray,
    object_pos: np.ndarray,
    object_wxyz: np.ndarray,
    threshold: float,
    min_consecutive: int,
    motion_gate: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Generate per-side hand-object contact labels via mesh proximity queries.

    Probes are the 11 FK frame origins per side in task-name order. A slot is active when
    its distance to the object surface is below ``threshold`` and it belongs to a run of at
    least ``min_consecutive`` consecutive sub-threshold frames. Contact positions are the
    closest surface points in world; normals are the inward (negated) mesh face normals
    rotated to world.

    Args:
        obj_path: Object OBJ mesh path.
        probes_w: World probe positions (T, 2 * PROBES_PER_SIDE, 3) — left slots then right.
        object_pos: Object positions (T, 3).
        object_wxyz: Unit object quaternions (T, 4) wxyz.
        threshold: Activation distance threshold in meters.
        min_consecutive: De-flicker minimum run length.
        motion_gate: Optional inclusive ``[lo, hi]`` frame window; when set, proximity
            activations outside it are suppressed before de-flicker.

    Returns:
        Dict with positions (2, T, 11, 3), normals (2, T, 11, 3), part_ids (2, T, 11),
        distances (2, T, 11), and active (2, T, 11).
    """
    mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"failed to load object mesh: {obj_path}")
    query = trimesh.proximity.ProximityQuery(mesh)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)

    num_frames = probes_w.shape[0]
    rot = Rotation.from_quat(object_wxyz, scalar_first=True).as_matrix()  # (T, 3, 3)

    in_window = np.ones(num_frames, dtype=bool)
    if motion_gate is not None:
        lo, hi = motion_gate
        in_window[:] = False
        in_window[lo : hi + 1] = True

    positions = np.zeros((2, num_frames, PROBES_PER_SIDE, 3), dtype=np.float64)
    normals = np.zeros((2, num_frames, PROBES_PER_SIDE, 3), dtype=np.float64)
    part_ids = np.zeros((2, num_frames, PROBES_PER_SIDE), dtype=np.int64)
    distances = np.zeros((2, num_frames, PROBES_PER_SIDE), dtype=np.float64)
    active_all = np.zeros((2, num_frames, PROBES_PER_SIDE), dtype=bool)

    for side_index in range(2):
        slots = slice(side_index * PROBES_PER_SIDE, (side_index + 1) * PROBES_PER_SIDE)
        side_probes_w = probes_w[:, slots, :]  # (T, 11, 3)
        rel = side_probes_w - object_pos[:, None, :]
        probes_obj = np.einsum("tji,tcj->tci", rot, rel)  # R^T @ rel
        closest_obj, dist, tri_ids = query.on_surface(probes_obj.reshape(-1, 3))
        closest_obj = closest_obj.reshape(num_frames, PROBES_PER_SIDE, 3)
        dist = dist.reshape(num_frames, PROBES_PER_SIDE)
        tri_ids = tri_ids.reshape(num_frames, PROBES_PER_SIDE)
        distances[side_index] = dist

        active = dist < threshold
        active[~in_window] = (
            False  # object-motion gate: suppress stationary-phase contacts
        )
        for slot in range(PROBES_PER_SIDE):
            active[:, slot] = deflicker(active[:, slot], min_consecutive)
        active_all[side_index] = active

        closest_w = np.einsum("tij,tcj->tci", rot, closest_obj) + object_pos[:, None, :]
        normal_obj = -face_normals[tri_ids]  # inward convention
        normal_w = np.einsum("tij,tcj->tci", rot, normal_obj)
        norm = np.linalg.norm(normal_w, axis=-1, keepdims=True)
        norm = np.where(norm < 1.0e-12, 1.0, norm)
        normal_w = normal_w / norm

        positions[side_index][active] = closest_w[active]
        normals[side_index][active] = normal_w[active]
        part_ids[side_index][active] = 1

    return {
        "positions": positions,
        "normals": normals,
        "part_ids": part_ids,
        "distances": distances,
        "active": active_all,
    }
