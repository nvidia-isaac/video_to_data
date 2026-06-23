# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symmetry-aware pose canonicalization for multi-view FoundationPose.

Loads BOP-style symmetry annotations produced by hoi-tools' compute_symmetry
script and picks the equivalence-class representative of a per-view 6D pose
that lies closest to a reference pose. The convention is right-multiplication:
P and `P @ R_s` render identical images for any self-symmetry R_s, so the
canonical form is `P @ R_s*` minimizing rotational geodesic distance from the
reference. Translation enters only through R_s and is zero for
centroid-aligned BOP symmetries.
"""

import json
from pathlib import Path

import numpy as np


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _close_group(transforms: list[np.ndarray], atol: float = 1e-6,
                 max_size: int = 1024) -> list[np.ndarray]:
    elems: list[np.ndarray] = [np.eye(4)]
    for T in transforms:
        T = np.asarray(T, dtype=float)
        if not any(np.allclose(T, e, atol=atol) for e in elems):
            elems.append(T)
    while True:
        new: list[np.ndarray] = []
        for a in elems:
            for b in elems:
                p = a @ b
                if any(np.allclose(p, e, atol=atol) for e in elems):
                    continue
                if any(np.allclose(p, n, atol=atol) for n in new):
                    continue
                new.append(p)
                if len(elems) + len(new) > max_size:
                    raise ValueError(
                        f"group closure exceeded {max_size} elements; "
                        "inputs may not generate a finite group"
                    )
        if not new:
            break
        elems.extend(new)
    return elems


def load_symmetry_group(
    json_path: str | Path,
    continuous_step_deg: float = 5.0,
) -> list[np.ndarray]:
    """Load a BOP-style symmetry JSON and return a flat list of (4,4) SE(3) group elements.

    Identity is included. Discrete transforms are closed under composition
    (idempotent if the JSON already stores a closed group). Continuous axes
    are discretized at `continuous_step_deg` and layered via Cartesian
    product. Continuous-axis offsets are ignored (BOP convention places the
    object centroid on the axis, so the axis line passes through the origin
    of the aligned mesh).
    """
    with open(json_path) as f:
        data = json.load(f)

    discrete = [
        np.array(t, dtype=float).reshape(4, 4)
        for t in data.get("symmetries_discrete", [])
    ]
    group = _close_group(discrete)

    n_steps = max(1, int(round(360.0 / continuous_step_deg)))
    for cont in data.get("symmetries_continuous", []):
        axis = np.asarray(cont["axis"], dtype=float)
        rotations: list[np.ndarray] = []
        for s in range(n_steps):
            R = _rotation_matrix(axis, 2 * np.pi * s / n_steps)
            T = np.eye(4)
            T[:3, :3] = R
            rotations.append(T)
        group = [b @ r for b in group for r in rotations]

    return group


def canonicalize_pose(
    pose: np.ndarray,
    group: list[np.ndarray],
    reference: np.ndarray,
) -> np.ndarray:
    """Pick `pose @ R_s*` (R_s ∈ group) minimizing rotational geodesic distance to reference."""
    ref_R = np.asarray(reference, dtype=float)[:3, :3]
    pose = np.asarray(pose, dtype=float)
    best = None
    best_dist = float("inf")
    for R_s in group:
        candidate = pose @ R_s
        cos_theta = (np.trace(ref_R.T @ candidate[:3, :3]) - 1.0) / 2.0
        d = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        if d < best_dist:
            best_dist = d
            best = candidate
    return best
