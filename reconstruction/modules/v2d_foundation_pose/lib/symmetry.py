# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded symmetry handling for multi-view FoundationPose."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG = 10.0
MAX_SYMMETRY_GROUP_SIZE = 4096
_AXIS_ATOL = 1e-6


@dataclass(frozen=True)
class SymmetrySpec:
    """Parsed symmetry behavior plus immutable runtime provenance."""

    transforms: np.ndarray
    mode: str
    axes: tuple[tuple[float, float, float], ...]
    continuous_step_deg: float
    full_rotational: bool
    candidate_count: int
    guard_limit: int = MAX_SYMMETRY_GROUP_SIZE

    def provenance(self) -> dict:
        return {
            "mode": self.mode,
            "continuous_axes": [list(axis) for axis in self.axes],
            "continuous_symmetry_step_deg": float(self.continuous_step_deg),
            "candidate_count": int(self.candidate_count),
            "full_rotation_detected": bool(self.full_rotational),
            "candidate_guard_limit": int(self.guard_limit),
        }


def identity_symmetry_spec(
    continuous_step_deg: float = DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG,
) -> SymmetrySpec:
    """Return provenance for an object without a symmetry annotation."""
    return SymmetrySpec(
        transforms=np.eye(4, dtype=float)[None, ...],
        mode="none",
        axes=(),
        continuous_step_deg=float(continuous_step_deg),
        full_rotational=False,
        candidate_count=1,
    )


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError("Continuous symmetry axis must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("Continuous symmetry axis must be nonzero")
    axis = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _close_group(
    transforms: list[np.ndarray],
    atol: float = 1e-6,
    max_size: int = MAX_SYMMETRY_GROUP_SIZE,
) -> list[np.ndarray]:
    elems: list[np.ndarray] = [np.eye(4)]
    for transform in transforms:
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Discrete symmetry must be a finite 4x4 transform")
        if not any(np.allclose(transform, elem, atol=atol) for elem in elems):
            elems.append(transform)
            if len(elems) > max_size:
                raise ValueError(f"symmetry group exceeds {max_size} candidates")
    while True:
        new: list[np.ndarray] = []
        for left in elems:
            for right in elems:
                product = left @ right
                if any(np.allclose(product, elem, atol=atol) for elem in elems):
                    continue
                if any(np.allclose(product, elem, atol=atol) for elem in new):
                    continue
                new.append(product)
                if len(elems) + len(new) > max_size:
                    raise ValueError(
                        f"symmetry group exceeds {max_size} candidates; "
                        "inputs may not generate a finite group"
                    )
        if not new:
            return elems
        elems.extend(new)


def _normalized_continuous_axes(
    continuous: list[dict],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return all axes and deduplicated centered axes.

    Parallel and antiparallel axes through the object origin describe the same
    axial symmetry and are collapsed before either full-rotation detection or
    discretization.
    """
    all_axes: list[np.ndarray] = []
    centered_unique: list[np.ndarray] = []
    for item in continuous:
        axis = np.asarray(item.get("axis"), dtype=float)
        offset = np.asarray(item.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        if axis.shape != (3,) or offset.shape != (3,):
            raise ValueError("Continuous symmetry axis/offset must have shape (3,)")
        if not np.isfinite(axis).all() or not np.isfinite(offset).all():
            raise ValueError("Continuous symmetry axis/offset must be finite")
        norm = float(np.linalg.norm(axis))
        if norm <= 0:
            raise ValueError("Continuous symmetry axis must be nonzero")
        normalized = axis / norm
        all_axes.append(normalized)
        if np.allclose(offset, 0.0, atol=1e-9) and not any(
            abs(float(np.dot(normalized, existing))) >= 1.0 - _AXIS_ATOL
            for existing in centered_unique
        ):
            centered_unique.append(normalized)
    return all_axes, centered_unique


def load_symmetry_spec(
    json_path: str | Path,
    continuous_step_deg: float = DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG,
    max_candidates: int = MAX_SYMMETRY_GROUP_SIZE,
) -> SymmetrySpec:
    """Load a BOP-style annotation without materializing unbounded products."""
    step_deg = float(continuous_step_deg)
    if not np.isfinite(step_deg) or step_deg <= 0 or step_deg > 360:
        raise ValueError("continuous symmetry step must be in (0, 360] degrees")
    if int(max_candidates) < 1:
        raise ValueError("symmetry candidate guard must be positive")

    data = json.loads(Path(json_path).read_text())
    continuous = data.get("symmetries_continuous", [])
    all_axes, centered_axes = _normalized_continuous_axes(continuous)
    effective_axes: list[np.ndarray] = []
    centered_seen_for_provenance: list[np.ndarray] = []
    for item, axis in zip(continuous, all_axes):
        offset = np.asarray(item.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        if np.allclose(offset, 0.0, atol=1e-9):
            if any(
                abs(float(np.dot(axis, existing))) >= 1.0 - _AXIS_ATOL
                for existing in centered_seen_for_provenance
            ):
                continue
            centered_seen_for_provenance.append(axis)
        effective_axes.append(axis)
    axes_for_provenance = tuple(
        tuple(float(v) for v in axis) for axis in effective_axes
    )

    if len(centered_axes) >= 2 and np.linalg.matrix_rank(
        np.stack(centered_axes), tol=_AXIS_ATOL,
    ) >= 2:
        return SymmetrySpec(
            transforms=np.eye(4, dtype=float)[None, ...],
            mode="full_rotational",
            axes=axes_for_provenance,
            continuous_step_deg=step_deg,
            full_rotational=True,
            candidate_count=1,
            guard_limit=int(max_candidates),
        )

    discrete = [
        np.asarray(value, dtype=float).reshape(4, 4)
        for value in data.get("symmetries_discrete", [])
    ]
    group = _close_group(discrete, max_size=int(max_candidates))
    steps = max(1, int(round(360.0 / step_deg)))

    # Deduplicate centered parallel/antiparallel axes. Non-centered axes retain
    # their historical treatment and are kept individually.
    sampling_axes: list[np.ndarray] = []
    centered_seen: list[np.ndarray] = []
    for item, axis in zip(continuous, all_axes):
        offset = np.asarray(item.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        if np.allclose(offset, 0.0, atol=1e-9):
            if any(
                abs(float(np.dot(axis, existing))) >= 1.0 - _AXIS_ATOL
                for existing in centered_seen
            ):
                continue
            centered_seen.append(axis)
        sampling_axes.append(axis)

    projected_count = len(group) * (steps ** len(sampling_axes))
    if projected_count > int(max_candidates):
        raise ValueError(
            f"symmetry group would contain {projected_count} candidates, "
            f"above guard limit {int(max_candidates)}"
        )

    for axis in sampling_axes:
        rotations = np.repeat(np.eye(4)[None, ...], steps, axis=0)
        rotations[:, :3, :3] = np.stack([
            _rotation_matrix(axis, 2.0 * np.pi * index / steps)
            for index in range(steps)
        ])
        group = list(
            (np.asarray(group)[:, None, :, :] @ rotations[None, :, :, :])
            .reshape(-1, 4, 4)
        )

    transforms = np.asarray(group, dtype=float)
    mode = "none" if len(transforms) == 1 and not continuous and not discrete else (
        "axial" if sampling_axes else "finite"
    )
    return SymmetrySpec(
        transforms=transforms,
        mode=mode,
        axes=axes_for_provenance,
        continuous_step_deg=step_deg,
        full_rotational=False,
        candidate_count=len(transforms),
        guard_limit=int(max_candidates),
    )


def load_symmetry_group(
    json_path: str | Path,
    continuous_step_deg: float = DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG,
) -> list[np.ndarray]:
    """Compatibility wrapper returning sampled finite group elements."""
    return list(load_symmetry_spec(json_path, continuous_step_deg).transforms)


def canonicalize_pose(
    pose: np.ndarray,
    group: list[np.ndarray] | np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Pick the symmetry-equivalent pose nearest to ``reference``."""
    pose = np.asarray(pose, dtype=float).reshape(4, 4)
    transforms = np.asarray(group, dtype=float).reshape(-1, 4, 4)
    if len(transforms) == 0:
        raise ValueError("symmetry group must contain at least one candidate")
    candidates = pose[None, :, :] @ transforms
    ref_rotation = np.asarray(reference, dtype=float).reshape(4, 4)[:3, :3]
    relative = np.einsum(
        "ij,njk->nik", ref_rotation.T, candidates[:, :3, :3], optimize=True,
    )
    cosines = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return candidates[int(np.argmax(cosines))]
