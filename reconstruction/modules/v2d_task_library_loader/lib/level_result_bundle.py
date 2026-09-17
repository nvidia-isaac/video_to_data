# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply a single world-leveling rotation to an ego reconstruction bundle.

Gravity alignment (GeoCalib) can leave a residual tilt that puts the object on an
edge rather than a face for an entire clip. Diagnose it with the contact
footprint: mesh vertices within 5 mm of the lowest world z at a rest frame form a
broad planar patch when the object is level and a thin strip when it is not. A
tilt that stays constant while the object is carried is a world-frame error, not
object motion, so it is corrected once, globally.

Neither loader flag reliably fixes this. ``--no_ground_align`` keeps the tilt,
and the loader's OBB path picks "up" by smallest PCA extent, which is arbitrary
when an object's two smaller extents are close. Level the bundle here, then load
the result with ``--no_ground_align``.

The correction rotates the object's canonical (bbox-volume-minimizing) vertical
axis at the anchor frame onto world +Z, and applies that rotation to every
``*_to_world`` transform. Camera-relative quantities are untouched: the loader
rebuilds hands from ``camera_to_world_transform`` composed with camera-frame
MANO parameters, so rotating the camera track carries the hands with it.

``--up_rank`` selects which canonical axis is vertical, by descending extent.
Prefer the rank that yields the smallest correction angle: a large angle means
re-orienting the object onto a different face, which the reconstruction's own
gravity estimate would have to be badly wrong to justify.

Example (the shipped ``tissue_box_simple``, which needed 8.4 degrees)::

    python -m v2d.task_library_loader.lib.level_result_bundle
        --src <clip>/result_slam_gravity_aligned_scaled
        --dst <clip>/result_leveled --up_rank 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import product
from pathlib import Path

import numpy as np

WORLD_TRANSFORM_KEYS = (
    "camera_to_world_transform",
    "object_to_world_transform",
    "hand_left_wrist_to_world_transform",
    "hand_right_wrist_to_world_transform",
)


def _rot(axis: int, theta: float) -> np.ndarray:
    """Elementary rotation of ``theta`` radians about ``axis``."""
    c, s = np.cos(theta), np.sin(theta)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _read_obj_vertices(path: Path) -> np.ndarray:
    """Parse ``v`` lines from an OBJ without pulling in a mesh library."""
    with path.open() as handle:
        rows = [
            [float(v) for v in line.split()[1:4]]
            for line in handle
            if line.startswith("v ")
        ]
    return np.asarray(rows, dtype=np.float64)


def canonical_rotation(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the rotation whose axis-aligned bbox of ``vertices`` is smallest.

    Raw bbox extents are meaningless when a mesh is tilted inside its own local
    frame, which this one is by 37 degrees. Returns the rotation and the
    resulting extents, both in mesh-local coordinates.
    """
    centered = vertices - vertices.mean(axis=0)
    best_volume, best_angles = np.inf, (0.0, 0.0, 0.0)

    coarse = np.deg2rad(np.arange(0.0, 90.0, 2.0))
    for angles in product(coarse, repeat=3):
        rotation = _rot(0, angles[0]) @ _rot(1, angles[1]) @ _rot(2, angles[2])
        extents = np.ptp(centered @ rotation.T, axis=0)
        volume = float(np.prod(extents))
        if volume < best_volume:
            best_volume, best_angles = volume, angles

    def refine(angle: float) -> np.ndarray:
        """Half-degree sweep spanning +-2 degrees around a coarse solution."""
        return np.deg2rad(np.rad2deg(angle) + np.arange(-2.0, 2.01, 0.25))

    for angles in product(*(refine(a) for a in best_angles)):
        rotation = _rot(0, angles[0]) @ _rot(1, angles[1]) @ _rot(2, angles[2])
        extents = np.ptp(centered @ rotation.T, axis=0)
        volume = float(np.prod(extents))
        if volume < best_volume:
            best_volume, best_angles = volume, angles

    rotation = (
        _rot(0, best_angles[0]) @ _rot(1, best_angles[1]) @ _rot(2, best_angles[2])
    )
    return rotation, np.ptp(centered @ rotation.T, axis=0)


def minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest rotation carrying ``source`` onto ``target`` (Rodrigues)."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    axis = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if np.linalg.norm(axis) < 1e-12:
        return np.eye(3) if cosine > 0 else -np.eye(3)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def contact_footprint(vertices: np.ndarray, band: float = 0.005) -> tuple[float, float]:
    """Planar extents of the vertices within ``band`` of the lowest world z."""
    low = vertices[vertices[:, 2] < vertices[:, 2].min() + band][:, :2]
    centered = low - low.mean(axis=0)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    extents = np.ptp(centered @ basis.T, axis=0)
    return float(extents[0]), float(extents[1])


def main() -> None:
    """Level a bundle and write the corrected copy, reporting before/after footprints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Source bundle dir.")
    parser.add_argument("--dst", type=Path, required=True, help="Output bundle dir.")
    parser.add_argument(
        "--anchor_frame",
        type=int,
        default=0,
        help="Frame whose resting pose defines level. Frame 0 is the reset pose.",
    )
    parser.add_argument(
        "--up_rank",
        type=int,
        default=1,
        help="Canonical axis to send to +Z, by descending extent: "
        "0=longest, 1=middle, 2=shortest.",
    )
    args = parser.parse_args()

    data = dict(np.load(args.src / "result.npz", allow_pickle=False))
    vertices = _read_obj_vertices(args.src / "mesh.obj")

    rotation, extents = canonical_rotation(vertices)
    order = np.argsort(extents)[::-1]
    print(f"canonical dims (cm): {np.round(extents[order] * 100, 2)}")

    pose = data["object_to_world_transform"][args.anchor_frame].astype(np.float64)
    up_world = pose[:3, :3] @ rotation[order[args.up_rank]]
    if up_world[2] < 0.0:
        up_world = -up_world
    correction = minimal_rotation(up_world, np.array([0.0, 0.0, 1.0]))

    angle = np.rad2deg(
        np.arccos(np.clip((np.trace(correction) - 1.0) / 2.0, -1.0, 1.0))
    )
    print(f"correction: {angle:.2f} deg about {np.round(up_world, 4)} -> +Z")
    assert abs(np.linalg.det(correction) - 1.0) < 1e-9, "correction is not a rotation"

    world_points = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
    before = contact_footprint(world_points)
    after = contact_footprint((correction @ world_points.T).T)
    print(f"footprint before: {before[0] * 100:.1f} x {before[1] * 100:.1f} cm")
    print(f"footprint after:  {after[0] * 100:.1f} x {after[1] * 100:.1f} cm")

    for key in WORLD_TRANSFORM_KEYS:
        transforms = data[key].astype(np.float64)
        transforms[:, :3, :3] = np.einsum(
            "ij,njk->nik", correction, transforms[:, :3, :3]
        )
        transforms[:, :3, 3] = (correction @ transforms[:, :3, 3].T).T
        data[key] = transforms.astype(np.float32)

    alignment = data["gravity_alignment_transform"].astype(np.float64)
    alignment[:3, :3] = correction @ alignment[:3, :3]
    alignment[:3, 3] = correction @ alignment[:3, 3]
    data["gravity_alignment_transform"] = alignment.astype(np.float32)

    args.dst.mkdir(parents=True, exist_ok=True)
    np.savez(args.dst / "result.npz", **data)
    for name in ("mesh.obj", "material.mtl", "material_0.png", "README.md"):
        source_file = args.src / name
        if source_file.exists():
            shutil.copy2(source_file, args.dst / name)

    manifest = json.loads((args.src / "manifest.json").read_text())
    manifest["world_leveling_correction"] = {
        "applied": True,
        "source_result_dir": str(args.src),
        "rotation_matrix": correction.tolist(),
        "rotation_angle_deg": float(angle),
        "anchor_frame": args.anchor_frame,
        "canonical_up_rank": args.up_rank,
        "canonical_dims_m": extents[order].tolist(),
        "contact_footprint_before_m": list(before),
        "contact_footprint_after_m": list(after),
        "rationale": (
            "GeoCalib left a constant residual tilt: the object's resting face was "
            f"{angle:.2f} deg off horizontal across all frames, including while the box "
            "was airborne, which identifies it as a world-frame error rather than "
            "object motion. Applied to every *_to_world transform; camera-relative "
            "quantities are unchanged."
        ),
        "keys_rotated": list(WORLD_TRANSFORM_KEYS) + ["gravity_alignment_transform"],
    }
    (args.dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
