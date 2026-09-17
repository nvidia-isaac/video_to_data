"""Compare commercial FoundationPose output with a frozen legacy trajectory.

The comparison is intentionally independent of either FoundationPose runtime.  It
operates on exported ``poses.npy`` arrays, optional validity masks, the aligned
object mesh, and the BOP-style symmetry annotation copied with that mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from scipy.spatial import cKDTree


DEFAULT_TOLERANCES = {
    "translation_m": {"median_max": 0.035, "p95_max": 0.08},
    "rotation_deg": {"median_max": 15.0, "p95_max": 45.0},
    "normalized_adds": {"median_max": 0.10, "p95_max": 0.25},
    "valid_coverage_min": 0.95,
    "newly_invalid_fraction_max": 0.02,
    "divergent_frame_fraction": {
        "max": 0.05,
        "translation_m": 0.08,
        "rotation_deg": 45.0,
        "normalized_adds": 0.25,
    },
}

POSE_FRAMES = ("aligned", "original")
DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG = 10.0
MAX_SYMMETRY_GROUP_SIZE = 4096


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm <= 0:
        raise ValueError("Continuous symmetry axis must be nonzero")
    axis = axis / norm
    x, y, z = axis
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
    ])


def _full_rotational_symmetry(continuous: list[dict]) -> bool:
    """Return whether continuous axes generate unrestricted SO(3) rotation.

    Continuous rotations around two nonparallel axes generate the full
    rotation group.  Detecting that case avoids materializing the Cartesian
    product of sampled rotations (72**3 candidates for the common three-axis
    sphere annotation).

    Nonzero axis offsets are excluded because their generated transforms may
    include translations rather than pure object-frame rotations.
    """
    axes = []
    for item in continuous:
        axis = np.asarray(item.get("axis"), dtype=float)
        offset = np.asarray(item.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        if axis.shape != (3,) or offset.shape != (3,):
            raise ValueError("Continuous symmetry axis/offset must have shape (3,)")
        norm = np.linalg.norm(axis)
        if norm <= 0:
            raise ValueError("Continuous symmetry axis must be nonzero")
        if not np.allclose(offset, 0.0, atol=1e-9):
            return False
        axes.append(axis / norm)
    return len(axes) >= 2 and np.linalg.matrix_rank(np.stack(axes), tol=1e-6) >= 2


def _symmetry_spec(
    path: str | Path | None,
    step_deg: float = DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG,
) -> tuple[list[np.ndarray], bool]:
    step_deg = float(step_deg)
    if not np.isfinite(step_deg) or step_deg <= 0 or step_deg > 360:
        raise ValueError("Continuous symmetry step must be in (0, 360] degrees")
    group = [np.eye(4)]
    if path is None or not Path(path).exists():
        return group, False
    data = json.loads(Path(path).read_text())
    continuous = data.get("symmetries_continuous", [])
    if _full_rotational_symmetry(continuous):
        # Every object-frame rotation is equivalent, so discrete rotations are
        # already contained in SO(3) and no sampled group is needed.
        return group, True
    for value in data.get("symmetries_discrete", []):
        transform = np.asarray(value, dtype=float).reshape(4, 4)
        if not any(np.allclose(transform, item, atol=1e-6) for item in group):
            group.append(transform)
    # The files produced for this pipeline normally already contain a closed
    # discrete group. Close small groups defensively without allowing malformed
    # annotations to grow without bound.
    changed = True
    while changed:
        changed = False
        for left in tuple(group):
            for right in tuple(group):
                value = left @ right
                if not any(np.allclose(value, item, atol=1e-6) for item in group):
                    group.append(value)
                    changed = True
                    if len(group) > MAX_SYMMETRY_GROUP_SIZE:
                        raise ValueError(
                            f"Symmetry group exceeds {MAX_SYMMETRY_GROUP_SIZE} elements"
                        )
    steps = max(1, int(round(360.0 / step_deg)))
    sampling_axes: list[np.ndarray] = []
    centered_axes: list[np.ndarray] = []
    for continuous_item in continuous:
        axis = np.asarray(continuous_item["axis"], dtype=float)
        norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or not np.isfinite(axis).all() or norm <= 0:
            raise ValueError("Continuous symmetry axis must be a finite nonzero 3-vector")
        axis = axis / norm
        offset = np.asarray(continuous_item.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Continuous symmetry offset must be a finite 3-vector")
        if np.allclose(offset, 0.0, atol=1e-9):
            if any(abs(float(np.dot(axis, seen))) >= 1.0 - 1e-6 for seen in centered_axes):
                continue
            centered_axes.append(axis)
        sampling_axes.append(axis)

    candidate_count = len(group) * (steps ** len(sampling_axes))
    if candidate_count > MAX_SYMMETRY_GROUP_SIZE:
        raise ValueError(
            f"Symmetry group would contain {candidate_count} candidates, "
            f"above guard limit {MAX_SYMMETRY_GROUP_SIZE}"
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
    return group, False


def _symmetry_group(
    path: str | Path | None,
    step_deg: float = DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG,
) -> list[np.ndarray]:
    """Return sampled symmetries for callers that do not need the mode flag."""
    return _symmetry_spec(path, step_deg=step_deg)[0]


def _aligned_to_original_transform(path: str | Path) -> np.ndarray:
    """Return the transform mapping aligned-mesh points to original-mesh points.

    ``hoi-tools`` records the forward mesh alignment as::

        aligned_pt = rotation @ (original_pt - centroid)

    A pose produced with the original mesh maps original object coordinates to
    the scene.  To express the same physical pose for ``output_aligned.glb``,
    compose it on the right with the inverse mesh alignment returned here.
    """
    data = json.loads(Path(path).read_text())
    alignment = data.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("Symmetry metadata is missing alignment information")
    centroid = np.asarray(alignment.get("centroid"), dtype=float)
    rotation = np.asarray(alignment.get("rotation"), dtype=float)
    if centroid.shape != (3,) or rotation.size != 16:
        raise ValueError("Invalid centroid/rotation in symmetry alignment metadata")
    rotation = rotation.reshape(4, 4)
    if not np.allclose(rotation[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous alignment rotation")
    rotation3 = rotation[:3, :3]
    if not np.allclose(rotation3.T @ rotation3, np.eye(3), atol=1e-5):
        raise ValueError("Alignment rotation is not orthonormal")
    aligned_to_original = np.eye(4)
    aligned_to_original[:3, :3] = rotation3.T
    aligned_to_original[:3, 3] = centroid
    return aligned_to_original


def _convert_legacy_poses_to_aligned_frame(
    poses: np.ndarray, symmetry_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    aligned_to_original = _aligned_to_original_transform(symmetry_path)
    converted = np.matmul(
        np.asarray(poses, dtype=float).reshape(-1, 4, 4),
        aligned_to_original,
    )
    return converted, aligned_to_original


def _valid_mask(path: str | Path | None, poses: np.ndarray) -> np.ndarray:
    finite = np.isfinite(poses).all(axis=(1, 2))
    if path is None or not Path(path).exists():
        return finite
    mask = np.asarray(np.load(path), dtype=bool).reshape(-1)
    if len(mask) != len(poses):
        raise ValueError("Pose validity mask length does not match pose count")
    return mask & finite


def _percentiles(values: np.ndarray) -> dict:
    if not len(values):
        return {"median": None, "p95": None, "max": None}
    return {
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _segments(mask: np.ndarray, minimum_length: int) -> list[dict]:
    segments: list[dict] = []
    start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= minimum_length:
                segments.append({"start_frame": start, "end_frame": index - 1,
                                 "frame_count": index - start})
            start = None
    return segments


def _configured_tolerances(tolerances: dict | None) -> dict:
    """Normalize comparison tolerances, including frozen legacy configs.

    Campaign 459's immutable configuration contains the former
    ``sustained_divergence`` block.  Its metric thresholds remain authoritative,
    but the consecutive-frame length is intentionally ignored and replaced by
    the global five-percent rule.  Future configurations use the explicit
    ``divergent_frame_fraction`` block.
    """
    configured = json.loads(json.dumps(DEFAULT_TOLERANCES))
    supplied = json.loads(json.dumps(tolerances or {}))
    legacy_divergence = supplied.pop("sustained_divergence", None)
    divergence = supplied.pop("divergent_frame_fraction", None)

    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(configured.get(key), dict):
            configured[key].update(value)
        else:
            configured[key] = value

    threshold_source = divergence if divergence is not None else legacy_divergence
    if threshold_source:
        for key in ("max", "translation_m", "rotation_deg", "normalized_adds"):
            if key in threshold_source:
                configured["divergent_frame_fraction"][key] = threshold_source[key]
    return configured


def _mesh_points(path: str | Path, maximum: int = 2000) -> tuple[np.ndarray, float]:
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) == 0:
        raise ValueError("Object mesh has no vertices")
    if len(vertices) > maximum:
        # Stable, deterministic downsampling keeps comparison reports repeatable.
        indices = np.linspace(0, len(vertices) - 1, maximum, dtype=int)
        vertices = vertices[indices]
    bounds = np.asarray(mesh.bounds, dtype=float)
    diameter = float(np.linalg.norm(bounds[1] - bounds[0]))
    if diameter <= 0:
        raise ValueError("Object mesh diameter must be positive")
    return vertices, diameter


def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def compare_pose_arrays(
    legacy_poses: np.ndarray,
    commercial_poses: np.ndarray,
    *,
    mesh_vertices: np.ndarray,
    mesh_diameter: float,
    legacy_valid: np.ndarray | None = None,
    commercial_valid: np.ndarray | None = None,
    symmetries: Iterable[np.ndarray] = (np.eye(4),),
    full_rotational_symmetry: bool = False,
    tolerances: dict | None = None,
) -> dict:
    legacy = np.asarray(legacy_poses, dtype=float).reshape(-1, 4, 4)
    commercial = np.asarray(commercial_poses, dtype=float).reshape(-1, 4, 4)
    if legacy.shape != commercial.shape:
        raise ValueError(
            f"Pose arrays differ in shape: legacy={legacy.shape}, commercial={commercial.shape}"
        )
    count = len(legacy)
    legacy_valid = (
        np.asarray(legacy_valid, dtype=bool).reshape(-1)
        if legacy_valid is not None else np.isfinite(legacy).all(axis=(1, 2))
    )
    commercial_valid = (
        np.asarray(commercial_valid, dtype=bool).reshape(-1)
        if commercial_valid is not None else np.isfinite(commercial).all(axis=(1, 2))
    )
    if len(legacy_valid) != count or len(commercial_valid) != count:
        raise ValueError("Validity-mask length does not match pose count")
    comparable = legacy_valid & commercial_valid
    translation = np.full(count, np.nan)
    rotation = np.full(count, np.nan)
    adds = np.full(count, np.nan)
    group = [np.asarray(item, dtype=float).reshape(4, 4) for item in symmetries]
    if not group:
        group = [np.eye(4)]
    for index in np.flatnonzero(comparable):
        old = legacy[index]
        new = commercial[index]
        if full_rotational_symmetry:
            # ADD-S already compares each transformed point to its nearest
            # neighbor, so it is symmetry-aware without enumerating rotations.
            # For unrestricted rotational symmetry, orientation error is
            # undefined/equivalent and therefore zero by protocol.
            candidate = new
            rotation[index] = 0.0
            translation[index] = float(
                np.linalg.norm(old[:3, 3] - new[:3, 3])
            )
        else:
            candidates = new[None, ...] @ np.asarray(group)
            relative = np.einsum(
                "ij,njk->nik", old[:3, :3].T, candidates[:, :3, :3], optimize=True,
            )
            cosines = np.clip(
                (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0,
            )
            rot_values = np.degrees(np.arccos(cosines))
            best = int(np.argmin(rot_values))
            candidate = candidates[best]
            rotation[index] = float(rot_values[best])
            translation[index] = float(
                np.linalg.norm(old[:3, 3] - candidate[:3, 3])
            )
        old_points = _transform(mesh_vertices, old)
        new_points = _transform(mesh_vertices, candidate)
        adds[index] = float(np.mean(cKDTree(old_points).query(new_points, k=1)[0])) / mesh_diameter

    configured = _configured_tolerances(tolerances)
    comparable_count = int(comparable.sum())
    legacy_valid_count = int(legacy_valid.sum())
    coverage = comparable_count / max(1, legacy_valid_count)
    newly_invalid = legacy_valid & ~commercial_valid
    newly_invalid_fraction = int(newly_invalid.sum()) / max(1, legacy_valid_count)
    metric_values = {
        "translation_m": translation[comparable],
        "rotation_deg": rotation[comparable],
        "normalized_adds": adds[comparable],
    }
    summaries = {name: _percentiles(values) for name, values in metric_values.items()}
    divergence_cfg = configured["divergent_frame_fraction"]
    divergent = comparable & (
        (translation > float(divergence_cfg["translation_m"]))
        | (rotation > float(divergence_cfg["rotation_deg"]))
        | (adds > float(divergence_cfg["normalized_adds"]))
    )
    divergent_count = int(divergent.sum())
    divergent_fraction = divergent_count / max(1, comparable_count)
    divergence_segments = _segments(divergent, 1)
    failures: list[str] = []
    if comparable_count == 0:
        failures.append("no_comparable_frames")
    if coverage < float(configured["valid_coverage_min"]):
        failures.append("valid_coverage")
    if newly_invalid_fraction > float(configured["newly_invalid_fraction_max"]):
        failures.append("newly_invalid_frames")
    for name in ("translation_m", "rotation_deg", "normalized_adds"):
        for statistic in ("median", "p95"):
            observed = summaries[name][statistic]
            maximum = float(configured[name][f"{statistic}_max"])
            if observed is None or observed > maximum:
                failures.append(f"{name}_{statistic}")
    if divergent_fraction > float(divergence_cfg["max"]):
        failures.append("divergent_frame_fraction")
    return {
        "schema": "v2d.mv_hoi.foundation_pose_comparison.v2",
        "status": "PASS" if not failures else "FAIL",
        "frame_count": count,
        "legacy_valid_frames": legacy_valid_count,
        "commercial_valid_frames": int(commercial_valid.sum()),
        "comparable_frames": comparable_count,
        "valid_coverage": coverage,
        "newly_invalid_frames": int(newly_invalid.sum()),
        "newly_invalid_fraction": newly_invalid_fraction,
        "full_rotational_symmetry": bool(full_rotational_symmetry),
        "symmetry_candidate_count": 1 if full_rotational_symmetry else len(group),
        "translation_m": summaries["translation_m"],
        "rotation_deg": summaries["rotation_deg"],
        "normalized_adds": summaries["normalized_adds"],
        "divergent_frames": divergent_count,
        "divergent_frame_fraction": divergent_fraction,
        "divergence_segments": divergence_segments,
        "failures": failures,
        "tolerances": configured,
        "tolerances_sha256": hashlib.sha256(_canonical_json(configured).encode()).hexdigest(),
    }


def compare_pose_files(
    *, legacy_pose_path: str | Path, commercial_pose_path: str | Path,
    mesh_path: str | Path, output_path: str | Path,
    legacy_valid_path: str | Path | None = None,
    commercial_valid_path: str | Path | None = None,
    symmetry_path: str | Path | None = None,
    tolerances_path: str | Path | None = None,
    legacy_pose_frame: str = "aligned",
    continuous_symmetry_step_deg: float = DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG,
) -> dict:
    legacy = np.load(legacy_pose_path)
    commercial = np.load(commercial_pose_path)
    if legacy_pose_frame not in POSE_FRAMES:
        raise ValueError(
            f"Unsupported legacy pose frame {legacy_pose_frame!r}; expected {POSE_FRAMES}"
        )
    alignment_transform = None
    if legacy_pose_frame == "original":
        if symmetry_path is None or not Path(symmetry_path).is_file():
            raise ValueError(
                "Original-frame legacy poses require output_symmetry.json alignment metadata"
            )
        legacy, alignment_transform = _convert_legacy_poses_to_aligned_frame(
            legacy, symmetry_path,
        )
    legacy_valid = _valid_mask(legacy_valid_path, legacy)
    commercial_valid = _valid_mask(commercial_valid_path, commercial)
    vertices, diameter = _mesh_points(mesh_path)
    tolerances = json.loads(Path(tolerances_path).read_text()) if tolerances_path else None
    if tolerances and "pose_comparison_tolerances" in tolerances:
        tolerances = tolerances["pose_comparison_tolerances"]
    symmetries, full_rotational_symmetry = _symmetry_spec(
        symmetry_path, step_deg=continuous_symmetry_step_deg,
    )
    result = compare_pose_arrays(
        legacy, commercial, mesh_vertices=vertices, mesh_diameter=diameter,
        legacy_valid=legacy_valid, commercial_valid=commercial_valid,
        symmetries=symmetries,
        full_rotational_symmetry=full_rotational_symmetry,
        tolerances=tolerances,
    )
    result.update({
        "legacy_pose_frame": legacy_pose_frame,
        "comparison_pose_frame": "aligned",
        "legacy_alignment_applied": alignment_transform is not None,
        "legacy_alignment_aligned_to_original": (
            alignment_transform.reshape(-1).tolist()
            if alignment_transform is not None else None
        ),
        "legacy_pose_sha256": hashlib.sha256(Path(legacy_pose_path).read_bytes()).hexdigest(),
        "legacy_comparison_pose_sha256": hashlib.sha256(
            np.ascontiguousarray(legacy).tobytes()
        ).hexdigest(),
        "commercial_pose_sha256": hashlib.sha256(Path(commercial_pose_path).read_bytes()).hexdigest(),
        "mesh_sha256": hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest(),
        "symmetry_sha256": (
            hashlib.sha256(Path(symmetry_path).read_bytes()).hexdigest()
            if symmetry_path is not None and Path(symmetry_path).is_file() else None
        ),
        "continuous_symmetry_step_deg": float(continuous_symmetry_step_deg),
    })
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-pose-path", required=True)
    parser.add_argument("--commercial-pose-path", required=True)
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--legacy-valid-path")
    parser.add_argument("--commercial-valid-path")
    parser.add_argument("--symmetry-path")
    parser.add_argument("--tolerances-path")
    parser.add_argument("--legacy-pose-frame", choices=POSE_FRAMES, default="aligned")
    parser.add_argument(
        "--continuous-symmetry-step-deg", type=float,
        default=DEFAULT_CONTINUOUS_SYMMETRY_STEP_DEG,
    )
    args = parser.parse_args()
    result = compare_pose_files(**vars(args))
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
