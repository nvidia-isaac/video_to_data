#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline geometry comparison for HOI reconstruction meshes.

The evaluator intentionally ignores texture and color. It samples points on
the mesh surfaces, registers each candidate to a reference with rotation and
translation only, and reports an unsquared symmetric Chamfer-L2 distance. The
distance is normalized by the reference axis-aligned bounding-box diagonal so
objects of different physical sizes can be summarized together.

This is analysis tooling, not a reconstruction-stage or EVT pass/fail gate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree


DEFAULT_METRIC_SAMPLES = 10_000
DEFAULT_ALIGNMENT_SAMPLES = 3_000
DEFAULT_ICP_ITERATIONS = 40
DEFAULT_ALIGNMENT_STARTS = 25
DEFAULT_TRIM_FRACTION = 0.9
DEFAULT_THRESHOLD_PERCENTS = (1.0, 2.0, 5.0)
_EPSILON = 1e-12


@dataclass(frozen=True)
class EvaluationConfig:
    """Controls deterministic sampling, registration, and metric reporting."""

    metric_samples: int = DEFAULT_METRIC_SAMPLES
    alignment_samples: int = DEFAULT_ALIGNMENT_SAMPLES
    icp_iterations: int = DEFAULT_ICP_ITERATIONS
    alignment_starts: int = DEFAULT_ALIGNMENT_STARTS
    trim_fraction: float = DEFAULT_TRIM_FRACTION
    threshold_percents: tuple[float, ...] = DEFAULT_THRESHOLD_PERCENTS
    seed: int = 0

    def validate(self) -> None:
        if self.metric_samples < 10:
            raise ValueError("metric_samples must be at least 10")
        if self.alignment_samples < 10:
            raise ValueError("alignment_samples must be at least 10")
        if self.icp_iterations < 1:
            raise ValueError("icp_iterations must be at least 1")
        if self.alignment_starts < 1:
            raise ValueError("alignment_starts must be at least 1")
        if not 0.5 <= self.trim_fraction <= 1.0:
            raise ValueError("trim_fraction must be in [0.5, 1.0]")
        if not self.threshold_percents:
            raise ValueError("at least one threshold percent is required")
        if any(value <= 0.0 for value in self.threshold_percents):
            raise ValueError("threshold percents must be positive")


def stable_seed(*parts: object, base_seed: int = 0) -> int:
    """Return a process-independent uint32 seed for a named sample."""

    payload = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _scene_meshes(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(np.asarray(transform, dtype=np.float64))
        meshes.append(mesh)
    return meshes


def load_mesh(path: Path | str) -> trimesh.Trimesh:
    """Load a mesh and apply scene-node transforms before concatenation."""

    mesh_path = Path(path).expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = _scene_meshes(loaded)
        if not meshes:
            raise ValueError(f"No triangle geometry found in {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise TypeError(f"Unsupported mesh type in {mesh_path}: {type(loaded)}")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(vertices) < 3 or len(faces) < 1:
        raise ValueError(f"Mesh has no usable triangle surface: {mesh_path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"Mesh contains non-finite vertices: {mesh_path}")
    if not math.isfinite(float(mesh.area)) or float(mesh.area) <= _EPSILON:
        raise ValueError(f"Mesh has zero or invalid surface area: {mesh_path}")
    return mesh


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> np.ndarray:
    """Sample surface points deterministically using area-weighted faces."""

    if count < 1:
        raise ValueError("sample count must be positive")
    points, _ = trimesh.sample.sample_surface(mesh, count, seed=int(seed))
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError("surface sampling produced invalid points")
    return points


def rms_surface_radius(points: np.ndarray) -> float:
    """Return RMS distance of surface points from their centroid."""

    values = _as_points(points, "points")
    centered = values - values.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.einsum("ij,ij->i", centered, centered))))
    if not math.isfinite(radius) or radius <= _EPSILON:
        raise ValueError("surface points have zero or invalid RMS radius")
    return radius


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError(f"{name} must have shape (N, 3) with N >= 3")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def _pca_basis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(centered), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    basis = eigenvectors[:, ::-1]
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    return basis


def _proper_axis_mappings() -> Iterable[np.ndarray]:
    for permutation in itertools.permutations(range(3)):
        permuted = np.eye(3, dtype=np.float64)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapping = permuted @ np.diag(signs)
            if np.linalg.det(mapping) > 0.0:
                yield mapping


def _initial_rotations(
    reference: np.ndarray,
    candidate: np.ndarray,
    maximum: int,
) -> list[np.ndarray]:
    reference_basis = _pca_basis(reference)
    candidate_basis = _pca_basis(candidate)
    rotations = [np.eye(3, dtype=np.float64)]
    seen = {tuple(np.round(rotations[0], decimals=10).ravel())}
    for mapping in _proper_axis_mappings():
        rotation = reference_basis @ mapping @ candidate_basis.T
        if np.linalg.det(rotation) < 0.0:
            continue
        key = tuple(np.round(rotation, decimals=10).ravel())
        if key in seen:
            continue
        seen.add(key)
        rotations.append(rotation)
        if len(rotations) >= maximum:
            break
    return rotations[:maximum]


def _kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def apply_rigid_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply a column-vector rigid transform to row-vector points."""

    return np.asarray(points, dtype=np.float64) @ rotation.T + translation


def _distance_arrays(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference_tree = cKDTree(reference)
    candidate_tree = cKDTree(candidate)
    candidate_to_reference = reference_tree.query(candidate, k=1, workers=1)[0]
    reference_to_candidate = candidate_tree.query(reference, k=1, workers=1)[0]
    return candidate_to_reference, reference_to_candidate


def _symmetric_mean_distance(reference: np.ndarray, candidate: np.ndarray) -> float:
    candidate_distances, reference_distances = _distance_arrays(reference, candidate)
    return 0.5 * (
        float(np.mean(candidate_distances))
        + float(np.mean(reference_distances))
    )


def _run_icp(
    reference: np.ndarray,
    candidate: np.ndarray,
    initial_rotation: np.ndarray,
    iterations: int,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    reference_center = reference.mean(axis=0)
    candidate_center = candidate.mean(axis=0)
    rotation = initial_rotation.copy()
    translation = reference_center - rotation @ candidate_center
    reference_tree = cKDTree(reference)
    previous_error = math.inf

    for _ in range(iterations):
        transformed = apply_rigid_transform(candidate, rotation, translation)
        distances, indices = reference_tree.query(transformed, k=1, workers=1)
        keep_count = max(3, int(math.ceil(len(distances) * trim_fraction)))
        if keep_count < len(distances):
            selected = np.argpartition(distances, keep_count - 1)[:keep_count]
        else:
            selected = np.arange(len(distances))
        matched = reference[indices[selected]]
        delta_rotation, delta_translation = _kabsch(transformed[selected], matched)
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation

        error = float(np.mean(distances[selected]))
        if math.isfinite(previous_error) and abs(previous_error - error) <= (
            1e-8 * max(previous_error, 1.0)
        ):
            break
        previous_error = error
    return rotation, translation


def align_rigid_multistart(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    iterations: int = DEFAULT_ICP_ITERATIONS,
    starts: int = DEFAULT_ALIGNMENT_STARTS,
    trim_fraction: float = DEFAULT_TRIM_FRACTION,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align candidate to reference without reflection or scale correction."""

    reference_points = _as_points(reference, "reference")
    candidate_points = _as_points(candidate, "candidate")
    if iterations < 1 or starts < 1:
        raise ValueError("iterations and starts must be positive")
    if not 0.5 <= trim_fraction <= 1.0:
        raise ValueError("trim_fraction must be in [0.5, 1.0]")

    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for initial_rotation in _initial_rotations(
        reference_points, candidate_points, starts
    ):
        rotation, translation = _run_icp(
            reference_points,
            candidate_points,
            initial_rotation,
            iterations,
            trim_fraction,
        )
        transformed = apply_rigid_transform(candidate_points, rotation, translation)
        score = _symmetric_mean_distance(reference_points, transformed)
        if best is None or score < best[2]:
            best = rotation, translation, score

    if best is None:
        raise RuntimeError("rigid alignment produced no candidate transform")
    return best


def _threshold_key(percent: float) -> str:
    rounded = round(percent)
    if math.isclose(percent, rounded, rel_tol=0.0, abs_tol=1e-9):
        return f"{rounded:d}pct"
    return f"{percent:g}pct".replace(".", "p")


def compute_surface_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    reference_diagonal: float,
    threshold_percents: Sequence[float] = DEFAULT_THRESHOLD_PERCENTS,
) -> dict[str, float]:
    """Compute normalized Chamfer-L2 and directional threshold metrics."""

    reference_points = _as_points(reference, "reference")
    candidate_points = _as_points(candidate, "candidate")
    if not math.isfinite(reference_diagonal) or reference_diagonal <= _EPSILON:
        raise ValueError("reference_diagonal must be positive and finite")

    candidate_distances, reference_distances = _distance_arrays(
        reference_points, candidate_points
    )
    normalization = 100.0 / reference_diagonal
    candidate_mean = float(np.mean(candidate_distances))
    reference_mean = float(np.mean(reference_distances))
    combined = np.concatenate((candidate_distances, reference_distances))
    result: dict[str, float] = {
        "chamfer_mean_pct_diag": 0.5
        * (candidate_mean + reference_mean)
        * normalization,
        "surface_p95_pct_diag": float(np.percentile(combined, 95.0))
        * normalization,
        "candidate_to_reference_mean_pct_diag": candidate_mean * normalization,
        "reference_to_candidate_mean_pct_diag": reference_mean * normalization,
    }

    for percent in threshold_percents:
        threshold = reference_diagonal * float(percent) / 100.0
        precision = 100.0 * float(np.mean(candidate_distances <= threshold))
        recall = 100.0 * float(np.mean(reference_distances <= threshold))
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        suffix = _threshold_key(float(percent))
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"fscore_{suffix}"] = fscore
    return result


def _mesh_metadata(mesh: trimesh.Trimesh, path: Path | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "surface_area": float(mesh.area),
    }
    if path is not None:
        metadata["path"] = str(path)
        metadata["file_size_mb"] = path.stat().st_size / 1_000_000.0
    return metadata


def evaluate_mesh_pair(
    reference_mesh: trimesh.Trimesh,
    candidate_mesh: trimesh.Trimesh,
    *,
    object_id: str,
    method: str,
    config: EvaluationConfig = EvaluationConfig(),
    reference_path: Path | None = None,
    candidate_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate mesh against one reference mesh."""

    config.validate()
    reference_metric = sample_surface(
        reference_mesh,
        config.metric_samples,
        stable_seed(object_id, "reference", "metric", base_seed=config.seed),
    )
    candidate_metric = sample_surface(
        candidate_mesh,
        config.metric_samples,
        stable_seed(object_id, method, "metric", base_seed=config.seed),
    )
    reference_alignment = sample_surface(
        reference_mesh,
        config.alignment_samples,
        stable_seed(object_id, "reference", "alignment", base_seed=config.seed),
    )
    candidate_alignment = sample_surface(
        candidate_mesh,
        config.alignment_samples,
        stable_seed(object_id, method, "alignment", base_seed=config.seed),
    )

    reference_radius = rms_surface_radius(reference_metric)
    candidate_radius = rms_surface_radius(candidate_metric)
    scale_ratio = candidate_radius / reference_radius
    symmetric_scale_error = (max(scale_ratio, 1.0 / scale_ratio) - 1.0) * 100.0
    reference_diagonal = float(np.linalg.norm(reference_mesh.extents))
    if not math.isfinite(reference_diagonal) or reference_diagonal <= _EPSILON:
        raise ValueError("reference mesh has zero or invalid bounding-box diagonal")

    rotation, translation, alignment_score = align_rigid_multistart(
        reference_alignment,
        candidate_alignment,
        iterations=config.icp_iterations,
        starts=config.alignment_starts,
        trim_fraction=config.trim_fraction,
    )
    delivered_candidate = apply_rigid_transform(
        candidate_metric, rotation, translation
    )
    delivered_metrics = compute_surface_metrics(
        reference_metric,
        delivered_candidate,
        reference_diagonal=reference_diagonal,
        threshold_percents=config.threshold_percents,
    )

    candidate_center = candidate_metric.mean(axis=0)
    normalized_metric = (
        candidate_metric - candidate_center
    ) / scale_ratio + candidate_center
    normalized_alignment = (
        candidate_alignment - candidate_center
    ) / scale_ratio + candidate_center
    normalized_rotation, normalized_translation, normalized_score = (
        align_rigid_multistart(
            reference_alignment,
            normalized_alignment,
            iterations=config.icp_iterations,
            starts=config.alignment_starts,
            trim_fraction=config.trim_fraction,
        )
    )
    shape_candidate = apply_rigid_transform(
        normalized_metric, normalized_rotation, normalized_translation
    )
    shape_metrics = compute_surface_metrics(
        reference_metric,
        shape_candidate,
        reference_diagonal=reference_diagonal,
        threshold_percents=config.threshold_percents,
    )

    return {
        "object_id": object_id,
        "method": method,
        "status": "evaluated",
        "reference": _mesh_metadata(reference_mesh, reference_path),
        "candidate": _mesh_metadata(candidate_mesh, candidate_path),
        "reference_diagonal": reference_diagonal,
        "scale_ratio_to_reference": scale_ratio,
        "symmetric_scale_error_pct": symmetric_scale_error,
        "as_delivered": delivered_metrics,
        "shape_scale_normalized": shape_metrics,
        "alignment": {
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "coarse_chamfer": alignment_score,
        },
        "shape_scale_normalized_alignment": {
            "rotation": normalized_rotation.tolist(),
            "translation": normalized_translation.tolist(),
            "coarse_chamfer": normalized_score,
        },
    }


def evaluate_paths(
    reference_path: Path | str,
    candidate_path: Path | str,
    *,
    object_id: str,
    method: str,
    config: EvaluationConfig = EvaluationConfig(),
) -> dict[str, Any]:
    reference = Path(reference_path).expanduser().resolve()
    candidate = Path(candidate_path).expanduser().resolve()
    return evaluate_mesh_pair(
        load_mesh(reference),
        load_mesh(candidate),
        object_id=object_id,
        method=method,
        config=config,
        reference_path=reference,
        candidate_path=candidate,
    )


def metric_definition(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "geometry_only": True,
        "texture_and_color_included": False,
        "metric_samples_per_mesh": config.metric_samples,
        "alignment_samples_per_mesh": config.alignment_samples,
        "alignment": (
            "Multi-start PCA initialization followed by trimmed rigid ICP; "
            "rotation and translation only, without reflection or scale change."
        ),
        "chamfer": (
            "0.5 * (mean candidate-to-reference nearest-neighbor L2 distance + "
            "mean reference-to-candidate nearest-neighbor L2 distance); unsquared."
        ),
        "distance_normalization": (
            "Reference axis-aligned bounding-box diagonal; reported as percent."
        ),
        "threshold_percents": list(config.threshold_percents),
        "precision": "Candidate surface points within the threshold of reference.",
        "recall": "Reference surface points within the threshold of candidate.",
        "scale_ratio": (
            "Candidate RMS surface radius divided by reference RMS surface radius."
        ),
        "shape_scale_normalized": (
            "Candidate is uniformly rescaled by the inverse RMS-radius ratio, then "
            "registered again."
        ),
        "config": asdict(config),
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "population_stddev": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25.0)),
        "p75": float(np.percentile(array, 75.0)),
        "p90": float(np.percentile(array, 90.0)),
    }


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for method in sorted({str(row["method"]) for row in results}):
        rows = [row for row in results if row["method"] == method]
        evaluated = [row for row in rows if row.get("status") == "evaluated"]
        summary: dict[str, Any] = {
            "attempted": len(rows),
            "evaluated": len(evaluated),
            "failed": len(rows) - len(evaluated),
        }
        if evaluated:
            summary["scale_ratio_to_reference"] = _distribution(
                [float(row["scale_ratio_to_reference"]) for row in evaluated]
            )
            summary["symmetric_scale_error_pct"] = _distribution(
                [float(row["symmetric_scale_error_pct"]) for row in evaluated]
            )
            summary["as_delivered_chamfer_pct_diag"] = _distribution(
                [
                    float(row["as_delivered"]["chamfer_mean_pct_diag"])
                    for row in evaluated
                ]
            )
            summary["shape_scale_normalized_chamfer_pct_diag"] = _distribution(
                [
                    float(
                        row["shape_scale_normalized"]["chamfer_mean_pct_diag"]
                    )
                    for row in evaluated
                ]
            )
            summary["watertight_count"] = sum(
                bool(row["candidate"]["watertight"]) for row in evaluated
            )
            precision_keys = sorted(
                key
                for key in evaluated[0]["as_delivered"]
                if key.startswith("precision_")
            )
            for key in precision_keys:
                suffix = key.removeprefix("precision_")
                recall_key = f"recall_{suffix}"
                summary[f"as_delivered_{key}"] = _distribution(
                    [float(row["as_delivered"][key]) for row in evaluated]
                )
                summary[f"as_delivered_{recall_key}"] = _distribution(
                    [float(row["as_delivered"][recall_key]) for row in evaluated]
                )
                fscore_key = f"fscore_{suffix}"
                summary[f"as_delivered_{fscore_key}"] = _distribution(
                    [float(row["as_delivered"][fscore_key]) for row in evaluated]
                )
        methods[method] = summary
    return {"result_count": len(results), "methods": methods}


def _resolve_manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest_dir / path).resolve()


def _manifest_tasks(manifest_path: Path) -> list[dict[str, str]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(cases, list):
        raise ValueError("manifest must be a list or an object containing a cases list")

    tasks: list[dict[str, str]] = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each manifest case must be an object")
        object_id = str(case.get("object_id", "")).strip()
        reference_value = case.get("reference")
        candidates = case.get("candidates")
        if not object_id or not isinstance(reference_value, str):
            raise ValueError("each case requires object_id and reference")
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError("each case requires a non-empty candidates object")
        reference = _resolve_manifest_path(reference_value, manifest_path.parent)
        for method, candidate_value in candidates.items():
            if not isinstance(candidate_value, str):
                raise ValueError("candidate paths must be strings")
            tasks.append(
                {
                    "object_id": object_id,
                    "method": str(method),
                    "reference": str(reference),
                    "candidate": str(
                        _resolve_manifest_path(candidate_value, manifest_path.parent)
                    ),
                }
            )
    return tasks


def evaluate_manifest(
    manifest_path: Path | str,
    *,
    config: EvaluationConfig = EvaluationConfig(),
    keep_going: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate every reference/candidate pair in a JSON manifest."""

    manifest = Path(manifest_path).expanduser().resolve()
    tasks = _manifest_tasks(manifest)
    results: list[dict[str, Any]] = []
    for task in tasks:
        try:
            result = evaluate_paths(
                task["reference"],
                task["candidate"],
                object_id=task["object_id"],
                method=task["method"],
                config=config,
            )
        except Exception as error:
            if not keep_going:
                raise
            result = {
                "object_id": task["object_id"],
                "method": task["method"],
                "status": "failed",
                "reference_path": task["reference"],
                "candidate_path": task["candidate"],
                "error": f"{type(error).__name__}: {error}",
            }
        results.append(result)
    return results


def _write_json(payload: Any, path: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metric-samples", type=int, default=DEFAULT_METRIC_SAMPLES)
    parser.add_argument(
        "--alignment-samples", type=int, default=DEFAULT_ALIGNMENT_SAMPLES
    )
    parser.add_argument("--icp-iterations", type=int, default=DEFAULT_ICP_ITERATIONS)
    parser.add_argument(
        "--alignment-starts", type=int, default=DEFAULT_ALIGNMENT_STARTS
    )
    parser.add_argument("--trim-fraction", type=float, default=DEFAULT_TRIM_FRACTION)
    parser.add_argument(
        "--threshold-percent",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLD_PERCENTS),
        dest="threshold_percents",
        metavar="PERCENT",
    )
    parser.add_argument("--seed", type=int, default=0)


def _config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    config = EvaluationConfig(
        metric_samples=args.metric_samples,
        alignment_samples=args.alignment_samples,
        icp_iterations=args.icp_iterations,
        alignment_starts=args.alignment_starts,
        trim_fraction=args.trim_fraction,
        threshold_percents=tuple(args.threshold_percents),
        seed=args.seed,
    )
    config.validate()
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pair = subparsers.add_parser("pair", help="evaluate one candidate/reference pair")
    pair.add_argument("--reference", type=Path, required=True)
    pair.add_argument("--candidate", type=Path, required=True)
    pair.add_argument("--object-id", required=True)
    pair.add_argument("--method", required=True)
    pair.add_argument("--output", type=Path)
    _add_metric_arguments(pair)

    batch = subparsers.add_parser("manifest", help="evaluate a JSON manifest")
    batch.add_argument("--manifest", type=Path, required=True)
    batch.add_argument("--output-dir", type=Path, required=True)
    batch.add_argument("--keep-going", action="store_true")
    _add_metric_arguments(batch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    if args.command == "pair":
        result = evaluate_paths(
            args.reference,
            args.candidate,
            object_id=args.object_id,
            method=args.method,
            config=config,
        )
        _write_json(
            {"metric_definition": metric_definition(config), "result": result},
            args.output,
        )
        return 0

    results = evaluate_manifest(
        args.manifest,
        config=config,
        keep_going=args.keep_going,
    )
    output_dir = args.output_dir.expanduser().resolve()
    _write_json(results, output_dir / "mesh_quality_results.json")
    _write_json(
        {
            "metric_definition": metric_definition(config),
            "summary": summarize_results(results),
        },
        output_dir / "mesh_quality_summary.json",
    )
    return 0 if all(row.get("status") == "evaluated" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
