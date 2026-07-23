"""Select a GraspGenX grasp and refine one parallel-jaw interaction.

The command consumes already-generated candidates and writes the same exact
12-key robot-neutral target contract as its base target.  It intentionally
does not run GraspGenX, render, or solve embodiment IK.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from inpainting import grasp_refinement
from inpainting.adapters import parallel_jaw_from_tracking
from inpainting.contracts import ContractError, validate_tracking_arrays
from inpainting.graspgenx_candidates import validate_candidates


RUN_SCHEMA = "v2d.inpainting.graspgenx-refinement-run/v1"
RUNNER_VERSION = "1.1.0"
THUMB_TIP = 4
INDEX_TIP = 8
DEFAULT_ANCHOR_WINDOW_RADIUS = 2
CONTACT_PAIR_REGISTRATION_STRATEGY = (
    "raw_pair_common_translation_via_nearest_surface_midpoint/v1"
)

_GALBOT_SEMANTIC_TO_TCP = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
_YAM_LEFT_SEMANTIC_TO_TCP = np.diag([-1.0, -1.0, 1.0])


class RefinementRunError(RuntimeError):
    """Raised when an orchestration input is ambiguous or inconsistent."""


class ArtifactExistsError(FileExistsError):
    """Raised when an output exists and overwrite was not requested."""


@dataclass(frozen=True)
class RobotProfile:
    """Auditable geometry needed before embodiment-specific IK."""

    robot_id: str
    aperture_limits_m: tuple[float, float]
    sweep_volume: grasp_refinement.ParallelJawSweepVolume
    gripper_base_to_contact_z_m: float
    semantic_to_tcp_rotation: Mapping[str, np.ndarray]
    profile_facts: Mapping[str, Any]

    def semantic_to_tcp(self, side: str) -> np.ndarray:
        return np.asarray(self.semantic_to_tcp_rotation[side], dtype=np.float64)

    def T_gripper_base_semantic_contact(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[2, 3] = self.gripper_base_to_contact_z_m
        return result

    def T_gripper_base_tcp(self, side: str) -> np.ndarray:
        result = self.T_gripper_base_semantic_contact()
        result[:3, :3] = self.semantic_to_tcp(side)
        return result


def resolve_robot_profile(name: str) -> RobotProfile:
    """Return pinned v1 profile facts for the two experiment embodiments."""

    if name == "galbot_one_golf":
        max_aperture = 0.12490876627340242
        return RobotProfile(
            robot_id=name,
            aperture_limits_m=(0.0, max_aperture),
            sweep_volume=grasp_refinement.ParallelJawSweepVolume(
                x_bounds_m=(-max_aperture / 2.0, max_aperture / 2.0),
                pad_y_bounds_m=(-0.01015, 0.01015),
                pad_z_bounds_m=(0.099338875, 0.158861285),
                samples_y=9,
                samples_z=15,
            ),
            gripper_base_to_contact_z_m=0.13996,
            semantic_to_tcp_rotation={
                "left": _GALBOT_SEMANTIC_TO_TCP.copy(),
                "right": _GALBOT_SEMANTIC_TO_TCP.copy(),
            },
            profile_facts={
                "profile_version": "parallel-jaw-graspgenx/v1",
                "gripper_type": "revolute_2f",
                "open_sweep_extents_m": [max_aperture, 0.0203, 0.05952241],
                "open_sweep_offset_m": [0.0, 0.0, 0.12910008],
                "mid_sweep_extents_m": [0.07423274, 0.0203, 0.0595224],
                "mid_sweep_offset_m": [0.0, 0.0, 0.15629605],
                "fingertip_depth_m": 0.13996,
                "exact_sources": {
                    "maximum_aperture": "derived Galbot bundle manifest",
                    "fingertip_depth": "gripper-base-to-TCP fixed URDF joint",
                },
                "provisional_assumptions": [
                    "sweep boxes conservatively enclose inner-pad motion",
                ],
            },
        )
    if name == "yam_bimanual":
        minimum = 0.002003575087686701
        maximum = 0.09490105891248288
        return RobotProfile(
            robot_id=name,
            aperture_limits_m=(minimum, maximum),
            sweep_volume=grasp_refinement.ParallelJawSweepVolume(
                x_bounds_m=(-maximum / 2.0, maximum / 2.0),
                pad_y_bounds_m=(-0.034, 0.034),
                pad_z_bounds_m=(0.07456, 0.14256),
                samples_y=9,
                samples_z=15,
            ),
            gripper_base_to_contact_z_m=0.14256,
            semantic_to_tcp_rotation={
                "left": _YAM_LEFT_SEMANTIC_TO_TCP.copy(),
                "right": np.eye(3, dtype=np.float64),
            },
            profile_facts={
                "profile_version": "parallel-jaw-graspgenx/v1",
                "gripper_type": "parallel_2f",
                "open_sweep_extents_m": [maximum, 0.068, 0.068],
                "open_sweep_offset_m": [0.0, 0.0, 0.10856],
                "mid_sweep_extents_m": [0.04742217, 0.068, 0.068],
                "mid_sweep_offset_m": [0.0, 0.0, 0.10856],
                "fingertip_depth_m": 0.14256,
                "exact_sources": {
                    "aperture_limits": "measured YAM bundle manifest",
                    "fingertip_depth": "link-6-to-TCP fixed URDF joint",
                },
                "provisional_assumptions": [
                    "sweep boxes conservatively enclose inner-pad motion",
                ],
            },
        )
    raise RefinementRunError(
        f"unknown robot profile {name!r}; expected galbot_one_golf or yam_bimanual"
    )


def _with_profile_overrides(
    profile: RobotProfile,
    *,
    aperture_limits_m: Sequence[float] | None = None,
    sweep_x_bounds_m: Sequence[float] | None = None,
    pad_y_bounds_m: Sequence[float] | None = None,
    pad_z_bounds_m: Sequence[float] | None = None,
    gripper_base_to_contact_z_m: float | None = None,
    sweep_samples_y: int | None = None,
    sweep_samples_z: int | None = None,
) -> tuple[RobotProfile, dict[str, Any]]:
    overrides: dict[str, Any] = {}

    def bounds(
        value: Sequence[float] | None,
        fallback: tuple[float, float],
        name: str,
    ) -> tuple[float, float]:
        if value is None:
            return fallback
        array = np.asarray(value, dtype=np.float64)
        if (
            array.shape != (2,)
            or not np.isfinite(array).all()
            or array[1] <= array[0]
        ):
            raise RefinementRunError(f"{name} must be two increasing finite numbers")
        result = (float(array[0]), float(array[1]))
        overrides[name] = list(result)
        return result

    aperture = bounds(
        aperture_limits_m,
        profile.aperture_limits_m,
        "aperture_limits_m",
    )
    if aperture[0] < 0.0:
        raise RefinementRunError("aperture_limits_m minimum must be nonnegative")
    sweep = grasp_refinement.ParallelJawSweepVolume(
        x_bounds_m=bounds(
            sweep_x_bounds_m,
            profile.sweep_volume.x_bounds_m,
            "sweep_x_bounds_m",
        ),
        pad_y_bounds_m=bounds(
            pad_y_bounds_m,
            profile.sweep_volume.pad_y_bounds_m,
            "pad_y_bounds_m",
        ),
        pad_z_bounds_m=bounds(
            pad_z_bounds_m,
            profile.sweep_volume.pad_z_bounds_m,
            "pad_z_bounds_m",
        ),
        samples_y=(
            profile.sweep_volume.samples_y
            if sweep_samples_y is None
            else int(sweep_samples_y)
        ),
        samples_z=(
            profile.sweep_volume.samples_z
            if sweep_samples_z is None
            else int(sweep_samples_z)
        ),
    )
    sweep.validate()
    depth = profile.gripper_base_to_contact_z_m
    if gripper_base_to_contact_z_m is not None:
        depth = float(gripper_base_to_contact_z_m)
        if not np.isfinite(depth) or depth <= 0.0:
            raise RefinementRunError(
                "gripper_base_to_contact_z_m must be positive and finite"
            )
        overrides["gripper_base_to_contact_z_m"] = depth
    return (
        replace(
            profile,
            aperture_limits_m=aperture,
            sweep_volume=sweep,
            gripper_base_to_contact_z_m=depth,
        ),
        overrides,
    )


def _scalar_text(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise RefinementRunError(f"{name} must be one scalar string")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _directory_record(directory: str | Path) -> dict[str, Any]:
    resolved = Path(directory).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    files = tuple(sorted(resolved.glob("*.json")))
    if not files:
        raise RefinementRunError(f"{resolved} contains no pose JSON files")
    records = [_file_record(path) for path in files]
    digest = hashlib.sha256()
    for record in records:
        digest.update(Path(record["path"]).name.encode("utf-8"))
        digest.update(record["sha256"].encode("ascii"))
    return {
        "path": str(resolved),
        "file_count": len(records),
        "manifest_sha256": digest.hexdigest(),
        "files": records,
    }


def _assert_record_unchanged(record: Mapping[str, Any]) -> None:
    if "files" in record:
        current = _directory_record(str(record["path"]))
        if (
            current["file_count"] != record["file_count"]
            or current["manifest_sha256"] != record["manifest_sha256"]
        ):
            raise RefinementRunError(
                f"input directory changed while running: {record['path']}"
            )
        return
    current = _file_record(str(record["path"]))
    if (
        current["size_bytes"] != record["size_bytes"]
        or current["sha256"] != record["sha256"]
    ):
        raise RefinementRunError(f"input changed while running: {record['path']}")


def _load_tracking(
    path: str | Path,
    *,
    expected_frames: int,
) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise RefinementRunError(f"cannot load V2D tracking {source}: {exc}") from exc
    try:
        validate_tracking_arrays(arrays, expected_frames=expected_frames)
    except ContractError as exc:
        raise RefinementRunError(str(exc)) from exc
    if _scalar_text(arrays["tracker"], name="tracking.tracker") != "v2d":
        raise RefinementRunError("tracking tracker must be exactly 'v2d'")
    return arrays


def _load_candidates(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"object_to_gripper_base", "confidence"}:
                raise RefinementRunError(
                    "candidate NPZ must contain exactly object_to_gripper_base "
                    "and confidence"
                )
            transforms = np.asarray(archive["object_to_gripper_base"])
            confidence = np.asarray(archive["confidence"])
    except RefinementRunError:
        raise
    except (OSError, ValueError) as exc:
        raise RefinementRunError(f"cannot load candidates {source}: {exc}") from exc
    try:
        transforms, confidence = validate_candidates(transforms, confidence)
    except ValueError as exc:
        raise RefinementRunError(str(exc)) from exc
    return transforms.astype(np.float64), confidence.astype(np.float64)


def _load_mesh(path: str | Path) -> Any:
    try:
        import trimesh
    except ImportError as exc:
        raise RefinementRunError(
            "trimesh is required to load and project the metric object mesh"
        ) from exc
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        loaded = trimesh.load(str(source), force="scene", process=False)
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    except Exception as exc:
        raise RefinementRunError(f"cannot load mesh {source}: {exc}") from exc
    if not isinstance(mesh, trimesh.Trimesh):
        raise RefinementRunError(f"{source} does not contain a triangular mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(vertices) < 3
        or len(faces) == 0
        or not np.isfinite(vertices).all()
    ):
        raise RefinementRunError(
            "metric mesh must be non-empty, finite, and triangular"
        )
    return mesh


def _project_to_mesh(
    mesh: Any,
    points_object: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import trimesh

        projected, distances, triangles = trimesh.proximity.closest_point(
            mesh,
            points_object,
        )
    except Exception as exc:
        raise RefinementRunError(
            "nearest mesh projection failed; ensure trimesh rtree is installed"
        ) from exc
    projected = np.asarray(projected, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if (
        projected.shape != (2, 3)
        or distances.shape != (2,)
        or triangles.shape != (2,)
        or not np.isfinite(projected).all()
        or not np.isfinite(distances).all()
    ):
        raise RefinementRunError("nearest mesh projection returned invalid contacts")
    return projected, distances, triangles


@dataclass(frozen=True)
class SurfaceRegisteredContactPair:
    """A raw contact pair translated together to the object's surface.

    Projecting the two fingertips independently can send both observations to
    one mesh vertex when the reconstructed hand and object are globally
    offset.  Registering only their midpoint yields one common translation,
    preserving the observed aperture and thumb-to-index direction exactly.
    """

    scoring_pair_object: np.ndarray
    raw_midpoint_object: np.ndarray
    projected_midpoint_object: np.ndarray
    midpoint_projection_distance_m: float
    common_translation_object: np.ndarray
    projected_triangle_index: int


def _surface_register_contact_pair(
    mesh: Any,
    raw_pair_object: np.ndarray,
) -> SurfaceRegisteredContactPair:
    raw_pair = np.asarray(raw_pair_object, dtype=np.float64)
    if raw_pair.shape != (2, 3) or not np.isfinite(raw_pair).all():
        raise RefinementRunError(
            "raw thumb/index contact pair must have finite shape (2,3)"
        )
    raw_midpoint = np.mean(raw_pair, axis=0)
    try:
        import trimesh

        projected, distances, triangles = trimesh.proximity.closest_point(
            mesh,
            raw_midpoint[None],
        )
    except Exception as exc:
        raise RefinementRunError(
            "nearest mesh midpoint projection failed; ensure trimesh rtree is installed"
        ) from exc
    projected = np.asarray(projected, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if (
        projected.shape != (1, 3)
        or distances.shape != (1,)
        or triangles.shape != (1,)
        or not np.isfinite(projected).all()
        or not np.isfinite(distances).all()
    ):
        raise RefinementRunError(
            "nearest mesh midpoint projection returned invalid values"
        )
    common_translation = projected[0] - raw_midpoint
    scoring_pair = raw_pair + common_translation
    if not np.allclose(
        scoring_pair[1] - scoring_pair[0],
        raw_pair[1] - raw_pair[0],
        atol=1e-12,
        rtol=0.0,
    ):
        raise RefinementRunError(
            "common midpoint registration failed to preserve the contact pair"
        )
    return SurfaceRegisteredContactPair(
        scoring_pair_object=scoring_pair,
        raw_midpoint_object=raw_midpoint,
        projected_midpoint_object=projected[0],
        midpoint_projection_distance_m=float(distances[0]),
        common_translation_object=common_translation,
        projected_triangle_index=int(triangles[0]),
    )


def _human_contacts(
    tracking: Mapping[str, np.ndarray],
    *,
    joints_world: np.ndarray,
    side: str,
    anchor_frame: int,
    window_radius: int,
    T_world_object_anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    frame_count = joints_world.shape[0]
    lo = max(0, anchor_frame - window_radius)
    hi = min(frame_count, anchor_frame + window_radius + 1)
    valid = np.asarray(tracking[f"{side}_valid"], dtype=np.bool_)
    finite = np.isfinite(
        joints_world[:, [THUMB_TIP, INDEX_TIP], :]
    ).all(axis=(1, 2))
    frames = tuple(
        int(frame)
        for frame in range(lo, hi)
        if valid[frame] and finite[frame]
    )
    if not frames:
        raise RefinementRunError(
            f"no valid {side} thumb/index observations in anchor window [{lo},{hi})"
        )
    raw_world = np.median(
        joints_world[np.asarray(frames)][:, [THUMB_TIP, INDEX_TIP], :],
        axis=0,
    )
    object_from_world = np.linalg.inv(T_world_object_anchor)
    raw_object = (
        raw_world @ object_from_world[:3, :3].T
        + object_from_world[:3, 3]
    )
    return raw_world, raw_object, frames


def _pose_as_lists(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).tolist()


def _score_as_dict(score: grasp_refinement.CandidateScore) -> dict[str, Any]:
    return {
        "candidate_index": score.candidate_index,
        "feasible": score.feasible,
        "total_cost": (
            score.total_cost if np.isfinite(score.total_cost) else None
        ),
        "contact_distance_m": score.contact_distance_m,
        "registered_contact_residual_m": score.registered_contact_residual_m,
        "registered_translation_m": (
            None
            if score.registered_translation_m is None
            else score.registered_translation_m.tolist()
        ),
        "registration_weight": score.registration_weight,
        "confidence_cost": score.confidence_cost,
        "pose_translation_m": score.pose_translation_m,
        "pose_rotation_rad": score.pose_rotation_rad,
        "approach_angle_rad": score.approach_angle_rad,
        "symmetry_flipped": score.symmetry_flipped,
        "reason": score.reason,
    }


def _write_npz_temp(directory: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=".grasp-refinement-",
        suffix=".partial.npz",
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o644)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _write_json_temp(directory: Path, payload: Mapping[str, Any]) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=".grasp-refinement-",
        suffix=".partial.json",
    )
    path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o644)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _commit_pair(
    npz_temp: Path,
    json_temp: Path,
    *,
    npz_output: Path,
    json_output: Path,
    overwrite: bool,
) -> None:
    targets = (npz_output, json_output)
    if npz_output == json_output:
        raise RefinementRunError("target NPZ and metadata JSON paths must differ")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        names = [str(path) for path in existing]
        raise ArtifactExistsError(
            f"refusing to overwrite existing outputs: {names}"
        )
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for target in existing:
            descriptor, name = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.name}.backup-",
            )
            os.close(descriptor)
            backup = Path(name)
            backup.unlink()
            os.replace(target, backup)
            backups[target] = backup
        os.replace(npz_temp, npz_output)
        installed.append(npz_output)
        os.replace(json_temp, json_output)
        installed.append(json_output)
    except Exception:
        for target in reversed(installed):
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def execute(
    *,
    base_target: str | Path,
    tracking: str | Path,
    T_camera_world: str | Path,
    foundationpose_poses: str | Path,
    mesh: str | Path,
    candidates: str | Path,
    robot_profile: str,
    side: str,
    object_name: str,
    event_start: int,
    event_end: int,
    event_anchor: int,
    starts_in_contact: bool,
    output_target: str | Path,
    output_metadata: str | Path | None = None,
    overwrite: bool = False,
    anchor_window_radius: int = DEFAULT_ANCHOR_WINDOW_RADIUS,
    approach_blend_frames: int = 8,
    release_blend_frames: int = 8,
    propagation_mode: grasp_refinement.GraspPropagationMode | str = (
        grasp_refinement.GraspPropagationMode.OBJECT_LOCK
    ),
    min_antipodal_score: float = 0.65,
    score_registration_weight: float = 1.0,
    score_contact_weight: float = 1.0,
    score_confidence_weight: float = 0.005,
    score_pose_translation_weight: float = 0.10,
    score_pose_rotation_weight: float = 0.005,
    score_approach_weight: float = 0.003,
    mesh_registration_residual_m: float | None = None,
    max_contact_registration_m: float | None = None,
    aperture_limits_m: Sequence[float] | None = None,
    sweep_x_bounds_m: Sequence[float] | None = None,
    pad_y_bounds_m: Sequence[float] | None = None,
    pad_z_bounds_m: Sequence[float] | None = None,
    gripper_base_to_contact_z_m: float | None = None,
    sweep_samples_y: int | None = None,
    sweep_samples_z: int | None = None,
) -> dict[str, Any]:
    """Run deterministic candidate selection and atomically commit both outputs."""

    if side not in {"left", "right"}:
        raise RefinementRunError("side must be exactly 'left' or 'right'")
    try:
        propagation_mode = grasp_refinement.GraspPropagationMode(propagation_mode)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(
            mode.value for mode in grasp_refinement.GraspPropagationMode
        )
        raise RefinementRunError(
            f"propagation_mode must be one of {choices}"
        ) from exc
    object_name = str(object_name).strip()
    if not object_name:
        raise RefinementRunError("object_name must not be empty")
    for value, name in (
        (event_start, "event_start"),
        (event_end, "event_end"),
        (event_anchor, "event_anchor"),
        (anchor_window_radius, "anchor_window_radius"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RefinementRunError(f"{name} must be an integer")
    if anchor_window_radius < 0:
        raise RefinementRunError("anchor_window_radius must be nonnegative")
    if starts_in_contact != (event_start == 0):
        raise RefinementRunError(
            "starts_in_contact must be true exactly when event_start is frame zero"
        )

    output_npz = Path(output_target).expanduser().resolve()
    output_json = (
        Path(output_metadata).expanduser().resolve()
        if output_metadata is not None
        else output_npz.with_suffix(".json")
    )
    if output_npz.suffix.lower() != ".npz":
        raise RefinementRunError("output_target must have a .npz suffix")
    if output_json.suffix.lower() != ".json":
        raise RefinementRunError("output_metadata must have a .json suffix")
    if not overwrite:
        existing = [path for path in (output_npz, output_json) if path.exists()]
        if existing:
            names = [str(path) for path in existing]
            raise ArtifactExistsError(
                f"refusing to overwrite existing outputs: {names}"
            )

    input_records: dict[str, Any] = {
        "base_target": _file_record(base_target),
        "tracking": _file_record(tracking),
        "T_camera_world": _file_record(T_camera_world),
        "foundationpose_poses": _directory_record(foundationpose_poses),
        "mesh": _file_record(mesh),
        "candidates": _file_record(candidates),
    }
    candidate_metadata_path = (
        Path(candidates).expanduser().resolve().with_suffix(".json")
    )
    if candidate_metadata_path.is_file():
        input_records["candidate_metadata"] = _file_record(candidate_metadata_path)

    base_arrays = grasp_refinement.load_parallel_jaw_target(base_target)
    frame_count = int(np.asarray(base_arrays["frame_indices"]).size)
    if _scalar_text(base_arrays["tracker"], name="base_target.tracker") != "v2d":
        raise RefinementRunError("base target tracker must be exactly 'v2d'")
    tracking_arrays = _load_tracking(tracking, expected_frames=frame_count)
    camera_from_world, _ = parallel_jaw_from_tracking.load_world_to_camera(
        T_camera_world,
        frame_count=frame_count,
    )
    world_from_object = grasp_refinement.load_foundationpose_wxyz_batch(
        foundationpose_poses,
        camera_from_world,
        expected_frames=frame_count,
    )
    if not (
        0 <= event_start <= event_anchor <= event_end < frame_count
    ):
        raise RefinementRunError(
            f"event must satisfy 0 <= start <= anchor <= end < {frame_count}"
        )

    joints_world_by_side = parallel_jaw_from_tracking.joints_in_world(
        tracking_arrays,
        world_to_camera=camera_from_world,
    )
    raw_world, raw_object, contact_frames = _human_contacts(
        tracking_arrays,
        joints_world=joints_world_by_side[side],
        side=side,
        anchor_frame=event_anchor,
        window_radius=anchor_window_radius,
        T_world_object_anchor=world_from_object[event_anchor],
    )
    metric_mesh = _load_mesh(mesh)
    projected_object, hand_mesh_distances, projected_triangles = _project_to_mesh(
        metric_mesh,
        raw_object,
    )
    registered_contacts = _surface_register_contact_pair(
        metric_mesh,
        raw_object,
    )
    candidate_poses, confidence = _load_candidates(candidates)

    profile, profile_overrides = _with_profile_overrides(
        resolve_robot_profile(robot_profile),
        aperture_limits_m=aperture_limits_m,
        sweep_x_bounds_m=sweep_x_bounds_m,
        pad_y_bounds_m=pad_y_bounds_m,
        pad_z_bounds_m=pad_z_bounds_m,
        gripper_base_to_contact_z_m=gripper_base_to_contact_z_m,
        sweep_samples_y=sweep_samples_y,
        sweep_samples_z=sweep_samples_z,
    )
    derived_contacts = grasp_refinement.derive_parallel_jaw_candidate_contacts(
        metric_mesh,
        candidate_poses,
        sweep_volume=profile.sweep_volume,
    )

    base_semantic_world = grasp_refinement.pose_matrix(
        np.asarray(base_arrays[f"{side}_position"])[event_anchor],
        np.asarray(base_arrays[f"{side}_wxyz"])[event_anchor],
    )
    semantic_from_tcp = np.eye(4, dtype=np.float64)
    semantic_from_tcp[:3, :3] = profile.semantic_to_tcp(side)
    gripper_base_from_tcp = np.linalg.inv(profile.T_gripper_base_tcp(side))
    human_gripper_base_object = (
        np.linalg.inv(world_from_object[event_anchor])
        @ base_semantic_world
        @ semantic_from_tcp
        @ gripper_base_from_tcp
    )
    scores = grasp_refinement.score_grasp_candidates(
        derived_contacts,
        registered_contacts.scoring_pair_object,
        confidences=confidence,
        human_pose_object=human_gripper_base_object,
        aperture_limits_m=profile.aperture_limits_m,
        min_antipodal_score=min_antipodal_score,
        registration_weight=score_registration_weight,
        mesh_registration_residual_m=mesh_registration_residual_m,
        weights=grasp_refinement.GraspScoreWeights(
            contact=score_contact_weight,
            confidence=score_confidence_weight,
            pose_translation=score_pose_translation_weight,
            pose_rotation=score_pose_rotation_weight,
            approach=score_approach_weight,
        ),
    )
    selected_score = grasp_refinement.select_best_candidate(scores)
    selected = next(
        contact
        for contact in derived_contacts
        if contact.candidate_index == selected_score.candidate_index
    )
    if selected.contact_points_object is None or selected.aperture_m is None:
        raise RefinementRunError("selected candidate is missing contact geometry")

    (
        raw_registration_residual,
        registration_translation,
        registration_swapped,
    ) = grasp_refinement.translation_registered_contact_residual(
        selected.contact_points_object,
        raw_object,
    )
    registration_magnitude = float(np.linalg.norm(registration_translation))
    if (
        max_contact_registration_m is not None
        and (
            not np.isfinite(max_contact_registration_m)
            or max_contact_registration_m < 0.0
        )
    ):
        raise RefinementRunError(
            "max_contact_registration_m must be finite and nonnegative"
        )
    if (
        max_contact_registration_m is not None
        and registration_magnitude > max_contact_registration_m
    ):
        raise RefinementRunError(
            f"selected contact registration {registration_magnitude:.6g} m exceeds "
            f"limit {max_contact_registration_m:.6g} m"
        )
    selected_gripper_base_aligned = selected.T_object_gripper.copy()
    if selected_score.symmetry_flipped:
        selected_gripper_base_aligned[:3, :3] = (
            selected_gripper_base_aligned[:3, :3]
            @ np.diag([-1.0, -1.0, 1.0])
        )
    registered_gripper_base = selected_gripper_base_aligned.copy()
    registered_gripper_base[:3, 3] += registration_translation

    # GraspGenX and the robot-neutral semantic target share +X closing/+Z
    # approach.  Move from its base to the contact origin; the existing bundle
    # applies semantic_target_to_tcp_rotation later during embodiment IK.
    object_from_semantic = (
        registered_gripper_base
        @ profile.T_gripper_base_semantic_contact()
    )
    object_from_tcp = object_from_semantic.copy()
    object_from_tcp[:3, :3] = (
        object_from_semantic[:3, :3] @ profile.semantic_to_tcp(side)
    )

    event = grasp_refinement.GraspEvent(
        start_frame=int(event_start),
        end_frame=int(event_end),
        anchor_frame=int(event_anchor),
        starts_in_contact=bool(starts_in_contact),
    )
    refined = grasp_refinement.apply_phase_aware_corrections(
        base_arrays,
        side=side,
        T_world_object=world_from_object,
        corrections=[
            grasp_refinement.GraspCorrection(
                event=event,
                T_object_gripper=object_from_semantic,
                aperture_m=float(selected.aperture_m),
                propagation_mode=propagation_mode,
            )
        ],
        approach_blend_frames=approach_blend_frames,
        release_blend_frames=release_blend_frames,
    )
    grasp_refinement.validate_parallel_jaw_target(
        refined,
        expected_frames=frame_count,
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    npz_temp: Path | None = None
    json_temp: Path | None = None
    try:
        npz_temp = _write_npz_temp(output_npz.parent, refined)
        output_npz_record = {
            "path": str(output_npz),
            "size_bytes": npz_temp.stat().st_size,
            "sha256": _sha256(npz_temp),
            "keys": sorted(refined),
            "dtypes": {
                key: str(np.asarray(value).dtype) for key, value in refined.items()
            },
        }
        selected_index = selected.candidate_index
        metadata: dict[str, Any] = {
            "schema_version": RUN_SCHEMA,
            "state": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "runner": {
                "version": RUNNER_VERSION,
                "source": _file_record(Path(__file__)),
            },
            "interaction": {
                "object_name": object_name,
                "side": side,
                "event": {
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "anchor_frame": event.anchor_frame,
                    "starts_in_contact": event.starts_in_contact,
                },
                "anchor_window_radius": anchor_window_radius,
                "contact_source_frames": list(contact_frames),
            },
            "inputs": input_records,
            "profile": {
                "robot_id": profile.robot_id,
                "aperture_limits_m": list(profile.aperture_limits_m),
                "sweep_volume": {
                    "x_bounds_m": list(profile.sweep_volume.x_bounds_m),
                    "pad_y_bounds_m": list(profile.sweep_volume.pad_y_bounds_m),
                    "pad_z_bounds_m": list(profile.sweep_volume.pad_z_bounds_m),
                    "samples_y": profile.sweep_volume.samples_y,
                    "samples_z": profile.sweep_volume.samples_z,
                },
                "T_gripper_base_semantic_contact": _pose_as_lists(
                    profile.T_gripper_base_semantic_contact()
                ),
                "T_gripper_base_tcp": _pose_as_lists(
                    profile.T_gripper_base_tcp(side)
                ),
                "semantic_target_to_tcp_rotation": profile.semantic_to_tcp(
                    side
                ).tolist(),
                "renderer_policy": (
                    "output remains +X-closing/+Z-approach robot-neutral semantic; "
                    "bundle semantic_target_to_tcp_rotation is applied downstream"
                ),
                "facts": dict(profile.profile_facts),
                "overrides": profile_overrides,
            },
            "human_contacts": {
                "joint_indices": {"thumb_tip": THUMB_TIP, "index_tip": INDEX_TIP},
                "aggregation": (
                    "coordinatewise median in world over valid anchor ± radius"
                ),
                "raw_world_m": raw_world.tolist(),
                "raw_object_m": raw_object.tolist(),
                "scoring_contact_estimation": {
                    "strategy": CONTACT_PAIR_REGISTRATION_STRATEGY,
                    "raw_aperture_m": float(
                        np.linalg.norm(raw_object[1] - raw_object[0])
                    ),
                    "raw_midpoint_object_m": (
                        registered_contacts.raw_midpoint_object.tolist()
                    ),
                    "projected_midpoint_object_m": (
                        registered_contacts.projected_midpoint_object.tolist()
                    ),
                    "midpoint_projection_distance_m": (
                        registered_contacts.midpoint_projection_distance_m
                    ),
                    "common_translation_object_m": (
                        registered_contacts.common_translation_object.tolist()
                    ),
                    "projected_triangle_index": (
                        registered_contacts.projected_triangle_index
                    ),
                    "scoring_pair_object_m": (
                        registered_contacts.scoring_pair_object.tolist()
                    ),
                    "scoring_aperture_m": float(
                        np.linalg.norm(
                            registered_contacts.scoring_pair_object[1]
                            - registered_contacts.scoring_pair_object[0]
                        )
                    ),
                    "preserves_raw_pair_vector": True,
                },
                "independent_nearest_point_diagnostics": {
                    "projected_object_m": projected_object.tolist(),
                    "raw_hand_to_mesh_distance_m": (
                        hand_mesh_distances.tolist()
                    ),
                    "projected_triangle_indices": (
                        projected_triangles.tolist()
                    ),
                    "projected_aperture_m": float(
                        np.linalg.norm(
                            projected_object[1] - projected_object[0]
                        )
                    ),
                    "used_for_scoring": False,
                },
                "projected_object_m": projected_object.tolist(),
                "raw_hand_to_mesh_distance_m": hand_mesh_distances.tolist(),
                "projected_triangle_indices": projected_triangles.tolist(),
            },
            "selection": {
                "candidate_count": len(derived_contacts),
                "derived_contact_valid_count": sum(
                    int(item.valid) for item in derived_contacts
                ),
                "hard_filters": {
                    "aperture_limits_m": list(profile.aperture_limits_m),
                    "minimum_antipodal_score": min_antipodal_score,
                },
                "weights": {
                    "contact_primary": score_contact_weight,
                    "confidence": score_confidence_weight,
                    "pose_translation_secondary": (
                        score_pose_translation_weight
                    ),
                    "pose_rotation_secondary": score_pose_rotation_weight,
                    "approach_secondary": score_approach_weight,
                    "score_registration_weight": score_registration_weight,
                    "mesh_registration_residual_m": mesh_registration_residual_m,
                },
                "scores": [_score_as_dict(score) for score in scores],
                "selected_candidate_index": selected_index,
                "selected_confidence": float(confidence[selected_index]),
                "selected_aperture_m": float(selected.aperture_m),
                "selected_antipodal_score": float(selected.antipodal_score),
                "selected_contacts_object_m": (
                    selected.contact_points_object.tolist()
                ),
                "T_object_gripper_base_before_registration": _pose_as_lists(
                    selected.T_object_gripper
                ),
                "parallel_jaw_symmetry_aligned_to_human_pose": (
                    selected_score.symmetry_flipped
                ),
                "T_object_gripper_base_symmetry_aligned": _pose_as_lists(
                    selected_gripper_base_aligned
                ),
            },
            "contact_registration": {
                "target": "raw_unprojected_human_thumb_index_contacts_object",
                "translation_object_m": registration_translation.tolist(),
                "translation_magnitude_m": registration_magnitude,
                "residual_m": raw_registration_residual,
                "jaw_assignment_swapped": registration_swapped,
                "maximum_allowed_m": max_contact_registration_m,
                "T_object_gripper_base_registered": _pose_as_lists(
                    registered_gripper_base
                ),
            },
            "frame_conversion": {
                "T_object_semantic": _pose_as_lists(object_from_semantic),
                "T_object_tcp_for_audit_only": _pose_as_lists(object_from_tcp),
            },
            "trajectory_correction": {
                "propagation_mode": propagation_mode.value,
                "algorithm": (
                    "anchor-derived constant base-local right offset throughout "
                    "approach, inclusive hold, and release; C2/Slerp blending"
                    if propagation_mode
                    is grasp_refinement.GraspPropagationMode.BASE_LOCAL_OFFSET
                    else (
                        "constant local offset during approach; "
                        "object-relative hold; constant local offset during "
                        "release; C2/Slerp blending"
                    )
                ),
                "foundationpose_dependency": (
                    "anchor_frame_only"
                    if propagation_mode
                    is grasp_refinement.GraspPropagationMode.BASE_LOCAL_OFFSET
                    else "every_hold_frame"
                ),
                "approach_blend_frames": approach_blend_frames,
                "release_blend_frames": release_blend_frames,
                "other_side_preserved": True,
                "input_can_be_prior_refinement_output": True,
            },
            "output": {
                "target": output_npz_record,
                "metadata": {"path": str(output_json)},
                "exact_target_key_count": len(refined),
                "tracker": _scalar_text(refined["tracker"], name="output.tracker"),
            },
        }
        json_temp = _write_json_temp(output_json.parent, metadata)
        for record in input_records.values():
            _assert_record_unchanged(record)
        _commit_pair(
            npz_temp,
            json_temp,
            npz_output=output_npz,
            json_output=output_json,
            overwrite=overwrite,
        )
        npz_temp = None
        json_temp = None
    finally:
        if npz_temp is not None:
            npz_temp.unlink(missing_ok=True)
        if json_temp is not None:
            json_temp.unlink(missing_ok=True)
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-target", required=True, type=Path)
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument(
        "--t-camera-world",
        "--T-camera-world",
        dest="T_camera_world",
        required=True,
        type=Path,
    )
    parser.add_argument("--foundationpose-poses", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument(
        "--robot-profile",
        required=True,
        choices=("galbot_one_golf", "yam_bimanual"),
    )
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--event-start", required=True, type=int)
    parser.add_argument("--event-end", required=True, type=int)
    parser.add_argument("--event-anchor", required=True, type=int)
    parser.add_argument(
        "--starts-in-contact",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    parser.add_argument("--output-target", required=True, type=Path)
    parser.add_argument("--output-metadata", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--anchor-window-radius",
        type=int,
        default=DEFAULT_ANCHOR_WINDOW_RADIUS,
    )
    parser.add_argument("--approach-blend-frames", type=int, default=8)
    parser.add_argument("--release-blend-frames", type=int, default=8)
    parser.add_argument(
        "--propagation-mode",
        choices=tuple(mode.value for mode in grasp_refinement.GraspPropagationMode),
        default=grasp_refinement.GraspPropagationMode.OBJECT_LOCK.value,
    )
    parser.add_argument("--min-antipodal-score", type=float, default=0.65)
    parser.add_argument("--score-registration-weight", type=float, default=1.0)
    parser.add_argument("--score-contact-weight", type=float, default=1.0)
    parser.add_argument("--score-confidence-weight", type=float, default=0.005)
    parser.add_argument(
        "--score-pose-translation-weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--score-pose-rotation-weight",
        type=float,
        default=0.005,
    )
    parser.add_argument("--score-approach-weight", type=float, default=0.003)
    parser.add_argument("--mesh-registration-residual-m", type=float)
    parser.add_argument("--max-contact-registration-m", type=float)
    parser.add_argument("--aperture-limits-m", type=float, nargs=2)
    parser.add_argument("--sweep-x-bounds-m", type=float, nargs=2)
    parser.add_argument("--pad-y-bounds-m", type=float, nargs=2)
    parser.add_argument("--pad-z-bounds-m", type=float, nargs=2)
    parser.add_argument("--gripper-base-to-contact-z-m", type=float)
    parser.add_argument("--sweep-samples-y", type=int)
    parser.add_argument("--sweep-samples-z", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metadata = execute(
        base_target=args.base_target,
        tracking=args.tracking,
        T_camera_world=args.T_camera_world,
        foundationpose_poses=args.foundationpose_poses,
        mesh=args.mesh,
        candidates=args.candidates,
        robot_profile=args.robot_profile,
        side=args.side,
        object_name=args.object_name,
        event_start=args.event_start,
        event_end=args.event_end,
        event_anchor=args.event_anchor,
        starts_in_contact=args.starts_in_contact,
        output_target=args.output_target,
        output_metadata=args.output_metadata,
        overwrite=args.overwrite,
        anchor_window_radius=args.anchor_window_radius,
        approach_blend_frames=args.approach_blend_frames,
        release_blend_frames=args.release_blend_frames,
        propagation_mode=args.propagation_mode,
        min_antipodal_score=args.min_antipodal_score,
        score_registration_weight=args.score_registration_weight,
        score_contact_weight=args.score_contact_weight,
        score_confidence_weight=args.score_confidence_weight,
        score_pose_translation_weight=args.score_pose_translation_weight,
        score_pose_rotation_weight=args.score_pose_rotation_weight,
        score_approach_weight=args.score_approach_weight,
        mesh_registration_residual_m=args.mesh_registration_residual_m,
        max_contact_registration_m=args.max_contact_registration_m,
        aperture_limits_m=args.aperture_limits_m,
        sweep_x_bounds_m=args.sweep_x_bounds_m,
        pad_y_bounds_m=args.pad_y_bounds_m,
        pad_z_bounds_m=args.pad_z_bounds_m,
        gripper_base_to_contact_z_m=args.gripper_base_to_contact_z_m,
        sweep_samples_y=args.sweep_samples_y,
        sweep_samples_z=args.sweep_samples_z,
    )
    print(
        json.dumps(
            {
                "output_target": metadata["output"]["target"]["path"],
                "output_metadata": metadata["output"]["metadata"]["path"],
                "selected_candidate_index": metadata["selection"][
                    "selected_candidate_index"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
