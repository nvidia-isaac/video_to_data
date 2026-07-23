"""Geometry and trajectory utilities for object-conditioned grasp refinement.

The module deliberately stops at the robot-neutral parallel-jaw target seam.
It does not run GraspGen(X), solve robot IK, or write renderer artifacts.  Its
inputs and outputs are explicit NumPy arrays so the orchestration layer can
record provenance and reject ambiguous frame conventions.

Conventions
-----------

* All quaternions are WXYZ.
* ``T_a_b`` maps points expressed in frame ``b`` into frame ``a``.
* TACO ``T_camera_world`` is world-to-camera.
* FoundationPose JSON poses are ``T_camera_object``.
* Grasp candidates are ``T_object_gripper`` with +Z approach and +X closing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from inpainting.adapters.parallel_jaw_from_tracking import (
    PARALLEL_JAW_KEYS,
    validate_parallel_jaw_arrays,
)


_HOMOGENEOUS_ROW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_FOUNDATIONPOSE_KEYS = frozenset({"rotation", "translation", "scale"})
_RZ_PI = np.diag([-1.0, -1.0, 1.0])
_EPS = np.finfo(np.float64).eps


class GraspRefinementError(ValueError):
    """Raised when grasp refinement would require an implicit convention."""


def _finite_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GraspRefinementError(f"{name} must be numeric") from exc
    if shape is not None and result.shape != shape:
        raise GraspRefinementError(
            f"{name} must have shape {shape}, got {result.shape}"
        )
    if not np.isfinite(result).all():
        raise GraspRefinementError(f"{name} contains non-finite values")
    return result


def _validate_transform_batch(
    value: Any,
    *,
    name: str,
    expected_frames: int | None = None,
) -> np.ndarray:
    transforms = _finite_array(value, name=name)
    if transforms.shape == (4, 4):
        transforms = transforms[None]
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise GraspRefinementError(
            f"{name} must have shape (4,4) or (N,4,4), got {transforms.shape}"
        )
    if expected_frames is not None and transforms.shape[0] != expected_frames:
        raise GraspRefinementError(
            f"{name} contains {transforms.shape[0]} frames, "
            f"expected {expected_frames}"
        )
    if not np.allclose(
        transforms[:, 3, :],
        _HOMOGENEOUS_ROW,
        atol=1e-7,
        rtol=0.0,
    ):
        raise GraspRefinementError(f"{name} contains non-homogeneous transforms")
    rotations = transforms[:, :3, :3]
    orthogonality = rotations @ np.swapaxes(rotations, 1, 2)
    if not np.allclose(
        orthogonality,
        np.eye(3, dtype=np.float64),
        atol=2e-5,
        rtol=0.0,
    ):
        raise GraspRefinementError(f"{name} rotations are not orthonormal")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=2e-5, rtol=0.0):
        raise GraspRefinementError(f"{name} rotations are not proper")
    return transforms


def validate_parallel_jaw_target(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_frames: int | None = None,
) -> int:
    """Validate the exact existing 12-key parallel-jaw target contract."""

    copied = {key: np.asarray(value) for key, value in arrays.items()}
    try:
        return validate_parallel_jaw_arrays(
            copied,
            expected_frames=expected_frames,
        )
    except Exception as exc:
        raise GraspRefinementError(str(exc)) from exc


def load_parallel_jaw_target(
    path: str | Path,
    *,
    expected_frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Load an NPZ without pickle and validate the exact 12-key target seam."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise GraspRefinementError(
            f"cannot load parallel-jaw target {source}: {exc}"
        ) from exc
    validate_parallel_jaw_target(arrays, expected_frames=expected_frames)
    return arrays


def wxyz_to_matrix(quaternion: Any) -> np.ndarray:
    """Convert one or a batch of unit WXYZ quaternions to rotation matrices."""

    values = _finite_array(quaternion, name="quaternion")
    if values.ndim == 1:
        values = values[None]
        squeeze = True
    elif values.ndim >= 2:
        squeeze = False
    else:
        raise GraspRefinementError("quaternion must end in a four-vector")
    if values.shape[-1] != 4:
        raise GraspRefinementError(
            f"quaternion must have shape (...,4), got {values.shape}"
        )
    norms = np.linalg.norm(values, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        index = int(np.argmax(np.abs(norms.reshape(-1) - 1.0)))
        raise GraspRefinementError(
            f"quaternion {index} has norm {norms.reshape(-1)[index]:.8g}, expected 1"
        )
    xyzw = np.concatenate((values[..., 1:], values[..., :1]), axis=-1)
    matrices = Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix()
    matrices = matrices.reshape(values.shape[:-1] + (3, 3))
    return matrices[0] if squeeze else matrices


def matrix_to_wxyz(matrix: Any) -> np.ndarray:
    """Convert one or a batch of proper rotation matrices to unit WXYZ."""

    matrices = _finite_array(matrix, name="rotation matrix")
    if matrices.shape == (3, 3):
        matrices = matrices[None]
        squeeze = True
    else:
        squeeze = False
    if matrices.ndim < 3 or matrices.shape[-2:] != (3, 3):
        raise GraspRefinementError(
            f"rotation matrix must have shape (...,3,3), got {matrices.shape}"
        )
    flat = matrices.reshape(-1, 3, 3)
    if not np.allclose(
        flat @ np.swapaxes(flat, 1, 2),
        np.eye(3),
        atol=2e-5,
        rtol=0.0,
    ) or not np.allclose(np.linalg.det(flat), 1.0, atol=2e-5, rtol=0.0):
        raise GraspRefinementError("rotation matrix is not a proper rotation")
    xyzw = Rotation.from_matrix(flat).as_quat()
    wxyz = np.concatenate((xyzw[:, 3:], xyzw[:, :3]), axis=1)
    # A deterministic hemisphere makes serialized outputs reproducible.
    negative = wxyz[:, 0] < 0.0
    wxyz[negative] *= -1.0
    wxyz = wxyz.reshape(matrices.shape[:-2] + (4,))
    return wxyz[0] if squeeze else wxyz


def pose_matrix(position: Any, quaternion_wxyz: Any) -> np.ndarray:
    """Form one or a batch of homogeneous poses from position and WXYZ."""

    positions = _finite_array(position, name="position")
    quaternions = _finite_array(quaternion_wxyz, name="quaternion")
    single = positions.shape == (3,) and quaternions.shape == (4,)
    if single:
        positions = positions[None]
        quaternions = quaternions[None]
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or quaternions.shape != (positions.shape[0], 4)
    ):
        raise GraspRefinementError(
            "position and quaternion must have shapes (3,)/(4,) or (N,3)/(N,4)"
        )
    result = np.repeat(np.eye(4, dtype=np.float64)[None], positions.shape[0], axis=0)
    result[:, :3, :3] = wxyz_to_matrix(quaternions)
    result[:, :3, 3] = positions
    return result[0] if single else result


def load_foundationpose_wxyz_batch(
    pose_directory: str | Path,
    T_camera_world: Any,
    *,
    expected_frames: int | None = None,
) -> np.ndarray:
    """Load exact numbered FoundationPose JSON and return ``T_world_object``.

    ``T_camera_world`` is the TACO world-to-camera transform.  It may be one
    transform, which is broadcast, or one transform per pose frame.
    """

    directory = Path(pose_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = tuple(sorted(directory.glob("*.json")))
    if not paths:
        raise GraspRefinementError(f"no FoundationPose JSON files in {directory}")
    frame_count = len(paths)
    if expected_frames is not None and frame_count != expected_frames:
        raise GraspRefinementError(
            f"FoundationPose contains {frame_count} frames, expected {expected_frames}"
        )
    expected_paths = tuple(
        directory / f"{frame:06d}.json" for frame in range(frame_count)
    )
    if paths != expected_paths:
        raise GraspRefinementError(
            "FoundationPose batch must contain exactly contiguous 000000.json..N"
        )

    camera_from_object = np.repeat(
        np.eye(4, dtype=np.float64)[None],
        frame_count,
        axis=0,
    )
    for frame, path in enumerate(paths):
        try:
            value = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GraspRefinementError(
                f"cannot parse FoundationPose frame {path}: {exc}"
            ) from exc
        if not isinstance(value, dict) or set(value) != _FOUNDATIONPOSE_KEYS:
            raise GraspRefinementError(
                f"{path} must contain exactly rotation, translation, and scale"
            )
        rotation = _finite_array(
            value["rotation"],
            name=f"{path} rotation",
            shape=(4,),
        )
        translation = _finite_array(
            value["translation"],
            name=f"{path} translation",
            shape=(3,),
        )
        scale = _finite_array(
            value["scale"],
            name=f"{path} scale",
            shape=(3,),
        )
        if not np.allclose(scale, 1.0, atol=1e-6, rtol=0.0):
            raise GraspRefinementError(
                f"{path} scale must be [1,1,1] because the mesh is already metric"
            )
        camera_from_object[frame, :3, :3] = wxyz_to_matrix(rotation)
        camera_from_object[frame, :3, 3] = translation

    camera_from_world = _validate_transform_batch(
        T_camera_world,
        name="T_camera_world",
    )
    if camera_from_world.shape[0] == 1:
        camera_from_world = np.repeat(camera_from_world, frame_count, axis=0)
    elif camera_from_world.shape[0] != frame_count:
        raise GraspRefinementError(
            f"T_camera_world contains {camera_from_world.shape[0]} frames, "
            f"expected 1 or {frame_count}"
        )
    world_from_camera = np.linalg.inv(camera_from_world)
    world_from_object = world_from_camera @ camera_from_object
    return _validate_transform_batch(
        world_from_object,
        name="T_world_object",
        expected_frames=frame_count,
    )


@dataclass(frozen=True)
class GraspEvent:
    """Inclusive contact interval detected from a hysteretic aperture trace."""

    start_frame: int
    end_frame: int
    anchor_frame: int
    starts_in_contact: bool = False

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def segment_grasp_events(
    aperture_m: Any,
    *,
    close_threshold_m: float,
    open_threshold_m: float,
    min_duration_frames: int = 2,
    max_gap_frames: int = 0,
) -> tuple[GraspEvent, ...]:
    """Segment multiple contact intervals with Schmitt-trigger hysteresis.

    A grasp begins at or below ``close_threshold_m`` and remains active until
    the aperture reaches ``open_threshold_m``.  Short open gaps can be merged
    before minimum-duration filtering.  A trace closed at frame zero produces
    an explicit ``starts_in_contact`` event.
    """

    aperture = _finite_array(aperture_m, name="aperture_m")
    if aperture.ndim != 1 or aperture.size == 0:
        raise GraspRefinementError("aperture_m must be a non-empty vector")
    if np.any(aperture < 0.0):
        raise GraspRefinementError("aperture_m must be nonnegative")
    if (
        not np.isfinite(close_threshold_m)
        or not np.isfinite(open_threshold_m)
        or close_threshold_m < 0.0
        or open_threshold_m <= close_threshold_m
    ):
        raise GraspRefinementError(
            "thresholds must satisfy 0 <= close_threshold_m < open_threshold_m"
        )
    for value, name in (
        (min_duration_frames, "min_duration_frames"),
        (max_gap_frames, "max_gap_frames"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise GraspRefinementError(f"{name} must be an integer")
    if min_duration_frames <= 0 or max_gap_frames < 0:
        raise GraspRefinementError(
            "min_duration_frames must be positive and max_gap_frames nonnegative"
        )

    raw: list[tuple[int, int]] = []
    active = bool(aperture[0] <= close_threshold_m)
    start = 0 if active else -1
    for frame in range(1 if active else 0, aperture.size):
        if not active and aperture[frame] <= close_threshold_m:
            active = True
            start = frame
        elif active and aperture[frame] >= open_threshold_m:
            raw.append((start, frame - 1))
            active = False
            start = -1
    if active:
        raw.append((start, aperture.size - 1))

    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] - 1 <= max_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    events: list[GraspEvent] = []
    for start, end in merged:
        if end - start + 1 < min_duration_frames:
            continue
        local = aperture[start : end + 1]
        anchor = start + int(np.argmin(local))
        events.append(
            GraspEvent(
                start_frame=start,
                end_frame=end,
                anchor_frame=anchor,
                starts_in_contact=(start == 0 and aperture[0] <= close_threshold_m),
            )
        )
    return tuple(events)


@dataclass(frozen=True)
class ParallelJawSweepVolume:
    """Rectangular gripper-frame volume swept by the two inner finger pads."""

    x_bounds_m: tuple[float, float]
    pad_y_bounds_m: tuple[float, float]
    pad_z_bounds_m: tuple[float, float]
    samples_y: int = 9
    samples_z: int = 13

    def validate(self) -> None:
        for bounds, name in (
            (self.x_bounds_m, "x_bounds_m"),
            (self.pad_y_bounds_m, "pad_y_bounds_m"),
            (self.pad_z_bounds_m, "pad_z_bounds_m"),
        ):
            values = _finite_array(bounds, name=name, shape=(2,))
            if values[1] <= values[0]:
                raise GraspRefinementError(f"{name} must be strictly increasing")
        for value, name in (
            (self.samples_y, "samples_y"),
            (self.samples_z, "samples_z"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 2
            ):
                raise GraspRefinementError(f"{name} must be an integer >= 2")


@dataclass(frozen=True)
class CandidateContacts:
    """One candidate's derived opposing contacts in the object frame."""

    candidate_index: int
    T_object_gripper: np.ndarray
    valid: bool
    contact_points_object: np.ndarray | None
    contact_normals_object: np.ndarray | None
    aperture_m: float | None
    antipodal_score: float | None
    ray_yz_gripper: np.ndarray | None
    reason: str | None = None


def _ray_first_hits(
    mesh: Any,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    try:
        locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False,
        )
    except Exception as exc:
        raise GraspRefinementError(
            "trimesh ray intersection failed; install its rtree/embreex dependency"
        ) from exc
    points = {
        int(ray): np.asarray(point, dtype=np.float64)
        for point, ray in zip(locations, ray_indices, strict=True)
    }
    triangles = {
        int(ray): int(triangle)
        for triangle, ray in zip(triangle_indices, ray_indices, strict=True)
    }
    return points, triangles


def derive_parallel_jaw_candidate_contacts(
    mesh: Any,
    T_object_gripper_candidates: Any,
    *,
    sweep_volume: ParallelJawSweepVolume,
) -> tuple[CandidateContacts, ...]:
    """Ray-cast opposing +X/-X pad contacts for each candidate pose.

    Paired rays share one pad ``(y,z)`` sample.  This enforces that the object
    lies inside the finite finger-pad/sweep bounds instead of using unbounded
    mesh extrema.  The best ray is the most antipodal, with central pad samples
    used as a deterministic tie-breaker.
    """

    sweep_volume.validate()
    poses = _validate_transform_batch(
        T_object_gripper_candidates,
        name="T_object_gripper_candidates",
    )
    vertices = _finite_array(mesh.vertices, name="mesh.vertices")
    faces = np.asarray(mesh.faces)
    face_normals = _finite_array(mesh.face_normals, name="mesh.face_normals")
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or face_normals.shape != (faces.shape[0], 3)
        or vertices.shape[0] == 0
        or faces.shape[0] == 0
    ):
        raise GraspRefinementError("mesh must be a non-empty triangular mesh")

    x_lo, x_hi = sweep_volume.x_bounds_m
    ys = np.linspace(*sweep_volume.pad_y_bounds_m, sweep_volume.samples_y)
    zs = np.linspace(*sweep_volume.pad_z_bounds_m, sweep_volume.samples_z)
    grid_y, grid_z = np.meshgrid(ys, zs, indexing="ij")
    yz = np.column_stack((grid_y.reshape(-1), grid_z.reshape(-1)))
    ray_count = yz.shape[0]
    plus_origins_gripper = np.column_stack(
        (np.full(ray_count, x_hi), yz[:, 0], yz[:, 1])
    )
    minus_origins_gripper = np.column_stack(
        (np.full(ray_count, x_lo), yz[:, 0], yz[:, 1])
    )

    results: list[CandidateContacts] = []
    for candidate_index, pose in enumerate(poses):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        plus_origins = plus_origins_gripper @ rotation.T + translation
        minus_origins = minus_origins_gripper @ rotation.T + translation
        plus_directions = np.repeat((-rotation[:, 0])[None], ray_count, axis=0)
        minus_directions = np.repeat(rotation[:, 0][None], ray_count, axis=0)
        plus_points, plus_triangles = _ray_first_hits(
            mesh,
            plus_origins,
            plus_directions,
        )
        minus_points, minus_triangles = _ray_first_hits(
            mesh,
            minus_origins,
            minus_directions,
        )

        pair_data: list[
            tuple[float, float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for ray_index in sorted(set(plus_points) & set(minus_points)):
            point_plus = plus_points[ray_index]
            point_minus = minus_points[ray_index]
            gripper_plus = rotation.T @ (point_plus - translation)
            gripper_minus = rotation.T @ (point_minus - translation)
            aperture = float(gripper_plus[0] - gripper_minus[0])
            if aperture <= 1e-8:
                continue
            if (
                gripper_minus[0] < x_lo - 1e-7
                or gripper_plus[0] > x_hi + 1e-7
            ):
                continue
            normal_plus = face_normals[plus_triangles[ray_index]]
            normal_minus = face_normals[minus_triangles[ray_index]]
            normal_plus_gripper = rotation.T @ normal_plus
            normal_minus_gripper = rotation.T @ normal_minus
            antipodal = float(
                np.clip(
                    min(
                        normal_plus_gripper[0],
                        -normal_minus_gripper[0],
                        -np.dot(normal_plus_gripper, normal_minus_gripper),
                    ),
                    0.0,
                    1.0,
                )
            )
            pad_center_distance = float(np.linalg.norm(yz[ray_index]))
            pair_data.append(
                (
                    antipodal,
                    -pad_center_distance,
                    ray_index,
                    point_minus,
                    point_plus,
                    normal_minus,
                    normal_plus,
                )
            )

        if not pair_data:
            results.append(
                CandidateContacts(
                    candidate_index=candidate_index,
                    T_object_gripper=pose.copy(),
                    valid=False,
                    contact_points_object=None,
                    contact_normals_object=None,
                    aperture_m=None,
                    antipodal_score=None,
                    ray_yz_gripper=None,
                    reason="no opposing mesh intersections inside the pad sweep volume",
                )
            )
            continue
        best = max(pair_data, key=lambda item: (item[0], item[1], -item[2]))
        (
            antipodal,
            _,
            ray_index,
            point_minus,
            point_plus,
            normal_minus,
            normal_plus,
        ) = best
        aperture = float(
            (rotation.T @ (point_plus - point_minus))[0]
        )
        results.append(
            CandidateContacts(
                candidate_index=candidate_index,
                T_object_gripper=pose.copy(),
                valid=True,
                contact_points_object=np.stack((point_minus, point_plus)),
                contact_normals_object=np.stack((normal_minus, normal_plus)),
                aperture_m=aperture,
                antipodal_score=antipodal,
                ray_yz_gripper=yz[ray_index].copy(),
            )
        )
    return tuple(results)


def unordered_contact_distance(
    candidate_pair: Any,
    human_pair: Any,
) -> tuple[float, bool]:
    """Return mean point distance under the better of the two jaw assignments."""

    candidate = _finite_array(
        candidate_pair,
        name="candidate contact pair",
        shape=(2, 3),
    )
    human = _finite_array(human_pair, name="human contact pair", shape=(2, 3))
    direct = float(np.mean(np.linalg.norm(candidate - human, axis=1)))
    swapped = float(np.mean(np.linalg.norm(candidate[::-1] - human, axis=1)))
    return (swapped, True) if swapped < direct else (direct, False)


def translation_registered_contact_residual(
    candidate_pair: Any,
    human_pair: Any,
) -> tuple[float, np.ndarray, bool]:
    """Fit one common translation and return RMS residual and jaw assignment."""

    candidate = _finite_array(
        candidate_pair,
        name="candidate contact pair",
        shape=(2, 3),
    )
    human = _finite_array(human_pair, name="human contact pair", shape=(2, 3))
    options: list[tuple[float, np.ndarray, bool]] = []
    for ordered, swapped in ((candidate, False), (candidate[::-1], True)):
        translation = np.mean(human - ordered, axis=0)
        residuals = ordered + translation - human
        rms = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
        options.append((rms, translation, swapped))
    return min(options, key=lambda item: item[0])


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def parallel_jaw_rotation_distance(
    candidate_rotation: Any,
    reference_rotation: Any,
) -> tuple[float, bool]:
    """Rotation distance modulo a 180-degree roll about gripper +Z."""

    candidate = _finite_array(
        candidate_rotation,
        name="candidate_rotation",
        shape=(3, 3),
    )
    reference = _finite_array(
        reference_rotation,
        name="reference_rotation",
        shape=(3, 3),
    )
    _validate_transform_batch(
        np.block(
            [
                [candidate, np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        ),
        name="candidate_rotation",
    )
    _validate_transform_batch(
        np.block(
            [
                [reference, np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        ),
        name="reference_rotation",
    )
    direct = _rotation_angle(reference.T @ candidate)
    flipped = _rotation_angle(reference.T @ candidate @ _RZ_PI)
    return (flipped, True) if flipped < direct else (direct, False)


@dataclass(frozen=True)
class GraspScoreWeights:
    """Cost weights; contact geometry intentionally dominates pose priors."""

    contact: float = 1.0
    confidence: float = 0.005
    pose_translation: float = 0.10
    pose_rotation: float = 0.005
    approach: float = 0.003
    registration_weight_fallback: float = 0.25

    def validate(self) -> None:
        values = (
            self.contact,
            self.confidence,
            self.pose_translation,
            self.pose_rotation,
            self.approach,
        )
        if not np.isfinite(values).all() or any(value < 0.0 for value in values):
            raise GraspRefinementError("score weights must be finite and nonnegative")
        if (
            not np.isfinite(self.registration_weight_fallback)
            or not 0.0 <= self.registration_weight_fallback <= 1.0
        ):
            raise GraspRefinementError(
                "registration_weight_fallback must be in [0,1]"
            )


@dataclass(frozen=True)
class CandidateScore:
    candidate_index: int
    feasible: bool
    total_cost: float
    contact_distance_m: float | None
    registered_contact_residual_m: float | None
    registered_translation_m: np.ndarray | None
    registration_weight: float
    confidence_cost: float | None
    pose_translation_m: float | None
    pose_rotation_rad: float | None
    approach_angle_rad: float | None
    symmetry_flipped: bool
    reason: str | None = None


def _registration_weight(
    *,
    supplied_weight: float | None,
    mesh_registration_residual_m: float | None,
    residual_scale_m: float,
    fallback: float,
) -> float:
    if supplied_weight is not None:
        if not np.isfinite(supplied_weight) or not 0.0 <= supplied_weight <= 1.0:
            raise GraspRefinementError("registration_weight must be in [0,1]")
        return float(supplied_weight)
    if mesh_registration_residual_m is None:
        return float(fallback)
    if (
        not np.isfinite(mesh_registration_residual_m)
        or mesh_registration_residual_m < 0.0
    ):
        raise GraspRefinementError(
            "mesh_registration_residual_m must be finite and nonnegative"
        )
    if not np.isfinite(residual_scale_m) or residual_scale_m <= 0.0:
        raise GraspRefinementError("registration_residual_scale_m must be positive")
    ratio = float(mesh_registration_residual_m / residual_scale_m)
    return float(np.exp(-(ratio * ratio)))


def score_grasp_candidates(
    candidates: Sequence[CandidateContacts],
    human_contacts_object: Any | None,
    *,
    confidences: Any | None = None,
    human_pose_object: Any | None = None,
    human_approach_object: Any | None = None,
    aperture_limits_m: tuple[float, float],
    min_antipodal_score: float = 0.5,
    registration_weight: float | None = None,
    mesh_registration_residual_m: float | None = None,
    registration_residual_scale_m: float = 0.02,
    weights: GraspScoreWeights = GraspScoreWeights(),
) -> tuple[CandidateScore, ...]:
    """Score feasible grasps using contacts first and weak pose priors second.

    The direct contact term is blended with a common-translation registered
    residual.  A measured mesh-registration residual determines that blend;
    when no registration diagnostic exists, the explicit fallback in
    ``weights`` is used instead of silently trusting reconstructed translation.
    """

    if not candidates:
        raise GraspRefinementError("at least one grasp candidate is required")
    weights.validate()
    aperture_limits = _finite_array(
        aperture_limits_m,
        name="aperture_limits_m",
        shape=(2,),
    )
    if aperture_limits[0] < 0.0 or aperture_limits[1] <= aperture_limits[0]:
        raise GraspRefinementError(
            "aperture_limits_m must satisfy 0 <= minimum < maximum"
        )
    if (
        not np.isfinite(min_antipodal_score)
        or not 0.0 <= min_antipodal_score <= 1.0
    ):
        raise GraspRefinementError("min_antipodal_score must be in [0,1]")

    human_contacts = (
        None
        if human_contacts_object is None
        else _finite_array(
            human_contacts_object,
            name="human_contacts_object",
            shape=(2, 3),
        )
    )
    human_pose = (
        None
        if human_pose_object is None
        else _validate_transform_batch(
            human_pose_object,
            name="human_pose_object",
            expected_frames=1,
        )[0]
    )
    human_approach = None
    if human_approach_object is not None:
        human_approach = _finite_array(
            human_approach_object,
            name="human_approach_object",
            shape=(3,),
        )
        norm = float(np.linalg.norm(human_approach))
        if norm <= 1e-10:
            raise GraspRefinementError("human_approach_object must be nonzero")
        human_approach = human_approach / norm
    elif human_pose is not None:
        human_approach = human_pose[:3, 2]

    if confidences is None:
        confidence_values = np.ones(len(candidates), dtype=np.float64)
    else:
        confidence_values = _finite_array(confidences, name="confidences")
        if confidence_values.shape != (len(candidates),):
            raise GraspRefinementError(
                f"confidences must have shape ({len(candidates)},)"
            )
        if np.any((confidence_values < 0.0) | (confidence_values > 1.0)):
            raise GraspRefinementError("confidences must lie in [0,1]")

    effective_registration_weight = _registration_weight(
        supplied_weight=registration_weight,
        mesh_registration_residual_m=mesh_registration_residual_m,
        residual_scale_m=registration_residual_scale_m,
        fallback=weights.registration_weight_fallback,
    )
    scores: list[CandidateScore] = []
    for offset, candidate in enumerate(candidates):
        confidence = float(confidence_values[offset])
        if not candidate.valid:
            scores.append(
                CandidateScore(
                    candidate_index=candidate.candidate_index,
                    feasible=False,
                    total_cost=np.inf,
                    contact_distance_m=None,
                    registered_contact_residual_m=None,
                    registered_translation_m=None,
                    registration_weight=effective_registration_weight,
                    confidence_cost=None,
                    pose_translation_m=None,
                    pose_rotation_rad=None,
                    approach_angle_rad=None,
                    symmetry_flipped=False,
                    reason=candidate.reason or "candidate has no derived contacts",
                )
            )
            continue
        if candidate.aperture_m is None or not (
            aperture_limits[0] - 1e-9
            <= candidate.aperture_m
            <= aperture_limits[1] + 1e-9
        ):
            reason = (
                "candidate aperture is unavailable"
                if candidate.aperture_m is None
                else (
                    f"aperture {candidate.aperture_m:.6g} m is outside "
                    f"[{aperture_limits[0]:.6g},{aperture_limits[1]:.6g}]"
                )
            )
            scores.append(
                CandidateScore(
                    candidate_index=candidate.candidate_index,
                    feasible=False,
                    total_cost=np.inf,
                    contact_distance_m=None,
                    registered_contact_residual_m=None,
                    registered_translation_m=None,
                    registration_weight=effective_registration_weight,
                    confidence_cost=None,
                    pose_translation_m=None,
                    pose_rotation_rad=None,
                    approach_angle_rad=None,
                    symmetry_flipped=False,
                    reason=reason,
                )
            )
            continue
        if (
            candidate.antipodal_score is None
            or candidate.antipodal_score < min_antipodal_score
        ):
            scores.append(
                CandidateScore(
                    candidate_index=candidate.candidate_index,
                    feasible=False,
                    total_cost=np.inf,
                    contact_distance_m=None,
                    registered_contact_residual_m=None,
                    registered_translation_m=None,
                    registration_weight=effective_registration_weight,
                    confidence_cost=None,
                    pose_translation_m=None,
                    pose_rotation_rad=None,
                    approach_angle_rad=None,
                    symmetry_flipped=False,
                    reason="candidate fails the antipodal hard filter",
                )
            )
            continue

        contact_distance = 0.0
        registered_residual = 0.0
        registered_translation = np.zeros(3, dtype=np.float64)
        if human_contacts is not None:
            if candidate.contact_points_object is None:
                raise GraspRefinementError(
                    "valid candidate is missing contact_points_object"
                )
            contact_distance, _ = unordered_contact_distance(
                candidate.contact_points_object,
                human_contacts,
            )
            (
                registered_residual,
                registered_translation,
                _,
            ) = translation_registered_contact_residual(
                candidate.contact_points_object,
                human_contacts,
            )
            contact_cost = (
                effective_registration_weight * contact_distance
                + (1.0 - effective_registration_weight) * registered_residual
            )
        else:
            contact_cost = 0.0

        pose_translation = 0.0
        pose_rotation = 0.0
        symmetry_flipped = False
        if human_pose is not None:
            pose_translation = float(
                np.linalg.norm(
                    candidate.T_object_gripper[:3, 3] - human_pose[:3, 3]
                )
            )
            pose_rotation, symmetry_flipped = parallel_jaw_rotation_distance(
                candidate.T_object_gripper[:3, :3],
                human_pose[:3, :3],
            )

        approach_angle = 0.0
        if human_approach is not None:
            candidate_approach = candidate.T_object_gripper[:3, 2]
            approach_angle = float(
                np.arccos(
                    np.clip(
                        np.dot(candidate_approach, human_approach),
                        -1.0,
                        1.0,
                    )
                )
            )
        confidence_cost = float(-np.log(max(confidence, 1e-8)))
        total = (
            weights.contact * contact_cost
            + weights.confidence * confidence_cost
            + weights.pose_translation * pose_translation
            + weights.pose_rotation * pose_rotation
            + weights.approach * approach_angle
        )
        scores.append(
            CandidateScore(
                candidate_index=candidate.candidate_index,
                feasible=True,
                total_cost=float(total),
                contact_distance_m=(
                    float(contact_distance) if human_contacts is not None else None
                ),
                registered_contact_residual_m=(
                    float(registered_residual) if human_contacts is not None else None
                ),
                registered_translation_m=(
                    registered_translation.copy()
                    if human_contacts is not None
                    else None
                ),
                registration_weight=effective_registration_weight,
                confidence_cost=confidence_cost,
                pose_translation_m=(
                    pose_translation if human_pose is not None else None
                ),
                pose_rotation_rad=(
                    pose_rotation if human_pose is not None else None
                ),
                approach_angle_rad=(
                    approach_angle if human_approach is not None else None
                ),
                symmetry_flipped=symmetry_flipped,
            )
        )
    return tuple(scores)


def select_best_candidate(scores: Sequence[CandidateScore]) -> CandidateScore:
    """Return the lowest-cost feasible candidate with deterministic tie-breaking."""

    feasible = [score for score in scores if score.feasible]
    if not feasible:
        reasons = sorted(
            {
                score.reason or "unknown rejection"
                for score in scores
            }
        )
        raise GraspRefinementError(
            f"no feasible grasp candidates; reasons={reasons}"
        )
    return min(feasible, key=lambda score: (score.total_cost, score.candidate_index))


class GraspPropagationMode(str, Enum):
    """How an anchor-frame grasp correction is propagated through an event."""

    OBJECT_LOCK = "object_lock"
    BASE_LOCAL_OFFSET = "base_local_offset"


@dataclass(frozen=True)
class GraspCorrection:
    """One event's selected object-relative gripper pose and jaw aperture."""

    event: GraspEvent
    T_object_gripper: np.ndarray
    aperture_m: float
    propagation_mode: GraspPropagationMode = GraspPropagationMode.OBJECT_LOCK


def _smootherstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _interpolate_pose(
    base_pose: np.ndarray,
    target_pose: np.ndarray,
    weight: float,
) -> np.ndarray:
    if weight <= 0.0:
        return base_pose.copy()
    if weight >= 1.0:
        return target_pose.copy()
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (
        (1.0 - weight) * base_pose[:3, 3]
        + weight * target_pose[:3, 3]
    )
    rotations = Rotation.from_matrix(
        np.stack((base_pose[:3, :3], target_pose[:3, :3]))
    )
    result[:3, :3] = Slerp(
        np.array([0.0, 1.0]),
        rotations,
    )(np.array([weight])).as_matrix()[0]
    return result


def _symmetry_aligned_grasp(
    T_object_gripper: np.ndarray,
    reference_object_gripper: np.ndarray,
) -> np.ndarray:
    _, flip = parallel_jaw_rotation_distance(
        T_object_gripper[:3, :3],
        reference_object_gripper[:3, :3],
    )
    if not flip:
        return T_object_gripper.copy()
    result = T_object_gripper.copy()
    result[:3, :3] = result[:3, :3] @ _RZ_PI
    return result


def apply_phase_aware_corrections(
    target: Mapping[str, np.ndarray],
    *,
    side: str,
    T_world_object: Any,
    corrections: Sequence[GraspCorrection],
    approach_blend_frames: int = 8,
    release_blend_frames: int = 8,
) -> dict[str, np.ndarray]:
    """Apply interaction-local corrections while preserving the 12-key schema.

    ``object_lock`` follows the tracked object pose during each inclusive hold
    interval.  ``base_local_offset`` computes exactly one right-multiplied
    correction at the event anchor,
    ``inv(T_base_anchor) @ (T_world_object_anchor @ T_object_gripper)``, and
    applies it to the base pose at every hold frame.  The latter intentionally
    uses no non-anchor object poses, avoiding propagation of object-tracker
    jitter.  Both modes use the same C2 translation/aperture and Slerp rotation
    blending before contact and after release.  Frames with zero blend weight
    and the other hand are copied exactly from the input arrays.
    """

    frame_count = validate_parallel_jaw_target(target)
    if side not in {"left", "right"}:
        raise GraspRefinementError("side must be exactly 'left' or 'right'")
    for value, name in (
        (approach_blend_frames, "approach_blend_frames"),
        (release_blend_frames, "release_blend_frames"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 0
        ):
            raise GraspRefinementError(f"{name} must be a nonnegative integer")
    object_poses = _validate_transform_batch(
        T_world_object,
        name="T_world_object",
        expected_frames=frame_count,
    )
    output = {key: np.asarray(value).copy() for key, value in target.items()}
    if set(output) != set(PARALLEL_JAW_KEYS):
        raise GraspRefinementError("target no longer matches the exact 12-key schema")
    if not corrections:
        return output

    base_poses = pose_matrix(
        np.asarray(target[f"{side}_position"], dtype=np.float64),
        np.asarray(target[f"{side}_wxyz"], dtype=np.float64),
    )
    base_aperture = np.asarray(
        target[f"{side}_aperture_m"],
        dtype=np.float64,
    )
    frame_owner = np.full(frame_count, -1, dtype=np.int64)
    frame_weight = np.zeros(frame_count, dtype=np.float64)
    desired_poses: dict[int, np.ndarray] = {}
    desired_apertures: dict[int, float] = {}

    ordered = sorted(
        enumerate(corrections),
        key=lambda item: (
            item[1].event.start_frame,
            item[1].event.end_frame,
            item[0],
        ),
    )
    previous_end = -1
    for correction_index, correction in ordered:
        event = correction.event
        if not (
            0 <= event.start_frame
            <= event.anchor_frame
            <= event.end_frame
            < frame_count
        ):
            raise GraspRefinementError(
                f"correction {correction_index} has invalid event bounds"
            )
        if event.start_frame <= previous_end:
            raise GraspRefinementError("grasp event hold intervals must not overlap")
        previous_end = event.end_frame
        if (
            not np.isfinite(correction.aperture_m)
            or correction.aperture_m < 0.0
        ):
            raise GraspRefinementError(
                f"correction {correction_index} aperture must be finite and nonnegative"
            )
        object_from_gripper = _validate_transform_batch(
            correction.T_object_gripper,
            name=f"correction[{correction_index}].T_object_gripper",
            expected_frames=1,
        )[0]
        try:
            propagation_mode = GraspPropagationMode(correction.propagation_mode)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(mode.value for mode in GraspPropagationMode)
            raise GraspRefinementError(
                f"correction {correction_index} propagation_mode must be one of "
                f"{choices}"
            ) from exc

        if propagation_mode is GraspPropagationMode.OBJECT_LOCK:
            reference_object_gripper = (
                np.linalg.inv(object_poses[event.anchor_frame])
                @ base_poses[event.anchor_frame]
            )
            object_from_gripper = _symmetry_aligned_grasp(
                object_from_gripper,
                reference_object_gripper,
            )
            event_poses = object_poses @ object_from_gripper
            start_delta = (
                np.linalg.inv(base_poses[event.start_frame])
                @ event_poses[event.start_frame]
            )
            end_delta = (
                np.linalg.inv(base_poses[event.end_frame])
                @ event_poses[event.end_frame]
            )
        else:
            anchor_target_pose = (
                object_poses[event.anchor_frame] @ object_from_gripper
            )
            anchor_delta = (
                np.linalg.inv(base_poses[event.anchor_frame])
                @ anchor_target_pose
            )
            event_poses = base_poses @ anchor_delta
            start_delta = anchor_delta
            end_delta = anchor_delta

        weights_for_event: dict[int, float] = {}
        for frame in range(event.start_frame, event.end_frame + 1):
            weights_for_event[frame] = 1.0
            desired_poses[frame] = event_poses[frame]
            desired_apertures[frame] = float(correction.aperture_m)
        for offset in range(1, approach_blend_frames + 1):
            frame = event.start_frame - offset
            if frame < 0:
                break
            linear = 1.0 - offset / float(approach_blend_frames + 1)
            weights_for_event[frame] = _smootherstep(linear)
            desired_poses[frame] = base_poses[frame] @ start_delta
            desired_apertures[frame] = float(correction.aperture_m)
        for offset in range(1, release_blend_frames + 1):
            frame = event.end_frame + offset
            if frame >= frame_count:
                break
            linear = 1.0 - offset / float(release_blend_frames + 1)
            weights_for_event[frame] = _smootherstep(linear)
            desired_poses[frame] = base_poses[frame] @ end_delta
            desired_apertures[frame] = float(correction.aperture_m)

        for frame, weight in weights_for_event.items():
            if frame_owner[frame] >= 0 and weight > _EPS and frame_weight[frame] > _EPS:
                raise GraspRefinementError(
                    "approach/release blend windows from different events overlap"
                )
            frame_owner[frame] = correction_index
            frame_weight[frame] = weight

    positions = np.asarray(output[f"{side}_position"])
    quaternions = np.asarray(output[f"{side}_wxyz"])
    apertures = np.asarray(output[f"{side}_aperture_m"])
    previous_quaternion: np.ndarray | None = None
    for frame in range(frame_count):
        weight = float(frame_weight[frame])
        if weight <= _EPS:
            previous_quaternion = np.asarray(quaternions[frame], dtype=np.float64)
            continue
        refined = _interpolate_pose(
            base_poses[frame],
            desired_poses[frame],
            weight,
        )
        quaternion = matrix_to_wxyz(refined[:3, :3])
        reference_quaternion = (
            previous_quaternion
            if previous_quaternion is not None
            else np.asarray(quaternions[frame], dtype=np.float64)
        )
        if np.dot(quaternion, reference_quaternion) < 0.0:
            quaternion *= -1.0
        positions[frame] = refined[:3, 3].astype(positions.dtype, copy=False)
        quaternions[frame] = quaternion.astype(quaternions.dtype, copy=False)
        apertures[frame] = np.asarray(
            (1.0 - weight) * base_aperture[frame]
            + weight * desired_apertures[frame],
            dtype=apertures.dtype,
        )
        previous_quaternion = quaternion

    validate_parallel_jaw_target(output, expected_frames=frame_count)
    return output
