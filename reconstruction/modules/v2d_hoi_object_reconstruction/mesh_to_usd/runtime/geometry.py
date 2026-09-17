"""Pure geometry helpers shared by mesh generation and drop testing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrincipalAxisPose:
    pose_id: str
    axis_index: int
    axis_sign: int
    local_up: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    principal_rotation_wxyz: tuple[float, float, float, float]
    axis_extent: float
    singular_value: float
    support_plane: SupportPlaneRefinement | None


@dataclass(frozen=True)
class SupportPlaneRefinement:
    applied: bool
    rotation: tuple[tuple[float, float, float], ...]
    reason: str
    correction_degrees: float = 0.0
    face_count: int = 0
    surface_area: float = 0.0
    projected_area: float = 0.0
    normal_before: tuple[float, float, float] | None = None
    plane_rms_error: float | None = None


SUPPORT_BOTTOM_BAND_RATIO = 0.15
SUPPORT_MAX_CORRECTION_DEGREES = 30.0
SUPPORT_NORMAL_TOLERANCE_DEGREES = 8.0
SUPPORT_PLANE_TOLERANCE_RATIO = 0.005
SUPPORT_MIN_AREA_RATIO = 0.01
SUPPORT_MAX_SEEDS = 128


def triangulate_faces(
    counts: Sequence[int],
    indices: Sequence[int],
    offset: int = 0,
) -> list[tuple[int, int, int]]:
    """Triangulate polygon indices with a deterministic fan."""

    triangles = []
    cursor = 0
    for raw_count in counts:
        count = int(raw_count)
        polygon = [int(value) + offset for value in indices[cursor : cursor + count]]
        cursor += count
        for index in range(1, len(polygon) - 1):
            triangles.append((polygon[0], polygon[index], polygon[index + 1]))
    if cursor != len(indices):
        raise ValueError("face counts do not match the index array")
    return triangles


def matrix_to_quaternion_wxyz(
    matrix: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a scalar-first quaternion."""

    matrix_array = np.asarray(matrix, dtype=np.float64)
    if matrix_array.shape != (3, 3) or not np.all(np.isfinite(matrix_array)):
        raise ValueError("rotation matrix must have shape (3, 3) and be finite")
    if not np.allclose(matrix_array.T @ matrix_array, np.eye(3), atol=1e-7):
        raise ValueError("rotation matrix must be orthonormal")
    if not math.isclose(
        float(np.linalg.det(matrix_array)), 1.0, rel_tol=0.0, abs_tol=1e-7
    ):
        raise ValueError("rotation matrix must have determinant +1")

    trace = float(np.trace(matrix_array))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix_array[2, 1] - matrix_array[1, 2]) / scale,
            (matrix_array[0, 2] - matrix_array[2, 0]) / scale,
            (matrix_array[1, 0] - matrix_array[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix_array)))
        if index == 0:
            scale = math.sqrt(
                1.0
                + matrix_array[0, 0]
                - matrix_array[1, 1]
                - matrix_array[2, 2]
            ) * 2.0
            values = (
                (matrix_array[2, 1] - matrix_array[1, 2]) / scale,
                0.25 * scale,
                (matrix_array[0, 1] + matrix_array[1, 0]) / scale,
                (matrix_array[0, 2] + matrix_array[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(
                1.0
                + matrix_array[1, 1]
                - matrix_array[0, 0]
                - matrix_array[2, 2]
            ) * 2.0
            values = (
                (matrix_array[0, 2] - matrix_array[2, 0]) / scale,
                (matrix_array[0, 1] + matrix_array[1, 0]) / scale,
                0.25 * scale,
                (matrix_array[1, 2] + matrix_array[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(
                1.0
                + matrix_array[2, 2]
                - matrix_array[0, 0]
                - matrix_array[1, 1]
            ) * 2.0
            values = (
                (matrix_array[1, 0] - matrix_array[0, 1]) / scale,
                (matrix_array[0, 2] + matrix_array[2, 0]) / scale,
                (matrix_array[1, 2] + matrix_array[2, 1]) / scale,
                0.25 * scale,
            )

    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return tuple(float(value) for value in quaternion)


def vectors_to_rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the minimum proper rotation that maps source onto target."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine <= 1e-12:
        if cosine > 0:
            return np.eye(3)
        pivot = np.zeros(3)
        pivot[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, pivot)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    skew = np.asarray(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        )
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def _fit_support_plane(
    face_points: np.ndarray,
    face_areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a plane to triangles using equal surface-area weight per triangle."""

    sample_points = np.asarray(face_points, dtype=np.float64).reshape((-1, 3))
    sample_weights = np.repeat(np.asarray(face_areas, dtype=np.float64) / 3.0, 3)
    centroid = np.average(sample_points, axis=0, weights=sample_weights)
    centered = sample_points - centroid
    covariance = (centered * sample_weights[:, None]).T @ centered
    covariance /= sample_weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    if normal[2] > 0:
        normal *= -1.0
    rms_error = math.sqrt(max(0.0, float(np.min(eigenvalues))))
    return centroid, normal, rms_error


def refine_rotation_to_support_plane(
    points: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray,
    rotation: Sequence[Sequence[float]] | np.ndarray,
    *,
    bottom_band_ratio: float = SUPPORT_BOTTOM_BAND_RATIO,
    max_correction_degrees: float = SUPPORT_MAX_CORRECTION_DEGREES,
    normal_tolerance_degrees: float = SUPPORT_NORMAL_TOLERANCE_DEGREES,
    plane_tolerance_ratio: float = SUPPORT_PLANE_TOLERANCE_RATIO,
    min_area_ratio: float = SUPPORT_MIN_AREA_RATIO,
) -> SupportPlaneRefinement:
    """Level the dominant low support patch after a coarse pose rotation.

    The search considers only low triangles whose unoriented normal is close to
    world -Z. Approximately coplanar triangles are scored together by projected
    area, allowing a tessellated base or several coplanar feet to form one
    support patch. If no reliable patch is found, the input rotation is returned.
    """

    vertices = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    coarse_rotation = np.asarray(rotation, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("points must be finite")
    if triangles.size == 0:
        triangles = np.empty((0, 3), dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("face indices must reference points")
    if coarse_rotation.shape != (3, 3) or not np.all(np.isfinite(coarse_rotation)):
        raise ValueError("rotation must have shape (3, 3) and be finite")
    if not 0 < bottom_band_ratio <= 1:
        raise ValueError("bottom_band_ratio must be in (0, 1]")
    if not 0 < max_correction_degrees < 90:
        raise ValueError("max_correction_degrees must be in (0, 90)")
    if not 0 < normal_tolerance_degrees < 90:
        raise ValueError("normal_tolerance_degrees must be in (0, 90)")
    if plane_tolerance_ratio <= 0 or min_area_ratio <= 0:
        raise ValueError("support-plane tolerance ratios must be positive")

    unchanged = tuple(tuple(float(value) for value in row) for row in coarse_rotation)
    if not len(triangles):
        return SupportPlaneRefinement(
            applied=False,
            rotation=unchanged,
            reason="no_faces",
        )

    rotated = vertices @ coarse_rotation.T
    extents = np.ptp(rotated, axis=0)
    characteristic_extent = float(np.max(extents))
    height = float(extents[2])
    if characteristic_extent <= 0 or height <= 0:
        return SupportPlaneRefinement(
            applied=False,
            rotation=unchanged,
            reason="zero_extent",
        )

    face_points = rotated[triangles]
    cross = np.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > np.finfo(np.float64).eps * characteristic_extent**2
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / double_area[valid, None]
    normals[normals[:, 2] > 0] *= -1.0
    areas = 0.5 * double_area
    centroids = face_points.mean(axis=1)

    bottom_limit = float(rotated[:, 2].min()) + bottom_band_ratio * height
    downward_cosine = math.cos(math.radians(max_correction_degrees))
    candidates = (
        valid
        & (centroids[:, 2] <= bottom_limit)
        & (-normals[:, 2] >= downward_cosine)
    )
    candidate_indices = np.flatnonzero(candidates)
    if not len(candidate_indices):
        return SupportPlaneRefinement(
            applied=False,
            rotation=unchanged,
            reason="no_low_downward_faces",
        )

    seed_order = candidate_indices[
        np.argsort(-areas[candidate_indices], kind="stable")[:SUPPORT_MAX_SEEDS]
    ]
    normal_cosine = math.cos(math.radians(normal_tolerance_degrees))
    plane_tolerance = plane_tolerance_ratio * characteristic_extent
    best_score = None
    best = None
    for seed_index in seed_order:
        seed_normal = normals[seed_index]
        seed_centroid = centroids[seed_index]
        aligned = normals @ seed_normal >= normal_cosine
        coplanar = (
            np.abs((centroids - seed_centroid) @ seed_normal) <= plane_tolerance
        )
        members = candidates & aligned & coplanar
        member_indices = np.flatnonzero(members)
        member_areas = areas[member_indices]
        surface_area = float(member_areas.sum())
        projected_area = float(
            np.sum(member_areas * np.maximum(0.0, -normals[member_indices, 2]))
        )
        mean_height = float(
            np.average(centroids[member_indices, 2], weights=member_areas)
        )
        score = (projected_area, surface_area, -mean_height, -int(seed_index))
        if best_score is None or score > best_score:
            best_score = score
            best = (
                member_indices,
                surface_area,
                projected_area,
            )

    if best is None:
        raise RuntimeError("support-plane candidate selection produced no result")
    member_indices, surface_area, projected_area = best
    _, support_normal, plane_rms_error = _fit_support_plane(
        face_points[member_indices],
        areas[member_indices],
    )
    if projected_area < min_area_ratio * characteristic_extent**2:
        return SupportPlaneRefinement(
            applied=False,
            rotation=unchanged,
            reason="support_area_too_small",
            face_count=len(member_indices),
            surface_area=surface_area,
            projected_area=projected_area,
            normal_before=tuple(float(value) for value in support_normal),
            plane_rms_error=plane_rms_error,
        )

    target = np.asarray((0.0, 0.0, -1.0))
    correction_degrees = math.degrees(
        math.acos(float(np.clip(np.dot(support_normal, target), -1.0, 1.0)))
    )
    correction = vectors_to_rotation_matrix(support_normal, target)
    refined_rotation = correction @ coarse_rotation
    return SupportPlaneRefinement(
        applied=True,
        rotation=tuple(
            tuple(float(value) for value in row) for row in refined_rotation
        ),
        reason="support_plane_aligned",
        correction_degrees=correction_degrees,
        face_count=len(member_indices),
        surface_area=surface_area,
        projected_area=projected_area,
        normal_before=tuple(float(value) for value in support_normal),
        plane_rms_error=plane_rms_error,
    )


def principal_axis_pose_candidates(
    points: Sequence[Sequence[float]] | np.ndarray,
    faces: Sequence[Sequence[int]] | np.ndarray | None = None,
    *,
    refine_support_plane: bool = False,
    support_points: Sequence[Sequence[float]] | np.ndarray | None = None,
    support_faces: Sequence[Sequence[int]] | np.ndarray | None = None,
) -> tuple[PrincipalAxisPose, ...]:
    """Return six signed principal-axis poses, optionally leveling support.

    The principal axes always come from ``points``. When supplied, the
    independent support geometry is used only to refine each coarse rotation.
    This lets physics validation level the collider surface that will actually
    contact the ground without allowing simplified colliders to bias PCA.
    """

    vertices = np.asarray(points, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(vertices) < 3:
        raise ValueError("at least three points are required")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("points must be finite")

    centered = vertices - vertices.mean(axis=0)
    if not np.any(np.linalg.norm(centered, axis=1) > 0):
        raise ValueError("points must span a non-zero extent")
    _, singular_values, raw_axes = np.linalg.svd(centered, full_matrices=False)

    refinement_vertices = vertices if support_points is None else support_points
    if support_faces is None:
        refinement_faces = (
            np.empty((0, 3), dtype=np.int64) if faces is None else faces
        )
    else:
        refinement_faces = support_faces

    axes = []
    for raw_axis in raw_axes:
        axis = np.asarray(raw_axis, dtype=np.float64)
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis = -axis
        axes.append(axis)
    axes_array = np.asarray(axes, dtype=np.float64)
    extents = np.ptp(centered @ axes_array.T, axis=0)

    candidates = []
    for axis_index in range(3):
        horizontal_index = max(
            (index for index in range(3) if index != axis_index),
            key=lambda index: (float(extents[index]), -index),
        )
        horizontal = axes_array[horizontal_index]
        for axis_sign in (1, -1):
            local_up = axis_sign * axes_array[axis_index]
            local_side = np.cross(local_up, horizontal)
            local_side /= np.linalg.norm(local_side)
            rotation = np.stack((horizontal, local_side, local_up))
            principal_rotation_wxyz = matrix_to_quaternion_wxyz(rotation)
            refinement = None
            if refine_support_plane:
                refinement = refine_rotation_to_support_plane(
                    refinement_vertices,
                    refinement_faces,
                    rotation,
                )
                rotation = np.asarray(refinement.rotation)
            direction = "positive" if axis_sign > 0 else "negative"
            candidates.append(
                PrincipalAxisPose(
                    pose_id=f"principal_axis_{axis_index}_{direction}",
                    axis_index=axis_index,
                    axis_sign=axis_sign,
                    local_up=tuple(float(value) for value in local_up),
                    rotation_wxyz=matrix_to_quaternion_wxyz(rotation),
                    principal_rotation_wxyz=principal_rotation_wxyz,
                    axis_extent=float(extents[axis_index]),
                    singular_value=float(singular_values[axis_index]),
                    support_plane=refinement,
                )
            )
    return tuple(candidates)
