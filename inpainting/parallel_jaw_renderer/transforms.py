"""Dependency-free rigid transform helpers and renderer conventions."""

from __future__ import annotations

import numpy as np


class TransformError(ValueError):
    """Raised when transform semantics cannot be validated exactly."""


def validate_rotation(
    rotation: np.ndarray,
    *,
    name: str = "rotation",
    atol: float = 1e-5,
) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise TransformError(f"{name} must have shape (3,3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise TransformError(f"{name} contains non-finite values")
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(3), atol=atol, rtol=0.0):
        raise TransformError(f"{name} is not orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise TransformError(f"{name} determinant must be +1, got {determinant:.8g}")
    return matrix


def validate_transform(
    transform: np.ndarray,
    *,
    name: str = "transform",
    atol: float = 1e-5,
) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise TransformError(f"{name} must have shape (4,4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise TransformError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=atol, rtol=0.0):
        raise TransformError(f"{name} has a non-homogeneous bottom row")
    validate_rotation(matrix[:3, :3], name=f"{name} rotation", atol=atol)
    return matrix


def validate_transform_batch(
    transforms: np.ndarray,
    *,
    name: str,
    atol: float = 1e-5,
) -> np.ndarray:
    matrices = np.asarray(transforms, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise TransformError(f"{name} must have shape (N,4,4), got {matrices.shape}")
    if matrices.shape[0] == 0:
        raise TransformError(f"{name} must not be empty")
    for index, matrix in enumerate(matrices):
        validate_transform(matrix, name=f"{name}[{index}]", atol=atol)
    return matrices


def invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ matrix[:3, 3])
    return result


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise TransformError(f"quaternion must have shape (4,), got {quat.shape}")
    if not np.isfinite(quat).all():
        raise TransformError("quaternion contains non-finite values")
    norm = float(np.linalg.norm(quat))
    if not np.isclose(norm, 1.0, atol=1e-3, rtol=0.0):
        raise TransformError(f"quaternion norm must be 1, got {norm:.8g}")
    w, x, y, z = quat / norm
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def rotation_from_json(value: object, *, name: str) -> np.ndarray:
    """Accept an explicit 3x3 matrix or scalar-first quaternion."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4,):
        return quaternion_wxyz_to_matrix(array)
    return validate_rotation(array, name=name)


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = validate_rotation(rotation)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                )
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                (
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    translation = np.asarray(position, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise TransformError("position must be a finite shape-(3,) vector")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_wxyz_to_matrix(quaternion_wxyz)
    result[:3, 3] = translation
    return result


def orientation_error_degrees(actual: np.ndarray, target: np.ndarray) -> float:
    relative = validate_rotation(actual).T @ validate_rotation(target)
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


# OpenCV camera coordinates (+x right, +y down, +z forward) to the OpenGL
# coordinates used by an identity pyrender camera (+x right, +y up, -z forward).
CV_TO_OPENGL = np.diag((1.0, -1.0, -1.0, 1.0)).astype(np.float64)
