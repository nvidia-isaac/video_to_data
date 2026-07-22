"""Small, dependency-free rigid-transform helpers.

Conventions used throughout this package:

* homogeneous transforms are named ``T_destination_source``;
* quaternions are scalar-first ``(w, x, y, z)``;
* camera coordinates use OpenCV axes: +x right, +y down, +z forward;
* world-to-camera calibration therefore maps ``p_world`` to ``p_camera``.
"""

from __future__ import annotations

import numpy as np


class TransformError(ValueError):
    """Raised when a matrix or quaternion is not a valid rigid transform."""


def validate_rigid_transform(
    transform: np.ndarray,
    *,
    name: str = "transform",
    atol: float = 1e-5,
) -> np.ndarray:
    """Return ``transform`` as float64 after strict SE(3) validation."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise TransformError(f"{name} must have shape (4,4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise TransformError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=atol, rtol=0.0):
        raise TransformError(f"{name} has a non-homogeneous bottom row: {matrix[3].tolist()}")
    rotation = matrix[:3, :3]
    gram = rotation.T @ rotation
    if not np.allclose(gram, np.eye(3), atol=atol, rtol=0.0):
        error = float(np.max(np.abs(gram - np.eye(3))))
        raise TransformError(f"{name} rotation is not orthonormal (max error {error:.3g})")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise TransformError(f"{name} rotation determinant must be +1, got {determinant:.8g}")
    return matrix


def validate_transform_batch(
    transforms: np.ndarray,
    *,
    name: str = "transforms",
    atol: float = 1e-5,
) -> np.ndarray:
    """Validate a non-empty ``(N,4,4)`` batch of SE(3) matrices."""

    matrices = np.asarray(transforms, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise TransformError(f"{name} must have shape (N,4,4), got {matrices.shape}")
    if matrices.shape[0] == 0:
        raise TransformError(f"{name} must contain at least one transform")
    if not np.isfinite(matrices).all():
        raise TransformError(f"{name} contains non-finite values")
    bottoms = matrices[:, 3, :]
    expected_bottom = np.broadcast_to((0.0, 0.0, 0.0, 1.0), bottoms.shape)
    if not np.allclose(bottoms, expected_bottom, atol=atol, rtol=0.0):
        bad = int(np.where(np.max(np.abs(bottoms - expected_bottom), axis=1) > atol)[0][0])
        raise TransformError(f"{name}[{bad}] has a non-homogeneous bottom row")
    rotations = matrices[:, :3, :3]
    grams = np.swapaxes(rotations, 1, 2) @ rotations
    orthogonal_error = np.max(np.abs(grams - np.eye(3)), axis=(1, 2))
    if np.any(orthogonal_error > atol):
        bad = int(np.argmax(orthogonal_error))
        raise TransformError(
            f"{name}[{bad}] rotation is not orthonormal "
            f"(max error {orthogonal_error[bad]:.3g})"
        )
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=atol, rtol=0.0):
        bad = int(np.argmax(np.abs(determinants - 1.0)))
        raise TransformError(
            f"{name}[{bad}] rotation determinant must be +1, got {determinants[bad]:.8g}"
        )
    return matrices


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert one unit WXYZ quaternion to a ``(3,3)`` rotation matrix."""

    quat = np.asarray(quaternion, dtype=np.float64)
    if quat.shape != (4,):
        raise TransformError(f"quaternion must have shape (4,), got {quat.shape}")
    if not np.isfinite(quat).all():
        raise TransformError("quaternion contains non-finite values")
    norm = float(np.linalg.norm(quat))
    if not np.isclose(norm, 1.0, atol=1e-3, rtol=0.0):
        raise TransformError(f"quaternion must be unit length, got norm {norm:.8g}")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized WXYZ quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise TransformError(f"rotation must have shape (3,3), got {matrix.shape}")
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, :3] = matrix
    validate_rigid_transform(candidate, name="rotation")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= np.linalg.norm(quat)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Build an SE(3) pose from a position and WXYZ quaternion."""

    translation = np.asarray(position, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise TransformError("position must be a finite shape-(3,) vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(quaternion_wxyz)
    transform[:3, 3] = translation
    return transform


def invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a validated SE(3) matrix without a generic matrix inverse."""

    matrix = validate_rigid_transform(transform)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -(result[:3, :3] @ matrix[:3, 3])
    return result


def transform_pose_batch(
    destination_from_source: np.ndarray,
    source_poses: np.ndarray,
) -> np.ndarray:
    """Compose aligned ``(N,4,4)`` batches or one transform with a pose batch."""

    poses = np.asarray(source_poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise TransformError(f"source_poses must have shape (N,4,4), got {poses.shape}")
    transforms = np.asarray(destination_from_source, dtype=np.float64)
    if transforms.shape == (4, 4):
        return transforms[None, ...] @ poses
    if transforms.shape == poses.shape:
        return transforms @ poses
    raise TransformError(
        "destination_from_source must be (4,4) or match source_poses; "
        f"got {transforms.shape} and {poses.shape}"
    )


# OpenCV camera coordinates (+x right, +y down, +z forward) to the OpenGL
# world convention used with an identity pyrender camera (+x right, +y up,
# camera looking down -z).
CV_TO_OPENGL = np.diag((1.0, -1.0, -1.0, 1.0)).astype(np.float64)
