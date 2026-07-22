"""Geometry conversions used by pinned Phantom/HaMeR inference."""

from __future__ import annotations

import numpy as np


def cam_crop_to_full(
    pred_cam: np.ndarray,
    box_center: np.ndarray,
    box_size: np.ndarray,
    image_size: np.ndarray,
    focal_length: float,
) -> np.ndarray:
    """HaMeR weak-perspective crop camera to full-frame translation."""

    pred_cam = np.asarray(pred_cam, dtype=np.float64).reshape(-1, 3)
    box_center = np.asarray(box_center, dtype=np.float64).reshape(-1, 2)
    box_size = np.asarray(box_size, dtype=np.float64).reshape(-1)
    image_size = np.asarray(image_size, dtype=np.float64).reshape(-1, 2)
    if not (
        len(pred_cam) == len(box_center) == len(box_size) == len(image_size)
    ):
        raise ValueError("camera arrays must have the same batch dimension")
    denominator = box_size * pred_cam[:, 0]
    if np.any(np.abs(denominator) < 1e-9):
        raise ValueError("HaMeR predicted a degenerate weak-perspective scale")
    translation = np.empty_like(pred_cam)
    translation[:, 2] = 2.0 * float(focal_length) / denominator
    translation[:, 0] = (
        2.0 * (box_center[:, 0] - image_size[:, 0] * 0.5) / denominator
        + pred_cam[:, 1]
    )
    translation[:, 1] = (
        2.0 * (box_center[:, 1] - image_size[:, 1] * 0.5) / denominator
        + pred_cam[:, 2]
    )
    return translation


def project_points(
    points_camera: np.ndarray, focal_length: float, width: int, height: int
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("points_camera must end in xyz")
    z = points[..., 2]
    projected = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float64)
    positive = np.isfinite(points).all(axis=-1) & (z > 1e-8)
    projected[..., 0][positive] = (
        float(focal_length) * points[..., 0][positive] / z[positive] + width * 0.5
    )
    projected[..., 1][positive] = (
        float(focal_length) * points[..., 1][positive] / z[positive] + height * 0.5
    )
    return projected


def project_points_intrinsics(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project camera-space points with a 3x3 pinhole matrix."""

    points = np.asarray(points_camera, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if points.shape[-1] != 3 or matrix.shape != (3, 3):
        raise ValueError("points must end in xyz and intrinsics must be 3x3")
    if (
        not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[[0, 1], [1, 0]], 0.0, atol=1e-8)
        or not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-8)
    ):
        raise ValueError("intrinsics must be a finite zero-skew pinhole matrix")
    z = points[..., 2]
    projected = np.full(points.shape[:-1] + (2,), np.nan, dtype=np.float64)
    positive = np.isfinite(points).all(axis=-1) & (z > 1e-8)
    projected[..., 0][positive] = (
        matrix[0, 0] * points[..., 0][positive] / z[positive] + matrix[0, 2]
    )
    projected[..., 1][positive] = (
        matrix[1, 1] * points[..., 1][positive] / z[positive] + matrix[1, 2]
    )
    return projected


def remap_virtual_camera_points(
    points_virtual: np.ndarray,
    virtual_focal_length: float,
    width: int,
    height: int,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Remap virtual-camera anchor points to calibrated TACO camera rays.

    The virtual projection is preserved exactly. Metric depth is rescaled by
    ``fx_real / focal_virtual``; x/y are then reconstructed through the real
    principal point and per-axis focal lengths. The Phantom pipeline applies
    this to HaMeR camera translation only; articulated MANO-local geometry is
    then added unchanged so the hand is not anisotropically distorted.
    """

    points = np.asarray(points_virtual, dtype=np.float64)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if not np.isfinite(virtual_focal_length) or virtual_focal_length <= 0:
        raise ValueError("virtual_focal_length must be positive and finite")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsics must be one finite 3x3 matrix")
    uv = project_points(points, virtual_focal_length, width, height)
    result = np.full_like(points, np.nan, dtype=np.float64)
    result[..., 2] = points[..., 2] * matrix[0, 0] / virtual_focal_length
    result[..., 0] = (uv[..., 0] - matrix[0, 2]) * result[..., 2] / matrix[0, 0]
    result[..., 1] = (uv[..., 1] - matrix[1, 2]) * result[..., 2] / matrix[1, 1]
    return result


def mirror_rotation_x(rotation: np.ndarray) -> np.ndarray:
    """Convert a canonical right-hand rotation to a mirrored left rotation."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in 3x3")
    mirror = np.diag([-1.0, 1.0, 1.0])
    return mirror @ rotation @ mirror


def cumulative_mano_joint_rotations(
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    parents: np.ndarray,
    output_joint_indices: tuple[int, ...],
    *,
    mirror_left: bool,
    anatomical_frames: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate, anatomy-align, and expand 16 MANO rotations to 21 joints.

    ``anatomical_frames`` is the native-side ``AxisLayerFK.TMPL_R_p_a``
    correction. Sharpa consumes those anatomy-aligned global joint frames,
    rather than MANO's raw internal transform axes.
    """

    root = np.asarray(global_orient, dtype=np.float64)
    fingers = np.asarray(hand_pose, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64).reshape(-1)
    if root.shape != (3, 3) or fingers.shape != (15, 3, 3) or parents.shape != (16,):
        raise ValueError("expected root (3,3), hand_pose (15,3,3), parents (16,)")
    local = np.concatenate([root[None], fingers], axis=0)
    cumulative = np.empty_like(local)
    cumulative[0] = local[0]
    for joint in range(1, 16):
        parent = int(parents[joint])
        if parent < 0 or parent >= joint:
            raise ValueError(f"invalid MANO parent {parent} for joint {joint}")
        cumulative[joint] = cumulative[parent] @ local[joint]
    if mirror_left:
        cumulative = mirror_rotation_x(cumulative)
    if anatomical_frames is not None:
        frames = np.asarray(anatomical_frames, dtype=np.float64)
        if frames.shape != (16, 3, 3) or not np.isfinite(frames).all():
            raise ValueError("anatomical_frames must be finite (16,3,3)")
        cumulative = cumulative @ frames
    expanded = cumulative[np.asarray(output_joint_indices, dtype=np.int64)]
    if expanded.shape != (21, 3, 3):
        raise ValueError("output_joint_indices must select exactly 21 MANO transforms")
    return expanded


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert one proper rotation matrix to a normalized WXYZ quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be one finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-3) or not np.isclose(
        np.linalg.det(matrix), 1.0, atol=2e-3
    ):
        raise ValueError("rotation must be a proper orthonormal matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        wxyz = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0))
            wxyz = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0))
            wxyz = np.array([(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0))
            wxyz = np.array([(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    if not np.isfinite(wxyz).all() or np.linalg.norm(wxyz) < 1e-12:
        raise ValueError("rotation could not be converted to a quaternion")
    if wxyz[0] < 0:
        wxyz = -wxyz
    return wxyz / np.linalg.norm(wxyz)
