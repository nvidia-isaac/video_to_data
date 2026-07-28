"""Retarget common MANO joint tracks to robot-neutral parallel-jaw targets.

The geometry and temporal filtering intentionally follow Phantom's
unconstrained bimanual policy, stopping before its robot-specific ``Rz(90)``
mount correction:

* reference revision: ``MarionLepert/phantom`` commit
  ``a8bb81c1bbe6ade129a1f6f0906482f510354a5e``;
* reference files: ``phantom/hand.py``,
  ``phantom/processors/smoothing_processor.py``, and
  ``phantom/processors/robotinpaint_processor.py``.

* position: thumb-tip / middle-tip midpoint;
* aperture: thumb-tip / index-tip distance;
* orientation: a semantic frame built from thumb-tip, index-tip, and index MCP;
* smoothing: Gaussian-process position/aperture and Gaussian-weighted SLERP;
* grasp interval: widths are capped at 20 percent of the sequence span above
  the minimum between the first and last below-threshold observations.

The output is an embodiment-independent seam.  A Galbot- or YAM-specific stage
must apply its own tool-frame transform and jaw limits after consuming it. The
reference checkout is not imported; this module consumes only the common
Video2Data tracking contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from inpainting.contracts import ContractError, TRACKERS, validate_tracking_arrays


PARALLEL_JAW_TRAJECTORY_SCHEMA = "v2d.inpainting.parallel-jaw-target/v1"
PARALLEL_JAW_RUN_SCHEMA = "v2d.inpainting.parallel-jaw-retarget-run/v1"
TRAJECTORY_FILENAME = "parallel_jaw_trajectory.npz"
METADATA_FILENAME = "parallel_jaw_trajectory.json"

SIDES = ("left", "right")
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_TIP = 12
GEOMETRY_EPSILON_M = 1e-8
GRASP_SPAN_FRACTION = 0.20
ORIENTATION_SIGMA = 10.0
ORIENTATION_KERNEL_SIZE = 21

PARALLEL_JAW_KEYS = frozenset(
    {
        "schema_version",
        "tracker",
        "coordinate_frame",
        "frame_indices",
        "left_valid",
        "right_valid",
        "left_position",
        "right_position",
        "left_wxyz",
        "right_wxyz",
        "left_aperture_m",
        "right_aperture_m",
    }
)

ArraySmoother = Callable[[np.ndarray], np.ndarray]


class ParallelJawRetargetError(ContractError):
    """Raised when a tracked sequence cannot safely produce gripper targets."""


def _scalar_text(value: np.ndarray | Any, key: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ParallelJawRetargetError(f"{key} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParallelJawRetargetError(f"{key} is not valid UTF-8") from exc
    text = str(item)
    if not text:
        raise ParallelJawRetargetError(f"{key} must not be empty")
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _assert_artifact_unchanged(
    label: str, path: Path, original: dict[str, Any]
) -> None:
    current = _artifact(path)
    if (
        current["size_bytes"] != original["size_bytes"]
        or current["sha256"] != original["sha256"]
    ):
        raise ParallelJawRetargetError(
            f"{label} changed while retargeting was running; refusing atomic commit"
        )


def load_tracking(path: str | Path) -> tuple[dict[str, np.ndarray], int]:
    """Load and strictly validate the common ``tracking.npz`` seam."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ParallelJawRetargetError(
            f"Cannot read tracking archive {path}: {exc}"
        ) from exc

    try:
        frame_count = validate_tracking_arrays(arrays)
    except ContractError as exc:
        raise ParallelJawRetargetError(str(exc)) from exc
    if frame_count == 0:
        raise ParallelJawRetargetError("tracking.npz contains no frames")

    for side in SIDES:
        joint_key = f"{side}_joints_3d"
        if joint_key not in arrays:
            raise ParallelJawRetargetError(
                f"Parallel-jaw retargeting requires {joint_key}"
            )
        valid = np.asarray(arrays[f"{side}_valid"])
        missing = np.flatnonzero(~valid)
        if missing.size:
            preview = ", ".join(str(int(index)) for index in missing[:8])
            suffix = "..." if missing.size > 8 else ""
            raise ParallelJawRetargetError(
                f"{side}_valid must cover every frame; invalid frames: "
                f"{preview}{suffix}"
            )
        joints = np.asarray(arrays[joint_key])
        if not np.isfinite(joints).all():
            raise ParallelJawRetargetError(f"{joint_key} must be finite on every frame")
    return arrays, frame_count


def _validate_transform_batch(
    transforms: np.ndarray, *, frame_count: int
) -> tuple[np.ndarray, bool]:
    transforms = np.asarray(transforms, dtype=np.float64)
    broadcast = transforms.shape == (4, 4)
    if broadcast:
        transforms = np.repeat(transforms[None], frame_count, axis=0)
    if transforms.shape != (frame_count, 4, 4):
        raise ParallelJawRetargetError(
            "T_camera_world must have shape (4,4) or "
            f"({frame_count},4,4), got {transforms.shape}"
        )
    if not np.isfinite(transforms).all():
        raise ParallelJawRetargetError("T_camera_world contains non-finite values")
    if not np.allclose(transforms[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ParallelJawRetargetError(
            "T_camera_world must contain homogeneous rigid transforms"
        )
    rotations = transforms[:, :3, :3]
    identities = np.einsum("nji,njk->nik", rotations, rotations)
    if not np.allclose(identities, np.eye(3), atol=1e-5):
        raise ParallelJawRetargetError("T_camera_world rotations must be orthonormal")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=1e-5):
        raise ParallelJawRetargetError(
            "T_camera_world rotations must be proper (determinant +1)"
        )
    return transforms, broadcast


def load_world_to_camera(
    path: str | Path, *, frame_count: int
) -> tuple[np.ndarray, bool]:
    """Load TACO ``T_camera_world`` (a world-to-camera transform batch)."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                candidates = [
                    key
                    for key in ("T_camera_world", "world_to_camera", "extrinsic")
                    if key in loaded.files
                ]
                if len(candidates) == 1:
                    transforms = np.asarray(loaded[candidates[0]])
                elif len(loaded.files) == 1:
                    transforms = np.asarray(loaded[loaded.files[0]])
                else:
                    raise ParallelJawRetargetError(
                        "Transform NPZ must contain one of T_camera_world, "
                        "world_to_camera, or extrinsic"
                    )
            finally:
                loaded.close()
        else:
            transforms = np.asarray(loaded)
    except ParallelJawRetargetError:
        raise
    except (OSError, ValueError) as exc:
        raise ParallelJawRetargetError(
            f"Cannot read T_camera_world archive {path}: {exc}"
        ) from exc
    return _validate_transform_batch(transforms, frame_count=frame_count)


def joints_in_world(
    tracking: dict[str, np.ndarray],
    *,
    world_to_camera: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Preserve world joints or invert T_camera_world for camera joints."""

    coordinate_frame = _scalar_text(tracking["coordinate_frame"], "coordinate_frame")
    frame_count = int(np.asarray(tracking["frame_indices"]).shape[0])
    if coordinate_frame == "world":
        return {
            side: np.asarray(tracking[f"{side}_joints_3d"], dtype=np.float64).copy()
            for side in SIDES
        }
    if coordinate_frame != "camera":
        raise ParallelJawRetargetError(
            f"Unsupported tracking coordinate frame {coordinate_frame!r}"
        )
    if world_to_camera is None:
        raise ParallelJawRetargetError(
            "Camera-frame tracking requires T_camera_world (world-to-camera); "
            "an identity transform is never inferred"
        )
    world_to_camera, _ = _validate_transform_batch(
        world_to_camera, frame_count=frame_count
    )
    try:
        camera_to_world = np.linalg.inv(world_to_camera)
    except np.linalg.LinAlgError as exc:
        raise ParallelJawRetargetError(
            "T_camera_world contains a non-invertible transform"
        ) from exc

    result: dict[str, np.ndarray] = {}
    for side in SIDES:
        joints_camera = np.asarray(tracking[f"{side}_joints_3d"], dtype=np.float64)
        result[side] = (
            np.einsum("nij,nkj->nki", camera_to_world[:, :3, :3], joints_camera)
            + camera_to_world[:, None, :3, 3]
        )
    return result


def derive_parallel_jaw_geometry(
    joints_3d: np.ndarray,
    *,
    side: str = "hand",
    epsilon_m: float = GEOMETRY_EPSILON_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Derive Phantom's unconstrained pose and aperture without ``Rz(90)``."""

    joints = np.asarray(joints_3d, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        raise ParallelJawRetargetError(
            f"{side} joints must have shape (N,21,3), got {joints.shape}"
        )
    if joints.shape[0] == 0:
        raise ParallelJawRetargetError(f"{side} joints contain no frames")
    if not np.isfinite(joints).all():
        raise ParallelJawRetargetError(f"{side} joints contain non-finite values")
    if not np.isfinite(epsilon_m) or epsilon_m <= 0.0:
        raise ValueError("epsilon_m must be finite and positive")

    thumb_tip = joints[:, THUMB_TIP]
    index_mcp = joints[:, INDEX_MCP]
    index_tip = joints[:, INDEX_TIP]
    middle_tip = joints[:, MIDDLE_TIP]

    positions = (thumb_tip + middle_tip) / 2.0
    opening_vectors = thumb_tip - index_tip
    apertures = np.linalg.norm(opening_vectors, axis=1)
    tip_midpoints = (thumb_tip + index_tip) / 2.0
    palm_axes = index_mcp - tip_midpoints
    palm_norms = np.linalg.norm(palm_axes, axis=1)

    bad_opening = np.flatnonzero(apertures <= epsilon_m)
    if bad_opening.size:
        raise ParallelJawRetargetError(
            f"{side} frame {int(bad_opening[0])} has coincident thumb/index tips"
        )
    bad_palm = np.flatnonzero(palm_norms <= epsilon_m)
    if bad_palm.size:
        raise ParallelJawRetargetError(
            f"{side} frame {int(bad_palm[0])} has a degenerate index-MCP palm axis"
        )

    x_axes = opening_vectors / apertures[:, None]
    z_seed = -palm_axes / palm_norms[:, None]
    y_axes = np.cross(z_seed, x_axes)
    cross_norms = np.linalg.norm(y_axes, axis=1)
    bad_cross = np.flatnonzero(cross_norms <= epsilon_m)
    if bad_cross.size:
        raise ParallelJawRetargetError(
            f"{side} frame {int(bad_cross[0])} has collinear opening and palm axes"
        )
    y_axes /= cross_norms[:, None]
    z_axes = np.cross(x_axes, y_axes)
    z_norms = np.linalg.norm(z_axes, axis=1)
    if np.any(z_norms <= epsilon_m):
        raise ParallelJawRetargetError(
            f"{side} semantic orientation could not be orthogonalized"
        )
    z_axes /= z_norms[:, None]

    # Match Phantom's palm-consistency check.  All three axes are flipped
    # together when needed; the determinant correction below restores a proper
    # frame exactly as Phantom's implementation does.
    flip = np.einsum("ni,ni->n", z_axes, palm_axes) > 0.0
    x_axes[flip] *= -1.0
    y_axes[flip] *= -1.0
    z_axes[flip] *= -1.0

    orientations = np.stack((x_axes, y_axes, z_axes), axis=2)
    determinants = np.linalg.det(orientations)
    improper = determinants < 0.0
    orientations[improper, :, 0] *= -1.0
    determinants = np.linalg.det(orientations)
    orthogonality = np.einsum("nji,njk->nik", orientations, orientations)
    if (
        not np.isfinite(orientations).all()
        or not np.allclose(orthogonality, np.eye(3), atol=1e-6)
        or not np.allclose(determinants, 1.0, atol=1e-6)
    ):
        raise ParallelJawRetargetError(
            f"{side} semantic orientation is not a proper rotation"
        )

    diagnostics = {
        "minimum_thumb_index_distance_m": float(np.min(apertures)),
        "minimum_index_mcp_axis_m": float(np.min(palm_norms)),
        "minimum_basis_cross_norm": float(np.min(cross_norms)),
    }
    return positions, orientations, apertures, diagnostics


def phantom_gaussian_process_smoothing(points: np.ndarray) -> np.ndarray:
    """Apply Phantom's RBF + white-noise Gaussian-process smoother."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] == 0:
        raise ParallelJawRetargetError(
            "Gaussian-process input must have shape (N,) or (N,D), with N > 0"
        )
    if not np.isfinite(values).all():
        raise ParallelJawRetargetError(
            "Gaussian-process input contains non-finite values"
        )
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    except ImportError as exc:
        raise ParallelJawRetargetError(
            "Phantom-style position/aperture smoothing requires scikit-learn"
        ) from exc

    time = np.arange(values.shape[0], dtype=np.float64)[:, None]
    kernel = RBF(length_scale=1) + WhiteKernel(noise_level=1)
    regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    if values.ndim == 1:
        result = regressor.fit(time, values).predict(time)
    else:
        result = np.column_stack(
            [
                regressor.fit(time, values[:, dimension]).predict(time)
                for dimension in range(values.shape[1])
            ]
        )
    return np.asarray(result, dtype=np.float64)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size <= 0 or size % 2 != 1:
        raise ValueError("Gaussian SLERP kernel size must be a positive odd integer")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Gaussian SLERP sigma must be finite and positive")
    offsets = np.arange(size, dtype=np.float64) - size // 2
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    return weights / weights.sum()


def phantom_gaussian_slerp_smoothing(
    rotation_matrices: np.ndarray,
    *,
    sigma: float = ORIENTATION_SIGMA,
    kernel_size: int = ORIENTATION_KERNEL_SIZE,
) -> np.ndarray:
    """Apply Phantom's local Gaussian-weighted iterative SLERP smoother."""

    matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not len(matrices):
        raise ParallelJawRetargetError(
            "Orientation smoother input must have shape (N,3,3), with N > 0"
        )
    if not np.isfinite(matrices).all():
        raise ParallelJawRetargetError(
            "Orientation smoother input contains non-finite values"
        )
    try:
        from scipy.spatial.transform import Rotation, Slerp
    except ImportError as exc:
        raise ParallelJawRetargetError(
            "Phantom-style orientation smoothing requires scipy"
        ) from exc

    half_kernel = kernel_size // 2
    weights = _gaussian_kernel(kernel_size, sigma)
    quaternions = Rotation.from_matrix(matrices).as_quat()
    continuous = [quaternions[0]]
    for quaternion in quaternions[1:]:
        if np.dot(quaternion, continuous[-1]) < 0.0:
            quaternion = -quaternion
        continuous.append(quaternion)
    quaternions = np.asarray(continuous)

    smoothed: list[np.ndarray] = []
    for index in range(len(matrices)):
        start = max(0, index - half_kernel)
        end = min(len(matrices), index + half_kernel + 1)
        local_quaternions = quaternions[start:end]
        # Copy before normalization.  Phantom's intent is a fixed Gaussian
        # kernel; mutating a shared view would make later frames order-dependent.
        local_weights = weights[
            half_kernel - (index - start) : half_kernel + (end - index)
        ].copy()
        local_weights /= local_weights.sum()
        average = Rotation.from_quat(local_quaternions[0])
        for local_index in range(1, len(local_quaternions)):
            next_rotation = Rotation.from_quat(local_quaternions[local_index])
            interpolation_weight = local_weights[local_index] / np.sum(
                local_weights[: local_index + 1]
            )
            keyframes = Rotation.concatenate([average, next_rotation])
            average = Slerp([0.0, 1.0], keyframes)([interpolation_weight])[0]
        smoothed.append(average.as_matrix())
    return np.stack(smoothed)


def apply_grasp_span_width_cap(
    widths_m: np.ndarray,
    *,
    fraction: float = GRASP_SPAN_FRACTION,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Phantom's sequence-relative grasp-interval width cap."""

    widths = np.asarray(widths_m, dtype=np.float64)
    if widths.ndim != 1 or widths.size == 0:
        raise ParallelJawRetargetError("Widths must have shape (N,), with N > 0")
    if not np.isfinite(widths).all() or np.any(widths < 0.0):
        raise ParallelJawRetargetError(
            "Smoothed widths must be finite and non-negative"
        )
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("Grasp-span fraction must be within [0,1]")

    minimum = float(np.min(widths))
    maximum = float(np.max(widths))
    span = maximum - minimum
    threshold = minimum + fraction * span
    below = np.flatnonzero(widths < threshold)
    capped = widths.copy()
    first: int | None = None
    last: int | None = None
    if below.size:
        first = int(below[0])
        last = int(below[-1])
        capped[first : last + 1] = np.minimum(capped[first : last + 1], threshold)
    changed = np.flatnonzero(~np.isclose(capped, widths, rtol=0.0, atol=0.0))
    return capped, {
        "fraction": float(fraction),
        "minimum_m": minimum,
        "maximum_m": maximum,
        "span_m": span,
        "threshold_m": float(threshold),
        "first_grasp_frame": first,
        "last_grasp_frame": last,
        "capped_frame_count": int(changed.size),
    }


def _validate_smoothed(
    values: np.ndarray, expected_shape: tuple[int, ...], label: str
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != expected_shape:
        raise ParallelJawRetargetError(
            f"{label} returned shape {result.shape}; expected {expected_shape}"
        )
    if not np.isfinite(result).all():
        raise ParallelJawRetargetError(f"{label} returned non-finite values")
    return result


def _rotation_matrices_to_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to continuous WXYZ quaternions."""

    rotations = np.asarray(matrices, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ParallelJawRetargetError(
            f"Rotations must have shape (N,3,3), got {rotations.shape}"
        )
    products = np.einsum("nji,njk->nik", rotations, rotations)
    if (
        not np.isfinite(rotations).all()
        or not np.allclose(products, np.eye(3), atol=1e-5)
        or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5)
    ):
        raise ParallelJawRetargetError(
            "Orientation smoother returned a non-rotation matrix"
        )

    quaternions = np.empty((len(rotations), 4), dtype=np.float64)
    for index, matrix in enumerate(rotations):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
        else:
            diagonal = np.diag(matrix)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                quaternion = np.array(
                    [
                        (matrix[2, 1] - matrix[1, 2]) / scale,
                        0.25 * scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                quaternion = np.array(
                    [
                        (matrix[0, 2] - matrix[2, 0]) / scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        0.25 * scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                    ]
                )
            else:
                scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                quaternion = np.array(
                    [
                        (matrix[1, 0] - matrix[0, 1]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ParallelJawRetargetError(
                f"Cannot convert orientation frame {index} to a quaternion"
            )
        quaternions[index] = quaternion / norm
        if index and np.dot(quaternions[index], quaternions[index - 1]) < 0.0:
            quaternions[index] *= -1.0
    return quaternions


def retarget_tracking_arrays(
    tracking: dict[str, np.ndarray],
    *,
    world_to_camera: np.ndarray | None = None,
    position_smoother: ArraySmoother | None = None,
    width_smoother: ArraySmoother | None = None,
    orientation_smoother: ArraySmoother | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Retarget validated common tracking arrays through one shared policy."""

    try:
        frame_count = validate_tracking_arrays(tracking)
    except ContractError as exc:
        raise ParallelJawRetargetError(str(exc)) from exc
    if frame_count == 0:
        raise ParallelJawRetargetError("tracking.npz contains no frames")
    for side in SIDES:
        if f"{side}_joints_3d" not in tracking:
            raise ParallelJawRetargetError(
                f"Parallel-jaw retargeting requires {side}_joints_3d"
            )
        valid = np.asarray(tracking[f"{side}_valid"])
        if not valid.all():
            raise ParallelJawRetargetError(
                f"{side}_valid must cover all {frame_count} frames"
            )

    if position_smoother is None:
        position_smoother = phantom_gaussian_process_smoothing
    if width_smoother is None:
        width_smoother = phantom_gaussian_process_smoothing
    if orientation_smoother is None:
        orientation_smoother = phantom_gaussian_slerp_smoothing

    world_joints = joints_in_world(tracking, world_to_camera=world_to_camera)
    tracker = _scalar_text(tracking["tracker"], "tracker")
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PARALLEL_JAW_TRAJECTORY_SCHEMA),
        "tracker": np.asarray(tracker),
        "coordinate_frame": np.asarray("world"),
        "frame_indices": np.asarray(tracking["frame_indices"], dtype=np.int32).copy(),
    }
    diagnostics: dict[str, Any] = {"sides": {}}
    for side in SIDES:
        positions, orientations, raw_widths, geometry_diagnostics = (
            derive_parallel_jaw_geometry(world_joints[side], side=side)
        )
        smoothed_positions = _validate_smoothed(
            position_smoother(positions),
            (frame_count, 3),
            f"{side} position smoother",
        )
        smoothed_orientations = _validate_smoothed(
            orientation_smoother(orientations),
            (frame_count, 3, 3),
            f"{side} orientation smoother",
        )
        smoothed_widths = _validate_smoothed(
            width_smoother(raw_widths),
            (frame_count,),
            f"{side} width smoother",
        )
        capped_widths, cap_diagnostics = apply_grasp_span_width_cap(smoothed_widths)
        output[f"{side}_valid"] = np.ones(frame_count, dtype=np.bool_)
        output[f"{side}_position"] = smoothed_positions.astype(np.float32)
        output[f"{side}_wxyz"] = _rotation_matrices_to_wxyz(
            smoothed_orientations
        ).astype(np.float32)
        output[f"{side}_aperture_m"] = capped_widths.astype(np.float32)
        diagnostics["sides"][side] = {
            "valid_frames": frame_count,
            "raw_aperture_m": raw_widths.tolist(),
            "raw_aperture_summary_m": {
                "minimum": float(np.min(raw_widths)),
                "maximum": float(np.max(raw_widths)),
                "mean": float(np.mean(raw_widths)),
            },
            "smoothed_aperture_summary_m": {
                "minimum": float(np.min(smoothed_widths)),
                "maximum": float(np.max(smoothed_widths)),
                "mean": float(np.mean(smoothed_widths)),
            },
            "geometry": geometry_diagnostics,
            "grasp_span_cap": cap_diagnostics,
        }
    validate_parallel_jaw_arrays(output, expected_frames=frame_count)
    return output, diagnostics


def validate_parallel_jaw_arrays(
    arrays: dict[str, np.ndarray], *, expected_frames: int | None = None
) -> int:
    """Validate the exact robot-neutral renderer integration seam."""

    keys = set(arrays)
    if keys != PARALLEL_JAW_KEYS:
        missing = sorted(PARALLEL_JAW_KEYS - keys)
        unexpected = sorted(keys - PARALLEL_JAW_KEYS)
        raise ParallelJawRetargetError(
            f"Parallel-jaw archive keys differ from the exact schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    schema = _scalar_text(arrays["schema_version"], "schema_version")
    if schema != PARALLEL_JAW_TRAJECTORY_SCHEMA:
        raise ParallelJawRetargetError(f"Unsupported parallel-jaw schema {schema!r}")
    tracker = _scalar_text(arrays["tracker"], "tracker")
    if tracker not in TRACKERS:
        raise ParallelJawRetargetError(f"Unsupported tracker {tracker!r}")
    if _scalar_text(arrays["coordinate_frame"], "coordinate_frame") != "world":
        raise ParallelJawRetargetError(
            "Parallel-jaw targets must be expressed in world coordinates"
        )
    frame_indices = np.asarray(arrays["frame_indices"])
    if frame_indices.ndim != 1 or not np.issubdtype(frame_indices.dtype, np.integer):
        raise ParallelJawRetargetError(
            "frame_indices must be a one-dimensional integer array"
        )
    frame_count = int(frame_indices.shape[0])
    if frame_count == 0:
        raise ParallelJawRetargetError("Parallel-jaw archive contains no frames")
    if expected_frames is not None and frame_count != expected_frames:
        raise ParallelJawRetargetError(
            f"Parallel-jaw archive has {frame_count} frames; expected {expected_frames}"
        )
    if not np.array_equal(frame_indices, np.arange(frame_count)):
        raise ParallelJawRetargetError("frame_indices must be contiguous 0..N-1")
    for side in SIDES:
        valid = np.asarray(arrays[f"{side}_valid"])
        if valid.shape != (frame_count,) or valid.dtype != np.dtype(np.bool_):
            raise ParallelJawRetargetError(
                f"{side}_valid must be boolean with shape ({frame_count},)"
            )
        if not valid.all():
            raise ParallelJawRetargetError(
                f"{side}_valid must be true for every output frame"
            )
        for suffix, shape in (
            ("position", (frame_count, 3)),
            ("wxyz", (frame_count, 4)),
            ("aperture_m", (frame_count,)),
        ):
            key = f"{side}_{suffix}"
            value = np.asarray(arrays[key])
            if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
                raise ParallelJawRetargetError(
                    f"{key} must be floating with shape {shape}"
                )
            if not np.isfinite(value).all():
                raise ParallelJawRetargetError(f"{key} contains non-finite values")
        quaternions = np.asarray(arrays[f"{side}_wxyz"], dtype=np.float64)
        if not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-5):
            raise ParallelJawRetargetError(f"{side}_wxyz contains non-unit quaternions")
        if np.any(np.asarray(arrays[f"{side}_aperture_m"]) < 0.0):
            raise ParallelJawRetargetError(
                f"{side}_aperture_m contains negative widths"
            )
    return frame_count


def validate_parallel_jaw_file(
    path: str | Path, *, expected_frames: int | None = None
) -> int:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ParallelJawRetargetError(
            f"Cannot read parallel-jaw archive {path}: {exc}"
        ) from exc
    return validate_parallel_jaw_arrays(arrays, expected_frames=expected_frames)


def _write_npz_temporary(directory: Path, arrays: dict[str, np.ndarray]) -> Path:
    file_descriptor, name = tempfile.mkstemp(
        dir=directory, prefix=".parallel-jaw-", suffix=".partial.npz"
    )
    temporary = Path(name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json_temporary(directory: Path, metadata: dict[str, Any]) -> Path:
    file_descriptor, name = tempfile.mkstemp(
        dir=directory, prefix=".parallel-jaw-", suffix=".partial.json"
    )
    temporary = Path(name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def execute(
    *,
    tracking: str | Path,
    output_dir: str | Path,
    world_to_camera: str | Path | None = None,
    overwrite: bool = False,
    position_smoother: ArraySmoother | None = None,
    width_smoother: ArraySmoother | None = None,
    orientation_smoother: ArraySmoother | None = None,
) -> dict[str, Any]:
    """Run retargeting and atomically commit NPZ followed by its JSON marker."""

    tracking_path = Path(tracking).expanduser().resolve()
    output_directory = Path(output_dir).expanduser().resolve()
    trajectory_path = output_directory / TRAJECTORY_FILENAME
    metadata_path = output_directory / METADATA_FILENAME
    calibration_path = (
        Path(world_to_camera).expanduser().resolve()
        if world_to_camera is not None
        else None
    )
    targets = (trajectory_path, metadata_path)
    if any(path in {tracking_path, calibration_path} for path in targets):
        raise ParallelJawRetargetError(
            "Output paths must be distinct from immutable source inputs"
        )
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )

    tracking_artifact = _artifact(tracking_path)
    calibration_artifact = (
        _artifact(calibration_path) if calibration_path is not None else None
    )
    implementation_paths = {
        "inpainting/adapters/parallel_jaw_from_tracking.py": Path(__file__).resolve(),
    }
    implementation_artifacts = {
        name: _artifact(path) for name, path in implementation_paths.items()
    }
    arrays, frame_count = load_tracking(tracking_path)
    coordinate_frame = _scalar_text(arrays["coordinate_frame"], "coordinate_frame")
    transform_batch: np.ndarray | None = None
    transform_broadcast = False
    if calibration_path is not None:
        transform_batch, transform_broadcast = load_world_to_camera(
            calibration_path, frame_count=frame_count
        )
    if coordinate_frame == "camera" and transform_batch is None:
        raise ParallelJawRetargetError(
            "Camera-frame tracking requires --world-to-camera"
        )

    output_arrays, diagnostics = retarget_tracking_arrays(
        arrays,
        world_to_camera=transform_batch,
        position_smoother=position_smoother,
        width_smoother=width_smoother,
        orientation_smoother=orientation_smoother,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_npz: Path | None = None
    temporary_json: Path | None = None
    try:
        temporary_npz = _write_npz_temporary(output_directory, output_arrays)
        validate_parallel_jaw_file(temporary_npz, expected_frames=frame_count)
        output_artifact = {
            "filename": trajectory_path.name,
            "size_bytes": temporary_npz.stat().st_size,
            "sha256": _sha256(temporary_npz),
        }
        metadata: dict[str, Any] = {
            "schema_version": PARALLEL_JAW_RUN_SCHEMA,
            "state": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "adapter": "parallel_jaw_from_tracking",
            "tracker": _scalar_text(arrays["tracker"], "tracker"),
            "coordinate_frame": "world",
            "frame_count": frame_count,
            "all_frames_valid": True,
            "source": {
                "tracking": tracking_artifact,
                "T_camera_world": calibration_artifact,
                "source_coordinate_frame": coordinate_frame,
                "transform_policy": (
                    "preserve_world_coordinates"
                    if coordinate_frame == "world"
                    else "invert_world_to_camera_per_frame"
                ),
                "single_transform_broadcast": transform_broadcast,
            },
            "algorithm": {
                "geometry": {
                    "mano_joint_indices": {
                        "thumb_tip": THUMB_TIP,
                        "index_mcp": INDEX_MCP,
                        "index_tip": INDEX_TIP,
                        "middle_tip": MIDDLE_TIP,
                    },
                    "position": "midpoint(thumb_tip,middle_tip)",
                    "aperture": "distance(thumb_tip,index_tip)",
                    "orientation_columns": [
                        "thumb_tip-index_tip",
                        "cross(-index_mcp_axis,x)",
                        "orthogonalized_z",
                    ],
                    "robot_mount_rotation": None,
                },
                "smoothing": {
                    "position": "GaussianProcessRegressor(RBF(1)+WhiteKernel(1), normalize_y=True)",
                    "aperture": "GaussianProcessRegressor(RBF(1)+WhiteKernel(1), normalize_y=True)",
                    "orientation": {
                        "method": "Gaussian-weighted iterative SLERP",
                        "sigma": ORIENTATION_SIGMA,
                        "kernel_size": ORIENTATION_KERNEL_SIZE,
                    },
                },
                "grasp_span_cap_fraction": GRASP_SPAN_FRACTION,
            },
            "diagnostics": diagnostics,
            "implementation_sources": implementation_artifacts,
            "output": {"trajectory": output_artifact},
        }
        temporary_json = _write_json_temporary(output_directory, metadata)

        _assert_artifact_unchanged("tracking.npz", tracking_path, tracking_artifact)
        if calibration_path is not None and calibration_artifact is not None:
            _assert_artifact_unchanged(
                "T_camera_world", calibration_path, calibration_artifact
            )
        for name, path in implementation_paths.items():
            _assert_artifact_unchanged(name, path, implementation_artifacts[name])
        os.replace(temporary_npz, trajectory_path)
        temporary_npz = None
        os.replace(temporary_json, metadata_path)
        temporary_json = None
    finally:
        if temporary_npz is not None:
            temporary_npz.unlink(missing_ok=True)
        if temporary_json is not None:
            temporary_json.unlink(missing_ok=True)
    return metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking",
        required=True,
        type=Path,
        help="Common GT, Video2Data, or Phantom tracking.npz.",
    )
    parser.add_argument(
        "--world-to-camera",
        type=Path,
        help=(
            "T_camera_world world-to-camera .npy/.npz. Required for "
            "camera-frame tracking and inverted per frame; world tracks are preserved."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing trajectory and commit marker.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metadata = execute(
        tracking=args.tracking,
        world_to_camera=args.world_to_camera,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        f"Retargeted {metadata['tracker']} ({metadata['frame_count']} frames) -> "
        f"{Path(args.output_dir).resolve() / TRAJECTORY_FILENAME}"
    )


if __name__ == "__main__":
    main()
