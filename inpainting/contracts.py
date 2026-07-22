"""Versioned file contracts shared by tracking, inpainting, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TRACKING_SCHEMA = "v2d.inpainting.tracking/v1"
ROBOT_TRAJECTORY_SCHEMA = "v2d.inpainting.robot-trajectory/v1"
EXPERIMENT_SCHEMA = "v2d.inpainting.experiment/v1"
RESOLVED_EXPERIMENT_SCHEMA = "v2d.inpainting.resolved-experiment/v1"
TRACKERS = ("phantom", "v2d", "ground_truth")
COORDINATE_FRAMES = ("camera", "world")


class ContractError(ValueError):
    """Raised when a stage output cannot be consumed safely."""


@dataclass(frozen=True)
class VideoGeometry:
    """Decoded geometry that every frame-aligned artifact must preserve."""

    frame_count: int
    width: int
    height: int
    fps: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


def _scalar_text(value: np.ndarray | Any, key: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ContractError(
            f"{key} must be a scalar string, got shape {array.shape} and dtype {array.dtype}"
        )
    value = array.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractError(f"{key} must contain valid UTF-8 text") from exc
    result = str(value)
    if not result:
        raise ContractError(f"{key} must not be empty")
    return result


def _require_shape(
    data: dict[str, np.ndarray], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    if key not in data:
        raise ContractError(f"Missing required array: {key}")
    array = np.asarray(data[key])
    if array.shape != shape:
        raise ContractError(f"{key} must have shape {shape}, got {array.shape}")
    return array


def _require_bool_array(
    data: dict[str, np.ndarray], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = _require_shape(data, key, shape)
    if array.dtype != np.dtype(np.bool_):
        raise ContractError(f"{key} must have boolean dtype, got {array.dtype}")
    return array


def _require_float_array(
    data: dict[str, np.ndarray], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = _require_shape(data, key, shape)
    if not np.issubdtype(array.dtype, np.floating):
        raise ContractError(f"{key} must have floating-point dtype, got {array.dtype}")
    return array


def _require_string_array(
    data: dict[str, np.ndarray], key: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = _require_shape(data, key, shape)
    if array.dtype.kind not in {"U", "S"}:
        raise ContractError(f"{key} must have string dtype, got {array.dtype}")
    return array


def _require_finite_valid_rows(array: np.ndarray, valid: np.ndarray, key: str) -> None:
    if valid.any() and not np.isfinite(array[valid]).all():
        raise ContractError(f"{key} contains non-finite values in valid rows")


def validate_tracking_arrays(
    data: dict[str, np.ndarray], expected_frames: int | None = None
) -> int:
    """Validate a loaded tracking archive and return its frame count."""

    required_scalars = ("schema_version", "tracker", "coordinate_frame")
    for key in required_scalars:
        if key not in data:
            raise ContractError(f"Missing required tracking value: {key}")

    schema = _scalar_text(data["schema_version"], "schema_version")
    tracker = _scalar_text(data["tracker"], "tracker")
    coordinate_frame = _scalar_text(data["coordinate_frame"], "coordinate_frame")
    if schema != TRACKING_SCHEMA:
        raise ContractError(f"Unsupported tracking schema {schema!r}; expected {TRACKING_SCHEMA!r}")
    if tracker not in TRACKERS:
        raise ContractError(f"Unsupported tracker {tracker!r}; expected one of {TRACKERS}")
    if coordinate_frame not in COORDINATE_FRAMES:
        raise ContractError(
            f"Unsupported coordinate frame {coordinate_frame!r}; expected one of {COORDINATE_FRAMES}"
        )

    if "frame_indices" not in data or np.asarray(data["frame_indices"]).ndim != 1:
        raise ContractError("frame_indices must be a one-dimensional array")
    frame_indices = np.asarray(data["frame_indices"])
    if not np.issubdtype(frame_indices.dtype, np.integer):
        raise ContractError(f"frame_indices must have integer dtype, got {frame_indices.dtype}")
    frame_count = int(frame_indices.shape[0])
    if expected_frames is not None and frame_count != expected_frames:
        raise ContractError(
            f"Tracking has {frame_count} frames but source video has {expected_frames}"
        )
    if not np.array_equal(frame_indices, np.arange(frame_count)):
        raise ContractError("Initial experiment requires contiguous frame_indices 0..N-1")

    for side in ("left", "right"):
        valid = _require_bool_array(data, f"{side}_valid", (frame_count,))
        positions = _require_float_array(
            data, f"{side}_wrist_position", (frame_count, 3)
        )
        quaternions = _require_float_array(
            data, f"{side}_wrist_wxyz", (frame_count, 4)
        )
        _require_finite_valid_rows(positions, valid, f"{side}_wrist_position")
        _require_finite_valid_rows(quaternions, valid, f"{side}_wrist_wxyz")
        if valid.any():
            norms = np.linalg.norm(np.asarray(quaternions[valid], dtype=np.float64), axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ContractError(f"{side}_wrist_wxyz contains non-unit valid quaternions")
        for suffix, trailing in (("joints_3d", (21, 3)), ("joints_2d", (21, 2))):
            key = f"{side}_{suffix}"
            if key in data:
                joints = _require_float_array(data, key, (frame_count, *trailing))
                _require_finite_valid_rows(joints, valid, key)
        joint_quaternion_key = f"{side}_joints_wxyz"
        if joint_quaternion_key in data:
            joint_quaternions = _require_float_array(
                data, joint_quaternion_key, (frame_count, 21, 4)
            )
            _require_finite_valid_rows(
                joint_quaternions, valid, joint_quaternion_key
            )
            if valid.any() and not np.allclose(
                np.linalg.norm(
                    np.asarray(joint_quaternions[valid], dtype=np.float64), axis=2
                ),
                1.0,
                atol=1e-3,
            ):
                raise ContractError(
                    f"{joint_quaternion_key} contains non-unit valid quaternions"
                )

        finger_key = f"{side}_finger_joints"
        names_key = f"{side}_finger_joint_names"
        if finger_key in data or names_key in data:
            if finger_key not in data or names_key not in data:
                raise ContractError(f"{finger_key} and {names_key} must be supplied together")
            fingers = np.asarray(data[finger_key])
            if fingers.ndim != 2 or fingers.shape[0] != frame_count:
                raise ContractError(f"{finger_key} must have shape (N,J)")
            if not np.issubdtype(fingers.dtype, np.floating):
                raise ContractError(
                    f"{finger_key} must have floating-point dtype, got {fingers.dtype}"
                )
            _require_string_array(data, names_key, (fingers.shape[1],))
            _require_finite_valid_rows(fingers, valid, finger_key)
    return frame_count


def validate_tracking_file(path: str | Path, expected_frames: int | None = None) -> int:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return validate_tracking_arrays(dict(archive), expected_frames=expected_frames)


def validate_mask_array(masks: np.ndarray, geometry: VideoGeometry) -> np.ndarray:
    """Validate an exact boolean arm/robot mask."""

    masks = np.asarray(masks)
    expected = (geometry.frame_count, geometry.height, geometry.width)
    if masks.shape != expected:
        raise ContractError(f"Mask must have shape {expected}, got {masks.shape}")
    if masks.dtype != np.dtype(np.bool_):
        raise ContractError(f"Mask must have boolean dtype, got {masks.dtype}")
    return masks


def validate_mask_file(path: str | Path, geometry: VideoGeometry) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return validate_mask_array(np.load(path, mmap_mode="r"), geometry)


def validate_depth_array(
    depth: np.ndarray,
    mask: np.ndarray,
    geometry: VideoGeometry,
    *,
    name: str = "Depth",
) -> np.ndarray:
    """Validate metric camera-z depth paired exactly with a boolean mask.

    Valid pixels must be finite and strictly positive. Pixels outside the mask
    must be positive infinity. Validation is frame-wise so large memory-mapped
    1080p sequences do not require a second full-size allocation.
    """

    depth = np.asarray(depth)
    mask = validate_mask_array(mask, geometry)
    expected = (geometry.frame_count, geometry.height, geometry.width)
    if depth.shape != expected:
        raise ContractError(f"{name} must have shape {expected}, got {depth.shape}")
    if depth.dtype != np.dtype(np.float32):
        raise ContractError(f"{name} must have float32 dtype, got {depth.dtype}")
    for frame_index in range(geometry.frame_count):
        frame_depth = depth[frame_index]
        frame_mask = mask[frame_index]
        valid_values = frame_depth[frame_mask]
        if valid_values.size and (
            not np.isfinite(valid_values).all() or np.any(valid_values <= 0.0)
        ):
            raise ContractError(
                f"{name} frame {frame_index} has non-finite or non-positive masked values"
            )
        if not np.isposinf(frame_depth[~frame_mask]).all():
            raise ContractError(
                f"{name} frame {frame_index} must be +inf outside its mask"
            )
    return depth


def validate_depth_file(
    path: str | Path,
    mask: np.ndarray,
    geometry: VideoGeometry,
    *,
    name: str = "Depth",
) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return validate_depth_array(
        np.load(path, mmap_mode="r"), mask, geometry, name=name
    )


def validate_robot_trajectory_arrays(
    data: dict[str, np.ndarray], expected_frames: int | None = None
) -> int:
    """Validate Sharpa wrist/finger targets before Vega arm IK and rendering."""

    for key in ("schema_version", "coordinate_frame", "robot", "gripper"):
        if key not in data:
            raise ContractError(f"Missing required robot trajectory value: {key}")
    schema = _scalar_text(data["schema_version"], "schema_version")
    if schema != ROBOT_TRAJECTORY_SCHEMA:
        raise ContractError(
            f"Unsupported robot trajectory schema {schema!r}; expected {ROBOT_TRAJECTORY_SCHEMA!r}"
        )
    coordinate_frame = _scalar_text(data["coordinate_frame"], "coordinate_frame")
    if coordinate_frame not in COORDINATE_FRAMES:
        raise ContractError(f"Unsupported coordinate frame: {coordinate_frame!r}")
    _scalar_text(data["robot"], "robot")
    _scalar_text(data["gripper"], "gripper")
    if "frame_indices" not in data or np.asarray(data["frame_indices"]).ndim != 1:
        raise ContractError("frame_indices must be a one-dimensional array")
    frame_indices = np.asarray(data["frame_indices"])
    if not np.issubdtype(frame_indices.dtype, np.integer):
        raise ContractError(
            f"Robot trajectory frame_indices must have integer dtype, got {frame_indices.dtype}"
        )
    frame_count = int(frame_indices.shape[0])
    if expected_frames is not None and frame_count != expected_frames:
        raise ContractError(
            f"Robot trajectory has {frame_count} frames but expected {expected_frames}"
        )
    if not np.array_equal(frame_indices, np.arange(frame_count)):
        raise ContractError("Robot trajectory frame_indices must be contiguous 0..N-1")
    for side in ("left", "right"):
        valid = _require_bool_array(data, f"{side}_valid", (frame_count,))
        positions = _require_float_array(
            data, f"{side}_wrist_position", (frame_count, 3)
        )
        quat = _require_float_array(data, f"{side}_wrist_wxyz", (frame_count, 4))
        _require_finite_valid_rows(positions, valid, f"{side}_wrist_position")
        _require_finite_valid_rows(quat, valid, f"{side}_wrist_wxyz")
        finger_key = f"{side}_finger_joints"
        names_key = f"{side}_finger_joint_names"
        if finger_key not in data:
            raise ContractError(f"{finger_key} must have shape (N,J)")
        fingers = np.asarray(data[finger_key])
        if fingers.ndim != 2:
            raise ContractError(f"{finger_key} must have shape (N,J)")
        if fingers.shape[0] != frame_count:
            raise ContractError(f"{finger_key} must have {frame_count} rows")
        if not np.issubdtype(fingers.dtype, np.floating):
            raise ContractError(
                f"{finger_key} must have floating-point dtype, got {fingers.dtype}"
            )
        _require_string_array(data, names_key, (fingers.shape[1],))
        _require_finite_valid_rows(fingers, valid, finger_key)
        if valid.any() and not np.allclose(
            np.linalg.norm(np.asarray(quat[valid], dtype=np.float64), axis=1),
            1.0,
            atol=1e-3,
        ):
            raise ContractError(f"{side}_wrist_wxyz contains non-unit valid quaternions")
    return frame_count


def validate_robot_trajectory_file(
    path: str | Path, expected_frames: int | None = None
) -> int:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return validate_robot_trajectory_arrays(dict(archive), expected_frames=expected_frames)
