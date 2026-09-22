# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time resampling shared by reference-schema loaders."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resample_times(num_frames: int, source_fps: float, target_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return uniform source and target timestamps spanning the same motion duration."""
    _validate_timing(num_frames, source_fps, target_fps)
    duration = (num_frames - 1) / source_fps
    num_target_frames = int(round(duration * target_fps)) + 1
    source_times = np.arange(num_frames, dtype=np.float64) / source_fps
    target_times = np.clip(np.arange(num_target_frames, dtype=np.float64) / target_fps, 0.0, duration)
    return source_times, target_times


def playback_times(
    num_frames: int,
    source_fps: float,
    control_fps: float,
    motion_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source timestamps sampled once per control step at ``motion_speed``."""
    _validate_timing(num_frames, source_fps, control_fps)
    if not np.isfinite(motion_speed) or motion_speed <= 0.0:
        raise ValueError(f"motion_speed must be finite and positive, got {motion_speed}")
    if motion_speed == 1.0:
        return resample_times(num_frames, source_fps, control_fps)

    duration = (num_frames - 1) / source_fps
    sample_count = int(duration * control_fps / motion_speed)
    if sample_count < 2:
        raise ValueError(
            f"motion_speed {motion_speed} produces {sample_count} samples; at least two are required"
        )
    source_times = np.arange(num_frames, dtype=np.float64) / source_fps
    return source_times, np.linspace(0.0, duration, sample_count)


def lerp(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate an array along its leading time axis."""
    values = np.asarray(values)
    flat = values.reshape(values.shape[0], -1)
    result = np.stack(
        [np.interp(target_times, source_times, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return result.reshape((len(target_times),) + values.shape[1:])


def nearest(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Sample discrete values from the nearest source timestamp."""
    values = np.asarray(values)
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(source_times[right] - target_times) < np.abs(target_times - source_times[left])
    indices = np.where(choose_right, right, left)
    return values[indices]


def slerp_wxyz(quaternions: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Spherically interpolate one ``(T, 4)`` quaternion track in ``wxyz`` order."""
    quaternions = np.asarray(quaternions)
    if quaternions.shape != (len(source_times), 4):
        raise ValueError(
            f"quaternions must have shape ({len(source_times)}, 4), got {quaternions.shape}"
        )
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    return Slerp(source_times, rotations)(target_times).as_quat()[:, [3, 0, 1, 2]]


def slerp_tracks_wxyz(
    quaternions: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Spherically interpolate every quaternion track in a ``(T, ..., 4)`` array."""
    quaternions = np.asarray(quaternions)
    if quaternions.ndim < 2 or quaternions.shape[0] != len(source_times) or quaternions.shape[-1] != 4:
        raise ValueError(
            "quaternions must have shape "
            f"({len(source_times)}, ..., 4), got {quaternions.shape}"
        )
    flat = quaternions.reshape(len(source_times), -1, 4)
    result = np.stack(
        [slerp_wxyz(flat[:, track], source_times, target_times) for track in range(flat.shape[1])],
        axis=1,
    )
    return result.reshape((len(target_times),) + quaternions.shape[1:])


def _validate_timing(num_frames: int, source_fps: float, target_fps: float) -> None:
    if isinstance(num_frames, bool) or not isinstance(num_frames, (int, np.integer)) or num_frames < 2:
        raise ValueError(f"num_frames must be an integer of at least two, got {num_frames!r}")
    for name, value in (("source_fps", source_fps), ("target_fps", target_fps)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
