# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for schema-independent reference resampling."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from flash_chord.data.resampling import (
    lerp,
    nearest,
    playback_times,
    resample_times,
    slerp_tracks_wxyz,
    slerp_wxyz,
)


def test_lerp_linear_ramp_is_exact():
    source_times, target_times = resample_times(11, 30.0, 20.0)
    values = (2.0 * np.arange(11)).reshape(11, 1)
    result = lerp(values, source_times, target_times)
    np.testing.assert_allclose(result[:, 0], 60.0 * target_times)


def test_nearest_preserves_discrete_values_and_dtype():
    source_times, target_times = resample_times(11, 30.0, 20.0)
    values = np.arange(11, dtype=np.int32).reshape(11, 1)
    result = nearest(values, source_times, target_times)
    assert result.dtype == values.dtype
    assert set(np.unique(result)).issubset(set(range(11)))


def test_slerp_constant_rate_gives_linear_angle():
    angle = 0.1 * np.arange(11)
    quaternions = Rotation.from_rotvec(np.outer(angle, [0.0, 0.0, 1.0])).as_quat()[:, [3, 0, 1, 2]]
    source_times, target_times = resample_times(11, 30.0, 20.0)
    result = slerp_wxyz(quaternions, source_times, target_times)
    rotvec = Rotation.from_quat(result[:, [1, 2, 3, 0]]).as_rotvec()
    np.testing.assert_allclose(rotvec[:, 2], 3.0 * target_times, atol=1.0e-6)
    np.testing.assert_allclose(rotvec[:, :2], 0.0, atol=1.0e-6)


def test_slerp_tracks_preserves_leading_track_shape():
    source_times, target_times = resample_times(11, 30.0, 20.0)
    identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (11, 2, 3, 1))
    result = slerp_tracks_wxyz(identity, source_times, target_times)
    assert result.shape == (len(target_times), 2, 3, 4)
    expected = np.broadcast_to(np.array([1.0, 0.0, 0.0, 0.0]), result.shape)
    np.testing.assert_allclose(result, expected)


def test_half_speed_playback_times_match_existing_schedule():
    source_times, target_times = playback_times(638, 30.0, 20.0, 0.5)
    assert len(source_times) == 638
    assert len(target_times) == 849
    assert target_times[0] == 0.0
    assert target_times[-1] == source_times[-1]


@pytest.mark.parametrize("motion_speed", [0.0, -0.5, np.nan, np.inf])
def test_playback_times_reject_invalid_motion_speed(motion_speed):
    with pytest.raises(ValueError, match="motion_speed must be finite and positive"):
        playback_times(638, 30.0, 20.0, motion_speed)


@pytest.mark.parametrize(
    ("num_frames", "source_fps", "target_fps"),
    [(1, 30.0, 20.0), (10, 0.0, 20.0), (10, 30.0, np.inf)],
)
def test_resample_times_reject_invalid_timing(num_frames, source_fps, target_fps):
    with pytest.raises(ValueError):
        resample_times(num_frames, source_fps, target_fps)
