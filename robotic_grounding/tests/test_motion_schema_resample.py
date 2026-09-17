# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for command-rate resampling of unified motion data."""

from __future__ import annotations

import math

import pytest
import torch
from robotic_grounding.motion_schema import MotionData, resolve_playback_timing


def _motion(num_frames: int = 51, fps: float = 50.0) -> MotionData:
    """Build a minimal motion whose X position equals source time."""
    times = torch.arange(num_frames, dtype=torch.float32) / fps
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_frames, 4).clone()
    return MotionData(
        motion_kind="single_robot",
        fps=fps,
        robot_root_position=torch.stack(
            [times, torch.zeros_like(times), torch.zeros_like(times)], dim=-1
        ),
        robot_root_wxyz=identity,
        robot_joint_positions=times.unsqueeze(-1),
    )


def test_resample_50_hz_to_20_hz_preserves_recorded_duration() -> None:
    """A one-second 50 Hz motion becomes 21 samples on a 20 Hz grid."""
    resampled = _motion().resample(20.0)

    assert resampled.fps == 20.0
    assert resampled.num_frames() == 21
    torch.testing.assert_close(
        resampled.robot_root_position[:, 0],
        torch.arange(21, dtype=torch.float32) / 20.0,
    )
    torch.testing.assert_close(
        resampled.robot_root_position[-1],
        torch.tensor([1.0, 0.0, 0.0]),
    )


def test_resample_to_higher_fps_supports_slow_motion_playback() -> None:
    """A 20 Hz env can consume a 40 Hz reference at half recorded speed."""
    resampled = _motion().resample(40.0)

    assert resampled.num_frames() == 41
    # One resampled frame per 50 ms environment step takes two seconds.
    assert math.isclose((resampled.num_frames() - 1) * 0.05, 2.0)
    torch.testing.assert_close(
        resampled.robot_root_position[-1],
        torch.tensor([1.0, 0.0, 0.0]),
    )


def test_playback_timing_matches_environment_step_and_motion_speed() -> None:
    playback_dt, target_fps = resolve_playback_timing(0.05, 0.05, 1.0)
    assert playback_dt == 0.05
    assert target_fps == 20.0

    _, slow_target_fps = resolve_playback_timing(0.05, 0.05, 0.5)
    assert slow_target_fps == 40.0


def test_playback_timing_rejects_motion_and_environment_step_mismatch() -> None:
    with pytest.raises(ValueError, match=r"motion\.dt .* must equal env\.step_dt"):
        resolve_playback_timing(0.02, 0.05, 1.0)


@pytest.mark.parametrize("motion_speed", [0.0, -1.0, math.inf, math.nan])
def test_playback_timing_rejects_invalid_motion_speed(motion_speed: float) -> None:
    with pytest.raises(ValueError, match="motion.motion_speed"):
        resolve_playback_timing(0.05, 0.05, motion_speed)


def test_resample_same_fps_is_no_op() -> None:
    motion = _motion()

    assert motion.resample(50.0) is motion


def test_resample_uses_shortest_path_slerp_for_quaternions() -> None:
    motion = _motion(num_frames=2, fps=1.0)
    motion.robot_root_wxyz = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    resampled = motion.resample(2.0)

    expected_midpoint = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    torch.testing.assert_close(
        resampled.robot_root_wxyz[1], expected_midpoint, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(resampled.robot_root_wxyz, dim=-1),
        torch.ones(3),
    )


def test_resample_preserves_contact_gaps_and_uses_nearest_discrete_values() -> None:
    motion = _motion(num_frames=3, fps=1.0)
    motion.left_object_contact_positions = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0]],
        ]
    )
    motion.left_object_contact_part_ids = torch.tensor([[1], [2], [3]])
    motion.left_hand_contact_active = torch.tensor([0.0, 1.0, 0.0])

    resampled = motion.resample(2.0)

    torch.testing.assert_close(
        resampled.left_object_contact_positions[:, 0, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 3.0]),
    )
    torch.testing.assert_close(
        resampled.left_object_contact_part_ids[:, 0],
        torch.tensor([1, 2, 2, 3, 3]),
    )
    torch.testing.assert_close(
        resampled.left_hand_contact_active,
        torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize("source_fps", [0.0, -1.0, math.inf, math.nan])
def test_resample_rejects_invalid_source_fps(source_fps: float) -> None:
    with pytest.raises(ValueError, match="MotionData.fps"):
        _motion(fps=source_fps).resample(20.0)


@pytest.mark.parametrize("target_fps", [0.0, -1.0, math.inf, math.nan])
def test_resample_rejects_invalid_target_fps(target_fps: float) -> None:
    with pytest.raises(ValueError, match="target_fps"):
        _motion().resample(target_fps)


def test_resample_rejects_misaligned_time_axis_fields() -> None:
    motion = _motion()
    motion.ee_pos_w = torch.zeros(3, 2, 3)

    with pytest.raises(ValueError, match=r"ee_pos_w has 3 frames; expected 51"):
        motion.resample(20.0)


def test_every_time_axis_field_has_a_resampling_strategy() -> None:
    tensor_fields = {
        *MotionData._LINEAR_TENSOR_FIELDS,
        *MotionData._QUATERNION_TENSOR_FIELDS,
        *MotionData._POSE_TENSOR_FIELDS,
        *MotionData._CONTACT_TENSOR_FIELDS,
        *MotionData._NEAREST_TENSOR_FIELDS,
    }
    tensor_list_fields = {
        *MotionData._LINEAR_TENSOR_LIST_FIELDS,
        *MotionData._POSE_TENSOR_LIST_FIELDS,
        *MotionData._CONTACT_TENSOR_LIST_FIELDS,
        *MotionData._NEAREST_TENSOR_LIST_FIELDS,
    }

    assert tensor_fields == set(MotionData._TIME_AXIS_TENSOR_FIELDS)
    assert tensor_list_fields == set(MotionData._TIME_AXIS_TENSOR_LIST_FIELDS)
