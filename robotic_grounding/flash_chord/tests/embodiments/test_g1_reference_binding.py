# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real snack-sequence binding checks for floating G1+Dex3."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.sequence_data]

_REFERENCE_PATH = (
    "src/flash_chord/assets/human_motion_data/whole_body/soma/"
    "sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01/robot_name=g1/data.parquet"
)


def _sample(reference):
    frame_ids = np.asarray([0, reference.num_frames // 2, reference.num_frames - 1], dtype=np.int64)
    return replace(
        reference,
        _robot_joint_pos=np.ascontiguousarray(reference.robot_joint_pos()[frame_ids]),
        _robot_root_pos_w=np.ascontiguousarray(reference.robot_root_pos_w()[frame_ids]),
        _robot_root_quat_w=np.ascontiguousarray(reference.robot_root_quat_w()[frame_ids]),
        _robot_frame_pos_w=np.ascontiguousarray(reference.robot_frame_pos_w()[frame_ids]),
        _robot_frame_quat_w=np.ascontiguousarray(reference.robot_frame_quat_w()[frame_ids]),
    )


def test_real_snack_reference_binds_named_joints_floating_root_and_palms():
    import newton

    from flash_chord.data import load_reference
    from flash_chord.embodiments.g1_dex3 import G1Dex3

    reference = _sample(load_reference(_REFERENCE_PATH))
    robot = G1Dex3()
    builder = newton.ModelBuilder()
    layout = robot.build(builder)
    binding = robot.bind_reference(builder, layout, reference)

    source_index = {name: index for index, name in enumerate(reference.robot_joint_names())}
    expected_scalar_q = reference.robot_joint_pos()[:, [source_index[name] for name in layout.scalar_joints.names]]
    np.testing.assert_array_equal(binding.joint_q[:, layout.scalar_joints.q_ids], expected_scalar_q)
    np.testing.assert_array_equal(binding.joint_q[:, :3], reference.robot_root_pos_w())
    np.testing.assert_allclose(
        binding.joint_q[:, 3:7],
        reference.robot_root_quat_w()[:, (1, 2, 3, 0)],
        atol=1.0e-7,
    )
    free_root_target = np.broadcast_to(
        np.asarray(builder.joint_target_pos, dtype=np.float32)[:6],
        (reference.num_frames, 6),
    )
    np.testing.assert_array_equal(binding.joint_target[:, :6], free_root_target)

    for side in layout.sides:
        frame_id = reference.robot_frame_names().index(f"{side}_hand_palm_link")
        actual = binding.hand(side)
        np.testing.assert_allclose(actual.wrist_pos_w, reference.robot_frame_pos_w()[:, frame_id], atol=1.0e-6)
        actual_quat = actual.wrist_quat_w / np.linalg.norm(actual.wrist_quat_w, axis=1, keepdims=True)
        expected_quat = reference.robot_frame_quat_w()[:, frame_id]
        expected_quat = expected_quat / np.linalg.norm(expected_quat, axis=1, keepdims=True)
        assert np.min(np.abs(np.sum(actual_quat * expected_quat, axis=1))) > 1.0 - 1.0e-7


def test_real_snack_reference_binding_is_permutation_invariant():
    import newton

    from flash_chord.data import load_reference
    from flash_chord.embodiments.g1_dex3 import G1Dex3

    reference = _sample(load_reference(_REFERENCE_PATH))
    robot = G1Dex3()
    builder = newton.ModelBuilder()
    layout = robot.build(builder)
    expected = robot.bind_reference(builder, layout, reference)

    permutation = np.arange(len(reference.robot_joint_names()))[::-1]
    permuted = replace(
        reference,
        _robot_joint_names=[reference.robot_joint_names()[index] for index in permutation],
        _robot_joint_pos=np.ascontiguousarray(reference.robot_joint_pos()[:, permutation]),
    )
    actual = robot.bind_reference(builder, layout, permuted)

    np.testing.assert_array_equal(actual.joint_q, expected.joint_q)
    np.testing.assert_array_equal(actual.joint_target, expected.joint_target)
