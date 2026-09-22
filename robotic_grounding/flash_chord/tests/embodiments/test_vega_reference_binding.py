# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-schema tests for binding named Vega trajectories to simulation topology."""

from dataclasses import replace

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.sequence_data]

_REFERENCE_PATH = (
    "src/flash_chord/assets/human_motion_data/arctic/arctic_processed/"
    "sequence_id=dataset_s01_mixer_use_01/robot_name=vega_sharpa/data.parquet"
)


@pytest.fixture(scope="module")
def vega():
    import newton
    from flash_chord.embodiments.vega_sharpa import VegaSharpa

    robot = VegaSharpa()
    builder = newton.ModelBuilder()
    layout = robot.build(builder)
    return robot, builder, layout


@pytest.fixture(scope="module")
def native_reference():
    from flash_chord.data import load_reference

    return load_reference(_REFERENCE_PATH)


@pytest.fixture(scope="module")
def half_speed_reference():
    from flash_chord.data import load_reference

    return load_reference(_REFERENCE_PATH, control_fps=20.0, motion_speed=0.5)


def _frames(reference, frame_ids):
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    return replace(
        reference,
        _robot_joint_pos=np.ascontiguousarray(reference.robot_joint_pos()[frame_ids]),
        _robot_root_pos_w=np.ascontiguousarray(reference.robot_root_pos_w()[frame_ids]),
        _robot_root_quat_w=np.ascontiguousarray(reference.robot_root_quat_w()[frame_ids]),
        _robot_frame_pos_w=np.ascontiguousarray(reference.robot_frame_pos_w()[frame_ids]),
        _robot_frame_quat_w=np.ascontiguousarray(reference.robot_frame_quat_w()[frame_ids]),
    )


def _assert_same_binding(left, right):
    np.testing.assert_array_equal(left.joint_q, right.joint_q)
    np.testing.assert_array_equal(left.joint_target, right.joint_target)
    for side in ("left", "right"):
        left_hand = left.hand(side)
        right_hand = right.hand(side)
        for name in (
            "wrist_pos_w",
            "wrist_quat_w",
            "arm_joint_pos",
            "finger_joint_pos",
            "dp_pos_w",
            "dp_quat_w",
            "fingertip_pos_w",
            "fingertip_quat_w",
        ):
            np.testing.assert_array_equal(getattr(left_hand, name), getattr(right_hand, name))


def test_real_named_joint_binding_is_permutation_invariant(vega, native_reference):
    robot, builder, layout = vega
    reference = _frames(native_reference, [0, 100, native_reference.num_frames - 1])
    expected = robot.bind_reference(builder, layout, reference)

    permutation = np.arange(len(reference.robot_joint_names()))[::-1]
    permuted = replace(
        reference,
        _robot_joint_names=[reference.robot_joint_names()[index] for index in permutation],
        _robot_joint_pos=np.ascontiguousarray(reference.robot_joint_pos()[:, permutation]),
    )
    actual = robot.bind_reference(builder, layout, permuted)

    _assert_same_binding(actual, expected)


def test_real_named_joint_binding_preserves_source_q_and_bounds_controller_targets(
    vega,
    native_reference,
):
    from flash_chord.embodiments.vega_sharpa import VegaSharpa, VegaSharpaConfig

    robot, builder, layout = vega
    binding = robot.bind_reference(builder, layout, native_reference)
    source_index = {name: index for index, name in enumerate(native_reference.robot_joint_names())}
    source = np.asarray(native_reference.robot_joint_pos(), dtype=np.float32)
    expected_q = source[:, [source_index[name] for name in layout.scalar_joints.names]]
    bound_q = binding.joint_q[:, layout.scalar_joints.q_ids]
    bound_target = binding.joint_target[:, layout.scalar_joints.dof_ids]
    lower = np.asarray(builder.joint_limit_lower, dtype=np.float32)[list(layout.scalar_joints.dof_ids)]
    upper = np.asarray(builder.joint_limit_upper, dtype=np.float32)[list(layout.scalar_joints.dof_ids)]

    np.testing.assert_array_equal(bound_q, expected_q)
    assert np.all(bound_target >= lower[None, :])
    assert np.all(bound_target <= upper[None, :])
    limit_adjustment = np.abs(bound_target - bound_q)
    assert np.max(limit_adjustment) > 0.0
    assert np.max(limit_adjustment) <= robot.config.joint_limit_tolerance

    strict = VegaSharpa(
        replace(
            VegaSharpaConfig(),
            joint_limit_tolerance=1.0e-7,
            validate_named_frames=False,
        )
    )
    with pytest.raises(ValueError, match="named joint .* outside its .* limit"):
        strict.bind_reference(builder, layout, native_reference)


def test_fixed_root_validation_rejects_motion_and_nonunit_orientation(vega, native_reference):
    robot, builder, layout = vega
    reference = _frames(native_reference, [0, native_reference.num_frames - 1])

    root_position = reference.robot_root_pos_w().copy()
    root_position[1, 0] = 1.0e-3
    with pytest.raises(ValueError, match="fixed-base but the reference root moves"):
        robot.bind_reference(builder, layout, replace(reference, _robot_root_pos_w=root_position))

    root_orientation = reference.robot_root_quat_w().copy()
    root_orientation[1] *= 2.0
    with pytest.raises(ValueError, match="non-unit quaternions"):
        robot.bind_reference(builder, layout, replace(reference, _robot_root_quat_w=root_orientation))


def test_native_frame_validation_is_strict(vega, native_reference):
    robot, builder, layout = vega
    reference = _frames(native_reference, [0, 100, native_reference.num_frames - 1])
    frame_position = reference.robot_frame_pos_w().copy()
    palm = reference.robot_frame_names().index("left_hand_C_MC")
    frame_position[1, palm, 0] += 1.0e-3

    with pytest.raises(ValueError, match="native named frame 'left_hand_C_MC'.*0.0001 m"):
        robot.bind_reference(builder, layout, replace(reference, _robot_frame_pos_w=frame_position))


def test_resampled_frame_validation_separates_endpoints_from_health_bound(vega, half_speed_reference):
    robot, builder, layout = vega
    reference = _frames(half_speed_reference, [0, 78, half_speed_reference.num_frames - 1])
    assert reference.metadata.is_resampled
    robot.bind_reference(builder, layout, reference)

    frame_position = reference.robot_frame_pos_w().copy()
    fingertip = reference.robot_frame_names().index("left_pinky_fingertip")
    frame_position[1, fingertip, 0] += 5.0e-2
    with pytest.raises(ValueError, match="interpolated named frame 'left_pinky_fingertip'.*0.02 m"):
        robot.bind_reference(builder, layout, replace(reference, _robot_frame_pos_w=frame_position))


def test_named_frame_diagnostics_can_be_explicitly_disabled(vega, native_reference):
    from flash_chord.embodiments.vega_sharpa import VegaSharpa, VegaSharpaConfig

    _, builder, layout = vega
    reference = _frames(native_reference, [0, native_reference.num_frames - 1])
    reference_without_frames = replace(
        reference,
        _robot_frame_names=[],
        _robot_frame_pos_w=np.empty((reference.num_frames, 0, 3), dtype=np.float32),
        _robot_frame_quat_w=np.empty((reference.num_frames, 0, 4), dtype=np.float32),
    )
    robot = VegaSharpa(replace(VegaSharpaConfig(), validate_named_frames=False))

    binding = robot.bind_reference(builder, layout, reference_without_frames)
    assert binding.joint_q.shape == (2, 58)
