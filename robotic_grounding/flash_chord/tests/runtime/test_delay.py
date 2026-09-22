# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for per-world delayed control targets."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_per_world_delay_supports_noncontiguous_dofs_and_immediate_targets():
    import warp as wp

    from flash_chord.runtime.delay import TargetDelayBuffer, TargetDelayConfig

    sequence = [
        [[10.0, 11.0, 100.0], [20.0, 21.0, 200.0]],
        [[12.0, 13.0, 101.0], [22.0, 23.0, 201.0]],
        [[14.0, 15.0, 102.0], [24.0, 25.0, 202.0]],
        [[16.0, 17.0, 103.0], [26.0, 27.0, 203.0]],
    ]
    with wp.ScopedDevice("cuda:0"):
        delay = TargetDelayBuffer.build(
            world_count=2,
            num_joint_dof=3,
            delayed_dof_ids=(0, 2),
            config=TargetDelayConfig(min_steps=0, max_steps=2),
        )
        delay.reset(wp.ones(2, dtype=wp.int32))
        delay.delay_steps.assign(np.array([0, 2], dtype=np.int32))
        outputs = []
        for target in sequence:
            target_d = wp.array(np.asarray(target, dtype=np.float32).reshape(-1), dtype=wp.float32)
            outputs.append(delay.advance(target_d).numpy().reshape(2, 3).copy())

    np.testing.assert_allclose(outputs[0], sequence[0])
    np.testing.assert_allclose(outputs[1][0], sequence[1][0])
    np.testing.assert_allclose(outputs[2][0], sequence[2][0])
    np.testing.assert_allclose(outputs[3][0], sequence[3][0])
    np.testing.assert_allclose(outputs[1][1], [20.0, 23.0, 200.0])
    np.testing.assert_allclose(outputs[2][1], [20.0, 25.0, 200.0])
    np.testing.assert_allclose(outputs[3][1], [22.0, 27.0, 201.0])
    assert delay.delayed_dof_ids == (0, 2)
    assert delay.delayed_slot_by_dof.numpy().tolist() == [0, -1, 1]


def test_delay_can_compose_selected_dofs_into_complete_target():
    import warp as wp

    from flash_chord.runtime.delay import TargetDelayBuffer, TargetDelayConfig

    with wp.ScopedDevice("cuda:0"):
        delay = TargetDelayBuffer.build(
            world_count=1,
            num_joint_dof=4,
            delayed_dof_ids=(1, 3),
            config=TargetDelayConfig(min_steps=1, max_steps=1),
        )
        delay.reset(wp.ones(1, dtype=wp.int32))
        output = wp.full(4, 9.0, dtype=wp.float32)
        delay.advance_into(wp.array([0.0, 1.0, 2.0, 3.0], dtype=wp.float32), output)
        delay.advance_into(wp.array([4.0, 5.0, 6.0, 7.0], dtype=wp.float32), output)

    np.testing.assert_array_equal(output.numpy(), [9.0, 1.0, 9.0, 3.0])


def test_masked_reset_restarts_only_selected_delay_stream():
    import warp as wp

    from flash_chord.runtime.delay import TargetDelayBuffer, TargetDelayConfig

    with wp.ScopedDevice("cuda:0"):
        delay = TargetDelayBuffer.build(
            world_count=2,
            num_joint_dof=1,
            delayed_dof_ids=(0,),
            config=TargetDelayConfig(min_steps=1, max_steps=1),
        )
        delay.reset(wp.ones(2, dtype=wp.int32))
        delay.advance(wp.array([1.0, 10.0], dtype=wp.float32))
        before_reset = delay.advance(wp.array([2.0, 20.0], dtype=wp.float32)).numpy().copy()
        delay.reset(wp.array([0, 1], dtype=wp.int32))
        after_reset = delay.advance(wp.array([3.0, 30.0], dtype=wp.float32)).numpy().copy()

    np.testing.assert_allclose(before_reset, [1.0, 10.0])
    np.testing.assert_allclose(after_reset, [2.0, 30.0])
    assert delay.num_pushes.numpy().tolist() == [2, 1]


def test_delay_config_rejects_invalid_ranges():
    from flash_chord.runtime.delay import TargetDelayConfig

    with pytest.raises(ValueError, match="non-negative"):
        TargetDelayConfig(min_steps=-1, max_steps=0)
    with pytest.raises(ValueError, match="must be >="):
        TargetDelayConfig(min_steps=2, max_steps=1)


@pytest.mark.parametrize(
    ("delayed_dof_ids", "message"),
    [
        ((), "at least one"),
        ((1, 1), "must be unique"),
        ((-1,), r"must be in \[0, 3\)"),
        ((3,), r"must be in \[0, 3\)"),
    ],
)
def test_delay_rejects_invalid_dof_mappings(delayed_dof_ids, message):
    from flash_chord.runtime.delay import TargetDelayBuffer, TargetDelayConfig

    with pytest.raises(ValueError, match=message):
        TargetDelayBuffer.build(
            world_count=1,
            num_joint_dof=3,
            delayed_dof_ids=delayed_dof_ids,
            config=TargetDelayConfig(),
        )
