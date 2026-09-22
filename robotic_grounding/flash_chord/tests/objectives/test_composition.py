# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for weighted objective composition."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_default_weights_match_reference_run():
    from flash_chord.objectives.config import ObjectiveConfig

    assert ObjectiveConfig().weights == (
        0.0,
        0.0,
        0.0,
        10.0,
        -1.0,
        -10.0,
        -100.0,
        -0.005,
        -0.002,
        0.0,
        0.0,
    )


def test_weighted_score_has_exact_term_order_signs_and_frame_dt():
    import warp as wp

    from flash_chord.objectives.composition import ObjectiveBuffers
    from flash_chord.objectives.config import (
        ContactSupportObjectiveTermConfig,
        ForceClosureObjectiveTermConfig,
        ObjectiveConfig,
        ObjectiveTermConfig,
        ShapedObjectiveTermConfig,
    )

    weights = np.array(
        [1.0, 2.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, 8.0],
        dtype=np.float32,
    )
    config = ObjectiveConfig(
        object_keypoints=ShapedObjectiveTermConfig(weight=float(weights[0])),
        hand_keypoints=ShapedObjectiveTermConfig(weight=float(weights[1])),
        hand_joint_pos=ShapedObjectiveTermConfig(weight=float(weights[2])),
        contact_wrench_support=ContactSupportObjectiveTermConfig(weight=float(weights[3])),
        missed_contact=ObjectiveTermConfig(weight=float(weights[4])),
        unintended_contact=ObjectiveTermConfig(weight=float(weights[5])),
        termination=ObjectiveTermConfig(weight=float(weights[6])),
        action_rate_l2=ObjectiveTermConfig(weight=float(weights[7])),
        action_l2=ObjectiveTermConfig(weight=float(weights[8])),
        contact_force_l2=ObjectiveTermConfig(weight=float(weights[9])),
        force_closure=ForceClosureObjectiveTermConfig(weight=float(weights[10])),
    )
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0, 9.0, 10.0],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0.0, 6.5, 7.5, 8.5, 9.5],
        ],
        dtype=np.float32,
    )
    frame_dt = 0.02
    with wp.ScopedDevice("cuda:0"):
        buffers = ObjectiveBuffers.build(world_count=2, frame_dt=frame_dt, config=config)
        buffers.object_keypoints.assign(values[:, 0])
        buffers.hand_keypoints.assign(values[:, 1])
        buffers.hand_joint_pos.assign(values[:, 2])
        buffers.contact_wrench_support.assign(values[:, 3])
        buffers.missed_contact.assign(values[:, 4])
        buffers.unintended_contact.assign(values[:, 5])
        buffers.contact_force_l2.assign(values[:, 9])
        buffers.force_closure.assign(values[:, 10])
        buffers.compose(
            terminated=wp.array(values[:, 6].astype(np.int32), dtype=wp.int32),
            action_rate_l2=wp.array(values[:, 7], dtype=wp.float32),
            action_l2=wp.array(values[:, 8], dtype=wp.float32),
        )

    np.testing.assert_allclose(buffers.terms.numpy().reshape(2, 11), values)
    np.testing.assert_allclose(buffers.score.numpy(), frame_dt * (values @ weights), atol=1.0e-6)


def test_enabled_false_disables_a_term_while_zero_weight_keeps_it_enabled():
    from flash_chord.objectives.config import (
        OBJECTIVE_TERM_NAMES,
        ObjectiveConfig,
        ShapedObjectiveTermConfig,
    )

    config = ObjectiveConfig(
        object_keypoints=ShapedObjectiveTermConfig(weight=0.0, enabled=False),
        hand_keypoints=ShapedObjectiveTermConfig(weight=0.0),
    )
    object_index = OBJECTIVE_TERM_NAMES.index("object_keypoints")
    hand_index = OBJECTIVE_TERM_NAMES.index("hand_keypoints")
    assert config.enabled[object_index] is False
    assert config.weights[object_index] == 0.0
    assert config.enabled[hand_index] is True
    assert config.weights[hand_index] == 0.0
