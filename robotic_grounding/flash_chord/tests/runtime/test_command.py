# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for object command mappings and device-side target gathering."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def test_object_layout_maps_rigid_and_articulated_bodies():
    from flash_chord.data.reference import ObjectJointDriveSpec
    from flash_chord.runtime.command import ObjectLayout
    from flash_chord.scene.collision import ShapeSpan
    from flash_chord.scene.objects import (
        ObjectBinding,
        ObjectBodyBinding,
        ObjectJointBinding,
        ObjectRootBinding,
    )

    rigid = [
        ObjectBinding(
            "a",
            bodies=(ObjectBodyBinding(10, 0),),
            root=ObjectRootBinding(10, 0, tuple(range(7)), tuple(range(6))),
            articulations=(),
            shapes=ShapeSpan(0, 1),
        ),
        ObjectBinding(
            "b",
            bodies=(ObjectBodyBinding(11, 1),),
            root=ObjectRootBinding(11, 1, tuple(range(7, 14)), tuple(range(6, 12))),
            articulations=(),
            shapes=ShapeSpan(1, 2),
        ),
    ]
    rigid_layout = ObjectLayout.from_bindings(rigid, num_reference_bodies=2, num_reference_articulations=0)
    assert rigid_layout.body_ids == (10, 11)
    assert rigid_layout.body_object_ids == (0, 1)
    assert rigid_layout.voc_body_ids == (10, 11)
    assert rigid_layout.voc_reference_body_ids == (0, 1)
    assert rigid_layout.voc_object_body_offsets == (0, 1, 2)
    assert rigid_layout.voc_object_body_ids == (10, 11)

    articulated = [
        ObjectBinding(
            "box",
            bodies=(ObjectBodyBinding(21, 1), ObjectBodyBinding(20, 0)),
            root=ObjectRootBinding(20, 0, tuple(range(7)), tuple(range(6))),
            articulations=(
                ObjectJointBinding(
                    q_id=7,
                    dof_id=6,
                    reference_id=0,
                    drive=ObjectJointDriveSpec(kp=50.0, kd=2.0, effort_limit=50.0),
                ),
            ),
            shapes=ShapeSpan(0, 4),
        )
    ]
    articulated_layout = ObjectLayout.from_bindings(
        articulated,
        num_reference_bodies=2,
        num_reference_articulations=1,
    )
    assert articulated_layout.body_ids == (20, 21)
    assert articulated_layout.body_object_ids == (0, 0)
    assert articulated_layout.voc_body_ids == (20,)
    assert articulated_layout.voc_reference_body_ids == (0,)
    assert articulated_layout.voc_object_body_offsets == (0, 2)
    assert articulated_layout.voc_object_body_ids == (21, 20)
    assert articulated_layout.articulation_q_ids == (7,)
    assert articulated_layout.articulation_dof_ids == (6,)
    assert articulated_layout.articulation_reference_ids == (0,)
    assert articulated_layout.articulation_kp == (50.0,)
    assert articulated_layout.articulation_kd == (2.0,)
    assert articulated_layout.articulation_effort_limit == (50.0,)


def test_object_layout_aligns_multiple_articulations_by_reference_column():
    from flash_chord.data.reference import ObjectJointDriveSpec
    from flash_chord.runtime.command import ObjectLayout
    from flash_chord.scene.collision import ShapeSpan
    from flash_chord.scene.objects import (
        ObjectBinding,
        ObjectBodyBinding,
        ObjectJointBinding,
        ObjectRootBinding,
    )

    early_drive = ObjectJointDriveSpec(kp=5.0, kd=0.5, effort_limit=8.0)
    late_drive = ObjectJointDriveSpec(kp=7.0, kd=0.75, effort_limit=9.0)
    bindings = [
        ObjectBinding(
            "late",
            bodies=(ObjectBodyBinding(11, 1),),
            root=ObjectRootBinding(11, 1, tuple(range(8, 15)), tuple(range(7, 13))),
            articulations=(ObjectJointBinding(15, 13, 1, late_drive),),
            shapes=ShapeSpan(1, 2),
        ),
        ObjectBinding(
            "early",
            bodies=(ObjectBodyBinding(10, 0),),
            root=ObjectRootBinding(10, 0, tuple(range(7)), tuple(range(6))),
            articulations=(ObjectJointBinding(7, 6, 0, early_drive),),
            shapes=ShapeSpan(0, 1),
        ),
    ]

    layout = ObjectLayout.from_bindings(bindings, num_reference_bodies=2, num_reference_articulations=2)

    assert layout.voc_body_ids == (11, 10)
    assert layout.voc_reference_body_ids == (1, 0)
    assert layout.articulation_reference_ids == (0, 1)
    assert layout.articulation_q_ids == (7, 15)
    assert layout.articulation_dof_ids == (6, 13)
    assert layout.articulation_kp == (early_drive.kp, late_drive.kp)
    assert layout.articulation_kd == (early_drive.kd, late_drive.kd)
    assert layout.articulation_effort_limit == (early_drive.effort_limit, late_drive.effort_limit)


def test_object_bindings_reject_cross_wired_or_overlapping_topology():
    from flash_chord.data.reference import ObjectJointDriveSpec
    from flash_chord.scene.collision import ShapeSpan
    from flash_chord.scene.objects import (
        ObjectBinding,
        ObjectBodyBinding,
        ObjectJointBinding,
        ObjectRootBinding,
    )

    with pytest.raises(ValueError, match="exactly seven"):
        ObjectRootBinding(10, 0, (0, 1, 2, 3, 4, 5, 7), tuple(range(6)))
    with pytest.raises(ValueError, match="exactly six"):
        ObjectRootBinding(10, 0, tuple(range(7)), (0, 1, 2, 3, 4, 6))

    bodies = (ObjectBodyBinding(10, 0), ObjectBodyBinding(11, 1))
    with pytest.raises(ValueError, match="root must match"):
        ObjectBinding(
            "cross_wired",
            bodies=bodies,
            root=ObjectRootBinding(10, 1, tuple(range(7)), tuple(range(6))),
            articulations=(),
            shapes=ShapeSpan(0, 1),
        )
    with pytest.raises(ValueError, match="reuses a free-root q ID"):
        ObjectBinding(
            "overlap",
            bodies=bodies,
            root=ObjectRootBinding(10, 0, tuple(range(7)), tuple(range(6))),
            articulations=(
                ObjectJointBinding(
                    q_id=6,
                    dof_id=6,
                    reference_id=0,
                    drive=ObjectJointDriveSpec(kp=1.0, kd=0.0, effort_limit=1.0),
                ),
            ),
            shapes=ShapeSpan(0, 1),
        )


def test_gather_object_command_uses_each_world_timestep_and_scale():
    import warp as wp

    from flash_chord.runtime.command import gather_object_command

    body_pos_w = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    body_quat_w = np.array(
        [
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    with wp.ScopedDevice("cuda:0"):
        timestep = wp.array([0, 1], dtype=wp.int32)
        body_target_pos_w = wp.zeros(4, dtype=wp.vec3)
        body_target_quat_w = wp.zeros(4, dtype=wp.quat)
        voc_target_pos_w = wp.zeros(2, dtype=wp.vec3)
        voc_target_quat_w = wp.zeros(2, dtype=wp.quat)
        articulation_target_pos = wp.zeros(4, dtype=wp.float32)
        wp.launch(
            gather_object_command,
            dim=2,
            inputs=[
                timestep,
                2,
                2,
                1,
                2,
                2,
                wp.array(body_pos_w.reshape(-1, 3), dtype=wp.vec3),
                wp.array(body_quat_w.reshape(-1, 4), dtype=wp.quat),
                wp.array([0.25, 0.5, 0.75, 1.0], dtype=wp.float32),
                wp.array([0, 0], dtype=wp.int32),
                wp.array([0], dtype=wp.int32),
                wp.array([0, 1], dtype=wp.int32),
                wp.array([1.0, 2.0], dtype=wp.float32),
                wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=wp.vec3),
            ],
            outputs=[
                body_target_pos_w,
                body_target_quat_w,
                voc_target_pos_w,
                voc_target_quat_w,
                articulation_target_pos,
            ],
        )
        body_target = body_target_pos_w.numpy().reshape(2, 2, 3)
        voc_target = voc_target_pos_w.numpy().reshape(2, 1, 3)
        articulation_target = articulation_target_pos.numpy().reshape(2, 2)

    np.testing.assert_allclose(body_target[0], body_pos_w[0])
    np.testing.assert_allclose(body_target[1], [[8.0, 8.0, 9.0], [14.0, 14.0, 15.0]])
    np.testing.assert_allclose(voc_target[:, 0], [[1.0, 2.0, 3.0], [8.0, 8.0, 9.0]])
    np.testing.assert_allclose(articulation_target, [[0.25, 0.5], [0.75, 1.0]])
