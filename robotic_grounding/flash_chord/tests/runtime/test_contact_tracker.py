# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fixed-slot live contact aggregation."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _tracker(palm_frame=None, *, link_geometry=None, world_count=1):
    from flash_chord.embodiments.base import (
        BodyFrame,
        EmbodimentLayout,
        HandLayout,
        HandLinkGeometry,
    )
    from flash_chord.runtime.contact import ContactTracker

    palm_frame = palm_frame or BodyFrame("right_palm", 0)
    link_geometry = link_geometry or (
        HandLinkGeometry("palm", (0,), True),
        HandLinkGeometry("finger", (1,), True),
    )
    embodiment = EmbodimentLayout(
        num_joint_q=0,
        num_joint_dof=0,
        hands=(
            HandLayout(
                side="right",
                link_geometry=link_geometry,
                palm_frame=palm_frame,
                dp_frames=(BodyFrame("right_index_DP", 1),),
                fingertip_frames=(BodyFrame("right_index_fingertip", 1),),
            ),
        ),
    )
    return ContactTracker.build(
        embodiment,
        object_body_ids=(2,),
        world_count=world_count,
        bodies_per_world=3,
        shapes_per_world=3,
        sides=("right",),
    )


def _contacts(shape0, shape1, point0, point1, force0):
    import warp as wp

    count = len(shape0)
    return SimpleNamespace(
        rigid_contact_max=count,
        rigid_contact_count=wp.array([count], dtype=wp.int32),
        rigid_contact_shape0=wp.array(shape0, dtype=wp.int32),
        rigid_contact_shape1=wp.array(shape1, dtype=wp.int32),
        rigid_contact_point0=wp.array(point0, dtype=wp.vec3),
        rigid_contact_point1=wp.array(point1, dtype=wp.vec3),
        force=wp.array(
            [wp.spatial_vector(*force, 0.0, 0.0, 0.0) for force in force0],
            dtype=wp.spatial_vector,
        ),
    )


def test_shape_ordering_produces_the_same_object_contact():
    import warp as wp

    with wp.ScopedDevice("cuda:0"):
        tracker = _tracker()
        model = SimpleNamespace(
            shape_body=wp.array([0, 1, 2], dtype=wp.int32),
            shape_count=3,
        )
        state = SimpleNamespace(
            body_q=wp.array(
                [
                    wp.transform_identity(),
                    wp.transform_identity(),
                    wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
                ],
                dtype=wp.transform,
            )
        )
        hand_first = _contacts(
            shape0=[0],
            shape1=[2],
            point0=[[0.0, 0.0, 0.0]],
            point1=[[0.0, 1.0, 0.0]],
            force0=[[-2.0, 0.0, 0.0]],
        )
        tracker.update(model, state, hand_first)
        hand_first_pos = tracker.contact_pos_w.numpy().copy()
        hand_first_force = tracker.contact_force_w.numpy().copy()

        object_first = _contacts(
            shape0=[2],
            shape1=[0],
            point0=[[0.0, 1.0, 0.0]],
            point1=[[0.0, 0.0, 0.0]],
            force0=[[2.0, 0.0, 0.0]],
        )
        tracker.update(model, state, object_first)

    np.testing.assert_allclose(hand_first_pos[0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(tracker.contact_pos_w.numpy(), hand_first_pos)
    np.testing.assert_allclose(tracker.contact_force_w.numpy(), hand_first_force)
    np.testing.assert_allclose(hand_first_force[0], [2.0, 0.0, 0.0])
    assert tracker.contact_active.numpy().tolist() == [1, 0]


def test_contact_coordinates_use_semantic_palm_frame():
    import warp as wp

    from flash_chord.embodiments.base import BodyFrame

    sine = float(np.sqrt(0.5))
    palm = BodyFrame(
        "right_palm",
        body_id=0,
        body_to_frame_pos=(1.0, 0.0, 0.0),
        body_to_frame_quat_xyzw=(0.0, 0.0, sine, sine),
    )
    with wp.ScopedDevice("cuda:0"):
        tracker = _tracker(palm)
        model = SimpleNamespace(
            shape_body=wp.array([0, 1, 2], dtype=wp.int32),
            shape_count=3,
        )
        state = SimpleNamespace(
            body_q=wp.array(
                [wp.transform_identity(), wp.transform_identity(), wp.transform_identity()],
                dtype=wp.transform,
            )
        )
        contacts = _contacts(
            shape0=[0],
            shape1=[2],
            point0=[[0.0, 0.0, 0.0]],
            point1=[[1.0, 1.0, 0.0]],
            force0=[[-2.0, 0.0, 0.0]],
        )
        tracker.update(model, state, contacts)

    np.testing.assert_allclose(tracker.contact_pos_w.numpy()[0], [1.0, 1.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(tracker.contact_pos_b.numpy()[0], [1.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(tracker.contact_force_direction_b.numpy()[0], [0.0, -1.0, 0.0], atol=1.0e-6)
    assert tracker.contact_active.numpy().tolist() == [1, 0]


def test_multiple_contacts_aggregate_and_transform_exactly():
    import warp as wp

    with wp.ScopedDevice("cuda:0"):
        tracker = _tracker()
        model = SimpleNamespace(
            shape_body=wp.array([0, 1, 2], dtype=wp.int32),
            shape_count=3,
        )
        quarter_turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)
        state = SimpleNamespace(
            body_q=wp.array(
                [
                    wp.transform(wp.vec3(0.0, 1.0, 0.0), quarter_turn),
                    wp.transform_identity(),
                    wp.transform(wp.vec3(1.0, 0.0, 0.0), quarter_turn),
                ],
                dtype=wp.transform,
            )
        )
        contacts = _contacts(
            shape0=[0, 2],
            shape1=[2, 0],
            point0=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            point1=[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            force0=[[-2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        )
        tracker.update(model, state, contacts)

    np.testing.assert_allclose(tracker.contact_force_w.numpy()[0], [5.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(tracker.contact_pos_w.numpy()[0], [0.5, 0.5, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(tracker.contact_pos_b.numpy()[0], [-0.5, -0.5, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(tracker.contact_pos_o.numpy()[0], [0.5, 0.5, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(
        tracker.contact_force_direction_b.numpy()[0],
        [0.0, -1.0, 0.0],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        tracker.contact_force_direction_o.numpy()[0],
        [0.0, -1.0, 0.0],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(tracker.contact_force_w.numpy()[1], 0.0)
    assert tracker.contact_active.numpy().tolist() == [1, 0]


def test_reported_contact_remains_active_when_net_force_cancels():
    import warp as wp

    with wp.ScopedDevice("cuda:0"):
        tracker = _tracker()
        model = SimpleNamespace(
            shape_body=wp.array([0, 1, 2], dtype=wp.int32),
            shape_count=3,
        )
        state = SimpleNamespace(
            body_q=wp.array(
                [wp.transform_identity(), wp.transform_identity(), wp.transform_identity()],
                dtype=wp.transform,
            )
        )
        contacts = _contacts(
            shape0=[0, 0],
            shape1=[2, 2],
            point0=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            point1=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            force0=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
        tracker.update(model, state, contacts)

    np.testing.assert_allclose(tracker.contact_pos_w.numpy()[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(tracker.contact_force_w.numpy()[0], 0.0)
    np.testing.assert_allclose(tracker.contact_force_direction_o.numpy()[0], 0.0)
    assert tracker.raw_contact_count.numpy().tolist() == [2, 0]
    assert tracker.contact_active.numpy().tolist() == [1, 0]


def test_noncontact_shape_sharing_the_palm_body_is_not_misclassified():
    import warp as wp

    from flash_chord.embodiments.base import HandLinkGeometry

    links = (
        HandLinkGeometry("palm", (0,), True),
        HandLinkGeometry("terminal_arm", (1,), False),
    )
    with wp.ScopedDevice("cuda:0"):
        tracker = _tracker(link_geometry=links)
        model = SimpleNamespace(
            shape_body=wp.array([0, 0, 2], dtype=wp.int32),
            shape_count=3,
        )
        state = SimpleNamespace(
            body_q=wp.array(
                [wp.transform_identity(), wp.transform_identity(), wp.transform_identity()],
                dtype=wp.transform,
            )
        )
        contacts = _contacts(
            shape0=[1, 0],
            shape1=[2, 2],
            point0=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            point1=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            force0=[[-7.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        )
        tracker.update(model, state, contacts)

    np.testing.assert_allclose(tracker.contact_force_w.numpy(), [[2.0, 0.0, 0.0]])
    assert tracker.contact_active.numpy().tolist() == [1]


def test_replicated_shape_stride_routes_worlds_and_ignores_appended_ground():
    import warp as wp

    with wp.ScopedDevice("cuda:0"):
        tracker = _tracker(world_count=2)
        model = SimpleNamespace(
            shape_body=wp.array([0, 1, 2, 3, 4, 5, -1], dtype=wp.int32),
            shape_count=7,
        )
        state = SimpleNamespace(body_q=wp.array([wp.transform_identity()] * 6, dtype=wp.transform))
        contacts = _contacts(
            shape0=[3, 6],
            shape1=[5, 2],
            point0=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            point1=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            force0=[[-3.0, 0.0, 0.0], [-11.0, 0.0, 0.0]],
        )
        tracker.update(model, state, contacts)

    np.testing.assert_allclose(
        tracker.contact_force_w.numpy(),
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    assert tracker.contact_active.numpy().tolist() == [0, 0, 1, 0]
