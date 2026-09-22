# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared replay/evaluation marker selection."""

from types import SimpleNamespace

import numpy as np

from flash_chord.visualization import markers


def test_configured_markers_dispatch_selected_groups(monkeypatch):
    calls = []
    renderer_names = (
        "log_wrist_axes",
        "log_object_axes",
        "log_hand_keypoint_spheres",
        "log_contact_position_spheres",
        "log_contact_normals",
    )
    for name in renderer_names:
        monkeypatch.setattr(markers, name, lambda *args, _name=name, **kwargs: calls.append(_name))
    viewer = SimpleNamespace(log_contacts=lambda *args: calls.append("raw_contacts"))
    config = SimpleNamespace(
        enabled=True,
        axes=True,
        keypoints=True,
        keypoint_frame="fingertip",
        contacts=True,
        raw_contacts=False,
    )

    markers.log_configured_markers(viewer, config, None, None, 0, None, None, None)

    assert calls == list(renderer_names)


def test_raw_contacts_remain_independently_configurable(monkeypatch):
    monkeypatch.setattr(markers, "log_wrist_axes", lambda *args: None)
    viewer_calls = []
    viewer = SimpleNamespace(log_contacts=lambda *args: viewer_calls.append(args))
    config = SimpleNamespace(enabled=False, axes=True, keypoints=True, contacts=True, raw_contacts=True)

    markers.log_configured_markers(viewer, config, None, None, 0, "state", "contacts", None)

    assert viewer_calls == [("contacts", "state")]


def test_hand_markers_use_bound_targets_and_semantic_frame_offsets(monkeypatch):
    from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout
    from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding

    palm = BodyFrame("left_palm", 0, body_to_frame_pos=(0.1, 0.0, 0.0))
    dp = BodyFrame("left_dp", 1)
    fingertip = BodyFrame("left_fingertip", 1, body_to_frame_pos=(0.5, 0.0, 0.0))
    layout = EmbodimentLayout(
        num_joint_q=0,
        num_joint_dof=0,
        hands=(
            HandLayout(
                side="left",
                palm_frame=palm,
                dp_frames=(dp,),
                fingertip_frames=(fingertip,),
            ),
        ),
    )
    hand = BoundHandReference(
        side="left",
        wrist_pos_w=np.array([[10.0, 0.0, 0.0]], dtype=np.float32),
        wrist_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        arm_joint_pos=np.empty((1, 0), dtype=np.float32),
        finger_joint_pos=np.empty((1, 0), dtype=np.float32),
        dp_pos_w=np.array([[[11.0, 0.0, 0.0]]], dtype=np.float32),
        dp_quat_w=np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        fingertip_pos_w=np.array([[[12.0, 0.0, 0.0]]], dtype=np.float32),
        fingertip_quat_w=np.array([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
    )
    binding = RobotReferenceBinding(
        num_frames=1,
        fps=20.0,
        num_joint_q=0,
        num_joint_dof=0,
        joint_q=np.empty((1, 0), dtype=np.float32),
        joint_target=np.empty((1, 0), dtype=np.float32),
        hands=(hand,),
    )
    scene = SimpleNamespace(layout=layout, robot_reference=binding)
    root = np.sqrt(0.5)
    body_q = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, root, root],
        ],
        dtype=np.float32,
    )
    state = SimpleNamespace(body_q=SimpleNamespace(numpy=lambda: body_q))
    points = {}
    monkeypatch.setattr(
        markers,
        "_points",
        lambda _viewer, name, values, *_args: points.__setitem__(name, np.asarray(values)),
    )

    markers.log_hand_keypoint_spheres(
        None,
        scene,
        None,
        0,
        state,
        keypoint_frame="fingertip",
    )

    np.testing.assert_allclose(points["/ref/hand_keypoints"], [[10, 0, 0], [12, 0, 0]])
    np.testing.assert_allclose(points["/sim/hand_keypoints"], [[1.1, 0, 0], [2, 0.5, 0]], atol=1.0e-6)
