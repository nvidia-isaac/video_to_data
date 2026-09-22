# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for lifecycle termination terms."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _hand(side, palm, **kwargs):
    from flash_chord.embodiments.base import BodyFrame, HandLayout

    return HandLayout(
        side=side,
        palm_frame=palm,
        dp_frames=(BodyFrame(f"{side}_index_DP", palm.body_id),),
        fingertip_frames=(BodyFrame(f"{side}_index_fingertip", palm.body_id),),
        **kwargs,
    )


def _robot_reference(layout, wrist_pos_w, wrist_quat_xyzw):
    from flash_chord.embodiments.binding import BoundHandReference, RobotReferenceBinding

    wrist_pos_w = np.asarray(wrist_pos_w, dtype=np.float32)
    wrist_quat_xyzw = np.asarray(wrist_quat_xyzw, dtype=np.float32)
    num_frames, num_hands, _ = wrist_pos_w.shape
    assert num_hands == len(layout.hands)
    hands = []
    for hand_id, hand in enumerate(layout.hands):
        keypoint_count = len(hand.dp_frames)
        keypoint_pos = np.zeros((num_frames, keypoint_count, 3), dtype=np.float32)
        keypoint_quat = np.zeros((num_frames, keypoint_count, 4), dtype=np.float32)
        keypoint_quat[..., 0] = 1.0
        hands.append(
            BoundHandReference(
                side=hand.side,
                wrist_pos_w=wrist_pos_w[:, hand_id],
                wrist_quat_w=wrist_quat_xyzw[:, hand_id][:, (3, 0, 1, 2)],
                arm_joint_pos=np.empty((num_frames, len(hand.arm_q_ids)), dtype=np.float32),
                finger_joint_pos=np.empty((num_frames, len(hand.finger_q_ids)), dtype=np.float32),
                dp_pos_w=keypoint_pos,
                dp_quat_w=keypoint_quat,
                fingertip_pos_w=keypoint_pos,
                fingertip_quat_w=keypoint_quat,
            )
        )
    return RobotReferenceBinding(
        num_frames=num_frames,
        fps=20.0,
        num_joint_q=layout.num_joint_q,
        num_joint_dof=layout.num_joint_dof,
        joint_q=np.empty((num_frames, layout.num_joint_q), dtype=np.float32),
        joint_target=np.empty((num_frames, layout.num_joint_dof), dtype=np.float32),
        hands=tuple(hands),
    )


def test_termination_buffers_follow_requested_hand_and_object_order():
    import warp as wp

    from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout
    from flash_chord.lifecycle.termination import TerminationBuffers

    layout = EmbodimentLayout(
        num_joint_q=0,
        num_joint_dof=0,
        hands=(
            _hand("left", BodyFrame("left_palm", 3)),
            _hand(
                "right",
                BodyFrame(
                    "right_palm",
                    7,
                    body_to_frame_pos=(0.1, 0.2, 0.3),
                    body_to_frame_quat_xyzw=(0.0, 0.0, 1.0, 0.0),
                ),
            ),
        ),
    )
    scene = SimpleNamespace(
        layout=layout,
        model=SimpleNamespace(body_count=20),
        objects=[object()],
        world_count=2,
    )
    with wp.ScopedDevice("cuda:0"):
        buffers = TerminationBuffers.from_scene(scene, sides=("right", "left"), num_objects=3)

    assert buffers.num_hands == 2
    assert buffers.num_objects == 3
    assert buffers.bodies_per_world == 10
    assert buffers.wrist_frames.body_ids.numpy().tolist() == [7, 3]
    np.testing.assert_allclose(
        buffers.wrist_frames.body_to_frame_pos.numpy(),
        [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        buffers.wrist_frames.body_to_frame_quat.numpy(),
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    assert buffers.object_pos_err.shape == (6,)
    assert buffers.packed_diagnostics.shape == (12,)


def test_tracking_conditions_are_independently_configurable():
    import warp as wp

    from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout
    from flash_chord.lifecycle.termination import (
        PackedTerminationDiagnostics,
        Termination,
        TerminationDiagnostics,
        ThresholdTerminationTermConfig,
        TrackingTermination,
        TrackingTerminationConfig,
    )

    layout = EmbodimentLayout(
        num_joint_q=0,
        num_joint_dof=0,
        hands=(_hand("right", BodyFrame("right_palm", 0)),),
    )
    scene = SimpleNamespace(
        layout=layout,
        model=SimpleNamespace(body_count=2),
        robot_reference=_robot_reference(
            layout,
            np.array([[[10.0, 0.0, 0.0]], [[20.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]], dtype=np.float32),
            np.tile([0.0, 0.0, 0.0, 1.0], (3, 1, 1)),
        ),
        objects=[object()],
        world_count=1,
    )
    with wp.ScopedDevice("cuda:0"):
        half_angle = 0.3
        body_q = wp.array(
            [
                wp.transform(
                    wp.vec3(1.0, 0.0, 0.0),
                    wp.quat(0.0, 0.0, np.sin(half_angle), np.cos(half_angle)),
                ),
                wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
            ],
            dtype=wp.transform,
        )
        object_body_ids_w = wp.array([1], dtype=wp.int32)
        object_ref_pos_w = wp.zeros(1, dtype=wp.vec3)
        object_ref_quat_w = wp.array([wp.quat_identity()], dtype=wp.quat)
        timestep = wp.array([2], dtype=wp.int32)
        termination = TrackingTermination.build(
            scene,
            config=TrackingTerminationConfig(
                wrist_position=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
                wrist_orientation=ThresholdTerminationTermConfig(threshold=0.5),
                object_position=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
                object_orientation=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
            ),
            object_body_ids_w=object_body_ids_w,
            object_ref_pos_w=object_ref_pos_w,
            object_ref_quat_w=object_ref_quat_w,
            num_objects=1,
        )
        termination.evaluate(body_q, timestep)

    assert isinstance(termination, Termination)
    assert isinstance(termination, TerminationDiagnostics)
    assert isinstance(termination, PackedTerminationDiagnostics)
    assert termination.cause_names == ("wrist", "object")
    assert termination.cause_arrays == (termination.buffers.wrist_done, termination.buffers.object_done)
    assert termination.error_names == (
        "wrist_position_m",
        "wrist_orientation_rad",
        "object_position_m",
        "object_orientation_rad",
    )
    assert termination.error_arrays == (
        termination.buffers.wrist_pos_err,
        termination.buffers.wrist_ori_err,
        termination.buffers.object_pos_err,
        termination.buffers.object_ori_err,
    )
    assert termination.packed_diagnostic_names == termination.cause_names + termination.error_names
    np.testing.assert_allclose(
        termination.packed_diagnostics.numpy().reshape(1, -1),
        [[1.0, 0.0, 1.0, 0.6, 2.0, 0.0]],
        atol=1e-6,
    )
    assert termination.buffers.wrist_done.numpy().tolist() == [1]
    assert termination.buffers.object_done.numpy().tolist() == [0]
    assert termination.terminated.numpy().tolist() == [1]
    assert termination.truncated.numpy().tolist() == [1]


def test_wrist_termination_measures_semantic_frame_not_retained_body_origin():
    import warp as wp

    from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout
    from flash_chord.lifecycle.termination import (
        ThresholdTerminationTermConfig,
        TrackingTermination,
        TrackingTerminationConfig,
    )

    sine = float(np.sqrt(0.5))
    palm = BodyFrame(
        "right_palm",
        body_id=0,
        body_to_frame_pos=(1.0, 0.0, 0.0),
        body_to_frame_quat_xyzw=(0.0, 0.0, sine, sine),
    )
    layout = EmbodimentLayout(
        num_joint_q=0,
        num_joint_dof=0,
        hands=(_hand("right", palm),),
    )
    scene = SimpleNamespace(
        layout=layout,
        model=SimpleNamespace(body_count=2),
        robot_reference=_robot_reference(
            layout,
            np.array([[[1.0, 1.0, 0.0]]], dtype=np.float32),
            np.array([[[0.0, 0.0, 1.0, 0.0]]], dtype=np.float32),
        ),
        objects=[object()],
        world_count=1,
    )
    with wp.ScopedDevice("cuda:0"):
        body_q = wp.array(
            [
                wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat(0.0, 0.0, sine, sine)),
                wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
            ],
            dtype=wp.transform,
        )
        termination = TrackingTermination.build(
            scene,
            config=TrackingTerminationConfig(
                wrist_position=ThresholdTerminationTermConfig(threshold=1.0e-4),
                wrist_orientation=ThresholdTerminationTermConfig(threshold=1.0e-4),
                object_position=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
                object_orientation=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
            ),
            object_body_ids_w=wp.array([1], dtype=wp.int32),
            object_ref_pos_w=wp.array([[2.0, 0.0, 0.0]], dtype=wp.vec3),
            object_ref_quat_w=wp.array([wp.quat_identity()], dtype=wp.quat),
            num_objects=1,
        )
        termination.evaluate(body_q, wp.zeros(1, dtype=wp.int32))

    np.testing.assert_allclose(termination.buffers.wrist_pos_err.numpy(), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(termination.buffers.wrist_ori_err.numpy(), 0.0, atol=1.0e-6)
    assert termination.buffers.wrist_done.numpy().tolist() == [0]
    assert termination.terminated.numpy().tolist() == [0]


def test_packed_diagnostics_contain_per_world_maxima_in_name_order():
    import warp as wp

    from flash_chord.embodiments.base import BodyFrame, EmbodimentLayout, HandLayout
    from flash_chord.lifecycle.termination import (
        ThresholdTerminationTermConfig,
        TrackingTermination,
        TrackingTerminationConfig,
    )

    layout = EmbodimentLayout(
        num_joint_q=0,
        num_joint_dof=0,
        hands=(
            _hand("left", BodyFrame("left_palm", 0)),
            _hand("right", BodyFrame("right_palm", 1)),
        ),
    )
    scene = SimpleNamespace(
        layout=layout,
        model=SimpleNamespace(body_count=8),
        robot_reference=_robot_reference(
            layout,
            np.zeros((1, 2, 3), dtype=np.float32),
            np.tile([0.0, 0.0, 0.0, 1.0], (1, 2, 1)),
        ),
        objects=[object(), object()],
        world_count=2,
    )
    with wp.ScopedDevice("cuda:0"):
        body_q = wp.array(
            [wp.transform(wp.vec3(float(x), 0.0, 0.0), wp.quat_identity()) for x in range(1, 9)],
            dtype=wp.transform,
        )
        termination = TrackingTermination.build(
            scene,
            config=TrackingTerminationConfig(
                wrist_position=ThresholdTerminationTermConfig(threshold=1.5),
                wrist_orientation=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
                object_position=ThresholdTerminationTermConfig(threshold=6.5),
                object_orientation=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
            ),
            object_body_ids_w=wp.array([2, 3, 6, 7], dtype=wp.int32),
            object_ref_pos_w=wp.zeros(4, dtype=wp.vec3),
            object_ref_quat_w=wp.array([wp.quat_identity()] * 4, dtype=wp.quat),
            num_objects=2,
        )
        termination.evaluate(body_q, wp.zeros(2, dtype=wp.int32))

    assert termination.packed_diagnostic_names == (
        "wrist",
        "object",
        "wrist_position_m",
        "wrist_orientation_rad",
        "object_position_m",
        "object_orientation_rad",
    )
    np.testing.assert_allclose(
        termination.packed_diagnostics.numpy().reshape(2, -1),
        [[1.0, 0.0, 2.0, 0.0, 4.0, 0.0], [1.0, 1.0, 6.0, 0.0, 8.0, 0.0]],
    )


def test_recon_body_termination_masks_palms_and_objects_but_not_pelvis_during_freeze():
    import warp as wp

    from flash_chord.lifecycle.termination import evaluate_recon_body_termination

    reference_joint_q = np.zeros((1, 7), dtype=np.float32)
    reference_joint_q[:, 6] = 1.0
    reference_wrist_quat = np.zeros((1, 2, 4), dtype=np.float32)
    reference_wrist_quat[..., 3] = 1.0

    with wp.ScopedDevice("cpu"):
        timestep = wp.zeros(1, dtype=wp.int32)
        age = wp.zeros(1, dtype=wp.int32)
        terminated = wp.zeros(1, dtype=wp.int32)
        truncated = wp.zeros(1, dtype=wp.int32)
        causes = [wp.zeros(1, dtype=wp.int32) for _ in range(3)]
        errors = [wp.zeros(1, dtype=wp.float32) for _ in range(6)]
        diagnostics = wp.zeros(9, dtype=wp.float32)

        def evaluate(pelvis_x=0.0):
            body_q = wp.array(
                [
                    wp.transform(wp.vec3(pelvis_x, 0.0, 0.0), wp.quat_identity()),
                    wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
                    wp.transform_identity(),
                    wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
                ],
                dtype=wp.transform,
            )
            wp.launch(
                evaluate_recon_body_termination,
                dim=1,
                inputs=[
                    body_q,
                    timestep,
                    age,
                    wp.array(reference_joint_q, dtype=wp.float32),
                    wp.zeros(2, dtype=wp.vec3),
                    wp.array(reference_wrist_quat.reshape(-1, 4), dtype=wp.quat),
                    wp.array([1, 2], dtype=wp.int32),
                    wp.zeros(2, dtype=wp.vec3),
                    wp.array([wp.quat_identity(), wp.quat_identity()], dtype=wp.quat),
                    wp.array([3], dtype=wp.int32),
                    wp.zeros(1, dtype=wp.vec3),
                    wp.array([wp.quat_identity()], dtype=wp.quat),
                    0,
                    1,
                    4,
                    2,
                    1,
                    50,
                    0.7,
                    1.5,
                    0.15,
                    1.5,
                    0.1,
                    1.5,
                    1,
                ],
                outputs=[terminated, truncated, *causes, *errors, diagnostics],
            )

        evaluate()
        assert terminated.numpy().tolist() == [0]
        np.testing.assert_allclose(diagnostics.numpy()[[5, 7]], [0.2, 0.2], atol=1.0e-6)
        np.testing.assert_array_equal(
            np.asarray([error.numpy()[0] for error in errors], dtype=np.float32),
            diagnostics.numpy()[3:],
        )

        age.assign([51])
        evaluate()
        assert terminated.numpy().tolist() == [1]
        assert [cause.numpy().tolist() for cause in causes] == [[0], [1], [1]]

        age.assign([0])
        evaluate(pelvis_x=0.8)
        assert terminated.numpy().tolist() == [1]
        assert causes[0].numpy().tolist() == [1]
