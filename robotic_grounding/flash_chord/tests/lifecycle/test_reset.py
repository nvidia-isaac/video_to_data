# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for lifecycle reset terms."""

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR

pytestmark = pytest.mark.gpu

_HOT3D = (
    ASSETS_DIR
    / "human_motion_data"
    / "hot3d"
    / "hot3d_processed"
    / "sequence_id=P0002_59a84a3a_seg025"
    / "robot_name=sharpa_wave"
)
_VEGA_MIXER = (
    ASSETS_DIR
    / "human_motion_data"
    / "arctic"
    / "arctic_processed"
    / "sequence_id=dataset_s01_mixer_use_01"
    / "robot_name=vega_sharpa"
    / "data.parquet"
)


@pytest.mark.sequence_data
def test_reset_places_wrists_and_objects_at_reference():
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.lifecycle.reset import reset_scene_to_frame
    from flash_chord.scene.builder import build_scene

    ref = load_mano_sharpa(str(_HOT3D))
    embodiment = SharpaHands()
    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(embodiment, ref, world_count=1)
        state = scene.model.state()
        reset_scene_to_frame(scene, ref, 0, state)
        body_q = state.body_q.numpy()

    # each wrist body (hand_C_MC) snaps to the reference wrist position
    for hand in scene.layout.hands:
        ref_pos = ref.wrist_pos_w(hand.side)[0]
        np.testing.assert_allclose(body_q[hand.wrist_body_id][:3], ref_pos, atol=2e-3)

    # objects snap to their reference frame-0 positions
    obj_pos = ref.object_body_pos_w()[0]
    for binding in scene.objects:
        for body in binding.bodies:
            np.testing.assert_allclose(
                body_q[body.body_id][:3],
                obj_pos[body.reference_body_id],
                atol=2e-3,
            )


@pytest.mark.sequence_data
def test_masked_reset_changes_only_selected_world_and_zeros_velocity():
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.lifecycle.reset import ReferenceResetTable, reset_worlds
    from flash_chord.scene.builder import build_scene

    ref = load_mano_sharpa(str(_HOT3D))
    embodiment = SharpaHands()
    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(embodiment, ref, world_count=2)
        state = scene.model.state()
        table = ReferenceResetTable.build(scene, ref)
        state.joint_qd.assign(np.ones(scene.model.joint_dof_count, dtype=np.float32))
        joint_q_before = state.joint_q.numpy().reshape(2, table.per_world_q).copy()
        reset_worlds(
            scene.model,
            table,
            state,
            reset_frame=wp.array([0, 1], dtype=wp.int32),
            reset_mask=wp.array([1, 0], dtype=wp.int32),
        )
        joint_q = state.joint_q.numpy().reshape(2, table.per_world_q)
        joint_qd = state.joint_qd.numpy().reshape(2, table.per_world_dof)

    np.testing.assert_allclose(joint_q[0], table.ref_joint_q.numpy()[0])
    np.testing.assert_allclose(joint_qd[0], 0.0)
    np.testing.assert_allclose(joint_q[1], joint_q_before[1])
    np.testing.assert_allclose(joint_qd[1], 1.0)


@pytest.mark.sequence_data
def test_vega_reset_uses_bound_named_joint_and_semantic_frame_trajectories():
    import warp as wp

    from flash_chord.data import load_reference
    from flash_chord.embodiments.vega_sharpa import VegaSharpa
    from flash_chord.lifecycle.reset import ReferenceResetTable, reset_worlds
    from flash_chord.scene.builder import build_scene

    reference = load_reference(_VEGA_MIXER)
    frame = 100
    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(
            VegaSharpa(),
            reference,
            world_count=1,
            decompose_objects=False,
        )
        table = ReferenceResetTable.build(scene, reference)
        state = scene.model.state()
        state.joint_qd.assign(np.ones(scene.model.joint_dof_count, dtype=np.float32))
        reset_worlds(scene.model, table, state, frame_id=frame)
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        body_q = state.body_q.numpy()

    np.testing.assert_array_equal(
        joint_q[: scene.layout.num_joint_q],
        scene.robot_reference.joint_q[frame],
    )
    np.testing.assert_array_equal(joint_qd, 0.0)
    for side in scene.layout.sides:
        hand_layout = scene.layout.hand(side)
        hand_reference = scene.robot_reference.hand(side)
        np.testing.assert_allclose(
            body_q[hand_layout.palm_frame.body_id, :3],
            hand_reference.wrist_pos_w[frame],
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            body_q[list(hand_layout.fingertip_body_ids), :3],
            hand_reference.dp_pos_w[frame],
            atol=2.0e-6,
        )
    assert len(scene.objects) == 1 and len(scene.objects[0].articulations) == 1
    articulation = scene.objects[0].articulations[0]
    np.testing.assert_allclose(
        joint_q[articulation.q_id],
        reference.object_articulation()[frame, articulation.reference_id],
    )


def test_reference_reset_samples_random_frames_open_fingers_and_first_frame_override():
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    world_count = 1024
    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count,
            num_frames=11,
            config=ReferenceResetConfig(seed=7),
        )
        mask = wp.ones(world_count, dtype=wp.int32)
        curriculum_voc = wp.array([1.0], dtype=wp.float32)
        reset.sample(mask, curriculum_voc)

        frames = reset.reset_frame.numpy()
        finger_scale = reset.finger_scale.numpy()
        assert frames.min() >= 0
        assert frames.max() <= 9
        assert np.unique(frames).size > 1
        assert np.all((finger_scale >= 0.0) & (finger_scale <= 0.7))
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 1.0)
        np.testing.assert_array_equal(reset.steps_since_reset.numpy(), 0)
        np.testing.assert_array_equal(reset.reset_count.numpy(), 1)

        first_frame = ReferenceReset.build(
            world_count,
            num_frames=11,
            config=ReferenceResetConfig(
                reset_to_first_frame_probability=1.0,
                first_frame_voc_threshold=0.1,
                seed=7,
            ),
        )
        first_frame.sample(mask, wp.array([0.0], dtype=wp.float32))
        np.testing.assert_array_equal(first_frame.reset_frame.numpy(), 0)


def test_reference_reset_curriculum_probability_updates_inside_existing_graph():
    import warp as wp

    from flash_chord.lifecycle.reset import FirstFrameResetCurriculum, ReferenceReset, ReferenceResetConfig

    world_count = 1024
    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count,
            num_frames=11,
            config=ReferenceResetConfig(reset_to_first_frame_probability=0.0, seed=7),
        )
        mask = wp.ones(world_count, dtype=wp.int32)
        curriculum_voc = wp.array([0.0], dtype=wp.float32)
        with wp.ScopedCapture("cuda:0") as capture:
            reset.sample(mask, curriculum_voc)

        assert isinstance(reset, FirstFrameResetCurriculum)
        assert reset.set_reset_to_first_frame_probability(1.0) == 1.0
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(reset.reset_frame.numpy(), 0)

        assert reset.set_reset_to_first_frame_probability(None) == 0.0
        wp.capture_launch(capture.graph)
        frames = reset.reset_frame.numpy()
        assert np.unique(frames).size > 1
        assert np.any(frames != 0)
        np.testing.assert_array_equal(reset.first_frame_probability.numpy(), [0.0])

        with pytest.raises(ValueError, match="reset_to_first_frame_probability"):
            reset.set_reset_to_first_frame_probability(float("nan"))


def test_reference_reset_holds_reference_and_voc_before_advancing():
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count=2,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=3, seed=3),
        )
        mask = wp.ones(2, dtype=wp.int32)
        curriculum_voc = wp.array([0.25], dtype=wp.float32)
        reset.sample(mask, curriculum_voc)
        timestep = wp.array([4, 6], dtype=wp.int32)
        episode_step = wp.zeros(2, dtype=wp.int32)
        terminated = wp.zeros(2, dtype=wp.int32)
        truncated = wp.zeros(2, dtype=wp.int32)

        reset.advance(terminated, truncated, timestep, episode_step, curriculum_voc)
        reset.advance(terminated, truncated, timestep, episode_step, curriculum_voc)
        np.testing.assert_array_equal(timestep.numpy(), [4, 6])
        np.testing.assert_array_equal(episode_step.numpy(), [2, 2])
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 1.0)

        reset.advance(terminated, truncated, timestep, episode_step, curriculum_voc)
        np.testing.assert_array_equal(timestep.numpy(), [5, 7])
        np.testing.assert_array_equal(episode_step.numpy(), [3, 3])
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 0.25)

        reset.prepare_explicit(mask, curriculum_voc)
        np.testing.assert_allclose(reset.finger_scale.numpy(), 1.0)
        np.testing.assert_array_equal(reset.steps_since_reset.numpy(), 3)
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 0.25)


def test_reference_reset_zero_decay_applies_target_voc_during_sample():
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count=2,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=0),
        )
        mask = wp.ones(2, dtype=wp.int32)
        target = wp.array([0.25], dtype=wp.float32)

        reset.sample(mask, target)

        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 0.25)
        np.testing.assert_array_equal(reset.steps_since_reset.numpy(), 0)


def test_reference_reset_linear_decay_is_configurable_and_validated():
    import warp as wp

    from flash_chord.lifecycle.reset import ReferenceReset, ReferenceResetConfig

    with pytest.raises(ValueError, match="reset_finger_openness"):
        ReferenceResetConfig(reset_finger_openness=1.1)
    with pytest.raises(ValueError, match="voc_decay_mode"):
        ReferenceResetConfig(voc_decay_mode="invalid")  # type: ignore[arg-type]

    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count=1,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=4, voc_decay_mode="linear"),
        )
        mask = wp.ones(1, dtype=wp.int32)
        curriculum_voc = wp.array([0.0], dtype=wp.float32)
        timestep = wp.array([3], dtype=wp.int32)
        episode_step = wp.zeros(1, dtype=wp.int32)
        done = wp.zeros(1, dtype=wp.int32)
        reset.sample(mask, curriculum_voc)
        reset.advance(done, done, timestep, episode_step, curriculum_voc)
        np.testing.assert_allclose(reset.applied_voc_scale.numpy(), 0.75)
        np.testing.assert_array_equal(timestep.numpy(), 3)


def test_reference_reset_publishes_applied_target_and_normalized_progress():
    import warp as wp

    from flash_chord.lifecycle.reset import RESET_CONTEXT_NAMES, ReferenceReset, ReferenceResetConfig, ResetContext

    with wp.ScopedDevice("cuda:0"):
        reset = ReferenceReset.build(
            world_count=4,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=4),
        )
        reset.applied_voc_scale.assign(np.asarray([1.0, 0.75, 0.25, 0.25], dtype=np.float32))
        reset.steps_since_reset.assign(np.asarray([0, 1, 4, 7], dtype=np.int32))
        target = wp.array([0.25], dtype=wp.float32)
        context = wp.zeros(4 * len(RESET_CONTEXT_NAMES), dtype=wp.float32)
        result = reset.compute_context(target, context)

        assert isinstance(reset, ResetContext)
        assert reset.context_names == RESET_CONTEXT_NAMES
        assert result is context
        np.testing.assert_allclose(
            context.numpy().reshape(4, 3),
            [
                [1.0, 0.25, 0.0],
                [0.75, 0.25, 0.25],
                [0.25, 0.25, 1.0],
                [0.25, 0.25, 1.0],
            ],
        )
        with pytest.raises(ValueError, match="reset context output has shape"):
            reset.compute_context(target, wp.zeros(11, dtype=wp.float32))

        zero_decay = ReferenceReset.build(
            world_count=1,
            num_frames=10,
            config=ReferenceResetConfig(voc_decay_steps=0),
        )
        mask = wp.ones(1, dtype=wp.int32)
        zero_decay.sample(mask, target)
        zero_context = wp.zeros(3, dtype=wp.float32)
        zero_decay.compute_context(target, zero_context)
        np.testing.assert_allclose(zero_context.numpy(), [0.25, 0.25, 1.0])

        zero_decay.prepare_explicit(mask, target)
        zero_decay.compute_context(target, zero_context)
        np.testing.assert_allclose(zero_context.numpy(), [0.25, 0.25, 1.0])


def test_recon_body_reset_spreads_shoulders_and_zeros_fingers_through_offsets():
    import warp as wp

    from flash_chord.lifecycle.reset import sample_recon_body_reset

    reference = np.asarray(
        [
            [0.0, 0.0, 0.3, -0.4, 9.0, 9.0],
            [0.0, 0.0, 0.5, -0.6, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    with wp.ScopedDevice("cpu"):
        outputs = (
            wp.zeros(1, dtype=wp.int32),
            wp.zeros(1, dtype=wp.int32),
            wp.ones(1, dtype=wp.float32),
            wp.zeros(1, dtype=wp.int32),
            wp.zeros(1, dtype=wp.float32),
            wp.zeros(6, dtype=wp.float32),
            wp.zeros(6, dtype=wp.float32),
        )
        wp.launch(
            sample_recon_body_reset,
            dim=1,
            inputs=[
                wp.ones(1, dtype=wp.int32),
                wp.array(reference, dtype=wp.float32),
                6,
                0,
                1,
                1,
                wp.zeros(1, dtype=wp.float32),
                wp.zeros(1, dtype=wp.float32),
                50,
                wp.array([0.0], dtype=wp.float32),
                1.0,
                0.2,
                0,
                1,
                wp.array([2, 3], dtype=wp.int32),
                42,
            ],
            outputs=list(outputs),
        )

    reset_count, reset_frame, finger_scale, age, voc, raw_offset, offset = outputs
    assert reset_count.numpy().tolist() == [1]
    assert reset_frame.numpy().tolist() == [0]
    np.testing.assert_array_equal(finger_scale.numpy(), [1.0])
    assert age.numpy().tolist() == [0]
    np.testing.assert_array_equal(voc.numpy(), [1.0])
    np.testing.assert_allclose(raw_offset.numpy(), [0.2, -0.2, -0.3, 0.4, 0.0, 0.0])
    np.testing.assert_array_equal(offset.numpy(), raw_offset.numpy())
    np.testing.assert_allclose(reference[0] + offset.numpy(), [0.2, -0.2, 0.0, 0.0, 9.0, 9.0])


def test_recon_body_immediate_reset_exactly_matches_unassisted_frame_zero_state():
    import warp as wp

    from flash_chord.lifecycle.reset import sample_recon_body_reset

    reference = np.asarray(
        [
            [0.0, 0.0, 0.3, -0.4, 9.0, 9.0],
            [0.0, 0.0, 0.5, -0.6, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    with wp.ScopedDevice("cpu"):
        outputs = (
            wp.zeros(1, dtype=wp.int32),
            wp.zeros(1, dtype=wp.int32),
            wp.ones(1, dtype=wp.float32),
            wp.zeros(1, dtype=wp.int32),
            wp.ones(1, dtype=wp.float32),
            wp.ones(6, dtype=wp.float32),
            wp.ones(6, dtype=wp.float32),
        )
        wp.launch(
            sample_recon_body_reset,
            dim=1,
            inputs=[
                wp.ones(1, dtype=wp.int32),
                wp.array(reference, dtype=wp.float32),
                6,
                0,
                1,
                0,
                wp.zeros(1, dtype=wp.float32),
                wp.ones(1, dtype=wp.float32),
                50,
                wp.array([0.0], dtype=wp.float32),
                1.0,
                0.2,
                0,
                1,
                wp.array([2, 3], dtype=wp.int32),
                42,
            ],
            outputs=list(outputs),
        )

    reset_count, reset_frame, finger_scale, age, voc, raw_offset, offset = outputs
    assert reset_count.numpy().tolist() == [1]
    assert reset_frame.numpy().tolist() == [0]
    np.testing.assert_array_equal(finger_scale.numpy(), [1.0])
    assert age.numpy().tolist() == [51]
    np.testing.assert_array_equal(voc.numpy(), [0.0])
    np.testing.assert_array_equal(raw_offset.numpy(), np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(offset.numpy(), np.zeros(6, dtype=np.float32))


def test_recon_body_settling_offsets_voc_and_reference_release_timeline():
    import warp as wp

    from flash_chord.lifecycle.reset import advance_recon_body_reset

    with wp.ScopedDevice("cpu"):
        terminated = wp.zeros(1, dtype=wp.int32)
        truncated = wp.zeros(1, dtype=wp.int32)
        curriculum = wp.array([0.2], dtype=wp.float32)
        raw_offset = wp.array([2.0, -2.0], dtype=wp.float32)
        age = wp.zeros(1, dtype=wp.int32)
        applied_voc = wp.ones(1, dtype=wp.float32)
        offset = wp.array([2.0, -2.0], dtype=wp.float32)
        timestep = wp.array([7], dtype=wp.int32)
        episode_step = wp.zeros(1, dtype=wp.int32)

        def advance(count=1):
            for _ in range(count):
                wp.launch(
                    advance_recon_body_reset,
                    dim=1,
                    inputs=[terminated, truncated, 50, 40, 10, 1.0, 2, curriculum, raw_offset],
                    outputs=[age, applied_voc, offset, timestep, episode_step],
                )

        advance(39)
        np.testing.assert_allclose(offset.numpy(), [0.05, -0.05], atol=1.0e-6)
        np.testing.assert_allclose(applied_voc.numpy(), [1.0])
        assert timestep.numpy().tolist() == [7]

        advance()
        np.testing.assert_array_equal(offset.numpy(), 0.0)
        np.testing.assert_allclose(applied_voc.numpy(), [1.0])

        advance(5)
        np.testing.assert_allclose(applied_voc.numpy(), [0.6], atol=1.0e-6)
        assert timestep.numpy().tolist() == [7]

        advance(5)
        np.testing.assert_allclose(applied_voc.numpy(), [0.2], atol=1.0e-6)
        assert (age.numpy().tolist(), episode_step.numpy().tolist(), timestep.numpy().tolist()) == ([50], [50], [7])

        advance()
        assert (age.numpy().tolist(), episode_step.numpy().tolist(), timestep.numpy().tolist()) == ([51], [51], [8])
