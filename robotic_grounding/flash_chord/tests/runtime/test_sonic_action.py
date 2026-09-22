# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SONIC tokenizer, history, residual, and real-inference contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flash_chord.runtime.actions.sonic import (
    SONIC_ACTUATOR_DELAY_GROUPS,
    SONIC_CURRENT_OBSERVATION_DIM,
    SONIC_DECODER_DIM,
    SONIC_ENCODER_DIM,
    SONIC_HISTORY_LENGTH,
    SONIC_JOINT_DIM,
    SONIC_LATENT_DIM,
    SONIC_POLICY_JOINT_NAMES,
    gather_sonic_encoder_input,
    append_sonic_history,
    build_sonic_encoder_reference,
    pack_sonic_decoder_input,
    process_sonic_joint_targets,
)


def test_encoder_reference_has_exact_upstream_layout_and_future_spacing():
    frame_count = 51
    q_ids = tuple(range(7, 7 + SONIC_JOINT_DIM))
    joint_q = np.zeros((frame_count, 7 + SONIC_JOINT_DIM), dtype=np.float32)
    frames = np.arange(frame_count, dtype=np.float32)[:, None]
    joints = np.arange(SONIC_JOINT_DIM, dtype=np.float32)[None, :]
    joint_q[:, q_ids] = frames * 100.0 + joints
    joint_q[:, 6] = 1.0
    scene = SimpleNamespace(
        layout=SimpleNamespace(
            scalar_joints=SimpleNamespace(names=SONIC_POLICY_JOINT_NAMES, q_ids=q_ids),
        ),
        robot_reference=SimpleNamespace(
            fps=50.0,
            num_frames=frame_count,
            joint_q=joint_q,
        ),
    )

    encoder = build_sonic_encoder_reference(scene)

    assert encoder.shape == (frame_count, SONIC_ENCODER_DIM)
    assert encoder.dtype == np.float32
    assert encoder.flags.c_contiguous
    np.testing.assert_array_equal(encoder[:, :4], 0.0)
    np.testing.assert_array_equal(encoder[:, 584:601], 0.0)
    np.testing.assert_array_equal(encoder[:, 661:], 0.0)

    future = np.arange(SONIC_HISTORY_LENGTH) * 5
    expected_position = joint_q[future][:, q_ids].reshape(-1)
    np.testing.assert_array_equal(encoder[0, 4:294], expected_position)
    np.testing.assert_array_equal(encoder[0, 294:584], 5000.0)
    expected_identity_6d = np.tile(np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), SONIC_HISTORY_LENGTH)
    np.testing.assert_array_equal(encoder[0, 601:661], expected_identity_6d)

    clamped_future = np.minimum(49 + future, frame_count - 1)
    np.testing.assert_array_equal(encoder[49, 4:294], joint_q[clamped_future][:, q_ids].reshape(-1))
    expected_velocity = np.zeros((SONIC_HISTORY_LENGTH, SONIC_JOINT_DIM), dtype=np.float32)
    expected_velocity[0] = 5000.0
    np.testing.assert_array_equal(encoder[49, 294:584], expected_velocity.reshape(-1))


def _expected_term_major_history(samples: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(samples)
    return np.concatenate(
        (
            stacked[:, :3].reshape(-1),
            stacked[:, 3:32].reshape(-1),
            stacked[:, 32:61].reshape(-1),
            stacked[:, 61:90].reshape(-1),
            stacked[:, 90:93].reshape(-1),
        )
    )


def test_decoder_history_first_fills_then_appends_oldest_to_newest_term_major():
    import warp as wp

    with wp.ScopedDevice("cpu"):
        first_host = np.arange(SONIC_CURRENT_OBSERVATION_DIM, dtype=np.float32)
        first = wp.array(first_host[None, :], dtype=wp.float32)
        history = wp.zeros(
            (1, SONIC_HISTORY_LENGTH, SONIC_CURRENT_OBSERVATION_DIM),
            dtype=wp.float32,
        )
        head = wp.full(1, SONIC_HISTORY_LENGTH - 1, dtype=wp.int32)
        initialized = wp.zeros(1, dtype=wp.int32)
        latent_host = np.arange(SONIC_LATENT_DIM, dtype=np.float32) + 1000.0
        latent = wp.array(latent_host[None, :], dtype=wp.float32)
        decoder = wp.zeros((1, SONIC_DECODER_DIM), dtype=wp.float32)

        wp.launch(
            append_sonic_history,
            dim=1,
            inputs=[first],
            outputs=[history, head, initialized],
        )
        wp.launch(
            pack_sonic_decoder_input,
            dim=1,
            inputs=[latent, history, head],
            outputs=[decoder],
        )
        expected_samples = [first_host] * SONIC_HISTORY_LENGTH
        np.testing.assert_array_equal(decoder.numpy()[0, :SONIC_LATENT_DIM], latent_host)
        np.testing.assert_array_equal(
            decoder.numpy()[0, SONIC_LATENT_DIM:],
            _expected_term_major_history(expected_samples),
        )
        assert head.numpy().tolist() == [SONIC_HISTORY_LENGTH - 1]
        assert initialized.numpy().tolist() == [1]

        second_host = first_host + 100.0
        second = wp.array(second_host[None, :], dtype=wp.float32)
        wp.launch(
            append_sonic_history,
            dim=1,
            inputs=[second],
            outputs=[history, head, initialized],
        )
        wp.launch(
            pack_sonic_decoder_input,
            dim=1,
            inputs=[latent, history, head],
            outputs=[decoder],
        )
        expected_samples = [first_host] * (SONIC_HISTORY_LENGTH - 1) + [second_host]
        np.testing.assert_array_equal(
            decoder.numpy()[0, SONIC_LATENT_DIM:],
            _expected_term_major_history(expected_samples),
        )
        assert head.numpy().tolist() == [0]


def test_reference_offsets_condition_future_positions_and_control_step_target_history():
    import warp as wp

    with wp.ScopedDevice("cpu"):
        q_per_world = 50
        encoder_reference = wp.zeros((1, SONIC_ENCODER_DIM), dtype=wp.float32)
        encoder_offset = np.zeros(q_per_world, dtype=np.float32)
        encoder_offset[:SONIC_JOINT_DIM] = np.arange(SONIC_JOINT_DIM, dtype=np.float32) + 1.0
        offset = wp.array(encoder_offset, dtype=wp.float32)
        sonic_q_ids = wp.array(np.arange(SONIC_JOINT_DIM), dtype=wp.int32)
        encoder_input = wp.zeros((1, SONIC_ENCODER_DIM), dtype=wp.float32)
        wp.launch(
            gather_sonic_encoder_input,
            dim=(1, SONIC_ENCODER_DIM),
            inputs=[wp.zeros(1, dtype=wp.int32), encoder_reference, offset, sonic_q_ids, 1, q_per_world],
            outputs=[encoder_input],
        )
        expected = np.tile(encoder_offset[:SONIC_JOINT_DIM], SONIC_HISTORY_LENGTH)
        np.testing.assert_array_equal(encoder_input.numpy()[0, 4:294], expected)
        np.testing.assert_array_equal(encoder_input.numpy()[0, 294:], 0.0)

        action_dim = processed_dim = 43
        finger_count = 14
        finger_q_ids = wp.array(np.arange(29, 43), dtype=wp.int32)
        finger_offset_host = np.zeros(q_per_world, dtype=np.float32)
        finger_offset_host[29:43] = 0.25
        finger_offset = wp.array(finger_offset_host, dtype=wp.float32)
        raw_action = wp.zeros(action_dim, dtype=wp.float32)
        processed_target = wp.zeros(processed_dim, dtype=wp.float32)
        last_sonic_action = wp.zeros((1, SONIC_JOINT_DIM), dtype=wp.float32)
        joint_target = wp.zeros(q_per_world, dtype=wp.float32)
        action_l2 = wp.zeros(1, dtype=wp.float32)
        action_rate_l2 = wp.zeros(1, dtype=wp.float32)
        actor_history = wp.zeros(3 * processed_dim, dtype=wp.float32)

        for value in (1.0, 2.0, 3.0, 4.0):
            wp.launch(
                process_sonic_joint_targets,
                dim=1,
                inputs=[
                    wp.full(action_dim, value, dtype=wp.float32),
                    wp.zeros(1, dtype=wp.int32),
                    wp.zeros((1, SONIC_JOINT_DIM), dtype=wp.float32),
                    wp.zeros((1, q_per_world), dtype=wp.float32),
                    finger_offset,
                    1,
                    q_per_world,
                    q_per_world,
                    action_dim,
                    processed_dim,
                    SONIC_JOINT_DIM,
                    finger_count,
                    sonic_q_ids,
                    wp.array(np.arange(SONIC_JOINT_DIM), dtype=wp.int32),
                    wp.array(np.arange(SONIC_JOINT_DIM), dtype=wp.int32),
                    finger_q_ids,
                    finger_q_ids,
                    finger_q_ids,
                    wp.ones(SONIC_JOINT_DIM, dtype=wp.float32),
                    wp.zeros(SONIC_JOINT_DIM, dtype=wp.float32),
                    1.0,
                    1.0,
                    False,
                ],
                outputs=[
                    raw_action,
                    processed_target,
                    last_sonic_action,
                    joint_target,
                    action_l2,
                    action_rate_l2,
                    actor_history,
                ],
            )

        history = actor_history.numpy().reshape(3, processed_dim)
        np.testing.assert_array_equal(
            history[:, :SONIC_JOINT_DIM],
            np.tile(np.asarray([2.0, 3.0, 4.0])[:, None], (1, SONIC_JOINT_DIM)),
        )
        np.testing.assert_array_equal(
            history[:, SONIC_JOINT_DIM:],
            np.tile(np.asarray([2.25, 3.25, 4.25])[:, None], (1, finger_count)),
        )


@pytest.mark.gpu
@pytest.mark.sequence_data
def test_real_snack_sonic_action_runs_cuda_inference_and_applies_exact_targets():
    import warp as wp

    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.base import BaseEnv, BaseEnvConfig
    from flash_chord.runtime.actions import Action, ActionSpec, PolicyAction, SonicJointResidualAction
    from flash_chord.runtime.replay import ReplayConfig, ReplayRunner
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    config = compose_config(
        "sonic_replay",
        [
            "scene.support=false",
            "scene.decompose_objects=false",
            "collision.robot_scope=none",
            "collision.robot_object_collision=false",
            "collision.robot_support_collision=false",
        ],
    )
    embodiment = instantiate_typed(config.embodiment, Embodiment)
    collision = instantiate_typed(config.collision, CollisionPolicy)
    replay_config = instantiate_typed(config.replay, ReplayConfig)
    assert isinstance(replay_config.action, ActionSpec)

    with wp.ScopedDevice("cuda:0"):
        setup = setup_scene(
            parquet=config.task.parquet,
            control_fps=replay_config.sim.fps,
            motion_speed=config.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=1,
            include_support=False,
            decompose_objects=False,
            source_frame_playback=bool(config.task.source_frame_playback),
        )
        runner = ReplayRunner(setup.scene, setup.reference, replay_config, device="cuda:0")
        runner.reset(0)
        assert setup.reference.num_frames == 800
        assert setup.reference.metadata.source_frame_playback
        action = runner.action
        assert isinstance(action, SonicJointResidualAction)
        assert isinstance(action, Action)
        assert isinstance(action, PolicyAction)
        assert (action.action_dim, action.processed_dim) == (43, 43)
        assert action.encoder_session.provider == "CUDAExecutionProvider"
        assert action.decoder_session.provider == "CUDAExecutionProvider"
        dof_by_name = dict(zip(action.residual_joint_names, action.sonic_dof_ids.numpy(), strict=True))
        assert tuple(delay.delayed_dof_ids for delay in action.delays) == tuple(
            tuple(int(dof_by_name[name]) for name in group) for group in SONIC_ACTUATOR_DELAY_GROUPS
        )

        action.process(runner.zero_action, runner.timestep, runner.state_0)
        wp.synchronize()
        np.testing.assert_array_equal(action.encoder_input.numpy(), action.encoder_reference.numpy()[[0]])
        for array in (
            action.latent,
            action.decoder_input,
            action.decoder_action,
            action.current_observation,
            action.joint_target,
            action.processed_target,
        ):
            assert np.isfinite(array.numpy()).all()

        decoder = action.decoder_action.numpy()[0]
        upstream_reconbody_decoder = np.asarray(
            [
                0.097205065,
                0.072333775,
                0.066371985,
                0.284888625,
                -0.335025430,
                0.026375547,
                0.029117335,
                -0.067300238,
                -0.333325714,
                -0.357283235,
                -0.487421989,
                -0.094483495,
                0.155918062,
                0.782079458,
                0.758527339,
                0.563644171,
                -0.496401489,
                -0.097642384,
                0.102923825,
                -0.160502851,
                0.112822652,
                -0.177561522,
                -0.076786943,
                0.378553569,
                -0.185023144,
                0.014389683,
                -0.144214198,
                -0.512896955,
                -0.126307115,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(decoder, upstream_reconbody_decoder, rtol=0.0, atol=5.0e-6)
        scale = action.sonic_scale.numpy()
        default = action.default_sonic_position.numpy()
        target = action.joint_target.numpy()
        sonic_dofs = action.sonic_dof_ids.numpy()
        np.testing.assert_allclose(target[sonic_dofs], decoder * scale + default, atol=1.0e-6)
        finger_dofs = action.finger_dof_ids.numpy()
        np.testing.assert_array_equal(target[finger_dofs], action.reference_joint_target.numpy()[0, finger_dofs])
        scalar_slots = np.concatenate((action.sonic_scalar_slots.numpy(), action.finger_scalar_slots.numpy()))
        mapped_dofs = np.concatenate((sonic_dofs, finger_dofs))
        processed = action.processed_target.numpy().reshape(1, action.processed_dim)[0]
        np.testing.assert_array_equal(processed[scalar_slots], target[mapped_dofs])

        residual_host = np.linspace(-0.8, 0.8, action.action_dim, dtype=np.float32)
        residual = wp.array(residual_host, dtype=wp.float32, device="cuda:0")
        action.process(residual, runner.timestep, runner.state_0)
        wp.synchronize()
        decoder = action.decoder_action.numpy()[0]
        target = action.joint_target.numpy()
        expected_sonic = decoder * scale + default
        expected_sonic += residual_host[:29] * action.residual_scale * scale
        np.testing.assert_allclose(target[sonic_dofs], expected_sonic, atol=1.0e-6)
        expected_fingers = (
            action.reference_joint_target.numpy()[0, finger_dofs] + residual_host[29:] * action.finger_residual_scale
        )
        np.testing.assert_allclose(target[finger_dofs], expected_fingers, atol=1.0e-6)
        np.testing.assert_array_equal(action.raw_action.numpy(), residual_host)
        np.testing.assert_allclose(action.action_l2.numpy(), np.sum(residual_host**2), atol=5.0e-6)
        np.testing.assert_allclose(action.action_rate_l2.numpy(), np.sum(residual_host**2), atol=5.0e-6)

        runner.step(0)
        wp.synchronize()
        assert runner.sim_time == pytest.approx(1.0 / 50.0)
        assert np.isfinite(runner.state_0.joint_q.numpy()).all()
        assert np.isfinite(runner.state_0.joint_qd.numpy()).all()

        environment = BaseEnv(
            setup.scene,
            setup.reference,
            config=BaseEnvConfig(
                sim=replay_config.sim,
                action=replay_config.action,
                contact=None,
                objective=None,
            ),
            action=action,
            device="cuda:0",
        )
        environment.reset(frame_id=0)
        environment.capture_transition()
        environment.step(runner.zero_action)
        wp.synchronize()
        assert environment.timestep.numpy().tolist() == [1]
        assert np.isfinite(environment.state_0.joint_q.numpy()).all()
        assert np.isfinite(environment.state_0.joint_qd.numpy()).all()
