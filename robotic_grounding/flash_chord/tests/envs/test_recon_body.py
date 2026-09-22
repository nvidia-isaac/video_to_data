# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic checks for the fused ReconBody observation schema."""

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR

_G1_REFERENCE = (
    ASSETS_DIR
    / "human_motion_data"
    / "whole_body"
    / "soma"
    / "sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01"
    / "robot_name=g1"
)


def test_recon_body_observation_has_exact_486_float_analytic_layout():
    import warp as wp

    from flash_chord.envs.recon_body import (
        RECON_BODY_OBSERVATION_BLOCKS,
        RECON_BODY_OBSERVATION_DIM,
        compute_recon_body_observation,
    )

    with wp.ScopedDevice("cpu"):
        num_frames = 11
        q_per_world = 50
        num_joints = 43
        reference_joint_q = np.zeros((num_frames, q_per_world), dtype=np.float32)
        reference_joint_q[:, 6] = 1.0
        reference_joint_q[:, 0] = np.arange(num_frames, dtype=np.float32)
        reference_joint_q[:, 7:] = np.arange(num_frames, dtype=np.float32)[:, None]
        reference_wrist_pos = np.zeros((num_frames, 2, 3), dtype=np.float32)
        reference_wrist_pos[:, 0, 0] = 1.0 + np.arange(num_frames)
        reference_wrist_pos[:, 1, 1] = 1.0 + np.arange(num_frames)
        reference_wrist_quat = np.zeros((num_frames, 2, 4), dtype=np.float32)
        reference_wrist_quat[..., 3] = 1.0
        reference_object_pos = np.zeros((num_frames, 3), dtype=np.float32)
        reference_object_pos[:, 0] = np.arange(num_frames)
        reference_object_quat = np.zeros((num_frames, 4), dtype=np.float32)
        reference_object_quat[:, 3] = 1.0
        history = np.arange(3 * num_joints, dtype=np.float32)

        output = wp.zeros(RECON_BODY_OBSERVATION_DIM, dtype=wp.float32)
        wp.launch(
            compute_recon_body_observation,
            dim=1,
            inputs=[
                wp.array(
                    [
                        wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                        wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
                        wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity()),
                        wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
                    ],
                    dtype=wp.transform,
                ),
                wp.array(
                    [
                        wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        wp.spatial_vector(1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
                        wp.spatial_vector(4.0, 5.0, 6.0, 0.0, 0.0, 0.0),
                        wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    ],
                    dtype=wp.spatial_vector,
                ),
                wp.zeros(q_per_world, dtype=wp.float32),
                wp.zeros(num_joints, dtype=wp.float32),
                wp.zeros(1, dtype=wp.int32),
                wp.zeros(q_per_world, dtype=wp.float32),
                wp.array(reference_joint_q, dtype=wp.float32),
                wp.array(reference_wrist_pos.reshape(-1, 3), dtype=wp.vec3),
                wp.array(reference_wrist_quat.reshape(-1, 4), dtype=wp.quat),
                wp.array(reference_object_pos, dtype=wp.vec3),
                wp.array(reference_object_quat, dtype=wp.quat),
                wp.array(history, dtype=wp.float32),
                wp.array([1, 2], dtype=wp.int32),
                wp.zeros(2, dtype=wp.vec3),
                wp.array([wp.quat_identity(), wp.quat_identity()], dtype=wp.quat),
                wp.array(np.arange(7, 50), dtype=wp.int32),
                wp.array(np.arange(num_joints), dtype=wp.int32),
                wp.zeros(num_joints, dtype=wp.float32),
                3,
                0,
                num_frames,
                4,
                q_per_world,
                num_joints,
                num_joints,
            ],
            outputs=[output],
        )

    values = output.numpy()
    starts = {}
    cursor = 0
    for name, width in RECON_BODY_OBSERVATION_BLOCKS:
        starts[name] = slice(cursor, cursor + width)
        cursor += width
    assert cursor == RECON_BODY_OBSERVATION_DIM
    np.testing.assert_array_equal(values[starts["wrist_position_pelvis_m[right,left]"]], [1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(
        values[starts["wrist_orientation_pelvis_6d[right,left]"]],
        np.tile([1, 0, 0, 1, 0, 0], 2),
    )
    np.testing.assert_array_equal(
        values[starts["wrist_linear_velocity_pelvis_mps[right,left]"]],
        [1, 2, 3, 4, 5, 6],
    )
    np.testing.assert_array_equal(values[starts["object_position_pelvis_m"]], [2, 0, 0])
    np.testing.assert_array_equal(
        values[starts["future_root_position_w_m[offsets=0,5,10]"]],
        [0, 0, 0, 5, 0, 0, 10, 0, 0],
    )
    future_joint = values[starts["future_joint_position_delta_rad[offsets=0,5,10]"]].reshape(3, 43)
    np.testing.assert_array_equal(future_joint[:, 0], [0, 5, 10])
    np.testing.assert_array_equal(
        values[starts["live_hand_pose_in_object[position,6d;left,right]"]].reshape(2, 9)[:, :3],
        [[-2, 1, 0], [-1, 0, 0]],
    )
    np.testing.assert_array_equal(values[starts["normalized_reference_phase"]], [0.0])
    np.testing.assert_array_equal(values[starts["processed_target_history[oldest,newest;3x43]"]], history)
    assert np.isfinite(values).all()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.sequence_data
def test_recon_body_flash_sac_composition_runs_eager_and_captured():
    import warp as wp

    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.recon_body import ReconBodyObservation
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.lifecycle.reset import ReconBodyReset
    from flash_chord.lifecycle.termination import ReconBodyTermination
    from flash_chord.objectives.recon_body import ReconBodyObjective
    from flash_chord.runtime.actions.sonic import SonicJointResidualAction
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    config = compose_config(
        "train_flash_sac",
        [
            "experiment=g1_recon_body_flash_sac",
            f"task.parquet={_G1_REFERENCE}",
            "scene.world_count=1",
            "logging.mode=disabled",
        ],
    )
    environment_config = instantiate_typed(config.env, RLEnvConfig)
    embodiment = instantiate_typed(config.embodiment, Embodiment)
    collision = instantiate_typed(config.collision, CollisionPolicy)

    with wp.ScopedDevice("cuda:0"):
        setup = setup_scene(
            parquet=config.task.parquet,
            control_fps=environment_config.sim.fps,
            motion_speed=config.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=1,
            include_support=config.scene.support,
            decompose_objects=config.scene.decompose_objects,
            contact_friction=config.scene.contact_friction,
            object_free_joint_damping=config.scene.object_free_joint_damping,
            source_frame_playback=bool(config.task.source_frame_playback),
            motion_start_frame=int(config.task.motion_start_frame),
            motion_end_frame=int(config.task.motion_end_frame),
        )
        environment = RLEnv(setup.scene, setup.reference, config=environment_config)
        assert setup.reference.num_frames == 500
        assert setup.reference.metadata.source_frame_playback
        assert isinstance(environment.action, SonicJointResidualAction)
        assert isinstance(environment.observation_strategy, ReconBodyObservation)
        assert isinstance(environment.objective, ReconBodyObjective)
        assert isinstance(environment.termination, ReconBodyTermination)
        assert isinstance(environment.reset_policy, ReconBodyReset)
        assert environment.observation_dim == 486
        assert environment.joint_actuator_limits is not None

        environment.set_voc_scale(0.0)
        environment.set_immediate_first_frame_probability(1.0)
        immediate_observation = environment.reset().numpy().copy()
        assert environment.reset_policy.reset_frame.numpy().tolist() == [0]
        assert environment.reset_policy.steps_since_reset.numpy().tolist() == [51]
        np.testing.assert_array_equal(environment.reset_policy.applied_voc_scale.numpy(), [0.0])
        np.testing.assert_array_equal(environment.reference_joint_q_offset.numpy(), 0.0)
        explicit_observation = environment.reset(frame_id=0).numpy().copy()
        np.testing.assert_array_equal(immediate_observation, explicit_observation)

        action = wp.zeros(environment.action.action_dim * environment.world_count, dtype=wp.float32)
        environment.reset(frame_id=0)
        eager = environment.step(action)
        assert all(np.isfinite(value.numpy()).all() for value in eager)
        history = environment.action.actor_action_history.numpy().reshape(3, 43)
        np.testing.assert_array_equal(history[:2], 0.0)

        environment.reset(frame_id=0)
        environment.capture_step()
        environment.reset(frame_id=0)
        captured = environment.step(action)
        assert all(np.isfinite(value.numpy()).all() for value in captured)
        assert environment.timestep.numpy().tolist() == [1]
