# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for batched object-pose recording against a live environment."""

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR

pytestmark = [pytest.mark.gpu, pytest.mark.sequence_data]

_VEGA_MIXER = (
    ASSETS_DIR
    / "human_motion_data"
    / "arctic"
    / "arctic_processed"
    / "sequence_id=dataset_s01_mixer_use_01"
    / "robot_name=vega_sharpa"
    / "data.parquet"
)

_DISABLE_FAILURE_TERMS = (
    "termination.wrist_position.enabled=false",
    "termination.wrist_orientation.enabled=false",
    "termination.object_position.enabled=false",
    "termination.object_orientation.enabled=false",
)


def _build_env(*extra_overrides):
    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    config = compose_config(
        "train",
        overrides=(
            f"task.parquet='{_VEGA_MIXER}'",
            "task.motion_speed=0.5",
            "scene.world_count=2",
            "scene.decompose_objects=false",
            "embodiment=dexmate_sharpa",
            "action=residual_joint_position",
            "observation=articulated_arm",
            "collision=manipulation",
            "env.auto_reset=false",
            "env.voc_scale=0.0",
            "reset.always_reset_to_first_frame=true",
            "reset.voc_decay_steps=0",
            *extra_overrides,
        ),
    )
    environment_config = instantiate_typed(config.env, RLEnvConfig)
    embodiment = instantiate_typed(config.embodiment, Embodiment)
    collision = instantiate_typed(config.collision, CollisionPolicy)
    setup = setup_scene(
        parquet=config.task.parquet,
        control_fps=environment_config.sim.fps,
        motion_speed=config.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=config.scene.world_count,
        include_support=config.scene.support,
        decompose_objects=config.scene.decompose_objects,
    )
    env = RLEnv(setup.scene, setup.reference, config=environment_config)
    env.reset()
    env.capture_step()
    return env, setup


def _zero_policy(action_dim: int):
    import jax.numpy as jnp

    def policy_action(_state, observation, key):
        return jnp.zeros((observation.shape[0], action_dim), dtype=jnp.float32), key

    return policy_action


def _record(env, setup, step_count: int):
    from flash_chord.evaluation.rollout import record_object_pose_rollout
    from flash_chord.training.environment import WarpRLEnv

    jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    observation = jax_env.reset()
    return record_object_pose_rollout(
        env,
        jax_env,
        _zero_policy(env.action.action_dim),
        None,
        None,
        observation,
        reference=setup.reference,
        step_count=step_count,
        progress_interval=0,
    )


def test_recorded_poses_match_a_direct_body_readback():
    from flash_chord.evaluation.rollout import record_object_pose_rollout
    from flash_chord.training.environment import WarpRLEnv

    env, setup = _build_env(*_DISABLE_FAILURE_TERMS)
    jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    observation = jax_env.reset()
    rollout = record_object_pose_rollout(
        env,
        jax_env,
        _zero_policy(env.action.action_dim),
        None,
        None,
        observation,
        reference=setup.reference,
        step_count=4,
        progress_interval=0,
    )
    body_ids = np.asarray(env.command.body_ids_w.numpy(), dtype=np.int64)
    final = np.asarray(env.state_0.body_q.numpy(), dtype=np.float64).reshape(-1, 7)
    expected = final[body_ids].reshape(env.world_count, -1, 7)
    assert rollout.achieved_pose_w[-1] == pytest.approx(expected, abs=1e-5)


def test_every_world_runs_the_whole_sequence_without_terminating():
    env, setup = _build_env(*_DISABLE_FAILURE_TERMS)
    steps = setup.reference.num_frames
    rollout = _record(env, setup, steps)

    assert not rollout.completion.terminated.any()
    assert rollout.completion.truncated.all()
    assert rollout.completion.completion_step.tolist() == [steps - 1] * env.world_count
    assert rollout.achieved_pose_w.shape[0] == steps
    # A zero-action rollout can diverge once the object is dropped; the guard must report and
    # zero those worlds rather than leaving a NaN to reach scoring.
    assert np.isfinite(rollout.achieved_pose_w).all()
    assert set(rollout.non_finite_worlds) <= set(range(env.world_count))


def test_enabled_failure_terms_freeze_the_reference_and_are_rejected():
    from flash_chord.evaluation.metrics import TrackingThresholds, chord_success

    env, setup = _build_env()
    rollout = _record(env, setup, setup.reference.num_frames)

    assert rollout.completion.terminated.any(), "a zero-action rollout should fail object tracking"
    with pytest.raises(ValueError, match="must disable the tracking-failure termination"):
        chord_success(rollout.tracking_error, TrackingThresholds(object_position_m=0.1), rollout.completion)


def test_tracking_error_matches_the_published_diagnostics():
    env, setup = _build_env(*_DISABLE_FAILURE_TERMS)
    rollout = _record(env, setup, 3)

    width = len(env.termination.packed_diagnostic_names)
    packed = np.asarray(env.termination.packed_diagnostics.numpy(), dtype=np.float64).reshape(env.world_count, width)
    assert rollout.tracking_error[-1] == pytest.approx(packed[:, width - 4 :], abs=1e-6)
    assert (rollout.tracking_error >= 0.0).all()


def test_reference_poses_use_xyzw_and_track_the_reference_arrays():
    from flash_chord.evaluation.rollout import reference_object_pose

    _, setup = _build_env(*_DISABLE_FAILURE_TERMS)
    steps = 5
    poses = reference_object_pose(setup.reference, 0, steps)
    position = np.asarray(setup.reference.object_body_pos_w())[:steps]
    quaternion_wxyz = np.asarray(setup.reference.object_body_quat_w())[:steps]

    assert poses[..., :3] == pytest.approx(position)
    assert poses[..., 6] == pytest.approx(quaternion_wxyz[..., 0])
    assert poses[..., 3:6] == pytest.approx(quaternion_wxyz[..., 1:])


def test_body_object_ids_assign_every_reference_body():
    from flash_chord.evaluation.rollout import body_object_ids

    env, _ = _build_env(*_DISABLE_FAILURE_TERMS)
    layout = env.command.layout
    ids = body_object_ids(layout)

    assert ids.shape == (layout.num_bodies,)
    assert set(ids.tolist()) == set(range(layout.num_objects))
    assert np.bincount(ids, minlength=layout.num_objects).sum() == layout.num_bodies


def test_sampled_vertices_come_from_the_simulated_geometry():
    """Vertices must carry the model's own scale and shape transform, not the raw mesh file."""
    import newton

    from flash_chord.evaluation.rollout import sample_object_vertices

    env, setup = _build_env(*_DISABLE_FAILURE_TERMS)
    radii = np.asarray(setup.reference.object_mesh_radius(), dtype=np.float64)
    vertices = sample_object_vertices(env.scene, count=64, seed=3)

    assert vertices.shape == (radii.size, 64, 3)
    assert np.isfinite(vertices).all()
    assert sample_object_vertices(env.scene, count=64, seed=3) == pytest.approx(vertices)
    assert not np.allclose(sample_object_vertices(env.scene, count=64, seed=4), vertices)

    model = env.scene.model
    visible = int(newton.ShapeFlags.VISIBLE)
    mesh_type = int(newton.GeoType.MESH)
    flags = np.asarray(model.shape_flags.numpy())
    shape_type = np.asarray(model.shape_type.numpy())
    shape_scale = np.asarray(model.shape_scale.numpy(), dtype=np.float64)
    binding = env.scene.objects[0]
    chosen = [
        s
        for s in range(binding.shapes.start, binding.shapes.stop)
        if (flags[s] & visible) and shape_type[s] == mesh_type
    ]
    # The source mesh survives convex decomposition as the only visible shape per body.
    assert len(chosen) == len(binding.bodies)
    scaled_extent = max(
        float(np.linalg.norm(np.asarray(model.shape_source[s].vertices) * shape_scale[s][None, :], axis=-1).max())
        for s in chosen
    )
    assert float(np.linalg.norm(vertices, axis=-1).max()) <= scaled_extent + 1e-9
