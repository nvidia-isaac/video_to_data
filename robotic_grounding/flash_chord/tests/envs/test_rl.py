# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RL output preservation, auto-reset ordering, and graph capture."""

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


def _build_vega_rl_env(*extra_overrides, config_name="train"):
    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    config = compose_config(
        config_name,
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
    return setup, RLEnv(setup.scene, setup.reference, config=environment_config)


def test_publish_rl_transition_preserves_outputs_and_episode_metrics():
    import warp as wp

    from flash_chord.envs.rl import publish_rl_transition

    with wp.ScopedDevice("cuda:0"):
        score = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32)
        terminated = wp.array([0, 1, 0, 1], dtype=wp.int32)
        truncated = wp.array([0, 0, 1, 1], dtype=wp.int32)
        timestep = wp.array([5, 6, 8, 9], dtype=wp.int32)
        episode_start_frame = wp.array([0, 2, 4, 8], dtype=wp.int32)
        running_return = wp.array([10.0, 20.0, 30.0, 40.0], dtype=wp.float32)
        running_length = wp.array([5, 6, 7, 8], dtype=wp.int32)
        reward = wp.zeros(4, dtype=wp.float32)
        done = wp.zeros(4, dtype=wp.int32)
        truncation = wp.zeros(4, dtype=wp.int32)
        reset_mask = wp.zeros(4, dtype=wp.int32)
        episode_return = wp.zeros(4, dtype=wp.float32)
        episode_length = wp.zeros(4, dtype=wp.int32)
        episode_reference_progress = wp.zeros(4, dtype=wp.float32)
        wp.launch(
            publish_rl_transition,
            dim=4,
            inputs=[score, terminated, truncated, timestep, episode_start_frame, 9, 1],
            outputs=[
                running_return,
                running_length,
                reward,
                done,
                truncation,
                reset_mask,
                episode_return,
                episode_length,
                episode_reference_progress,
            ],
        )

    np.testing.assert_allclose(reward.numpy(), [1, 2, 3, 4])
    assert done.numpy().tolist() == [0, 1, 0, 1]
    assert truncation.numpy().tolist() == [0, 0, 1, 1]
    assert reset_mask.numpy().tolist() == [0, 1, 1, 1]
    np.testing.assert_allclose(running_return.numpy(), [11, 0, 0, 0])
    assert running_length.numpy().tolist() == [6, 0, 0, 0]
    np.testing.assert_allclose(episode_return.numpy(), [0, 22, 33, 44])
    assert episode_length.numpy().tolist() == [0, 7, 8, 9]
    np.testing.assert_allclose(episode_reference_progress.numpy(), [0, 4 / 7, 0.8, 1.0])


@pytest.mark.slow
@pytest.mark.sequence_data
def test_vega_rl_eager_capture_and_zero_copy_jax_parity():
    import jax
    import jax.numpy as jnp
    import warp as wp

    from flash_chord.envs.rl import RLEnv
    from flash_chord.training.environment import WarpRLEnv

    with wp.ScopedDevice("cuda:0"):
        setup, eager_env = _build_vega_rl_env(
            "collision=kinematic",
            "scene.support=false",
            "observation.object_body_velocity.enabled=true",
            "env.critic_context.object_body_velocity=false",
            config_name="train_flash_sac",
        )
        reference = setup.reference
        assert reference.fps == 20.0
        assert setup.support_usda is None
        assert eager_env.observation_dim == 778
        velocity_block = eager_env.observation_strategy.block_names.index("object_body_velocity_w_mps_radps")
        assert eager_env.observation_strategy.block_ranges[velocity_block] == (170, 182)
        assert len(eager_env.critic_context_names) == 4
        assert eager_env.observation_dim + len(eager_env.critic_context_names) + eager_env.action.action_dim == 840
        captured_env = RLEnv(setup.scene, reference, config=eager_env.config)
        jax_native_env = RLEnv(setup.scene, reference, config=eager_env.config)
        action = wp.zeros(2 * eager_env.action.action_dim, dtype=wp.float32)

        eager_env.reset(frame_id=0)
        eager_env.step(action)
        eager_env.reset(frame_id=0)
        eager_outputs = eager_env.step(action)
        eager = tuple(output.numpy().copy() for output in eager_outputs)
        eager_joint_q = eager_env.state_0.joint_q.numpy().copy()
        eager_context = eager_env.critic_context.numpy().copy()

        captured_env.reset(frame_id=0)
        captured_env.capture_step()
        captured_env.reset(frame_id=0)
        captured_outputs = captured_env.step()
        captured = tuple(output.numpy().copy() for output in captured_outputs)
        captured_joint_q = captured_env.state_0.joint_q.numpy().copy()
        captured_context = captured_env.critic_context.numpy().copy()

        jax_native_env.reset(frame_id=0)
        jax_native_env.capture_step()
        jax_native_env.reset(frame_id=0)
        jax_env = WarpRLEnv(jax_native_env, publish_diagnostics=False)
        assert jax_env.critic_context_dim == 4
        jax_step = jax_env.step(jnp.zeros((2, jax_native_env.action.action_dim), dtype=jnp.float32))
        jax.block_until_ready(jax_step.observation)
        through_jax = (
            np.asarray(jax_step.observation).reshape(-1),
            np.asarray(jax_step.reward),
            np.asarray(jax_step.terminated),
            np.asarray(jax_step.truncated),
        )

    for output in (*eager, eager_joint_q):
        assert np.isfinite(output).all()
    for actual, expected in zip(captured, eager, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)
    np.testing.assert_allclose(captured_joint_q, eager_joint_q, atol=1.0e-5, rtol=1.0e-5)
    np.testing.assert_allclose(captured_context, eager_context, atol=1.0e-4, rtol=1.0e-5)
    for actual, expected in zip(through_jax, captured, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5)
    np.testing.assert_allclose(
        np.asarray(jax_step.critic_context).reshape(-1),
        jax_native_env.critic_context.numpy(),
        atol=1.0e-4,
        rtol=1.0e-5,
    )


@pytest.mark.slow
@pytest.mark.sequence_data
def test_vega_zero_voc_leaves_object_articulation_passive():
    import jax
    import jax.numpy as jnp
    import newton
    import warp as wp

    from flash_chord.training.environment import WarpRLEnv

    with wp.ScopedDevice("cuda:0"):
        _, env = _build_vega_rl_env("env.voc_scale=0.0", "reset.voc_decay_steps=0")
        action = wp.zeros(2 * env.action.action_dim, dtype=wp.float32)
        articulation_q_ids = np.asarray(env.command.layout.articulation_q_ids)
        articulation_dof_ids = np.asarray(env.command.layout.articulation_dof_ids)
        assert articulation_q_ids.size > 0

        def displace_object_articulations():
            joint_q = env.state_0.joint_q.numpy()
            for world in range(env.world_count):
                joint_q[world * env.command.num_joint_q + articulation_q_ids] += 0.2
            env.state_0.joint_q.assign(joint_q)
            newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)

        def assert_articulation_drive_is_passive():
            assert env.object_joint_drive is not None
            armature = env.solver.mjw_model.dof_armature.numpy()
            friction = env.solver.mjw_model.dof_frictionloss.numpy()
            np.testing.assert_allclose(armature[:, articulation_dof_ids], 0.01)
            np.testing.assert_allclose(friction[:, articulation_dof_ids], 0.1)
            position_ids = env.object_joint_drive.position_actuator_ids.numpy()
            velocity_ids = env.object_joint_drive.velocity_actuator_ids.numpy()
            actuator_ids = np.concatenate((position_ids, velocity_ids))
            gain = env.solver.mjw_model.actuator_gainprm.numpy()
            bias = env.solver.mjw_model.actuator_biasprm.numpy()
            np.testing.assert_array_equal(gain[:, actuator_ids], 0.0)
            np.testing.assert_array_equal(bias[:, actuator_ids], 0.0)
            actuator_force = env.solver.mjw_data.actuator_force.numpy()
            np.testing.assert_array_equal(actuator_force[:, actuator_ids], 0.0)
            joint_f = env.control.joint_f.numpy().reshape(env.world_count, env.command.num_joint_dof)
            np.testing.assert_array_equal(joint_f[:, articulation_dof_ids], 0.0)

        env.reset()
        np.testing.assert_array_equal(env.reset_policy.applied_voc_scale.numpy(), 0.0)
        displace_object_articulations()
        env.step(action)
        assert_articulation_drive_is_passive()

        env.reset(frame_id=200)
        displace_object_articulations()
        env.step(action)
        assert_articulation_drive_is_passive()

        env.reset(frame_id=200)
        env.capture_step()
        env.reset(frame_id=200)
        displace_object_articulations()
        env.step()
        assert_articulation_drive_is_passive()

        env.reset(frame_id=200)
        displace_object_articulations()
        jax_env = WarpRLEnv(env, publish_diagnostics=False)
        step = jax_env.step(jnp.zeros((env.world_count, env.action.action_dim), dtype=jnp.float32))
        jax.block_until_ready(step.observation)
        assert_articulation_drive_is_passive()


@pytest.mark.slow
@pytest.mark.sequence_data
def test_rl_env_auto_reset_preserves_transition_outputs_and_capture():
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.lifecycle.termination import ThresholdTerminationTermConfig, TrackingTerminationConfig
    from flash_chord.scene.builder import build_scene
    from flash_chord.scene.collision import CollisionPolicy

    class _ConstantObjective:
        def __init__(self, values):
            self.values = wp.array(values, dtype=wp.float32)
            self.score = wp.zeros(len(values), dtype=wp.float32)
            self.terms = wp.zeros(len(values), dtype=wp.float32)
            self.weights = wp.ones(1, dtype=wp.float32)

        def evaluate(self, state, timestep, terminated):
            wp.copy(self.score, self.values)

        def reset(self, reset_mask):
            pass

    reference = load_mano_sharpa(str(_HOT3D), control_fps=50.0)
    embodiment = SharpaHands()
    termination = TrackingTerminationConfig(
        wrist_position=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
        wrist_orientation=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
        object_position=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
        object_orientation=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
    )
    config = RLEnvConfig(termination=termination, objective=None)

    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(
            embodiment,
            reference,
            world_count=2,
            collision=CollisionPolicy(robot_scope="none"),
            decompose_objects=False,
        )
        env = RLEnv(
            scene,
            reference,
            config=config,
            objective=_ConstantObjective([1.0, 2.0]),
        )
        assert env.terminal_observation is None
        assert env.critic_context is None
        assert env.terminal_critic_context is None
        reset_frame = wp.array([reference.num_frames - 1, 0], dtype=wp.int32)
        action = wp.zeros(2 * env.action.action_dim, dtype=wp.float32)

        env.set_voc_scale(0.25)
        env.reset(frame_id=None, reset_frame=reset_frame)
        env.reset_policy.reset_count.assign([0, 0])
        eager_observation, eager_reward, eager_done, eager_truncation = env.step(action)
        eager = {
            "observation": eager_observation.numpy().copy(),
            "reward": eager_reward.numpy().copy(),
            "done": eager_done.numpy().copy(),
            "truncation": eager_truncation.numpy().copy(),
            "joint_q": env.state_0.joint_q.numpy().copy(),
            "joint_qd": env.state_0.joint_qd.numpy().copy(),
            "timestep": env.timestep.numpy().copy(),
            "finger_scale": env.reset_policy.finger_scale.numpy().copy(),
            "applied_voc_scale": env.reset_policy.applied_voc_scale.numpy().copy(),
            "steps_since_reset": env.reset_policy.steps_since_reset.numpy().copy(),
            "episode_reference_progress": env.episode_reference_progress.numpy().copy(),
        }

        np.testing.assert_allclose(eager["reward"], [1, 2])
        assert eager["done"].tolist() == [0, 0]
        assert eager["truncation"].tolist() == [1, 0]
        assert env.reset_mask.numpy().tolist() == [1, 0]
        assert env.truncated.numpy().tolist() == [0, 0]
        sampled_frame = int(eager["timestep"][0])
        assert 0 <= sampled_frame < reference.num_frames - 1
        assert eager["timestep"].tolist() == [sampled_frame, 1]
        assert env.episode_step.numpy().tolist() == [0, 1]
        assert 0.0 <= eager["finger_scale"][0] <= 0.7
        assert eager["finger_scale"][1] == 1.0
        np.testing.assert_allclose(eager["applied_voc_scale"], [1.0, 0.25])
        assert eager["steps_since_reset"].tolist() == [0, 21]
        np.testing.assert_allclose(env.running_return.numpy(), [0, 2])
        assert env.running_length.numpy().tolist() == [0, 1]
        np.testing.assert_allclose(env.episode_return.numpy(), [1, 0])
        assert env.episode_length.numpy().tolist() == [1, 0]
        np.testing.assert_allclose(env.episode_reference_progress.numpy(), [1, 0])

        per_world_q = scene.model.joint_coord_count // scene.world_count
        joint_q = eager["joint_q"].reshape(2, per_world_q)
        expected_reset = env.reset_table.ref_joint_q.numpy()[sampled_frame].copy()
        for finger, q_id in enumerate(env.reset_table.finger_q_ids.numpy()):
            expected_reset[q_id] = np.clip(
                eager["finger_scale"][0] * expected_reset[q_id],
                env.reset_table.finger_lower.numpy()[finger],
                env.reset_table.finger_upper.numpy()[finger],
            )
        np.testing.assert_allclose(joint_q[0], expected_reset, atol=1e-6)
        observation = eager["observation"].reshape(2, env.observation_dim)
        contact_start = env.observation_strategy.layout.contact_position_starts[0]
        np.testing.assert_allclose(observation[0, contact_start:], 0.0)
        assert np.isfinite(eager["observation"]).all()

        env.reset(frame_id=None, reset_frame=reset_frame)
        env.reset_policy.reset_count.assign([0, 0])
        env.capture_step()
        env.reset(frame_id=None, reset_frame=reset_frame)
        env.reset_policy.reset_count.assign([0, 0])
        captured_observation, captured_reward, captured_done, captured_truncation = env.step()
        captured = {
            "observation": captured_observation.numpy(),
            "reward": captured_reward.numpy(),
            "done": captured_done.numpy(),
            "truncation": captured_truncation.numpy(),
            "joint_q": env.state_0.joint_q.numpy(),
            "joint_qd": env.state_0.joint_qd.numpy(),
            "timestep": env.timestep.numpy(),
            "finger_scale": env.reset_policy.finger_scale.numpy(),
            "applied_voc_scale": env.reset_policy.applied_voc_scale.numpy(),
            "steps_since_reset": env.reset_policy.steps_since_reset.numpy(),
            "episode_reference_progress": env.episode_reference_progress.numpy(),
        }

    for name in eager:
        np.testing.assert_allclose(captured[name], eager[name], atol=1e-6)


@pytest.mark.slow
@pytest.mark.sequence_data
def test_rl_env_terminal_state_capture_is_pre_reset_and_graph_capturable():
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.embodiments.sharpa_hands import SharpaHands
    from flash_chord.envs.rl import CriticContextConfig, RLEnv, RLEnvConfig
    from flash_chord.lifecycle.termination import ThresholdTerminationTermConfig, TrackingTerminationConfig
    from flash_chord.scene.builder import build_scene
    from flash_chord.scene.collision import CollisionPolicy

    class _ConstantObjective:
        def __init__(self, values):
            self.values = wp.array(values, dtype=wp.float32)
            self.score = wp.zeros(len(values), dtype=wp.float32)
            self.terms = wp.zeros(len(values), dtype=wp.float32)
            self.weights = wp.ones(1, dtype=wp.float32)

        def evaluate(self, state, timestep, terminated):
            wp.copy(self.score, self.values)

        def reset(self, reset_mask):
            pass

    reference = load_mano_sharpa(str(_HOT3D), control_fps=50.0)
    embodiment = SharpaHands()
    termination = TrackingTerminationConfig(
        wrist_position=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
        wrist_orientation=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
        object_position=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
        object_orientation=ThresholdTerminationTermConfig(threshold=0.0, enabled=False),
    )
    config = RLEnvConfig(
        termination=termination,
        objective=None,
        capture_terminal_state=True,
        critic_context=CriticContextConfig(reference_phase=True, object_body_velocity=True),
    )

    with wp.ScopedDevice("cuda:0"):
        scene = build_scene(
            embodiment,
            reference,
            world_count=2,
            collision=CollisionPolicy(robot_scope="none"),
            decompose_objects=False,
        )
        env = RLEnv(
            scene,
            reference,
            config=config,
            objective=_ConstantObjective([1.0, 2.0]),
        )
        reset_frame = wp.array([reference.num_frames - 1, 0], dtype=wp.int32)
        action = wp.zeros(2 * env.action.action_dim, dtype=wp.float32)
        env.set_voc_scale(0.25)
        env.reset(frame_id=None, reset_frame=reset_frame)
        context_dim = len(env.critic_context_names)
        velocity_start = 4
        assert env.critic_context_names[:4] == (
            "applied_voc_scale",
            "target_voc_scale",
            "normalized_settling_progress",
            "normalized_reference_phase",
        )
        assert context_dim == 4 + 6 * env.command.layout.num_bodies
        initial_context = env.critic_context.numpy().reshape(2, context_dim)
        np.testing.assert_allclose(
            initial_context[:, :3],
            [[0.25, 0.25, 1.0], [0.25, 0.25, 1.0]],
        )
        np.testing.assert_allclose(initial_context[:, 3], [1.0, 0.0])
        np.testing.assert_allclose(initial_context[:, velocity_start:], 0.0, atol=1.0e-6)
        env.reset_policy.reset_count.assign([0, 0])
        observation, _, done, truncation = env.step(action)
        eager = {
            "observation": observation.numpy().copy(),
            "terminal_observation": env.terminal_observation.numpy().copy(),
            "critic_context": env.critic_context.numpy().copy(),
            "terminal_critic_context": env.terminal_critic_context.numpy().copy(),
            "done": done.numpy().copy(),
            "truncation": truncation.numpy().copy(),
        }

        assert eager["done"].tolist() == [0, 0]
        assert eager["truncation"].tolist() == [1, 0]
        terminal_observation = eager["terminal_observation"].reshape(2, env.observation_dim)
        post_reset_observation = eager["observation"].reshape(2, env.observation_dim)
        assert not np.allclose(terminal_observation[0], post_reset_observation[0])
        np.testing.assert_allclose(terminal_observation[1], post_reset_observation[1], atol=1.0e-6)
        terminal_context = eager["terminal_critic_context"].reshape(2, context_dim)
        post_reset_context = eager["critic_context"].reshape(2, context_dim)
        np.testing.assert_allclose(
            terminal_context[:, :3],
            [[0.25, 0.25, 1.0], [0.25, 0.25, 1.0]],
        )
        np.testing.assert_allclose(
            post_reset_context[:, :3],
            [[1.0, 0.25, 0.0], [0.25, 0.25, 1.0]],
        )
        assert terminal_context[0, 3] == pytest.approx(1.0)
        assert post_reset_context[0, 3] == pytest.approx(
            env.reset_policy.reset_frame.numpy()[0] / (reference.num_frames - 1)
        )
        assert post_reset_context[1, 3] == pytest.approx(1.0 / (reference.num_frames - 1))
        np.testing.assert_allclose(terminal_context[1], post_reset_context[1], atol=1.0e-6)

        object_body_ids_w = env.command.body_ids_w.numpy().reshape(2, -1)
        body_qd = env.state_0.body_qd.numpy()
        expected_velocity = body_qd[object_body_ids_w].reshape(2, -1)
        np.testing.assert_allclose(post_reset_context[:, velocity_start:], expected_velocity, atol=1.0e-6)

        env.reset(frame_id=None, reset_frame=reset_frame)
        env.reset_policy.reset_count.assign([0, 0])
        env.capture_step()
        env.reset(frame_id=None, reset_frame=reset_frame)
        env.reset_policy.reset_count.assign([0, 0])
        observation, _, done, truncation = env.step()
        captured = {
            "observation": observation.numpy(),
            "terminal_observation": env.terminal_observation.numpy(),
            "critic_context": env.critic_context.numpy(),
            "terminal_critic_context": env.terminal_critic_context.numpy(),
            "done": done.numpy(),
            "truncation": truncation.numpy(),
        }

    for name in eager:
        tolerance = 1.0e-4 if "critic_context" in name else 1.0e-6
        np.testing.assert_allclose(captured[name], eager[name], atol=tolerance)
