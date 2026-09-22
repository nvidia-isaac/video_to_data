# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared JAX vector-environment boundary."""

import numpy as np
import pytest


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    return jax, jnp


def test_vector_step_remains_a_jax_pytree():
    jax, jnp = _dependencies()

    from flash_chord.training.environment import VectorStep

    step = VectorStep(
        observation=jnp.ones((2, 3)),
        reward=jnp.ones(2),
        terminated=jnp.zeros(2),
        truncated=jnp.zeros(2),
        episode_return=jnp.zeros(2),
        episode_length=jnp.zeros(2, dtype=jnp.int32),
        episode_reference_progress=jnp.zeros(2),
    )

    leaves = jax.tree.leaves(step)
    assert len(leaves) == 7
    assert step.objective_terms is None
    assert step.termination_causes is None
    assert step.tracking_errors is None
    assert step.terminal_observation is None
    assert step.critic_context is None
    assert step.terminal_critic_context is None


def test_runtime_protocol_accepts_minimal_vector_environment():
    _, jnp = _dependencies()

    from flash_chord.training.environment import JAXVectorEnv, VectorStep

    class Env:
        world_count = 2
        observation_dim = 3
        action_dim = 1

        def reset(self):
            return jnp.zeros((self.world_count, self.observation_dim))

        def step(self, action):
            return VectorStep(
                observation=self.reset(),
                reward=jnp.zeros(self.world_count),
                terminated=jnp.zeros(self.world_count),
                truncated=jnp.zeros(self.world_count),
                episode_return=jnp.zeros(self.world_count),
                episode_length=jnp.zeros(self.world_count, dtype=jnp.int32),
                episode_reference_progress=jnp.zeros(self.world_count),
            )

    assert isinstance(Env(), JAXVectorEnv)


def test_warp_adapter_rejects_incorrect_action_shape_before_stepping():
    _, jnp = _dependencies()

    from flash_chord.training.environment import WarpRLEnv

    class Env:
        world_count = 2
        observation_dim = 3
        action = type("Action", (), {"action_dim": 1})()
        objective = None
        step_called = False

        def step(self, action):
            self.step_called = True
            raise AssertionError("shape validation must happen before the environment step")

    env = Env()
    adapter = WarpRLEnv(env)

    with pytest.raises(ValueError, match=r"action has shape \(2, 2\); expected \(2, 1\)"):
        adapter.step(jnp.zeros((2, 2)))
    assert env.step_called is False


def test_warp_adapter_updates_existing_curriculum_buffers():
    _dependencies()

    from flash_chord.lifecycle.curriculum import CurriculumStage
    from flash_chord.training.environment import WarpRLEnv

    class Objective:
        term_names = tuple(f"term_{index}" for index in range(9))
        assigned = None

        def set_weights(self, values):
            self.assigned = values

    class Env:
        world_count = 2
        observation_dim = 3
        action = type("Action", (), {"action_dim": 2})()
        objective = Objective()
        assigned_scale = None
        reset_to_first_frame_probability = 0.1
        immediate_first_frame_probability = 0.0
        assigned_reset_probabilities = []
        assigned_immediate_probabilities = []

        def set_voc_scale(self, scale):
            self.assigned_scale = scale

        def set_reset_to_first_frame_probability(self, probability):
            self.assigned_reset_probabilities.append(probability)
            self.reset_to_first_frame_probability = 0.1 if probability is None else probability
            return self.reset_to_first_frame_probability

        def set_immediate_first_frame_probability(self, probability):
            self.assigned_immediate_probabilities.append(probability)
            self.immediate_first_frame_probability = 0.0 if probability is None else probability
            return self.immediate_first_frame_probability

    env = Env()
    adapter = WarpRLEnv(env)
    weights = {name: float(index) for index, name in enumerate(Objective.term_names)}
    stage = CurriculumStage(
        0.25,
        weights,
        reset_to_first_frame_probability=1.0,
        immediate_first_frame_probability=0.5,
    )
    adapter.set_curriculum(stage)

    assert env.assigned_scale == 0.25
    assert env.objective.assigned == stage.objective_weights
    assert env.assigned_reset_probabilities == [1.0]
    assert adapter.reset_to_first_frame_probability == 1.0
    assert env.assigned_immediate_probabilities == [0.5]
    assert adapter.immediate_first_frame_probability == 0.5

    adapter.set_curriculum(CurriculumStage(0.25, weights))
    assert env.assigned_reset_probabilities == [1.0, None]
    assert adapter.reset_to_first_frame_probability == 0.1
    assert env.assigned_immediate_probabilities == [0.5, None]
    assert adapter.immediate_first_frame_probability == 0.0

    with pytest.raises(ValueError, match="curriculum objective weights mismatch"):
        adapter.set_curriculum(
            CurriculumStage(
                0.5,
                {"term_0": 1.0},
                reset_to_first_frame_probability=0.5,
            )
        )
    assert env.assigned_scale == 0.25
    assert env.objective.assigned == weights
    assert env.assigned_reset_probabilities == [1.0, None]
    assert adapter.reset_to_first_frame_probability == 0.1
    assert env.assigned_immediate_probabilities == [0.5, None]
    assert adapter.immediate_first_frame_probability == 0.0


def test_warp_adapter_rejects_reset_override_without_environment_capability():
    _dependencies()

    from flash_chord.lifecycle.curriculum import CurriculumStage
    from flash_chord.training.environment import WarpRLEnv

    class Objective:
        term_names = ("tracking",)

        def set_weights(self, values):
            pass

    class Env:
        world_count = 2
        observation_dim = 3
        action = type("Action", (), {"action_dim": 2})()
        objective = Objective()

        def set_voc_scale(self, scale):
            pass

    adapter = WarpRLEnv(Env())
    with pytest.raises(TypeError, match="reset_to_first_frame_probability"):
        adapter.set_curriculum(
            CurriculumStage(
                0.0,
                {"tracking": 1.0},
                reset_to_first_frame_probability=1.0,
            )
        )

    with pytest.raises(TypeError, match="immediate_first_frame_probability"):
        adapter.set_curriculum(
            CurriculumStage(
                0.0,
                {"tracking": 1.0},
                immediate_first_frame_probability=1.0,
            )
        )


@pytest.mark.gpu
def test_warp_adapter_exposes_fresh_stream_ordered_terminal_zero_copy_views():
    jax, jnp = _dependencies()
    wp = pytest.importorskip("warp")

    from flash_chord.training.environment import WarpRLEnv

    class Env:
        world_count = 2
        observation_dim = 3
        action = type("Action", (), {"action_dim": 1})()
        objective = None
        critic_context_names = (
            "applied_voc_scale",
            "target_voc_scale",
            "normalized_settling_progress",
            "normalized_reference_phase",
            "object_body_0_linear_velocity_w_x",
            "object_body_0_angular_velocity_w_x",
        )

        def __init__(self):
            self.step_value = 0
            self.action_pointers = []
            self.observation = wp.array([1, 2, 3, 4, 5, 6], dtype=wp.float32, device="cuda:0")
            self.terminal_observation = wp.array([11, 12, 13, 14, 15, 16], dtype=wp.float32, device="cuda:0")
            self.critic_context = wp.zeros(12, dtype=wp.float32, device="cuda:0")
            self.terminal_critic_context = wp.array(
                [0.25, 0.25, 1, 1, 2, 3, 0.25, 0.25, 1, 0, 4, 5],
                dtype=wp.float32,
                device="cuda:0",
            )
            self.reward = wp.zeros(2, dtype=wp.float32, device="cuda:0")
            self.done = wp.zeros(2, dtype=wp.int32, device="cuda:0")
            self.truncation = wp.zeros(2, dtype=wp.int32, device="cuda:0")
            self.episode_return = wp.zeros(2, dtype=wp.float32, device="cuda:0")
            self.episode_length = wp.zeros(2, dtype=wp.int32, device="cuda:0")
            self.episode_reference_progress = wp.zeros(2, dtype=wp.float32, device="cuda:0")
            self.termination = type(
                "Termination",
                (),
                {
                    "cause_names": ("failure",),
                    "cause_arrays": (wp.array([1, 0], dtype=wp.int32, device="cuda:0"),),
                    "error_names": ("error",),
                    "error_arrays": (wp.array([1, 2, 3, 4], dtype=wp.float32, device="cuda:0"),),
                    "packed_diagnostic_names": ("failure", "error"),
                    "packed_diagnostics": wp.array([1, 2, 0, 4], dtype=wp.float32, device="cuda:0"),
                },
            )()

        def reset(self):
            return self.observation

        def step(self, action):
            assert action.shape == (2,)
            self.action_pointers.append(action.ptr)
            self.step_value += 1
            self.terminal_observation.fill_(10.0 + self.step_value)
            self.critic_context.fill_(20.0 + self.step_value)
            self.terminal_critic_context.fill_(30.0 + self.step_value)
            return self.observation, self.reward, self.done, self.truncation

    with wp.ScopedDevice("cuda:0"):
        env = Env()
        adapter = WarpRLEnv(env)
        reset = adapter.reset()
        first_action = jnp.zeros((2, 1), dtype=jnp.float32)
        step = adapter.step(first_action)
        consume_terminal = jax.jit(
            lambda terminal, context, terminal_context: jnp.stack(
                (terminal.sum(), context.sum(), terminal_context.sum())
            )
        )
        first_sums = consume_terminal(
            step.terminal_observation,
            step.critic_context,
            step.terminal_critic_context,
        )
        next_action = jnp.broadcast_to(first_sums[0] * 0.0, (2, 1))
        next_step = adapter.step(next_action)
        second_sums = consume_terminal(
            next_step.terminal_observation,
            next_step.critic_context,
            next_step.terminal_critic_context,
        )
        quiet_adapter = WarpRLEnv(
            env,
            publish_diagnostics=False,
            publish_step_metrics=False,
        )
        quiet_action = jnp.broadcast_to(second_sums[0] * 0.0, (2, 1))
        quiet_step = quiet_adapter.step(quiet_action)
        on_demand_metrics = quiet_adapter.transition_metrics()
        on_demand_causes, on_demand_errors = quiet_adapter.transition_diagnostics()
        episode_progress, packed_diagnostics = quiet_adapter.transition_logging_inputs()

    assert adapter.critic_context_dim == 6
    assert adapter.critic_context_names == Env.critic_context_names
    assert reset.shape == (2, 3)
    assert adapter.initial_critic_context.shape == (2, 6)
    assert step.observation.shape == (2, 3)
    assert step.terminal_observation.shape == (2, 3)
    assert step.critic_context.shape == (2, 6)
    assert step.terminal_critic_context.shape == (2, 6)
    np.testing.assert_array_equal(np.asarray(step.termination_causes), [[1], [0]])
    np.testing.assert_array_equal(np.asarray(step.tracking_errors), [[2], [4]])
    assert quiet_step.termination_causes is None
    assert quiet_step.tracking_errors is None
    assert quiet_step.reward is None
    assert quiet_step.episode_return is None
    assert quiet_step.episode_length is None
    assert quiet_step.episode_reference_progress is None
    for metric in on_demand_metrics:
        np.testing.assert_array_equal(np.asarray(metric), [0, 0])
    np.testing.assert_array_equal(np.asarray(on_demand_causes), [[1], [0]])
    np.testing.assert_array_equal(np.asarray(on_demand_errors), [[2], [4]])
    np.testing.assert_array_equal(np.asarray(episode_progress), [0, 0])
    np.testing.assert_array_equal(np.asarray(packed_diagnostics), [[1, 2], [0, 4]])
    assert episode_progress.unsafe_buffer_pointer() == env.episode_reference_progress.ptr
    assert packed_diagnostics.unsafe_buffer_pointer() == env.termination.packed_diagnostics.ptr
    assert step.terminal_observation.unsafe_buffer_pointer() == env.terminal_observation.ptr
    assert step.critic_context.unsafe_buffer_pointer() == env.critic_context.ptr
    assert step.terminal_critic_context.unsafe_buffer_pointer() == env.terminal_critic_context.ptr
    assert env.action_pointers[0] == first_action.unsafe_buffer_pointer()
    assert env.action_pointers[1] == next_action.unsafe_buffer_pointer()
    np.testing.assert_array_equal(np.asarray(first_sums), [66, 252, 372])
    np.testing.assert_array_equal(np.asarray(second_sums), [72, 264, 384])


def test_warp_adapter_default_has_no_terminal_state_fields():
    _dependencies()

    from flash_chord.training.environment import WarpRLEnv

    class Env:
        world_count = 2
        observation_dim = 3
        action = type("Action", (), {"action_dim": 1})()
        objective = None
        termination = None

    adapter = WarpRLEnv(Env())
    assert adapter.critic_context_dim == 0
    with pytest.raises(RuntimeError, match="capture terminal state"):
        _ = adapter.initial_critic_context
