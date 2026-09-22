# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Algorithm-neutral JAX boundary for vectorized reinforcement-learning environments."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import flax
import jax
import jax.numpy as jnp

from flash_chord.lifecycle.curriculum import CurriculumStage
from flash_chord.lifecycle.termination import PackedTerminationDiagnostics, TerminationDiagnostics
from flash_chord.objectives.composition import CurriculumObjective, ObjectiveDiagnostics
from flash_chord.runtime.jax import from_jax, to_jax


@flax.struct.dataclass
class VectorStep:
    """One vectorized environment transition and completed-episode metrics."""

    observation: jax.Array
    reward: jax.Array | None
    terminated: jax.Array
    truncated: jax.Array
    episode_return: jax.Array | None
    episode_length: jax.Array | None
    episode_reference_progress: jax.Array | None
    objective_terms: jax.Array | None = None
    termination_causes: jax.Array | None = None
    tracking_errors: jax.Array | None = None
    terminal_observation: jax.Array | None = None
    critic_context: jax.Array | None = None
    terminal_critic_context: jax.Array | None = None


@runtime_checkable
class JAXVectorEnv(Protocol):
    """Minimal environment surface consumed by reinforcement-learning collectors."""

    world_count: int
    observation_dim: int
    action_dim: int

    def reset(self) -> jax.Array:
        """Reset all worlds and return the current observation."""
        ...

    def step(self, action: jax.Array) -> VectorStep:
        """Advance all worlds by one action."""
        ...


class WarpRLEnv:
    """Zero-copy JAX view adapter for :class:`flash_chord.envs.rl.RLEnv`."""

    def __init__(
        self,
        env,
        *,
        publish_diagnostics: bool = True,
        publish_step_metrics: bool = True,
    ) -> None:
        self.env = env
        self.publish_diagnostics = publish_diagnostics
        self.publish_step_metrics = publish_step_metrics
        self.world_count = env.world_count
        self.observation_dim = env.observation_dim
        self.action_dim = env.action.action_dim
        self.frame_dt = float(getattr(env, "frame_dt", 1.0))
        self.reset_to_first_frame_probability = getattr(env, "reset_to_first_frame_probability", None)
        self.immediate_first_frame_probability = getattr(env, "immediate_first_frame_probability", None)
        objective = getattr(env, "objective", None)
        if isinstance(objective, ObjectiveDiagnostics):
            self.objective_term_names = objective.term_names
            self._objective_terms = objective.terms
        else:
            self.objective_term_names = ()
            self._objective_terms = None
        termination = getattr(env, "termination", None)
        if isinstance(termination, TerminationDiagnostics):
            self.termination_cause_names = termination.cause_names
            self.tracking_error_names = termination.error_names
            self._termination_cause_arrays = termination.cause_arrays
            self._tracking_error_arrays = termination.error_arrays
        else:
            self.termination_cause_names = ()
            self.tracking_error_names = ()
            self._termination_cause_arrays = ()
            self._tracking_error_arrays = ()
        if isinstance(termination, PackedTerminationDiagnostics):
            expected_names = self.termination_cause_names + self.tracking_error_names
            if termination.packed_diagnostic_names != expected_names:
                raise ValueError(
                    f"packed termination diagnostics are {termination.packed_diagnostic_names}; "
                    f"expected {expected_names}"
                )
            self._packed_termination_diagnostics = termination.packed_diagnostics
        else:
            self._packed_termination_diagnostics = None
        self._episode_reference_progress_array = getattr(env, "episode_reference_progress", None)
        self.critic_context_names = tuple(getattr(env, "critic_context_names", ()))
        self.critic_context_dim = len(self.critic_context_names)
        terminal_observation = getattr(env, "terminal_observation", None)
        critic_context = getattr(env, "critic_context", None)
        terminal_critic_context = getattr(env, "terminal_critic_context", None)
        terminal_outputs = (terminal_observation, critic_context, terminal_critic_context)
        if any(output is None for output in terminal_outputs):
            if not all(output is None for output in terminal_outputs):
                raise ValueError("terminal observation and both critic contexts must be configured together")
            if self.critic_context_names:
                raise ValueError("critic context names require configured terminal-state buffers")
            self._terminal_observation_array = None
            self._critic_context_array = None
            self._terminal_critic_context_array = None
        else:
            if self.critic_context_dim == 0:
                raise ValueError("configured terminal-state buffers require non-empty critic context names")
            self._terminal_observation_array = terminal_observation
            self._critic_context_array = critic_context
            self._terminal_critic_context_array = terminal_critic_context

    @property
    def initial_critic_context(self) -> jax.Array:
        """Return the zero-copy context initialized by the most recent reset."""
        if self._critic_context_array is None:
            raise RuntimeError("environment was not configured to capture terminal state")
        return to_jax(
            self._critic_context_array,
            (self.world_count, self.critic_context_dim),
        )

    def reset(self) -> jax.Array:
        observation = self.env.reset()
        return to_jax(observation, (self.world_count, self.observation_dim))

    def step(self, action: jax.Array) -> VectorStep:
        expected = (self.world_count, self.action_dim)
        if action.shape != expected:
            raise ValueError(f"action has shape {action.shape}; expected {expected}")
        warp_action = from_jax(action).reshape((self.world_count * self.action_dim,))
        observation, _, terminated, truncated = self.env.step(warp_action)
        objective_terms = None
        if self._objective_terms is not None:
            objective_terms = to_jax(
                self._objective_terms,
                (self.world_count, len(self.objective_term_names)),
            )
        termination_causes = None
        tracking_errors = None
        if self.publish_diagnostics:
            termination_causes, tracking_errors = self.transition_diagnostics()
        reward = None
        episode_return = None
        episode_length = None
        episode_reference_progress = None
        if self.publish_step_metrics:
            reward, episode_return, episode_length, episode_reference_progress = self.transition_metrics()
        terminal_observation = None
        critic_context = None
        terminal_critic_context = None
        if self._terminal_observation_array is not None:
            terminal_observation = to_jax(
                self._terminal_observation_array,
                (self.world_count, self.observation_dim),
            )
            critic_context = to_jax(
                self._critic_context_array,
                (self.world_count, self.critic_context_dim),
            )
            terminal_critic_context = to_jax(
                self._terminal_critic_context_array,
                (self.world_count, self.critic_context_dim),
            )
        return VectorStep(
            observation=to_jax(observation, (self.world_count, self.observation_dim)),
            reward=reward,
            terminated=to_jax(terminated),
            truncated=to_jax(truncated),
            episode_return=episode_return,
            episode_length=episode_length,
            episode_reference_progress=episode_reference_progress,
            objective_terms=objective_terms,
            termination_causes=termination_causes,
            tracking_errors=tracking_errors,
            terminal_observation=terminal_observation,
            critic_context=critic_context,
            terminal_critic_context=terminal_critic_context,
        )

    def transition_metrics(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Create zero-copy views of latest reward and completed-episode summaries on demand."""
        return (
            to_jax(self.env.reward),
            to_jax(self.env.episode_return),
            to_jax(self.env.episode_length),
            to_jax(self.env.episode_reference_progress),
        )

    def transition_diagnostics(self) -> tuple[jax.Array | None, jax.Array | None]:
        """Reduce optional termination diagnostics on demand before the next environment step."""
        termination_causes = None
        if self._termination_cause_arrays:
            termination_causes = jnp.stack([to_jax(array) for array in self._termination_cause_arrays], axis=-1)
        tracking_errors = None
        if self._tracking_error_arrays:
            per_world_maxima = [
                to_jax(array, (self.world_count, array.size // self.world_count)).max(axis=-1)
                for array in self._tracking_error_arrays
            ]
            tracking_errors = jnp.stack(per_world_maxima, axis=-1)
        return termination_causes, tracking_errors

    def transition_logging_inputs(self) -> tuple[jax.Array, jax.Array]:
        """Return stream-ordered zero-copy inputs for the compiled logging-window reduction."""
        if self._episode_reference_progress_array is None:
            raise RuntimeError("environment does not expose completed-episode reference progress")
        if self._packed_termination_diagnostics is None:
            raise RuntimeError("environment termination does not expose packed diagnostics")
        diagnostic_width = len(self.termination_cause_names) + len(self.tracking_error_names)
        return (
            to_jax(self._episode_reference_progress_array, (self.world_count,)),
            to_jax(
                self._packed_termination_diagnostics,
                (self.world_count, diagnostic_width),
            ),
        )

    def set_curriculum(self, stage: CurriculumStage) -> None:
        """Update captured VOC, objective, and optional reset buffers between learner iterations."""
        if not isinstance(self.env.objective, CurriculumObjective):
            raise TypeError("curriculum objective weights require a CurriculumObjective strategy")
        stage.weights_for(self.env.objective.term_names)
        reset_setter = getattr(self.env, "set_reset_to_first_frame_probability", None)
        if reset_setter is None:
            if stage.reset_to_first_frame_probability is not None:
                raise TypeError(
                    "reset_to_first_frame_probability requires an environment exposing the corresponding setter"
                )
        else:
            self.reset_to_first_frame_probability = reset_setter(stage.reset_to_first_frame_probability)
        immediate_setter = getattr(self.env, "set_immediate_first_frame_probability", None)
        if immediate_setter is None:
            if stage.immediate_first_frame_probability is not None:
                raise TypeError(
                    "immediate_first_frame_probability requires an environment exposing the corresponding setter"
                )
        else:
            self.immediate_first_frame_probability = immediate_setter(stage.immediate_first_frame_probability)
        self.env.set_voc_scale(stage.voc_scale)
        self.env.objective.set_weights(stage.objective_weights)
