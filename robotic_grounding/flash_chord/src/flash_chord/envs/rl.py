# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL episode semantics over the shared device-native base environment."""

from __future__ import annotations

from dataclasses import dataclass, field

import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.envs.base import BaseEnv, BaseEnvConfig
from flash_chord.envs.observation import (
    Observation,
    ObservationInto,
    ObservationSpec,
    PolicyObservationConfig,
)
from flash_chord.lifecycle.reset import (
    RESET_CONTEXT_NAMES,
    FirstFrameResetCurriculum,
    ImmediateFirstFrameResetCurriculum,
    ReferenceOffsetReset,
    ReferenceResetConfig,
    ResetContext,
    ResetPolicy,
    ResetSpec,
)
from flash_chord.lifecycle.termination import Termination
from flash_chord.objectives.composition import Objective
from flash_chord.runtime.actions import Action, PolicyAction
from flash_chord.scene.builder import Scene


@dataclass(frozen=True)
class CriticContextConfig:
    """Optional critic-only state appended to the reset/VOC context."""

    reference_phase: bool = False
    object_body_velocity: bool = False


@dataclass
class RLEnvConfig(BaseEnvConfig):
    """Shared dynamics plus RL observation and automatic-reset choices."""

    observation: PolicyObservationConfig = field(default_factory=PolicyObservationConfig)
    reset: ReferenceResetConfig = field(default_factory=ReferenceResetConfig)
    auto_reset: bool = True
    capture_terminal_state: bool = False
    critic_context: CriticContextConfig = field(default_factory=CriticContextConfig)


_OBJECT_BODY_VELOCITY_NAMES = (
    "linear_velocity_w_x",
    "linear_velocity_w_y",
    "linear_velocity_w_z",
    "angular_velocity_w_x",
    "angular_velocity_w_y",
    "angular_velocity_w_z",
)
_RESET_CONTEXT_DIM = len(RESET_CONTEXT_NAMES)


def critic_context_names(config: CriticContextConfig, num_object_bodies: int) -> tuple[str, ...]:
    """Return the stable scalar layout for the configured critic-only context."""
    names = list(RESET_CONTEXT_NAMES)
    if config.reference_phase:
        names.append("normalized_reference_phase")
    if config.object_body_velocity:
        for body in range(num_object_bodies):
            names.extend(f"object_body_{body}_{component}" for component in _OBJECT_BODY_VELOCITY_NAMES)
    return tuple(names)


@wp.kernel
def publish_critic_context(
    reset_context: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    body_qd: wp.array(dtype=wp.spatial_vector),
    object_body_ids_w: wp.array(dtype=wp.int32),
    num_frames: int,
    num_object_bodies: int,
    context_dim: int,
    phase_offset: int,
    velocity_offset: int,
    context: wp.array(dtype=wp.float32),
) -> None:
    """Append reference phase and mapped-object world velocity to reset context."""
    world = wp.tid()
    reset_input = world * _RESET_CONTEXT_DIM
    output = world * context_dim
    for index in range(_RESET_CONTEXT_DIM):
        context[output + index] = reset_context[reset_input + index]

    if phase_offset >= 0:
        phase = float(0.0)
        if num_frames > 1:
            phase = wp.clamp(float(timestep[world]) / float(num_frames - 1), 0.0, 1.0)
        context[output + phase_offset] = phase

    if velocity_offset >= 0:
        body_input = world * num_object_bodies
        for body in range(num_object_bodies):
            velocity = body_qd[object_body_ids_w[body_input + body]]
            linear = wp.spatial_top(velocity)
            angular = wp.spatial_bottom(velocity)
            velocity_output = output + velocity_offset + body * 6
            for axis in range(3):
                context[velocity_output + axis] = linear[axis]
                context[velocity_output + 3 + axis] = angular[axis]


@wp.kernel
def publish_rl_transition(
    score: wp.array(dtype=wp.float32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    timestep: wp.array(dtype=wp.int32),
    episode_start_frame: wp.array(dtype=wp.int32),
    last_frame: int,
    auto_reset: int,
    running_return: wp.array(dtype=wp.float32),
    running_length: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.int32),
    truncation: wp.array(dtype=wp.int32),
    reset_mask: wp.array(dtype=wp.int32),
    episode_return: wp.array(dtype=wp.float32),
    episode_length: wp.array(dtype=wp.int32),
    episode_reference_progress: wp.array(dtype=wp.float32),
) -> None:
    """Preserve one transition and publish metrics for episodes ending on it."""
    world = wp.tid()
    transition_reward = score[world]
    transition_done = terminated[world]
    transition_truncation = truncated[world]
    completed = wp.max(transition_done, transition_truncation)
    total_return = running_return[world] + transition_reward
    total_length = running_length[world] + 1

    reward[world] = transition_reward
    done[world] = transition_done
    truncation[world] = transition_truncation
    reset_mask[world] = completed * auto_reset
    episode_return[world] = 0.0
    episode_length[world] = 0
    episode_reference_progress[world] = 0.0
    if completed != 0:
        episode_return[world] = total_return
        episode_length[world] = total_length
        remaining_frames = last_frame - episode_start_frame[world]
        advanced_frames = timestep[world] - episode_start_frame[world]
        if remaining_frames <= 0:
            episode_reference_progress[world] = 1.0
        else:
            if advanced_frames < 0:
                advanced_frames = 0
            if advanced_frames > remaining_frames:
                advanced_frames = remaining_frames
            episode_reference_progress[world] = float(advanced_frames) / float(remaining_frames)
        running_return[world] = 0.0
        running_length[world] = 0
    else:
        running_return[world] = total_return
        running_length[world] = total_length


@wp.kernel
def reset_rl_state(
    reset_mask: wp.array(dtype=wp.int32),
    running_return: wp.array(dtype=wp.float32),
    running_length: wp.array(dtype=wp.int32),
    reward: wp.array(dtype=wp.float32),
    done: wp.array(dtype=wp.int32),
    truncation: wp.array(dtype=wp.int32),
    auto_reset_mask: wp.array(dtype=wp.int32),
    episode_return: wp.array(dtype=wp.float32),
    episode_length: wp.array(dtype=wp.int32),
    episode_reference_progress: wp.array(dtype=wp.float32),
) -> None:
    """Clear RL-only history and outputs for explicitly reset worlds."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    running_return[world] = 0.0
    running_length[world] = 0
    reward[world] = 0.0
    done[world] = 0
    truncation[world] = 0
    auto_reset_mask[world] = 0
    episode_return[world] = 0.0
    episode_length[world] = 0
    episode_reference_progress[world] = 0.0


class RLEnv(BaseEnv):
    """Base transition plus RL outputs, episode accounting, auto-reset, and observations."""

    def __init__(
        self,
        scene: Scene,
        reference: Reference,
        config: RLEnvConfig | None = None,
        device=None,
        *,
        action: Action | None = None,
        termination: Termination | None = None,
        objective: Objective | None = None,
        observation: Observation | None = None,
        reset_policy: ResetPolicy | None = None,
    ) -> None:
        config = config or RLEnvConfig()
        super().__init__(
            scene,
            reference,
            config=config,
            device=device,
            action=action,
            termination=termination,
            objective=objective,
        )
        if self.objective is None:
            raise ValueError("RLEnv requires an objective strategy or objective configuration")
        if reset_policy is None:
            if not isinstance(config.reset, ResetSpec):
                raise TypeError("configured reset must implement ResetSpec")
            reset_policy = config.reset.build(
                self.world_count,
                reference.num_frames,
                device=self.device,
            )
        if not isinstance(reset_policy, ResetPolicy):
            raise TypeError("reset_policy must implement the ResetPolicy protocol")
        if config.capture_terminal_state and not isinstance(reset_policy, ResetContext):
            raise TypeError("capture_terminal_state requires a reset policy implementing ResetContext")
        self.reset_policy = reset_policy
        if isinstance(reset_policy, ReferenceOffsetReset):
            reset_policy.bind_reference_joint_q_offset(
                scene,
                reference,
                self.reference_joint_q_offset,
            )
        self.object_control_scale = reset_policy.applied_voc_scale
        self.set_reset_to_first_frame_probability(None)
        self.set_immediate_first_frame_probability(None)
        if observation is None:
            if not isinstance(self.action, PolicyAction):
                raise TypeError(
                    "the configured observation requires a PolicyAction; "
                    "select a compatible observation strategy for this action"
                )
            if not isinstance(config.observation, ObservationSpec):
                raise TypeError("configured observation must implement ObservationSpec")
            observation = config.observation.build(
                scene,
                self.action,
                self.command,
                self.contact,
                self.timestep,
                self.episode_step,
                self.reference_joint_q_offset,
                device=self.device,
            )
        if not isinstance(observation, Observation):
            raise TypeError("observation must implement the Observation protocol")
        if config.capture_terminal_state and not isinstance(observation, ObservationInto):
            raise TypeError("capture_terminal_state requires an observation implementing ObservationInto")
        expected_observation = self.world_count * observation.observation_dim
        if observation.observation.shape != (expected_observation,):
            raise ValueError(
                f"observation buffer has shape {observation.observation.shape}; expected ({expected_observation},)"
            )
        self.observation_strategy = observation
        self.observation = observation.observation
        self.observation_dim = observation.observation_dim
        self.terminal_observation = None
        self.critic_context = None
        self.terminal_critic_context = None
        self.critic_context_names: tuple[str, ...] = ()
        self._reset_critic_context = None
        self._critic_phase_offset = -1
        self._critic_velocity_offset = -1
        if config.capture_terminal_state:
            if tuple(reset_policy.context_names) != RESET_CONTEXT_NAMES:
                raise ValueError(
                    f"critic context requires reset fields {RESET_CONTEXT_NAMES}, "
                    f"got {tuple(reset_policy.context_names)}"
                )
            self.critic_context_names = critic_context_names(
                config.critic_context,
                self.command.layout.num_bodies,
            )
            context_dim = len(self.critic_context_names)
            if config.critic_context.reference_phase:
                self._critic_phase_offset = _RESET_CONTEXT_DIM
            if config.critic_context.object_body_velocity:
                self._critic_velocity_offset = _RESET_CONTEXT_DIM + int(config.critic_context.reference_phase)
            if context_dim > _RESET_CONTEXT_DIM:
                self._reset_critic_context = wp.zeros(
                    self.world_count * _RESET_CONTEXT_DIM,
                    dtype=wp.float32,
                    device=self.device,
                )
            self.terminal_observation = wp.zeros(expected_observation, dtype=wp.float32, device=self.device)
            self.critic_context = wp.zeros(self.world_count * context_dim, dtype=wp.float32, device=self.device)
            self.terminal_critic_context = wp.zeros(
                self.world_count * context_dim,
                dtype=wp.float32,
                device=self.device,
            )

        self.reward = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.done = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.truncation = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.reset_mask = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.running_return = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.running_length = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.episode_return = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self.episode_length = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.episode_reference_progress = wp.zeros(self.world_count, dtype=wp.float32, device=self.device)
        self._step_graph = None

    def _compute_critic_context(self, state, output: wp.array) -> wp.array:
        """Publish reset context and configured Markov state into one persistent buffer."""
        if self._reset_critic_context is None:
            return self.reset_policy.compute_context(self.voc_scale, output)
        self.reset_policy.compute_context(self.voc_scale, self._reset_critic_context)
        wp.launch(
            publish_critic_context,
            dim=self.world_count,
            inputs=[
                self._reset_critic_context,
                self.timestep,
                state.body_qd,
                self.command.body_ids_w,
                self.reference.num_frames,
                self.command.layout.num_bodies,
                len(self.critic_context_names),
                self._critic_phase_offset,
                self._critic_velocity_offset,
            ],
            outputs=[output],
        )
        return output

    def reset(
        self,
        frame_id: int | None = None,
        reset_frame: wp.array | None = None,
        reset_mask: wp.array | None = None,
    ) -> wp.array:
        """Reset selected worlds and return their current shared observation buffer."""
        selected = self.reset_table.all_reset_mask if reset_mask is None else reset_mask
        if frame_id is None and reset_frame is None:
            self.reset_policy.sample(selected, self.voc_scale)
            reset_frame = self.reset_policy.reset_frame
        else:
            self.reset_policy.prepare_explicit(selected, self.voc_scale)
        BaseEnv.reset(
            self,
            frame_id=frame_id,
            reset_frame=reset_frame,
            reset_mask=selected,
            finger_scale=self.reset_policy.finger_scale,
        )
        wp.launch(
            reset_rl_state,
            dim=self.world_count,
            inputs=[selected],
            outputs=[
                self.running_return,
                self.running_length,
                self.reward,
                self.done,
                self.truncation,
                self.reset_mask,
                self.episode_return,
                self.episode_length,
                self.episode_reference_progress,
            ],
        )
        observation = self.observation_strategy.compute(self.state_0)
        if self.config.capture_terminal_state:
            self._compute_critic_context(self.state_0, self.critic_context)
        return observation

    def _rl_step(self) -> None:
        self._transition()
        if self.config.capture_terminal_state:
            self.observation_strategy.compute_into(self.state_0, self.terminal_observation)
            self._compute_critic_context(self.state_0, self.terminal_critic_context)
        wp.launch(
            publish_rl_transition,
            dim=self.world_count,
            inputs=[
                self.objective.score,
                self.terminated,
                self.truncated,
                self.timestep,
                self.episode_start_frame,
                self.reference.num_frames - 1,
                int(self.config.auto_reset),
            ],
            outputs=[
                self.running_return,
                self.running_length,
                self.reward,
                self.done,
                self.truncation,
                self.reset_mask,
                self.episode_return,
                self.episode_length,
                self.episode_reference_progress,
            ],
        )
        if self.config.auto_reset:
            self.reset_policy.sample(self.reset_mask, self.voc_scale)
            BaseEnv.reset(
                self,
                frame_id=None,
                reset_frame=self.reset_policy.reset_frame,
                reset_mask=self.reset_mask,
                finger_scale=self.reset_policy.finger_scale,
            )
        self.observation_strategy.compute(self.state_0)
        if self.config.capture_terminal_state:
            self._compute_critic_context(self.state_0, self.critic_context)

    def _advance_lifecycle(self) -> None:
        """Advance the reference reset hold, per-world VOC, and reference counters."""
        self.reset_policy.advance(
            self.terminated,
            self.truncated,
            self.timestep,
            self.episode_step,
            self.voc_scale,
        )

    def set_voc_scale(self, scale: float) -> None:
        """Change the curriculum target without disturbing worlds still settling."""
        super().set_voc_scale(scale)
        self.reset_policy.sync_curriculum(self.voc_scale)
        if self.config.capture_terminal_state:
            self._compute_critic_context(self.state_0, self.critic_context)

    def set_reset_to_first_frame_probability(self, probability: float | None) -> float | None:
        """Set an optional reset-stage override without requiring it of every reset strategy."""
        if not isinstance(self.reset_policy, FirstFrameResetCurriculum):
            if probability is not None:
                raise TypeError(
                    "reset_to_first_frame_probability requires a reset policy implementing FirstFrameResetCurriculum"
                )
            return None
        resolved = self.reset_policy.set_reset_to_first_frame_probability(probability)
        self.reset_to_first_frame_probability = resolved
        return resolved

    def set_immediate_first_frame_probability(self, probability: float | None) -> float | None:
        """Set the fraction of auto-resets that exactly match immediate frame-zero deployment."""
        if not isinstance(self.reset_policy, ImmediateFirstFrameResetCurriculum):
            if probability is not None:
                raise TypeError(
                    "immediate_first_frame_probability requires a reset policy implementing the corresponding setter"
                )
            return None
        resolved = self.reset_policy.set_immediate_first_frame_probability(probability)
        self.immediate_first_frame_probability = resolved
        return resolved

    def step(self, action: wp.array | None = None) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        """Advance one RL step and return observation, reward, done, and truncation buffers."""
        if action is not None and action.shape != self.action_input.shape:
            raise ValueError(f"action has shape {action.shape}; expected {self.action_input.shape}")
        if self._step_graph is None:
            action_input = self.action_input if action is None else action
            self.action.process(action_input, self.timestep, self.state_0)
            self._rl_step()
        else:
            if action is not None and action is not self.action_input:
                wp.copy(self.action_input, action)
            self.action.process(self.action_input, self.timestep, self.state_0)
            wp.capture_launch(self._step_graph)
        return self.observation, self.reward, self.done, self.truncation

    def capture_step(self) -> None:
        """Capture transition, output preservation, auto-reset, and observation as one graph."""
        if self._step_graph is not None:
            raise RuntimeError("RL step is already captured")
        with wp.ScopedCapture(self.device) as capture:
            self._rl_step()
        self._step_graph = capture.graph

    def capture_transition(self) -> None:
        """Capture the complete RL step rather than the base transition alone."""
        self.capture_step()
