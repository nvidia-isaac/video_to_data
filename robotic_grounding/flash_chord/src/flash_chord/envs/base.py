# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared device-native environment transition for RL and sampling workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import newton
import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.lifecycle.reset import ReferenceResetTable, reset_worlds
from flash_chord.lifecycle.termination import (
    Termination,
    TerminationSpec,
    TrackingTerminationConfig,
)
from flash_chord.objectives.composition import Objective, ObjectiveSpec
from flash_chord.objectives.config import ObjectiveConfig
from flash_chord.runtime.actions import (
    Action,
    ActionConfig,
    ActionSpec,
    ReferenceOffsetAction,
)
from flash_chord.runtime.command import CommandBuffers
from flash_chord.runtime.contact import ContactTracker, ContactTrackerConfig
from flash_chord.runtime.joint_drive import (
    JointActuatorLimits,
    JointPositionIntegral,
    JointPositionIntegralConfig,
    ScaledJointTargetDrive,
)
from flash_chord.runtime.object_control import VOCParams, apply_object_control
from flash_chord.runtime.sim import SimConfig, build_force_contacts, build_solver
from flash_chord.scene.builder import Scene


@dataclass
class BaseEnvConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    action: ActionSpec = field(default_factory=ActionConfig)
    voc: VOCParams = field(default_factory=VOCParams)
    termination: TrackingTerminationConfig = field(default_factory=TrackingTerminationConfig)
    contact: ContactTrackerConfig | None = field(default_factory=ContactTrackerConfig)
    objective: ObjectiveConfig | None = field(default_factory=ObjectiveConfig)
    voc_scale: float = 1.0
    arm_position_integral: JointPositionIntegralConfig = field(default_factory=JointPositionIntegralConfig)
    enforce_joint_velocity_limits: bool = False


@wp.kernel
def finish_transition(
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
) -> None:
    """Advance counters for worlds that remain active after a transition."""
    world = wp.tid()
    if terminated[world] == 0 and truncated[world] == 0:
        timestep[world] += 1
        episode_step[world] += 1


@wp.kernel
def reset_env_state(
    reset_mask: wp.array(dtype=wp.int32),
    reset_frame: wp.array(dtype=wp.int32),
    timestep: wp.array(dtype=wp.int32),
    episode_start_frame: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
) -> None:
    """Reset lifecycle state for selected worlds."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    timestep[world] = reset_frame[world]
    episode_start_frame[world] = reset_frame[world]
    episode_step[world] = 0
    terminated[world] = 0
    truncated[world] = 0


class BaseEnv:
    """World-parallel physics, control, reference, termination, and reset state."""

    def __init__(
        self,
        scene: Scene,
        reference: Reference,
        config: BaseEnvConfig | None = None,
        device=None,
        *,
        action: Action | None = None,
        termination: Termination | None = None,
        objective: Objective | None = None,
    ) -> None:
        self.scene = scene
        self.reference = reference
        self.config = config or BaseEnvConfig()
        self.device = wp.get_device(device)
        self.model = scene.model
        self.world_count = scene.world_count
        self.num_joint_dof = self.model.joint_dof_count // self.world_count
        self.frame_dt = self.config.sim.frame_dt
        self.sim_dt = self.config.sim.sim_dt
        if not np.isfinite(self.config.voc_scale) or self.config.voc_scale < 0.0:
            raise ValueError(f"voc_scale must be finite and non-negative, got {self.config.voc_scale}")
        self.voc_scale = wp.array([self.config.voc_scale], dtype=wp.float32, device=self.device)
        self.object_control_scale = self.voc_scale

        if abs(reference.fps - self.config.sim.fps) > 1.0e-6:
            raise ValueError(
                f"reference fps ({reference.fps}) != control fps ({self.config.sim.fps}); "
                "resample the reference to the configured control rate"
            )
        if not self.config.sim.use_mujoco_contacts:
            raise ValueError("BaseEnv requires MuJoCo contacts for measured contact force")

        self.solver = build_solver(self.model, self.config.sim)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = build_force_contacts(self.model, self.solver, self.config.sim, self.device)

        if action is None:
            if not isinstance(self.config.action, ActionSpec):
                raise TypeError("configured action must implement ActionSpec")
            action = self.config.action.build(
                scene,
                device=self.device,
            )
        self.action = action
        if not isinstance(self.action, Action):
            raise TypeError("action must implement the Action protocol")
        expected_targets = self.world_count * self.num_joint_dof
        if self.action.joint_target.shape != (expected_targets,):
            raise ValueError(
                f"action joint_target has shape {self.action.joint_target.shape}; expected ({expected_targets},)"
            )
        self.reference_joint_q_offset = wp.zeros(
            self.world_count * (self.model.joint_coord_count // self.world_count),
            dtype=wp.float32,
            device=self.device,
        )
        if isinstance(self.action, ReferenceOffsetAction):
            self.action.bind_reference_joint_q_offset(self.reference_joint_q_offset)
        self.command = CommandBuffers.build(scene, reference, device=self.device)
        self.timestep = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.episode_start_frame = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.episode_step = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.arm_position_integral = None
        if self.config.arm_position_integral.gain_s_inv2 > 0.0:
            arm_joints = scene.layout.select_scalar_joints(scene.layout.sides, ("arm",), require_names=True)
            self.arm_position_integral = JointPositionIntegral.build(
                self.model,
                arm_joints.q_ids,
                arm_joints.dof_ids,
                world_count=self.world_count,
                dt=self.sim_dt,
                config=self.config.arm_position_integral,
                device=self.device,
            )
        self.joint_actuator_limits = None
        if self.config.enforce_joint_velocity_limits:
            scalar = scene.layout.scalar_joints
            if scalar is None:
                raise ValueError("joint velocity-limit enforcement requires a scalar-joint layout")
            self.joint_actuator_limits = JointActuatorLimits.build(
                self.model,
                scalar.q_ids,
                scalar.dof_ids,
                world_count=self.world_count,
                device=self.device,
            )
        self.object_joint_drive = None
        if self.command.layout.num_articulations:
            self.object_joint_drive = ScaledJointTargetDrive.build(
                self.solver,
                self.command.layout.articulation_dof_ids,
                world_count=self.world_count,
                device=self.device,
            )
        self.contact = None
        if self.config.contact is not None:
            self.contact = ContactTracker.build(
                scene.layout,
                object_body_ids=self.command.layout.body_ids,
                world_count=self.world_count,
                bodies_per_world=self.command.bodies_per_world,
                shapes_per_world=scene.collision_layout.shape_count,
                config=self.config.contact,
                device=self.device,
            )
        if termination is None:
            if not isinstance(self.config.termination, TerminationSpec):
                raise TypeError("configured termination must implement TerminationSpec")
            termination = self.config.termination.build(
                scene,
                object_body_ids_w=self.command.body_ids_w,
                object_ref_pos_w=self.command.body_target_pos_w,
                object_ref_quat_w=self.command.body_target_quat_w,
                episode_step=self.episode_step,
                num_objects=self.command.layout.num_bodies,
                device=self.device,
            )
        if not isinstance(termination, Termination):
            raise TypeError("termination must implement the Termination protocol")
        self.termination = termination
        if objective is None and self.config.objective is not None:
            if not isinstance(self.config.objective, ObjectiveSpec):
                raise TypeError("configured objective must implement ObjectiveSpec")
            objective = self.config.objective.build(
                self.model,
                scene.layout,
                scene.robot_reference,
                reference,
                self.command,
                self.contact,
                self.action,
                world_count=self.world_count,
                frame_dt=self.frame_dt,
                device=self.device,
                reference_joint_q_offset=self.reference_joint_q_offset,
                episode_start_frame=self.episode_start_frame,
            )
        if objective is not None and not isinstance(objective, Objective):
            raise TypeError("objective must implement the Objective protocol")
        self.objective = objective
        self.reset_table = ReferenceResetTable.build(scene, reference, device=self.device)

        self.terminated = self.termination.terminated
        self.truncated = self.termination.truncated
        self.action_input = wp.zeros(
            self.world_count * self.action.action_dim,
            dtype=wp.float32,
            device=self.device,
        )
        self._transition_graph = None

    def reset(
        self,
        frame_id: int | None = 0,
        reset_frame: wp.array | None = None,
        reset_mask: wp.array | None = None,
        finger_scale: wp.array | None = None,
    ) -> None:
        """Reset selected worlds to reference frames and clear their lifecycle/action state."""
        if reset_frame is None:
            if frame_id is None:
                raise ValueError("pass frame_id or reset_frame")
            self.reset_table.reset_frame.assign(np.full(self.world_count, int(frame_id), dtype=np.int32))
            reset_frame = self.reset_table.reset_frame
        if reset_mask is None:
            reset_mask = self.reset_table.all_reset_mask
        reset_worlds(
            self.model,
            self.reset_table,
            self.state_0,
            reset_frame=reset_frame,
            reset_mask=reset_mask,
            finger_scale=finger_scale,
            reference_joint_q_offset=self.reference_joint_q_offset,
        )
        self.action.reset(reset_mask)
        if self.arm_position_integral is not None:
            self.arm_position_integral.reset(reset_mask)
        if self.objective is not None:
            self.objective.reset(reset_mask)
        wp.launch(
            reset_env_state,
            dim=self.world_count,
            inputs=[reset_mask, reset_frame],
            outputs=[
                self.timestep,
                self.episode_start_frame,
                self.episode_step,
                self.terminated,
                self.truncated,
            ],
        )

    def _transition(self) -> None:
        self.command.gather(self.timestep)
        self.action.prepare_control(self.control)
        if self.object_joint_drive is not None:
            self.object_joint_drive.apply(self.object_control_scale)
        for substep in range(self.config.sim.substeps):
            self.state_0.clear_forces()
            self.action.apply_control(self.state_0, self.control)
            if self.arm_position_integral is not None:
                self.arm_position_integral.apply(self.state_0, self.control)
            apply_object_control(
                self.model,
                self.state_0,
                self.control,
                self.command,
                scale=self.object_control_scale,
                params=self.config.voc,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if self.joint_actuator_limits is not None:
                self.joint_actuator_limits.enforce_velocity(self.state_1)
                newton.eval_fk(
                    self.model,
                    self.state_1.joint_q,
                    self.state_1.joint_qd,
                    self.state_1,
                )
            if self.config.sim.substeps % 2 != 0 and substep == self.config.sim.substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)
        if self.contact is not None:
            self.contact.update(self.model, self.state_0, self.contacts)
        self.termination.evaluate(self.state_0.body_q, self.timestep)
        if self.objective is not None:
            self.objective.evaluate(self.state_0, self.timestep, self.terminated)
        self._advance_lifecycle()

    def _advance_lifecycle(self) -> None:
        """Advance the default reference and episode counters after one transition."""
        wp.launch(
            finish_transition,
            dim=self.world_count,
            inputs=[self.terminated, self.truncated],
            outputs=[self.timestep, self.episode_step],
        )

    def step(self, action: wp.array) -> None:
        """Advance one shared transition without auto-reset or workflow-specific output conversion."""
        if self._transition_graph is None:
            self.action.process(action, self.timestep, self.state_0)
            self._transition()
            return
        wp.copy(self.action_input, action)
        self.action.process(self.action_input, self.timestep, self.state_0)
        wp.capture_launch(self._transition_graph)

    def set_voc_scale(self, scale: float) -> None:
        """Update the VOC assist seen by eager and already-captured transitions."""
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(f"voc_scale must be finite and non-negative, got {scale}")
        self.voc_scale.assign([scale])

    def capture_transition(self) -> None:
        """Capture physics after action preprocessing updates its persistent device targets."""
        with wp.ScopedCapture(self.device) as capture:
            self._transition()
        self._transition_graph = capture.graph
