# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference replay through the same zero-residual action and command path used for training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.lifecycle.reset import ReferenceResetTable, reset_worlds
from flash_chord.lifecycle.termination import (
    TerminationResult,
    TrackingTermination,
    TrackingTerminationConfig,
    evaluate_termination,
    read_termination,
)
from flash_chord.runtime.actions import Action, ActionConfig, ActionSpec
from flash_chord.runtime.command import CommandBuffers
from flash_chord.runtime.joint_drive import ScaledJointTargetDrive
from flash_chord.runtime.object_control import VOCParams, apply_object_control
from flash_chord.runtime.sim import SimConfig, build_force_contacts, build_solver
from flash_chord.scene.builder import Scene


@dataclass
class ReplayConfig:
    sim: SimConfig = field(default_factory=SimConfig)  # physics timing + solver
    action: ActionSpec = field(default_factory=ActionConfig)
    voc_scale: float = 1.0  # 1.0 = full virtual assist (kinematic-faithful object motion)
    voc_params: VOCParams = field(default_factory=VOCParams)


class ReplayRunner:
    """Steps a scene through a reference with zero policy residual and optional VOC."""

    def __init__(
        self,
        scene: Scene,
        reference: Reference,
        config: ReplayConfig | None = None,
        device=None,
    ):
        self.scene = scene
        self.reference = reference
        self.model = scene.model
        self.world_count = scene.world_count
        self.config = config or ReplayConfig()
        self.device = wp.get_device(device)

        self.sim = self.config.sim
        if abs(reference.fps - self.sim.fps) > 1e-6:
            raise ValueError(
                f"reference fps ({reference.fps}) != control fps ({self.sim.fps}); load with "
                f"load_reference(path, control_fps={self.sim.fps}) so replay uses the configured rate"
            )
        self.frame_dt = self.sim.frame_dt
        self.sim_dt = self.sim.sim_dt
        self.sim_time = 0.0

        c = self.config
        if not np.isfinite(c.voc_scale) or c.voc_scale < 0.0:
            raise ValueError(f"voc_scale must be finite and non-negative, got {c.voc_scale}")
        self.voc_scale = wp.array([c.voc_scale], dtype=wp.float32, device=self.device)
        self.solver = build_solver(self.model, self.sim)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = build_force_contacts(self.model, self.solver, self.sim, self.device)

        self.num_joint_dof = self.model.joint_dof_count // self.world_count
        if not isinstance(c.action, ActionSpec):
            raise TypeError("configured replay action must implement ActionSpec")
        self.action = c.action.build(scene, device=self.device)
        if not isinstance(self.action, Action):
            raise TypeError("configured replay action must build an Action")
        expected_targets = self.world_count * self.num_joint_dof
        if self.action.joint_target.shape != (expected_targets,):
            raise ValueError(
                f"replay action joint_target has shape {self.action.joint_target.shape}; "
                f"expected ({expected_targets},)"
            )
        self.zero_action = wp.zeros(
            self.world_count * self.action.action_dim,
            dtype=wp.float32,
            device=self.device,
        )
        self.timestep = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.command = CommandBuffers.build(scene, reference, device=self.device)
        self.object_joint_drive = None
        if self.command.layout.num_articulations:
            self.object_joint_drive = ScaledJointTargetDrive.build(
                self.solver,
                self.command.layout.articulation_dof_ids,
                world_count=self.world_count,
                device=self.device,
            )
        self.termination = TrackingTermination.build(
            scene,
            config=TrackingTerminationConfig(),
            object_body_ids_w=self.command.body_ids_w,
            object_ref_pos_w=self.command.body_target_pos_w,
            object_ref_quat_w=self.command.body_target_quat_w,
            num_objects=self.command.layout.num_bodies,
            device=self.device,
        )
        self.reset_table = ReferenceResetTable.build(scene, reference, device=self.device)

    def reset(self, frame_id: int = 0) -> None:
        """Snap the scene to ``reference`` frame ``frame_id`` (robot + objects) via the reset [WK]."""
        reset_worlds(self.model, self.reset_table, self.state_0, frame_id=frame_id)
        self.action.reset(self.reset_table.all_reset_mask)
        self.timestep.assign(np.full(self.world_count, frame_id, dtype=np.int32))
        self.sim_time = 0.0

    def step(self, frame_id: int) -> None:
        """Advance one control frame, tracking ``reference`` frame ``frame_id``."""
        self.timestep.assign(np.full(self.world_count, frame_id, dtype=np.int32))
        self.action.process(self.zero_action, self.timestep, self.state_0)
        self.command.gather(self.timestep)
        self.action.prepare_control(self.control)
        if self.object_joint_drive is not None:
            self.object_joint_drive.apply(self.voc_scale)
        for _ in range(self.sim.substeps):
            if not self.sim.use_mujoco_contacts:
                self.model.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.action.apply_control(self.state_0, self.control)
            apply_object_control(
                self.model,
                self.state_0,
                self.control,
                self.command,
                scale=self.voc_scale,
                params=self.config.voc_params,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        if self.sim.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)
        self.sim_time += self.frame_dt

    def check_termination(self, config: TrackingTerminationConfig | None = None) -> TerminationResult:
        """Run the termination [WK] for the current state vs the loaded reference frame and read it back
        (verification: per-world peak errors + whether any world terminated). The RL/MPPI step instead
        consumes ``self.termination``'s device ``done`` masks without the host readback."""
        termination_config = config or self.termination.config
        evaluate_termination(
            self.termination.buffers,
            self.state_0.body_q,
            self.timestep,
            self.termination.reference,
            self.command.body_ids_w,
            self.command.body_target_pos_w,
            self.command.body_target_quat_w,
            termination_config,
        )
        return read_termination(self.termination.buffers)
