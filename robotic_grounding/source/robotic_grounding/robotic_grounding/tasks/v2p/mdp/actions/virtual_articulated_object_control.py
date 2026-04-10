# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

logger = logging.getLogger(__name__)


class VirtualArticulatedObjectControl(ActionTerm):
    """Virtual articulated object control: PD force on root body + joint teleportation.

    Applies a PD-based wrench to the root body (identical to VirtualRigidObjectControl)
    to track the reference base position/orientation, then directly teleports all joints
    to the reference joint angles each step.  Controlling child bodies via independent
    forces is avoided because joint coupling transmits child-body control forces back to
    the base, causing unstable coupled oscillations when joints are undamped.
    """

    cfg: actions_cfg.VirtualArticulatedObjectControlCfg
    """The configuration of the action term."""

    def __init__(
        self, cfg: actions_cfg.VirtualArticulatedObjectControlCfg, env: ManagerBasedEnv
    ) -> None:
        """Initialize the action term."""
        super().__init__(cfg, env)

        # Pointer to the command term and object attribute
        self.command = env.command_manager.get_term(self.cfg.command_name)
        self.object_idx = self.command.cfg.object_body_names.index(cfg.asset_name)
        self.object = self.command.objects[self.object_idx]
        self.num_bodies = len(self.object.data.body_names)

        assert self.num_bodies == self.command.object_position_e.shape[1], (
            f"Currently only support single articulated object. "
            f"Find {self.command.object_position_e.shape[1]} body in motion file, "
            f"but {self.num_bodies} body in the object."
        )
        assert (
            self.command.retargeted_object_body_names == self.object.data.body_names
        ), (
            f"The body names in the motion file and the object do not match. "
            f"Find {self.command.retargeted_object_body_names} in motion file, "
            f"but {self.object.data.body_names} in the object."
        )

        # Root body physical properties (index 0) — shape matches VirtualRigidObjectControl
        root_masses = self.object.root_physx_view.get_masses().to(self.device)
        self.object_mass = root_masses[:, 0:1]  # (num_envs, 1)
        root_inertias = self.object.root_physx_view.get_inertias().to(self.device)
        self.object_inertia = root_inertias[:, 0, :]  # (num_envs, 9)
        root_coms = self.object.root_physx_view.get_coms().to(self.device)
        self.object_com = root_coms[:, 0, :]  # (num_envs, 7)

        # Joint IDs for teleportation
        joint_ids, _ = self.object.find_joints(".*")
        self._joint_ids = joint_ids
        self._num_joints = len(joint_ids)

        # Raw/processed actions store root-body 6D wrench only
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # PD gains
        self._tracking_controller_linear_stiffness = float(
            self.cfg.tracking_controller_linear_stiffness
        )
        self._tracking_controller_linear_damping = float(
            self.cfg.tracking_controller_linear_damping
        )
        self._tracking_controller_angular_stiffness = float(
            self.cfg.tracking_controller_angular_stiffness
        )
        self._tracking_controller_angular_damping = float(
            self.cfg.tracking_controller_angular_damping
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Policy should not control the object joints."""
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions that are applied to the object joints."""
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:  # noqa: N802
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "VirtualArticulatedObjectControl"
        self._IO_descriptor.num_bodies = self.num_bodies
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the actions."""
        del actions  # unused, should be empty

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term."""
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def apply_actions(self) -> None:
        """Apply PD wrench to root body and teleport joints to reference trajectory."""
        # ---- Root body state (identical to VirtualRigidObjectControl) ----
        object_position_e = self.command.object_position_e[:, 0, :]  # (num_envs, 3)
        object_wxyz = self.command.object_orientation_e[:, 0, :]  # (num_envs, 4)
        object_linvel_b = self.object.data.root_link_lin_vel_b  # (num_envs, 3)
        object_angvel_b = self.object.data.root_link_ang_vel_b  # (num_envs, 3)

        # ---- PD force (root body position) ----
        object_position_error_e = (
            self.command.object_body_position_command_e[:, 0, :] - object_position_e
        )
        object_position_error_b = math_utils.quat_apply_inverse(
            object_wxyz, object_position_error_e
        )
        force = (
            self._tracking_controller_linear_stiffness * object_position_error_b
            - self._tracking_controller_linear_damping * object_linvel_b
        )

        # ---- PD torque (root body orientation) ----
        object_orientation_error_b = math_utils.quat_mul(
            math_utils.quat_inv(object_wxyz),
            self.command.object_body_wxyz_command_e[:, 0, :],
        )
        object_orientation_error_b = math_utils.axis_angle_from_quat(
            object_orientation_error_b
        )
        torque = (
            self._tracking_controller_angular_stiffness * object_orientation_error_b
            - self._tracking_controller_angular_damping * object_angvel_b
        )

        # ---- Gravity compensation (root body) ----
        gravity_compensation_force = (
            -9.81 * self.object_mass * self.object.data.projected_gravity_b
        )
        force = force + gravity_compensation_force

        gravity_compensation_torque = torch.cross(
            self.object_com[..., :3], gravity_compensation_force, dim=-1
        )
        torque = torque + gravity_compensation_torque

        # ---- Curriculum scale ----
        force = force * self.command.virtual_object_controller_scale_factor_per_env
        torque = torque * self.command.virtual_object_controller_scale_factor_per_env

        # ---- Clip ----
        force = torch.clamp(force, min=-self.cfg.max_force, max=self.cfg.max_force)
        torque = torch.clamp(torque, min=-self.cfg.max_torque, max=self.cfg.max_torque)

        self._raw_actions[..., :3] = force
        self._raw_actions[..., 3:] = torque
        self._processed_actions = self._raw_actions

        # Apply force to root body only; zero out child bodies
        all_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        all_torques = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        all_forces[:, 0] = force
        all_torques[:, 0] = torque

        self.object.set_external_force_and_torque(
            forces=all_forces,
            torques=all_torques,
            is_global=False,
        )

        # ---- Teleport joints to reference trajectory ----
        if self._num_joints > 0:
            joint_pos = self.command.retargeted_object_articulation[
                self.command.timestep_counter
            ].float()  # (num_envs, N_joints)
            if joint_pos.dim() == 1:
                joint_pos = joint_pos.unsqueeze(-1)
            self.object.write_joint_state_to_sim(
                joint_pos,
                torch.zeros_like(joint_pos),
            )
