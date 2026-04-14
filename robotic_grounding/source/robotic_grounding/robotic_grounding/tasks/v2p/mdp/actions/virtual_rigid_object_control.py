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


class VirtualRigidObjectControl(ActionTerm):
    """Virtual rigid object control action that applies to the object's joints."""

    cfg: actions_cfg.VirtualRigidObjectControlCfg
    """The configuration of the action term."""

    def __init__(
        self, cfg: actions_cfg.VirtualRigidObjectControlCfg, env: ManagerBasedEnv
    ) -> None:
        """Initialize the action term."""
        # initialize the action term
        super().__init__(cfg, env)

        # Pointer to the command term and object attribute
        self.command = env.command_manager.get_term(self.cfg.command_name)
        self.object_idx = self.command.cfg.object_body_names.index(cfg.asset_name)
        self.object = self.command.objects[self.object_idx]

        self.object_mass = self.object.root_physx_view.get_masses().to(
            self.device
        )  # (num_envs, 1)
        self.object_inertia = self.object.root_physx_view.get_inertias().to(
            self.device
        )  # (num_envs, 9)
        self.object_com = self.object.root_physx_view.get_coms().to(
            self.device
        )  # (num_envs, 7)

        # Create tensors for raw and processed actions with force and torque
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Set stiffness and damping for the tracking controller
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
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the joint action.
        It adds the following information to the base descriptor:
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "VirtualRigidObjectControl"
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

        # clear external forces and torques
        self.object.set_external_force_and_torque(
            forces=self._raw_actions[env_ids, :3].view(-1, 1, 3),
            torques=self._raw_actions[env_ids, 3:].view(-1, 1, 3),
            env_ids=env_ids,
            is_global=False,
        )

    def apply_actions(self) -> None:
        """Apply virtual force torque to the rigid object using a Position PD Controller."""
        # 1. Extract current object state
        object_position_e = self.command.object_position_e[
            :, self.object_idx
        ]  # world_p_object
        object_wxyz = self.command.object_orientation_e[
            :, self.object_idx
        ]  # world_q_object
        object_linvel_b = self.object.data.root_link_lin_vel_b
        object_angvel_b = self.object.data.root_link_ang_vel_b

        # 2. PD for force control
        object_position_error_e = (
            self.command.object_body_position_command_e[:, self.object_idx]
            - object_position_e
        )
        object_position_error_b = math_utils.quat_apply_inverse(
            object_wxyz, object_position_error_e
        )
        force = (
            self._tracking_controller_linear_stiffness * object_position_error_b
            - self._tracking_controller_linear_damping * object_linvel_b
        )

        # 3. PD for torque control
        object_orientation_error_b = math_utils.quat_mul(
            math_utils.quat_inv(object_wxyz),
            self.command.object_body_wxyz_command_e[:, self.object_idx],
        )
        object_orientation_error_b = math_utils.axis_angle_from_quat(
            object_orientation_error_b
        )
        torque = (
            self._tracking_controller_angular_stiffness * object_orientation_error_b
            - self._tracking_controller_angular_damping * object_angvel_b
        )

        # 4. Gravity compensation
        gravity_compensation_force = (
            -9.81 * self.object_mass * self.object.data.projected_gravity_b
        )
        force = force + gravity_compensation_force

        # gravity_compensation_torque = torch.cross(
        #     self.object_com[..., :3], gravity_compensation_force, dim=-1
        # )
        # torque = torque + gravity_compensation_torque

        # 5. Scale based on curriculum
        force = force * self.command.virtual_object_controller_scale_factor_per_env
        torque = torque * self.command.virtual_object_controller_scale_factor_per_env

        # 6. Clip
        force = torch.clamp(force, min=-self.cfg.max_force, max=self.cfg.max_force)
        torque = torch.clamp(torque, min=-self.cfg.max_torque, max=self.cfg.max_torque)

        self._raw_actions[..., :3] = force
        self._raw_actions[..., 3:] = torque
        self._processed_actions = self._raw_actions

        self.object.set_external_force_and_torque(
            forces=force.reshape(self.num_envs, 1, 3),
            torques=torque.reshape(self.num_envs, 1, 3),
            is_global=False,
        )
