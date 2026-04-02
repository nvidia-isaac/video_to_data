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
    """Virtual rigid object control action that applies to the object's joints."""

    cfg: actions_cfg.VirtualArticulatedObjectControlCfg
    """The configuration of the action term."""

    def __init__(
        self, cfg: actions_cfg.VirtualArticulatedObjectControlCfg, env: ManagerBasedEnv
    ) -> None:
        """Initialize the action term."""
        # initialize the action term
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

        self.object_mass = (
            self.object.root_physx_view.get_masses().to(self.device).unsqueeze(2)
        )  # (num_envs, 2, 1)
        self.object_inertia = self.object.root_physx_view.get_inertias().to(
            self.device
        )  # (num_envs, 2, 9)
        self.object_com = self.object.root_physx_view.get_coms().to(
            self.device
        )  # (num_envs, 2, 7)

        # Create tensors for raw and processed actions with force and torque
        self._raw_actions = torch.zeros(self.num_envs, 2 * 6, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self.GRAVITY_VEC_E = self.object.data.GRAVITY_VEC_W.unsqueeze(1).expand(
            -1, self.num_bodies, -1
        )

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
        """Apply virtual force torque to the rigid object using a Position PD Controller."""
        # 1. Extract current object state
        object_position_e = self.command.object_position_e  # (num_envs, 2, 3)
        object_wxyz = self.command.object_orientation_e  # (num_envs, 2, 4)
        object_linvel_w = self.object.data.body_lin_vel_w  # (num_envs, 2, 3)
        object_linvel_b = math_utils.quat_apply_inverse(object_wxyz, object_linvel_w)
        object_angvel_w = self.object.data.body_ang_vel_w  # (num_envs, 2, 3)
        object_angvel_b = math_utils.quat_apply_inverse(object_wxyz, object_angvel_w)

        # 2. PD for force control
        object_position_error_e = (
            self.command.object_body_position_command_e - object_position_e
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
            self.command.object_body_wxyz_command_e,
        )
        object_orientation_error_b = math_utils.axis_angle_from_quat(
            object_orientation_error_b
        )
        torque = (
            self._tracking_controller_angular_stiffness * object_orientation_error_b
            - self._tracking_controller_angular_damping * object_angvel_b
        )

        # 4. Gravity compensation
        body_projected_gravity_b = math_utils.quat_apply_inverse(
            object_wxyz, self.GRAVITY_VEC_E
        )
        gravity_compensation_force = -9.81 * self.object_mass * body_projected_gravity_b
        force = force + gravity_compensation_force

        gravity_compensation_torque = torch.cross(
            self.object_com[..., :3], gravity_compensation_force, dim=-1
        )
        torque = torque + gravity_compensation_torque

        # 5. Scale based on curriculum
        force = (
            force
            * self.command.virtual_object_controller_scale_factor_per_env.unsqueeze(1)
        )
        torque = (
            torque
            * self.command.virtual_object_controller_scale_factor_per_env.unsqueeze(1)
        )

        # 6. Clip
        force = torch.clamp(force, min=-self.cfg.max_force, max=self.cfg.max_force)
        torque = torch.clamp(torque, min=-self.cfg.max_torque, max=self.cfg.max_torque)

        self._raw_actions[..., : 3 * self.num_bodies] = force.view(
            -1, 3 * self.num_bodies
        )
        self._raw_actions[..., 3 * self.num_bodies :] = torque.view(
            -1, 3 * self.num_bodies
        )
        self._processed_actions = self._raw_actions

        self.object.set_external_force_and_torque(
            forces=force.reshape(self.num_envs, self.num_bodies, 3),
            torques=torque.reshape(self.num_envs, self.num_bodies, 3),
            is_global=False,
        )
