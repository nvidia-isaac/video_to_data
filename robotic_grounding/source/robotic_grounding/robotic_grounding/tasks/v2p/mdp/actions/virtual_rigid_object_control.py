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
from typing import TYPE_CHECKING, Tuple

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

logger = logging.getLogger(__name__)


@torch.jit.script
def _compute_wrench(
    object_position_e: torch.Tensor,
    object_wxyz: torch.Tensor,
    root_link_vel_w: torch.Tensor,
    root_link_quat_w: torch.Tensor,
    root_com_pos_b: torch.Tensor,
    command_position_e: torch.Tensor,
    command_wxyz_e: torch.Tensor,
    object_mass: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    scale_factor: torch.Tensor,
    linear_stiffness: float,
    linear_damping: float,
    angular_stiffness: float,
    angular_damping: float,
    max_force: float,
    max_torque: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Correct COM->link offset for linear velocity (mutates caller tensor, matching eager path).
    root_link_vel_w[:, :3] += torch.linalg.cross(
        root_link_vel_w[:, 3:],
        quat_apply(root_link_quat_w, -root_com_pos_b),
        dim=-1,
    )
    object_linvel_b = quat_apply_inverse(root_link_quat_w, root_link_vel_w[:, :3])
    object_angvel_b = quat_apply_inverse(root_link_quat_w, root_link_vel_w[:, 3:])

    # PD force (position in body frame).
    position_error_e = command_position_e - object_position_e
    position_error_b = quat_apply_inverse(object_wxyz, position_error_e)
    force = linear_stiffness * position_error_b - linear_damping * object_linvel_b

    # PD torque (orientation in body frame).
    orientation_error_b = axis_angle_from_quat(
        quat_mul(quat_inv(object_wxyz), command_wxyz_e)
    )
    torque = angular_stiffness * orientation_error_b - angular_damping * object_angvel_b

    # Gravity compensation.
    force = force + (-9.81) * object_mass * projected_gravity_b

    # Curriculum scale + clamp.
    force = torch.clamp(force * scale_factor, min=-max_force, max=max_force)
    torque = torch.clamp(torque * scale_factor, min=-max_torque, max=max_torque)

    return force, torque


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

        self.root_com_pose_b = self.object.data._root_physx_view.get_coms().to(
            self.device
        )  # quat in xyzw format

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
        force, torque = _compute_wrench(
            object_position_e=self.command.object_position_e[:, self.object_idx],
            object_wxyz=self.command.object_orientation_e[:, self.object_idx],
            root_link_vel_w=self.object.data.root_com_vel_w,
            root_link_quat_w=self.object.data.root_link_quat_w,
            root_com_pos_b=self.root_com_pose_b[..., :3],
            command_position_e=self.command.object_body_position_command_e[
                :, self.object_idx
            ],
            command_wxyz_e=self.command.object_body_wxyz_command_e[:, self.object_idx],
            object_mass=self.object_mass,
            projected_gravity_b=self.object.data.projected_gravity_b,
            scale_factor=self.command.virtual_object_controller_scale_factor_per_env,
            linear_stiffness=self._tracking_controller_linear_stiffness,
            linear_damping=self._tracking_controller_linear_damping,
            angular_stiffness=self._tracking_controller_angular_stiffness,
            angular_damping=self._tracking_controller_angular_damping,
            max_force=float(self.cfg.max_force),
            max_torque=float(self.cfg.max_torque),
        )

        self._raw_actions[..., :3] = force
        self._raw_actions[..., 3:] = torque
        self._processed_actions = self._raw_actions

        self.object.set_external_force_and_torque(
            forces=force.reshape(self.num_envs, 1, 3),
            torques=torque.reshape(self.num_envs, 1, 3),
            is_global=False,
        )
