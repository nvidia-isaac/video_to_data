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
def _compute_wrench_and_effort(
    root_body_position_e: torch.Tensor,
    root_body_wxyz: torch.Tensor,
    root_link_vel_w: torch.Tensor,
    root_link_quat_w: torch.Tensor,
    root_com_pos_b: torch.Tensor,
    command_position_e: torch.Tensor,
    command_wxyz_e: torch.Tensor,
    object_mass: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    scale_factor: torch.Tensor,
    command_joint_pos: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    linear_stiffness: float,
    linear_damping: float,
    angular_stiffness: float,
    angular_damping: float,
    max_force: float,
    max_torque: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Correct COM->link offset for linear velocity (mutates caller tensor, matching eager path).
    root_link_vel_w[:, :3] += torch.linalg.cross(
        root_link_vel_w[:, 3:],
        quat_apply(root_link_quat_w, -root_com_pos_b),
        dim=-1,
    )
    root_body_linvel_b = quat_apply_inverse(root_link_quat_w, root_link_vel_w[:, :3])
    root_body_angvel_b = quat_apply_inverse(root_link_quat_w, root_link_vel_w[:, 3:])

    # PD force (position in body frame).
    position_error_e = command_position_e - root_body_position_e
    position_error_b = quat_apply_inverse(root_body_wxyz, position_error_e)
    force = linear_stiffness * position_error_b - linear_damping * root_body_linvel_b

    # PD torque (orientation in body frame).
    orientation_error_b = axis_angle_from_quat(
        quat_mul(quat_inv(root_body_wxyz), command_wxyz_e)
    )
    torque = (
        angular_stiffness * orientation_error_b - angular_damping * root_body_angvel_b
    )

    # Gravity compensation on the root body.
    force = force + (-9.81) * object_mass * projected_gravity_b

    # Curriculum scale + clamp.
    force = torch.clamp(force * scale_factor, min=-max_force, max=max_force)
    torque = torch.clamp(torque * scale_factor, min=-max_torque, max=max_torque)

    # Joint PD effort (reusing linear stiffness / angular damping to match eager path).
    effort = (
        linear_stiffness * (command_joint_pos - joint_pos) - angular_damping * joint_vel
    )
    effort = torch.clamp(effort * scale_factor, min=-max_force, max=max_force)

    return force, torque, effort


class VirtualArticulatedObjectControl(ActionTerm):
    """Virtual articulated object control: PD wrench on root body + joint torques.

    Applies a PD-based wrench to the root body (identical to VirtualRigidObjectControl)
    to track the reference base position/orientation, then apply target joint torques to the joints
    at each step.
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

        # Object physical properties
        self._root_body_idx = self.object.find_bodies([self.cfg.root_body_name])[0][0]
        body_masses = self.object.root_physx_view.get_masses().to(self.device)
        self.object_mass = body_masses.sum(dim=-1).unsqueeze(-1)  # (num_envs, 1)
        body_inertias = self.object.root_physx_view.get_inertias().to(self.device)
        self.object_inertia = body_inertias[:, self._root_body_idx, :]  # (num_envs, 9)
        body_coms = self.object.root_physx_view.get_coms().to(self.device)
        self.object_com = body_coms[:, self._root_body_idx, :]  # (num_envs, 7)

        self.root_com_pose_b = self.object.data._root_physx_view.get_coms()[:, 0].to(
            self.device
        )  # quat in xyzw format

        # Joint IDs
        joint_ids, _ = self.object.find_joints(".*")
        self._joint_ids = joint_ids
        self._num_joints = len(joint_ids)
        assert (
            self._num_joints == 1
        ), "Currently only support single joint articulated object."

        # Raw/processed actions store root-body 6D wrench and joint effort
        self._raw_actions = torch.zeros(
            self.num_envs, 6 + self._num_joints, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Forces, torques, and efforts tensors
        self._forces = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.device
        )
        self._torques = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.device
        )
        self._efforts = torch.zeros(self.num_envs, self._num_joints, device=self.device)

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
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the joint action.

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
        self._forces[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self._efforts[env_ids] = 0.0

        # clear external forces, torques, and joint efforts
        self.object.set_external_force_and_torque(
            forces=self._forces[env_ids],
            torques=self._torques[env_ids],
            env_ids=env_ids,
            is_global=False,
        )
        self.object.set_joint_effort_target(self._efforts[env_ids], env_ids=env_ids)
        self.object.write_data_to_sim()

    def apply_actions(self) -> None:
        """Apply PD wrench to root body and effort to joints based on reference trajectory."""
        command_joint_pos = self.command.retargeted_object_articulation[
            self.command.timestep_counter
        ].view(-1, self._num_joints)

        force, torque, effort = _compute_wrench_and_effort(
            root_body_position_e=self.command.object_position_e[
                :, self._root_body_idx, :
            ],
            root_body_wxyz=self.command.object_orientation_e[:, self._root_body_idx, :],
            root_link_vel_w=self.object.data.root_com_vel_w,
            root_link_quat_w=self.object.data.root_link_quat_w,
            root_com_pos_b=self.root_com_pose_b[..., :3],
            command_position_e=self.command.object_body_position_command_e[
                :, self._root_body_idx, :
            ],
            command_wxyz_e=self.command.object_body_wxyz_command_e[
                :, self._root_body_idx, :
            ],
            object_mass=self.object_mass,
            projected_gravity_b=self.object.data.projected_gravity_b,
            scale_factor=self.command.virtual_object_controller_scale_factor_per_env,
            command_joint_pos=command_joint_pos,
            joint_pos=self.object.data.joint_pos,
            joint_vel=self.object.data.joint_vel,
            linear_stiffness=self._tracking_controller_linear_stiffness,
            linear_damping=self._tracking_controller_linear_damping,
            angular_stiffness=self._tracking_controller_angular_stiffness,
            angular_damping=self._tracking_controller_angular_damping,
            max_force=float(self.cfg.max_force),
            max_torque=float(self.cfg.max_torque),
        )

        # Apply wrench to root body.
        self._forces[:, self._root_body_idx] = force
        self._torques[:, self._root_body_idx] = torque
        self.object.set_external_force_and_torque(
            forces=self._forces,
            torques=self._torques,
            is_global=False,
        )

        # Apply joint effort.
        self.object.set_joint_effort_target(effort)
        self.object.write_data_to_sim()

        self._raw_actions[..., :3] = force
        self._raw_actions[..., 3:6] = torque
        self._raw_actions[..., 6:] = effort
        self._processed_actions = self._raw_actions
