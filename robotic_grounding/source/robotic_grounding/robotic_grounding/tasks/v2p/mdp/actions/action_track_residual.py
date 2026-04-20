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
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

logger = logging.getLogger(__name__)


@torch.jit.script
def _process_residual_actions(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    raw_actions: torch.Tensor,
    processed_actions: torch.Tensor,
    wrist_pose_command_e: torch.Tensor,
    finger_joint_pos_command: torch.Tensor,
    scale: torch.Tensor,
    clip: torch.Tensor,
    ema_factor: float,
) -> None:
    # Store raw policy outputs.
    raw_actions[:] = actions

    # EMA-filtered, scaled, clipped residual.
    actions = ema_factor * prev_actions + (1.0 - ema_factor) * actions * scale
    actions = torch.clamp(actions, min=-clip, max=clip)
    prev_actions[:] = actions

    # Wrist position target = command + residual.
    processed_actions[:, :3] = wrist_pose_command_e[:, :3] + actions[:, :3]

    # Wrist orientation target = command * quat(residual euler xyz).
    wrist_orientation_residual = quat_from_euler_xyz(
        actions[:, 3], actions[:, 4], actions[:, 5]
    )
    processed_actions[:, 3:7] = quat_mul(
        wrist_pose_command_e[:, 3:7], wrist_orientation_residual
    )

    # Finger joint targets = command + residual.
    processed_actions[:, 7:] = finger_joint_pos_command + actions[:, 6:]


@torch.jit.script
def _compute_wrist_wrench_and_finger_target(
    wrist_position: torch.Tensor,
    wrist_wxyz: torch.Tensor,
    wrist_link_vel_w: torch.Tensor,
    wrist_link_quat_w: torch.Tensor,
    wrist_com_pos_b: torch.Tensor,
    wrist_pose_target: torch.Tensor,
    finger_joint_target: torch.Tensor,
    gravity_comp_e: torch.Tensor,
    joint_damping: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_effort_limits: torch.Tensor,
    joint_stiffness: torch.Tensor,
    joint_pos: torch.Tensor,
    linear_stiffness: float,
    linear_damping: float,
    angular_stiffness: float,
    angular_damping: float,
    max_force: float,
    max_torque: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # COM->link correction for linear velocity (mutates caller tensor, matching eager path).
    wrist_link_vel_w[:, :3] += torch.linalg.cross(
        wrist_link_vel_w[:, 3:],
        quat_apply(wrist_link_quat_w, -wrist_com_pos_b),
        dim=-1,
    )
    wrist_linvel_b = quat_apply_inverse(wrist_link_quat_w, wrist_link_vel_w[:, :3])
    wrist_angvel_b = quat_apply_inverse(wrist_link_quat_w, wrist_link_vel_w[:, 3:])

    # PD force (position in body frame).
    position_error_e = wrist_pose_target[:, :3] - wrist_position
    position_error_b = quat_apply_inverse(wrist_wxyz, position_error_e)
    force = linear_stiffness * position_error_b - linear_damping * wrist_linvel_b

    # PD torque (orientation in body frame).
    orientation_error_b = axis_angle_from_quat(
        quat_mul(quat_inv(wrist_wxyz), wrist_pose_target[:, 3:7])
    )
    torque = angular_stiffness * orientation_error_b - angular_damping * wrist_angvel_b

    # Gravity compensation (world -> body).
    gravity_force_b = quat_apply_inverse(wrist_wxyz, gravity_comp_e[..., :3])
    gravity_torque_b = quat_apply_inverse(wrist_wxyz, gravity_comp_e[..., 3:6])
    force = torch.clamp(force + gravity_force_b, min=-max_force, max=max_force)
    torque = torch.clamp(torque + gravity_torque_b, min=-max_torque, max=max_torque)

    # Finger joint position clamp from torque limits (inlined clip_actions_to_torque_limit).
    kv_times_vel = joint_damping * joint_vel
    max_joint_action = (
        joint_effort_limits + kv_times_vel
    ) / joint_stiffness + joint_pos
    min_joint_action = (
        -joint_effort_limits + kv_times_vel
    ) / joint_stiffness + joint_pos
    finger_target_clipped = torch.clamp(
        finger_joint_target, min=min_joint_action, max=max_joint_action
    )

    return force, torque, finger_target_clipped


class JointResidualWithTrackingAction(ActionTerm):
    """Joint-space action term: tracking controller plus policy residual.

    A low-level tracking controller drives the hand to follow the reference
    trajectory (from the command term). The policy outputs a residual that is
    added to that reference; the controller then tracks reference + residual.

    Control is split by subsystem:

    - **Wrist**: Position and orientation are tracked by applying external
      force and torque to the wrist link (stiffness/damping from config) with gravity compensation.
    - **Finger joints**: Tracked by setting joint position targets (the
      simulation PD controller drives the joints to these targets).

    Action dimension: 3 (wrist position) + 3 (wrist orientation, e.g. Euler delta)
    + num_finger_joints. Scales for each block are set in the action config.
    """

    cfg: actions_cfg.JointResidualWithTrackingActionCfg
    """The configuration of the action term."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action."""

    def __init__(
        self, cfg: actions_cfg.JointResidualWithTrackingActionCfg, env: ManagerBasedEnv
    ) -> None:
        """Initialize the action term."""
        # initialize the action term
        super().__init__(cfg, env)

        # Pointer to the command term and robot attributes
        self.side = cfg.asset_name.split("_")[0]
        self.command = env.command_manager.get_term(self.cfg.command_name)
        if self.side == "right":
            self.robot = self.command.right_robot
            self.wrist_body_id = self.command.right_wrist_body_id
            self.finger_joint_names = self.command.right_finger_joint_names
            self.finger_joint_ids = self.command.right_finger_joint_ids
            self._wrist_position_e = lambda: self.command.right_hand_wrist_position_e
            self._wrist_wxyz_e = lambda: self.command.right_hand_wrist_wxyz_e
            self._wrist_pose_command_e = (
                lambda: self.command.right_hand_wrist_pose_command_e
            )
            self._finger_joint_pos_command = (
                lambda: self.command.right_hand_finger_joint_pos_command
            )
        else:
            self.robot = self.command.left_robot
            self.wrist_body_id = self.command.left_wrist_body_id
            self.finger_joint_names = self.command.left_finger_joint_names
            self.finger_joint_ids = self.command.left_finger_joint_ids
            self._wrist_position_e = lambda: self.command.left_hand_wrist_position_e
            self._wrist_wxyz_e = lambda: self.command.left_hand_wrist_wxyz_e
            self._wrist_pose_command_e = (
                lambda: self.command.left_hand_wrist_pose_command_e
            )
            self._finger_joint_pos_command = (
                lambda: self.command.left_hand_finger_joint_pos_command
            )

        # Save the wrist com pose in the body frame
        self.wrist_com_pose_b = self.robot.data._root_physx_view.get_coms()[:, 0].to(
            self.device
        )  # quat in xyzw format

        # Create tensors for tracking controller, policy raw and applied actions
        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,  # Store wrist orientation as an Euler angle
        )
        self._processed_actions = torch.zeros(
            self.num_envs,
            self.action_dim + 1,
            device=self.device,  # Store wrist orientation as a quaternion
        )
        self._processed_actions[..., 3] = 1.0  # Make quaternion correct
        self.prev_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )

        self.wrist_forces = torch.zeros(self.num_envs, 3, device=self.device)
        self.wrist_torques = torch.zeros(self.num_envs, 3, device=self.device)
        self.finger_joint_pos = torch.zeros(
            self.num_envs, len(self.finger_joint_ids), device=self.device
        )
        self.zero_force_torque = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Parse scale
        scale_tensor = torch.ones(self.num_envs, self.action_dim, device=self.device)
        scale_tensor[:, :3] = cfg.wrist_position_scale
        scale_tensor[:, 3:6] = cfg.wrist_orientation_scale
        scale_tensor[:, 6:] = cfg.finger_joint_scale
        self._scale: torch.Tensor = scale_tensor

        # Parse clip
        clip_tensor = torch.ones(self.num_envs, self.action_dim, device=self.device)
        clip_tensor[:, :3] = cfg.wrist_position_clip
        clip_tensor[:, 3:6] = cfg.wrist_orientation_clip
        clip_tensor[:, 6:] = cfg.finger_joint_clip
        self._clip: torch.Tensor = clip_tensor

        # Parse EMA decay factor
        self.ema_factor = float(cfg.ema_factor)

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
        """The dimension of the action."""
        return (
            len(self.finger_joint_ids) + 3 + 3
        )  # finger joints + wrist position + wrist Euler rotation

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions that are applied to the robot joints.

        This is the sum of the actions from the tracking controller and the policy.
        """
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:  # noqa: N802
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the joint action.
        It adds the following information to the base descriptor:
        - joint_names: The names of the joints.
        - scale: The scale of the action term.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointResidualWithTrackingAction"
        self._IO_descriptor.joint_names = self.finger_joint_ids
        self._IO_descriptor.scale = self._scale
        return self._IO_descriptor

    """
    Operations.
    """

    def clip_actions_to_torque_limit(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip finger joint position targets so PD torque stays within effort limits.

        The simulation uses a PD controller: τ = Kp * (q_target - q) + Kd * (v_target - v),
        with v_target = 0. To keep τ in [-L, L] (L = joint effort limit), we solve for
        the allowed range of q_target and clamp the given position targets into that
        range. Uses the robot's current joint stiffness, damping, velocity, and position.
        """
        # Compute the projected limits of the position action based on the torque limit
        kv_times_vel = self.robot.data.joint_damping * self.robot.data.joint_vel
        max_actions = (
            self.robot.data.joint_effort_limits + kv_times_vel
        ) / self.robot.data.joint_stiffness + self.robot.data.joint_pos
        min_actions = (
            -self.robot.data.joint_effort_limits + kv_times_vel
        ) / self.robot.data.joint_stiffness + self.robot.data.joint_pos

        return torch.clamp(
            actions,
            min=min_actions,
            max=max_actions,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the actions."""
        _process_residual_actions(
            actions=actions,
            prev_actions=self.prev_actions,
            raw_actions=self._raw_actions,
            processed_actions=self._processed_actions,
            wrist_pose_command_e=self._wrist_pose_command_e(),
            finger_joint_pos_command=self._finger_joint_pos_command(),
            scale=self._scale,
            clip=self._clip,
            ema_factor=self.ema_factor,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term."""
        self._raw_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.wrist_forces[env_ids] = 0.0
        self.wrist_torques[env_ids] = 0.0
        self.finger_joint_pos[env_ids] = 0.0

        # Clear the external forces and torques
        self.robot.set_external_force_and_torque(
            forces=self.zero_force_torque[env_ids],
            torques=self.zero_force_torque[env_ids],
            body_ids=self.wrist_body_id,
            env_ids=env_ids,
            is_global=False,
        )

    def apply_actions(self) -> None:
        """Apply the actions, recomputing the tracking actions to allow better performance."""
        force, torque, finger_target = _compute_wrist_wrench_and_finger_target(
            wrist_position=self._wrist_position_e(),
            wrist_wxyz=self._wrist_wxyz_e(),
            wrist_link_vel_w=self.robot.data.root_com_vel_w,
            wrist_link_quat_w=self.robot.data.root_link_quat_w,
            wrist_com_pos_b=self.wrist_com_pose_b[..., :3],
            wrist_pose_target=self._processed_actions[:, :7],
            finger_joint_target=self._processed_actions[:, 7:],
            gravity_comp_e=self.robot.root_physx_view.get_gravity_compensation_forces(),
            joint_damping=self.robot.data.joint_damping,
            joint_vel=self.robot.data.joint_vel,
            joint_effort_limits=self.robot.data.joint_effort_limits,
            joint_stiffness=self.robot.data.joint_stiffness,
            joint_pos=self.robot.data.joint_pos,
            linear_stiffness=self._tracking_controller_linear_stiffness,
            linear_damping=self._tracking_controller_linear_damping,
            angular_stiffness=self._tracking_controller_angular_stiffness,
            angular_damping=self._tracking_controller_angular_damping,
            max_force=float(self.cfg.max_force),
            max_torque=float(self.cfg.max_torque),
        )

        self.wrist_forces[:] = force
        self.wrist_torques[:] = torque
        self.finger_joint_pos[:] = finger_target

        self.robot.set_external_force_and_torque(
            forces=self.wrist_forces.reshape(self.num_envs, 1, 3),
            torques=self.wrist_torques.reshape(self.num_envs, 1, 3),
            body_ids=self.wrist_body_id,
            is_global=False,
        )
        self._asset.set_joint_position_target(
            self.finger_joint_pos, joint_ids=self.finger_joint_ids
        )
