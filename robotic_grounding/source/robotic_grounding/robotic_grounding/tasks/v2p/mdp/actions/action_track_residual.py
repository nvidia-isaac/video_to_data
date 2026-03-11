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
        # 1. Store the raw actions from the policy
        self._raw_actions[:] = actions
        wrist_pose_command_e = self._wrist_pose_command_e()
        wrist_position_command_e = wrist_pose_command_e[:, :3]
        wrist_orientation_command_e = wrist_pose_command_e[:, 3:]

        # 2. Scale, filter and clip the actions
        actions = (
            self.ema_factor * self.prev_actions
            + (1 - self.ema_factor) * actions * self._scale
        )
        actions = torch.clamp(actions, min=-self._clip, max=self._clip)
        self.prev_actions[:] = actions

        # 3. Compute tracking target for wrist position
        self._processed_actions[:, :3] = wrist_position_command_e + actions[:, :3]

        # 4. Compute tracking target for wrist orientation
        wrist_orientation_residual = math_utils.quat_from_euler_xyz(
            actions[:, 3],
            actions[:, 4],
            actions[:, 5],
        )
        self._processed_actions[:, 3:7] = math_utils.quat_mul(
            wrist_orientation_command_e,
            wrist_orientation_residual,
        )

        # 5. Compute tracking target for finger joints
        self._processed_actions[:, 7:] = (
            self._finger_joint_pos_command() + actions[:, 6:]
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term."""
        self._raw_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.wrist_forces[env_ids] = 0.0
        self.wrist_torques[env_ids] = 0.0
        self.finger_joint_pos[env_ids] = 0.0

    def apply_actions(self) -> None:
        """Apply the actions, recomputing the tracking actions to allow better performance."""
        # 1. Extract current wrist position and orientation
        wrist_position = self._wrist_position_e()
        wrist_wxyz = self._wrist_wxyz_e()
        wrist_linvel_b = self.robot.data.root_link_lin_vel_b
        wrist_angvel_b = self.robot.data.root_link_ang_vel_b

        # 2. PD for wrist force control
        wrist_position_error_e = self.processed_actions[:, :3] - wrist_position
        wrist_position_error_b = math_utils.quat_apply_inverse(
            wrist_wxyz, wrist_position_error_e
        )
        force = (
            self.cfg.tracking_controller_linear_stiffness * wrist_position_error_b
            - self.cfg.tracking_controller_linear_damping * wrist_linvel_b
        )

        # 3. PD for wrist torque control
        wrist_orientation_error_b = math_utils.quat_mul(
            math_utils.quat_inv(wrist_wxyz),
            self.processed_actions[:, 3:7],  # w_t_current.inv() * w_t_target
        )
        wrist_orientation_error_b = math_utils.axis_angle_from_quat(
            wrist_orientation_error_b
        )
        torque = (
            self.cfg.tracking_controller_angular_stiffness * wrist_orientation_error_b
            - self.cfg.tracking_controller_angular_damping * wrist_angvel_b
        )

        # 4. Gravity compensation
        gravity_compensation_e = (
            self.robot.root_physx_view.get_gravity_compensation_forces()
        )
        gravity_compensation_force_b = math_utils.quat_apply_inverse(
            wrist_wxyz, gravity_compensation_e[..., :3]
        )
        self.wrist_forces[:] = torch.clamp(
            force + gravity_compensation_force_b,
            min=-self.cfg.max_force,
            max=self.cfg.max_force,
        )

        gravity_compensation_torque_b = math_utils.quat_apply_inverse(
            wrist_wxyz, gravity_compensation_e[..., 3:6]
        )
        self.wrist_torques[:] = torch.clamp(
            torque + gravity_compensation_torque_b,
            min=-self.cfg.max_torque,
            max=self.cfg.max_torque,
        )

        # 5. Clip finger joint position to torque limits
        self.finger_joint_pos[:] = self.clip_actions_to_torque_limit(
            self.processed_actions[:, 7:]
        )

        # 6. Set wrist wrench control
        self.robot.set_external_force_and_torque(
            forces=self.wrist_forces.reshape(self.num_envs, 1, 3),
            torques=self.wrist_torques.reshape(self.num_envs, 1, 3),
            body_ids=self.wrist_body_id,
            is_global=False,
        )

        # 7. Set finger joint position control
        self._asset.set_joint_position_target(
            self.finger_joint_pos, joint_ids=self.finger_joint_ids
        )
