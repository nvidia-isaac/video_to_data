# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg


class JointDirectPositionAction(ActionTerm):
    """Joint-space action term: policy directly outputs PD targets.

    Unlike JointResidualWithTrackingAction, there is no base tracking
    controller following a reference trajectory. The policy output
    (after scaling, EMA, clipping) IS the target position directly.

    Control is split by subsystem:

    - **Wrist**: Policy outputs target position and orientation.
      External force and torque are applied via PD control with
      gravity compensation.
    - **Finger joints**: Policy outputs target joint positions.
      The simulation PD controller drives the joints.

    Action dimension: 3 (wrist position) + 3 (wrist orientation,
    Euler) + num_finger_joints.
    """

    cfg: actions_cfg.JointDirectPositionActionCfg
    """The configuration of the action term."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action."""

    def __init__(
        self, cfg: actions_cfg.JointDirectPositionActionCfg, env: ManagerBasedEnv
    ) -> None:
        """Initialize the action term."""
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
        else:
            self.robot = self.command.left_robot
            self.wrist_body_id = self.command.left_wrist_body_id
            self.finger_joint_names = self.command.left_finger_joint_names
            self.finger_joint_ids = self.command.left_finger_joint_ids
            self._wrist_position_e = lambda: self.command.left_hand_wrist_position_e
            self._wrist_wxyz_e = lambda: self.command.left_hand_wrist_wxyz_e

        # Create tensors for policy raw and applied actions
        self._raw_actions = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
        )
        self._processed_actions = torch.zeros(
            self.num_envs,
            self.action_dim + 1,
            device=self.device,  # Store wrist orientation as a quaternion (+1 dim)
        )
        # Identity quaternion for wrist orientation.
        self._processed_actions[:, 3] = 1.0
        # Track which envs need target re-init after reset
        self._needs_target_init = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
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

        # Set stiffness and damping for the PD controller
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
        """The processed actions: target positions for PD control."""
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:  # noqa: N802
        """The IO descriptor of the action term."""
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointDirectPositionAction"
        self._IO_descriptor.joint_names = self.finger_joint_ids
        self._IO_descriptor.scale = self._scale
        return self._IO_descriptor

    """
    Operations.
    """

    def clip_actions_to_torque_limit(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip finger joint position targets so PD torque stays within effort limits."""
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
        """Process actions: wrist accumulates deltas, fingers are direct.

        Wrist: target_t = target_{t-1} + clipped(scaled_action).
        The policy accumulates deltas to overcome persistent errors.

        Fingers: scaled action is the target directly.
        """
        # 0. Initialize target for freshly reset envs
        #    Deferred from reset() because action reset runs BEFORE
        #    command reset. By process_actions time, sim buffers are
        #    fresh (sim.forward + scene.update in _resample_command).
        if self._needs_target_init.any():
            init_ids = self._needs_target_init.nonzero(as_tuple=False).squeeze(-1)
            self._processed_actions[init_ids, :3] = self._wrist_position_e()[init_ids]
            self._processed_actions[init_ids, 3:7] = self._wrist_wxyz_e()[init_ids]
            self._needs_target_init[init_ids] = False

        # 1. Store the raw actions from the policy
        self._raw_actions[:] = actions

        # 2. Scale and EMA filter
        actions = (
            self.ema_factor * self.prev_actions
            + (1 - self.ema_factor) * actions * self._scale
        )
        self.prev_actions[:] = actions

        # 3. Clip deltas around zero
        actions = torch.clamp(actions, min=-self._clip, max=self._clip)

        # 4. Wrist position: accumulate delta on previous target
        self._processed_actions[:, :3] = self._processed_actions[:, :3] + actions[:, :3]

        # 5. Wrist orientation: accumulate delta rotation
        ori_delta = math_utils.quat_from_euler_xyz(
            actions[:, 3],
            actions[:, 4],
            actions[:, 5],
        )
        self._processed_actions[:, 3:7] = math_utils.quat_mul(
            self._processed_actions[:, 3:7],
            ori_delta,
        )

        # 6. Finger joint positions: direct target
        self._processed_actions[:, 7:] = actions[:, 6:]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term."""
        self._raw_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.wrist_forces[env_ids] = 0.0
        self.wrist_torques[env_ids] = 0.0
        self.finger_joint_pos[env_ids] = 0.0

        # Mark envs for target re-init in process_actions
        # (command reset runs AFTER action reset, so command
        #  reference is stale here)
        self._needs_target_init[env_ids] = True

        # Clear the external forces and torques
        self.robot.set_external_force_and_torque(
            forces=self.zero_force_torque[env_ids],
            torques=self.zero_force_torque[env_ids],
            body_ids=self.wrist_body_id,
            env_ids=env_ids,
            is_global=False,
        )

    def apply_actions(self) -> None:
        """Apply the PD control to track the target positions."""
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
            self.processed_actions[:, 3:7],
        )
        wrist_orientation_error_b = math_utils.axis_angle_from_quat(
            wrist_orientation_error_b
        )
        torque = (
            self.cfg.tracking_controller_angular_stiffness * wrist_orientation_error_b
            - self.cfg.tracking_controller_angular_damping * wrist_angvel_b
        )

        # 4. Gravity compensation (PhysX articulation-aware)
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
