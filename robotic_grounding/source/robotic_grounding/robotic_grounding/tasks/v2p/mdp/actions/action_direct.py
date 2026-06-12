# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

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


@torch.jit.script
def _process_direct_actions(
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    raw_actions: torch.Tensor,
    processed_actions: torch.Tensor,
    finger_joint_limits: torch.Tensor,
    scale: torch.Tensor,
    clip: torch.Tensor,
    ema_factor: float,
) -> None:
    # Store raw policy outputs.
    raw_actions[:] = actions

    # EMA-filtered, scaled, and delta-clipped.
    actions = ema_factor * prev_actions + (1.0 - ema_factor) * actions * scale
    actions = torch.clamp(actions, min=-clip, max=clip)

    # Clip finger targets to joint position limits.
    actions[:, 6:] = torch.clamp(
        actions[:, 6:],
        min=finger_joint_limits[..., 0],
        max=finger_joint_limits[..., 1],
    )
    prev_actions[:] = actions

    # Wrist position: accumulate delta on previous target.
    processed_actions[:, :3] = processed_actions[:, :3] + actions[:, :3]

    # Wrist orientation: accumulate delta rotation.
    ori_delta = quat_from_euler_xyz(actions[:, 3], actions[:, 4], actions[:, 5])
    processed_actions[:, 3:7] = quat_mul(processed_actions[:, 3:7], ori_delta)

    # Finger joint targets: direct (already clipped above).
    processed_actions[:, 7:] = actions[:, 6:]


@torch.jit.script
def _compute_wrist_wrench_direct(
    wrist_position: torch.Tensor,
    wrist_wxyz: torch.Tensor,
    wrist_link_vel_w: torch.Tensor,
    wrist_link_quat_w: torch.Tensor,
    wrist_com_pos_b: torch.Tensor,
    wrist_pose_target: torch.Tensor,
    gravity_comp_e: torch.Tensor,
    linear_stiffness: float,
    linear_damping: float,
    angular_stiffness: float,
    angular_damping: float,
    max_force: float,
    max_torque: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    return force, torque


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

        # Save the wrist com pose in the body frame
        self.wrist_com_pose_b = self.robot.data._root_physx_view.get_coms()[:, 0].to(
            self.device
        )  # quat in xyzw format

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

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process actions: wrist accumulates deltas, fingers are direct.

        Wrist: target_t = target_{t-1} + clipped(scaled_action).
        The policy accumulates deltas to overcome persistent errors.

        Fingers: scaled action is the target directly.
        """
        # Initialize target for freshly reset envs. Deferred from reset()
        # because action reset runs BEFORE command reset — by the time
        # process_actions runs, sim buffers are fresh.
        if self._needs_target_init.any():
            init_ids = self._needs_target_init.nonzero(as_tuple=False).squeeze(-1)
            self._processed_actions[init_ids, :3] = self._wrist_position_e()[init_ids]
            self._processed_actions[init_ids, 3:7] = self._wrist_wxyz_e()[init_ids]
            self._needs_target_init[init_ids] = False

        _process_direct_actions(
            actions=actions,
            prev_actions=self.prev_actions,
            raw_actions=self._raw_actions,
            processed_actions=self._processed_actions,
            finger_joint_limits=self.robot.data.joint_pos_limits[
                :, self.finger_joint_ids
            ],
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
        force, torque = _compute_wrist_wrench_direct(
            wrist_position=self._wrist_position_e(),
            wrist_wxyz=self._wrist_wxyz_e(),
            wrist_link_vel_w=self.robot.data.root_com_vel_w,
            wrist_link_quat_w=self.robot.data.root_link_quat_w,
            wrist_com_pos_b=self.wrist_com_pose_b[..., :3],
            wrist_pose_target=self._processed_actions[:, :7],
            gravity_comp_e=self.robot.root_physx_view.get_gravity_compensation_forces(),
            linear_stiffness=self._tracking_controller_linear_stiffness,
            linear_damping=self._tracking_controller_linear_damping,
            angular_stiffness=self._tracking_controller_angular_stiffness,
            angular_damping=self._tracking_controller_angular_damping,
            max_force=float(self.cfg.max_force),
            max_torque=float(self.cfg.max_torque),
        )

        self.wrist_forces[:] = force
        self.wrist_torques[:] = torque
        self.finger_joint_pos[:] = self._processed_actions[:, 7:]

        self.robot.set_external_force_and_torque(
            forces=self.wrist_forces.reshape(self.num_envs, 1, 3),
            torques=self.wrist_torques.reshape(self.num_envs, 1, 3),
            body_ids=self.wrist_body_id,
            is_global=False,
        )
        self._asset.set_joint_position_target(
            self.finger_joint_pos, joint_ids=self.finger_joint_ids
        )
