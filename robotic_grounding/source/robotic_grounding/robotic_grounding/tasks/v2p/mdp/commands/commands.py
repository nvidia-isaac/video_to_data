# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Command definitions for the V2P environment."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TrackingCommand(CommandTerm):
    """Command term that generates pose commands for tracking task."""

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Load motion data
        assert os.path.exists(cfg.file_path), f"Motion file not found: {cfg.file_path}"
        qpos_data = torch.from_numpy(np.load(cfg.file_path)).to(self.device)
        self.pos_trajectory_w = qpos_data[:, :3] + torch.tensor(
            cfg.pos_offset, dtype=torch.float, device=self.device
        )
        self.quat_trajectory_w = qpos_data[:, 3:7]
        self.num_timesteps = len(qpos_data)

        self.joint_ids, self.joint_names = self.robot.find_joints(self.cfg.joint_names)
        MUJOCO_TO_ISAAC_UPPER_BODY = [
            cfg.file_joint_order.index(joint_name) for joint_name in self.joint_names
        ]
        self.target_full_body_joint_pos = qpos_data[:, 7:]
        self.target_upper_body_joint_pos = self.target_full_body_joint_pos[
            :, MUJOCO_TO_ISAAC_UPPER_BODY
        ]

        # Load tips_distance if available (ManipTrans approach)
        self.tips_distance: torch.Tensor | None = None
        if (
            hasattr(cfg, "tips_distance_file_path")
            and cfg.tips_distance_file_path is not None
        ):
            tips_dist_path = Path(cfg.tips_distance_file_path)
            if tips_dist_path.exists():
                tips_data = (
                    torch.from_numpy(np.load(tips_dist_path)).to(self.device).float()
                )
                # Verify shape: (T, 10) - 5 per hand
                if (
                    tips_data.shape[0] == self.num_timesteps
                    and tips_data.shape[1] == 10
                ):
                    self.tips_distance = tips_data
                else:
                    print(
                        f"Warning: tips_distance shape mismatch. "
                        f"Expected ({self.num_timesteps}, 10), got {tips_data.shape}"
                    )
            else:
                print(f"Warning: tips_distance file not found: {tips_dist_path}")

        # -- buffer
        self.timestep_counter = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.pos_command_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat_command_e = torch.zeros(self.num_envs, 4, device=self.device)
        self.upper_body_joint_pos_command = torch.zeros(
            self.num_envs, len(self.joint_names), device=self.device
        )

        # -- unit vectors
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["joint_pos_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """String representation of the command term."""
        msg = "TrackingCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 38)."""
        return torch.cat(
            (
                self.pos_command_e,
                self.quat_command_e,
                self.upper_body_joint_pos_command,
            ),
            dim=-1,
        )

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics."""
        pos_command_w = self.pos_command_e + self._env.scene.env_origins
        self.metrics["position_error"] = torch.norm(
            self.robot.data.root_pos_w - pos_command_w, dim=1
        )

        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.robot.data.root_quat_w, self.quat_command_e
        )

        self.metrics["joint_pos_error"] = torch.norm(
            self.robot.data.joint_pos[..., self.joint_ids]
            - self.upper_body_joint_pos_command,
            dim=1,
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command."""
        # Reset the timestep counter to 0
        # FIXME: this can also be reset to a random value between 0 and num_timesteps - 1
        self.timestep_counter[env_ids] *= 0
        # Reset the command
        self.pos_command_e[env_ids] = self.pos_trajectory_w[
            self.timestep_counter[env_ids]
        ].float()
        self.quat_command_e[env_ids] = (
            math_utils.quat_unique(
                self.quat_trajectory_w[self.timestep_counter[env_ids]].float()
            )
            if self.cfg.make_quat_unique
            else self.quat_trajectory_w[self.timestep_counter[env_ids]].float()
        )
        self.upper_body_joint_pos_command[env_ids] = self.target_upper_body_joint_pos[
            self.timestep_counter[env_ids]
        ].float()

        self._reset_robot(env_ids)

    def _reset_robot(self, env_ids: Sequence[int]) -> None:
        """Reset the robot."""
        # Reset the robot base
        positions = self.pos_command_e[env_ids] + self._env.scene.env_origins[env_ids]
        orientations = self.quat_command_e[env_ids]
        self.robot.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), env_ids=env_ids
        )
        self.robot.write_root_velocity_to_sim(
            torch.zeros_like(self.robot.data.root_vel_w[env_ids]), env_ids=env_ids
        )

        # Reset the robot joints
        self.robot.write_joint_state_to_sim(
            self.robot.data.joint_pos[env_ids],
            torch.zeros_like(self.robot.data.joint_vel[env_ids]),
            env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(
            self.upper_body_joint_pos_command[env_ids],
            torch.zeros_like(self.robot.data.joint_vel[..., self.joint_ids][env_ids]),
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )

    def _update_command(self) -> None:
        """Update the command."""
        if self.cfg.update_goal_on_reach:
            pos_command_w = self.pos_command_e + self._env.scene.env_origins
            pos_error = torch.norm(self.robot.data.root_pos_w - pos_command_w, dim=1)
            if pos_error < self.cfg.goal_reach_threshold:
                self.timestep_counter += 1
        else:
            self.timestep_counter += 1

        self.pos_command_e = self.pos_trajectory_w[self.timestep_counter].float()
        self.quat_command_e = (
            math_utils.quat_unique(
                self.quat_trajectory_w[self.timestep_counter].float()
            )
            if self.cfg.make_quat_unique
            else self.quat_trajectory_w[self.timestep_counter].float()
        )
        self.upper_body_joint_pos_command = self.target_upper_body_joint_pos[
            self.timestep_counter
        ].float()

    def get_tips_distance(self) -> torch.Tensor | None:
        """Get pre-computed tips_distance for current timestep per environment.

        This follows the ManipTrans approach of using pre-computed reference distances
        from MANO fingertips to object surface for contact rewards.

        Returns:
            Tips distance tensor of shape (num_envs, 10) with distances for each fingertip,
            or None if tips_distance data is not available.
            Order: left_thumb, left_index, left_middle, left_ring, left_pinky,
                   right_thumb, right_index, right_middle, right_ring, right_pinky.
        """
        if self.tips_distance is None:
            return None
        # Index by per-env timestep counter (clamped to valid range)
        indices = self.timestep_counter.clamp(0, self.num_timesteps - 1)
        return self.tips_distance[indices]  # (num_envs, 10)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set the debug visibility."""
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event: Any) -> None:
        """Visualize the goal marker."""
        del event  # unused
        # visualize the goal marker
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_e,
            orientations=self.quat_command_e,
        )
