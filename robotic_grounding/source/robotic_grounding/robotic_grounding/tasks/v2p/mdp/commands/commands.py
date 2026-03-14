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
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers

from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.tasks.v2p.mdp.utils import (
    interpolate_robot_motion_data,
)

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
        self.QUAT_UNIT_VEC = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat((self.num_envs, 1))

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


class DualHandsObjectTrackingCommand(CommandTerm):
    """Command term that generates pose commands for dual-hand object tracking.

    This term is the central component for the tracking task. It is responsible for:

    1. **Scene references**: Robot (left/right hand) and object asset pointers and
       body/joint indices used for reading state and applying commands.

    2. **Command trajectories**: Loading retargeted motion data (from parquet),
       interpolating to the env FPS, and exposing current command poses (wrist
       position/orientation, finger joint positions, object pose) in world and
       env frames.

    3. **Current state**: Reading robot and object state (positions, orientations,
       joint positions/velocities, fingertip poses) for rewards and observations.

    4. **Resampling and reset**: Sampling new trajectories and resetting envs to
       initial or arbitrary states from saved reset data.

    5. **Visualization**: Goal-pose and trajectory visualization when debug vis
       is enabled.

    All task-related attributes and methods are defined here when possible.
    """

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the dual-hand object tracking command term.

        Loads motion data from the configured folder, resolves scene assets
        (object and left/right robots), and sets up command buffers and reset
        state tensors.

        Args:
            cfg: Command term configuration (motion path, asset names, FPS, etc.).
            env: The RL environment instance; used to resolve scene assets.
        """
        # initialize the base class
        super().__init__(cfg, env)
        self.step_dt = env.step_dt

        # Retrive object asset from the environment
        self.object: RigidObject = env.scene[cfg.object_name]
        self.object_body_ids, self.object_body_names = self.object.find_bodies(
            cfg.object_body_names
        )
        self.object_body_ids = torch.tensor(self.object_body_ids, device=self.device)

        # Retrive both side robots
        for side in ["right", "left"]:
            side_robot_name = getattr(cfg, f"{side}_robot_name")
            side_robot = env.scene[side_robot_name]
            side_finger_joint_names = side_robot.data.joint_names
            finger_joint_ids, _ = side_robot.find_joints(side_finger_joint_names)
            finger_joint_ids = torch.tensor(finger_joint_ids, device=self.device)
            setattr(self, f"{side}_robot", side_robot)
            setattr(self, f"{side}_finger_joint_names", side_finger_joint_names)
            setattr(self, f"{side}_finger_joint_ids", finger_joint_ids)

            wrist_body_ids, wrist_body_name = side_robot.find_bodies(
                cfg.wrist_body_name
            )
            setattr(
                self,
                f"{side}_wrist_body_id",
                torch.tensor(wrist_body_ids, device=self.device),
            )
            setattr(self, f"{side}_wrist_body_name", wrist_body_name)

            fingertip_body_ids, fingertip_body_names = side_robot.find_bodies(
                cfg.fingertip_body_name
            )
            setattr(
                self,
                f"{side}_fingertip_body_ids",
                torch.tensor(fingertip_body_ids, device=self.device),
            )
            setattr(self, f"{side}_fingertip_body_names", fingertip_body_names)

        # Retrive contact sensor from the environment
        self.right_object_contact_sensor = env.scene[
            env.cfg.object_contact_sensor_names[0]
        ]
        self.left_object_contact_sensor = env.scene[
            env.cfg.object_contact_sensor_names[1]
        ]

        # Load motion data
        try:
            self._retargeted_motion_data = ManoSharpaData.from_parquet(
                root_path=str(cfg.motion_folder),
                filters=cfg.motion_filters,
                trajectory_id=cfg.motion_id,
            )
            # Interpolate the motion data to the target FPS
            target_fps = (
                cfg.target_fps if cfg.target_fps is not None else 1 / self.step_dt
            )
            self._retargeted_motion_data = interpolate_robot_motion_data(
                motion_data=self._retargeted_motion_data,
                target_fps=target_fps,
            )

        except Exception as e:
            raise ValueError(
                "Failed to load retargeted motion data from "
                f"{cfg.motion_folder} with filters {cfg.motion_filters} and "
                f"trajectory_id {cfg.motion_id}. Please check if the data exists "
                f"and is valid. Error: {e}"
            ) from e

        # Buffers
        self.retargeted_horizon = len(
            self._retargeted_motion_data.robot_right_wrist_position
        )
        self.timestep_counter = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.tracking_lengths = self.retargeted_horizon * torch.ones(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.steps_since_last_reset = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # Reset wrist pose buffers (written by _resample_command,
        # read by action terms for PD target initialization)
        self.reset_right_wrist_position_e = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.reset_right_wrist_wxyz = torch.zeros(self.num_envs, 4, device=self.device)
        self.reset_left_wrist_position_e = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.reset_left_wrist_wxyz = torch.zeros(self.num_envs, 4, device=self.device)

        # Hand data
        for side in ["right", "left"]:
            # Store wrist position and orientation
            retargeted_wrist_position = getattr(
                self._retargeted_motion_data, f"robot_{side}_wrist_position"
            )
            retargeted_wrist_wxyz = getattr(
                self._retargeted_motion_data, f"robot_{side}_wrist_wxyz"
            )
            setattr(
                self,
                f"retargeted_{side}_wrist_position",
                torch.tensor(retargeted_wrist_position, device=self.device),
            )
            setattr(
                self,
                f"retargeted_{side}_wrist_wxyz",
                torch.tensor(retargeted_wrist_wxyz, device=self.device),
            )

            # Store finger joints in ISAAC joint order
            retargeted_finger_joint_names = getattr(
                self._retargeted_motion_data, f"{side}_robot_finger_joint_names"
            )
            isaac_finger_joint_names = getattr(self, f"{side}_finger_joint_names")
            retargeted_to_isaac_joint_order = [
                retargeted_finger_joint_names.index(joint_name)
                for joint_name in isaac_finger_joint_names
            ]
            retargeted_finger_joints = getattr(
                self._retargeted_motion_data, f"robot_{side}_finger_joints"
            )
            retargeted_finger_joints = torch.tensor(
                retargeted_finger_joints, device=self.device
            )[:, retargeted_to_isaac_joint_order]
            setattr(self, f"retargeted_{side}_finger_joints", retargeted_finger_joints)

            # Store hand frame data
            retargeted_hand_frames = getattr(
                self._retargeted_motion_data, f"robot_{side}_frames"
            )
            retargeted_hand_frame_names = getattr(
                self._retargeted_motion_data, f"{side}_robot_frame_names"
            )
            setattr(
                self,
                f"retargeted_{side}_hand_frames",
                torch.tensor(retargeted_hand_frames, device=self.device),
            )
            setattr(
                self, f"retargeted_{side}_hand_frame_names", retargeted_hand_frame_names
            )

            # Command fingertip index
            fingertip_body_names = getattr(self, f"{side}_fingertip_body_names")
            retargeted_fingertip_indices = []
            for fingertip_body_name in fingertip_body_names:
                fingertip_index = retargeted_hand_frame_names.index(fingertip_body_name)
                retargeted_fingertip_indices.append(fingertip_index)
            setattr(
                self,
                f"retargeted_{side}_fingertip_indices",
                torch.tensor(retargeted_fingertip_indices, device=self.device),
            )

        # Object data
        self.retargeted_object_body_position = torch.tensor(
            self._retargeted_motion_data.object_body_position, device=self.device
        )
        self.retargeted_object_body_wxyz = torch.tensor(
            self._retargeted_motion_data.object_body_wxyz, device=self.device
        )
        self.retargeted_object_articulation = torch.tensor(
            self._retargeted_motion_data.object_articulation, device=self.device
        )

        # Load contact links positions, object contact positions, and object contact normals from parquet
        # Shape: (horizon, num_links, 4)
        # 4 = xyz/normal + part_id
        self.retargeted_left_link_contact_positions_e = torch.tensor(
            self._retargeted_motion_data.mano_left_link_contact_positions,
            dtype=torch.float32,
            device=self.device,
        )
        self.retargeted_left_object_contact_positions_e = torch.tensor(
            self._retargeted_motion_data.mano_left_object_contact_positions,
            dtype=torch.float32,
            device=self.device,
        )
        self.retargeted_left_object_contact_normals_e = torch.tensor(
            self._retargeted_motion_data.mano_left_object_contact_normals,
            dtype=torch.float32,
            device=self.device,
        )

        self.retargeted_right_link_contact_positions_e = torch.tensor(
            self._retargeted_motion_data.mano_right_link_contact_positions,
            dtype=torch.float32,
            device=self.device,
        )
        self.retargeted_right_object_contact_positions_e = torch.tensor(
            self._retargeted_motion_data.mano_right_object_contact_positions,
            dtype=torch.float32,
            device=self.device,
        )
        self.retargeted_right_object_contact_normals_e = torch.tensor(
            self._retargeted_motion_data.mano_right_object_contact_normals,
            dtype=torch.float32,
            device=self.device,
        )

        # Pre-compute left hand contact positions in object frame
        self.retargeted_left_object_contact_positions_o = torch.zeros_like(
            self.retargeted_left_link_contact_positions_e[..., :3]
        )
        self.retargeted_left_object_contact_is_valid = (
            self.retargeted_left_link_contact_positions_e[..., 3] > 1e-5
        )
        self.retargeted_left_object_has_contact = (
            self.retargeted_left_object_contact_is_valid.sum(dim=-1) > 1e-5
        )
        for link_idx in range(self.retargeted_left_object_contact_positions_e.shape[1]):
            self.retargeted_left_object_contact_positions_o[:, link_idx], _ = (
                math_utils.subtract_frame_transforms(
                    self.retargeted_object_body_position[
                        :, self.object_body_ids
                    ].squeeze(),
                    self.retargeted_object_body_wxyz[:, self.object_body_ids].squeeze(),
                    self.retargeted_left_object_contact_positions_e[:, link_idx, :3],
                    q02=None,
                )
            )
        self.retargeted_left_object_contact_positions_o[
            ~self.retargeted_left_object_contact_is_valid
        ] = 0.0

        # Pre-compute right hand contact positions in object frame
        self.retargeted_right_object_contact_positions_o = torch.zeros_like(
            self.retargeted_right_link_contact_positions_e[..., :3]
        )
        self.retargeted_right_object_contact_is_valid = (
            self.retargeted_right_link_contact_positions_e[..., 3] > 1e-5
        )
        self.retargeted_right_object_has_contact = (
            self.retargeted_right_object_contact_is_valid.sum(dim=-1) > 1e-5
        )
        for link_idx in range(
            self.retargeted_right_object_contact_positions_e.shape[1]
        ):
            self.retargeted_right_object_contact_positions_o[:, link_idx], _ = (
                math_utils.subtract_frame_transforms(
                    self.retargeted_object_body_position[
                        :, self.object_body_ids
                    ].squeeze(),
                    self.retargeted_object_body_wxyz[:, self.object_body_ids].squeeze(),
                    self.retargeted_right_object_contact_positions_e[:, link_idx, :3],
                    q02=None,
                )
            )
        self.retargeted_right_object_contact_positions_o[
            ~self.retargeted_right_object_contact_is_valid
        ] = 0.0

        # Pre-compute left hand frame positions in object frame
        self.retargeted_left_hand_frame_positions_o = torch.zeros_like(
            self.retargeted_left_hand_frames[..., :3]
        )
        for frame_idx in range(self.retargeted_left_hand_frames.shape[1]):
            self.retargeted_left_hand_frame_positions_o[:, frame_idx], _ = (
                math_utils.subtract_frame_transforms(
                    self.retargeted_object_body_position[
                        :, self.object_body_ids
                    ].squeeze(),
                    self.retargeted_object_body_wxyz[:, self.object_body_ids].squeeze(),
                    self.retargeted_left_hand_frames[:, frame_idx, :3],
                    q02=None,
                )
            )

        # Pre-compute right hand frame positions in object frame
        self.retargeted_right_hand_frame_positions_o = torch.zeros_like(
            self.retargeted_right_hand_frames[..., :3]
        )
        for frame_idx in range(self.retargeted_right_hand_frames.shape[1]):
            self.retargeted_right_hand_frame_positions_o[:, frame_idx], _ = (
                math_utils.subtract_frame_transforms(
                    self.retargeted_object_body_position[
                        :, self.object_body_ids
                    ].squeeze(),
                    self.retargeted_object_body_wxyz[:, self.object_body_ids].squeeze(),
                    self.retargeted_right_hand_frames[:, frame_idx, :3],
                    q02=None,
                )
            )

        # Pre-compute left hand wrist position and wxyz in object frame
        self.retargeted_left_wrist_position_o, self.retargeted_left_wrist_wxyz_o = (
            math_utils.subtract_frame_transforms(
                self.retargeted_object_body_position[:, self.object_body_ids].squeeze(),
                self.retargeted_object_body_wxyz[:, self.object_body_ids].squeeze(),
                self.retargeted_left_wrist_position,
                self.retargeted_left_wrist_wxyz,
            )
        )

        # Pre-compute right hand wrist position and wxyz in object frame
        self.retargeted_right_wrist_position_o, self.retargeted_right_wrist_wxyz_o = (
            math_utils.subtract_frame_transforms(
                self.retargeted_object_body_position[:, self.object_body_ids].squeeze(),
                self.retargeted_object_body_wxyz[:, self.object_body_ids].squeeze(),
                self.retargeted_right_wrist_position,
                self.retargeted_right_wrist_wxyz,
            )
        )

        # If debug vis is on, create contact markers
        self._get_contact_counts()
        self._set_contact_vis_impl(getattr(self.cfg, "debug_vis", False))

        # Virtual object controller scale factor for curriculum
        self.virtual_object_controller_scale_factor = torch.tensor(
            [cfg.initial_virtual_object_control_curriculum_scale], device=self.device
        )  # (1,)
        self.virtual_object_controller_scale_factor_per_env = (
            cfg.initial_virtual_object_control_curriculum_scale
            * torch.ones(self.num_envs, 1, device=self.device)  # (num_envs, 1)
        )

        # Unit vectors
        self.X_UNIT_VEC = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.Y_UNIT_VEC = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.Z_UNIT_VEC = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        # Precompute unit vectors for reward computation
        self.QUAT_UNIT_VEC = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat((self.num_envs, 1))
        self.KEYPOINT_VECS = (
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1, 1)
        )  # (num_envs, 6, 3)

        # Metrics
        self.metrics["right_hand_wrist_position_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["right_hand_wrist_wxyz_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["right_hand_finger_joints_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

        self.metrics["left_hand_wrist_position_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["left_hand_wrist_wxyz_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["left_hand_finger_joints_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

        self.metrics["object_body_position_error"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["object_body_wxyz_error"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["object_articulation_error"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.metrics["virtual_object_controller_scale_factor"] = (
            cfg.initial_virtual_object_control_curriculum_scale
            * torch.ones(self.num_envs, device=self.device)
        )

    def __str__(self) -> str:
        """String representation of the command term."""
        msg = f"{self.__class__.__name__}:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    ##########################################################
    # Commands
    ##########################################################

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, -1)."""
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        right_wrist_current_p_command, right_wrist_current_q_command = (
            math_utils.subtract_frame_transforms(
                self.right_hand_wrist_position_e,
                self.right_hand_wrist_wxyz_e,
                right_hand_wrist_pose_command_e[:, :3],
                right_hand_wrist_pose_command_e[:, 3:],
            )
        )
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        left_wrist_current_p_command, left_wrist_current_q_command = (
            math_utils.subtract_frame_transforms(
                self.left_hand_wrist_position_e,
                self.left_hand_wrist_wxyz_e,
                left_hand_wrist_pose_command_e[:, :3],
                left_hand_wrist_pose_command_e[:, 3:],
            )
        )

        right_joint_pos_delta = (
            self.right_hand_finger_joint_pos_command - self.right_robot.data.joint_pos
        )
        left_joint_pos_delta = (
            self.left_hand_finger_joint_pos_command - self.left_robot.data.joint_pos
        )

        object_current_p_command, object_current_q_command = (
            math_utils.subtract_frame_transforms(
                self.object_position_e.squeeze(1),
                self.object_orientation_e.squeeze(1),
                self.object_body_position_command_e,
                self.object_body_wxyz_command_e,
            )
        )

        return torch.cat(
            (
                right_wrist_current_p_command,
                right_wrist_current_q_command,
                left_wrist_current_p_command,
                left_wrist_current_q_command,
                right_joint_pos_delta,
                left_joint_pos_delta,
                object_current_p_command,
                object_current_q_command,
            ),
            dim=-1,
        )

    @property
    def right_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """The desired goal position and wxyz in the environment frame for the right hand wrist. Shape is (num_envs, 7)."""
        position, wxyz = math_utils.combine_frame_transforms(
            self.object_position_e.squeeze(1),
            self.object_orientation_e.squeeze(1),
            self.retargeted_right_wrist_position_o[self.timestep_counter],
            self.retargeted_right_wrist_wxyz_o[self.timestep_counter],
        )
        wxyz = math_utils.quat_unique(wxyz) if self.cfg.make_quat_unique else wxyz
        return torch.cat((position, wxyz), dim=-1)

    @property
    def left_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """The desired goal position and wxyz in the environment frame for the left hand wrist. Shape is (num_envs, 7)."""
        position, wxyz = math_utils.combine_frame_transforms(
            self.object_position_e.squeeze(1),
            self.object_orientation_e.squeeze(1),
            self.retargeted_left_wrist_position_o[self.timestep_counter],
            self.retargeted_left_wrist_wxyz_o[self.timestep_counter],
        )
        wxyz = math_utils.quat_unique(wxyz) if self.cfg.make_quat_unique else wxyz
        return torch.cat((position, wxyz), dim=-1)

    @property
    def right_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """The desired goal finger joint position for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.retargeted_right_finger_joints[self.timestep_counter].float()

    @property
    def left_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """The desired goal finger joint position for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.retargeted_left_finger_joints[self.timestep_counter].float()

    @property
    def right_hand_fingertip_position_command_o(self) -> torch.Tensor:
        """The desired goal fingertip position in the object frame for the right hand. Shape is (num_envs, NUM_FINGERTIPS, 7)."""
        return self.retargeted_right_hand_frame_positions_o[self.timestep_counter][
            :, self.retargeted_right_fingertip_indices
        ].float()

    @property
    def left_hand_fingertip_position_command_o(self) -> torch.Tensor:
        """The desired goal fingertip position in the object frame for the left hand. Shape is (num_envs, NUM_FINGERTIPS, 7)."""
        return self.retargeted_left_hand_frame_positions_o[self.timestep_counter][
            :, self.retargeted_left_fingertip_indices
        ].float()

    @property
    def right_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """The desired goal fingertip position in the environment frame for the right hand. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return math_utils.combine_frame_transforms(
            self.object_position_e.expand(
                -1, len(self.retargeted_right_fingertip_indices), -1
            ),
            self.object_orientation_e.expand(
                -1, len(self.retargeted_right_fingertip_indices), -1
            ),
            self.right_hand_fingertip_position_command_o,
            q12=None,
        )[0]

    @property
    def left_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """The desired goal fingertip position in the environment frame for the left hand. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return math_utils.combine_frame_transforms(
            self.object_position_e.expand(
                -1, len(self.retargeted_left_fingertip_indices), -1
            ),
            self.object_orientation_e.expand(
                -1, len(self.retargeted_left_fingertip_indices), -1
            ),
            self.left_hand_fingertip_position_command_o,
            q12=None,
        )[0]

    @property
    def object_body_position_command_e(self) -> torch.Tensor:
        """The desired goal position in the environment frame for the object. Shape is (num_envs, NUM_OBJECT_BODY x 3)."""
        return (
            self.retargeted_object_body_position[self.timestep_counter][
                :, self.object_body_ids
            ]
            .reshape(self.num_envs, -1)
            .float()
        )

    @property
    def object_body_wxyz_command_e(self) -> torch.Tensor:
        """The desired goal orientation in the environment frame for the object. Shape is (num_envs, NUM_OBJECT_BODY x 4)."""
        retargeted_object_wxyz = self.retargeted_object_body_wxyz[
            self.timestep_counter
        ][:, self.object_body_ids].float()
        retargeted_object_wxyz = (
            math_utils.quat_unique(retargeted_object_wxyz)
            if self.cfg.make_quat_unique
            else retargeted_object_wxyz
        )
        return retargeted_object_wxyz.reshape(self.num_envs, -1).float()

    @property
    def right_hand_object_contact_command_positions_o(self) -> torch.Tensor:
        """The target contact positions in the object frame for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        return self.retargeted_right_object_contact_positions_o[self.timestep_counter]

    @property
    def right_hand_object_contact_command_positions_e(self) -> torch.Tensor:
        """The target contact positions in the environment frame for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        valid_contact_mask = self.retargeted_right_object_contact_is_valid[
            self.timestep_counter
        ]  # (num_envs, NUM_RIGHT_HAND_OBJECT_CONTACT_POINTS)
        right_hand_object_contact_command_positions_e, _ = (
            math_utils.combine_frame_transforms(
                self.object_position_e.expand(-1, self.num_contacts_right, -1),
                self.object_orientation_e.expand(-1, self.num_contacts_right, -1),
                self.right_hand_object_contact_command_positions_o,
                q12=None,
            )
        )
        right_hand_object_contact_command_positions_e[~valid_contact_mask] = 0.0
        return right_hand_object_contact_command_positions_e

    @property
    def left_hand_object_contact_command_positions_o(self) -> torch.Tensor:
        """The target contact positions in the object frame for the left hand. Shape is (num_envs, NUM_LEFT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        return self.retargeted_left_object_contact_positions_o[self.timestep_counter]

    @property
    def left_hand_object_contact_command_positions_e(self) -> torch.Tensor:
        """The target contact positions in the environment frame for the left hand. Shape is (num_envs, NUM_LEFT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        valid_contact_mask = self.retargeted_left_object_contact_is_valid[
            self.timestep_counter
        ]  # (num_envs, NUM_LEFT_HAND_OBJECT_CONTACT_POINTS)
        left_hand_object_contact_command_positions_e, _ = (
            math_utils.combine_frame_transforms(
                self.object_position_e.expand(-1, self.num_contacts_left, -1),
                self.object_orientation_e.expand(-1, self.num_contacts_left, -1),
                self.left_hand_object_contact_command_positions_o,
                q12=None,
            )
        )
        left_hand_object_contact_command_positions_e[~valid_contact_mask] = 0.0
        return left_hand_object_contact_command_positions_e

    ##########################################################
    # Observations
    ##########################################################

    @property
    def right_hand_wrist_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the right hand wrist. Shape is (num_envs, 3)."""
        return self.right_robot.data.root_link_pos_w

    @property
    def left_hand_wrist_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the left hand wrist. Shape is (num_envs, 3)."""
        return self.left_robot.data.root_link_pos_w

    @property
    def right_hand_wrist_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the right hand wrist. Shape is (num_envs, 3)."""
        right_hand_wrist_position_e = (
            self.right_hand_wrist_position_w - self._env.scene.env_origins
        )
        return right_hand_wrist_position_e.float()

    @property
    def left_hand_wrist_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the left hand wrist. Shape is (num_envs, 3)."""
        left_hand_wrist_position_e = (
            self.left_hand_wrist_position_w - self._env.scene.env_origins
        )
        return left_hand_wrist_position_e.float()

    @property
    def right_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """The current wxyz in the environment frame for the right hand wrist. Shape is (num_envs, 4)."""
        right_wrist_wxyz = self.right_robot.data.root_link_quat_w.float()
        return (
            math_utils.quat_unique(right_wrist_wxyz)
            if self.cfg.make_quat_unique
            else right_wrist_wxyz
        )

    @property
    def left_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """The current wxyz in the environment frame for the left hand wrist. Shape is (num_envs, 4)."""
        left_wrist_wxyz = self.left_robot.data.root_link_quat_w.float()
        return (
            math_utils.quat_unique(left_wrist_wxyz)
            if self.cfg.make_quat_unique
            else left_wrist_wxyz
        )

    @property
    def right_hand_wrist_velocity_b(self) -> torch.Tensor:
        """The current velocity in the body frame for the right hand wrist. Shape is (num_envs, 6)."""
        return torch.cat(
            [
                self.right_robot.data.root_lin_vel_b,
                self.right_robot.data.root_ang_vel_b,
            ],
            dim=-1,
        ).float()

    @property
    def left_hand_wrist_velocity_b(self) -> torch.Tensor:
        """The current velocity in the body frame for the left hand wrist. Shape is (num_envs, 6)."""
        return torch.cat(
            [
                self.left_robot.data.root_lin_vel_b,
                self.left_robot.data.root_ang_vel_b,
            ],
            dim=-1,
        ).float()

    @property
    def right_hand_finger_joint_pos(self) -> torch.Tensor:
        """The current joint position for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.right_robot.data.joint_pos.float()

    @property
    def left_hand_finger_joint_pos(self) -> torch.Tensor:
        """The current joint position for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.left_robot.data.joint_pos.float()

    @property
    def right_hand_finger_joint_vel(self) -> torch.Tensor:
        """The current joint velocity for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.right_robot.data.joint_vel.float()

    @property
    def left_hand_finger_joint_vel(self) -> torch.Tensor:
        """The current joint velocity for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.left_robot.data.joint_vel.float()

    @property
    def right_hand_fingertip_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.right_robot.data.body_link_pos_w[:, self.right_fingertip_body_ids]

    @property
    def left_hand_fingertip_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.left_robot.data.body_link_pos_w[:, self.left_fingertip_body_ids]

    @property
    def right_hand_fingertip_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        fingertip_position_e = (
            self.right_hand_fingertip_position_w
            - self._env.scene.env_origins.unsqueeze(1)
        )
        return fingertip_position_e.float()

    @property
    def left_hand_fingertip_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        fingertip_position_e = (
            self.left_hand_fingertip_position_w
            - self._env.scene.env_origins.unsqueeze(1)
        )
        return fingertip_position_e.float()

    @property
    def right_hand_fingertip_orientation_e(self) -> torch.Tensor:
        """The current orientation in the environment frame for the left and right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 4)."""
        return self.right_robot.data.body_link_quat_w[
            :, self.right_fingertip_body_ids
        ].float()

    @property
    def left_hand_fingertip_orientation_e(self) -> torch.Tensor:
        """The current orientation in the environment frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 4)."""
        return self.left_robot.data.body_link_quat_w[
            :, self.left_fingertip_body_ids
        ].float()

    @property
    def object_position_w(self) -> torch.Tensor:
        """The current position in the environment frame for the object. Shape is (num_envs, NUM_BODY, 3)."""
        return self.object.data.body_link_pos_w[:, self.object_body_ids]

    @property
    def object_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the object. Shape is (num_envs, NUM_BODY, 3)."""
        object_position_e = (
            self.object_position_w - self._env.scene.env_origins.unsqueeze(1)
        )
        return object_position_e.float()

    @property
    def object_orientation_e(self) -> torch.Tensor:
        """The current orientation in the environment frame for the object. Shape is (num_envs, NUM_BODY, 4)."""
        return self.object.data.body_link_quat_w[:, self.object_body_ids].float()

    @property
    def right_hand_object_contact_positions_w(self) -> torch.Tensor:
        """The current contact positions in the world frame for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        contact_positions = self.right_object_contact_sensor.data.contact_pos_w
        contact_positions = torch.nan_to_num(
            contact_positions.view(self.num_envs, -1, 3), nan=0.0
        )
        return contact_positions

    @property
    def right_hand_object_contact_positions_e(self) -> torch.Tensor:
        """The current contact positions in the environment frame for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        _right_hand_object_contact_positions_w = (
            self.right_hand_object_contact_positions_w
        )
        num_links = _right_hand_object_contact_positions_w.shape[1]
        return (
            _right_hand_object_contact_positions_w
            - self._env.scene.env_origins.reshape(self.num_envs, 1, 3).expand(
                -1, num_links, -1
            )
        )

    @property
    def right_hand_object_contact_forces_w(self) -> torch.Tensor:
        """The current contact forces in the world frame for the right hand. Shape is (num_envs, TIMESTEPS, NUM_RIGHT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        return self.right_object_contact_sensor.data.force_matrix_w_history.view(
            self.num_envs, self.right_object_contact_sensor.cfg.history_length, -1, 3
        )

    @property
    def left_hand_object_contact_positions_w(self) -> torch.Tensor:
        """The current contact positions in the environment frame for the left hand. Shape is (num_envs, NUM_LEFT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        contact_positions = self.left_object_contact_sensor.data.contact_pos_w
        contact_positions = torch.nan_to_num(
            contact_positions.view(self.num_envs, -1, 3), nan=0.0
        )
        return contact_positions

    @property
    def left_hand_object_contact_positions_e(self) -> torch.Tensor:
        """The current contact positions in the environment frame for the left hand. Shape is (num_envs, NUM_LEFT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        _left_hand_object_contact_positions_w = (
            self.left_hand_object_contact_positions_w
        )
        num_links = _left_hand_object_contact_positions_w.shape[1]
        return (
            _left_hand_object_contact_positions_w
            - self._env.scene.env_origins.reshape(self.num_envs, 1, 3).expand(
                -1, num_links, -1
            )
        )

    @property
    def left_hand_object_contact_forces_w(self) -> torch.Tensor:
        """The current contact forces in the world frame for the left hand. Shape is (num_envs, TIMESTEPS, NUM_LEFT_HAND_OBJECT_CONTACT_POINTS, 3)."""
        return self.left_object_contact_sensor.data.force_matrix_w_history.view(
            self.num_envs, self.left_object_contact_sensor.cfg.history_length, -1, 3
        )

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics."""
        # Right hand
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        self.metrics["right_hand_wrist_position_error"] = torch.norm(
            self.right_hand_wrist_position_e - right_hand_wrist_pose_command_e[:, :3],
            dim=-1,
        )
        self.metrics["right_hand_wrist_wxyz_error"] = math_utils.quat_error_magnitude(
            self.right_hand_wrist_wxyz_e, right_hand_wrist_pose_command_e[:, 3:]
        )
        self.metrics["right_hand_finger_joints_error"] = torch.norm(
            self.right_hand_finger_joint_pos - self.right_hand_finger_joint_pos_command,
            dim=-1,
        )
        # Left hand
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        self.metrics["left_hand_wrist_position_error"] = torch.norm(
            self.left_hand_wrist_position_e - left_hand_wrist_pose_command_e[:, :3],
            dim=-1,
        )
        self.metrics["left_hand_wrist_wxyz_error"] = math_utils.quat_error_magnitude(
            self.left_hand_wrist_wxyz_e, left_hand_wrist_pose_command_e[:, 3:]
        )
        self.metrics["left_hand_finger_joints_error"] = torch.norm(
            self.left_hand_finger_joint_pos - self.left_hand_finger_joint_pos_command,
            dim=-1,
        )
        # Object
        self.metrics["object_body_position_error"] = torch.norm(
            self.object_position_e.squeeze(1) - self.object_body_position_command_e,
            dim=-1,
        )
        self.metrics["object_body_wxyz_error"] = math_utils.quat_error_magnitude(
            self.object_orientation_e.squeeze(1),
            self.object_body_wxyz_command_e.squeeze(1),
        )

        self.metrics["virtual_object_controller_scale_factor"] = (
            self.virtual_object_controller_scale_factor
            * torch.ones(self.num_envs, device=self.device)
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command."""
        ##########################################################
        # Resample the command
        ##########################################################

        # Reset to a random frame from the original retargeted motion data
        self.timestep_counter[env_ids] = torch.randint(
            low=0,
            high=self.retargeted_horizon - 1,
            size=(len(env_ids),),
            device=self.device,
            dtype=self.timestep_counter.dtype,
        )

        # Get object position and orientation from the retargeted motion data
        object_position_e = self.retargeted_object_body_position[
            self.timestep_counter[env_ids]
        ][:, self.object_body_ids].squeeze(1)
        object_wxyz = self.retargeted_object_body_wxyz[self.timestep_counter[env_ids]][
            :, self.object_body_ids
        ].squeeze(1)

        # Get robot wrist position and orientation from the retargeted motion data
        right_hand_wrist_position = self.retargeted_right_wrist_position[
            self.timestep_counter[env_ids]
        ]
        right_hand_wrist_wxyz = self.retargeted_right_wrist_wxyz[
            self.timestep_counter[env_ids]
        ]
        left_hand_wrist_position = self.retargeted_left_wrist_position[
            self.timestep_counter[env_ids]
        ]
        left_hand_wrist_wxyz = self.retargeted_left_wrist_wxyz[
            self.timestep_counter[env_ids]
        ]

        # Store reset wrist poses (env frame, no env_origins) for action terms
        self.reset_right_wrist_position_e[env_ids] = right_hand_wrist_position
        self.reset_right_wrist_wxyz[env_ids] = right_hand_wrist_wxyz
        self.reset_left_wrist_position_e[env_ids] = left_hand_wrist_position
        self.reset_left_wrist_wxyz[env_ids] = left_hand_wrist_wxyz

        # Finger joints: interpolate between open (0) and reference
        finger_factor = (
            torch.rand(len(env_ids), 1, device=self.device)
            * self.cfg.reset_finger_openness
        )
        right_hand_finger_joint_pos = (
            finger_factor
            * self.retargeted_right_finger_joints[self.timestep_counter[env_ids]]
        )
        left_hand_finger_joint_pos = (
            finger_factor
            * self.retargeted_left_finger_joints[self.timestep_counter[env_ids]]
        )

        # Update the tracking length
        self.tracking_lengths[env_ids] = (
            self.retargeted_horizon - self.timestep_counter[env_ids]
        ).clamp(min=1)

        # Reset virtual object control curriculum scale factor
        self.virtual_object_controller_scale_factor_per_env[env_ids] = 1.0
        self.steps_since_last_reset[env_ids] = 0

        ##########################################################
        # Reset the object pose
        ##########################################################

        object_position = object_position_e + self._env.scene.env_origins[env_ids]
        object_pose = torch.cat([object_position, object_wxyz], dim=-1).float()
        object_velocity = torch.zeros_like(object_pose[..., :6]).float()

        # Set into the physics simulation
        self.object.write_root_pose_to_sim(object_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(object_velocity, env_ids=env_ids)

        ##########################################################
        # Reset the robot
        ##########################################################

        # Robot wrist position and orientation
        right_hand_wrist_position = (
            right_hand_wrist_position + self._env.scene.env_origins[env_ids]
        )
        right_hand_wrist_pose = torch.cat(
            [right_hand_wrist_position, right_hand_wrist_wxyz], dim=-1
        ).float()
        right_hand_wrist_velocity = torch.zeros_like(
            right_hand_wrist_pose[..., :6]
        ).float()

        left_hand_wrist_position = (
            left_hand_wrist_position + self._env.scene.env_origins[env_ids]
        )
        left_hand_wrist_pose = torch.cat(
            [left_hand_wrist_position, left_hand_wrist_wxyz], dim=-1
        ).float()
        left_hand_wrist_velocity = torch.zeros_like(
            left_hand_wrist_pose[..., :6]
        ).float()

        self.right_robot.write_root_pose_to_sim(right_hand_wrist_pose, env_ids=env_ids)
        self.right_robot.write_root_velocity_to_sim(
            right_hand_wrist_velocity, env_ids=env_ids
        )
        self.left_robot.write_root_pose_to_sim(left_hand_wrist_pose, env_ids=env_ids)
        self.left_robot.write_root_velocity_to_sim(
            left_hand_wrist_velocity, env_ids=env_ids
        )

        # Clear residual external forces from previous episode
        zero_forces = torch.zeros(len(env_ids), 1, 3, device=self.device)
        self.right_robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=zero_forces,
            body_ids=self.right_wrist_body_id,
            env_ids=env_ids,
            is_global=False,
        )
        self.left_robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=zero_forces,
            body_ids=self.left_wrist_body_id,
            env_ids=env_ids,
            is_global=False,
        )

        # Right robot finger joint positions
        right_hand_joint_velocity = torch.zeros_like(
            right_hand_finger_joint_pos
        ).float()
        right_hand_joint_pos_limits = self.right_robot.data.soft_joint_pos_limits[
            env_ids, :
        ]
        right_hand_finger_joint_pos = right_hand_finger_joint_pos.clamp_(
            right_hand_joint_pos_limits[..., 0], right_hand_joint_pos_limits[..., 1]
        )
        self.right_robot.write_joint_state_to_sim(
            right_hand_finger_joint_pos, right_hand_joint_velocity, env_ids=env_ids
        )

        # Left robot finger joint positions
        left_hand_joint_velocity = torch.zeros_like(left_hand_finger_joint_pos).float()
        left_hand_joint_pos_limits = self.left_robot.data.soft_joint_pos_limits[
            env_ids, :
        ]
        left_hand_finger_joint_pos = left_hand_finger_joint_pos.clamp_(
            left_hand_joint_pos_limits[..., 0], left_hand_joint_pos_limits[..., 1]
        )
        self.left_robot.write_joint_state_to_sim(
            left_hand_finger_joint_pos, left_hand_joint_velocity, env_ids=env_ids
        )

        # Force a kinematic/data refresh after reset writes so the first
        # post-reset control step reads synchronized wrist states.
        self._env.sim.forward()
        self._env.scene.update(dt=self._env.physics_dt)

    def _update_command(self) -> None:
        """Update the command."""
        self.steps_since_last_reset += 1

        # Decay virtual object control scale factor toward curriculum scale
        if self.cfg.virtual_object_control_decay_mode == "linear":
            progress = self.steps_since_last_reset.float() / max(
                self.cfg.virtual_object_control_decay_steps, 1
            )
            self.virtual_object_controller_scale_factor_per_env[:] = (
                (
                    1.0
                    + (self.virtual_object_controller_scale_factor - 1.0)
                    * progress.clamp(max=1.0)
                )
                .clamp(min=0.0)
                .view(self.num_envs, 1)
            )
        elif self.cfg.virtual_object_control_decay_mode == "step":
            self.virtual_object_controller_scale_factor_per_env[
                self.steps_since_last_reset
                >= self.cfg.virtual_object_control_decay_steps
            ] = self.virtual_object_controller_scale_factor

        # If still in the reset phase, don't step the timestep counter
        # Note: This will make each episode length vary, which might cause the episode rewards to be variant.
        not_in_reset_phase_env_ids = (
            self.steps_since_last_reset >= self.cfg.virtual_object_control_decay_steps
        )
        self.timestep_counter[not_in_reset_phase_env_ids] += 1

    def _get_contact_counts(self) -> None:
        """Return (num_contacts_left, num_contacts_right) from retargeted contact data."""
        self.num_contacts_left = self.retargeted_left_object_contact_positions_o.shape[
            1
        ]
        self.num_contacts_right = (
            self.retargeted_right_object_contact_positions_o.shape[1]
        )
        self.total_num_contacts = self.num_contacts_left + self.num_contacts_right

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set the debug visibility."""
        if debug_vis:
            # Current state visualizers
            if not hasattr(self, "object_pose_visualizer_cfg"):
                self.object_pose_visualizer = VisualizationMarkers(
                    self.cfg.object_pose_visualizer_cfg
                )
            self.object_pose_visualizer.set_visibility(True)
            if not hasattr(self, "right_hand_pose_visualizer"):
                self.right_hand_pose_visualizer = VisualizationMarkers(
                    self.cfg.right_hand_pose_visualizer_cfg
                )
            self.right_hand_pose_visualizer.set_visibility(True)
            if not hasattr(self, "left_hand_pose_visualizer"):
                self.left_hand_pose_visualizer = VisualizationMarkers(
                    self.cfg.left_hand_pose_visualizer_cfg
                )
            self.left_hand_pose_visualizer.set_visibility(True)

            # Command visualizers
            if not hasattr(self, "object_goal_pose_visualizer_cfg"):
                self.object_goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.object_goal_pose_visualizer_cfg
                )
            self.object_goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "right_hand_goal_pose_visualizer"):
                self.right_hand_goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.right_hand_goal_pose_visualizer_cfg
                )
            self.right_hand_goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "left_hand_goal_pose_visualizer"):
                self.left_hand_goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.left_hand_goal_pose_visualizer_cfg
                )
            self.left_hand_goal_pose_visualizer.set_visibility(True)

        elif hasattr(self, "goal_pose_visualizer"):
            self.object_pose_visualizer.set_visibility(False)
            self.right_hand_pose_visualizer.set_visibility(False)
            self.left_hand_pose_visualizer.set_visibility(False)
            self.object_goal_pose_visualizer.set_visibility(False)
            self.right_hand_goal_pose_visualizer.set_visibility(False)
            self.left_hand_goal_pose_visualizer.set_visibility(False)
            for visualizer in getattr(self, "contact_marker_visualizers", []):
                visualizer.set_visibility(False)
            for visualizer in getattr(self, "policy_contact_marker_visualizers", []):
                visualizer.set_visibility(False)

    def _set_contact_vis_impl(self, debug_vis: bool) -> None:
        """Set the contact visibility."""
        if debug_vis:
            self.command_contact_marker_visualizers = []
            self.policy_contact_marker_visualizers = []
            for i in range(self.total_num_contacts):
                command_contact_vis_cfg = (
                    self.cfg.target_contact_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/TargetContact_{i}"
                    )
                )
                command_contact_marker_visualizer = VisualizationMarkers(
                    command_contact_vis_cfg
                )
                command_contact_marker_visualizer.set_visibility(True)
                self.command_contact_marker_visualizers.append(
                    command_contact_marker_visualizer
                )

                policy_contact_vis_cfg = (
                    self.cfg.current_contact_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/CurrentContact_{i}"
                    )
                )
                policy_contact_marker_visualizer = VisualizationMarkers(
                    policy_contact_vis_cfg
                )
                policy_contact_marker_visualizer.set_visibility(True)
                self.policy_contact_marker_visualizers.append(
                    policy_contact_marker_visualizer
                )

            self.command_fingertip_marker_visualizers = []
            self.policy_fingertip_marker_visualizers = []

            for i in range(
                len(self.right_fingertip_body_ids) + len(self.left_fingertip_body_ids)
            ):
                command_fingertip_vis_cfg = (
                    self.cfg.target_fingertip_position_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/TargetFingertip_{i}"
                    )
                )
                command_fingertip_marker_visualizer = VisualizationMarkers(
                    command_fingertip_vis_cfg
                )
                command_fingertip_marker_visualizer.set_visibility(True)
                self.command_fingertip_marker_visualizers.append(
                    command_fingertip_marker_visualizer
                )

                policy_fingertip_vis_cfg = (
                    self.cfg.current_fingertip_position_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/CurrentFingertip_{i}"
                    )
                )
                policy_fingertip_marker_visualizer = VisualizationMarkers(
                    policy_fingertip_vis_cfg
                )
                policy_fingertip_marker_visualizer.set_visibility(True)
                self.policy_fingertip_marker_visualizers.append(
                    policy_fingertip_marker_visualizer
                )

    def _debug_vis_callback(self, event: Any) -> None:
        """Visualize the goal marker."""
        del event  # unused

        # Current state visualizers
        self.object_pose_visualizer.visualize(
            translations=self.object.data.body_link_pos_w[
                :, self.object_body_ids
            ].squeeze(1),
            orientations=self.object.data.body_link_quat_w[
                :, self.object_body_ids
            ].squeeze(1),
        )
        self.right_hand_pose_visualizer.visualize(
            translations=self.right_robot.data.root_link_pos_w,
            orientations=self.right_robot.data.root_link_quat_w,
        )
        self.left_hand_pose_visualizer.visualize(
            translations=self.left_robot.data.root_link_pos_w,
            orientations=self.left_robot.data.root_link_quat_w,
        )
        # Command visualizers
        self.object_goal_pose_visualizer.visualize(
            translations=self.object_body_position_command_e
            + self._env.scene.env_origins,
            orientations=self.object_body_wxyz_command_e,
        )
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        self.right_hand_goal_pose_visualizer.visualize(
            translations=right_hand_wrist_pose_command_e[:, :3]
            + self._env.scene.env_origins,
            orientations=right_hand_wrist_pose_command_e[:, 3:],
        )
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        self.left_hand_goal_pose_visualizer.visualize(
            translations=left_hand_wrist_pose_command_e[:, :3]
            + self._env.scene.env_origins,
            orientations=left_hand_wrist_pose_command_e[:, 3:],
        )

        # Visualize target contact location on object
        left_hand_contact_command_positions_e = (
            self.left_hand_object_contact_command_positions_e
        )
        right_hand_contact_command_positions_e = (
            self.right_hand_object_contact_command_positions_e
        )
        for i in range(self.total_num_contacts):
            # Select left or right hand contacts
            if i < self.num_contacts_left:
                # Left contacts
                contact_command_positions_e = left_hand_contact_command_positions_e[
                    :, i
                ]
            else:
                # Right contacts
                contact_command_positions_e = right_hand_contact_command_positions_e[
                    :, i - self.num_contacts_left
                ]

            self.command_contact_marker_visualizers[i].visualize(
                translations=contact_command_positions_e + self._env.scene.env_origins,
                orientations=self.QUAT_UNIT_VEC,
            )

        # Visualize current contacts
        right_hand_contact_positions_w = self.right_hand_object_contact_positions_w
        for i in range(len(self.right_robot.data.body_names)):
            if i < right_hand_contact_positions_w.shape[1]:
                self.policy_contact_marker_visualizers[i].visualize(
                    translations=right_hand_contact_positions_w[:, i],
                    orientations=self.QUAT_UNIT_VEC,
                )

        left_hand_contact_positions_w = self.left_hand_object_contact_positions_w
        for i in range(len(self.right_robot.data.body_names)):
            if i < left_hand_contact_positions_w.shape[1]:
                self.policy_contact_marker_visualizers[
                    i + self.num_contacts_left
                ].visualize(
                    translations=left_hand_contact_positions_w[:, i],
                    orientations=self.QUAT_UNIT_VEC,
                )

        # Visualize fingertip positions
        for i in range(
            len(self.right_fingertip_body_ids) + len(self.left_fingertip_body_ids)
        ):
            if i < len(self.right_fingertip_body_ids):
                self.command_fingertip_marker_visualizers[i].visualize(
                    translations=self.right_hand_fingertip_position_command_e[:, i]
                    + self._env.scene.env_origins,
                    orientations=self.QUAT_UNIT_VEC,
                )
                self.policy_fingertip_marker_visualizers[i].visualize(
                    translations=self.right_hand_fingertip_position_e[:, i]
                    + self._env.scene.env_origins,
                    orientations=self.QUAT_UNIT_VEC,
                )
            else:
                self.command_fingertip_marker_visualizers[i].visualize(
                    translations=self.left_hand_fingertip_position_command_e[
                        :, i - len(self.right_fingertip_body_ids)
                    ]
                    + self._env.scene.env_origins,
                    orientations=self.QUAT_UNIT_VEC,
                )
                self.policy_fingertip_marker_visualizers[i].visualize(
                    translations=self.left_hand_fingertip_position_e[
                        :, i - len(self.right_fingertip_body_ids)
                    ]
                    + self._env.scene.env_origins,
                    orientations=self.QUAT_UNIT_VEC,
                )


class DualHandsTrackingCommand(CommandTerm):
    """Command term that generates pose commands for dual-hand tracking WITHOUT an object.

    This is a tracking-only variant of DualHandsObjectTrackingCommand. It removes all
    object-related functionality (object asset, object reset, object commands/observations,
    contact sensors, contact data, virtual object controller curriculum, object-frame
    transforms). Wrist and fingertip commands are expressed directly in the environment
    frame using the retargeted motion data, rather than relative to an object.
    """

    cfg: CommandTermCfg
    """Configuration for the command term."""

    def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the dual-hand tracking command term.

        Args:
            cfg: Command term configuration (motion path, asset names, FPS, etc.).
            env: The RL environment instance; used to resolve scene assets.
        """
        super().__init__(cfg, env)
        self.step_dt = env.step_dt

        # Retrieve both side robots
        for side in ["right", "left"]:
            side_robot_name = getattr(cfg, f"{side}_robot_name")
            side_robot = env.scene[side_robot_name]
            side_finger_joint_names = side_robot.data.joint_names
            finger_joint_ids, _ = side_robot.find_joints(side_finger_joint_names)
            finger_joint_ids = torch.tensor(finger_joint_ids, device=self.device)
            setattr(self, f"{side}_robot", side_robot)
            setattr(self, f"{side}_finger_joint_names", side_finger_joint_names)
            setattr(self, f"{side}_finger_joint_ids", finger_joint_ids)

            wrist_body_ids, wrist_body_name = side_robot.find_bodies(
                cfg.wrist_body_name
            )
            setattr(
                self,
                f"{side}_wrist_body_id",
                torch.tensor(wrist_body_ids, device=self.device),
            )
            setattr(self, f"{side}_wrist_body_name", wrist_body_name)

            fingertip_body_ids, fingertip_body_names = side_robot.find_bodies(
                cfg.fingertip_body_name
            )
            setattr(
                self,
                f"{side}_fingertip_body_ids",
                torch.tensor(fingertip_body_ids, device=self.device),
            )
            setattr(self, f"{side}_fingertip_body_names", fingertip_body_names)

        # Load motion data
        try:
            self._retargeted_motion_data = ManoSharpaData.from_parquet(
                root_path=str(cfg.motion_folder),
                filters=cfg.motion_filters,
                trajectory_id=cfg.motion_id,
            )
            target_fps = (
                cfg.target_fps if cfg.target_fps is not None else 1 / self.step_dt
            )
            self._retargeted_motion_data = interpolate_robot_motion_data(
                motion_data=self._retargeted_motion_data,
                target_fps=target_fps,
            )
        except Exception as e:
            raise ValueError(
                "Failed to load retargeted motion data from "
                f"{cfg.motion_folder} with filters {cfg.motion_filters} and "
                f"trajectory_id {cfg.motion_id}. Please check if the data exists "
                f"and is valid. Error: {e}"
            ) from e

        # Buffers
        self.retargeted_horizon = len(
            self._retargeted_motion_data.robot_right_wrist_position
        )
        self.timestep_counter = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.tracking_lengths = self.retargeted_horizon * torch.ones(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.steps_since_last_reset = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # Reset wrist pose buffers (written by _resample_command,
        # read by action terms for PD target initialization)
        self.reset_right_wrist_position_e = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.reset_right_wrist_wxyz = torch.zeros(self.num_envs, 4, device=self.device)
        self.reset_left_wrist_position_e = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.reset_left_wrist_wxyz = torch.zeros(self.num_envs, 4, device=self.device)

        # Hand data
        for side in ["right", "left"]:
            # Store wrist position and orientation
            retargeted_wrist_position = getattr(
                self._retargeted_motion_data, f"robot_{side}_wrist_position"
            )
            retargeted_wrist_wxyz = getattr(
                self._retargeted_motion_data, f"robot_{side}_wrist_wxyz"
            )
            setattr(
                self,
                f"retargeted_{side}_wrist_position",
                torch.tensor(retargeted_wrist_position, device=self.device),
            )
            setattr(
                self,
                f"retargeted_{side}_wrist_wxyz",
                torch.tensor(retargeted_wrist_wxyz, device=self.device),
            )

            # Store finger joints in ISAAC joint order
            retargeted_finger_joint_names = getattr(
                self._retargeted_motion_data, f"{side}_robot_finger_joint_names"
            )
            isaac_finger_joint_names = getattr(self, f"{side}_finger_joint_names")
            retargeted_to_isaac_joint_order = [
                retargeted_finger_joint_names.index(joint_name)
                for joint_name in isaac_finger_joint_names
            ]
            retargeted_finger_joints = getattr(
                self._retargeted_motion_data, f"robot_{side}_finger_joints"
            )
            retargeted_finger_joints = torch.tensor(
                retargeted_finger_joints, device=self.device
            )[:, retargeted_to_isaac_joint_order]
            setattr(self, f"retargeted_{side}_finger_joints", retargeted_finger_joints)

            # Store hand frame data
            retargeted_hand_frames = getattr(
                self._retargeted_motion_data, f"robot_{side}_frames"
            )
            retargeted_hand_frame_names = getattr(
                self._retargeted_motion_data, f"{side}_robot_frame_names"
            )
            setattr(
                self,
                f"retargeted_{side}_hand_frames",
                torch.tensor(retargeted_hand_frames, device=self.device),
            )
            setattr(
                self, f"retargeted_{side}_hand_frame_names", retargeted_hand_frame_names
            )

            # Command fingertip index
            fingertip_body_names = getattr(self, f"{side}_fingertip_body_names")
            retargeted_fingertip_indices = []
            for fingertip_body_name in fingertip_body_names:
                fingertip_index = retargeted_hand_frame_names.index(fingertip_body_name)
                retargeted_fingertip_indices.append(fingertip_index)
            setattr(
                self,
                f"retargeted_{side}_fingertip_indices",
                torch.tensor(retargeted_fingertip_indices, device=self.device),
            )

        # Unit vectors
        self.X_UNIT_VEC = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.Y_UNIT_VEC = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.Z_UNIT_VEC = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.QUAT_UNIT_VEC = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat((self.num_envs, 1))

        # Metrics
        self.metrics["right_hand_wrist_position_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["right_hand_wrist_wxyz_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["right_hand_finger_joints_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["left_hand_wrist_position_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["left_hand_wrist_wxyz_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["left_hand_finger_joints_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        """String representation of the command term."""
        msg = f"{self.__class__.__name__}:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    ##########################################################
    # Commands
    ##########################################################

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, -1)."""
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        right_wrist_current_p_command, right_wrist_current_q_command = (
            math_utils.subtract_frame_transforms(
                self.right_hand_wrist_position_e,
                self.right_hand_wrist_wxyz_e,
                right_hand_wrist_pose_command_e[:, :3],
                right_hand_wrist_pose_command_e[:, 3:],
            )
        )
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        left_wrist_current_p_command, left_wrist_current_q_command = (
            math_utils.subtract_frame_transforms(
                self.left_hand_wrist_position_e,
                self.left_hand_wrist_wxyz_e,
                left_hand_wrist_pose_command_e[:, :3],
                left_hand_wrist_pose_command_e[:, 3:],
            )
        )

        right_joint_pos_delta = (
            self.right_hand_finger_joint_pos_command - self.right_robot.data.joint_pos
        )
        left_joint_pos_delta = (
            self.left_hand_finger_joint_pos_command - self.left_robot.data.joint_pos
        )

        return torch.cat(
            (
                right_wrist_current_p_command,
                right_wrist_current_q_command,
                left_wrist_current_p_command,
                left_wrist_current_q_command,
                right_joint_pos_delta,
                left_joint_pos_delta,
            ),
            dim=-1,
        )

    @property
    def right_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """The desired goal position and wxyz in the environment frame for the right hand wrist. Shape is (num_envs, 7)."""
        position = self.retargeted_right_wrist_position[self.timestep_counter].float()
        wxyz = self.retargeted_right_wrist_wxyz[self.timestep_counter].float()
        wxyz = math_utils.quat_unique(wxyz) if self.cfg.make_quat_unique else wxyz
        return torch.cat((position, wxyz), dim=-1)

    @property
    def left_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """The desired goal position and wxyz in the environment frame for the left hand wrist. Shape is (num_envs, 7)."""
        position = self.retargeted_left_wrist_position[self.timestep_counter].float()
        wxyz = self.retargeted_left_wrist_wxyz[self.timestep_counter].float()
        wxyz = math_utils.quat_unique(wxyz) if self.cfg.make_quat_unique else wxyz
        return torch.cat((position, wxyz), dim=-1)

    @property
    def right_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """The desired goal finger joint position for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.retargeted_right_finger_joints[self.timestep_counter].float()

    @property
    def left_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """The desired goal finger joint position for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.retargeted_left_finger_joints[self.timestep_counter].float()

    @property
    def right_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """The desired goal fingertip position in the environment frame for the right hand. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.retargeted_right_hand_frames[self.timestep_counter][
            :, self.retargeted_right_fingertip_indices, :3
        ].float()

    @property
    def left_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """The desired goal fingertip position in the environment frame for the left hand. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.retargeted_left_hand_frames[self.timestep_counter][
            :, self.retargeted_left_fingertip_indices, :3
        ].float()

    ##########################################################
    # Observations
    ##########################################################

    @property
    def right_hand_wrist_position_w(self) -> torch.Tensor:
        """The current position in the world frame for the right hand wrist. Shape is (num_envs, 3)."""
        return self.right_robot.data.root_link_pos_w

    @property
    def left_hand_wrist_position_w(self) -> torch.Tensor:
        """The current position in the world frame for the left hand wrist. Shape is (num_envs, 3)."""
        return self.left_robot.data.root_link_pos_w

    @property
    def right_hand_wrist_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the right hand wrist. Shape is (num_envs, 3)."""
        return (self.right_hand_wrist_position_w - self._env.scene.env_origins).float()

    @property
    def left_hand_wrist_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the left hand wrist. Shape is (num_envs, 3)."""
        return (self.left_hand_wrist_position_w - self._env.scene.env_origins).float()

    @property
    def right_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """The current wxyz in the environment frame for the right hand wrist. Shape is (num_envs, 4)."""
        right_wrist_wxyz = self.right_robot.data.root_link_quat_w.float()
        return (
            math_utils.quat_unique(right_wrist_wxyz)
            if self.cfg.make_quat_unique
            else right_wrist_wxyz
        )

    @property
    def left_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """The current wxyz in the environment frame for the left hand wrist. Shape is (num_envs, 4)."""
        left_wrist_wxyz = self.left_robot.data.root_link_quat_w.float()
        return (
            math_utils.quat_unique(left_wrist_wxyz)
            if self.cfg.make_quat_unique
            else left_wrist_wxyz
        )

    @property
    def right_hand_wrist_velocity_b(self) -> torch.Tensor:
        """The current velocity in the body frame for the right hand wrist. Shape is (num_envs, 6)."""
        return torch.cat(
            [
                self.right_robot.data.root_lin_vel_b,
                self.right_robot.data.root_ang_vel_b,
            ],
            dim=-1,
        ).float()

    @property
    def left_hand_wrist_velocity_b(self) -> torch.Tensor:
        """The current velocity in the body frame for the left hand wrist. Shape is (num_envs, 6)."""
        return torch.cat(
            [
                self.left_robot.data.root_lin_vel_b,
                self.left_robot.data.root_ang_vel_b,
            ],
            dim=-1,
        ).float()

    @property
    def right_hand_finger_joint_pos(self) -> torch.Tensor:
        """The current joint position for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.right_robot.data.joint_pos.float()

    @property
    def left_hand_finger_joint_pos(self) -> torch.Tensor:
        """The current joint position for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.left_robot.data.joint_pos.float()

    @property
    def right_hand_finger_joint_vel(self) -> torch.Tensor:
        """The current joint velocity for the right hand. Shape is (num_envs, NUM_RIGHT_HAND_FINGER_JOINTS)."""
        return self.right_robot.data.joint_vel.float()

    @property
    def left_hand_finger_joint_vel(self) -> torch.Tensor:
        """The current joint velocity for the left hand. Shape is (num_envs, NUM_LEFT_HAND_FINGER_JOINTS)."""
        return self.left_robot.data.joint_vel.float()

    @property
    def right_hand_fingertip_position_w(self) -> torch.Tensor:
        """The current position in the world frame for the right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.right_robot.data.body_link_pos_w[:, self.right_fingertip_body_ids]

    @property
    def left_hand_fingertip_position_w(self) -> torch.Tensor:
        """The current position in the world frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return self.left_robot.data.body_link_pos_w[:, self.left_fingertip_body_ids]

    @property
    def right_hand_fingertip_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return (
            self.right_hand_fingertip_position_w
            - self._env.scene.env_origins.unsqueeze(1)
        ).float()

    @property
    def left_hand_fingertip_position_e(self) -> torch.Tensor:
        """The current position in the environment frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 3)."""
        return (
            self.left_hand_fingertip_position_w
            - self._env.scene.env_origins.unsqueeze(1)
        ).float()

    @property
    def right_hand_fingertip_orientation_e(self) -> torch.Tensor:
        """The current orientation in the environment frame for the right fingertip. Shape is (num_envs, NUM_FINGERTIPS, 4)."""
        return self.right_robot.data.body_link_quat_w[
            :, self.right_fingertip_body_ids
        ].float()

    @property
    def left_hand_fingertip_orientation_e(self) -> torch.Tensor:
        """The current orientation in the environment frame for the left fingertip. Shape is (num_envs, NUM_FINGERTIPS, 4)."""
        return self.left_robot.data.body_link_quat_w[
            :, self.left_fingertip_body_ids
        ].float()

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics."""
        # Right hand
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        self.metrics["right_hand_wrist_position_error"] = torch.norm(
            self.right_hand_wrist_position_e - right_hand_wrist_pose_command_e[:, :3],
            dim=-1,
        )
        self.metrics["right_hand_wrist_wxyz_error"] = math_utils.quat_error_magnitude(
            self.right_hand_wrist_wxyz_e, right_hand_wrist_pose_command_e[:, 3:]
        )
        self.metrics["right_hand_finger_joints_error"] = torch.norm(
            self.right_hand_finger_joint_pos - self.right_hand_finger_joint_pos_command,
            dim=-1,
        )
        # Left hand
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        self.metrics["left_hand_wrist_position_error"] = torch.norm(
            self.left_hand_wrist_position_e - left_hand_wrist_pose_command_e[:, :3],
            dim=-1,
        )
        self.metrics["left_hand_wrist_wxyz_error"] = math_utils.quat_error_magnitude(
            self.left_hand_wrist_wxyz_e, left_hand_wrist_pose_command_e[:, 3:]
        )
        self.metrics["left_hand_finger_joints_error"] = torch.norm(
            self.left_hand_finger_joint_pos - self.left_hand_finger_joint_pos_command,
            dim=-1,
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command."""
        # Reset to a random frame from the original retargeted motion data
        self.timestep_counter[env_ids] = torch.randint(
            low=0,
            high=self.retargeted_horizon - 1,
            size=(len(env_ids),),
            device=self.device,
            dtype=self.timestep_counter.dtype,
        )

        # Get robot wrist position and orientation from the retargeted motion data
        right_hand_wrist_position = self.retargeted_right_wrist_position[
            self.timestep_counter[env_ids]
        ]
        right_hand_wrist_wxyz = self.retargeted_right_wrist_wxyz[
            self.timestep_counter[env_ids]
        ]
        left_hand_wrist_position = self.retargeted_left_wrist_position[
            self.timestep_counter[env_ids]
        ]
        left_hand_wrist_wxyz = self.retargeted_left_wrist_wxyz[
            self.timestep_counter[env_ids]
        ]

        # Store reset wrist poses (env frame, no env_origins) for action terms
        self.reset_right_wrist_position_e[env_ids] = right_hand_wrist_position
        self.reset_right_wrist_wxyz[env_ids] = right_hand_wrist_wxyz
        self.reset_left_wrist_position_e[env_ids] = left_hand_wrist_position
        self.reset_left_wrist_wxyz[env_ids] = left_hand_wrist_wxyz

        # Finger joints: interpolate between open (0) and reference
        finger_factor = (
            torch.rand(len(env_ids), 1, device=self.device)
            * self.cfg.reset_finger_openness
        )
        right_hand_finger_joint_pos = (
            finger_factor
            * self.retargeted_right_finger_joints[self.timestep_counter[env_ids]]
        )
        left_hand_finger_joint_pos = (
            finger_factor
            * self.retargeted_left_finger_joints[self.timestep_counter[env_ids]]
        )

        # Update the tracking length
        self.tracking_lengths[env_ids] = (
            self.retargeted_horizon - self.timestep_counter[env_ids]
        ).clamp(min=1)

        self.steps_since_last_reset[env_ids] = 0

        ##########################################################
        # Reset the robot
        ##########################################################

        # Robot wrist position and orientation
        right_hand_wrist_position = (
            right_hand_wrist_position + self._env.scene.env_origins[env_ids]
        )
        right_hand_wrist_pose = torch.cat(
            [right_hand_wrist_position, right_hand_wrist_wxyz], dim=-1
        ).float()
        right_hand_wrist_velocity = torch.zeros_like(
            right_hand_wrist_pose[..., :6]
        ).float()

        left_hand_wrist_position = (
            left_hand_wrist_position + self._env.scene.env_origins[env_ids]
        )
        left_hand_wrist_pose = torch.cat(
            [left_hand_wrist_position, left_hand_wrist_wxyz], dim=-1
        ).float()
        left_hand_wrist_velocity = torch.zeros_like(
            left_hand_wrist_pose[..., :6]
        ).float()

        self.right_robot.write_root_pose_to_sim(right_hand_wrist_pose, env_ids=env_ids)
        self.right_robot.write_root_velocity_to_sim(
            right_hand_wrist_velocity, env_ids=env_ids
        )
        self.left_robot.write_root_pose_to_sim(left_hand_wrist_pose, env_ids=env_ids)
        self.left_robot.write_root_velocity_to_sim(
            left_hand_wrist_velocity, env_ids=env_ids
        )

        # Clear residual external forces from previous episode
        zero_forces = torch.zeros(len(env_ids), 1, 3, device=self.device)
        self.right_robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=zero_forces,
            body_ids=self.right_wrist_body_id,
            env_ids=env_ids,
            is_global=False,
        )
        self.left_robot.set_external_force_and_torque(
            forces=zero_forces,
            torques=zero_forces,
            body_ids=self.left_wrist_body_id,
            env_ids=env_ids,
            is_global=False,
        )

        # Right robot finger joint positions
        right_hand_joint_velocity = torch.zeros_like(
            right_hand_finger_joint_pos
        ).float()
        right_hand_joint_pos_limits = self.right_robot.data.soft_joint_pos_limits[
            env_ids, :
        ]
        right_hand_finger_joint_pos = right_hand_finger_joint_pos.clamp_(
            right_hand_joint_pos_limits[..., 0], right_hand_joint_pos_limits[..., 1]
        )
        self.right_robot.write_joint_state_to_sim(
            right_hand_finger_joint_pos, right_hand_joint_velocity, env_ids=env_ids
        )

        # Left robot finger joint positions
        left_hand_joint_velocity = torch.zeros_like(left_hand_finger_joint_pos).float()
        left_hand_joint_pos_limits = self.left_robot.data.soft_joint_pos_limits[
            env_ids, :
        ]
        left_hand_finger_joint_pos = left_hand_finger_joint_pos.clamp_(
            left_hand_joint_pos_limits[..., 0], left_hand_joint_pos_limits[..., 1]
        )
        self.left_robot.write_joint_state_to_sim(
            left_hand_finger_joint_pos, left_hand_joint_velocity, env_ids=env_ids
        )

        # Force a kinematic/data refresh after reset writes so the first
        # post-reset control step reads synchronized wrist states.
        self._env.sim.forward()
        self._env.scene.update(dt=self._env.physics_dt)

    def _update_command(self) -> None:
        """Update the command."""
        self.steps_since_last_reset += 1
        self.timestep_counter += 1

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set the debug visibility."""
        if debug_vis:
            if not hasattr(self, "right_hand_pose_visualizer"):
                self.right_hand_pose_visualizer = VisualizationMarkers(
                    self.cfg.right_hand_pose_visualizer_cfg
                )
            self.right_hand_pose_visualizer.set_visibility(True)
            if not hasattr(self, "left_hand_pose_visualizer"):
                self.left_hand_pose_visualizer = VisualizationMarkers(
                    self.cfg.left_hand_pose_visualizer_cfg
                )
            self.left_hand_pose_visualizer.set_visibility(True)
            if not hasattr(self, "right_hand_goal_pose_visualizer"):
                self.right_hand_goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.right_hand_goal_pose_visualizer_cfg
                )
            self.right_hand_goal_pose_visualizer.set_visibility(True)
            if not hasattr(self, "left_hand_goal_pose_visualizer"):
                self.left_hand_goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.left_hand_goal_pose_visualizer_cfg
                )
            self.left_hand_goal_pose_visualizer.set_visibility(True)

            # Per-body-frame markers (deferred: hand frame data may not
            # exist yet when super().__init__ calls set_debug_vis)
            if hasattr(self, "retargeted_right_hand_frames"):
                num_target = (
                    self.retargeted_right_hand_frames.shape[1]
                    + self.retargeted_left_hand_frames.shape[1]
                )
                num_current = len(self.right_robot.data.body_names) + len(
                    self.left_robot.data.body_names
                )
                num_markers = max(num_target, num_current)
                if not hasattr(self, "target_frame_visualizers"):
                    self.target_frame_visualizers = []
                    self.current_frame_visualizers = []
                    for i in range(num_markers):
                        target_cfg = self.cfg.target_fingertip_visualizer_cfg.replace(
                            prim_path=f"/Visuals/Command/TargetFrame_{i}"
                        )
                        self.target_frame_visualizers.append(
                            VisualizationMarkers(target_cfg)
                        )
                        current_cfg = self.cfg.current_fingertip_visualizer_cfg.replace(
                            prim_path=f"/Visuals/Command/CurrentFrame_{i}"
                        )
                        self.current_frame_visualizers.append(
                            VisualizationMarkers(current_cfg)
                        )
                    self._num_target_frames = num_target
                    self._num_current_frames = num_current
                for vis in self.target_frame_visualizers:
                    vis.set_visibility(True)
                for vis in self.current_frame_visualizers:
                    vis.set_visibility(True)
        else:
            for attr in [
                "right_hand_pose_visualizer",
                "left_hand_pose_visualizer",
                "right_hand_goal_pose_visualizer",
                "left_hand_goal_pose_visualizer",
            ]:
                if hasattr(self, attr):
                    getattr(self, attr).set_visibility(False)
            for vis in getattr(self, "target_frame_visualizers", []):
                vis.set_visibility(False)
            for vis in getattr(self, "current_frame_visualizers", []):
                vis.set_visibility(False)

    def _debug_vis_callback(self, event: Any) -> None:
        """Visualize the goal marker."""
        del event  # unused

        # Current state visualizers
        self.right_hand_pose_visualizer.visualize(
            translations=self.right_robot.data.root_link_pos_w,
            orientations=self.right_robot.data.root_link_quat_w,
        )
        self.left_hand_pose_visualizer.visualize(
            translations=self.left_robot.data.root_link_pos_w,
            orientations=self.left_robot.data.root_link_quat_w,
        )
        # Command visualizers
        right_hand_wrist_pose_command_e = self.right_hand_wrist_pose_command_e
        self.right_hand_goal_pose_visualizer.visualize(
            translations=right_hand_wrist_pose_command_e[:, :3]
            + self._env.scene.env_origins,
            orientations=right_hand_wrist_pose_command_e[:, 3:],
        )
        left_hand_wrist_pose_command_e = self.left_hand_wrist_pose_command_e
        self.left_hand_goal_pose_visualizer.visualize(
            translations=left_hand_wrist_pose_command_e[:, :3]
            + self._env.scene.env_origins,
            orientations=left_hand_wrist_pose_command_e[:, 3:],
        )

        # Hand body frame visualizers
        if not hasattr(self, "target_frame_visualizers"):
            # Deferred creation: first callback after __init__ completes
            self._set_debug_vis_impl(True)
        if hasattr(self, "target_frame_visualizers"):
            env_origins = self._env.scene.env_origins

            # Target: retargeted hand frame positions (env frame → world)
            num_right_target = self.retargeted_right_hand_frames.shape[1]
            right_target = self.retargeted_right_hand_frames[self.timestep_counter][
                :, :, :3
            ].float()
            left_target = self.retargeted_left_hand_frames[self.timestep_counter][
                :, :, :3
            ].float()
            for i in range(self._num_target_frames):
                if i < num_right_target:
                    pos = right_target[:, i] + env_origins
                else:
                    pos = left_target[:, i - num_right_target] + env_origins
                self.target_frame_visualizers[i].visualize(
                    translations=pos,
                    orientations=self.QUAT_UNIT_VEC,
                )

            # Current: all sim body link positions (already world frame)
            num_right_bodies = len(self.right_robot.data.body_names)
            right_body_pos_w = self.right_robot.data.body_link_pos_w
            left_body_pos_w = self.left_robot.data.body_link_pos_w
            for i in range(self._num_current_frames):
                if i < num_right_bodies:
                    pos = right_body_pos_w[:, i]
                else:
                    pos = left_body_pos_w[:, i - num_right_bodies]
                self.current_frame_visualizers[i].visualize(
                    translations=pos,
                    orientations=self.QUAT_UNIT_VEC,
                )
