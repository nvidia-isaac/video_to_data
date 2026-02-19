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

        # Load motion data
        try:
            self._retargeted_motion_data = ManoSharpaData.from_parquet(
                root_path=str(cfg.motion_folder),
                filters=cfg.motion_filters,
                trajectory_id=cfg.motion_id,
            )
            # Interpolate the motion data to the target FPS
            target_fps = (
                cfg.target_fps if cfg.target_fps is not None else 1 / self._env.step_dt
            )
            self._retargeted_motion_data = interpolate_robot_motion_data(
                motion_data=self._retargeted_motion_data,
                target_fps=target_fps,
            )

            # Load reset states
            reset_data_exists = os.path.exists(cfg.reset_state_file_path)
            reset_to_any_state = reset_data_exists and not cfg.reset_to_initial
            self.reset_to_initial = not reset_to_any_state
            if not self.reset_to_initial:
                reset_states = torch.load(cfg.reset_state_file_path)
                self.reset_object_position = torch.cat(
                    [
                        reset_states[step]["object_position"].unsqueeze(0)
                        for step in range(1, len(reset_states))
                    ],
                    dim=0,
                ).to(self.device)
                self.reset_object_wxyz = torch.cat(
                    [
                        reset_states[step]["object_wxyz"].unsqueeze(0)
                        for step in range(1, len(reset_states))
                    ],
                    dim=0,
                ).to(self.device)
                self.reset_right_hand_wrist_position = torch.cat(
                    [
                        reset_states[step]["right_hand_wrist_position"].unsqueeze(0)
                        for step in range(1, len(reset_states))
                    ],
                    dim=0,
                ).to(self.device)
                self.reset_right_hand_wrist_wxyz = torch.cat(
                    [
                        reset_states[step]["right_hand_wrist_wxyz"].unsqueeze(0)
                        for step in range(1, len(reset_states))
                    ],
                    dim=0,
                ).to(self.device)
                self.reset_left_hand_wrist_position = torch.cat(
                    [
                        reset_states[step]["left_hand_wrist_position"].unsqueeze(0)
                        for step in range(1, len(reset_states))
                    ],
                    dim=0,
                ).to(self.device)
                self.reset_left_hand_wrist_wxyz = torch.cat(
                    [
                        reset_states[step]["left_hand_wrist_wxyz"].unsqueeze(0)
                        for step in range(1, len(reset_states))
                    ],
                    dim=0,
                ).to(self.device)
                self.reset_right_hand_finger_joint_pos = torch.cat(
                    [
                        reset_states[step][
                            "right_hand_finger_joint_position"
                        ].unsqueeze(0)
                        for step in range(1, len(reset_states))
                    ],
                    dim=0,
                ).to(self.device)
                self.reset_left_hand_finger_joint_pos = torch.cat(
                    [
                        reset_states[step]["left_hand_finger_joint_position"].unsqueeze(
                            0
                        )
                        for step in range(1, len(reset_states))
                    ],
                    dim=0,
                ).to(self.device)
                num_reset_states = len(self.reset_object_position[0])
                assert num_reset_states == len(self.reset_object_wxyz[0])
                assert num_reset_states == len(self.reset_right_hand_wrist_position[0])
                assert num_reset_states == len(self.reset_right_hand_wrist_wxyz[0])
                assert num_reset_states == len(self.reset_left_hand_wrist_position[0])
                assert num_reset_states == len(self.reset_left_hand_wrist_wxyz[0])
                assert num_reset_states == len(
                    self.reset_right_hand_finger_joint_pos[0]
                )
                assert num_reset_states == len(self.reset_left_hand_finger_joint_pos[0])

        except Exception as e:
            raise ValueError(
                "Failed to load retargeted motion data from "
                f"{cfg.motion_folder} with filters {cfg.motion_filters} and "
                f"trajectory_id {cfg.motion_id}. Please check if the data exists "
                f"and is valid. Error: {e}"
            ) from e

        self.retargeted_horizon = len(
            self._retargeted_motion_data.robot_right_wrist_position
        )

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
        # FIXME: add support for multiple object bodies and multiple objects
        # FIXME: add velocity computation
        self.retargeted_object_body_position = torch.tensor(
            self._retargeted_motion_data.object_body_position, device=self.device
        )
        self.retargeted_object_body_wxyz = torch.tensor(
            self._retargeted_motion_data.object_body_wxyz, device=self.device
        )
        self.retargeted_object_articulation = torch.tensor(
            self._retargeted_motion_data.object_articulation, device=self.device
        )

        # Buffers
        self.timestep_counter = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        self.X_UNIT_VEC = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.Y_UNIT_VEC = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.Z_UNIT_VEC = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        # Precompute unit vectors and quaternions for reward computation
        self.QUAT_UNIT_VEC = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        ).repeat(
            (self.num_envs, 6, 1)
        )  # (num_envs, 6, 4)
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
        return torch.cat(
            (
                self.right_hand_wrist_position_command_e,
                self.right_hand_wrist_wxyz_command_e,
                self.left_hand_wrist_position_command_e,
                self.left_hand_wrist_wxyz_command_e,
                self.right_hand_finger_joint_pos_command,
                self.left_hand_finger_joint_pos_command,
                self.object_body_position_command_e,
                self.object_body_wxyz_command_e,
            ),
            dim=-1,
        )

    @property
    def right_hand_wrist_position_command_e(self) -> torch.Tensor:
        """The desired goal position in the environment frame for the right hand wrist. Shape is (num_envs, 3)."""
        return self.retargeted_right_wrist_position[self.timestep_counter].float()

    @property
    def right_hand_wrist_wxyz_command_e(self) -> torch.Tensor:
        """The desired goal wxyz in the environment frame for the right hand wrist. Shape is (num_envs, 4)."""
        right_wrist_wxyz = self.retargeted_right_wrist_wxyz[
            self.timestep_counter
        ].float()
        return (
            math_utils.quat_unique(right_wrist_wxyz)
            if self.cfg.make_quat_unique
            else right_wrist_wxyz
        )

    @property
    def left_hand_wrist_position_command_e(self) -> torch.Tensor:
        """The desired goal position in the environment frame for the left hand wrist. Shape is (num_envs, 3)."""
        return self.retargeted_left_wrist_position[self.timestep_counter]

    @property
    def left_hand_wrist_wxyz_command_e(self) -> torch.Tensor:
        """The desired goal wxyz in the environment frame for the left hand wrist. Shape is (num_envs, 4)."""
        left_wrist_wxyz = self.retargeted_left_wrist_wxyz[self.timestep_counter].float()
        return (
            math_utils.quat_unique(left_wrist_wxyz)
            if self.cfg.make_quat_unique
            else left_wrist_wxyz
        )

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
        """The desired goal fingertip position in the environment frame for the right hand. Shape is (num_envs, NUM_FINGERTIPS, 7)."""
        return self.retargeted_right_hand_frames[self.timestep_counter][
            :, self.retargeted_right_fingertip_indices
        ].float()

    @property
    def left_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """The desired goal fingertip position in the environment frame for the left hand. Shape is (num_envs, NUM_FINGERTIPS, 7)."""
        return self.retargeted_left_hand_frames[self.timestep_counter][
            :, self.retargeted_left_fingertip_indices
        ].float()

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

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        """Update the metrics."""
        # Right hand
        self.metrics["right_hand_wrist_position_error"] = torch.norm(
            self.right_hand_wrist_position_e - self.right_hand_wrist_position_command_e,
            dim=-1,
        )
        self.metrics["right_hand_wrist_wxyz_error"] = math_utils.quat_error_magnitude(
            self.right_hand_wrist_wxyz_e, self.right_hand_wrist_wxyz_command_e
        )
        self.metrics["right_hand_finger_joints_error"] = torch.norm(
            self.right_hand_finger_joint_pos - self.right_hand_finger_joint_pos_command,
            dim=-1,
        )
        # Left hand
        self.metrics["left_hand_wrist_position_error"] = torch.norm(
            self.left_hand_wrist_position_e - self.left_hand_wrist_position_command_e,
            dim=-1,
        )
        self.metrics["left_hand_wrist_wxyz_error"] = math_utils.quat_error_magnitude(
            self.left_hand_wrist_wxyz_e, self.left_hand_wrist_wxyz_command_e
        )
        self.metrics["left_hand_finger_joints_error"] = torch.norm(
            self.left_hand_finger_joint_pos - self.left_hand_finger_joint_pos_command,
            dim=-1,
        )
        # Object
        self.metrics["object_body_position_error"] = torch.norm(
            self.object_position_e.squeeze() - self.object_body_position_command_e,
            dim=-1,
        )
        self.metrics["object_body_wxyz_error"] = math_utils.quat_error_magnitude(
            self.object_orientation_e.squeeze(),
            self.object_body_wxyz_command_e.squeeze(),
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample the command."""
        ##########################################################
        # Resample the command
        ##########################################################

        if self.reset_to_initial:
            self.timestep_counter[env_ids] *= 0
            # Get object position and orientation from the retargeted motion data
            object_position_e = (
                self.retargeted_object_body_position[self.timestep_counter[env_ids]][
                    :, self.object_body_ids
                ]
                .squeeze(1)
                .clone()
            )
            object_wxyz = (
                self.retargeted_object_body_wxyz[self.timestep_counter[env_ids]][
                    :, self.object_body_ids
                ]
                .squeeze(1)
                .clone()
            )
            # Get robot wrist position and orientation from the retargeted motion data
            right_hand_wrist_position = self.retargeted_right_wrist_position[
                self.timestep_counter[env_ids]
            ].clone()
            right_hand_wrist_wxyz = self.retargeted_right_wrist_wxyz[
                self.timestep_counter[env_ids]
            ].clone()
            left_hand_wrist_position = self.retargeted_left_wrist_position[
                self.timestep_counter[env_ids]
            ].clone()
            left_hand_wrist_wxyz = self.retargeted_left_wrist_wxyz[
                self.timestep_counter[env_ids]
            ].clone()
            # Get robot finger joint positions from the retargeted motion data
            right_hand_finger_joint_pos = self.retargeted_right_finger_joints[
                self.timestep_counter[env_ids]
            ].clone()
            left_hand_finger_joint_pos = self.retargeted_left_finger_joints[
                self.timestep_counter[env_ids]
            ].clone()

        else:
            self.timestep_counter[env_ids] = torch.randint(
                low=0,
                high=self.retargeted_horizon - 1,
                size=(len(env_ids),),
                device=self.device,
                dtype=self.timestep_counter.dtype,
            )
            timestep = (
                len(self.reset_object_position)
                * self.timestep_counter[env_ids]
                / self.retargeted_horizon
            ).to(torch.int)
            reset_state_idx = torch.randint(
                low=0,
                high=len(self.reset_object_position[0]) - 1,
                size=(len(env_ids),),
                device=self.device,
                dtype=self.timestep_counter.dtype,
            )

            # Get object position and orientation from the reset state
            object_position_e = self.reset_object_position[
                timestep, reset_state_idx
            ].clone()
            object_wxyz = self.reset_object_wxyz[timestep, reset_state_idx].clone()

            # Get robot wrist position and orientation from the reset state
            right_hand_wrist_position = self.reset_right_hand_wrist_position[
                timestep, reset_state_idx
            ].clone()
            right_hand_wrist_wxyz = self.reset_right_hand_wrist_wxyz[
                timestep, reset_state_idx
            ].clone()
            left_hand_wrist_position = self.reset_left_hand_wrist_position[
                timestep, reset_state_idx
            ].clone()
            left_hand_wrist_wxyz = self.reset_left_hand_wrist_wxyz[
                timestep, reset_state_idx
            ].clone()

            # Get robot finger joint positions from the reset state
            right_hand_finger_joint_pos = self.reset_right_hand_finger_joint_pos[
                timestep, reset_state_idx
            ].clone()
            left_hand_finger_joint_pos = self.reset_left_hand_finger_joint_pos[
                timestep, reset_state_idx
            ].clone()

        ##########################################################
        # Reset the object joints
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

    def _update_command(self) -> None:
        """Update the command."""
        self.timestep_counter += 1

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
        self.right_hand_goal_pose_visualizer.visualize(
            translations=self.right_hand_wrist_position_command_e
            + self._env.scene.env_origins,
            orientations=self.right_hand_wrist_wxyz_command_e,
        )
        self.left_hand_goal_pose_visualizer.visualize(
            translations=self.left_hand_wrist_position_command_e
            + self._env.scene.env_origins,
            orientations=self.left_hand_wrist_wxyz_command_e,
        )
