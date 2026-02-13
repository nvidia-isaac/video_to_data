from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    quat_unique,
)

from .tracking_utils import load_motion_data

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .tracking_command_cfg import TrackingCommandCfg


class TrackingCommand(CommandTerm):
    """Load reference motion data and provide observations for tracking."""

    cfg: TrackingCommandCfg

    def __init__(self, cfg: TrackingCommandCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize tracking command from config and environment."""
        super().__init__(cfg, env)

        # Robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Object
        self.object: RigidObject = env.scene[cfg.object_name]

        # Cache env origins for world frame transforms
        self._env_origins = env.scene.env_origins

        # Find anchor body index (pelvis)
        self._anchor_body_ids, _ = self.robot.find_bodies([cfg.anchor_body_name])
        if len(self._anchor_body_ids) == 0:
            raise ValueError(
                f"Anchor body '{cfg.anchor_body_name}' not found in robot."
            )
        self._anchor_body_id = self._anchor_body_ids[0]

        # Load motion data from HDF5, YAML, or Parquet
        motion_data = load_motion_data(cfg, self.robot, self.device)
        qpos_data = motion_data.qpos_data
        object_pos_w = motion_data.object_pos_w
        object_quat_w = motion_data.object_quat_w

        # Store EE data if available
        if motion_data.ee_pos_w is not None:
            self.ee_link_names = motion_data.ee_link_names
            self.ee_link_ids = motion_data.ee_link_ids
            self.ee_pos_w = motion_data.ee_pos_w
            self.ee_quat_w = motion_data.ee_quat_w

        # Extract components from qpos: [pos(3), quat(4), joints(N)]
        self.root_pos_w = qpos_data[:, :3].float() + torch.tensor(
            cfg.robot_anchor_pos_offset, device=self.device
        )  # (T, 3)
        self.root_quat_w = qpos_data[:, 3:7].float()  # (T, 4)
        self._joint_pos_file = qpos_data[:, 7:].float()  # (T, N)
        self._object_pos_w = object_pos_w.float() + torch.tensor(
            cfg.object_pos_offset, device=self.device
        )  # (T, 3)
        self._object_quat_w = object_quat_w.float()  # (T, 4)
        self.num_timesteps = len(qpos_data)

        # Precompute peak object height timestep (for phase detection in rewards)
        object_heights = self._object_pos_w[:, 2]  # Z-coordinate
        self.object_height_peak_timestep = torch.argmax(object_heights).item()

        # Precompute joint velocities via finite differences
        self._joint_vel_file = torch.zeros_like(self._joint_pos_file)
        self._joint_vel_file[:-1] = (
            self._joint_pos_file[1:] - self._joint_pos_file[:-1]
        ) / cfg.dt

        # Future frame configuration
        self.num_future_frames = cfg.num_future_frames
        self.frame_step = int(cfg.dt_future_frames / cfg.dt)

        # Precompute future frame offsets
        self._future_frame_offsets = torch.arange(
            0,
            self.num_future_frames * self.frame_step,
            self.frame_step,
            dtype=torch.int32,
            device=self.device,
        )

        # Tracked joint indices and ordering conversion
        self._tracked_joint_ids, self._tracked_joint_names = self.robot.find_joints(
            cfg.joint_names
        )
        self.file_joint_names = cfg.file_joint_names
        self.is_ee_motion = cfg.is_ee_motion

        # For EE-based motion (floating-base hands), joint mapping is handled during loading
        # TODO: make this cleaner
        if self.file_joint_names is not None and not self.is_ee_motion:
            # Reorder from file ordering to IsaacLab ordering (for full-body motion files)
            file_to_isaac = [
                self.file_joint_names.index(joint_name)
                for joint_name in self._tracked_joint_names
            ]
            self.joint_pos = self._joint_pos_file[:, file_to_isaac]
            self.joint_vel = self._joint_vel_file[:, file_to_isaac]
        else:
            self.joint_pos = self._joint_pos_file
            self.joint_vel = self._joint_vel_file

        # Current timestep counter
        self.timestep = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # Track where each env started from (for normalized progress calculation)
        self.reset_timestep = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # Set encoder mode to G1 (TODO: make this dynamic)
        self._encoder_mode = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def future_timesteps(self) -> torch.Tensor:
        """Future timesteps. Shape: (num_envs, num_future_frames)."""
        return torch.clamp(
            self.timestep[:, None] + self._future_frame_offsets[None, :],
            0,
            self.num_timesteps - 1,
        )

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Robot anchor (pelvis) position in world frame. Shape: (num_envs, 3)."""
        return self.robot.data.body_pos_w[:, self._anchor_body_id]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Robot anchor (pelvis) quaternion in world frame. Shape: (num_envs, 4)."""
        return self.robot.data.body_quat_w[:, self._anchor_body_id]

    @property
    def robot_ee_pos_w(self) -> torch.Tensor:
        """Robot EE position in world frame. Shape: (num_envs, num_ee_links, 3)."""
        return self.robot.data.body_pos_w[:, self.ee_link_ids]

    @property
    def robot_ee_quat_w(self) -> torch.Tensor:
        """Robot EE quaternion in world frame. Shape: (num_envs, num_ee_links, 4)."""
        return self.robot.data.body_quat_w[:, self.ee_link_ids]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """Robot joint positions. Shape: (num_envs, num_joints)."""
        return self.robot.data.joint_pos[:, self._tracked_joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """Robot joint velocities. Shape: (num_envs, num_joints)."""
        return self.robot.data.joint_vel[:, self._tracked_joint_ids]

    @property
    def object_pos_w(self) -> torch.Tensor:
        """Object position in world frame. Shape: (num_envs, 3)."""
        return self.object.data.root_pos_w

    @property
    def object_quat_w(self) -> torch.Tensor:
        """Object quaternion in world frame. Shape: (num_envs, 4)."""
        return self.object.data.root_quat_w

    @property
    def encoder_mode(self) -> torch.Tensor:
        """
        Encoder mode selection. Shape: (num_envs, 3).

        One-hot vector for [G1, teleop, SMPL] modes. Only G1 is active for this command.
        """
        return self._encoder_mode

    @property
    def command_object_pos_w(self) -> torch.Tensor:
        """Command object position in world frame. Shape: (num_envs, 3)."""
        return self._object_pos_w[self.timestep] + self._env_origins

    @property
    def command_object_pos_multi_future(self) -> torch.Tensor:
        """Future object positions. Shape: (num_envs, num_future_frames, 3)."""
        return self._object_pos_w[self.future_timesteps]

    @property
    def command_object_quat_w(self) -> torch.Tensor:
        """Command object quaternion in world frame. Shape: (num_envs, 4)."""
        return quat_unique(self._object_quat_w[self.timestep])

    @property
    def command_anchor_pos_w(self) -> torch.Tensor:
        """Command anchor position in world frame. Shape: (num_envs, 3)."""
        return self.root_pos_w[self.timestep] + self._env_origins

    @property
    def command_ee_pos_w(self) -> torch.Tensor:
        """Command EE position in world frame. Shape: (num_envs, num_ee_links, 3)."""
        return self.ee_pos_w[self.timestep] + self._env_origins.unsqueeze(1)

    @property
    def command_ee_pos_multi_future(self) -> torch.Tensor:
        """Future EE positions. Shape: (num_envs, num_future_frames, num_ee_links, 3)."""
        return self.ee_pos_w[self.future_timesteps] + self._env_origins.unsqueeze(
            1
        ).unsqueeze(1)

    @property
    def command_ee_quat_w(self) -> torch.Tensor:
        """Command EE quaternion in world frame. Shape: (num_envs, num_ee_links, 4)."""
        return quat_unique(self.ee_quat_w[self.timestep])  # type: ignore[index]

    @property
    def command_ee_quat_multi_future(self) -> torch.Tensor:
        """Future EE quaternions. Shape: (num_envs, num_future_frames, num_ee_links, 4)."""
        return quat_unique(self.ee_quat_w[self.future_timesteps])  # type: ignore[index]

    @property
    def command_anchor_pos_multi_future(self) -> torch.Tensor:
        """Future anchor positions. Shape: (num_envs, num_future_frames, 3)."""
        return self.root_pos_w[self.future_timesteps] + self._env_origins.unsqueeze(1)

    @property
    def command_anchor_quat_w(self) -> torch.Tensor:
        """Command anchor quaternion in world frame. Shape: (num_envs, 4)."""
        return quat_unique(self.root_quat_w[self.timestep])

    @property
    def command_multi_future(self) -> torch.Tensor:
        """Future joint positions and velocities concatenated. Shape: (num_envs, num_future_frames, num_joints * 2)."""
        return torch.cat(
            [
                self.joint_pos[self.future_timesteps],
                self.joint_vel[self.future_timesteps],
            ],
            dim=-1,
        )

    @property
    def command_joint_pos(self) -> torch.Tensor:
        """Command joint positions. Shape: (num_envs, num_joints)."""
        return self.joint_pos[self.timestep]

    @property
    def command_joint_pos_multi_future(self) -> torch.Tensor:
        """Future joint positions. Shape: (num_envs, num_future_frames, num_joints)."""
        return self.joint_pos[self.future_timesteps]

    @property
    def command_joint_vel_multi_future(self) -> torch.Tensor:
        """Future joint velocities. Shape: (num_envs, num_future_frames, num_joints)."""
        return self.joint_vel[self.future_timesteps]

    @property
    def command_z_multi_future(self) -> torch.Tensor:
        """Future z (height) positions. Shape: (num_envs, num_future_frames)."""
        future_root_pos = self.root_pos_w[
            self.future_timesteps
        ]  # (num_envs, num_future_frames, 3)
        return future_root_pos[..., 2]  # Extract z coordinate

    @property
    def command_root_rot_dif_l_multi_future(self) -> torch.Tensor:
        """Future orientation differences from robot as 6D rotation. Shape: (num_envs, num_future_frames, 6).

        This computes the rotation difference between reference motion and current robot orientation.
        """
        ref_root_quat = self.root_quat_w[
            self.future_timesteps
        ]  # (num_envs, num_future_frames, 4)

        # Compute rotation difference: quat_inv(robot_quat) * ref_quat
        robot_quat_inv = (
            quat_inv(self.robot_anchor_quat_w)
            .view(self.num_envs, 1, 4)
            .repeat(1, self.num_future_frames, 1)
        )
        root_rot_dif = quat_mul(robot_quat_inv, ref_root_quat)

        # Convert to 6D representation
        mat = matrix_from_quat(root_rot_dif)
        root_rot_dif_l = mat[..., :2]

        return root_rot_dif_l.reshape(self.num_envs, self.num_future_frames, 6)

    @property
    def command(self) -> torch.Tensor:
        """The command tensor (required by CommandTerm base class).

        Returns the multi-future command for compatibility.
        """
        return self.command_multi_future

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Reset timestep for specified environments.

        For now, just reset to beginning.
        """
        del env_ids  # no resampling

    def _update_command(self) -> None:
        """Update timestep counter each step."""
        self.timestep += 1
        self.timestep = torch.clamp(self.timestep, 0, self.num_timesteps - 1)

    def _update_metrics(self) -> None:
        """Update tracking metrics."""
        if "root_pos_error" not in self.metrics:
            self.metrics["root_pos_error"] = torch.zeros(
                self.num_envs, self.num_timesteps, device=self.device
            )
            self.metrics["root_quat_error"] = torch.zeros(
                self.num_envs, self.num_timesteps, device=self.device
            )
            self.metrics["joint_pos_error"] = torch.zeros(
                self.num_envs, self.num_timesteps, device=self.device
            )
            self.metrics["joint_vel_error"] = torch.zeros(
                self.num_envs, self.num_timesteps, device=self.device
            )
            self.metrics["object_pos_error"] = torch.zeros(
                self.num_envs, self.num_timesteps, device=self.device
            )
        if self.cfg.ee_link_names and "ee_pos_error" not in self.metrics:
            self.metrics["ee_pos_error"] = torch.zeros(
                self.num_envs, self.num_timesteps, device=self.device
            )
            self.metrics["ee_quat_error"] = torch.zeros(
                self.num_envs, self.num_timesteps, device=self.device
            )

        self.metrics["root_pos_error"][:, self.timestep] = torch.norm(
            self.robot_anchor_pos_w - self.command_anchor_pos_w, dim=-1
        )
        self.metrics["root_quat_error"][:, self.timestep] = (
            quat_error_magnitude(self.robot_anchor_quat_w, self.command_anchor_quat_w)
            ** 2
        )
        self.metrics["joint_pos_error"][:, self.timestep] = torch.norm(
            self.robot_joint_pos - self.command_joint_pos_multi_future[:, 0, :], dim=-1
        )
        self.metrics["joint_vel_error"][:, self.timestep] = torch.norm(
            self.robot_joint_vel - self.command_joint_vel_multi_future[:, 0, :], dim=-1
        )
        self.metrics["object_pos_error"][:, self.timestep] = torch.norm(
            self.object_pos_w - self.command_object_pos_w, dim=-1
        )
        if self.cfg.ee_link_names:
            ee_pos_error = torch.norm(
                self.robot_ee_pos_w.reshape(-1, 3)
                - self.command_ee_pos_w.reshape(-1, 3),
                dim=-1,
            )
            ee_pos_error = torch.sum(
                ee_pos_error.reshape(self.num_envs, len(self.cfg.ee_link_names)), dim=-1
            )
            ee_quat_error = (
                quat_error_magnitude(
                    self.robot_ee_quat_w.reshape(-1, 4),
                    self.command_ee_quat_w.reshape(-1, 4),
                )
                ** 2
            )
            ee_quat_error = torch.sum(
                ee_quat_error.reshape(self.num_envs, len(self.cfg.ee_link_names)),
                dim=-1,
            )
            self.metrics["ee_pos_error"][:, self.timestep] = ee_pos_error
            self.metrics["ee_quat_error"][:, self.timestep] = ee_quat_error

    def _set_debug_vis_impl(self, debug_vis: bool = True) -> None:
        """Enable/disable debug visualization of tracking target."""
        if debug_vis:
            # Create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                goal_pose_visualizer_cfg = self.cfg.pose_visualizer_cfg.replace(
                    prim_path="/Visuals/Command/goal_marker"
                )
                self.goal_pose_visualizer = VisualizationMarkers(
                    goal_pose_visualizer_cfg
                )
            if not hasattr(self, "object_pose_visualizer"):
                object_pose_visualizer_cfg = self.cfg.pose_visualizer_cfg.replace(
                    prim_path="/Visuals/Command/object_marker"
                )
                self.object_pose_visualizer = VisualizationMarkers(
                    object_pose_visualizer_cfg
                )
            if not hasattr(self, "ee_pose_visualizer"):
                self.ee_pose_visualizer = {}
                for ee_name in self.cfg.ee_link_names:
                    ee_pose_visualizer_cfg = self.cfg.pose_visualizer_cfg.replace(
                        prim_path=f"/Visuals/Command/ee_marker_{ee_name}"
                    )
                    ee_pose_visualizer_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)
                    self.ee_pose_visualizer[ee_name] = VisualizationMarkers(
                        ee_pose_visualizer_cfg
                    )
            self.goal_pose_visualizer.set_visibility(True)
            self.object_pose_visualizer.set_visibility(True)
            for ee_name in self.cfg.ee_link_names:
                self.ee_pose_visualizer[ee_name].set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)
            self.object_pose_visualizer.set_visibility(False)
            for ee_name in self.cfg.ee_link_names:
                self.ee_pose_visualizer[ee_name].set_visibility(False)

    def _debug_vis_callback(self, event: Any) -> None:
        """Visualize the current tracking target."""
        if hasattr(self, "goal_pose_visualizer") and hasattr(
            self, "object_pose_visualizer"
        ):
            pos = self.command_anchor_pos_w
            quat = self.command_anchor_quat_w
            obj_pos = self.command_object_pos_w
            obj_quat = self.command_object_quat_w
            self.goal_pose_visualizer.visualize(translations=pos, orientations=quat)
            self.object_pose_visualizer.visualize(
                translations=obj_pos, orientations=obj_quat
            )
            for i, ee_name in enumerate(self.cfg.ee_link_names):
                ee_pos = self.command_ee_pos_w[:, i]
                ee_quat = self.command_ee_quat_w[:, i]
                self.ee_pose_visualizer[ee_name].visualize(
                    translations=ee_pos, orientations=ee_quat
                )
