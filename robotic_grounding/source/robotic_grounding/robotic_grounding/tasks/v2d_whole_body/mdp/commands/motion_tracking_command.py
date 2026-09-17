# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-only whole-body tracking over a bank of motion sequences."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

from .tracking_utils import load_motion_data

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .tracking_command_cfg import MotionTrackingCommandCfg


class MotionTrackingCommand(CommandTerm):
    """Track robot references without object, contact, VOC, or wrench state.

    Motions are padded into a dense bank at initialization. Each environment
    owns a ``motion_id`` and a local ``timestep``; command tensors are gathered
    with both indices so trajectories of different lengths can run together.
    """

    cfg: MotionTrackingCommandCfg

    def __init__(self, cfg: MotionTrackingCommandCfg, env: ManagerBasedRLEnv) -> None:
        """Initialize the motion bank and per-environment tracking state."""
        super().__init__(cfg, env)
        self._init_scene_references(cfg, env)
        self._load_and_process_motion(cfg)
        self._init_buffers(cfg)
        self._init_metrics()
        self._init_hand_data(cfg)

        if cfg.debug_vis:
            self.set_debug_vis(True)

    def _init_scene_references(
        self, cfg: MotionTrackingCommandCfg, env: ManagerBasedRLEnv
    ) -> None:
        """Resolve only robot bodies needed for reference tracking."""
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._env_origins = env.scene.env_origins
        anchor_ids, _ = self.robot.find_bodies([cfg.anchor_body_name])
        self._anchor_body_id = anchor_ids[0]

        self._left_wrist_body_id: int | None = None
        self._right_wrist_body_id: int | None = None
        self._left_fingertip_body_ids: list[int] = []
        self._right_fingertip_body_ids: list[int] = []
        self._left_finger_joint_ids: list[int] = []
        self._right_finger_joint_ids: list[int] = []

    @staticmethod
    def _metadata_equal(values: list[Any], label: str) -> Any:
        """Return common motion metadata or reject an incompatible bank."""
        first = values[0]
        if any(value != first for value in values[1:]):
            raise ValueError(
                f"MotionTrackingCommand requires matching {label} across motions; "
                f"got {values}."
            )
        return first

    def _pad_bank(
        self, tensors: list[torch.Tensor | None], label: str
    ) -> torch.Tensor | None:
        """Stack time-major tensors, repeating each sequence's final frame."""
        present = [tensor is not None for tensor in tensors]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(
                f"MotionTrackingCommand requires {label} in either every motion "
                "or no motions."
            )
        concrete = [tensor for tensor in tensors if tensor is not None]
        tail_shape = concrete[0].shape[1:]
        if any(tensor.shape[1:] != tail_shape for tensor in concrete[1:]):
            raise ValueError(
                f"MotionTrackingCommand requires matching {label} shapes; got "
                f"{[tuple(tensor.shape) for tensor in concrete]}."
            )
        padded: list[torch.Tensor] = []
        for source_tensor in concrete:
            if source_tensor.shape[0] == 0:
                raise ValueError(f"Motion has no frames for required field {label}.")
            pad_count = self.num_timesteps - source_tensor.shape[0]
            padded_tensor = source_tensor
            if pad_count:
                padded_tensor = torch.cat(
                    [
                        source_tensor,
                        source_tensor[-1:].expand(pad_count, *tail_shape),
                    ],
                    dim=0,
                )
            padded.append(padded_tensor)
        return torch.stack(padded)

    def _load_and_process_motion(self, cfg: MotionTrackingCommandCfg) -> None:
        """Load, validate, and pad all configured motion sequences."""
        self.playback_dt = float(self._env.step_dt)
        motion_files = list(cfg.motion_files) if cfg.motion_files else [cfg.motion_file]
        if not motion_files or not all(
            isinstance(path, str) and path for path in motion_files
        ):
            raise ValueError(
                "Set motion_files or the single-motion motion_file fallback."
            )

        motions = [
            load_motion_data(
                cfg,
                self.robot,
                self.device,
                step_dt=self.playback_dt,
                motion_file=path,
            )
            for path in motion_files
        ]
        self.motion_files = motion_files
        self._motion_data_bank = motions
        self._motion_data = motions[0]  # hand metadata is resolved once below
        self.num_motions = len(motions)
        self.motion_lengths = torch.tensor(
            [motion.robot_root_position.shape[0] for motion in motions],
            dtype=torch.long,
            device=self.device,
        )
        self.num_timesteps = int(self.motion_lengths.max().item())

        root_offset = torch.tensor(cfg.robot_anchor_pos_offset, device=self.device)
        self.root_pos_w = self._pad_bank(
            [motion.robot_root_position.float() + root_offset for motion in motions],
            "robot root positions",
        )
        self.root_quat_w = self._pad_bank(
            [motion.robot_root_wxyz.float() for motion in motions],
            "robot root orientations",
        )

        self._tracked_joint_ids, self._tracked_joint_names = self.robot.find_joints(
            cfg.joint_names
        )
        ordered_joint_pos: list[torch.Tensor] = []
        for motion in motions:
            file_names = motion.file_joint_names or cfg.file_joint_names
            positions = motion.robot_joint_positions.float()
            if file_names is not None:
                missing = [
                    name for name in self._tracked_joint_names if name not in file_names
                ]
                if missing:
                    raise ValueError(
                        f"Motion is missing tracked joints {missing} ({motion_files[len(ordered_joint_pos)]})."
                    )
                positions = positions[
                    :, [file_names.index(name) for name in self._tracked_joint_names]
                ]
            ordered_joint_pos.append(positions)
        joint_pos = self._pad_bank(ordered_joint_pos, "joint positions")
        if joint_pos is None:
            raise RuntimeError("Joint-position data is required for motion tracking.")
        self.joint_pos = joint_pos
        self.joint_vel = torch.zeros_like(self.joint_pos)
        for motion_id, length in enumerate(self.motion_lengths.tolist()):
            if length > 1:
                self.joint_vel[motion_id, : length - 1] = (
                    self.joint_pos[motion_id, 1:length]
                    - self.joint_pos[motion_id, : length - 1]
                ) / self.playback_dt

        ee_names = self._metadata_equal(
            [list(motion.ee_link_names or []) for motion in motions], "EE link names"
        )
        self.ee_link_names = ee_names
        self.ee_link_ids = motions[0].ee_link_ids or []
        self.ee_pos_w = self._pad_bank(
            [
                motion.ee_pos_w.float() if motion.ee_pos_w is not None else None
                for motion in motions
            ],
            "EE positions",
        )
        self.ee_quat_w = self._pad_bank(
            [
                motion.ee_quat_w.float() if motion.ee_quat_w is not None else None
                for motion in motions
            ],
            "EE orientations",
        )
        for body_id, body_name in zip(
            self.ee_link_ids, self.ee_link_names, strict=False
        ):
            if "left" in body_name.lower():
                self._left_wrist_body_id = body_id
            elif "right" in body_name.lower():
                self._right_wrist_body_id = body_id

        self.num_future_frames = cfg.num_future_frames
        self.frame_step = max(1, round(cfg.dt_future_frames / self.playback_dt))
        self._future_frame_offsets = torch.arange(
            0,
            self.num_future_frames * self.frame_step,
            self.frame_step,
            dtype=torch.long,
            device=self.device,
        )

    def _init_buffers(self, cfg: MotionTrackingCommandCfg) -> None:
        """Allocate per-environment sequence selection and tracking state."""
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.timestep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_timestep = torch.zeros_like(self.timestep)
        self.trajectory_end_timestep = self.selected_motion_lengths - 1
        self.steps_since_last_reset = torch.zeros_like(self.timestep)
        self._encoder_mode = torch.zeros(self.num_envs, 4, device=self.device)
        self.tracking_lengths = torch.minimum(
            self.selected_motion_lengths,
            torch.full_like(
                self.selected_motion_lengths, int(self._env.max_episode_length)
            ),
        )
        self._action_history: torch.Tensor | None = None
        self._action_history_len = cfg.action_history_length
        self._spread_joint_offset = torch.zeros(
            self.num_envs, self.joint_pos.shape[-1], device=self.device
        )
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)

    def _init_metrics(self) -> None:
        """Allocate reference-tracking metrics only."""
        for name in ("anchor_position_error", "anchor_wxyz_error", "joint_pos_error"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        for side in ("left", "right"):
            for suffix in (
                "hand_wrist_position_error",
                "hand_wrist_wxyz_error",
                "hand_finger_joints_error",
            ):
                self.metrics[f"{side}_{suffix}"] = torch.zeros(
                    self.num_envs, device=self.device
                )

    def _init_hand_data(self, cfg: MotionTrackingCommandCfg) -> None:
        """Resolve robot hand indices and bank retargeted hand references."""
        if cfg.fingertip_body_name:
            tip_ids, tip_names = self.robot.find_bodies(cfg.fingertip_body_name)
            self._left_fingertip_body_ids = [
                i
                for i, name in zip(tip_ids, tip_names, strict=False)
                if "left" in name.lower()
            ]
            self._right_fingertip_body_ids = [
                i
                for i, name in zip(tip_ids, tip_names, strict=False)
                if "right" in name.lower()
            ]

        self._left_finger_joint_names: list[str] = []
        self._right_finger_joint_names: list[str] = []
        if cfg.finger_joint_names:
            ids, names = self.robot.find_joints(cfg.finger_joint_names)
            for joint_id, name in zip(ids, names, strict=False):
                if "left" in name.lower():
                    self._left_finger_joint_ids.append(joint_id)
                    self._left_finger_joint_names.append(name)
                elif "right" in name.lower():
                    self._right_finger_joint_ids.append(joint_id)
                    self._right_finger_joint_names.append(name)

        for side in ("left", "right"):
            frame_names = self._metadata_equal(
                [
                    list(getattr(md, f"{side}_hand_frame_names") or [])
                    for md in self._motion_data_bank
                ],
                f"{side} hand frame names",
            )
            setattr(self, f"retargeted_{side}_hand_frame_names", frame_names)
            frames = self._pad_bank(
                [getattr(md, f"{side}_hand_frames") for md in self._motion_data_bank],
                f"{side} hand frames",
            )
            setattr(self, f"retargeted_{side}_hand_frames", frames)
            setattr(
                self,
                f"retargeted_{side}_wrist_position",
                self._pad_bank(
                    [
                        getattr(md, f"{side}_wrist_position")
                        for md in self._motion_data_bank
                    ],
                    f"{side} wrist positions",
                ),
            )
            setattr(
                self,
                f"retargeted_{side}_wrist_wxyz",
                self._pad_bank(
                    [
                        getattr(md, f"{side}_wrist_wxyz")
                        for md in self._motion_data_bank
                    ],
                    f"{side} wrist orientations",
                ),
            )

            tip_names = [
                self.robot.body_names[index]
                for index in getattr(self, f"_{side}_fingertip_body_ids")
            ]
            tip_indices = [
                frame_names.index(name) for name in tip_names if name in frame_names
            ]
            setattr(
                self,
                f"_retargeted_{side}_fingertip_indices",
                torch.tensor(tip_indices, dtype=torch.long, device=self.device),
            )

            sim_names = getattr(self, f"_{side}_finger_joint_names")
            ordered = []
            for md in self._motion_data_bank:
                values = getattr(md, f"{side}_finger_joints")
                parquet_names = getattr(md, f"{side}_finger_joint_names") or []
                if values is not None and sim_names:
                    missing = [name for name in sim_names if name not in parquet_names]
                    if missing:
                        raise ValueError(
                            f"Motion is missing {side} finger joints {missing}."
                        )
                    values = values[
                        :, [parquet_names.index(name) for name in sim_names]
                    ]
                ordered.append(values)
            setattr(
                self,
                f"retargeted_{side}_finger_joints",
                self._pad_bank(ordered, f"{side} finger joints"),
            )

    @property
    def selected_motion_lengths(self) -> torch.Tensor:
        """Return the trajectory length selected by each environment."""
        return self.motion_lengths[self.motion_ids]

    def sample_motions(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        """Uniformly select a new motion for the requested environments."""
        count = len(env_ids)
        self.motion_ids[env_ids] = torch.randint(
            self.num_motions, (count,), device=self.device
        )

    def _gather(
        self, bank: torch.Tensor, timesteps: torch.Tensor | None = None
    ) -> torch.Tensor:
        timesteps = self.timestep if timesteps is None else timesteps
        return bank[
            self.motion_ids[:, None] if timesteps.ndim > 1 else self.motion_ids,
            timesteps,
        ]

    def get_frame(
        self, bank: torch.Tensor, env_ids: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Gather arbitrary reset frames for a subset of environments."""
        return bank[self.motion_ids[env_ids], timesteps]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Return current robot anchor positions in world coordinates."""
        return self.robot.data.body_pos_w[:, self._anchor_body_id]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Return current robot anchor orientations in world coordinates."""
        return self.robot.data.body_quat_w[:, self._anchor_body_id]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """Return current positions for the tracked robot joints."""
        return self.robot.data.joint_pos[:, self._tracked_joint_ids]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """Return current velocities for the tracked robot joints."""
        return self.robot.data.joint_vel[:, self._tracked_joint_ids]

    @property
    def robot_ee_pos_w(self) -> torch.Tensor:
        """Return current end-effector positions in world coordinates."""
        if self.ee_link_ids:
            return self.robot.data.body_pos_w[:, self.ee_link_ids]
        return torch.zeros(self.num_envs, 0, 3, device=self.device)

    @property
    def robot_ee_quat_w(self) -> torch.Tensor:
        """Return current end-effector orientations in world coordinates."""
        if self.ee_link_ids:
            return self.robot.data.body_quat_w[:, self.ee_link_ids]
        return torch.zeros(self.num_envs, 0, 4, device=self.device)

    @property
    def encoder_mode(self) -> torch.Tensor:
        """Return the compatibility encoder-mode buffer."""
        return self._encoder_mode

    @property
    def timestep_counter(self) -> torch.Tensor:
        """Return the current trajectory timestep for each environment."""
        return self.timestep

    @property
    def future_timesteps(self) -> torch.Tensor:
        """Return clamped future trajectory indices for each environment."""
        future = self.timestep[:, None] + self._future_frame_offsets[None, :]
        return torch.minimum(future, self.trajectory_end_timestep[:, None])

    def update_action_history(self, actions: torch.Tensor) -> None:
        """Append the latest actions to the fixed-length history buffer."""
        if self._action_history is None:
            self._action_history = torch.zeros(
                self.num_envs,
                self._action_history_len,
                actions.shape[-1],
                device=self.device,
            )
        self._action_history = torch.roll(self._action_history, -1, dims=1)
        self._action_history[:, -1] = actions

    @property
    def action_history(self) -> torch.Tensor:
        """Return flattened recent processed actions."""
        if self._action_history is None:
            action_dim = self._env.action_manager.get_term(
                "joint_pos"
            ).processed_actions.shape[-1]
            self._action_history = torch.zeros(
                self.num_envs, self._action_history_len, action_dim, device=self.device
            )
        return self._action_history.reshape(self.num_envs, -1)

    @property
    def _spread_blend_factor(self) -> torch.Tensor:
        if self.cfg.reset_shoulder_spread == 0.0 or self.cfg.reset_freeze_steps == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return (
            1.0 - self.steps_since_last_reset.float() / self.cfg.reset_freeze_steps
        ).clamp(min=0.0)

    @property
    def _spread_offset_blended(self) -> torch.Tensor:
        return self._spread_joint_offset * self._spread_blend_factor.unsqueeze(-1)

    @property
    def command(self) -> torch.Tensor:
        """Return the current reference joint-position command."""
        return self.command_joint_pos

    @property
    def command_anchor_pos_w(self) -> torch.Tensor:
        """Return current reference anchor positions in world coordinates."""
        return self._gather(self.root_pos_w) + self._env_origins

    @property
    def command_anchor_quat_w(self) -> torch.Tensor:
        """Return current reference anchor orientations in world coordinates."""
        return math_utils.quat_unique(self._gather(self.root_quat_w))

    @property
    def command_joint_pos(self) -> torch.Tensor:
        """Return current reference joint positions."""
        return self._gather(self.joint_pos) + self._spread_offset_blended

    @property
    def command_joint_pos_multi_future(self) -> torch.Tensor:
        """Return current and future reference joint positions."""
        return self._gather(
            self.joint_pos, self.future_timesteps
        ) + self._spread_offset_blended.unsqueeze(1)

    @property
    def command_joint_vel_multi_future(self) -> torch.Tensor:
        """Return current and future reference joint velocities."""
        return self._gather(self.joint_vel, self.future_timesteps)

    @property
    def command_anchor_pos_w_multi_future(self) -> torch.Tensor:
        """Return current and future reference anchor positions."""
        return self._gather(
            self.root_pos_w, self.future_timesteps
        ) + self._env_origins.unsqueeze(1)

    @property
    def command_ee_pos_w(self) -> torch.Tensor:
        """Return current reference end-effector positions."""
        if self.ee_pos_w is None:
            return torch.zeros(self.num_envs, 0, 3, device=self.device)
        return self._gather(self.ee_pos_w) + self._env_origins.unsqueeze(1)

    @property
    def command_ee_quat_w(self) -> torch.Tensor:
        """Return current reference end-effector orientations."""
        if self.ee_quat_w is None:
            return torch.zeros(self.num_envs, 0, 4, device=self.device)
        return math_utils.quat_unique(self._gather(self.ee_quat_w))

    @property
    def command_ee_pos_w_multi_future(self) -> torch.Tensor:
        """Return current and future reference end-effector positions."""
        if self.ee_pos_w is None:
            return torch.zeros(
                self.num_envs, self.num_future_frames, 0, 3, device=self.device
            )
        return (
            self._gather(self.ee_pos_w, self.future_timesteps)
            + self._env_origins[:, None, None]
        )

    @property
    def command_ee_quat_w_multi_future(self) -> torch.Tensor:
        """Return current and future reference end-effector orientations."""
        if self.ee_quat_w is None:
            return torch.zeros(
                self.num_envs, self.num_future_frames, 0, 4, device=self.device
            )
        return math_utils.quat_unique(
            self._gather(self.ee_quat_w, self.future_timesteps)
        )

    @property
    def command_anchor_rot_diff_l_multi_future(self) -> torch.Tensor:
        """Return future anchor rotations relative to the current anchor."""
        future = self._gather(self.root_quat_w, self.future_timesteps)
        current_inv = math_utils.quat_conjugate(
            self._gather(self.root_quat_w)
        ).unsqueeze(1)
        diff = math_utils.quat_mul(future, current_inv.expand_as(future))
        return math_utils.matrix_from_quat(diff)[..., :2].reshape(
            self.num_envs, self.num_future_frames, 6
        )

    @property
    def command_anchor_z_multi_future(self) -> torch.Tensor:
        """Return current and future reference anchor heights."""
        return self._gather(self.root_pos_w, self.future_timesteps)[..., 2]

    @property
    def command_multi_future(self) -> torch.Tensor:
        """Return concatenated future joint-position and velocity commands."""
        return torch.cat(
            [self.command_joint_pos_multi_future, self.command_joint_vel_multi_future],
            dim=-1,
        )

    def _current_wrist_position(self, side: str) -> torch.Tensor:
        body_id = getattr(self, f"_{side}_wrist_body_id")
        if body_id is None:
            return torch.zeros(self.num_envs, 3, device=self.device)
        return self.robot.data.body_pos_w[:, body_id] - self._env_origins

    def _current_wrist_quat(self, side: str) -> torch.Tensor:
        body_id = getattr(self, f"_{side}_wrist_body_id")
        if body_id is None:
            quat = torch.zeros(self.num_envs, 4, device=self.device)
            quat[:, 0] = 1.0
            return quat
        return math_utils.quat_unique(self.robot.data.body_quat_w[:, body_id])

    def _wrist_command_e(self, side: str) -> torch.Tensor:
        positions = getattr(self, f"retargeted_{side}_wrist_position")
        quaternions = getattr(self, f"retargeted_{side}_wrist_wxyz")
        if positions is None or quaternions is None:
            return torch.zeros(self.num_envs, 7, device=self.device)
        return torch.cat([self._gather(positions), self._gather(quaternions)], dim=-1)

    def _fingertip_command_e(self, side: str) -> torch.Tensor:
        frames = getattr(self, f"retargeted_{side}_hand_frames")
        indices = getattr(self, f"_retargeted_{side}_fingertip_indices")
        if frames is None or len(indices) == 0:
            return torch.zeros(self.num_envs, 0, 3, device=self.device)
        return self._gather(frames)[:, indices, :3]

    @property
    def left_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """Return the left-wrist reference pose in environment coordinates."""
        return self._wrist_command_e("left")

    @property
    def right_hand_wrist_pose_command_e(self) -> torch.Tensor:
        """Return the right-wrist reference pose in environment coordinates."""
        return self._wrist_command_e("right")

    @property
    def left_hand_wrist_position_e(self) -> torch.Tensor:
        """Return the current left-wrist position in environment coordinates."""
        return self._current_wrist_position("left")

    @property
    def right_hand_wrist_position_e(self) -> torch.Tensor:
        """Return the current right-wrist position in environment coordinates."""
        return self._current_wrist_position("right")

    @property
    def left_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """Return the current left-wrist orientation."""
        return self._current_wrist_quat("left")

    @property
    def right_hand_wrist_wxyz_e(self) -> torch.Tensor:
        """Return the current right-wrist orientation."""
        return self._current_wrist_quat("right")

    @property
    def left_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """Return left-fingertip reference positions."""
        return self._fingertip_command_e("left")

    @property
    def right_hand_fingertip_position_command_e(self) -> torch.Tensor:
        """Return right-fingertip reference positions."""
        return self._fingertip_command_e("right")

    def _current_fingertips(self, side: str) -> torch.Tensor:
        ids = getattr(self, f"_{side}_fingertip_body_ids")
        if not ids:
            return torch.zeros(self.num_envs, 0, 3, device=self.device)
        return self.robot.data.body_pos_w[:, ids] - self._env_origins.unsqueeze(1)

    @property
    def left_hand_fingertip_position_e(self) -> torch.Tensor:
        """Return current left-fingertip positions."""
        return self._current_fingertips("left")

    @property
    def right_hand_fingertip_position_e(self) -> torch.Tensor:
        """Return current right-fingertip positions."""
        return self._current_fingertips("right")

    def _finger_joint_pos(self, side: str) -> torch.Tensor:
        ids = getattr(self, f"_{side}_finger_joint_ids")
        if not ids:
            return torch.zeros(self.num_envs, 0, device=self.device)
        return self.robot.data.joint_pos[:, ids]

    def _finger_joint_command(self, side: str) -> torch.Tensor:
        values = getattr(self, f"retargeted_{side}_finger_joints")
        if values is None:
            return torch.zeros(self.num_envs, 0, device=self.device)
        return self._gather(values)

    @property
    def left_hand_finger_joint_pos(self) -> torch.Tensor:
        """Return current left-hand joint positions."""
        return self._finger_joint_pos("left")

    @property
    def right_hand_finger_joint_pos(self) -> torch.Tensor:
        """Return current right-hand joint positions."""
        return self._finger_joint_pos("right")

    @property
    def left_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """Return reference left-hand joint positions."""
        return self._finger_joint_command("left")

    @property
    def right_hand_finger_joint_pos_command(self) -> torch.Tensor:
        """Return reference right-hand joint positions."""
        return self._finger_joint_command("right")

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Clear episode-local state; the reset event samples the motion first."""
        self.steps_since_last_reset[env_ids] = 0
        if self._action_history is not None:
            self._action_history[env_ids] = 0.0

    def _update_command(self) -> None:
        self.steps_since_last_reset += 1
        if self.cfg.reset_freeze_steps > 0:
            self.timestep[
                self.steps_since_last_reset > self.cfg.reset_freeze_steps
            ] += 1
        else:
            self.timestep += 1
        self.timestep.copy_(torch.minimum(self.timestep, self.trajectory_end_timestep))

    def _update_metrics(self) -> None:
        self.metrics["anchor_position_error"] = torch.norm(
            self.robot_anchor_pos_w - self.command_anchor_pos_w, dim=-1
        )
        self.metrics["anchor_wxyz_error"] = math_utils.quat_error_magnitude(
            self.robot_anchor_quat_w, self.command_anchor_quat_w
        )
        self.metrics["joint_pos_error"] = torch.norm(
            self.robot_joint_pos - self.command_joint_pos, dim=-1
        )
        for side in ("left", "right"):
            wrist_command = self._wrist_command_e(side)
            self.metrics[f"{side}_hand_wrist_position_error"] = torch.norm(
                self._current_wrist_position(side) - wrist_command[:, :3], dim=-1
            )
            body_id = getattr(self, f"_{side}_wrist_body_id")
            if body_id is not None:
                self.metrics[f"{side}_hand_wrist_wxyz_error"] = (
                    math_utils.quat_error_magnitude(
                        self._current_wrist_quat(side), wrist_command[:, 3:]
                    )
                )
            else:
                self.metrics[f"{side}_hand_wrist_wxyz_error"].zero_()
            self.metrics[f"{side}_hand_finger_joints_error"] = torch.norm(
                self._finger_joint_pos(side) - self._finger_joint_command(side), dim=-1
            )

    def _set_debug_vis_impl(self, debug_vis: bool = True) -> None:
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                cfg = self.cfg.pose_visualizer_cfg.replace(
                    prim_path="/Visuals/Command/free_space_goal_marker"
                )
                self.goal_pose_visualizer = VisualizationMarkers(cfg)
            self.goal_pose_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event: object) -> None:
        if hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.visualize(
                translations=self.command_anchor_pos_w,
                orientations=self.command_anchor_quat_w,
            )
