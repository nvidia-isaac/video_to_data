# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable dual-hand kinematic-replay helpers.

Extracted so both ``scripts/replay_motion.py`` (kinematic viewer) and
``scripts/rsl_rl/record_dataset.py`` (camera/recorder env) can teleport the
floating Sharpa hands to a reference wrist+finger trajectory. Dual-hand only.
"""
from __future__ import annotations

from typing import Any

import torch
from scipy.spatial.transform import Rotation as R

from robotic_grounding.tasks.scene_utils.replay_data import (
    DualHandTrajectory,
    load_replay_trajectory,
)


class DualHandReplay:
    """Reference wrist poses + finger joints for left/right floating hands."""

    def __init__(
        self,
        motion_file: str,
        device: torch.device,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        """Load a dual-hand reference trajectory and cache wrist/finger tensors on ``device``."""
        replay = load_replay_trajectory(
            motion_file, start_frame=start_frame, end_frame=end_frame
        )
        if not isinstance(replay, DualHandTrajectory):
            raise ValueError(
                "replay_kinematics supports dual-hand layouts only; got "
                f"{type(replay).__name__}"
            )
        self.fps = replay.fps
        self.num_frames = int(replay.num_frames)
        self.right_joint_names = list(replay.right_joint_names)
        self.left_joint_names = list(replay.left_joint_names)

        self.right_wrist_pos = torch.tensor(
            replay.right_wrist_position, dtype=torch.float32, device=device
        )
        self.left_wrist_pos = torch.tensor(
            replay.left_wrist_position, dtype=torch.float32, device=device
        )
        if replay.wrist_orientation_format == "wxyz":
            self.right_wrist_wxyz = torch.tensor(
                replay.right_wrist_orientation, dtype=torch.float32, device=device
            )
            self.left_wrist_wxyz = torch.tensor(
                replay.left_wrist_orientation, dtype=torch.float32, device=device
            )
        else:
            r = [
                R.from_euler("XYZ", xyz, degrees=False).as_quat(scalar_first=True)
                for xyz in replay.right_wrist_orientation
            ]
            l = [
                R.from_euler("XYZ", xyz, degrees=False).as_quat(scalar_first=True)
                for xyz in replay.left_wrist_orientation
            ]
            self.right_wrist_wxyz = torch.tensor(r, dtype=torch.float32, device=device)
            self.left_wrist_wxyz = torch.tensor(l, dtype=torch.float32, device=device)
        self.right_finger_joints = torch.tensor(
            replay.right_finger_joints, dtype=torch.float32, device=device
        )
        self.left_finger_joints = torch.tensor(
            replay.left_finger_joints, dtype=torch.float32, device=device
        )


def build_joint_reorder(
    parquet_names: list[str], sim_names: list[str]
) -> torch.Tensor | None:
    """Map Parquet joint order -> Isaac joint order. None if already identical."""
    if parquet_names == sim_names:
        return None
    sim_name_to_idx = {n: i for i, n in enumerate(sim_names)}
    indices = [sim_name_to_idx[n] for n in parquet_names]
    return torch.tensor(indices, dtype=torch.long)


def disable_gravity_in_articulation_cfg(cfg: Any) -> Any:
    """Return an articulation cfg with gravity disabled (for teleported bodies)."""
    spawn = getattr(cfg, "spawn", None)
    rigid_props = getattr(spawn, "rigid_props", None)
    if spawn is None or rigid_props is None:
        return cfg
    return cfg.replace(
        spawn=spawn.replace(rigid_props=rigid_props.replace(disable_gravity=True))
    )


def _write_one_hand(
    robot: Any,
    wrist_pos: torch.Tensor,
    wrist_wxyz: torch.Tensor,
    finger_joints: torch.Tensor,
    frame_idx: torch.Tensor,
    env_origins: torch.Tensor,
    reorder: torch.Tensor | None,
    sim_joint_names: list[str],
    device: torch.device,
) -> None:
    # Per-env frame index -> per-env pose/joints (handles staggered resets).
    pos = wrist_pos[frame_idx] + env_origins  # (N, 3)
    wxyz = wrist_wxyz[frame_idx]  # (N, 4)
    robot.write_root_pose_to_sim(torch.cat([pos, wxyz], dim=-1))
    robot.write_root_velocity_to_sim(torch.zeros(frame_idx.shape[0], 6, device=device))
    joints = finger_joints[frame_idx]  # (N, n_parquet)
    if reorder is not None:
        sim_joints = torch.zeros(
            frame_idx.shape[0], len(sim_joint_names), device=device
        )
        sim_joints[:, reorder] = joints
    else:
        sim_joints = joints
    robot.write_joint_state_to_sim(sim_joints, torch.zeros_like(sim_joints))


def write_dual_hand_frame_per_env(
    right_robot: Any,
    left_robot: Any,
    traj: DualHandReplay,
    frame_idx: torch.Tensor,
    env_origins: torch.Tensor,
    right_reorder: torch.Tensor | None,
    left_reorder: torch.Tensor | None,
    right_sim_joint_names: list[str],
    left_sim_joint_names: list[str],
    device: torch.device,
) -> None:
    """Teleport both hands to ``traj`` at the per-env ``frame_idx`` (shape (num_envs,))."""
    _write_one_hand(
        right_robot,
        traj.right_wrist_pos,
        traj.right_wrist_wxyz,
        traj.right_finger_joints,
        frame_idx,
        env_origins,
        right_reorder,
        right_sim_joint_names,
        device,
    )
    _write_one_hand(
        left_robot,
        traj.left_wrist_pos,
        traj.left_wrist_wxyz,
        traj.left_finger_joints,
        frame_idx,
        env_origins,
        left_reorder,
        left_sim_joint_names,
        device,
    )
