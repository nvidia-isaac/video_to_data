# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from robotic_grounding.tasks.v2p_whole_body.mdp.commands import TrackingCommand


def reset_robot_to_trajectory_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg | None = None,
    trajectory_time_index: tuple[int, int] = (0, 0),
) -> None:
    """Reset robot and object to a frame in the motion trajectory.

    Frame-window selection:
    - trajectory_time_index is the inclusive command window [start, end].
    - always_reset_to_first_frame: reset to the window start.
    - Otherwise: random reset inside the window.
    - The timestep termination ends the episode at the window end.

    Post-processing (all configurable on TrackingCommandCfg):
    - Optional root Z clamp (reset_root_height_min)
    - Optional yaw-only root quaternion (reset_yaw_only)
    - Optional shoulder spread + finger zeroing during freeze (reset_shoulder_spread)
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]

    # --- Frame selection ---
    low = max(0, int(trajectory_time_index[0]))
    high = min(command.num_timesteps - 1, int(trajectory_time_index[1]))
    high = max(low, high)
    if command.cfg.always_reset_to_first_frame:
        reset_ts = torch.full(
            (len(env_ids),), low, dtype=torch.int32, device=env.device
        )
    else:
        reset_ts = torch.randint(
            low,
            high + 1,
            (len(env_ids),),
            dtype=torch.int32,
            device=env.device,
        )

    command.timestep[env_ids] = reset_ts
    command.reset_timestep[env_ids] = reset_ts
    command.trajectory_end_timestep[env_ids] = high
    command.tracking_lengths[env_ids] = (high - reset_ts + 1).clamp(min=1)

    # --- Read trajectory frame ---
    initial_root_pos = command.root_pos_w[reset_ts].clone()
    initial_root_quat = command.root_quat_w[reset_ts].clone()
    initial_joint_pos = command.joint_pos[reset_ts].clone()

    # --- Root Z clamp ---
    if command.cfg.reset_root_height_min is not None:
        initial_root_pos[:, 2] = initial_root_pos[:, 2].clamp(
            min=command.cfg.reset_root_height_min
        )

    # --- Yaw-only root quaternion ---
    if command.cfg.reset_yaw_only:
        w = initial_root_quat[:, 0]
        x = initial_root_quat[:, 1]
        y = initial_root_quat[:, 2]
        z = initial_root_quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        half_yaw = yaw * 0.5
        initial_root_quat = torch.stack(
            [
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ],
            dim=-1,
        )

    # --- Shoulder spread + finger zeroing ---
    freeze_steps = command.cfg.reset_freeze_steps
    shoulder_spread = command.cfg.reset_shoulder_spread
    if shoulder_spread > 0.0 and freeze_steps > 0:
        spread_offset = torch.zeros_like(initial_joint_pos)
        joint_names = robot.joint_names
        for i, name in enumerate(joint_names):
            if name == "left_shoulder_yaw_joint":
                spread_offset[:, i] = shoulder_spread
            elif name == "right_shoulder_yaw_joint":
                spread_offset[:, i] = -shoulder_spread
            elif any(
                finger in name
                for finger in ("thumb", "index", "middle", "ring", "pinky")
            ):
                spread_offset[:, i] = -initial_joint_pos[:, i]
        initial_joint_pos = initial_joint_pos + spread_offset
        command._spread_joint_offset[env_ids] = spread_offset
    else:
        command._spread_joint_offset[env_ids] = 0.0

    # --- Write to sim ---
    root_pos_w = initial_root_pos + env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(
        torch.cat([root_pos_w, initial_root_quat], dim=-1), env_ids=env_ids
    )
    robot.write_root_velocity_to_sim(
        torch.zeros_like(robot.data.root_vel_w[env_ids]), env_ids=env_ids
    )
    robot.write_joint_state_to_sim(
        initial_joint_pos,
        torch.zeros_like(robot.data.joint_vel[env_ids]),
        env_ids=env_ids,
    )

    # --- Reset objects ---
    scene_objects = getattr(command, "objects", None) or [
        env.scene[command.cfg.object_name]
    ]
    object_pose = torch.cat(
        [
            command._object_body_pos_w[reset_ts] + env.scene.env_origins[env_ids, None],
            command._object_body_quat_w[reset_ts],
        ],
        dim=-1,
    )
    object_velocity = torch.zeros(
        object_pose.shape[0],
        object_pose.shape[1],
        6,
        device=object_pose.device,
        dtype=object_pose.dtype,
    )
    object_joint_pos = None
    if command.retargeted_object_articulation.numel() > 0:
        object_joint_pos = command.retargeted_object_articulation[reset_ts]
        if object_joint_pos.dim() == 1:
            object_joint_pos = object_joint_pos.unsqueeze(-1)

    for object_idx, scene_object in enumerate(scene_objects):
        scene_object.write_root_pose_to_sim(object_pose[:, object_idx], env_ids=env_ids)
        scene_object.write_root_velocity_to_sim(
            object_velocity[:, object_idx], env_ids=env_ids
        )
        if isinstance(scene_object, Articulation) and object_joint_pos is not None:
            if len(scene_objects) > 1:
                raise NotImplementedError(
                    "Multi-object trajectory reset currently supports separate "
                    "RigidObject assets or a single Articulation asset."
                )
            scene_object.write_joint_state_to_sim(
                object_joint_pos,
                torch.zeros_like(scene_object.data.joint_vel[env_ids]),
                env_ids=env_ids,
            )
