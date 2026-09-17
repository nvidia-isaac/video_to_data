# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from robotic_grounding.tasks.v2d_whole_body.mdp.commands import TrackingCommand


def reset_robot_to_trajectory_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    *,
    command_name: str = "motion",
    asset_cfg: SceneEntityCfg | None = None,
    trajectory_time_index: tuple[int, int] = (0, 0),
    joint_position_noise_groups: dict[str, dict] | None = None,
    object_xy_noise_range: tuple[float, float] = (0.0, 0.0),
    object_yaw_noise_range: tuple[float, float] = (0.0, 0.0),
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

    Optional initial-condition noise changes only the simulator state constructed from
    the selected frame. Motion-reference tensors owned by the command are never mutated.
    Degenerate ranges disable the corresponding perturbation.
    """
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
    command: TrackingCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]

    # --- Motion and frame selection ---
    # Multi-motion commands must sample before the reset pose is read. Command
    # manager resampling happens after reset-mode events, which is too late.
    if hasattr(command, "sample_motions"):
        command.sample_motions(env_ids)

    low = max(0, int(trajectory_time_index[0]))
    requested_high = int(trajectory_time_index[1])
    if hasattr(command, "selected_motion_lengths"):
        low_ts = torch.minimum(
            command.selected_motion_lengths[env_ids] - 1,
            torch.full((len(env_ids),), low, dtype=torch.long, device=env.device),
        )
        high = torch.minimum(
            command.selected_motion_lengths[env_ids] - 1,
            torch.full(
                (len(env_ids),), requested_high, dtype=torch.long, device=env.device
            ),
        )
        high = torch.maximum(low_ts, high)
    else:
        scalar_high = max(low, min(command.num_timesteps - 1, requested_high))
        low_ts = torch.full((len(env_ids),), low, dtype=torch.long, device=env.device)
        high = torch.full(
            (len(env_ids),), scalar_high, dtype=torch.long, device=env.device
        )
    if command.cfg.always_reset_to_first_frame:
        reset_ts = low_ts
    else:
        # torch.randint does not accept a per-element upper bound.
        reset_ts = (
            low_ts
            + (torch.rand(len(env_ids), device=env.device) * (high - low_ts + 1)).long()
        )

    command.timestep[env_ids] = reset_ts
    command.reset_timestep[env_ids] = reset_ts
    command.trajectory_end_timestep[env_ids] = high
    command.tracking_lengths[env_ids] = (high - reset_ts + 1).clamp(min=1)

    # --- Read trajectory frame ---
    if hasattr(command, "get_frame"):
        initial_root_pos = command.get_frame(
            command.root_pos_w, env_ids, reset_ts
        ).clone()
        initial_root_quat = command.get_frame(
            command.root_quat_w, env_ids, reset_ts
        ).clone()
        initial_joint_pos = command.get_frame(
            command.joint_pos, env_ids, reset_ts
        ).clone()
    else:
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

    # --- Optional collection-time joint perturbations ---
    for group_name, group in (joint_position_noise_groups or {}).items():
        joint_names = list(group.get("joint_names", ()))
        noise_range = tuple(group.get("range", (0.0, 0.0)))
        if len(noise_range) != 2 or noise_range[0] > noise_range[1]:
            raise ValueError(
                f"invalid joint noise range for {group_name!r}: {noise_range}"
            )
        if noise_range == (0.0, 0.0):
            continue
        if not joint_names:
            raise ValueError(f"joint noise group {group_name!r} has no joints")
        joint_ids, resolved_names = robot.find_joints(joint_names, preserve_order=True)
        if list(resolved_names) != joint_names:
            raise ValueError(
                f"joint noise group {group_name!r} did not resolve exactly: "
                f"requested={joint_names}, resolved={resolved_names}"
            )
        offsets = torch.empty(
            (len(env_ids), len(joint_ids)),
            dtype=initial_joint_pos.dtype,
            device=env.device,
        ).uniform_(float(noise_range[0]), float(noise_range[1]))
        limits = robot.data.joint_pos_limits[env_ids][:, joint_ids]
        initial_joint_pos[:, joint_ids] = (
            initial_joint_pos[:, joint_ids] + offsets
        ).clamp(min=limits[..., 0], max=limits[..., 1])

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
    scene_objects = getattr(command, "objects", None)
    if not scene_objects:
        return

    if hasattr(command, "get_frame"):
        object_positions = command.get_frame(
            command._object_body_pos_w, env_ids, reset_ts
        )
        object_quaternions = command.get_frame(
            command._object_body_quat_w, env_ids, reset_ts
        )
    else:
        object_positions = command._object_body_pos_w[reset_ts]
        object_quaternions = command._object_body_quat_w[reset_ts]
    object_pose = torch.cat(
        [
            object_positions + env.scene.env_origins[env_ids, None],
            object_quaternions,
        ],
        dim=-1,
    )
    for label, noise_range in (
        ("object_xy_noise_range", object_xy_noise_range),
        ("object_yaw_noise_range", object_yaw_noise_range),
    ):
        if len(noise_range) != 2 or noise_range[0] > noise_range[1]:
            raise ValueError(f"invalid {label}: {noise_range}")
    if object_xy_noise_range != (0.0, 0.0):
        object_pose[..., :2] += torch.empty(
            (*object_pose.shape[:2], 2),
            dtype=object_pose.dtype,
            device=object_pose.device,
        ).uniform_(*object_xy_noise_range)
    if object_yaw_noise_range != (0.0, 0.0):
        yaw = torch.empty(
            object_pose.shape[:2],
            dtype=object_pose.dtype,
            device=object_pose.device,
        ).uniform_(*object_yaw_noise_range)
        old_quat = object_pose[..., 3:7].clone()
        cosine = torch.cos(0.5 * yaw)
        sine = torch.sin(0.5 * yaw)
        # World-frame yaw: q_new = q_yaw * q_old, with quaternions in wxyz order.
        object_pose[..., 3] = cosine * old_quat[..., 0] - sine * old_quat[..., 3]
        object_pose[..., 4] = cosine * old_quat[..., 1] - sine * old_quat[..., 2]
        object_pose[..., 5] = cosine * old_quat[..., 2] + sine * old_quat[..., 1]
        object_pose[..., 6] = cosine * old_quat[..., 3] + sine * old_quat[..., 0]
    object_velocity = torch.zeros(
        object_pose.shape[0],
        object_pose.shape[1],
        6,
        device=object_pose.device,
        dtype=object_pose.dtype,
    )
    object_joint_pos = None
    if command.retargeted_object_articulation.numel() > 0:
        if hasattr(command, "get_frame"):
            object_joint_pos = command.get_frame(
                command.retargeted_object_articulation, env_ids, reset_ts
            )
        else:
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
