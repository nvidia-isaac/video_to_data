# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def finger_contact_forces(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Contact force magnitudes for finger sensors.

    Args:
        env: The environment instance.
        sensor_cfg: If provided, return force for this specific sensor only.
            If None, return forces for all sensors in env.cfg.finger_sensor_names.

    Returns:
        If sensor_cfg is None: Tensor of shape (num_envs, num_fingers) with force magnitudes.
        If sensor_cfg is provided: Tensor of shape (num_envs, 1) with force magnitude.
    """
    if sensor_cfg is not None:
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        net_forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
        return torch.norm(net_forces, dim=-1)  # (num_envs, num_bodies)

    # Return all finger sensors
    sensor_names = env.cfg.finger_sensor_names
    force_list: list[torch.Tensor] = []
    for sensor_name in sensor_names:
        s: ContactSensor = env.scene[sensor_name]
        net_forces = s.data.net_forces_w  # (num_envs, 1, 3)
        force_magnitude = torch.norm(net_forces, dim=-1)  # (num_envs, 1)
        force_list.append(force_magnitude)

    return torch.cat(force_list, dim=-1)  # (num_envs, num_fingers)


def _get_contact_pos(env: ManagerBasedEnv, sensor: ContactSensor) -> torch.Tensor:
    """Helper to extract contact position from a sensor.

    Returns:
        Tensor of shape (num_envs, 1, 3) with contact position.
    """
    contact_pos_w = sensor.data.contact_pos_w
    # Return zeros if contact position tracking is not enabled
    if contact_pos_w is None:
        return torch.zeros(env.num_envs, 1, 3, device=env.device)
    # contact_pos_w shape: (num_envs, num_bodies, num_filter_bodies, 3)
    # Take first body and first filter body
    pos = contact_pos_w[:, 0, 0, :].unsqueeze(1)  # (num_envs, 1, 3)
    return torch.nan_to_num(pos, nan=0.0)


def finger_contact_positions(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Contact positions in world frame.

    Returns 0 for positions when there is no contact (NaN replaced with 0).

    Args:
        env: The environment instance.
        sensor_cfg: If provided, return position for this specific sensor only.
            If None, return positions for all sensors in env.cfg.finger_sensor_names.

    Note:
        Requires ContactSensorCfg.track_contact_points=True and
        ContactSensorCfg.max_contact_data_per_prim >= 1.

    Returns:
        If sensor_cfg is None: Tensor of shape (num_envs, num_fingers, 3) with contact positions.
        If sensor_cfg is provided: Tensor of shape (num_envs, 1, 3) with contact position.
    """
    if sensor_cfg is not None:
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        return _get_contact_pos(env, sensor)

    # Return all finger sensors
    sensor_names = env.cfg.finger_sensor_names
    pos_list: list[torch.Tensor] = []
    for sensor_name in sensor_names:
        s: ContactSensor = env.scene[sensor_name]
        pos_list.append(_get_contact_pos(env, s))

    return torch.cat(pos_list, dim=1)  # (num_envs, num_fingers, 3)


def total_contact_force(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Total contact force across all fingertips (num_envs, 1)."""
    force_magnitudes = finger_contact_forces(env, sensor_cfg)
    return force_magnitudes.sum(dim=-1, keepdim=True)


def finger_contact_force_vectors(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Contact force vectors for all finger sensors (num_envs, num_fingers, 3).

    Collects 3D force vectors from contact sensors (one per finger).
    Each sensor is filtered to report only contacts with the object.
    """
    sensor_names = env.cfg.finger_sensor_names
    force_list = []
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene[sensor_name]
        force_list.append(sensor.data.net_forces_w)  # (num_envs, 1, 3)

    # Concatenate to (num_envs, num_fingers, 3)
    return torch.cat(force_list, dim=1)


def contact_link_pos_and_valid(
    env: ManagerBasedEnv,
    side: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-link contact position and validity grouped by object part for one hand.

    Args:
        env: The environment instance.
        side: 'left' or 'right'.

    Returns:
        contact_pos: Tensor shaped ``(num_envs, 2, num_links, 3)`` with contact
            points in world frame for each part/link slot.
        contact_valid: Tensor shaped ``(num_envs, 2, num_links)`` where entries
            are ``True`` when the sensor force norm exceeds its threshold.
    """
    num_parts = 2
    attr = f"contact_link_sensor_names_{side}"
    sensor_names = list(getattr(env.cfg, attr, []))

    if len(sensor_names) == 0:
        num_envs = env.num_envs
        return (
            torch.zeros(num_envs, num_parts, 0, 3, device=env.device),
            torch.zeros(num_envs, num_parts, 0, dtype=torch.bool, device=env.device),
        )

    if len(sensor_names) % num_parts != 0:
        raise ValueError(
            f"Expected an even number of {side} contact-link sensors, got {len(sensor_names)}."
        )

    pos_list: list[torch.Tensor] = []
    valid_list: list[torch.Tensor] = []
    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene[sensor_name]
        pos_list.append(_get_contact_pos(env, sensor))  # (num_envs, 1, 3)
        force = torch.norm(sensor.data.net_forces_w, dim=-1)  # (num_envs, 1)
        threshold = getattr(getattr(sensor, "cfg", None), "force_threshold", 0.1)
        valid_list.append(force > threshold)
    pos_flat = torch.cat(pos_list, dim=1)  # (num_envs, 2 * num_links, 3)
    valid_flat = torch.cat(valid_list, dim=1)  # (num_envs, 2 * num_links)
    # Reorder to (part, link): part 0 uses indices [0, 2, ...], part 1 uses [1, 3, ...].
    part0_idx = torch.arange(0, len(sensor_names), num_parts, device=env.device)
    part1_idx = torch.arange(1, len(sensor_names), num_parts, device=env.device)
    contact_pos = torch.stack(
        [
            pos_flat[:, part0_idx, :],
            pos_flat[:, part1_idx, :],
        ],
        dim=1,
    )  # (num_envs, 2, num_links, 3)
    contact_valid = torch.stack(
        [
            valid_flat[:, part0_idx],
            valid_flat[:, part1_idx],
        ],
        dim=1,
    )  # (num_envs, 2, num_links)
    return contact_pos, contact_valid


def finger_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Finger joint positions.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, num_fingers) with finger joint positions.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            math_utils.scale_transform(
                command.right_hand_finger_joint_pos,
                command.right_robot.data.joint_pos_limits[..., 0],
                command.right_robot.data.joint_pos_limits[..., 1],
            ),
            math_utils.scale_transform(
                command.left_hand_finger_joint_pos,
                command.left_robot.data.joint_pos_limits[..., 0],
                command.left_robot.data.joint_pos_limits[..., 1],
            ),
        ],
        dim=-1,
    ).float()


def finger_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Finger joint velocity.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, num_fingers) with finger joint velocities.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.right_hand_finger_joint_vel,
            command.left_hand_finger_joint_vel,
        ],
        dim=-1,
    ).float()


def wrist_position_e(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Wrist position in the environment frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, 3 x NUM_HANDS) with wrist position in the environment frame.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.right_hand_wrist_position_e,
            command.left_hand_wrist_position_e,
        ],
        dim=-1,
    )


def wrist_orientation_e(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Wrist orientation in the environment frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, 4 x NUM_HANDS) with wrist orientation in the environment frame.
    """
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.right_hand_wrist_wxyz_e,
            command.left_hand_wrist_wxyz_e,
        ],
        dim=-1,
    )


def object_position_e(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object position in the environment frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.
    """
    command = env.command_manager.get_term(command_name)
    return command.object_position_e.reshape(command.num_envs, -1)


def object_orientation_e(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object orientation in the environment frame.

    Args:
        env: The environment instance.
        command_name: The name of the command.
    """
    command = env.command_manager.get_term(command_name)
    return command.object_orientation_e.reshape(command.num_envs, -1)


def object_t_wrist(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Wrist in object transformation.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, 7 x NUM_HANDS) transformation of wrist in object frame.
    """
    command = env.command_manager.get_term(command_name)

    object_p_right_wrist, object_q_right_wrist = math_utils.subtract_frame_transforms(
        command.object_position_e.squeeze(),
        command.object_orientation_e.squeeze(),  # world_t_object
        command.right_hand_wrist_position_e,
        command.right_hand_wrist_wxyz_e,  # world_t_wrist
    )
    object_p_left_wrist, object_q_left_wrist = math_utils.subtract_frame_transforms(
        command.object_position_e.squeeze(),
        command.object_orientation_e.squeeze(),  # world_t_object
        command.left_hand_wrist_position_e,
        command.left_hand_wrist_wxyz_e,  # world_t_wrist
    )

    return torch.cat(
        [
            object_p_right_wrist,
            object_q_right_wrist,
            object_p_left_wrist,
            object_q_left_wrist,
        ],
        dim=-1,
    ).float()


def object_p_fingertip(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Fingertips to object transformation fingertip_t_object.

    Args:
        env: The environment instance.
        command_name: The name of the command.

    Returns:
        Tensor of shape (num_envs, 7 x NUM_FINGERTIPS) with fingertips to object transformation.
    """
    command = env.command_manager.get_term(command_name)

    num_fingertips = len(command.right_fingertip_body_ids)

    object_position_e = command.object_position_e.expand(-1, num_fingertips, -1)
    object_orientation_e = command.object_orientation_e.expand(-1, num_fingertips, -1)

    object_p_right_fingertip, _ = math_utils.subtract_frame_transforms(
        object_position_e.reshape(-1, 3),
        object_orientation_e.reshape(-1, 4),  # world_t_object
        command.right_hand_fingertip_position_e.reshape(-1, 3),
        command.right_hand_fingertip_orientation_e.reshape(-1, 4),  # world_t_fingertip
    )
    object_p_right_fingertip = object_p_right_fingertip.reshape(env.num_envs, -1, 3)

    object_p_left_fingertip, _ = math_utils.subtract_frame_transforms(
        object_position_e.reshape(-1, 3),
        object_orientation_e.reshape(-1, 4),  # world_t_object
        command.left_hand_fingertip_position_e.reshape(-1, 3),
        command.left_hand_fingertip_orientation_e.reshape(-1, 4),  # world_t_fingertip
    )
    object_p_left_fingertip = object_p_left_fingertip.reshape(env.num_envs, -1, 3)

    return torch.cat(
        [
            object_p_right_fingertip.reshape(env.num_envs, -1),
            object_p_left_fingertip.reshape(env.num_envs, -1),
        ],
        dim=-1,
    ).float()
