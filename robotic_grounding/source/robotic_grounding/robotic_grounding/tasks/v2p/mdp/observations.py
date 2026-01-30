from __future__ import annotations

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
