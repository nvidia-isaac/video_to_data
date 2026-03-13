# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Debug reward functions for visualizing sensor data."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor

# Axis name to index mapping
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# =============================================================================
# Contact Force Functions
# =============================================================================


def contact_force(
    env: ManagerBasedRLEnv,
    sensor_name: str = "right_pinky_contact_sensor",
    axis: str = "x",
) -> torch.Tensor:
    """Return the specified component of contact force in world frame.

    Args:
        env: The RL environment.
        sensor_name: Name of the contact sensor.
        axis: Which axis to return ('x', 'y', or 'z').

    Returns:
        The force component along the specified axis. Shape: (num_envs,)
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of: 'x', 'y', 'z'")
    sensor: ContactSensor = env.scene[sensor_name]
    net_forces_w = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    total_force = net_forces_w.sum(dim=1)  # (num_envs, 3)
    return total_force[:, _AXIS_INDEX[axis]]  # (num_envs,)


# =============================================================================
# Contact Position Functions (actual contact point in world frame)
# =============================================================================


def contact_pos(
    env: ManagerBasedRLEnv,
    sensor_name: str = "right_pinky_contact_sensor",
    axis: str = "x",
) -> torch.Tensor:
    """Return the specified position component of the actual contact point in world frame.

    Uses contact_pos_w which is the average position of contact points.
    Returns 0 when there is no contact (NaN replaced with 0).

    Args:
        env: The RL environment.
        sensor_name: Name of the contact sensor.
        axis: Which axis to return ('x', 'y', or 'z').

    Returns:
        The position component along the specified axis. Shape: (num_envs,)

    Note:
        Requires ContactSensorCfg.track_contact_points=True and
        ContactSensorCfg.max_contact_data_per_prim >= 1.
    """
    if axis not in _AXIS_INDEX:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of: 'x', 'y', 'z'")
    sensor: ContactSensor = env.scene[sensor_name]
    contact_pos_w = sensor.data.contact_pos_w
    # Return zeros if contact position tracking is not enabled
    if contact_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    # contact_pos_w shape: (num_envs, num_bodies, num_filter_bodies, 3)
    # Take first body and first filter body, specified axis component
    pos = contact_pos_w[:, 0, 0, _AXIS_INDEX[axis]]  # (num_envs,)
    return torch.nan_to_num(pos, nan=0.0)
