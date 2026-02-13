"""Utility functions for loading motion data in tracking commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from .tracking_command_cfg import TrackingCommandCfg


@dataclass
class MotionData:
    """Container for loaded motion data."""

    # Robot motion data
    qpos_data: torch.Tensor  # (T, 3 + 4 + num_joints) - [pos, quat, joints]
    object_pos_w: torch.Tensor  # (T, 3)
    object_quat_w: torch.Tensor  # (T, 4)

    # Optional EE motion data
    ee_pos_w: torch.Tensor | None = None  # (T, num_ee, 3)
    ee_quat_w: torch.Tensor | None = None  # (T, num_ee, 4)
    ee_link_ids: list[int] | None = None
    ee_link_names: list[str] | None = None


def load_motion_data(
    cfg: TrackingCommandCfg,
    robot: Articulation,
    device: torch.device,
) -> MotionData:
    """Load motion data from file.

    Dispatches to the appropriate loader based on file extension.

    Args:
        cfg: The tracking command configuration.
        robot: The robot articulation for joint info.
        device: The torch device to load tensors to.

    Returns:
        MotionData containing the loaded trajectories.
    """
    if cfg.motion_file.endswith(".h5"):
        return _load_h5_motion(cfg, robot, device)
    elif cfg.motion_file.endswith(".yaml"):
        return _load_yaml_motion(cfg, robot, device)
    elif cfg.motion_file.endswith(".parquet") and cfg.is_ee_motion:
        return _load_ee_parquet_motion(cfg, robot, device)
    else:
        raise ValueError(f"Unsupported file type: {cfg.motion_file}")


def _load_h5_motion(
    cfg: TrackingCommandCfg,
    robot: Articulation,
    device: torch.device,
) -> MotionData:
    """Load motion data from an HDF5 file."""
    with h5py.File(cfg.motion_file, "r") as f:
        qpos_data = torch.from_numpy(f["qpos"][()]).to(device)
        object_pos_w = torch.from_numpy(f[cfg.object_position_key][()]).to(device)
        object_quat_w = torch.from_numpy(f[cfg.object_quaternion_key][()]).to(device)

    return MotionData(
        qpos_data=qpos_data,
        object_pos_w=object_pos_w,
        object_quat_w=object_quat_w,
    )


def _load_yaml_motion(
    cfg: TrackingCommandCfg,
    robot: Articulation,
    device: torch.device,
) -> MotionData:
    """Load motion data from a YAML file."""
    with open(cfg.motion_file, "r") as f:
        data = yaml.safe_load(f)

    qpos_data = torch.tensor(data["qpos"]).to(device)
    object_pos_w = torch.tensor(data[cfg.object_position_key]).to(device)
    object_quat_w = torch.tensor(data[cfg.object_quaternion_key]).to(device)

    # Extract EE poses if specified
    ee_pos_w = None
    ee_quat_w = None
    ee_link_ids = None
    ee_link_names = None

    if cfg.ee_link_names:
        ee_link_ids, ee_link_names = robot.find_bodies(cfg.ee_link_names)
        ee_pos_w = torch.stack(
            [torch.tensor(data[ee_name + "_position"]) for ee_name in ee_link_names],
            dim=1,
        ).to(
            device
        )  # (T, num_ee_links, 3)
        ee_quat_w = torch.stack(
            [torch.tensor(data[ee_name + "_wxyz"]) for ee_name in ee_link_names], dim=1
        ).to(
            device
        )  # (T, num_ee_links, 4)

    return MotionData(
        qpos_data=qpos_data,
        object_pos_w=object_pos_w,
        object_quat_w=object_quat_w,
        ee_pos_w=ee_pos_w,
        ee_quat_w=ee_quat_w,
        ee_link_ids=ee_link_ids,
        ee_link_names=ee_link_names,
    )


def _load_ee_parquet_motion(
    cfg: TrackingCommandCfg,
    robot: Articulation,
    device: torch.device,
) -> MotionData:
    """Load motion data from a parquet file.

    Parquet files contain EE-based motion data:
    - nvhuman_root_translation / nvhuman_root_wxyz: root pose trajectory
    - robot_left_qpos / robot_right_qpos: 13 values per hand (6dof EE + 7 finger joints)
    - object_translation / object_axis_angle: object pose trajectory
    """
    table = pq.read_table(cfg.motion_file)
    data = table.to_pydict()

    # Get the first trajectory for now
    # Load root pose
    root_trans = np.array(data["nvhuman_root_translation"][0])  # (T, 3)
    root_wxyz = np.array(data["nvhuman_root_wxyz"][0])  # (T, 4)

    # Load object pose
    obj_trans = np.array(data["object_translation"][0])  # (T, 3)
    obj_axis_angle = np.array(data["object_axis_angle"][0])  # (T, 3)

    # Convert object axis-angle to quaternion (wxyz format)
    obj_quat = np.zeros((obj_axis_angle.shape[0], 4))
    for i, aa in enumerate(obj_axis_angle):
        rot = R.from_rotvec(aa)
        obj_quat[i] = rot.as_quat(scalar_first=True)

    num_timesteps = root_trans.shape[0]

    # Get robot's default joint positions to use for joints not in parquet
    default_joint_pos = robot.data.default_joint_pos[0].cpu().numpy()  # (num_joints,)

    # Initialize joint positions with default pose for all timesteps
    joint_pos = np.tile(default_joint_pos, (num_timesteps, 1))  # (T, num_joints)

    # Load hand qpos data from parquet
    left_qpos = (
        np.array(data.get("robot_left_qpos", [[]])[0])
        if "robot_left_qpos" in data
        else None
    )
    right_qpos = (
        np.array(data.get("robot_right_qpos", [[]])[0])
        if "robot_right_qpos" in data
        else None
    )

    # Use file_joint_names from config to determine joint mapping
    if cfg.file_joint_names is not None:
        left_joint_names = cfg.file_joint_names
        right_joint_names = [
            name.replace("left_", "right_") for name in cfg.file_joint_names
        ]

        # Get all robot joint names for matching
        all_robot_joint_names = robot.joint_names

        # Map parquet data to robot joints (skip joints not found on robot)
        if left_qpos is not None and len(left_qpos) > 0:
            for parquet_idx, joint_name in enumerate(left_joint_names):
                if parquet_idx >= left_qpos.shape[1]:
                    break
                if joint_name in all_robot_joint_names:
                    robot_joint_idx = all_robot_joint_names.index(joint_name)
                    joint_pos[:, robot_joint_idx] = left_qpos[:, parquet_idx]

        if right_qpos is not None and len(right_qpos) > 0:
            for parquet_idx, joint_name in enumerate(right_joint_names):
                if parquet_idx >= right_qpos.shape[1]:
                    break
                if joint_name in all_robot_joint_names:
                    robot_joint_idx = all_robot_joint_names.index(joint_name)
                    joint_pos[:, robot_joint_idx] = right_qpos[:, parquet_idx]
    elif left_qpos is not None and len(left_qpos) > 0:
        num_parquet_joints = left_qpos.shape[1]
        joint_pos[:, :num_parquet_joints] = left_qpos
    elif right_qpos is not None and len(right_qpos) > 0:
        num_parquet_joints = right_qpos.shape[1]
        joint_pos[:, :num_parquet_joints] = right_qpos

    # Construct qpos_data: [pos(3), quat(4), joints(N)]
    qpos_data = np.concatenate([root_trans, root_wxyz, joint_pos], axis=1)

    # Load EE poses for tracking from parquet data
    ee_pos_w = None
    ee_quat_w = None
    ee_link_ids = None
    ee_link_names = None

    if cfg.ee_link_names:
        ee_link_ids, ee_link_names = robot.find_bodies(cfg.ee_link_names)

        ee_pos_list = []
        ee_quat_list = []

        for ee_name in ee_link_names:
            # Map EE link name to parquet column
            if "left" in ee_name.lower():
                ee_qpos = (
                    left_qpos
                    if left_qpos is not None
                    else np.zeros((num_timesteps, 13))
                )
            elif "right" in ee_name.lower():
                ee_qpos = (
                    right_qpos
                    if right_qpos is not None
                    else np.zeros((num_timesteps, 13))
                )
            else:
                # Default to left hand
                ee_qpos = (
                    left_qpos
                    if left_qpos is not None
                    else np.zeros((num_timesteps, 13))
                )

            # Palm pose from floating base joints
            ee_pos = ee_qpos[:, 0:3]  # (T, 3) - base_x, base_y, base_z
            ee_euler = ee_qpos[:, 3:6]  # (T, 3) - base_roll, base_pitch, base_yaw

            # Convert euler to quaternion (wxyz)
            # Use intrinsic XYZ (body-frame rotations) to match URDF joint chain
            ee_quat = np.zeros((num_timesteps, 4))
            for i, euler in enumerate(ee_euler):
                rot = R.from_euler("XYZ", euler)
                ee_quat[i] = rot.as_quat(scalar_first=True)

            ee_pos_list.append(ee_pos)
            ee_quat_list.append(ee_quat)

        ee_pos_w = torch.tensor(
            np.stack(ee_pos_list, axis=1), device=device
        ).float()  # (T, num_ee, 3)
        ee_quat_w = torch.tensor(
            np.stack(ee_quat_list, axis=1), device=device
        ).float()  # (T, num_ee, 4)

        # Apply robot anchor offset to EE positions (same offset as root pose)
        if cfg.robot_anchor_pos_offset:
            offset = torch.tensor(cfg.robot_anchor_pos_offset, device=device).float()
            ee_pos_w = ee_pos_w + offset.unsqueeze(0).unsqueeze(0)  # (T, num_ee, 3)

    # Convert to tensors
    qpos_data = torch.tensor(qpos_data, device=device).float()
    object_pos_w = torch.tensor(obj_trans, device=device).float()
    object_quat_w = torch.tensor(obj_quat, device=device).float()

    return MotionData(
        qpos_data=qpos_data,
        object_pos_w=object_pos_w,
        object_quat_w=object_quat_w,
        ee_pos_w=ee_pos_w,
        ee_quat_w=ee_quat_w,
        ee_link_ids=ee_link_ids,
        ee_link_names=ee_link_names,
    )
