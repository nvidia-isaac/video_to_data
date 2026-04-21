"""Motion data loading for tracking commands.

Loads all motion data from a single Hive-partitioned parquet file produced
by the planner. The parquet contains body qpos, EE targets, hand keypoints,
contacts, object trajectory, and scene metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from .tracking_command_cfg import TrackingCommandCfg


@dataclass
class MotionData:
    """Container for all motion data loaded from planner parquet."""

    # Body motion (from planner)
    qpos_data: torch.Tensor  # (T, 3 + 4 + num_joints)
    object_pos_w: torch.Tensor  # (T, 3) — first object body
    object_quat_w: torch.Tensor  # (T, 4) — first object body

    # EE targets
    ee_pos_w: torch.Tensor | None = None  # (T, num_ee, 3)
    ee_quat_w: torch.Tensor | None = None  # (T, num_ee, 4)
    ee_link_ids: list[int] | None = None
    ee_link_names: list[str] | None = None

    # Hand data (from V2P retargeting, embedded in planner parquet)
    left_wrist_position: torch.Tensor | None = None  # (T, 3)
    left_wrist_wxyz: torch.Tensor | None = None  # (T, 4)
    right_wrist_position: torch.Tensor | None = None  # (T, 3)
    right_wrist_wxyz: torch.Tensor | None = None  # (T, 4)
    left_finger_joints: torch.Tensor | None = None  # (T, J_left)
    right_finger_joints: torch.Tensor | None = None  # (T, J_right)
    left_hand_frames: torch.Tensor | None = None  # (T, K, 7)
    right_hand_frames: torch.Tensor | None = None  # (T, K, 7)
    left_hand_frame_names: list[str] | None = None
    right_hand_frame_names: list[str] | None = None

    # Contact data (from V2P retargeting)
    left_link_contact_positions: torch.Tensor | None = None  # (T, N, 4)
    left_object_contact_positions: torch.Tensor | None = None  # (T, N, 4)
    right_link_contact_positions: torch.Tensor | None = None  # (T, N, 4)
    right_object_contact_positions: torch.Tensor | None = None  # (T, N, 4)

    # Contact normals (for wrench computation)
    left_link_contact_normals: torch.Tensor | None = None  # (T, N, 4)
    left_object_contact_normals: torch.Tensor | None = None  # (T, N, 4)
    right_link_contact_normals: torch.Tensor | None = None  # (T, N, 4)
    right_object_contact_normals: torch.Tensor | None = None  # (T, N, 4)

    # Contact part IDs (which object body each contact is on)
    left_object_contact_part_ids: torch.Tensor | None = None  # (T, N)
    right_object_contact_part_ids: torch.Tensor | None = None  # (T, N)

    # Object body data (multi-body)
    object_body_position: torch.Tensor | None = None  # (T, B, 3)
    object_body_wxyz: torch.Tensor | None = None  # (T, B, 4)

    # Binary contact labels (for force closure reward)
    left_hand_contact_active: torch.Tensor | None = None  # (T,)
    right_hand_contact_active: torch.Tensor | None = None  # (T,)

    # Object mesh radius (for wrench computation)
    object_mesh_radius: list[float] | None = None

    # Finger joint names (for reordering to robot joint order)
    left_finger_joint_names: list[str] | None = None
    right_finger_joint_names: list[str] | None = None

    # File-order joint names (for qpos reordering)
    file_joint_names: list[str] | None = None


def load_motion_data(
    cfg: TrackingCommandCfg,
    robot: Articulation,
    device: torch.device,
) -> MotionData:
    """Load motion data from a planner parquet file.

    Reads body qpos, EE targets, hand keypoints, contacts, and object
    trajectory from a single Hive-partitioned parquet.

    Args:
        cfg: The tracking command configuration with motion_file path.
        robot: The robot articulation (for joint/body lookups).
        device: Torch device for tensors.

    Returns:
        MotionData with all fields populated from the parquet.
    """
    motion_file = cfg.motion_file

    # Resolve Hive-partitioned directory to actual parquet file
    path = Path(motion_file)
    if path.is_dir():
        parquet_files = list(path.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {motion_file}")
        motion_file = str(parquet_files[0])

    data = pq.read_table(motion_file).to_pydict()

    # --- Body qpos ---
    qpos = np.array(data["qpos"][0], dtype=np.float32)  # (T, nq)
    qpos_layout = json.loads(data["qpos_layout"][0])
    joint_names = data.get("joint_names", [None])[0]

    # Decompose qpos into root + joints using layout
    root_pos_slice = slice(*qpos_layout["root_pos"])
    root_quat_slice = slice(*qpos_layout["root_quat_wxyz"])
    root_pos = qpos[:, root_pos_slice]  # (T, 3)
    root_quat = qpos[:, root_quat_slice]  # (T, 4) wxyz

    # Body joints: everything after root, before fingers
    body_start = qpos_layout.get("body_joints", [7, qpos.shape[1]])[0]

    # Build qpos_data in tracking format: [pos(3), quat(4), joints(N)]
    # Include all joints from body_start onward
    all_joints = qpos[:, body_start:]
    qpos_data = np.concatenate([root_pos, root_quat, all_joints], axis=1)

    # --- Object ---
    obj_pos = np.array(data["object_body_position"][0], dtype=np.float32)  # (T, B, 3)
    obj_quat = np.array(data["object_body_wxyz"][0], dtype=np.float32)  # (T, B, 4)
    # Single-body: squeeze
    if obj_pos.ndim == 3:
        obj_pos_single = obj_pos[:, 0]  # (T, 3)
        obj_quat_single = obj_quat[:, 0]  # (T, 4)
    else:
        obj_pos_single = obj_pos
        obj_quat_single = obj_quat

    # --- EE targets (derived from per-side wrist data or legacy ee_pos_w) ---
    ee_pos_w = None
    ee_quat_w = None
    ee_link_ids = None
    ee_link_names = None

    ee_names = data.get("ee_link_names", [None])[0]
    if ee_names:
        ee_link_ids, ee_link_names = robot.find_bodies(ee_names)

    if "ee_pos_w" in data:
        ee_pos_w = torch.tensor(
            np.array(data["ee_pos_w"][0], dtype=np.float32), device=device
        )
        ee_quat_w = torch.tensor(
            np.array(data["ee_quat_w"][0], dtype=np.float32), device=device
        )
    elif "robot_left_wrist_position" in data and "robot_right_wrist_position" in data:
        # Derive stacked EE from per-side wrist columns
        lp = np.array(data["robot_left_wrist_position"][0], dtype=np.float32)
        rp = np.array(data["robot_right_wrist_position"][0], dtype=np.float32)
        lq = np.array(data["robot_left_wrist_wxyz"][0], dtype=np.float32)
        rq = np.array(data["robot_right_wrist_wxyz"][0], dtype=np.float32)
        ee_pos_w = torch.tensor(np.stack([lp, rp], axis=1), device=device)
        ee_quat_w = torch.tensor(np.stack([lq, rq], axis=1), device=device)

    # --- Hand data ---
    def _load_optional(key: str, dtype: type = np.float32) -> torch.Tensor | None:
        val = data.get(key)
        if val is not None and val[0] is not None:
            arr = np.array(val[0], dtype=dtype)
            return torch.tensor(arr, device=device)
        return None

    left_wrist_pos = _load_optional("robot_left_wrist_position")
    left_wrist_wxyz = _load_optional("robot_left_wrist_wxyz")
    right_wrist_pos = _load_optional("robot_right_wrist_position")
    right_wrist_wxyz = _load_optional("robot_right_wrist_wxyz")
    left_finger_joints = _load_optional("robot_left_finger_joints")
    right_finger_joints = _load_optional("robot_right_finger_joints")
    left_frames = _load_optional("robot_left_frames")
    right_frames = _load_optional("robot_right_frames")

    left_frame_names = data.get("left_robot_frame_names", [None])[0]
    right_frame_names = data.get("right_robot_frame_names", [None])[0]

    # --- Contact data ---
    left_link_contacts = _load_optional("mano_left_link_contact_positions")
    left_obj_contacts = _load_optional("mano_left_object_contact_positions")
    right_link_contacts = _load_optional("mano_right_link_contact_positions")
    right_obj_contacts = _load_optional("mano_right_object_contact_positions")

    # Contact normals
    left_link_normals = _load_optional("mano_left_link_contact_normals")
    left_obj_normals = _load_optional("mano_left_object_contact_normals")
    right_link_normals = _load_optional("mano_right_link_contact_normals")
    right_obj_normals = _load_optional("mano_right_object_contact_normals")

    # Contact part IDs
    left_part_ids = _load_optional("mano_left_object_contact_part_ids", dtype=np.int64)
    right_part_ids = _load_optional(
        "mano_right_object_contact_part_ids", dtype=np.int64
    )

    # Binary contact labels
    left_contact_active = _load_optional("left_hand_contact_active")
    right_contact_active = _load_optional("right_hand_contact_active")

    # Object mesh radius (for wrench computation)
    object_mesh_radius = data.get("object_mesh_radius", [None])[0]

    # --- Finger joint names ---
    left_fj_names = data.get("left_robot_finger_joint_names", [None])[0]
    right_fj_names = data.get("right_robot_finger_joint_names", [None])[0]

    return MotionData(
        qpos_data=torch.tensor(qpos_data, device=device).float(),
        object_pos_w=torch.tensor(obj_pos_single, device=device).float(),
        object_quat_w=torch.tensor(obj_quat_single, device=device).float(),
        ee_pos_w=ee_pos_w,
        ee_quat_w=ee_quat_w,
        ee_link_ids=ee_link_ids,
        ee_link_names=ee_link_names,
        left_wrist_position=left_wrist_pos,
        left_wrist_wxyz=left_wrist_wxyz,
        right_wrist_position=right_wrist_pos,
        right_wrist_wxyz=right_wrist_wxyz,
        left_finger_joints=left_finger_joints,
        right_finger_joints=right_finger_joints,
        left_hand_frames=left_frames,
        right_hand_frames=right_frames,
        left_hand_frame_names=left_frame_names,
        right_hand_frame_names=right_frame_names,
        left_link_contact_positions=left_link_contacts,
        left_object_contact_positions=left_obj_contacts,
        right_link_contact_positions=right_link_contacts,
        right_object_contact_positions=right_obj_contacts,
        left_link_contact_normals=left_link_normals,
        left_object_contact_normals=left_obj_normals,
        right_link_contact_normals=right_link_normals,
        right_object_contact_normals=right_obj_normals,
        left_object_contact_part_ids=left_part_ids,
        right_object_contact_part_ids=right_part_ids,
        object_body_position=(
            torch.tensor(obj_pos, device=device).float() if obj_pos.ndim == 3 else None
        ),
        object_body_wxyz=(
            torch.tensor(obj_quat, device=device).float()
            if obj_quat.ndim == 3
            else None
        ),
        left_hand_contact_active=left_contact_active,
        right_hand_contact_active=right_contact_active,
        object_mesh_radius=object_mesh_radius,
        left_finger_joint_names=left_fj_names,
        right_finger_joint_names=right_fj_names,
        file_joint_names=joint_names,
    )
