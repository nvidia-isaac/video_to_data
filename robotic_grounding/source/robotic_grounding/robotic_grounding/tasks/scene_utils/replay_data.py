"""Schema-aware replay trajectory loading for scene playback scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from robotic_grounding.motion_schema import (
    DUAL_HAND,
    SCHEMA_VERSION,
    SINGLE_ROBOT,
    MotionData,
    load_motion_data_parquet,
)
from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.tasks.scene_utils.scene_config import SceneConfig


@dataclass
class ObjectTrajectory:
    """Object root trajectory in world coordinates."""

    root_position: np.ndarray  # (T, 3)
    root_wxyz: np.ndarray  # (T, 4)


@dataclass
class SingleRobotTrajectory:
    """Canonical replay data for a single whole-body robot."""

    schema: Literal["motion_v1"]
    robot_layout: Literal["single_robot"]
    fps: float
    num_frames: int
    robot_joint_names: list[str]
    robot_root_position: np.ndarray  # (T, 3)
    robot_root_wxyz: np.ndarray  # (T, 4)
    robot_joint_positions: np.ndarray  # (T, N)
    object_traj: ObjectTrajectory | None


@dataclass
class DualHandTrajectory:
    """Canonical replay data for dual floating-hand robots."""

    schema: Literal["mano_sharpa", "motion_v1"]
    robot_layout: Literal["dual_hand"]
    fps: float
    num_frames: int
    right_joint_names: list[str]
    left_joint_names: list[str]
    right_wrist_position: np.ndarray  # (T, 3)
    left_wrist_position: np.ndarray  # (T, 3)
    wrist_orientation_format: Literal["wxyz", "euler_xyz"]
    right_wrist_orientation: np.ndarray  # (T, 4) or (T, 3)
    left_wrist_orientation: np.ndarray  # (T, 4) or (T, 3)
    right_finger_joints: np.ndarray
    left_finger_joints: np.ndarray
    object_traj: ObjectTrajectory | None


ReplayTrajectory = SingleRobotTrajectory | DualHandTrajectory


def _resolve_path_and_filters(
    motion_file: str,
) -> tuple[str, list[tuple[str, str, str]] | None]:
    """Resolve path and infer deterministic partition filters."""
    resolved = SceneConfig._resolve_motion_file(motion_file)
    partition = SceneConfig._parse_partition_path(resolved)
    filters = partition.get("motion_filters")
    parts = Path(resolved).parts
    has_seq_partition = any(p.startswith("sequence_id=") for p in parts)
    has_robot_partition = any(p.startswith("robot_name=") for p in parts)
    if has_seq_partition and has_robot_partition:
        filters = None
    return resolved, filters


def _build_object_traj_from_arrays(
    object_root_position: Any,
    object_root_axis_angle: Any,
) -> ObjectTrajectory | None:
    """Convert object root position + axis-angle arrays to replay trajectory."""
    if object_root_position is None or object_root_axis_angle is None:
        return None
    pos = np.asarray(object_root_position, dtype=np.float32)
    aa = np.asarray(object_root_axis_angle, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        return None
    if aa.ndim != 2 or aa.shape[1] != 3:
        return None
    root_wxyz = np.asarray(
        [R.from_rotvec(v).as_quat(scalar_first=True) for v in aa],
        dtype=np.float32,
    )
    return ObjectTrajectory(root_position=pos, root_wxyz=root_wxyz)


def _to_np(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _motion_v1_to_replay(md: MotionData) -> ReplayTrajectory:
    """Map a `motion_v1` MotionData into the appropriate replay trajectory shape.

    Branches on the file's explicit `motion_kind`:
    - `single_robot` → `SingleRobotTrajectory` driven by whole-body joint state.
    - `dual_hand`    → `DualHandTrajectory` driven by `ee_pose_w`.
    """
    object_traj = _build_object_traj_from_arrays(
        object_root_position=_to_np(md.object_root_position),
        object_root_axis_angle=_to_np(md.object_root_axis_angle),
    )

    if md.motion_kind == SINGLE_ROBOT:
        root_pos = _to_np(md.robot_root_position)
        root_wxyz = _to_np(md.robot_root_wxyz)
        joint_pos = _to_np(md.robot_joint_positions)
        if root_pos is None or root_wxyz is None or joint_pos is None:
            raise ValueError(
                "motion_v1 single_robot file is missing required whole-body "
                "joint tensors. Re-run the producing retarget/planner script."
            )
        return SingleRobotTrajectory(
            schema="motion_v1",
            robot_layout="single_robot",
            fps=float(md.fps),
            num_frames=int(root_pos.shape[0]),
            robot_joint_names=list(md.robot_joint_names),
            robot_root_position=root_pos.astype(np.float32),
            robot_root_wxyz=root_wxyz.astype(np.float32),
            robot_joint_positions=joint_pos.astype(np.float32),
            object_traj=object_traj,
        )

    if md.motion_kind != DUAL_HAND:
        raise ValueError(
            f"Cannot build replay trajectory: unsupported motion_kind={md.motion_kind!r}."
        )

    ee_pose = _to_np(md.ee_pose_w)
    if ee_pose is None or ee_pose.ndim != 3 or ee_pose.shape[1] < 2:
        raise ValueError(
            "motion_v1 dual_hand file has fewer than 2 end-effector poses; cannot build a replay trajectory."
        )
    left_idx, right_idx = 0, 1
    names = md.ee_link_names or []
    for i, name in enumerate(names):
        lname = (name or "").lower()
        if "left" in lname:
            left_idx = i
        elif "right" in lname:
            right_idx = i
    left_pos = ee_pose[:, left_idx, 0:3]
    left_quat = ee_pose[:, left_idx, 3:7]
    right_pos = ee_pose[:, right_idx, 0:3]
    right_quat = ee_pose[:, right_idx, 3:7]

    left_fj_arr = _to_np(md.left_finger_joints)
    right_fj_arr = _to_np(md.right_finger_joints)
    left_fj_names = md.left_finger_joint_names or []
    right_fj_names = md.right_finger_joint_names or []

    return DualHandTrajectory(
        schema="motion_v1",
        robot_layout="dual_hand",
        fps=float(md.fps),
        num_frames=int(right_pos.shape[0]),
        right_joint_names=list(right_fj_names),
        left_joint_names=list(left_fj_names),
        right_wrist_position=right_pos.astype(np.float32),
        left_wrist_position=left_pos.astype(np.float32),
        wrist_orientation_format="wxyz",
        right_wrist_orientation=right_quat.astype(np.float32),
        left_wrist_orientation=left_quat.astype(np.float32),
        right_finger_joints=(
            right_fj_arr.astype(np.float32)
            if right_fj_arr is not None
            else np.zeros((0,), dtype=np.float32)
        ),
        left_finger_joints=(
            left_fj_arr.astype(np.float32)
            if left_fj_arr is not None
            else np.zeros((0,), dtype=np.float32)
        ),
        object_traj=object_traj,
    )


def _sharpa_to_dual_hand(data: ManoSharpaData) -> DualHandTrajectory:
    object_traj = _build_object_traj_from_arrays(
        object_root_position=getattr(data, "object_root_position", None),
        object_root_axis_angle=getattr(data, "object_root_axis_angle", None),
    )
    right_pos = np.asarray(data.robot_right_wrist_position, dtype=np.float32)
    left_pos = np.asarray(data.robot_left_wrist_position, dtype=np.float32)
    return DualHandTrajectory(
        schema="mano_sharpa",
        robot_layout="dual_hand",
        fps=float(data.fps),
        num_frames=int(right_pos.shape[0]),
        right_joint_names=list(data.right_robot_finger_joint_names),
        left_joint_names=list(data.left_robot_finger_joint_names),
        right_wrist_position=right_pos,
        left_wrist_position=left_pos,
        wrist_orientation_format="wxyz",
        right_wrist_orientation=np.asarray(
            data.robot_right_wrist_wxyz, dtype=np.float32
        ),
        left_wrist_orientation=np.asarray(data.robot_left_wrist_wxyz, dtype=np.float32),
        right_finger_joints=np.asarray(
            data.robot_right_finger_joints, dtype=np.float32
        ),
        left_finger_joints=np.asarray(data.robot_left_finger_joints, dtype=np.float32),
        object_traj=object_traj,
    )


def load_replay_trajectory(
    motion_file: str,
    trajectory_id: int = 0,
) -> ReplayTrajectory:
    """Load replay trajectory from a motion parquet path for known schemas."""
    resolved, filters = _resolve_path_and_filters(motion_file)

    resolved_path = Path(resolved)
    if resolved_path.is_dir():
        matches = list(resolved_path.rglob("*.parquet"))
        if not matches:
            raise FileNotFoundError(f"No parquet files under {resolved_path}")
        first_file = matches[0]
    else:
        first_file = resolved_path
    columns = set(pq.ParquetFile(str(first_file)).schema_arrow.names)

    # motion_v1 is the unified format; if the file declares it, use the reader.
    if "schema_version" in columns:
        md = load_motion_data_parquet(resolved)
        if md.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported motion schema version: {md.schema_version!r}. "
                f"Run scripts/motion_schema/migrate_to_v1.py to upgrade."
            )
        return _motion_v1_to_replay(md)

    # Legacy: ManoSharpaData (dual-hand V2P pipeline, not yet on motion_v1).
    if {
        "robot_right_wrist_position",
        "robot_right_wrist_wxyz",
        "robot_right_finger_joints",
        "robot_left_wrist_position",
        "robot_left_wrist_wxyz",
        "robot_left_finger_joints",
    }.issubset(columns):
        data = ManoSharpaData.from_parquet(
            root_path=resolved,
            filters=filters,
            trajectory_id=trajectory_id,
        )
        return _sharpa_to_dual_hand(data)

    raise ValueError(
        "Unsupported replay schema. Expected motion_v1 (run migrate_to_v1.py on "
        "legacy G1/Dex3 parquets) or ManoSharpaData."
    )
