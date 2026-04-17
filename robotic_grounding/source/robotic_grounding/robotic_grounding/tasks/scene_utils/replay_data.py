"""Schema-aware replay trajectory loading for scene playback scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from robotic_grounding.retarget.data_logger import (
    ManoSharpaData,
    NvhumanDex3Data,
    NvhumanG1Data,
)
from robotic_grounding.tasks.scene_utils.scene_config import SceneConfig


@dataclass
class ObjectTrajectory:
    """Object root trajectory in world coordinates."""

    root_position: np.ndarray  # (T, 3)
    root_wxyz: np.ndarray  # (T, 4)


@dataclass
class SingleRobotTrajectory:
    """Canonical replay data for a single whole-body robot."""

    schema: Literal["nvhuman_g1"]
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

    schema: Literal["mano_sharpa", "nvhuman_dex3"]
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
    # If the resolved path already points to the sequence/robot partition leaf,
    # partition filters can exclude all rows because partition columns may no
    # longer be materialized at that scope. In that case, load directly.
    parts = Path(resolved).parts
    has_seq_partition = any(p.startswith("sequence_id=") for p in parts)
    has_robot_partition = any(p.startswith("robot_name=") for p in parts)
    if has_seq_partition and has_robot_partition:
        filters = None
    return resolved, filters


def _build_object_traj(
    object_root_position: list[list[float]] | None,
    object_root_axis_angle: list[list[float]] | None,
) -> ObjectTrajectory | None:
    """Convert object root position + axis-angle arrays to replay trajectory."""
    if not object_root_position or not object_root_axis_angle:
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


def _is_g1_schema(columns: set[str]) -> bool:
    return {
        "robot_root_position",
        "robot_root_wxyz",
        "robot_joint_positions",
    }.issubset(columns)


def _is_sharpa_schema(columns: set[str]) -> bool:
    return {
        "robot_right_wrist_position",
        "robot_right_wrist_wxyz",
        "robot_right_finger_joints",
        "robot_left_wrist_position",
        "robot_left_wrist_wxyz",
        "robot_left_finger_joints",
    }.issubset(columns)


def _is_dex3_schema(columns: set[str]) -> bool:
    return {
        "robot_right_wrist_position",
        "robot_right_wrist_euler_xyz",
        "robot_right_finger_joints",
        "robot_left_wrist_position",
        "robot_left_wrist_euler_xyz",
        "robot_left_finger_joints",
    }.issubset(columns)


def _to_single_robot(data: Any) -> SingleRobotTrajectory:
    object_traj = _build_object_traj(
        object_root_position=getattr(data, "object_root_position", None),
        object_root_axis_angle=getattr(data, "object_root_axis_angle", None),
    )
    root_pos = np.asarray(data.robot_root_position, dtype=np.float32)
    root_wxyz = np.asarray(data.robot_root_wxyz, dtype=np.float32)
    joint_pos = np.asarray(data.robot_joint_positions, dtype=np.float32)
    return SingleRobotTrajectory(
        schema="nvhuman_g1",
        robot_layout="single_robot",
        fps=float(data.fps),
        num_frames=int(root_pos.shape[0]),
        robot_joint_names=list(data.robot_joint_names),
        robot_root_position=root_pos,
        robot_root_wxyz=root_wxyz,
        robot_joint_positions=joint_pos,
        object_traj=object_traj,
    )


def _to_dual_hand_wxyz(data: Any) -> DualHandTrajectory:
    object_traj = _build_object_traj(
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


def _to_dual_hand_euler(data: Any) -> DualHandTrajectory:
    object_traj = _build_object_traj(
        object_root_position=getattr(data, "object_root_position", None),
        object_root_axis_angle=getattr(data, "object_root_axis_angle", None),
    )
    right_pos = np.asarray(data.robot_right_wrist_position, dtype=np.float32)
    left_pos = np.asarray(data.robot_left_wrist_position, dtype=np.float32)
    return DualHandTrajectory(
        schema="nvhuman_dex3",
        robot_layout="dual_hand",
        fps=float(data.fps),
        num_frames=int(right_pos.shape[0]),
        right_joint_names=list(data.right_robot_finger_joint_names),
        left_joint_names=list(data.left_robot_finger_joint_names),
        right_wrist_position=right_pos,
        left_wrist_position=left_pos,
        wrist_orientation_format="euler_xyz",
        right_wrist_orientation=np.asarray(
            data.robot_right_wrist_euler_xyz, dtype=np.float32
        ),
        left_wrist_orientation=np.asarray(
            data.robot_left_wrist_euler_xyz, dtype=np.float32
        ),
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
    columns = set(pq.read_table(resolved).schema.names)

    if _is_g1_schema(columns):
        data = NvhumanG1Data.from_parquet(
            root_path=resolved,
            filters=filters,
            trajectory_id=trajectory_id,
        )
        return _to_single_robot(data)

    if _is_sharpa_schema(columns):
        data = ManoSharpaData.from_parquet(
            root_path=resolved,
            filters=filters,
            trajectory_id=trajectory_id,
        )
        return _to_dual_hand_wxyz(data)

    if _is_dex3_schema(columns):
        data = NvhumanDex3Data.from_parquet(
            root_path=resolved,
            filters=filters,
            trajectory_id=trajectory_id,
        )
        return _to_dual_hand_euler(data)

    raise ValueError(
        "Unsupported replay schema. Expected one of: "
        "{NvhumanG1Data, ManoSharpaData, NvhumanDex3Data}."
    )
