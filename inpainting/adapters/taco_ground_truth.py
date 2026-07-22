"""Convert a processed TACO ManoSharpa parquet to common tracking contracts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from inpainting.contracts import (
    ROBOT_TRAJECTORY_SCHEMA,
    TRACKING_SCHEMA,
    validate_robot_trajectory_arrays,
    validate_tracking_arrays,
)


REQUIRED_COLUMNS = [
    "sequence_id",
    "robot_name",
    "fps",
    "mano_left_joints",
    "mano_right_joints",
    "mano_left_joints_wxyz",
    "mano_right_joints_wxyz",
    "left_robot_finger_joint_names",
    "right_robot_finger_joint_names",
    "robot_left_wrist_position",
    "robot_right_wrist_position",
    "robot_left_wrist_wxyz",
    "robot_right_wrist_wxyz",
    "robot_left_finger_joints",
    "robot_right_finger_joints",
]


def _load_row(path: Path) -> dict:
    table = pq.read_table(path, columns=REQUIRED_COLUMNS)
    if table.num_rows != 1:
        raise ValueError(f"Expected one sequence row in {path}, got {table.num_rows}")
    return table.to_pylist()[0]


def arrays_from_row(row: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    left_joints = np.asarray(row["mano_left_joints"], dtype=np.float32)
    right_joints = np.asarray(row["mano_right_joints"], dtype=np.float32)
    left_joint_quats = np.asarray(row["mano_left_joints_wxyz"], dtype=np.float32)
    right_joint_quats = np.asarray(row["mano_right_joints_wxyz"], dtype=np.float32)
    frame_count = left_joints.shape[0]
    if right_joints.shape[0] != frame_count:
        raise ValueError("Left/right MANO frame counts differ")
    left_valid = np.isfinite(left_joints).all(axis=(1, 2)) & np.isfinite(
        left_joint_quats
    ).all(axis=(1, 2))
    right_valid = np.isfinite(right_joints).all(axis=(1, 2)) & np.isfinite(
        right_joint_quats
    ).all(axis=(1, 2))

    robot_targets = {
        side: {
            "position": np.asarray(row[f"robot_{side}_wrist_position"], dtype=np.float32),
            "quaternion": np.asarray(row[f"robot_{side}_wrist_wxyz"], dtype=np.float32),
            "fingers": np.asarray(row[f"robot_{side}_finger_joints"], dtype=np.float32),
        }
        for side in ("left", "right")
    }

    def robot_valid(side: str) -> np.ndarray:
        fields = robot_targets[side]
        valid = np.ones(frame_count, dtype=np.bool_)
        for name, values in fields.items():
            if values.ndim < 2 or values.shape[0] != frame_count:
                raise ValueError(
                    f"robot_{side}_{name} must have {frame_count} rows, got {values.shape}"
                )
            valid &= np.isfinite(values).reshape(frame_count, -1).all(axis=1)
        return valid

    robot_left_valid = robot_valid("left")
    robot_right_valid = robot_valid("right")
    tracking = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray("ground_truth"),
        "coordinate_frame": np.asarray("world"),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
        "left_valid": left_valid,
        "right_valid": right_valid,
        "left_wrist_position": left_joints[:, 0],
        "right_wrist_position": right_joints[:, 0],
        "left_wrist_wxyz": left_joint_quats[:, 0],
        "right_wrist_wxyz": right_joint_quats[:, 0],
        "left_joints_3d": left_joints,
        "right_joints_3d": right_joints,
    }
    trajectory = {
        "schema_version": np.asarray(ROBOT_TRAJECTORY_SCHEMA),
        "coordinate_frame": np.asarray("world"),
        "robot": np.asarray("dexmate_vega"),
        "gripper": np.asarray(row["robot_name"]),
        "frame_indices": np.arange(frame_count, dtype=np.int32),
        "left_valid": robot_left_valid,
        "right_valid": robot_right_valid,
        "left_wrist_position": robot_targets["left"]["position"],
        "right_wrist_position": robot_targets["right"]["position"],
        "left_wrist_wxyz": robot_targets["left"]["quaternion"],
        "right_wrist_wxyz": robot_targets["right"]["quaternion"],
        "left_finger_joints": robot_targets["left"]["fingers"],
        "right_finger_joints": robot_targets["right"]["fingers"],
        "left_finger_joint_names": np.asarray(row["left_robot_finger_joint_names"]),
        "right_finger_joint_names": np.asarray(row["right_robot_finger_joint_names"]),
    }
    validate_tracking_arrays(tracking)
    validate_robot_trajectory_arrays(trajectory, expected_frames=frame_count)
    return tracking, trajectory


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.partial.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def convert(
    parquet: Path,
    tracking_output: Path,
    trajectory_output: Path,
    metadata_output: Path | None = None,
) -> dict:
    row = _load_row(parquet)
    tracking, trajectory = arrays_from_row(row)
    _atomic_savez(tracking_output, tracking)
    _atomic_savez(trajectory_output, trajectory)
    metadata = {
        "schema_version": "v2d.inpainting.adapter-run/v1",
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "adapter": "taco_ground_truth",
        "sequence_id": row["sequence_id"],
        "fps": float(row["fps"]),
        "frame_count": int(tracking["frame_indices"].shape[0]),
        "coordinate_frame": "world",
        "source_parquet": str(parquet.resolve()),
        "tracking_output": str(tracking_output.resolve()),
        "trajectory_output": str(trajectory_output.resolve()),
        "camera_projection_status": "not_evaluated_by_adapter",
        "camera_projection_note": (
            "Validate with the matching TACO egocentric calibration as a separate stage"
        ),
    }
    if metadata_output is None:
        metadata_output = tracking_output.with_suffix(".json")
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--tracking-output", required=True, type=Path)
    parser.add_argument("--trajectory-output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    metadata = convert(
        args.parquet, args.tracking_output, args.trajectory_output, args.metadata_output
    )
    print(
        f"Converted {metadata['sequence_id']} ({metadata['frame_count']} frames) -> "
        f"{metadata['tracking_output']}"
    )


if __name__ == "__main__":
    main()
