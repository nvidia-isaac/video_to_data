# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from dataclasses import MISSING, field, fields, make_dataclass
from typing import Any, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Type alias for field specification
# Format: (field_name, pyarrow_type, python_type, is_time_series)
FieldSpec = tuple[str, pa.DataType, type, bool]

#############################################################
# Base fields
#############################################################
BASE_FIELDS: list[FieldSpec] = [
    ("sequence_id", pa.string(), str, False),
    ("raw_motion_file", pa.string(), str, False),
    ("robot_name", pa.string(), str, False),
    ("fps", pa.float32(), float, False),
]

#############################################################
# Body model fields
#############################################################
MANO_FIELDS: list[FieldSpec] = [
    ("mano_to_robot_scale", pa.float32(), float, False),
    ("mano_right_betas", pa.list_(pa.float32()), list[float], False),
    ("mano_left_betas", pa.list_(pa.float32()), list[float], False),
    ("mano_link_names", pa.list_(pa.string()), list[str], False),
    # Time series
    ("mano_right_trans", pa.list_(pa.list_(pa.float32(), 3)), list[list[float]], True),
    (
        "mano_right_global_orient",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "mano_right_finger_pose",
        pa.list_(pa.list_(pa.float32(), 45)),
        list[list[float]],
        True,
    ),
    (
        "mano_right_joints",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 21)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_right_joints_wxyz",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), 21)),
        list[list[list[float]]],
        True,
    ),
    ("mano_right_fitting_err", pa.list_(pa.float32()), list[float], True),
    (
        "mano_right_tips_distance",
        pa.list_(pa.list_(pa.float32(), 5)),
        list[list[float]],
        True,
    ),
    (
        "mano_right_link_contact_positions",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_right_object_contact_positions",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), 16)),
        list[list[list[float]]],
        True,
    ),
    # MANO left hand (time series)
    ("mano_left_trans", pa.list_(pa.list_(pa.float32(), 3)), list[list[float]], True),
    (
        "mano_left_global_orient",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "mano_left_finger_pose",
        pa.list_(pa.list_(pa.float32(), 45)),
        list[list[float]],
        True,
    ),
    (
        "mano_left_joints",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 21)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_left_joints_wxyz",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), 21)),
        list[list[list[float]]],
        True,
    ),
    ("mano_left_fitting_err", pa.list_(pa.float32()), list[float], True),
    (
        "mano_left_tips_distance",
        pa.list_(pa.list_(pa.float32(), 5)),
        list[list[float]],
        True,
    ),
    (
        "mano_left_link_contact_positions",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_left_object_contact_positions",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), 16)),
        list[list[list[float]]],
        True,
    ),
]

NVHUMAN_NUM_JOINTS = 93
NVHUMAN_FIELDS: list[FieldSpec] = [
    ("nvhuman_betas", pa.list_(pa.float32()), list[float], False),
    # Time series
    (
        "nvhuman_joints",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), NVHUMAN_NUM_JOINTS)),
        list[list[list[float]]],
        True,
    ),
    (
        "nvhuman_joints_wxyz",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), NVHUMAN_NUM_JOINTS)),
        list[list[list[float]]],
        True,
    ),
    # Head trajectory
    (
        "nvhuman_head_translation",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "nvhuman_head_wxyz",
        pa.list_(pa.list_(pa.float32(), 4)),
        list[list[float]],
        True,
    ),
    # Root trajectory
    (
        "nvhuman_root_translation",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "nvhuman_root_wxyz",
        pa.list_(pa.list_(pa.float32(), 4)),
        list[list[float]],
        True,
    ),
]

#############################################################
# Robot fields
#############################################################
SHARPA_NUM_FRAMES = 67
SHARPA_FIELDS: list[FieldSpec] = [
    ("right_robot_finger_joint_names", pa.list_(pa.string()), list[str], False),
    ("right_robot_frame_names", pa.list_(pa.string()), list[str], False),
    ("right_robot_frame_task_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_finger_joint_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_frame_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_frame_task_names", pa.list_(pa.string()), list[str], False),
    ("object_body_names", pa.list_(pa.string()), list[str], False),
    # Time series
    (
        "robot_right_wrist_position",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "robot_right_wrist_wxyz",
        pa.list_(pa.list_(pa.float32(), 4)),
        list[list[float]],
        True,
    ),
    (
        "robot_right_finger_joints",
        pa.list_(pa.list_(pa.float32(), 22)),
        list[list[float]],
        True,
    ),
    (
        "robot_right_frames",
        pa.list_(pa.list_(pa.list_(pa.float32(), 7), SHARPA_NUM_FRAMES)),
        list[list[list[float]]],
        True,
    ),
    (
        "robot_right_frame_task_errors",
        pa.list_(pa.list_(pa.float32(), 11)),
        list[list[float]],
        True,
    ),
    ("robot_right_num_optimization_iterations", pa.list_(pa.int32()), list[int], True),
    (
        "robot_left_wrist_position",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "robot_left_wrist_wxyz",
        pa.list_(pa.list_(pa.float32(), 4)),
        list[list[float]],
        True,
    ),
    (
        "robot_left_finger_joints",
        pa.list_(pa.list_(pa.float32(), 22)),
        list[list[float]],
        True,
    ),
    (
        "robot_left_frames",
        pa.list_(pa.list_(pa.list_(pa.float32(), 7), SHARPA_NUM_FRAMES)),
        list[list[list[float]]],
        True,
    ),
    (
        "robot_left_frame_task_errors",
        pa.list_(pa.list_(pa.float32(), 11)),
        list[list[float]],
        True,
    ),
    ("robot_left_num_optimization_iterations", pa.list_(pa.int32()), list[int], True),
]

DEX3_NUM_FINGER_JOINTS = 7
DEX3_NUM_FRAMES = 36
DEX3_FIELDS: list[FieldSpec] = [
    ("right_robot_finger_joint_names", pa.list_(pa.string()), list[str], False),
    ("right_robot_frame_names", pa.list_(pa.string()), list[str], False),
    ("right_robot_frame_task_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_finger_joint_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_frame_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_frame_task_names", pa.list_(pa.string()), list[str], False),
    # Time series
    (
        "robot_right_wrist_position",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "robot_right_wrist_euler_xyz",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "robot_right_finger_joints",
        pa.list_(pa.list_(pa.float32(), DEX3_NUM_FINGER_JOINTS)),
        list[list[float]],
        True,
    ),
    (
        "robot_right_frames",
        pa.list_(pa.list_(pa.list_(pa.float32(), 7), DEX3_NUM_FRAMES)),
        list[list[list[float]]],
        True,
    ),
    (
        "robot_right_frame_task_errors",
        pa.list_(pa.list_(pa.float32())),
        list[list[float]],
        True,
    ),
    ("robot_right_ik_error", pa.list_(pa.float32()), list[float], True),
    ("robot_right_num_optimization_iterations", pa.list_(pa.int32()), list[int], True),
    (
        "robot_left_wrist_position",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "robot_left_wrist_euler_xyz",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "robot_left_finger_joints",
        pa.list_(pa.list_(pa.float32(), DEX3_NUM_FINGER_JOINTS)),
        list[list[float]],
        True,
    ),
    (
        "robot_left_frames",
        pa.list_(pa.list_(pa.list_(pa.float32(), 7), DEX3_NUM_FRAMES)),
        list[list[list[float]]],
        True,
    ),
    (
        "robot_left_frame_task_errors",
        pa.list_(pa.list_(pa.float32())),
        list[list[float]],
        True,
    ),
    ("robot_left_ik_error", pa.list_(pa.float32()), list[float], True),
    ("robot_left_num_optimization_iterations", pa.list_(pa.int32()), list[int], True),
]

#############################################################
# Object fields
#############################################################
OBJECT_FIELDS: list[FieldSpec] = [
    ("object_name", pa.string(), str, False),
    # Time series
    ("object_articulation", pa.list_(pa.float32()), list[float], True),
    (
        "object_root_axis_angle",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "object_root_position",
        pa.list_(pa.list_(pa.float32(), 3)),
        list[list[float]],
        True,
    ),
    (
        "object_body_position",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3))),
        list[list[list[float]]],
        True,
    ),
    (
        "object_body_wxyz",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4))),
        list[list[list[float]]],
        True,
    ),
]


#############################################################
# Data logger class factory
#############################################################
def create_data_logger_class(
    name: str,
    field_specs: list[FieldSpec],
) -> type:
    """Create a data logger class from field specifications.

    Args:
        name: Name of the generated class.
        field_specs: List of field specifications to include.

    Returns:
        A dataclass type with logging and serialization methods.
    """
    # Build schema
    schema = pa.schema([(name, pa_type) for name, pa_type, _, _ in field_specs])

    # Sort fields, keep metadata before time-series
    sorted_specs = sorted(field_specs, key=lambda x: x[3])

    # Build dataclass fields
    dataclass_fields: list[tuple[str, Any, Any]] = []
    for field_name, _, py_type, is_time_series in sorted_specs:
        if is_time_series:
            dataclass_fields.append((field_name, py_type, field(default_factory=list)))
        else:
            dataclass_fields.append((field_name, py_type, MISSING))

    # Create base dataclass
    base_cls = make_dataclass(name, dataclass_fields)

    # Add methods
    class DataLoggerMixin:
        """Mixin providing logging and serialization methods."""

        _schema = schema
        _field_specs = field_specs

        def log_timestep(self, **kwargs: Any) -> None:
            """Log data for a single timestep."""
            time_series_fields = {
                fname for fname, _, _, is_ts in self._field_specs if is_ts
            }
            for key, value in kwargs.items():
                if value is not None:
                    if key not in time_series_fields:
                        raise ValueError(
                            f"'{key}' is not a valid time-series field. "
                            f"Valid fields: {sorted(time_series_fields)}"
                        )
                    getattr(self, key).append(value)

        def to_dict(self) -> dict:
            """Convert to dictionary for serialization."""
            return {f.name: getattr(self, f.name) for f in fields(self)}  # type: ignore[arg-type]

        def save_to_parquet(
            self, root_path: str, partition_cols: Optional[list[str]] = None
        ) -> None:
            """Save to Parquet file."""
            table = pa.Table.from_pylist([self.to_dict()], schema=self._schema)
            pq.write_to_dataset(
                table,
                root_path=root_path,
                partition_cols=partition_cols,
                compression="zstd",
            )

        @classmethod
        def from_parquet(
            cls,
            root_path: str,
            filters: Optional[list] = None,
            trajectory_id: int = 0,
        ) -> Any:
            """Load from Parquet file."""
            parquet_filters: Optional[list[tuple[str, str, Any]]] = None
            contains_filters: list[tuple[str, str]] = []
            if filters:
                parquet_filters = []
                for col, op, val in filters:
                    if op == "contains":
                        contains_filters.append((col, val))
                    else:
                        parquet_filters.append((col, op, val))
                if not parquet_filters:
                    parquet_filters = None

            dataset = pq.read_table(
                root_path, filters=parquet_filters, schema=cls._schema
            )
            for col, substring in contains_filters:
                dataset = dataset.filter(pc.match_substring(dataset[col], substring))

            num_rows = len(dataset)
            if num_rows == 0:
                raise ValueError(f"No data found in {root_path} with filters {filters}")
            if num_rows > 1:
                print(f"Multiple rows {num_rows} found. Using row {trajectory_id}.")
                dataset = dataset.slice(trajectory_id, trajectory_id + 1)

            data_dict = dataset.to_pylist()[0]
            return cls(**data_dict)

    # Combine base class with mixin
    combined_cls = type(name, (DataLoggerMixin, base_cls), {})
    return combined_cls


#############################################################
# Logger Classes
#############################################################
ManoSharpaData = create_data_logger_class(
    "ManoSharpaData",
    BASE_FIELDS + MANO_FIELDS + SHARPA_FIELDS + OBJECT_FIELDS,
)

NvhumanDex3Data = create_data_logger_class(
    "NvhumanDex3Data",
    BASE_FIELDS + NVHUMAN_FIELDS + DEX3_FIELDS + OBJECT_FIELDS,
)
