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

# Define field specifications once to avoid duplication
# Format: (field_name, pyarrow_type, python_type, is_time_series)
FIELD_SPECS = [
    # Metadata fields (non-time-series)
    ("sequence_id", pa.string(), str, False),
    ("raw_motion_file", pa.string(), str, False),
    ("object_name", pa.string(), str, False),
    ("robot_name", pa.string(), str, False),
    ("fps", pa.float32(), float, False),
    ("mano_to_robot_scale", pa.float32(), float, False),
    ("mano_right_betas", pa.list_(pa.float32()), list[float], False),
    ("mano_left_betas", pa.list_(pa.float32()), list[float], False),
    ("right_robot_finger_joint_names", pa.list_(pa.string()), list[str], False),
    ("right_robot_frame_names", pa.list_(pa.string()), list[str], False),
    ("right_robot_frame_task_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_finger_joint_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_frame_names", pa.list_(pa.string()), list[str], False),
    ("left_robot_frame_task_names", pa.list_(pa.string()), list[str], False),
    ("object_body_names", pa.list_(pa.string()), list[str], False),
    # MANO right hand (time series)
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
        list[list[float]],
        True,
    ),
    (
        "mano_right_joints_wxyz",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), 21)),
        list[list[float]],
        True,
    ),
    ("mano_right_fitting_err", pa.list_(pa.float32()), list[float], True),
    (
        "mano_right_tips_distance",
        pa.list_(pa.list_(pa.float32(), 5)),
        list[list[float]],
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
        list[list[float]],
        True,
    ),
    (
        "mano_left_joints_wxyz",
        pa.list_(pa.list_(pa.list_(pa.float32(), 4), 21)),
        list[list[float]],
        True,
    ),
    ("mano_left_fitting_err", pa.list_(pa.float32()), list[float], True),
    (
        "mano_left_tips_distance",
        pa.list_(pa.list_(pa.float32(), 5)),
        list[list[float]],
        True,
    ),
    # Object (time series)
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
    # Robot right hand (time series)
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
        pa.list_(pa.list_(pa.list_(pa.float32(), 7), 67)),
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
    # Robot left hand (time series)
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
        pa.list_(pa.list_(pa.list_(pa.float32(), 7), 67)),
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

# Generate PyArrow schema from field specifications
MANO_SHARPA_SCHEMA = pa.schema([(name, pa_type) for name, pa_type, _, _ in FIELD_SPECS])


# Helper function to create dataclass fields from FIELD_SPECS
def _make_dataclass_fields() -> list[tuple[str, type, Any]]:
    """Generate dataclass field specifications from FIELD_SPECS.

    Returns:
        List of tuples (field_name, field_type, field_default_or_default_factory)
    """
    dataclass_fields: list[tuple[str, Any, Any]] = []
    for field_name, _, py_type, is_time_series in FIELD_SPECS:
        if is_time_series:
            # Time series fields get a default_factory for empty list
            dataclass_fields.append((field_name, py_type, field(default_factory=list)))
        else:
            # Metadata fields are required (no default)
            dataclass_fields.append((field_name, py_type, MISSING))
    return dataclass_fields


# Dynamically create the base dataclass from FIELD_SPECS
_ManoSharpaDataBase: type = make_dataclass(
    "ManoSharpaData",
    _make_dataclass_fields(),
    namespace={
        "__doc__": """Data class for storing kinematic data.

        All fields are dynamically generated from FIELD_SPECS to avoid duplication.
        Metadata fields are required, time-series fields default to empty lists.
        """
    },
)


class ManoSharpaData(_ManoSharpaDataBase):  # type: ignore[misc, valid-type]
    """Data class for storing kinematic data with methods for logging and serialization."""

    def log_timestep(self, **kwargs: Any) -> None:
        """Log data for a single timestep.

        Accepts any time-series field as a keyword argument. For scalar time-series
        (like fitting_err, articulation, num_optimization_iterations), pass the scalar value.
        For array time-series, pass the array.

        Example:
            logger.log_timestep(
                mano_right_trans=[1.0, 2.0, 3.0],
                mano_right_fitting_err=0.05,
                robot_right_qpos=[0.1, 0.2, ...],
            )
        """
        # Get time-series field names from FIELD_SPECS
        time_series_fields = {name for name, _, _, is_ts in FIELD_SPECS if is_ts}

        for key, value in kwargs.items():
            if value is not None:
                if key not in time_series_fields:
                    raise ValueError(
                        f"'{key}' is not a valid time-series field. "
                        f"Valid fields: {sorted(time_series_fields)}"
                    )
                getattr(self, key).append(value)

    def to_dict(self) -> dict:
        """Convert the dataclass to a dictionary for PyArrow serialization."""
        # Use dataclass fields() to get all field names dynamically
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def save_to_parquet(
        self, root_path: str, partition_cols: Optional[list[str]] = None
    ) -> None:
        """Save the data to a Parquet file.

        Args:
            root_path: Path to the parquet dataset directory
            partition_cols: Optional columns to partition the data by

        Raises:
            ValueError: If the data is not valid
        """
        table = pa.Table.from_pylist([self.to_dict()], schema=MANO_SHARPA_SCHEMA)
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
    ) -> "ManoSharpaData":
        """Create a new ManoSharpaData instance from a Parquet file.

        Args:
            root_path: Path to the parquet dataset directory
            filters: Optional filters in PyArrow format, e.g.,
                     [('sequence_id', '=', 'seq_001')]
            trajectory_id: Index of the row to load when multiple rows match
        """
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

        # Read the parquet dataset
        dataset = pq.read_table(
            root_path, filters=parquet_filters, schema=MANO_SHARPA_SCHEMA
        )
        for col, substring in contains_filters:
            dataset = dataset.filter(pc.match_substring(dataset[col], substring))

        # Check the number of rows
        num_rows = len(dataset)

        if num_rows == 0:
            raise ValueError(f"No data found in {root_path} with filters {filters}")

        if num_rows > 1:
            print(
                f"Multiple rows {num_rows} found in {root_path} for filters {filters}. "
                f"Using the {trajectory_id}th row."
            )
            dataset = dataset.slice(trajectory_id, trajectory_id + 1)

        # Convert only the selected row(s) to Python list
        data_dict = dataset.to_pylist()[0]

        # Create instance with all fields from data_dict
        return cls(**data_dict)
