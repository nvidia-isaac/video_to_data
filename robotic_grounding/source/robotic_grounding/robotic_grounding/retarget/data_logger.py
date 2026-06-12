# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import re
import shutil
from dataclasses import MISSING, field, fields, make_dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

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
    ("mano_flat_hand_mean", pa.bool_(), bool, False),
    ("mano_center_idx", pa.int32(), int, False),
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
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_right_link_contact_normals",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_right_object_contact_positions",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_right_object_contact_normals",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_right_object_contact_part_ids",
        pa.list_(pa.list_(pa.int32(), 16)),
        list[list[int]],
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
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_left_link_contact_normals",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_left_object_contact_positions",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_left_object_contact_normals",
        pa.list_(pa.list_(pa.list_(pa.float32(), 3), 16)),
        list[list[list[float]]],
        True,
    ),
    (
        "mano_left_object_contact_part_ids",
        pa.list_(pa.list_(pa.int32(), 16)),
        list[list[int]],
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
DEX3_MANO_NUM_FRAMES = 23
DEX3_NUM_FRAME_TASKS = 4
DEX3_MANO_FIELDS: list[FieldSpec] = [
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
        "robot_right_wrist_wxyz",
        pa.list_(pa.list_(pa.float32(), 4)),
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
        pa.list_(pa.list_(pa.list_(pa.float32(), 7), DEX3_MANO_NUM_FRAMES)),
        list[list[list[float]]],
        True,
    ),
    (
        "robot_right_frame_task_errors",
        pa.list_(pa.list_(pa.float32(), DEX3_NUM_FRAME_TASKS)),
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
        pa.list_(pa.list_(pa.float32(), DEX3_NUM_FINGER_JOINTS)),
        list[list[float]],
        True,
    ),
    (
        "robot_left_frames",
        pa.list_(pa.list_(pa.list_(pa.float32(), 7), DEX3_MANO_NUM_FRAMES)),
        list[list[list[float]]],
        True,
    ),
    (
        "robot_left_frame_task_errors",
        pa.list_(pa.list_(pa.float32(), DEX3_NUM_FRAME_TASKS)),
        list[list[float]],
        True,
    ),
    ("robot_left_num_optimization_iterations", pa.list_(pa.int32()), list[int], True),
]

#############################################################
# Object fields
#############################################################
OBJECT_FIELDS: list[FieldSpec] = [
    ("object_name", pa.string(), str, False),
    ("safe_object_name", pa.string(), str, False),
    ("object_body_names", pa.list_(pa.string()), list[str], False),
    ("safe_object_body_names", pa.list_(pa.string()), list[str], False),
    ("object_mesh_paths", pa.list_(pa.string()), list[str], False),
    ("object_urdf_paths", pa.list_(pa.string()), list[str], False),
    ("object_mesh_radius", pa.list_(pa.float32()), list[float], False),
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
            """Save to Parquet file.

            If partition_cols are specified and the partition directory already
            exists, it is removed first so stale data is not left behind.
            """
            table = pa.Table.from_pylist([self.to_dict()], schema=self._schema)

            # Remove existing partition directory to avoid duplicate files.
            # Read partition values from the table (after pyarrow casting) so
            # directory names match exactly what write_to_dataset creates.
            if partition_cols:
                partition_dir = Path(root_path)
                row = table.to_pydict()
                for col in partition_cols:
                    val = row[col][0]
                    partition_dir = partition_dir / f"{col}={val}"
                if partition_dir.is_dir():
                    shutil.rmtree(partition_dir)

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

            # Select columns in schema order and cast so partition columns have correct types
            dataset = dataset.select(cls._schema.names)
            dataset = dataset.cast(cls._schema)
            data_dict = dataset.to_pylist()[0]
            return cls(**data_dict)

    # Combine base class with mixin
    combined_cls = type(name, (DataLoggerMixin, base_cls), {})
    return combined_cls


def list_sequence_ids(root_path: str) -> list[str]:
    """Return sorted list of unique sequence_id values in the Parquet dataset.

    Reads partition directory names when ``sequence_id`` is a partition column,
    avoiding a full table scan.  Falls back to reading the column if the
    directory structure doesn't match.
    """
    root = Path(root_path)
    if not root.is_dir():
        print(f"Input directory not found: {root_path}. Run the loader first.")
        return []
    # Partitioned datasets store sequence_id as directory names: sequence_id=<value>/
    partition_dirs = sorted(root.glob("sequence_id=*/"))
    if partition_dirs:
        return [unquote(d.name.split("=", 1)[1]) for d in partition_dirs]
    # Fallback: read the column from Parquet files
    table = pq.read_table(root_path, columns=["sequence_id"])
    ids = pc.unique(table["sequence_id"])
    return sorted(ids.to_pylist())


def shard_matches(key: str, shard_id: int, num_shards: int) -> bool:
    """Return True iff ``key`` belongs to this shard.

    Uses md5 for a stable, process-independent partition so the same key
    always lands in the same shard regardless of ``PYTHONHASHSEED``.
    Pass ``num_shards <= 1`` to disable sharding (matches everything).
    """
    if num_shards <= 1:
        return True
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return h % num_shards == shard_id


def add_sequence_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add --sequence_pattern, --sequence_id, --max_sequences, --shard_id, --num_shards."""
    group = parser.add_argument_group("sequence filtering")
    group.add_argument(
        "--sequence_id",
        type=str,
        default=None,
        help="Process a single sequence by exact ID.",
    )
    group.add_argument(
        "--sequence_pattern",
        type=str,
        default=None,
        help="Regex pattern to filter sequence IDs (e.g., '.*box.*').",
    )
    group.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Limit to first N sequences after filtering.",
    )
    group.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard index (0-based) for parallel processing.",
    )
    group.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards.  1 = no sharding (default).",
    )


def filter_sequence_ids(sequence_ids: list[str], args: argparse.Namespace) -> list[str]:
    """Apply all sequence filters, including shard partitioning (applied last)."""
    if getattr(args, "sequence_id", None):
        sequence_ids = [s for s in sequence_ids if s == args.sequence_id]
    if getattr(args, "sequence_pattern", None):
        pat = re.compile(args.sequence_pattern)
        sequence_ids = [s for s in sequence_ids if pat.search(s)]
    if getattr(args, "max_sequences", None):
        sequence_ids = sequence_ids[: args.max_sequences]

    num_shards = getattr(args, "num_shards", 1) or 1
    shard_id = getattr(args, "shard_id", 0) or 0
    if num_shards > 1:
        sequence_ids = [
            s for s in sequence_ids if shard_matches(s, shard_id, num_shards)
        ]
    return sequence_ids


#############################################################
# Logger Classes
#############################################################
# ManoSharpaData remains the logger for the dual-hand V2P pipeline.
# The whole-body G1 and hand-only Dex3 producers write the unified
# ``motion_v1`` schema directly via ``robotic_grounding.motion_schema``;
# no dedicated logger class is needed for them.
ManoSharpaData = create_data_logger_class(
    "ManoSharpaData",
    BASE_FIELDS + MANO_FIELDS + SHARPA_FIELDS + OBJECT_FIELDS,
)

ManoDex3Data = create_data_logger_class(
    "ManoDex3Data",
    BASE_FIELDS + MANO_FIELDS + DEX3_MANO_FIELDS + OBJECT_FIELDS,
)
