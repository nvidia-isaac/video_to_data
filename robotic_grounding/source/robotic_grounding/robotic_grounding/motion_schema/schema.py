# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Unified `MotionData` parquet schema (version `motion_v1`).

This module is the single source of truth for the on-disk layout used by all
whole-body retargeting scripts, the planner, and the training loader.

- `SCHEMA_VERSION` is the on-disk version tag written to every file.
- `build_schema()` returns the flat pyarrow schema used by writer/reader.
- `MotionData` is the in-memory dataclass carried across the pipeline; its
  attribute names are a stable contract with consumers (notably
  `tasks/v2p_whole_body/mdp/commands/tracking_command.py`).

Design notes:

- All per-side data (hands, contacts) is stored on disk as `hand_sides`-indexed
  lists. The in-memory dataclass exposes flattened `left_*` / `right_*`
  attributes for consumer convenience.
- Quaternions are `wxyz` everywhere, both on disk and in memory.
- Poses that combine position and orientation are packed as `[x, y, z, qw, qx,
  qy, qz]` (length 7) per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

SCHEMA_VERSION: str = "motion_v1"
"""Current schema version. Bumped on breaking changes."""

# ---------------------------------------------------------------------------
# Field inventory
# ---------------------------------------------------------------------------


def _ts(dtype: pa.DataType) -> pa.DataType:
    """Wrap a per-timestep value type as a time-series list."""
    return pa.list_(dtype)


_POSE7 = pa.list_(pa.float32(), 7)  # [x, y, z, qw, qx, qy, qz]
_VEC3 = pa.list_(pa.float32(), 3)
_VEC4 = pa.list_(pa.float32(), 4)


METADATA_FIELDS: list[tuple[str, pa.DataType]] = [
    ("schema_version", pa.string()),
    ("sequence_id", pa.string()),
    ("robot_name", pa.string()),
    ("source_dataset", pa.string()),
    ("raw_motion_file", pa.string()),
    ("fps", pa.float32()),
    ("coord_frame", pa.string()),
]

ROBOT_FIELDS: list[tuple[str, pa.DataType]] = [
    ("robot_joint_names", pa.list_(pa.string())),
    ("robot_root_position", _ts(_VEC3)),
    ("robot_root_wxyz", _ts(_VEC4)),
    ("robot_joint_positions", _ts(pa.list_(pa.float32()))),
]

EE_FIELDS: list[tuple[str, pa.DataType]] = [
    ("ee_link_names", pa.list_(pa.string())),
    ("ee_pose_w", _ts(pa.list_(_POSE7))),  # (T, E, 7)
]

HAND_FIELDS: list[tuple[str, pa.DataType]] = [
    ("hand_sides", pa.list_(pa.string())),
    ("hand_frame_names", pa.list_(pa.list_(pa.string()))),
    # Per-side (T, K, 7) stored as list of list of list of float. pyarrow can't
    # express "ragged inner list with fixed inner pose length," so the inner
    # `pose7` dimension uses a fixed-size list.
    ("hand_frames_w", pa.list_(_ts(pa.list_(_POSE7)))),
    ("hand_finger_joint_names", pa.list_(pa.list_(pa.string()))),
    ("hand_finger_joints", pa.list_(_ts(pa.list_(pa.float32())))),
]

OBJECT_FIELDS: list[tuple[str, pa.DataType]] = [
    ("object_name", pa.string()),
    ("safe_object_name", pa.string()),
    ("object_body_names", pa.list_(pa.string())),
    ("safe_object_body_names", pa.list_(pa.string())),
    ("object_mesh_paths", pa.list_(pa.string())),
    ("object_urdf_paths", pa.list_(pa.string())),
    ("object_mesh_radius", pa.list_(pa.float32())),
    ("object_articulation", _ts(pa.float32())),
    ("object_root_axis_angle", _ts(_VEC3)),
    ("object_root_position", _ts(_VEC3)),
    # (T, B, 3) — ragged B, inner vec3 fixed-size
    ("object_body_position", _ts(pa.list_(_VEC3))),
    ("object_body_wxyz", _ts(pa.list_(_VEC4))),
]

CONTACT_FIELDS: list[tuple[str, pa.DataType]] = [
    # All per-side, aligned with `hand_sides` index.
    ("hand_contact_link_names", pa.list_(pa.list_(pa.string()))),
    # Per-side (T, N, 3) with ragged N.
    ("hand_link_contact_positions", pa.list_(_ts(pa.list_(_VEC3)))),
    ("hand_link_contact_normals", pa.list_(_ts(pa.list_(_VEC3)))),
    ("hand_object_contact_positions", pa.list_(_ts(pa.list_(_VEC3)))),
    ("hand_object_contact_normals", pa.list_(_ts(pa.list_(_VEC3)))),
    ("hand_object_contact_part_ids", pa.list_(_ts(pa.list_(pa.int32())))),
    ("hand_contact_active", pa.list_(_ts(pa.float32()))),
]

SOURCE_FIELDS: list[tuple[str, pa.DataType]] = [
    ("source_kind", pa.string()),
    ("source_payload", pa.binary()),  # pickled dict; opaque to training
    ("source_joint_names", pa.list_(pa.string())),
]

DIAGNOSTICS_FIELDS: list[tuple[str, pa.DataType]] = [
    ("ik_error_per_frame", _ts(pa.float32())),
    ("ik_num_iterations", _ts(pa.int32())),
    ("frame_task_errors", _ts(pa.list_(pa.float32()))),
]


ALL_FIELDS: list[tuple[str, pa.DataType]] = (
    METADATA_FIELDS
    + ROBOT_FIELDS
    + EE_FIELDS
    + HAND_FIELDS
    + OBJECT_FIELDS
    + CONTACT_FIELDS
    + SOURCE_FIELDS
    + DIAGNOSTICS_FIELDS
)


# Names of fields required for a training-eligible file. Reader uses this to
# fail fast if a file promises `motion_v1` but is missing core data.
REQUIRED_TRAINING_FIELDS: tuple[str, ...] = (
    "schema_version",
    "sequence_id",
    "robot_name",
    "fps",
    "robot_joint_names",
    "robot_root_position",
    "robot_root_wxyz",
    "robot_joint_positions",
    "ee_link_names",
    "ee_pose_w",
    "object_body_names",
    "object_body_position",
    "object_body_wxyz",
)


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------


def build_schema() -> pa.Schema:
    """Return the unified `motion_v1` pyarrow schema."""
    return pa.schema([(name, dtype) for name, dtype in ALL_FIELDS])


# ---------------------------------------------------------------------------
# In-memory dataclass
# ---------------------------------------------------------------------------


@dataclass
class MotionData:
    """In-memory container for motion data.

    Field naming preserves the contract consumed by
    `tasks/v2p_whole_body/mdp/commands/tracking_command.py`: `left_*` /
    `right_*` attributes for hand and contact data, even though on disk these
    live under `hand_sides`-indexed columns. The reader is responsible for
    populating both the on-disk view (useful for debugging / re-serialization)
    and the flattened view (consumed by training).

    Tensors are `None` when the corresponding group was not written by the
    producer. Training code already guards each optional field.
    """

    # ---- Metadata ----------------------------------------------------------
    schema_version: str = SCHEMA_VERSION
    sequence_id: str = ""
    robot_name: str = ""
    source_dataset: str = ""
    raw_motion_file: str = ""
    fps: float = 0.0
    coord_frame: str = ""

    # ---- Robot state (required for training) -------------------------------
    robot_joint_names: list[str] = field(default_factory=list)
    robot_root_position: Any = None  # torch.Tensor (T, 3), world frame
    robot_root_wxyz: Any = None  # torch.Tensor (T, 4), wxyz
    robot_joint_positions: Any = None  # torch.Tensor (T, J)

    # ---- End-effector frames ----------------------------------------------
    ee_link_names: list[str] | None = None
    ee_pose_w: Any = None  # torch.Tensor (T, E, 7); on-disk form
    # Flattened for training: ee_pos_w + ee_quat_w split from ee_pose_w.
    ee_pos_w: Any = None  # torch.Tensor (T, E, 3)
    ee_quat_w: Any = None  # torch.Tensor (T, E, 4)
    ee_link_ids: list[int] | None = None

    # ---- Object ------------------------------------------------------------
    object_name: str = ""
    safe_object_name: str = ""
    object_body_names: list[str] = field(default_factory=list)
    safe_object_body_names: list[str] = field(default_factory=list)
    object_mesh_paths: list[str] = field(default_factory=list)
    object_urdf_paths: list[str] = field(default_factory=list)
    object_mesh_radius: list[float] | None = None
    object_articulation: Any = None
    object_root_axis_angle: Any = None
    object_root_position: Any = None
    object_body_position: Any = None  # torch.Tensor (T, B, 3)
    object_body_wxyz: Any = None  # torch.Tensor (T, B, 4)

    # Convenience aliases used by training: primary object body (index 0).
    object_pos_w: Any = None  # torch.Tensor (T, 3)
    object_quat_w: Any = None  # torch.Tensor (T, 4)

    # ---- Hands (on-disk view) ---------------------------------------------
    hand_sides: list[str] = field(default_factory=list)
    hand_frame_names: list[list[str]] = field(default_factory=list)
    hand_frames_w: list[Any] = field(default_factory=list)
    hand_finger_joint_names: list[list[str]] = field(default_factory=list)
    hand_finger_joints: list[Any] = field(default_factory=list)

    # ---- Hands (flattened view for training) -------------------------------
    left_wrist_position: Any = None
    left_wrist_wxyz: Any = None
    right_wrist_position: Any = None
    right_wrist_wxyz: Any = None
    left_finger_joints: Any = None
    right_finger_joints: Any = None
    left_hand_frames: Any = None
    right_hand_frames: Any = None
    left_hand_frame_names: list[str] | None = None
    right_hand_frame_names: list[str] | None = None
    left_finger_joint_names: list[str] | None = None
    right_finger_joint_names: list[str] | None = None

    # ---- Contacts (on-disk view) ------------------------------------------
    hand_contact_link_names: list[list[str]] = field(default_factory=list)
    hand_link_contact_positions: list[Any] = field(default_factory=list)
    hand_link_contact_normals: list[Any] = field(default_factory=list)
    hand_object_contact_positions: list[Any] = field(default_factory=list)
    hand_object_contact_normals: list[Any] = field(default_factory=list)
    hand_object_contact_part_ids: list[Any] = field(default_factory=list)
    hand_contact_active: list[Any] = field(default_factory=list)

    # ---- Contacts (flattened view) ----------------------------------------
    left_link_contact_positions: Any = None
    left_object_contact_positions: Any = None
    right_link_contact_positions: Any = None
    right_object_contact_positions: Any = None
    left_link_contact_normals: Any = None
    left_object_contact_normals: Any = None
    right_link_contact_normals: Any = None
    right_object_contact_normals: Any = None
    left_object_contact_part_ids: Any = None
    right_object_contact_part_ids: Any = None
    left_hand_contact_active: Any = None
    right_hand_contact_active: Any = None

    # ---- Source raw motion -------------------------------------------------
    source_kind: str = ""
    source_payload: bytes = b""
    source_joint_names: list[str] = field(default_factory=list)

    # ---- Diagnostics -------------------------------------------------------
    ik_error_per_frame: Any = None
    ik_num_iterations: Any = None
    frame_task_errors: Any = None

    # ---- Consumer contract ------------------------------------------------
    # Kept so the training loader preserves the legacy `file_joint_names`
    # semantic (alias of `robot_joint_names` for joint reordering).
    file_joint_names: list[str] | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SchemaVersionMismatch(RuntimeError):  # noqa: N818
    """Raised when a parquet file's `schema_version` does not match this module.

    Kept without the "Error" suffix to preserve the public API name; the
    corresponding N818 ruff warning is intentionally suppressed.
    """

    def __init__(self, got: str, expected: str, path: str) -> None:
        """Build a readable error with migration hint."""
        super().__init__(
            f"Motion parquet at {path} has schema_version={got!r} but this "
            f"codebase expects {expected!r}. Run "
            f"`python scripts/motion_schema/migrate_to_v1.py {path}` to migrate, "
            f"or re-run the producing retarget/planner script on the latest "
            f"source tree."
        )
        self.got = got
        self.expected = expected
        self.path = path


class MissingRequiredField(RuntimeError):  # noqa: N818
    """Raised when a training-eligible parquet lacks a required column.

    Kept without the "Error" suffix to preserve the public API name; the
    corresponding N818 ruff warning is intentionally suppressed.
    """

    def __init__(self, missing: list[str], path: str) -> None:
        """Build a readable error listing missing fields."""
        super().__init__(
            f"Motion parquet at {path} is missing required fields: {missing}. "
            f"This file claims schema_version={SCHEMA_VERSION!r} but is not "
            f"training-eligible. Check the producer."
        )
        self.missing = missing
        self.path = path
