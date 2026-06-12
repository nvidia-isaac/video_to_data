# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from dataclasses import dataclass, field, replace
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
    ("motion_kind", pa.string()),
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


# Recognized values for the `motion_kind` discriminator. The schema branches
# its required-field check on this value: `single_robot` files carry
# whole-body joint state, while `dual_hand` files carry only floating-wrist
# end-effector state plus per-side hand frames.
SINGLE_ROBOT: str = "single_robot"
DUAL_HAND: str = "dual_hand"
KNOWN_MOTION_KINDS: frozenset[str] = frozenset({SINGLE_ROBOT, DUAL_HAND})


# Fields required for any training-eligible file regardless of motion kind.
# `motion_kind` is included so loaders fail fast on legacy files that predate
# the discriminator instead of silently inferring a kind.
COMMON_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "sequence_id",
    "robot_name",
    "motion_kind",
    "fps",
    "ee_link_names",
    "ee_pose_w",
    "object_body_names",
    "object_body_position",
    "object_body_wxyz",
)


# Required for `motion_kind == "single_robot"`: whole-body joint trajectories.
SINGLE_ROBOT_REQUIRED_FIELDS: tuple[str, ...] = (
    "robot_joint_names",
    "robot_root_position",
    "robot_root_wxyz",
    "robot_joint_positions",
)


# Required for `motion_kind == "dual_hand"`: floating-wrist trajectories
# carry per-side hand frames and finger joints aligned by `hand_sides`.
DUAL_HAND_REQUIRED_FIELDS: tuple[str, ...] = (
    "hand_sides",
    "hand_frame_names",
    "hand_frames_w",
    "hand_finger_joint_names",
    "hand_finger_joints",
)


# Per-side fields that must have an outer length equal to `len(hand_sides)`.
# This catches producers that set `hand_sides` but forgot to populate one
# side's payload, which would otherwise silently produce ragged data.
DUAL_HAND_PER_SIDE_FIELDS: tuple[str, ...] = (
    "hand_frame_names",
    "hand_frames_w",
    "hand_finger_joint_names",
    "hand_finger_joints",
)


# Backwards-compatible alias. New code should consult the per-kind tuples
# above. This union is the conservative super-set used only by callers that
# do not yet know the file's `motion_kind`.
REQUIRED_TRAINING_FIELDS: tuple[str, ...] = tuple(
    dict.fromkeys(
        COMMON_REQUIRED_FIELDS
        + SINGLE_ROBOT_REQUIRED_FIELDS
        + DUAL_HAND_REQUIRED_FIELDS
    )
)


def resolve_motion_kind(source: Any) -> str:
    """Return the explicit `motion_kind` from a `MotionData` or pyarrow row dict.

    Raises:
        ValueError: If `motion_kind` is missing, empty, or not one of the
            values in `KNOWN_MOTION_KINDS`. Inference from `robot_joint_names`
            or `hand_sides` is intentionally not performed; producers must
            tag every file explicitly.
    """
    if isinstance(source, dict):
        raw = source.get("motion_kind")
        if isinstance(raw, list):
            raw = raw[0] if raw else None
    else:
        raw = getattr(source, "motion_kind", None)
    if raw is None or raw == "":
        raise ValueError(
            "Motion file is missing `motion_kind`. Producers must set "
            f"motion_kind to one of {sorted(KNOWN_MOTION_KINDS)}; legacy "
            "files predating this field are not supported and must be "
            "regenerated."
        )
    if raw not in KNOWN_MOTION_KINDS:
        raise ValueError(
            f"Unknown motion_kind={raw!r}. Expected one of {sorted(KNOWN_MOTION_KINDS)}."
        )
    return raw


def required_fields_for(motion_kind: str) -> tuple[str, ...]:
    """Return the required field tuple for a given resolved `motion_kind`."""
    if motion_kind == SINGLE_ROBOT:
        return COMMON_REQUIRED_FIELDS + SINGLE_ROBOT_REQUIRED_FIELDS
    if motion_kind == DUAL_HAND:
        return COMMON_REQUIRED_FIELDS + DUAL_HAND_REQUIRED_FIELDS
    raise ValueError(
        f"Unknown motion_kind={motion_kind!r}. Expected one of {sorted(KNOWN_MOTION_KINDS)}."
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
    # Discriminator between layout shapes. Required to be non-empty before
    # writing or after loading; see `KNOWN_MOTION_KINDS` for accepted values.
    motion_kind: str = ""
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

    # ------------------------------------------------------------------
    # Time-axis subsetting
    # ------------------------------------------------------------------

    # Field names whose first axis is the time axis T. Used by `trim()` so
    # the schema stays the single source of truth for time-axis layout.
    # Not annotated → class attribute only, not a dataclass field.
    _TIME_AXIS_TENSOR_FIELDS = (
        "robot_root_position",
        "robot_root_wxyz",
        "robot_joint_positions",
        "ee_pose_w",
        "ee_pos_w",
        "ee_quat_w",
        "object_articulation",
        "object_root_axis_angle",
        "object_root_position",
        "object_body_position",
        "object_body_wxyz",
        "object_pos_w",
        "object_quat_w",
        "left_wrist_position",
        "left_wrist_wxyz",
        "right_wrist_position",
        "right_wrist_wxyz",
        "left_finger_joints",
        "right_finger_joints",
        "left_hand_frames",
        "right_hand_frames",
        "left_link_contact_positions",
        "left_object_contact_positions",
        "right_link_contact_positions",
        "right_object_contact_positions",
        "left_link_contact_normals",
        "left_object_contact_normals",
        "right_link_contact_normals",
        "right_object_contact_normals",
        "left_object_contact_part_ids",
        "right_object_contact_part_ids",
        "left_hand_contact_active",
        "right_hand_contact_active",
        "ik_error_per_frame",
        "ik_num_iterations",
        "frame_task_errors",
    )

    # List-of-tensor fields where each element is `(T, ...)` and should be
    # sliced per-element.
    _TIME_AXIS_TENSOR_LIST_FIELDS = (
        "hand_frames_w",
        "hand_finger_joints",
        "hand_link_contact_positions",
        "hand_link_contact_normals",
        "hand_object_contact_positions",
        "hand_object_contact_normals",
        "hand_object_contact_part_ids",
        "hand_contact_active",
    )

    def num_frames(self) -> int:
        """Return motion length T inferred from a required time-axis field."""
        ref = self.robot_root_position
        if ref is None:
            return 0
        return int(ref.shape[0])

    def trim(self, start_frame: int = 0, end_frame: int | None = None) -> "MotionData":
        """Return a copy of this MotionData restricted to ``[start, end)`` frames.

        Slices every time-axis tensor (and per-element of every time-axis tensor
        list) along axis 0. Non-tensor metadata is shared with the source.

        Args:
            start_frame: First frame to keep (inclusive). Must be ``>= 0``.
            end_frame: One past the last frame to keep. ``None`` means the
                end of the sequence.

        Returns:
            A new ``MotionData`` whose time-axis fields contain frames
            ``[start_frame, end_frame)`` of the original.

        Raises:
            ValueError: If the range is empty or out of bounds.
        """
        total = self.num_frames()
        if total == 0:
            # Nothing to trim; return self to avoid masking a missing-field bug
            # behind a silent no-op clone.
            if start_frame == 0 and end_frame in (None, 0):
                return self
            raise ValueError(
                f"Cannot trim MotionData with no time-axis tensors: requested "
                f"[{start_frame}, {end_frame})."
            )

        end = total if end_frame is None else end_frame
        if start_frame < 0 or end > total or start_frame >= end:
            raise ValueError(
                f"Invalid trim range [{start_frame}, {end}) for motion of length "
                f"{total}. Require 0 <= start_frame < end_frame <= num_frames."
            )

        if start_frame == 0 and end == total:
            return self

        updates: dict[str, Any] = {}
        for name in self._TIME_AXIS_TENSOR_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            updates[name] = value[start_frame:end]
        for name in self._TIME_AXIS_TENSOR_LIST_FIELDS:
            value = getattr(self, name)
            if not value:
                continue
            updates[name] = [None if v is None else v[start_frame:end] for v in value]
        return replace(self, **updates)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SchemaVersionMismatch(RuntimeError):  # noqa: N818
    """Raised when a parquet file's `schema_version` does not match this module.

    Kept without the "Error" suffix to preserve the public API name; the
    corresponding N818 ruff warning is intentionally suppressed.
    """

    def __init__(self, got: str, expected: str, path: str) -> None:
        """Build a readable error pointing producers at the right action."""
        super().__init__(
            f"Motion parquet at {path} has schema_version={got!r} but this "
            f"codebase expects {expected!r}. Re-run the producing retarget "
            f"or planner script on the latest source tree to regenerate this "
            f"file; no migrator is shipped."
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
