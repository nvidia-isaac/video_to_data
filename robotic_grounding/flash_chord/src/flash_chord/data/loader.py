# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema-aware public reference loader."""

from __future__ import annotations

from pathlib import Path

from flash_chord.data.mano_sharpa import load_mano_sharpa
from flash_chord.data.motion_v1 import load_motion_v1
from flash_chord.data.parquet import read_parquet_row
from flash_chord.data.reference import Reference

_DISPATCH_COLUMNS = (
    "schema_version",
    "fps",
    "object_name",
    "object_body_names",
    "object_body_position",
    "object_body_wxyz",
    "object_mesh_paths",
    "object_mesh_radius",
    "robot_left_wrist_position",
    "robot_left_wrist_wxyz",
    "robot_left_finger_joints",
    "left_robot_finger_joint_names",
    "robot_left_frames",
    "left_robot_frame_names",
    "mano_left_object_contact_positions",
    "mano_left_object_contact_normals",
    "mano_left_object_contact_part_ids",
    "robot_right_wrist_position",
    "robot_right_wrist_wxyz",
    "robot_right_finger_joints",
    "right_robot_finger_joint_names",
    "robot_right_frames",
    "right_robot_frame_names",
    "mano_right_object_contact_positions",
    "mano_right_object_contact_normals",
    "mano_right_object_contact_part_ids",
)
_MANO_SHARPA_SIGNATURE = frozenset(_DISPATCH_COLUMNS[1:])


def load_reference(
    parquet_path: str | Path,
    control_fps: float | None = None,
    motion_speed: float = 1.0,
    source_frame_playback: bool = False,
) -> Reference:
    """Load a declared schema or the current unversioned ``ManoSharpaData`` format."""
    row = read_parquet_row(parquet_path, columns=_DISPATCH_COLUMNS)
    schema_version = row.cells.get("schema_version")
    if schema_version == "motion_v1":
        return load_motion_v1(
            str(row.path),
            control_fps=control_fps,
            motion_speed=motion_speed,
            source_frame_playback=source_frame_playback,
        )
    if schema_version not in (None, ""):
        raise ValueError(f"unsupported reference schema_version={schema_version!r}: {row.path}")
    if _MANO_SHARPA_SIGNATURE <= row.cells.keys():
        if source_frame_playback:
            raise ValueError("source-frame playback is supported only for motion_v1 references")
        return load_mano_sharpa(str(row.path), control_fps=control_fps, motion_speed=motion_speed)
    raise ValueError(
        "unversioned reference does not match the ManoSharpaData format; "
        f"declare a supported schema_version: {row.path}"
    )
