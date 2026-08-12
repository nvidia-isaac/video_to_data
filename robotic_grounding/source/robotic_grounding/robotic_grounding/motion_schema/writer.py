# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer for `motion_v1` parquets.

Producers populate a `MotionData` dataclass and call `save_motion_parquet(...)`.
The writer serializes a single row per parquet file and writes it under a
Hive partition (`sequence_id=<seq>/robot_name=<robot>/`).
"""

from __future__ import annotations

import shutil
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    ALL_FIELDS,
    DUAL_HAND,
    DUAL_HAND_PER_SIDE_FIELDS,
    SCHEMA_VERSION,
    MotionData,
    build_schema,
    required_fields_for,
    resolve_motion_kind,
)

# Optional dependency: torch is only needed when producers pass tensors to the
# coerce path. Import at module level so ruff's PLC0415 stays happy, but keep
# the module importable in thin environments that don't ship torch.
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_wxyz(tag: str, wxyz: Any) -> None:
    """Cheap sanity check that a quaternion series is plausibly `wxyz`.

    Guards against producers accidentally packing `xyzw`. Two heuristics:

    - `|w|` must exceed 0.3 at least once (identity or near-identity poses
      are near-universal in any long trajectory; if the first component is
      never the scalar, the convention is likely swapped).
    - `|w|` must not exceed 1.01 (quaternions must be unit; a value > 1.01
      indicates raw axis-angle or some other mis-stashed representation).
    """
    if wxyz is None:
        return
    arr = np.asarray(wxyz, dtype=np.float32)
    if arr.size == 0:
        return
    if arr.ndim < 2 or arr.shape[-1] != 4:
        return
    flat = arr.reshape(-1, 4)
    w = np.abs(flat[:, 0])
    last = np.abs(flat[:, 3])
    w_max = float(w.max())
    last_max = float(last.max())
    if w_max < 0.3 and last_max > 0.9:
        # Classic xyzw layout: w stays small, last component hugs 1.
        raise ValueError(
            f"[{tag}] quaternion series looks like xyzw, not wxyz "
            f"(max |first|={w_max:.3f}, max |last|={last_max:.3f}). "
            f"Producer must swap conventions before writing."
        )
    if w_max > 1.01:
        raise ValueError(
            f"[{tag}] quaternion w component exceeds 1.01 (max={w_max:.3f}); not unit quaternions."
        )


def _coerce(value: Any) -> Any:
    """Coerce torch tensors / numpy arrays to nested python lists for pyarrow."""
    if value is None:
        return None
    if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list | tuple):
        return [_coerce(v) for v in value]
    if isinstance(value, bytes | str | int | float | bool):
        return value
    return value


def _row_dict(md: MotionData) -> dict[str, Any]:
    """Build a single-row pyarrow-friendly dict from a `MotionData`.

    Each field is wrapped in a one-element list because each parquet file
    stores a single trajectory as a single row; wrapping matches the legacy
    `ManoSharpaData` convention and plays nicely with partitioned writes.
    """
    md.schema_version = SCHEMA_VERSION

    per_side_count = len(md.hand_sides or [])

    def _per_side(seq: list[Any] | None) -> list[Any]:
        if not seq:
            return [[] for _ in range(per_side_count)]
        return [_coerce(v) for v in seq]

    row = {
        # Metadata
        "schema_version": [md.schema_version],
        "sequence_id": [md.sequence_id],
        "robot_name": [md.robot_name],
        "motion_kind": [md.motion_kind],
        "source_dataset": [md.source_dataset],
        "raw_motion_file": [md.raw_motion_file],
        "fps": [float(md.fps) if md.fps is not None else 0.0],
        "coord_frame": [md.coord_frame],
        # Robot
        "robot_joint_names": [list(md.robot_joint_names or [])],
        "robot_root_position": [_coerce(md.robot_root_position)],
        "robot_root_wxyz": [_coerce(md.robot_root_wxyz)],
        "robot_joint_positions": [_coerce(md.robot_joint_positions)],
        # EE
        "ee_link_names": [list(md.ee_link_names or [])],
        "ee_pose_w": [_coerce(md.ee_pose_w)],
        # Hands
        "hand_sides": [list(md.hand_sides or [])],
        "hand_frame_names": [list(md.hand_frame_names or [])],
        "hand_frames_w": [_per_side(md.hand_frames_w)],
        "hand_finger_joint_names": [list(md.hand_finger_joint_names or [])],
        "hand_finger_joints": [_per_side(md.hand_finger_joints)],
        # Object
        "object_name": [md.object_name],
        "safe_object_name": [md.safe_object_name],
        "object_body_names": [list(md.object_body_names or [])],
        "safe_object_body_names": [list(md.safe_object_body_names or [])],
        "object_mesh_paths": [list(md.object_mesh_paths or [])],
        "object_urdf_paths": [list(md.object_urdf_paths or [])],
        "object_mesh_radius": [list(md.object_mesh_radius or [])],
        "object_articulation": [_coerce(md.object_articulation)],
        "object_root_axis_angle": [_coerce(md.object_root_axis_angle)],
        "object_root_position": [_coerce(md.object_root_position)],
        "object_body_position": [_coerce(md.object_body_position)],
        "object_body_wxyz": [_coerce(md.object_body_wxyz)],
        # Contacts
        "hand_contact_link_names": [list(md.hand_contact_link_names or [])],
        "hand_link_contact_positions": [_per_side(md.hand_link_contact_positions)],
        "hand_link_contact_normals": [_per_side(md.hand_link_contact_normals)],
        "hand_object_contact_positions": [_per_side(md.hand_object_contact_positions)],
        "hand_object_contact_normals": [_per_side(md.hand_object_contact_normals)],
        "hand_object_contact_part_ids": [_per_side(md.hand_object_contact_part_ids)],
        "hand_contact_active": [_per_side(md.hand_contact_active)],
        # Source raw
        "source_kind": [md.source_kind],
        "source_payload": [md.source_payload or b""],
        "source_joint_names": [list(md.source_joint_names or [])],
        # Diagnostics
        "ik_error_per_frame": [_coerce(md.ik_error_per_frame)],
        "ik_num_iterations": [_coerce(md.ik_num_iterations)],
        "frame_task_errors": [_coerce(md.frame_task_errors)],
    }

    # Pyarrow rejects column counts mismatching the schema; keep them aligned.
    assert set(row.keys()) == {
        name for name, _ in ALL_FIELDS
    }, f"writer row dict out of sync with schema; diff={set(row.keys()) ^ {name for name, _ in ALL_FIELDS}}"
    return row


def _validate_required(md: MotionData) -> None:
    """Fail fast if required training fields are empty or missing.

    Validation is branched on `md.motion_kind`: whole-body files require
    `robot_*` joint state, while dual-hand files require per-side hand frames
    and finger joints. The discriminator must be set explicitly; producers
    that omit `motion_kind` raise here.
    """
    motion_kind = resolve_motion_kind(md)
    required = required_fields_for(motion_kind)
    missing: list[str] = []
    for name in required:
        val = getattr(md, name, None)
        if name == "fps":
            if val is None or float(val) <= 0.0:
                missing.append(name)
        elif val is None or (
            hasattr(val, "__len__") and len(val) == 0 and name != "schema_version"
        ):
            missing.append(name)
    if missing:
        raise ValueError(
            f"Cannot write motion parquet (motion_kind={motion_kind!r}): "
            f"required fields are empty: {missing}. Producer must populate at "
            f"least: {list(required)}."
        )

    if motion_kind == DUAL_HAND:
        n_sides = len(md.hand_sides or [])
        misaligned: list[tuple[str, int]] = []
        for name in DUAL_HAND_PER_SIDE_FIELDS:
            val = getattr(md, name, None)
            actual = len(cast(Sized, val)) if hasattr(val, "__len__") else 0
            if actual != n_sides:
                misaligned.append((name, actual))
        if misaligned:
            details = ", ".join(
                f"{name} has {actual} entries" for name, actual in misaligned
            )
            raise ValueError(
                f"Cannot write motion parquet (motion_kind=dual_hand): per-side "
                f"fields must align with hand_sides (len={n_sides}); {details}."
            )


# ---------------------------------------------------------------------------
# Public writer
# ---------------------------------------------------------------------------


DEFAULT_PARTITION_COLS: list[str] = ["sequence_id", "robot_name"]


def save_motion_parquet(
    md: MotionData,
    root_path: str | Path,
    partition_cols: list[str] | None = None,
    validate: bool = True,
    file_name: str | None = None,
) -> Path:
    """Write a `MotionData` row to a Hive-partitioned parquet dataset.

    Args:
        md: Populated `MotionData`.
        root_path: Dataset root (e.g. `.../whole_body/soma`).
        partition_cols: Hive partition keys. Defaults to `sequence_id`, `robot_name`.
        validate: If True, run required-fields and wxyz sanity checks before writing.
        file_name: Optional stable basename for the single parquet file (e.g.
            `"data.parquet"`). When ``None`` (default), pyarrow's auto-generated
            UUID-prefixed name is kept. Since ``pq.write_to_dataset`` doesn't
            expose a basename, we rename after the fact; safe because the writer
            clears the partition dir above so exactly one file is produced.

    Returns:
        The partition directory that was written.
    """
    if validate:
        _validate_required(md)
        _validate_wxyz("robot_root_wxyz", md.robot_root_wxyz)

    partition_cols = partition_cols or DEFAULT_PARTITION_COLS
    schema = build_schema()
    table = pa.Table.from_pydict(_row_dict(md), schema=schema)

    root = Path(root_path)
    partition_dir = root
    row = table.to_pydict()
    for col in partition_cols:
        val = row[col][0]
        partition_dir = partition_dir / f"{col}={val}"
    if partition_dir.is_dir():
        shutil.rmtree(partition_dir)

    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=partition_cols,
        compression="zstd",
    )

    if file_name is not None:
        written = list(partition_dir.glob("*.parquet"))
        if len(written) != 1:
            raise RuntimeError(
                f"Expected exactly one parquet under {partition_dir} after "
                f"write_to_dataset, found {len(written)}: {written}"
            )
        written[0].rename(partition_dir / file_name)

    return partition_dir
