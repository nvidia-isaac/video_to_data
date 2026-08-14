# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reader for `motion_v1` parquets.

Training and downstream tools call `load_motion_data_parquet(path)` to get a
populated `MotionData`. The reader handles:

- Directory-or-file path resolution (matches the legacy behaviour of
  `tracking_utils.load_motion_data` and `SceneConfig.from_motion_file`).
- Schema version enforcement with actionable error messages.
- On-disk → in-memory mapping: `hand_sides`-indexed lists are also exposed as
  flattened `left_*` / `right_*` attributes for consumers that predate the
  unified schema.
- Convenience derived fields: `ee_pos_w`, `ee_quat_w`, `object_pos_w`,
  `object_quat_w`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    DUAL_HAND,
    DUAL_HAND_PER_SIDE_FIELDS,
    SCHEMA_VERSION,
    MissingRequiredField,
    MotionData,
    SchemaVersionMismatch,
    required_fields_for,
    resolve_motion_kind,
)

# Optional dependency: the reader only needs torch when producing tensor
# outputs. Import at module level for ruff PLC0415 but keep the module
# importable in thin environments that don't ship torch.
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_parquet_file(path: str | Path) -> Path:
    """Resolve a path to a concrete parquet file.

    Accepts either a direct parquet file, a Hive partition directory, or a
    dataset root that contains exactly one partition.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        matches = list(p.rglob("*.parquet"))
        if not matches:
            raise FileNotFoundError(f"No parquet files under {p}")
        return matches[0]
    raise FileNotFoundError(f"Motion parquet path does not exist: {p}")


def _read_single_file(path: Path) -> pa.Table:
    """Read a single parquet file, bypassing Hive partition auto-inference.

    `pq.read_table(path)` on a partitioned path tree can infer partition
    columns as dictionary types and then clash with the explicit string
    columns in the file body. We side-step that by using `ParquetFile`.
    """
    pf = pq.ParquetFile(str(path))
    return pf.read()


def _partition_values_from_path(path: Path) -> dict[str, str]:
    """Extract Hive partition values (`col=val`) from the path components."""
    values: dict[str, str] = {}
    for parent in path.parents:
        if "=" in parent.name:
            col, val = parent.name.split("=", 1)
            values[col] = val
    return values


def _is_empty(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, list | tuple | str | bytes):
        return len(val) == 0
    return False


def _as_tensor(
    value: Any,
    device: str | None = None,
    dtype: Any = None,
) -> Any:
    """Coerce a nested-list value into a torch tensor, or return None.

    Returns `Any` because torch is an optional dependency at the module level;
    callers that care about the exact type should narrow themselves.
    """
    if value is None:
        return None
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "torch is required to materialize motion_v1 tensors but is not installed in this environment."
        )

    if isinstance(value, torch.Tensor):
        out = value
    else:
        arr = np.asarray(value)
        if arr.dtype == object or arr.size == 0:
            return None
        if dtype is None:
            # Preserve int dtype when the input is integer (e.g. part_ids).
            if np.issubdtype(arr.dtype, np.integer):
                dtype = torch.long
            else:
                dtype = torch.float32
        out = torch.as_tensor(arr, dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


# ---------------------------------------------------------------------------
# Core reader
# ---------------------------------------------------------------------------


def _flatten_per_side(
    data: dict[str, Any],
    disk_column: str,
    side_index: int,
) -> Any:
    """Return the side-indexed entry from a per-side list, or None if absent/empty."""
    col = data.get(disk_column)
    if not col or col[0] is None:
        return None
    per_side_list = col[0]
    if side_index >= len(per_side_list):
        return None
    value = per_side_list[side_index]
    if _is_empty(value):
        return None
    return value


def _validate_version(data: dict[str, Any], path: Path) -> None:
    version = data.get("schema_version", [None])
    if not version or not version[0]:
        raise SchemaVersionMismatch(
            got="<missing>", expected=SCHEMA_VERSION, path=str(path)
        )
    got = version[0]
    if got != SCHEMA_VERSION:
        raise SchemaVersionMismatch(got=got, expected=SCHEMA_VERSION, path=str(path))


def _validate_required(data: dict[str, Any], path: Path) -> None:
    """Branch required-field validation on the file's `motion_kind`.

    The reader refuses any file that lacks an explicit `motion_kind`.
    Producers must regenerate files that predate the discriminator instead
    of relying on inference from neighboring columns.
    """
    raw_kind = (data.get("motion_kind") or [None])[0]
    if not raw_kind:
        raise MissingRequiredField(missing=["motion_kind"], path=str(path))
    try:
        motion_kind = resolve_motion_kind({"motion_kind": [raw_kind]})
    except ValueError as exc:
        raise MissingRequiredField(missing=["motion_kind"], path=str(path)) from exc

    required = required_fields_for(motion_kind)
    missing: list[str] = []
    for name in required:
        if name not in data:
            missing.append(name)
            continue
        val = data[name]
        if not val or val[0] is None:
            missing.append(name)
            continue
        if name == "fps":
            try:
                if float(val[0]) <= 0.0:
                    missing.append(name)
            except (TypeError, ValueError):
                missing.append(name)
            continue
        if hasattr(val[0], "__len__") and len(val[0]) == 0:
            missing.append(name)
    if missing:
        raise MissingRequiredField(missing=missing, path=str(path))

    if motion_kind == DUAL_HAND:
        n_sides = len(data.get("hand_sides", [[]])[0] or [])
        misaligned: list[tuple[str, int]] = []
        for name in DUAL_HAND_PER_SIDE_FIELDS:
            per_side = data.get(name, [[]])[0] or []
            if len(per_side) != n_sides:
                misaligned.append((name, len(per_side)))
        if misaligned:
            details = ", ".join(
                f"{name} has {actual} entries" for name, actual in misaligned
            )
            raise MissingRequiredField(
                missing=[name for name, _ in misaligned], path=str(path)
            ) from ValueError(
                f"motion_kind=dual_hand per-side fields must align with hand_sides (len={n_sides}); {details}."
            )


def load_motion_data_parquet(
    path: str | Path,
    device: str | None = None,
) -> MotionData:
    """Load a `motion_v1` parquet and return a populated `MotionData`.

    Args:
        path: File or directory containing the parquet.
        device: Optional torch device string to place tensors on.

    Returns:
        `MotionData` with both on-disk and flattened attribute views populated.

    Raises:
        SchemaVersionMismatch: If the file's `schema_version` is not `motion_v1`.
        MissingRequiredField: If required training fields are absent.
        FileNotFoundError: If the path doesn't resolve.
    """
    parquet_path = _resolve_parquet_file(path)
    table = _read_single_file(parquet_path)
    data = table.to_pydict()

    # `pq.write_to_dataset` strips partition columns from the file body. When
    # we bypass Hive auto-inference above, we must backfill them from the
    # partition directory names ourselves.
    for col, val in _partition_values_from_path(parquet_path).items():
        if col not in data or not data[col] or data[col][0] is None:
            data[col] = [val]

    _validate_version(data, parquet_path)
    _validate_required(data, parquet_path)

    md = MotionData()

    # ---- Metadata ---------------------------------------------------------
    md.schema_version = data["schema_version"][0]
    md.sequence_id = data.get("sequence_id", [""])[0] or ""
    md.robot_name = data.get("robot_name", [""])[0] or ""
    md.motion_kind = data.get("motion_kind", [""])[0] or ""
    md.source_dataset = data.get("source_dataset", [""])[0] or ""
    md.raw_motion_file = data.get("raw_motion_file", [""])[0] or ""
    md.fps = float(data.get("fps", [0.0])[0] or 0.0)
    md.coord_frame = data.get("coord_frame", [""])[0] or ""

    # ---- Robot state ------------------------------------------------------
    # Robot fields are only required for `single_robot`; for `dual_hand` they
    # may be absent or empty, and the loaded tensors are correspondingly None.
    md.robot_joint_names = list(data.get("robot_joint_names", [[]])[0] or [])
    md.robot_root_position = _as_tensor(
        data.get("robot_root_position", [None])[0], device=device
    )
    md.robot_root_wxyz = _as_tensor(
        data.get("robot_root_wxyz", [None])[0], device=device
    )
    md.robot_joint_positions = _as_tensor(
        data.get("robot_joint_positions", [None])[0], device=device
    )

    # `file_joint_names` is the legacy attribute used by tracking_command.py
    # for joint reordering; we alias it to robot_joint_names.
    md.file_joint_names = md.robot_joint_names or None

    # ---- End-effector frames ---------------------------------------------
    ee_link_names = list(data.get("ee_link_names", [[]])[0] or [])
    md.ee_link_names = ee_link_names or None
    ee_pose = _as_tensor(data["ee_pose_w"][0], device=device)
    md.ee_pose_w = ee_pose
    if ee_pose is not None and ee_pose.ndim == 3 and ee_pose.shape[-1] == 7:
        md.ee_pos_w = ee_pose[..., 0:3].contiguous()
        md.ee_quat_w = ee_pose[..., 3:7].contiguous()

    # ---- Object -----------------------------------------------------------
    md.object_name = data.get("object_name", [""])[0] or ""
    md.safe_object_name = data.get("safe_object_name", [""])[0] or ""
    md.object_body_names = list(data.get("object_body_names", [[]])[0] or [])
    md.safe_object_body_names = list(data.get("safe_object_body_names", [[]])[0] or [])
    md.object_mesh_paths = list(data.get("object_mesh_paths", [[]])[0] or [])
    md.object_urdf_paths = list(data.get("object_urdf_paths", [[]])[0] or [])
    md.object_mesh_radius = list(data.get("object_mesh_radius", [[]])[0] or []) or None
    md.object_articulation = _as_tensor(
        data.get("object_articulation", [None])[0], device=device
    )
    md.object_root_axis_angle = _as_tensor(
        data.get("object_root_axis_angle", [None])[0], device=device
    )
    md.object_root_position = _as_tensor(
        data.get("object_root_position", [None])[0], device=device
    )
    obj_pos = _as_tensor(data["object_body_position"][0], device=device)
    obj_quat = _as_tensor(data["object_body_wxyz"][0], device=device)
    md.object_body_position = obj_pos
    md.object_body_wxyz = obj_quat
    # Primary body (index 0).
    if obj_pos is not None and obj_pos.ndim == 3:
        md.object_pos_w = obj_pos[:, 0].contiguous()
    elif obj_pos is not None:
        md.object_pos_w = obj_pos
    if obj_quat is not None and obj_quat.ndim == 3:
        md.object_quat_w = obj_quat[:, 0].contiguous()
    elif obj_quat is not None:
        md.object_quat_w = obj_quat

    # ---- Hands (on-disk + flattened) --------------------------------------
    hand_sides = list(data.get("hand_sides", [[]])[0] or [])
    md.hand_sides = hand_sides
    md.hand_frame_names = list(data.get("hand_frame_names", [[]])[0] or [])
    md.hand_finger_joint_names = list(
        data.get("hand_finger_joint_names", [[]])[0] or []
    )
    # Convert per-side tensor lists.
    per_side_frames = data.get("hand_frames_w", [[]])[0] or []
    md.hand_frames_w = [_as_tensor(x, device=device) for x in per_side_frames]
    per_side_fj = data.get("hand_finger_joints", [[]])[0] or []
    md.hand_finger_joints = [_as_tensor(x, device=device) for x in per_side_fj]

    for side in ("left", "right"):
        if side not in hand_sides:
            continue
        idx = hand_sides.index(side)
        # Frame names, frames.
        frame_names = (
            md.hand_frame_names[idx] if idx < len(md.hand_frame_names) else None
        )
        frames = md.hand_frames_w[idx] if idx < len(md.hand_frames_w) else None
        setattr(md, f"{side}_hand_frame_names", list(frame_names or []) or None)
        setattr(md, f"{side}_hand_frames", frames)

        # Finger joints / names.
        fj_names = (
            md.hand_finger_joint_names[idx]
            if idx < len(md.hand_finger_joint_names)
            else None
        )
        fj = md.hand_finger_joints[idx] if idx < len(md.hand_finger_joints) else None
        setattr(md, f"{side}_finger_joint_names", list(fj_names or []) or None)
        setattr(md, f"{side}_finger_joints", fj)

        # Wrist pose is derived from the corresponding ee slot when we can.
        # ee_link_names convention: produce `[left_*, right_*]` in order; the
        # side->ee mapping is left-first when both sides are present.
        if md.ee_pose_w is not None and md.ee_link_names:
            # Match side to the first ee link whose name contains the side.
            matches = [
                i for i, n in enumerate(md.ee_link_names) if side in (n or "").lower()
            ]
            if matches:
                ee_idx = matches[0]
                pose = md.ee_pose_w[:, ee_idx, :]
                setattr(md, f"{side}_wrist_position", pose[:, 0:3].contiguous())
                setattr(md, f"{side}_wrist_wxyz", pose[:, 3:7].contiguous())

    # ---- Contacts ---------------------------------------------------------
    md.hand_contact_link_names = list(
        data.get("hand_contact_link_names", [[]])[0] or []
    )
    for disk_col, attr in (
        ("hand_link_contact_positions", "link_contact_positions"),
        ("hand_link_contact_normals", "link_contact_normals"),
        ("hand_object_contact_positions", "object_contact_positions"),
        ("hand_object_contact_normals", "object_contact_normals"),
        ("hand_object_contact_part_ids", "object_contact_part_ids"),
        ("hand_contact_active", "hand_contact_active"),
    ):
        per_side = data.get(disk_col, [[]])[0] or []
        tensors = [_as_tensor(x, device=device) for x in per_side]
        setattr(md, disk_col, tensors)
        for side in ("left", "right"):
            if side in hand_sides:
                idx = hand_sides.index(side)
                if idx < len(tensors):
                    # Flattened attribute names follow the legacy tracking_command.py
                    # convention: `left_link_contact_positions` etc.
                    if disk_col == "hand_contact_active":
                        setattr(md, f"{side}_hand_contact_active", tensors[idx])
                    else:
                        setattr(md, f"{side}_{attr}", tensors[idx])

    # ---- Source raw -------------------------------------------------------
    md.source_kind = data.get("source_kind", [""])[0] or ""
    md.source_payload = data.get("source_payload", [b""])[0] or b""
    md.source_joint_names = list(data.get("source_joint_names", [[]])[0] or [])

    # ---- Diagnostics -----------------------------------------------------
    md.ik_error_per_frame = _as_tensor(
        data.get("ik_error_per_frame", [None])[0], device=device
    )
    md.ik_num_iterations = _as_tensor(
        data.get("ik_num_iterations", [None])[0], device=device
    )
    md.frame_task_errors = _as_tensor(
        data.get("frame_task_errors", [None])[0], device=device
    )

    return md
