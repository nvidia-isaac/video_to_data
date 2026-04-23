# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Unit tests for the `motion_v1` unified motion schema.

Covers (from the plan's test plan):
    U1. Schema round-trip with every optional group populated.
    U2. Minimal file (only required groups) is loadable.
    U3. `schema_version` mismatch raises `SchemaVersionMismatch`.
    U4. `hand_sides`-indexed alignment for single/bimanual.
    U5. Quaternion convention guard (wxyz vs xyzw).
    U6. `ee_pose_w` shape invariant for E in {1, 2, 3}.

Run with pytest or as a script (the latter is nice inside the isaac-sim
container where pytest may not be available):

    pytest tests/test_motion_schema.py
    python tests/test_motion_schema.py
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from robotic_grounding.motion_schema import (
    SCHEMA_VERSION,
    MissingRequiredField,
    MotionData,
    SchemaVersionMismatch,
    build_schema,
    load_motion_data_parquet,
    save_motion_parquet,
)
from robotic_grounding.motion_schema.writer import _row_dict


def _load_migrator_module() -> Any:
    """Load scripts/motion_schema/migrate_to_v1.py without requiring it on PYTHONPATH."""
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "motion_schema"
        / "migrate_to_v1.py"
    )
    spec = importlib.util.spec_from_file_location("migrate_to_v1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load migrator at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _identity_wxyz(t: int) -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0] for _ in range(t)]


def _pose7_series(t: int, e: int) -> list[list[list[float]]]:
    """Build a (T, E, 7) pose series centered at origin with identity rotation."""
    return [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for _ in range(e)] for _ in range(t)]


def _minimal_motion_data(
    t: int = 4,
    j: int = 5,
    e: int = 2,
    num_bodies: int = 1,
) -> MotionData:
    """Construct a MotionData with only the REQUIRED training fields populated."""
    return MotionData(
        sequence_id="seq_unit_test",
        robot_name="test_robot",
        source_dataset="synthetic",
        raw_motion_file="memory://unit_test",
        fps=30.0,
        coord_frame="robot_base_z_up",
        robot_joint_names=[f"joint_{i}" for i in range(j)],
        robot_root_position=[[0.0, 0.0, 0.8] for _ in range(t)],
        robot_root_wxyz=_identity_wxyz(t),
        robot_joint_positions=[[0.0 for _ in range(j)] for _ in range(t)],
        ee_link_names=[f"ee_{i}" for i in range(e)],
        ee_pose_w=_pose7_series(t, e),
        object_body_names=[f"body_{i}" for i in range(num_bodies)],
        object_body_position=[
            [[0.3, 0.0, 0.4] for _ in range(num_bodies)] for _ in range(t)
        ],
        object_body_wxyz=[
            [[1.0, 0.0, 0.0, 0.0] for _ in range(num_bodies)] for _ in range(t)
        ],
    )


def _fully_populated_motion_data(t: int = 4) -> MotionData:
    """Construct a MotionData with every optional group populated."""
    j = 3
    e = 2
    k = 3  # hand frames per side
    jf = 4  # finger joints per side
    n = 2  # contact links per side

    md = _minimal_motion_data(t=t, j=j, e=e, num_bodies=1)
    md.source_dataset = "nvhuman"
    md.ee_link_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    md.safe_object_name = "body_0"
    md.object_name = "body_0"
    md.safe_object_body_names = ["body_0"]
    md.object_mesh_paths = ["mesh://0"]
    md.object_urdf_paths = ["urdf://0"]
    md.object_mesh_radius = [0.05]
    md.object_articulation = [0.0 for _ in range(t)]
    md.object_root_axis_angle = [[0.0, 0.0, 0.0] for _ in range(t)]
    md.object_root_position = [[0.3, 0.0, 0.4] for _ in range(t)]

    # Hands
    md.hand_sides = ["left", "right"]
    md.hand_frame_names = [
        [f"{side}_frame_{i}" for i in range(k)] for side in md.hand_sides
    ]
    md.hand_frames_w = [
        _pose7_series(t, k),
        _pose7_series(t, k),
    ]
    md.hand_finger_joint_names = [
        [f"{side}_fj_{i}" for i in range(jf)] for side in md.hand_sides
    ]
    md.hand_finger_joints = [
        [[0.1 for _ in range(jf)] for _ in range(t)],
        [[0.2 for _ in range(jf)] for _ in range(t)],
    ]

    # Contacts
    md.hand_contact_link_names = [
        [f"{side}_contact_{i}" for i in range(n)] for side in md.hand_sides
    ]
    zero_vec3_series = [[[0.0, 0.0, 0.0] for _ in range(n)] for _ in range(t)]
    zero_part_ids = [[0 for _ in range(n)] for _ in range(t)]
    md.hand_link_contact_positions = [zero_vec3_series, zero_vec3_series]
    md.hand_link_contact_normals = [zero_vec3_series, zero_vec3_series]
    md.hand_object_contact_positions = [zero_vec3_series, zero_vec3_series]
    md.hand_object_contact_normals = [zero_vec3_series, zero_vec3_series]
    md.hand_object_contact_part_ids = [zero_part_ids, zero_part_ids]
    md.hand_contact_active = [[0.0 for _ in range(t)] for _ in md.hand_sides]

    # Source + diagnostics
    md.source_kind = "nvhuman"
    md.source_payload = b"\x00\x01\x02"
    md.source_joint_names = ["n0", "n1"]
    md.ik_error_per_frame = [0.0 for _ in range(t)]
    md.ik_num_iterations = [1 for _ in range(t)]
    md.frame_task_errors = [[0.0, 0.0] for _ in range(t)]

    return md


def _round_trip(md: MotionData, tmp_dir: Path) -> MotionData:
    save_motion_parquet(md, root_path=str(tmp_dir))
    partition_dir = (
        tmp_dir / f"sequence_id={md.sequence_id}" / f"robot_name={md.robot_name}"
    )
    return load_motion_data_parquet(str(partition_dir))


# ---------------------------------------------------------------------------
# U1. Schema round-trip with every optional group populated
# ---------------------------------------------------------------------------


def test_u1_full_roundtrip(tmp_path: Path) -> None:
    """Every optional group populated, save then load, compare key fields."""
    md = _fully_populated_motion_data()
    loaded = _round_trip(md, tmp_path)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.sequence_id == md.sequence_id
    assert loaded.robot_name == md.robot_name
    assert loaded.source_dataset == "nvhuman"
    assert abs(loaded.fps - md.fps) < 1e-6

    # Robot state
    assert loaded.robot_joint_names == md.robot_joint_names
    t = len(md.robot_root_position)
    j = len(md.robot_joint_names)
    assert tuple(loaded.robot_root_position.shape) == (t, 3)
    assert tuple(loaded.robot_root_wxyz.shape) == (t, 4)
    assert tuple(loaded.robot_joint_positions.shape) == (t, j)

    # EE
    assert loaded.ee_link_names == md.ee_link_names
    assert loaded.ee_pose_w is not None
    assert tuple(loaded.ee_pose_w.shape) == (4, 2, 7)
    assert tuple(loaded.ee_pos_w.shape) == (4, 2, 3)
    assert tuple(loaded.ee_quat_w.shape) == (4, 2, 4)

    # Object
    assert loaded.object_body_names == md.object_body_names
    assert loaded.object_mesh_paths == md.object_mesh_paths
    assert tuple(loaded.object_body_position.shape) == (4, 1, 3)
    assert tuple(loaded.object_body_wxyz.shape) == (4, 1, 4)
    assert loaded.object_pos_w is not None
    assert tuple(loaded.object_pos_w.shape) == (4, 3)

    # Hands — on-disk view
    assert loaded.hand_sides == ["left", "right"]
    assert loaded.hand_frame_names == md.hand_frame_names
    assert len(loaded.hand_frames_w) == 2

    # Hands — flattened view
    assert loaded.left_hand_frames is not None
    assert loaded.right_hand_frames is not None
    assert tuple(loaded.left_hand_frames.shape) == (4, 3, 7)
    assert loaded.left_hand_frame_names == md.hand_frame_names[0]
    assert loaded.right_hand_frame_names == md.hand_frame_names[1]
    assert tuple(loaded.left_finger_joints.shape) == (4, 4)
    assert tuple(loaded.right_finger_joints.shape) == (4, 4)

    # Wrists derived from ee
    assert loaded.left_wrist_position is not None
    assert tuple(loaded.left_wrist_position.shape) == (4, 3)
    assert tuple(loaded.right_wrist_wxyz.shape) == (4, 4)

    # Contacts
    assert loaded.left_link_contact_positions is not None
    assert loaded.right_object_contact_part_ids is not None
    assert tuple(loaded.left_link_contact_positions.shape) == (4, 2, 3)

    # Source + diagnostics
    assert loaded.source_kind == "nvhuman"
    assert loaded.source_payload == b"\x00\x01\x02"
    assert loaded.ik_error_per_frame is not None


# ---------------------------------------------------------------------------
# U2. Minimal file — only required groups are populated
# ---------------------------------------------------------------------------


def test_u2_minimal_file_loads(tmp_path: Path) -> None:
    """Round-trip a MotionData with no hands, no contacts, no source."""
    md = _minimal_motion_data()
    loaded = _round_trip(md, tmp_path)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.robot_root_position is not None
    assert loaded.robot_joint_positions is not None
    assert loaded.ee_pose_w is not None

    # Optional groups must all be None / empty on the flattened view.
    for attr in (
        "left_wrist_position",
        "left_wrist_wxyz",
        "right_wrist_position",
        "right_wrist_wxyz",
        "left_hand_frames",
        "right_hand_frames",
        "left_finger_joints",
        "right_finger_joints",
        "left_link_contact_positions",
        "right_link_contact_positions",
        "left_object_contact_part_ids",
        "left_hand_contact_active",
        "ik_error_per_frame",
    ):
        assert getattr(loaded, attr) is None, f"expected {attr} to be None"

    assert loaded.hand_sides == []
    assert loaded.source_kind == ""
    assert loaded.source_payload == b""


# ---------------------------------------------------------------------------
# U3. schema_version enforcement
# ---------------------------------------------------------------------------


def _write_file_with_version(path: Path, version: str) -> Path:
    """Build a parquet with a fake schema_version but otherwise valid columns."""
    md = _minimal_motion_data()
    md.schema_version = version  # bypass writer's forced SCHEMA_VERSION
    # We have to build the row dict ourselves because save_motion_parquet
    # always overwrites schema_version to SCHEMA_VERSION.
    row = _row_dict(md)
    row["schema_version"] = [version]
    table = pa.Table.from_pydict(row, schema=build_schema())
    partition_dir = (
        path / f"sequence_id={md.sequence_id}" / f"robot_name={md.robot_name}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    file_path = partition_dir / "data.parquet"
    pq.write_table(table, str(file_path))
    return file_path


def test_u3_version_mismatch_raises(tmp_path: Path) -> None:
    """Reader raises SchemaVersionMismatch with an actionable message."""
    file_path = _write_file_with_version(tmp_path, version="motion_v0")
    try:
        load_motion_data_parquet(str(file_path))
    except SchemaVersionMismatch as exc:
        assert "motion_v0" in str(exc)
        assert SCHEMA_VERSION in str(exc)
        assert str(file_path) in str(exc)
        return
    raise AssertionError("expected SchemaVersionMismatch to be raised")


def test_u3_missing_version_raises(tmp_path: Path) -> None:
    """Reader raises if schema_version is empty string."""
    file_path = _write_file_with_version(tmp_path, version="")
    try:
        load_motion_data_parquet(str(file_path))
    except SchemaVersionMismatch:
        return
    raise AssertionError("expected SchemaVersionMismatch for empty version")


# ---------------------------------------------------------------------------
# U4. hand_sides alignment (single-hand right-only)
# ---------------------------------------------------------------------------


def test_u4_single_hand_right_only(tmp_path: Path) -> None:
    """A right-only file populates right_* attributes and leaves left_* None."""
    t, k, jf, n = 4, 2, 3, 2
    md = _minimal_motion_data(t=t)
    md.ee_link_names = ["right_wrist_yaw_link"]
    md.ee_pose_w = _pose7_series(t, 1)

    md.hand_sides = ["right"]
    md.hand_frame_names = [[f"right_frame_{i}" for i in range(k)]]
    md.hand_frames_w = [_pose7_series(t, k)]
    md.hand_finger_joint_names = [[f"right_fj_{i}" for i in range(jf)]]
    md.hand_finger_joints = [[[0.0 for _ in range(jf)] for _ in range(t)]]
    md.hand_contact_link_names = [[f"right_contact_{i}" for i in range(n)]]
    zero_vec3 = [[[0.0, 0.0, 0.0] for _ in range(n)] for _ in range(t)]
    md.hand_link_contact_positions = [zero_vec3]
    md.hand_link_contact_normals = [zero_vec3]
    md.hand_object_contact_positions = [zero_vec3]
    md.hand_object_contact_normals = [zero_vec3]
    md.hand_object_contact_part_ids = [[[0 for _ in range(n)] for _ in range(t)]]
    md.hand_contact_active = [[0.0 for _ in range(t)]]

    loaded = _round_trip(md, tmp_path)

    # Right-side populated.
    assert loaded.right_hand_frames is not None
    assert tuple(loaded.right_hand_frames.shape) == (t, k, 7)
    assert loaded.right_finger_joints is not None
    assert tuple(loaded.right_finger_joints.shape) == (t, jf)
    assert loaded.right_link_contact_positions is not None
    assert loaded.right_hand_contact_active is not None
    assert loaded.right_wrist_position is not None

    # Left-side strictly None (no blind [0, 1] indexing).
    for attr in (
        "left_hand_frames",
        "left_hand_frame_names",
        "left_finger_joints",
        "left_link_contact_positions",
        "left_object_contact_positions",
        "left_hand_contact_active",
        "left_wrist_position",
        "left_wrist_wxyz",
    ):
        assert (
            getattr(loaded, attr) is None
        ), f"expected {attr} to be None in single-hand case"

    assert loaded.hand_sides == ["right"]


# ---------------------------------------------------------------------------
# U5. Quaternion convention guard
# ---------------------------------------------------------------------------


def test_u5_writer_rejects_xyzw(tmp_path: Path) -> None:
    """Writer raises when quaternions look like xyzw instead of wxyz."""
    md = _minimal_motion_data()
    # Build an xyzw-like series (real part last). This should fail the guard.
    md.robot_root_wxyz = [[0.0, 0.0, 0.0, 1.0] for _ in range(4)]
    try:
        save_motion_parquet(md, root_path=str(tmp_path))
    except ValueError as exc:
        assert "wxyz" in str(exc)
        return
    raise AssertionError("expected ValueError when writing xyzw-ordered quaternions")


def test_u5_writer_accepts_plausible_rotations(tmp_path: Path) -> None:
    """Writer accepts quaternions with moderate rotations (w still dominant enough)."""
    md = _minimal_motion_data()
    # 45deg about z axis: w=cos(22.5deg)~=0.924, z=sin(22.5deg)~=0.383. Still wxyz-ish.
    w, z = 0.924, 0.383
    md.robot_root_wxyz = [[w, 0.0, 0.0, z] for _ in range(4)]
    save_motion_parquet(md, root_path=str(tmp_path))  # should not raise


# ---------------------------------------------------------------------------
# U6. ee_pose_w shape invariant (E in {1, 2, 3})
# ---------------------------------------------------------------------------


def test_u6_variable_num_ee(tmp_path: Path) -> None:
    """ee_pose_w round-trips for E in {1, 2, 3} without reshape regressions."""
    for e in (1, 2, 3):
        t = 3
        md = _minimal_motion_data(t=t, e=e)
        md.ee_link_names = [f"ee_{i}" for i in range(e)]
        md.ee_pose_w = _pose7_series(t, e)
        sub_dir = tmp_path / f"e_{e}"
        loaded = _round_trip(md, sub_dir)
        assert tuple(loaded.ee_pose_w.shape) == (t, e, 7), f"E={e}"
        assert tuple(loaded.ee_pos_w.shape) == (t, e, 3), f"E={e}"
        assert tuple(loaded.ee_quat_w.shape) == (t, e, 4), f"E={e}"


# ---------------------------------------------------------------------------
# Missing-required-field behaviour
# ---------------------------------------------------------------------------


def test_missing_required_field_raises_on_write(tmp_path: Path) -> None:
    """Writer fails fast with a pointer to the missing required field."""
    md = _minimal_motion_data()
    md.ee_pose_w = None  # required, removed
    try:
        save_motion_parquet(md, root_path=str(tmp_path))
    except ValueError as exc:
        assert "ee_pose_w" in str(exc)
        return
    raise AssertionError("expected ValueError from writer")


def test_missing_required_field_raises_on_read(tmp_path: Path) -> None:
    """If a file lacks a required column at read time, reader raises."""
    md = _minimal_motion_data()
    # Write via the writer but then truncate the column.
    save_motion_parquet(md, root_path=str(tmp_path))
    # Rewrite with one required column cleared.
    partition_dir = (
        tmp_path / f"sequence_id={md.sequence_id}" / f"robot_name={md.robot_name}"
    )
    parquet_files = list(partition_dir.glob("*.parquet"))
    assert len(parquet_files) == 1
    file_path = parquet_files[0]
    table = pq.ParquetFile(str(file_path)).read()
    pydict = table.to_pydict()
    # pq.write_to_dataset strips partition columns from the file body;
    # rebuild them here before writing the full-schema table back.
    pydict["sequence_id"] = [md.sequence_id]
    pydict["robot_name"] = [md.robot_name]
    pydict["ee_pose_w"] = [None]
    new_table = pa.Table.from_pydict(pydict, schema=build_schema())
    pq.write_table(new_table, str(file_path))

    try:
        load_motion_data_parquet(str(partition_dir))
    except MissingRequiredField as exc:
        assert "ee_pose_w" in exc.missing
        return
    raise AssertionError("expected MissingRequiredField")


# ---------------------------------------------------------------------------
# Migrator tests (M1-M4)
# ---------------------------------------------------------------------------


def _build_fake_planner_parquet(tmp_path: Path, t: int = 4) -> Path:
    """Build a tiny planner-schema parquet at tmp_path."""
    j = 3  # body joints
    nq = 7 + j  # root pose + body joints
    qpos = np.zeros((t, nq), dtype=np.float32)
    qpos[:, 2] = 0.8  # root z
    qpos[:, 3] = 1.0  # root quat w
    qpos[:, 7] = np.linspace(0.0, 1.0, t, dtype=np.float32)

    qpos_layout = json.dumps(
        {
            "root_pos": [0, 3],
            "root_quat_wxyz": [3, 7],
            "body_joints": [7, nq],
        }
    )
    joint_names = [
        "root_x",
        "root_y",
        "root_z",
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
        "j0",
        "j1",
        "j2",
    ]
    ee_pos = np.zeros((t, 2, 3), dtype=np.float32)
    ee_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (t, 2, 1))
    row = {
        "sequence_id": ["planner_fixture"],
        "robot_name": ["g1"],
        "raw_motion_file": ["memory://planner_fixture"],
        "fps": [30.0],
        "qpos": [qpos.tolist()],
        "qpos_layout": [qpos_layout],
        "joint_names": [joint_names],
        "ee_link_names": [["left_wrist_yaw_link", "right_wrist_yaw_link"]],
        "ee_pos_w": [ee_pos.tolist()],
        "ee_quat_w": [ee_quat.tolist()],
        "object_name": ["apple"],
        "safe_object_name": ["apple"],
        "object_body_names": [["apple"]],
        "safe_object_body_names": [["apple"]],
        "object_mesh_paths": [["mesh://0"]],
        "object_urdf_paths": [["urdf://0"]],
        "object_mesh_radius": [[0.05]],
        "object_articulation": [[0.0 for _ in range(t)]],
        "object_root_axis_angle": [[[0.0, 0.0, 0.0] for _ in range(t)]],
        "object_root_position": [[[0.3, 0.0, 0.4] for _ in range(t)]],
        "object_body_position": [[[[0.3, 0.0, 0.4]] for _ in range(t)]],
        "object_body_wxyz": [[[[1.0, 0.0, 0.0, 0.0]] for _ in range(t)]],
    }
    table = pa.Table.from_pydict(row)
    partition = tmp_path / "sequence_id=planner_fixture" / "robot_name=g1"
    partition.mkdir(parents=True, exist_ok=True)
    file_path = partition / "data.parquet"
    pq.write_table(table, str(file_path))
    return file_path


def _build_fake_nvhuman_g1_parquet(tmp_path: Path, t: int = 4) -> Path:
    """Build a tiny NvhumanG1Data-shaped parquet at tmp_path."""
    j = 4  # robot joints (not split the way planner splits)
    frame_names = [
        "pelvis",
        "left_hand_palm_link",
        "right_hand_palm_link",
    ]
    robot_frames = np.zeros((t, len(frame_names), 7), dtype=np.float32)
    robot_frames[..., 3] = 1.0  # identity wxyz
    row = {
        "sequence_id": ["nvhuman_g1_fixture"],
        "robot_name": ["g1"],
        "raw_motion_file": ["memory://nvhuman_g1_fixture"],
        "fps": [30.0],
        "nvhuman_betas": [[0.0] * 10],
        "robot_joint_names": [[f"j{i}" for i in range(j)]],
        "robot_frame_names": [frame_names],
        "robot_frame_task_names": [frame_names],
        "source_to_robot_scale": [1.0],
        "robot_root_position": [[[0.0, 0.0, 0.8] for _ in range(t)]],
        "robot_root_wxyz": [[[1.0, 0.0, 0.0, 0.0] for _ in range(t)]],
        "robot_joint_positions": [[[0.0] * j for _ in range(t)]],
        "robot_frames": [robot_frames.tolist()],
        "robot_frame_task_errors": [[[0.0] * len(frame_names) for _ in range(t)]],
        "robot_ik_error": [[0.0] * t],
        "robot_num_optimization_iterations": [[1] * t],
        "object_name": ["bottle"],
        "safe_object_name": ["bottle"],
        "object_body_names": [["bottle"]],
        "safe_object_body_names": [["bottle"]],
        "object_mesh_paths": [[]],
        "object_urdf_paths": [[]],
        "object_mesh_radius": [[0.05]],
        "object_articulation": [[0.0 for _ in range(t)]],
        "object_root_axis_angle": [[[0.0, 0.0, 0.0] for _ in range(t)]],
        "object_root_position": [[[0.4, 0.0, 0.5] for _ in range(t)]],
        "object_body_position": [[[[0.4, 0.0, 0.5]] for _ in range(t)]],
        "object_body_wxyz": [[[[1.0, 0.0, 0.0, 0.0]] for _ in range(t)]],
        "nvhuman_joints": [[[[0.0, 0.0, 0.0] for _ in range(93)] for _ in range(t)]],
        "nvhuman_joints_wxyz": [
            [[[1.0, 0.0, 0.0, 0.0] for _ in range(93)] for _ in range(t)]
        ],
        "nvhuman_head_translation": [[[0.0, 0.0, 1.7] for _ in range(t)]],
        "nvhuman_head_wxyz": [[[1.0, 0.0, 0.0, 0.0] for _ in range(t)]],
        "nvhuman_root_translation": [[[0.0, 0.0, 0.9] for _ in range(t)]],
        "nvhuman_root_wxyz": [[[1.0, 0.0, 0.0, 0.0] for _ in range(t)]],
    }
    table = pa.Table.from_pydict(row)
    partition = tmp_path / "sequence_id=nvhuman_g1_fixture" / "robot_name=g1"
    partition.mkdir(parents=True, exist_ok=True)
    file_path = partition / "data.parquet"
    pq.write_table(table, str(file_path))
    return file_path


def test_m1_migrator_idempotent(tmp_path: Path) -> None:
    """Running the migrator on a motion_v1 file prints SKIP and does not rewrite."""
    migrator = _load_migrator_module()
    md = _minimal_motion_data()
    save_motion_parquet(md, root_path=str(tmp_path))
    partition_dir = (
        tmp_path / f"sequence_id={md.sequence_id}" / f"robot_name={md.robot_name}"
    )
    parquet_files = list(partition_dir.glob("*.parquet"))
    assert len(parquet_files) == 1
    file_path = parquet_files[0]
    original_mtime = file_path.stat().st_mtime

    # Call _migrate_one directly (CLI is tested via the same function).
    msg = migrator._migrate_one(
        file_path,
        schema="auto",
        output_root=tmp_path,
        dry_run=False,
    )
    assert msg.startswith("SKIP"), f"expected SKIP, got: {msg}"
    assert (
        file_path.stat().st_mtime == original_mtime
    ), "migrator rewrote an up-to-date file"


def test_m2_planner_adapter_bit_equivalent(tmp_path: Path) -> None:
    """Planner-adapter output must decode to the same qpos slices after migration."""
    migrator = _load_migrator_module()
    planner_parquet = _build_fake_planner_parquet(tmp_path / "input")

    # Read the source data for later comparison.
    source = pq.ParquetFile(str(planner_parquet)).read().to_pydict()
    qpos = np.asarray(source["qpos"][0], dtype=np.float32)
    layout = json.loads(source["qpos_layout"][0])

    out_root = tmp_path / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    migrator._migrate_one(
        planner_parquet, schema="planner", output_root=out_root, dry_run=False
    )

    loaded = load_motion_data_parquet(
        str(out_root / "sequence_id=planner_fixture" / "robot_name=g1")
    )
    assert loaded.schema_version == SCHEMA_VERSION

    # Field-level equivalence.
    rp = np.asarray(
        loaded.robot_root_position.cpu().numpy()
        if hasattr(loaded.robot_root_position, "cpu")
        else loaded.robot_root_position
    )
    np.testing.assert_allclose(
        rp, qpos[:, layout["root_pos"][0] : layout["root_pos"][1]]
    )

    rw = np.asarray(
        loaded.robot_root_wxyz.cpu().numpy()
        if hasattr(loaded.robot_root_wxyz, "cpu")
        else loaded.robot_root_wxyz
    )
    np.testing.assert_allclose(
        rw, qpos[:, layout["root_quat_wxyz"][0] : layout["root_quat_wxyz"][1]]
    )

    rj = np.asarray(
        loaded.robot_joint_positions.cpu().numpy()
        if hasattr(loaded.robot_joint_positions, "cpu")
        else loaded.robot_joint_positions
    )
    np.testing.assert_allclose(
        rj, qpos[:, layout["body_joints"][0] : layout["body_joints"][1]]
    )

    # EE pose round-trip.
    ee_pos = np.asarray(
        loaded.ee_pos_w.cpu().numpy()
        if hasattr(loaded.ee_pos_w, "cpu")
        else loaded.ee_pos_w
    )
    np.testing.assert_allclose(
        ee_pos, np.asarray(source["ee_pos_w"][0], dtype=np.float32)
    )
    ee_quat = np.asarray(
        loaded.ee_quat_w.cpu().numpy()
        if hasattr(loaded.ee_quat_w, "cpu")
        else loaded.ee_quat_w
    )
    np.testing.assert_allclose(
        ee_quat, np.asarray(source["ee_quat_w"][0], dtype=np.float32)
    )


def test_m3_nvhuman_g1_adapter_bit_equivalent(tmp_path: Path) -> None:
    """NvhumanG1 adapter populates robot_root_*, joint_positions, EE and object."""
    migrator = _load_migrator_module()
    src = _build_fake_nvhuman_g1_parquet(tmp_path / "input")
    source = pq.ParquetFile(str(src)).read().to_pydict()

    out_root = tmp_path / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    migrator._migrate_one(src, schema="nvhuman_g1", output_root=out_root, dry_run=False)

    loaded = load_motion_data_parquet(
        str(out_root / "sequence_id=nvhuman_g1_fixture" / "robot_name=g1")
    )
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.robot_joint_names == source["robot_joint_names"][0]

    rp = np.asarray(
        loaded.robot_root_position.cpu().numpy()
        if hasattr(loaded.robot_root_position, "cpu")
        else loaded.robot_root_position
    )
    np.testing.assert_allclose(rp, np.asarray(source["robot_root_position"][0]))

    rj = np.asarray(
        loaded.robot_joint_positions.cpu().numpy()
        if hasattr(loaded.robot_joint_positions, "cpu")
        else loaded.robot_joint_positions
    )
    np.testing.assert_allclose(rj, np.asarray(source["robot_joint_positions"][0]))

    # EE mapped to palm frames (first frame is pelvis, skipped).
    assert loaded.ee_link_names == ["left_hand_palm_link", "right_hand_palm_link"]
    assert loaded.ee_pose_w is not None
    # Original frames were identity wxyz at origin for all frames.
    expected = np.asarray(source["robot_frames"][0])  # (T, K, 7)
    l_idx = source["robot_frame_names"][0].index("left_hand_palm_link")
    r_idx = source["robot_frame_names"][0].index("right_hand_palm_link")
    ee_actual = np.asarray(
        loaded.ee_pose_w.cpu().numpy()
        if hasattr(loaded.ee_pose_w, "cpu")
        else loaded.ee_pose_w
    )
    np.testing.assert_allclose(ee_actual[:, 0, :], expected[:, l_idx, :])
    np.testing.assert_allclose(ee_actual[:, 1, :], expected[:, r_idx, :])


def test_m4_source_payload_roundtrip(tmp_path: Path) -> None:
    """NVHuman source joints survive migration via source_payload."""
    migrator = _load_migrator_module()
    src = _build_fake_nvhuman_g1_parquet(tmp_path / "input")
    out_root = tmp_path / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    migrator._migrate_one(src, schema="nvhuman_g1", output_root=out_root, dry_run=False)

    loaded = load_motion_data_parquet(
        str(out_root / "sequence_id=nvhuman_g1_fixture" / "robot_name=g1")
    )
    assert loaded.source_kind == "nvhuman"
    assert loaded.source_payload, "source_payload should be non-empty"
    payload = pickle.loads(loaded.source_payload)
    assert "nvhuman_joints" in payload
    assert "nvhuman_head_translation" in payload
    np.testing.assert_allclose(
        np.asarray(payload["nvhuman_head_translation"][0]),
        np.asarray([0.0, 0.0, 1.7]),
    )


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    """--dry-run must not produce output files."""
    migrator = _load_migrator_module()
    src = _build_fake_planner_parquet(tmp_path / "input")
    out_root = tmp_path / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    msg = migrator._migrate_one(
        src, schema="planner", output_root=out_root, dry_run=True
    )
    assert msg.startswith("DRY-RUN")
    assert not any(out_root.rglob("*.parquet")), "dry-run wrote files"


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


TESTS: list[tuple[str, Any]] = [
    ("U1 full round-trip", test_u1_full_roundtrip),
    ("U2 minimal file loads", test_u2_minimal_file_loads),
    ("U3 version mismatch raises", test_u3_version_mismatch_raises),
    ("U3 missing version raises", test_u3_missing_version_raises),
    ("U4 single-hand right only", test_u4_single_hand_right_only),
    ("U5 writer rejects xyzw", test_u5_writer_rejects_xyzw),
    ("U5 writer accepts rotations", test_u5_writer_accepts_plausible_rotations),
    ("U6 variable num ee", test_u6_variable_num_ee),
    ("missing required raises on write", test_missing_required_field_raises_on_write),
    ("missing required raises on read", test_missing_required_field_raises_on_read),
    ("M1 migrator idempotent", test_m1_migrator_idempotent),
    ("M2 planner adapter equivalence", test_m2_planner_adapter_bit_equivalent),
    ("M3 nvhuman_g1 adapter equivalence", test_m3_nvhuman_g1_adapter_bit_equivalent),
    ("M4 source payload round-trip", test_m4_source_payload_roundtrip),
    ("dry-run does not write", test_dry_run_does_not_write),
]


def _run_as_script() -> int:
    """Run all tests without pytest."""
    failures = 0
    for name, fn in TESTS:
        with tempfile.TemporaryDirectory(prefix="motion_schema_test_") as tmp_dir:
            print(f"[RUN]  {name}", flush=True)
            try:
                fn(Path(tmp_dir))
                print(f"[PASS] {name}", flush=True)
            except Exception:
                print(f"[FAIL] {name}", flush=True)
                traceback.print_exc()
                failures += 1
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_as_script())
