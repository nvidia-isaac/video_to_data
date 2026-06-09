# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Unit tests for the `motion_v1` unified motion schema.

Covers:
    U1. Schema round-trip with every optional group populated.
    U2. Minimal single-robot file (only required groups) is loadable.
    U3. `schema_version` mismatch raises `SchemaVersionMismatch`.
    U4. `hand_sides`-indexed alignment for single/bimanual.
    U5. Quaternion convention guard (wxyz vs xyzw).
    U6. `ee_pose_w` shape invariant for E in {1, 2, 3}.
    K1-K5. `motion_kind` validation: dual-hand round-trip, missing/empty/
           unknown kind, single-robot/dual-hand required-field enforcement,
           per-side alignment.

Run with pytest or as a script (the latter is nice inside the isaac-sim
container where pytest may not be available):

    pytest tests/test_motion_schema.py
    python tests/test_motion_schema.py
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Any

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
    """Construct a single-robot MotionData with only the REQUIRED fields populated."""
    return MotionData(
        sequence_id="seq_unit_test",
        robot_name="test_robot",
        motion_kind="single_robot",
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


def _minimal_dual_hand_motion_data(
    t: int = 4,
    k: int = 3,
    jf: int = 4,
    num_bodies: int = 1,
) -> MotionData:
    """Construct a dual-hand MotionData mirroring the Dex3 producer shape.

    Whole-body joint state is intentionally left empty; required validation
    must succeed for `motion_kind="dual_hand"` regardless of `robot_*` fields.
    """
    sides = ["left", "right"]
    return MotionData(
        sequence_id="seq_dual_hand",
        robot_name="dex3",
        motion_kind="dual_hand",
        source_dataset="synthetic",
        raw_motion_file="memory://unit_test",
        fps=30.0,
        coord_frame="robot_base_z_up",
        ee_link_names=["left_wrist_link", "right_wrist_link"],
        ee_pose_w=_pose7_series(t, 2),
        object_body_names=[f"body_{i}" for i in range(num_bodies)],
        object_body_position=[
            [[0.3, 0.0, 0.4] for _ in range(num_bodies)] for _ in range(t)
        ],
        object_body_wxyz=[
            [[1.0, 0.0, 0.0, 0.0] for _ in range(num_bodies)] for _ in range(t)
        ],
        hand_sides=sides,
        hand_frame_names=[[f"{side}_frame_{i}" for i in range(k)] for side in sides],
        hand_frames_w=[_pose7_series(t, k), _pose7_series(t, k)],
        hand_finger_joint_names=[
            [f"{side}_fj_{i}" for i in range(jf)] for side in sides
        ],
        hand_finger_joints=[
            [[0.0 for _ in range(jf)] for _ in range(t)],
            [[0.0 for _ in range(jf)] for _ in range(t)],
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
    md.source_dataset = "soma"
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
    md.source_kind = "soma"
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
    assert loaded.source_dataset == "soma"
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
    assert loaded.source_kind == "soma"
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
    md.ee_pose_w = None  # common-required, removed
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
# K1-K5. motion_kind validation
# ---------------------------------------------------------------------------


def test_dual_hand_round_trip(tmp_path: Path) -> None:
    """Dex3-style dual-hand file round-trips without whole-body joints."""
    md = _minimal_dual_hand_motion_data()
    save_motion_parquet(md, root_path=str(tmp_path))
    partition_dir = (
        tmp_path / f"sequence_id={md.sequence_id}" / f"robot_name={md.robot_name}"
    )
    loaded = load_motion_data_parquet(str(partition_dir))

    assert loaded.motion_kind == "dual_hand"
    assert loaded.robot_joint_names == []
    assert loaded.hand_sides == ["left", "right"]
    assert loaded.left_hand_frames is not None
    assert loaded.right_hand_frames is not None
    assert loaded.left_finger_joints is not None
    assert loaded.right_finger_joints is not None
    # Whole-body fields stay None for dual-hand parquets.
    assert loaded.robot_root_position is None
    assert loaded.robot_joint_positions is None


def test_single_robot_missing_joints_raises(tmp_path: Path) -> None:
    """`motion_kind=single_robot` with empty robot_joint_names raises."""
    md = _minimal_motion_data()
    md.robot_joint_names = []
    md.robot_joint_positions = []
    try:
        save_motion_parquet(md, root_path=str(tmp_path))
    except ValueError as exc:
        assert "robot_joint_names" in str(exc)
        return
    raise AssertionError("expected ValueError for single_robot without joints")


def test_dual_hand_missing_hand_sides_raises(tmp_path: Path) -> None:
    """`motion_kind=dual_hand` without `hand_sides` raises on write."""
    md = _minimal_dual_hand_motion_data()
    md.hand_sides = []
    try:
        save_motion_parquet(md, root_path=str(tmp_path))
    except ValueError as exc:
        assert "hand_sides" in str(exc)
        return
    raise AssertionError("expected ValueError for dual_hand without hand_sides")


def test_dual_hand_misaligned_per_side_raises(tmp_path: Path) -> None:
    """Per-side outer length must equal len(hand_sides)."""
    md = _minimal_dual_hand_motion_data()
    md.hand_finger_joints = [md.hand_finger_joints[0]]
    try:
        save_motion_parquet(md, root_path=str(tmp_path))
    except ValueError as exc:
        assert "hand_finger_joints" in str(exc)
        assert "align" in str(exc)
        return
    raise AssertionError(
        "expected ValueError for dual_hand with misaligned per-side data"
    )


def test_missing_motion_kind_raises_on_write(tmp_path: Path) -> None:
    """Writer rejects a MotionData with empty motion_kind."""
    md = _minimal_motion_data()
    md.motion_kind = ""
    try:
        save_motion_parquet(md, root_path=str(tmp_path))
    except ValueError as exc:
        assert "motion_kind" in str(exc)
        return
    raise AssertionError("expected ValueError for missing motion_kind on write")


def test_missing_motion_kind_raises_on_read(tmp_path: Path) -> None:
    """Reader rejects a parquet whose motion_kind column is empty."""
    md = _minimal_motion_data()
    save_motion_parquet(md, root_path=str(tmp_path))
    partition_dir = (
        tmp_path / f"sequence_id={md.sequence_id}" / f"robot_name={md.robot_name}"
    )
    parquet_files = list(partition_dir.glob("*.parquet"))
    assert len(parquet_files) == 1
    file_path = parquet_files[0]
    table = pq.ParquetFile(str(file_path)).read()
    pydict = table.to_pydict()
    pydict["sequence_id"] = [md.sequence_id]
    pydict["robot_name"] = [md.robot_name]
    pydict["motion_kind"] = [""]
    new_table = pa.Table.from_pydict(pydict, schema=build_schema())
    pq.write_table(new_table, str(file_path))

    try:
        load_motion_data_parquet(str(partition_dir))
    except MissingRequiredField as exc:
        assert "motion_kind" in exc.missing
        return
    raise AssertionError("expected MissingRequiredField for empty motion_kind")


def test_unknown_motion_kind_raises_on_write(tmp_path: Path) -> None:
    """Writer rejects an unrecognised motion_kind value."""
    md = _minimal_motion_data()
    md.motion_kind = "quadruped"
    try:
        save_motion_parquet(md, root_path=str(tmp_path))
    except ValueError as exc:
        assert "quadruped" in str(exc) or "Unknown motion_kind" in str(exc)
        return
    raise AssertionError("expected ValueError for unknown motion_kind")


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
    ("K1 dual-hand round-trip", test_dual_hand_round_trip),
    ("K2 single_robot missing joints raises", test_single_robot_missing_joints_raises),
    (
        "K3 dual_hand missing hand_sides raises",
        test_dual_hand_missing_hand_sides_raises,
    ),
    (
        "K4 dual_hand misaligned per-side raises",
        test_dual_hand_misaligned_per_side_raises,
    ),
    (
        "K5 missing motion_kind raises on write",
        test_missing_motion_kind_raises_on_write,
    ),
    ("K5 missing motion_kind raises on read", test_missing_motion_kind_raises_on_read),
    (
        "K5 unknown motion_kind raises on write",
        test_unknown_motion_kind_raises_on_write,
    ),
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
