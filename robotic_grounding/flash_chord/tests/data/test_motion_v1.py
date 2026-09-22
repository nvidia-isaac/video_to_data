# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the versioned motion_v1 single-robot loader."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flash_chord.assets import ASSETS_DIR
from flash_chord.data.motion_v1 import load_motion_v1
from flash_chord.data.reference import HandPoseReference, NamedFrameReference, NamedRobotReference, Reference

_VEGA_MIXER = str(
    ASSETS_DIR
    / "human_motion_data"
    / "arctic"
    / "arctic_processed"
    / "sequence_id=dataset_s01_mixer_use_01"
    / "robot_name=vega_sharpa"
)


def _columns() -> dict:
    return {
        "schema_version": ["motion_v1"],
        "motion_kind": ["single_robot"],
        "source_dataset": ["synthetic"],
        "raw_motion_file": ["raw/demo"],
        "fps": [2.0],
        "coord_frame": ["world"],
        "robot_joint_names": [["joint_b", "joint_a"]],
        "robot_root_position": [[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        "robot_root_wxyz": [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        "robot_joint_positions": [[[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]]],
        "ee_link_names": [["tip_b", "tip_a"]],
        "ee_pose_w": [
            [
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            ]
        ],
        "hand_sides": [["right", "left"]],
        "hand_frame_names": [[["right_index_DP"], ["left_index_DP"]]],
        "hand_frames_w": [
            [
                [[[_frame + 30.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]] for _frame in range(3)],
                [[[_frame + 40.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]] for _frame in range(3)],
            ]
        ],
        "object_name": ["object"],
        "object_body_names": [["body"]],
        "object_mesh_paths": [["mesh.obj"]],
        "object_urdf_paths": [["object.urdf"]],
        "object_mesh_radius": [[0.1]],
        "object_articulation": [[]],
        "object_body_position": [[[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]],
        "object_body_wxyz": [
            [
                [[1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0]],
            ]
        ],
        "hand_object_contact_positions": [
            [
                [[[10.0, 0.0, 0.0]], [[11.0, 0.0, 0.0]], [[12.0, 0.0, 0.0]]],
                [[[20.0, 0.0, 0.0]], [[21.0, 0.0, 0.0]], [[22.0, 0.0, 0.0]]],
            ]
        ],
        "hand_object_contact_normals": [
            [
                [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]]],
            ]
        ],
        "hand_object_contact_part_ids": [[[[1], [1], [1]], [[0], [0], [0]]]],
    }


def _write(tmp_path: Path, **overrides) -> Path:
    columns = _columns()
    columns.update({name: [value] for name, value in overrides.items()})
    path = tmp_path / "sequence_id=demo" / "robot_name=vega" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def test_loads_named_capabilities_and_preserves_source_order(tmp_path):
    reference = load_motion_v1(str(_write(tmp_path)))

    assert isinstance(reference, Reference)
    assert isinstance(reference, NamedRobotReference)
    assert isinstance(reference, NamedFrameReference)
    assert not isinstance(reference, HandPoseReference)
    assert reference.robot_joint_names() == ["joint_b", "joint_a"]
    assert reference.robot_frame_names() == [
        "tip_b",
        "tip_a",
        "right_index_DP",
        "left_index_DP",
    ]
    np.testing.assert_array_equal(reference.robot_frame_pos_w()[:, 2, 0], [30.0, 31.0, 32.0])
    np.testing.assert_array_equal(reference.robot_frame_pos_w()[:, 3, 0], [40.0, 41.0, 42.0])
    np.testing.assert_array_equal(reference.robot_joint_pos()[0], [0.0, 10.0])
    assert reference.object_articulation().shape == (3, 0)
    assert reference.object_assets()[0].root_reference_name == "body"
    assert reference.metadata.sequence_id == "demo"
    assert reference.metadata.robot_name == "vega"
    assert reference.metadata.source_dataset == "synthetic"
    assert not reference.metadata.is_resampled


def test_accepts_robot_base_z_up_world_coordinate_convention(tmp_path):
    reference = load_motion_v1(str(_write(tmp_path, coord_frame="robot_base_z_up")))

    assert reference.metadata.coord_frame == "robot_base_z_up"
    np.testing.assert_array_equal(reference.robot_root_pos_w()[0], [-1.0, 0.0, 0.0])
    np.testing.assert_array_equal(reference.robot_frame_pos_w()[0, 1], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(reference.object_body_pos_w()[0, 0], [0.0, 0.0, 0.0])


def test_rigid_object_zero_articulation_placeholder_is_normalized(tmp_path):
    reference = load_motion_v1(str(_write(tmp_path, object_articulation=[0.0, 0.0, 0.0])))

    assert reference.object_articulation().shape == (3, 0)

    with pytest.raises(ValueError, match="all-zero articulation placeholder"):
        load_motion_v1(str(_write(tmp_path, object_articulation=[0.0, 0.1, 0.0])))


def test_hand_sides_define_contact_axis_order(tmp_path):
    reference = load_motion_v1(str(_write(tmp_path)))
    assert reference.sides == ("right", "left")
    np.testing.assert_array_equal(reference.contact_pos_w("right")[:, 0, 0], [10.0, 11.0, 12.0])
    np.testing.assert_array_equal(reference.contact_pos_w("left")[:, 0, 0], [20.0, 21.0, 22.0])
    np.testing.assert_array_equal(reference.contact_part_ids("right")[:, 0], [1, 1, 1])
    np.testing.assert_array_equal(reference.contact_active("right"), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(reference.contact_active("left"), [0.0, 0.0, 0.0])


def test_contact_activity_is_validated_and_nearest_resampled(tmp_path):
    path = _write(tmp_path, hand_contact_active=[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    reference = load_motion_v1(str(path), control_fps=4.0)

    assert reference.contact_active("right").shape == (5,)
    assert set(reference.contact_active("right")) <= {0.0, 1.0}
    assert set(reference.contact_active("left")) <= {0.0, 1.0}

    with pytest.raises(ValueError, match="binary 0/1"):
        load_motion_v1(str(_write(tmp_path / "invalid", hand_contact_active=[[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])))

    derived = load_motion_v1(str(_write(tmp_path / "derived", hand_contact_active=[None, None])))
    np.testing.assert_array_equal(derived.contact_active("right"), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(derived.contact_active("left"), [0.0, 0.0, 0.0])


def test_aligned_empty_per_side_contacts_load_as_empty_trajectories(tmp_path):
    reference = load_motion_v1(
        str(
            _write(
                tmp_path,
                hand_object_contact_positions=[[], []],
                hand_object_contact_normals=[[], []],
                hand_object_contact_part_ids=[[], []],
            )
        )
    )

    for side in ("right", "left"):
        assert reference.contact_pos_w(side).shape == (3, 0, 3)
        assert reference.contact_normal_w(side).shape == (3, 0, 3)
        assert reference.contact_part_ids(side).shape == (3, 0)
        np.testing.assert_array_equal(reference.contact_active(side), 0.0)


def test_contact_slot_counts_may_differ_by_hand_side(tmp_path):
    reference = load_motion_v1(
        str(
            _write(
                tmp_path,
                hand_object_contact_positions=[
                    [[[10.0, 0.0, 0.0]], [[11.0, 0.0, 0.0]], [[12.0, 0.0, 0.0]]],
                    [
                        [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]],
                        [[21.0, 0.0, 0.0], [0.0, 21.0, 0.0]],
                        [[22.0, 0.0, 0.0], [0.0, 22.0, 0.0]],
                    ],
                ],
                hand_object_contact_normals=[
                    [[[1.0, 0.0, 0.0]]] * 3,
                    [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]] * 3,
                ],
                hand_object_contact_part_ids=[
                    [[1], [1], [1]],
                    [[1, 0], [1, 0], [1, 0]],
                ],
            )
        )
    )

    assert reference.contact_pos_w("right").shape == (3, 1, 3)
    assert reference.contact_pos_w("left").shape == (3, 2, 3)


def test_resampling_preserves_endpoints_and_uses_nearest_contacts(tmp_path):
    reference = load_motion_v1(str(_write(tmp_path)), control_fps=4.0, motion_speed=0.5)
    assert reference.fps == 4.0
    assert reference.num_frames == 8
    assert reference.metadata.is_resampled
    np.testing.assert_array_equal(reference.robot_joint_pos()[[0, -1]], [[0.0, 10.0], [2.0, 12.0]])
    np.testing.assert_array_equal(reference.robot_frame_pos_w()[[0, -1], 0, 0], [0.0, 2.0])
    np.testing.assert_array_equal(reference.robot_frame_pos_w()[[0, -1], 2, 0], [30.0, 32.0])
    assert set(reference.contact_pos_w("right")[:, 0, 0]).issubset({10.0, 11.0, 12.0})
    np.testing.assert_allclose(np.linalg.norm(reference.robot_root_quat_w(), axis=-1), 1.0)


def test_source_frame_playback_preserves_samples_and_reinterprets_control_rate(tmp_path):
    path = str(_write(tmp_path))
    source = load_motion_v1(path)
    reference = load_motion_v1(path, control_fps=50.0, source_frame_playback=True)

    assert reference.fps == 50.0
    assert reference.num_frames == source.num_frames
    assert not reference.metadata.is_resampled
    assert reference.metadata.source_frame_playback
    np.testing.assert_array_equal(reference.robot_joint_pos(), source.robot_joint_pos())
    np.testing.assert_array_equal(reference.robot_root_pos_w(), source.robot_root_pos_w())


def test_frame_window_slices_every_time_aligned_field_and_preserves_static_contract(tmp_path):
    source = load_motion_v1(str(_write(tmp_path)))
    reference = source.frame_window(1, -1)

    assert reference.num_frames == 2
    assert reference.metadata is source.metadata
    assert reference.robot_joint_names() is source.robot_joint_names()
    assert reference.robot_frame_names() is source.robot_frame_names()
    assert reference.object_assets() is source.object_assets()
    np.testing.assert_array_equal(reference.robot_joint_pos(), source.robot_joint_pos()[1:])
    np.testing.assert_array_equal(reference.robot_root_pos_w(), source.robot_root_pos_w()[1:])
    np.testing.assert_array_equal(reference.robot_root_quat_w(), source.robot_root_quat_w()[1:])
    np.testing.assert_array_equal(reference.robot_frame_pos_w(), source.robot_frame_pos_w()[1:])
    np.testing.assert_array_equal(reference.robot_frame_quat_w(), source.robot_frame_quat_w()[1:])
    np.testing.assert_array_equal(reference.object_body_pos_w(), source.object_body_pos_w()[1:])
    np.testing.assert_array_equal(reference.object_body_quat_w(), source.object_body_quat_w()[1:])
    np.testing.assert_array_equal(reference.object_articulation(), source.object_articulation()[1:])
    for side in source.sides:
        np.testing.assert_array_equal(reference.contact_pos_w(side), source.contact_pos_w(side)[1:])
        np.testing.assert_array_equal(reference.contact_normal_w(side), source.contact_normal_w(side)[1:])
        np.testing.assert_array_equal(reference.contact_part_ids(side), source.contact_part_ids(side)[1:])
        np.testing.assert_array_equal(reference.contact_active(side), source.contact_active(side)[1:])

    assert source.frame_window() is source
    for invalid in ((-1, -1), (3, -1), (2, 2), (0, 4)):
        with pytest.raises(ValueError, match="invalid reference frame window"):
            source.frame_window(*invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "motion_v2", "schema_version"),
        ("motion_kind", "dual_hand", "single_robot"),
        ("coord_frame", "robot", "common world-coordinate convention"),
        ("robot_joint_names", ["joint", "joint"], "unique names"),
        ("ee_link_names", ["tip", "tip"], "unique names"),
    ],
)
def test_rejects_invalid_schema_or_names(tmp_path, field, value, message):
    with pytest.raises(ValueError, match=message):
        load_motion_v1(str(_write(tmp_path, **{field: value})))


def test_rejects_nonfinite_or_mismatched_robot_trajectory(tmp_path):
    with pytest.raises(ValueError, match="finite numeric"):
        load_motion_v1(str(_write(tmp_path, robot_joint_positions=[[0.0, 10.0], [np.nan, 11.0], [2.0, 12.0]])))

    with pytest.raises(ValueError, match="name-axis mismatch"):
        load_motion_v1(str(_write(tmp_path, robot_joint_positions=[[0.0], [1.0], [2.0]])))


def test_rejects_incomplete_or_duplicate_optional_hand_frames(tmp_path):
    with pytest.raises(ValueError, match="names and poses must be provided together"):
        load_motion_v1(str(_write(tmp_path, hand_frames_w=[])))

    with pytest.raises(ValueError, match="globally unique"):
        load_motion_v1(
            str(
                _write(
                    tmp_path / "duplicate",
                    hand_frame_names=[["tip_b"], ["left_index_DP"]],
                )
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("object_mesh_paths", [], "must contain 1 nonempty paths"),
        ("object_mesh_radius", [0.0], "positive finite"),
        ("object_urdf_paths", [], "explicit supported asset adapter"),
        (
            "hand_object_contact_part_ids",
            [[[0.5], [1.0], [1.0]], [[0.0], [0.0], [0.0]]],
            "must be integers",
        ),
        (
            "hand_object_contact_part_ids",
            [[[2], [1], [1]], [[0], [0], [0]]],
            "must be in",
        ),
    ],
)
def test_rejects_invalid_object_asset_or_contact_metadata(tmp_path, field, value, message):
    with pytest.raises(ValueError, match=message):
        load_motion_v1(str(_write(tmp_path, **{field: value})))


def test_requires_control_fps_for_nonunit_motion_speed(tmp_path):
    with pytest.raises(ValueError, match="control_fps is required"):
        load_motion_v1(str(_write(tmp_path)), motion_speed=0.5)


@pytest.mark.sequence_data
def test_loads_vendored_vega_mixer_reference():
    reference = load_motion_v1(_VEGA_MIXER)
    assert reference.robot_joint_pos().shape == (638, 58)
    assert reference.robot_frame_pos_w().shape == (638, 22, 3)
    assert reference.robot_joint_names()[:7] == [f"L_arm_j{index}" for index in range(1, 8)]
    assert "left_index_fingertip" in reference.robot_frame_names()
    assert reference.metadata.source_dataset == "arctic"
    assert reference.metadata.sequence_id == "dataset_s01_mixer_use_01"
    assert reference.metadata.robot_name == "vega_sharpa"
    articulation = reference.object_assets()[0].articulations[0]
    assert articulation.simulation_joint_name == "rotation"
    assert articulation.physics.armature == pytest.approx(0.01)
    assert articulation.physics.friction == pytest.approx(0.1)
    assert articulation.drive.kp == pytest.approx(50.0)
    assert articulation.drive.kd == pytest.approx(2.0)
    assert articulation.drive.effort_limit == pytest.approx(50.0)
