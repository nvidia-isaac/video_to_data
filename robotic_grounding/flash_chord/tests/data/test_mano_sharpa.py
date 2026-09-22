# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ManoSharpa parquet loader + resampling."""

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR
from flash_chord.data.mano_sharpa import load_mano_sharpa
from flash_chord.data.reference import HandPoseReference, NamedFrameReference, NamedRobotReference, Reference

# A vendored, fully retargeted Sharpa parquet (filled robot fields) — arctic box, articulated (no per-body URDFs).
_BOX = str(
    ASSETS_DIR
    / "human_motion_data"
    / "arctic"
    / "arctic_processed"
    / "sequence_id=dataset_s07_box_grab_01"
    / "robot_name=sharpa_wave"
)
_MIXER = str(
    ASSETS_DIR
    / "human_motion_data"
    / "arctic"
    / "arctic_processed"
    / "sequence_id=dataset_s01_mixer_use_01"
    / "robot_name=sharpa_wave"
)
_HOT3D = str(
    ASSETS_DIR
    / "human_motion_data"
    / "hot3d"
    / "hot3d_processed"
    / "sequence_id=P0002_59a84a3a_seg025"
    / "robot_name=sharpa_wave"
)


def test_motion_speed_requires_control_fps():
    with pytest.raises(ValueError, match="control_fps is required"):
        load_mano_sharpa("/unused", motion_speed=0.5)


# --- real parquet ---
@pytest.mark.sequence_data
def test_load_real_parquet_shapes():
    r = load_mano_sharpa(_BOX)
    assert isinstance(r, Reference)
    assert isinstance(r, HandPoseReference)
    assert not isinstance(r, NamedRobotReference)
    assert not isinstance(r, NamedFrameReference)
    assert r.metadata.source_fps == 30.0
    assert not r.metadata.is_resampled
    assert r.metadata.schema_version is None
    assert r.metadata.source_dataset == "arctic"
    assert r.metadata.sequence_id == "dataset_s07_box_grab_01"
    assert r.metadata.robot_name == "sharpa_wave"
    assert r.metadata.raw_motion_file == "dataset/s07/box_grab_01"
    assert r.num_frames == 725
    assert r.fps == 30.0
    assert r.sides == ("left", "right")
    for s in r.sides:
        assert r.wrist_pos_w(s).shape == (725, 3)
        assert r.wrist_quat_w(s).shape == (725, 4)
        assert r.finger_joint_pos(s).shape == (725, 22)
        assert len(r.finger_joint_names(s)) == 22
        assert all(name.startswith(f"{s}_") for name in r.finger_joint_names(s))
        assert r.contact_pos_w(s).shape == (725, 16, 3)
        assert r.contact_part_ids(s).shape == (725, 16)
    assert r.object_body_pos_w().shape == (725, 2, 3)
    assert r.object_body_quat_w().shape == (725, 2, 4)
    assert r.object_body_names() == ["bottom", "top"]
    assert len(r.object_mesh_paths()) == 2
    assert np.allclose(np.linalg.norm(r.wrist_quat_w("left"), axis=-1), 1.0, atol=1e-5)


@pytest.mark.sequence_data
def test_resample_changes_frame_count_and_keeps_unit_quats():
    r = load_mano_sharpa(_BOX, control_fps=20.0)
    assert r.fps == 20.0
    assert r.metadata.is_resampled
    assert r.num_frames == round(724 / 30 * 20) + 1
    assert r.wrist_pos_w("left").shape[0] == r.num_frames
    assert np.allclose(np.linalg.norm(r.wrist_quat_w("left"), axis=-1), 1.0, atol=1e-5)


@pytest.mark.sequence_data
def test_half_speed_mixer_resample_matches_expected_reference():
    native = load_mano_sharpa(_MIXER)
    reference = load_mano_sharpa(_MIXER, control_fps=20.0, motion_speed=0.5)
    assert reference.fps == 20.0
    assert reference.num_frames == 849
    assert reference.metadata.is_resampled
    assert np.allclose(reference.object_body_pos_w()[[0, -1]], native.object_body_pos_w()[[0, -1]])
    assert np.allclose(reference.wrist_pos_w("left")[[0, -1]], native.wrist_pos_w("left")[[0, -1]])


@pytest.mark.sequence_data
def test_independent_rigid_mano_sharpa_placeholder_is_explicitly_adapted():
    reference = load_mano_sharpa(_HOT3D)

    assert reference.object_articulation().shape == (reference.num_frames, 0)


@pytest.mark.parametrize(("width", "value"), [(2, 0.0), (1, 0.1)])
@pytest.mark.sequence_data
def test_independent_rigid_mano_sharpa_rejects_non_placeholder_articulation(monkeypatch, width, value):
    from flash_chord.data import mano_sharpa
    from flash_chord.data.parquet import ParquetRow, read_parquet_row

    row = read_parquet_row(_HOT3D)
    cells = dict(row.cells)
    num_frames = len(cells["object_body_position"])
    cells["object_articulation"] = np.full((num_frames, width), value)
    monkeypatch.setattr(
        mano_sharpa,
        "read_parquet_row",
        lambda path: ParquetRow(path=row.path, cells=cells),
    )

    with pytest.raises(ValueError, match="only one all-zero articulation placeholder"):
        mano_sharpa.load_mano_sharpa(_HOT3D)
