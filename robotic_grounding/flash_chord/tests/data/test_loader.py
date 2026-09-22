# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for explicit reference-schema dispatch."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flash_chord.assets import ASSETS_DIR
from flash_chord.data.loader import load_reference
from flash_chord.data.mano_sharpa import load_mano_sharpa

_MIXER = str(
    ASSETS_DIR
    / "human_motion_data"
    / "arctic"
    / "arctic_processed"
    / "sequence_id=dataset_s01_mixer_use_01"
    / "robot_name=sharpa_wave"
)


def _write(tmp_path: Path, columns: dict) -> Path:
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table(columns), path)
    return path


def test_dispatches_declared_motion_v1_and_forwards_playback(monkeypatch, tmp_path):
    path = _write(tmp_path, {"schema_version": ["motion_v1"]})
    sentinel = object()
    calls = []

    def fake_loader(source_path, *, control_fps, motion_speed, source_frame_playback):
        calls.append((source_path, control_fps, motion_speed, source_frame_playback))
        return sentinel

    monkeypatch.setattr("flash_chord.data.loader.load_motion_v1", fake_loader)
    assert load_reference(path, control_fps=20.0, motion_speed=1.0, source_frame_playback=True) is sentinel
    assert calls == [(str(path), 20.0, 1.0, True)]


def test_dispatches_only_complete_unversioned_mano_sharpa_signature(monkeypatch, tmp_path):
    columns = {
        "fps": [30.0],
        "object_name": ["object"],
        "object_body_names": [["body"]],
        "object_body_position": [[[[0.0, 0.0, 0.0]]]],
        "object_body_wxyz": [[[[1.0, 0.0, 0.0, 0.0]]]],
        "object_mesh_paths": [["mesh.obj"]],
        "object_mesh_radius": [[0.1]],
    }
    for side in ("left", "right"):
        columns.update(
            {
                f"robot_{side}_wrist_position": [[[0.0, 0.0, 0.0]]],
                f"robot_{side}_wrist_wxyz": [[[1.0, 0.0, 0.0, 0.0]]],
                f"robot_{side}_finger_joints": [[[0.0]]],
                f"{side}_robot_finger_joint_names": [[f"{side}_joint"]],
                f"robot_{side}_frames": [[[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]],
                f"{side}_robot_frame_names": [[f"{side}_tip"]],
                f"mano_{side}_object_contact_positions": [[[[0.0, 0.0, 0.0]]]],
                f"mano_{side}_object_contact_normals": [[[[1.0, 0.0, 0.0]]]],
                f"mano_{side}_object_contact_part_ids": [[[1]]],
            }
        )
    path = _write(tmp_path, columns)
    sentinel = object()
    monkeypatch.setattr("flash_chord.data.loader.load_mano_sharpa", lambda *_args, **_kwargs: sentinel)
    assert load_reference(path) is sentinel


@pytest.mark.parametrize("schema_version", ["motion_v2", "mano_sharpa_v2"])
def test_rejects_unknown_declared_schema(schema_version, tmp_path):
    path = _write(tmp_path, {"schema_version": [schema_version]})
    with pytest.raises(ValueError, match="unsupported reference schema_version"):
        load_reference(path)


def test_rejects_arbitrary_unversioned_parquet(tmp_path):
    path = _write(tmp_path, {"fps": [30.0], "value": [1]})
    with pytest.raises(ValueError, match="does not match the ManoSharpaData"):
        load_reference(path)


@pytest.mark.sequence_data
def test_real_mano_sharpa_dispatch_matches_direct_loader_at_half_speed():
    direct = load_mano_sharpa(_MIXER, control_fps=20.0, motion_speed=0.5)
    dispatched = load_reference(_MIXER, control_fps=20.0, motion_speed=0.5)
    assert type(dispatched) is type(direct)
    assert dispatched.num_frames == direct.num_frames == 849
    np.testing.assert_array_equal(dispatched.wrist_pos_w("left"), direct.wrist_pos_w("left"))
    np.testing.assert_array_equal(dispatched.object_body_quat_w(), direct.object_body_quat_w())
    np.testing.assert_array_equal(dispatched.contact_part_ids("right"), direct.contact_part_ids("right"))
