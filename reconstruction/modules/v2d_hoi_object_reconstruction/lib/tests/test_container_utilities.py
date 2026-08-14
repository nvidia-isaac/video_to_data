# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from v2d_hoi_object_reconstruction.lib import stitch_mp4 as stitch_module
from v2d_hoi_object_reconstruction.lib.convert_poses_to_matrix import (
    convert_poses_to_matrix as convert_pose_files,
)


def test_pose_converter_changes_quaternion_json_and_skips_matrices(tmp_path):
    pose_path = tmp_path / "000000.json"
    pose_path.write_text(json.dumps({
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "translation": [1.0, 2.0, 3.0],
        "scale": [1.0, 1.0, 1.0],
    }))
    matrix_path = tmp_path / "000001.json"
    existing_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    matrix_path.write_text(json.dumps(existing_matrix))

    assert convert_pose_files(tmp_path) == 1
    converted = json.loads(pose_path.read_text())
    assert len(converted) == 4
    assert all(len(row) == 4 for row in converted)
    assert json.loads(matrix_path.read_text()) == existing_matrix


def test_stitch_module_falls_back_to_next_codec(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[command.index("-c:v") + 1] == "libx264":
            return SimpleNamespace(returncode=1, stderr="libx264 failed")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(stitch_module.subprocess, "run", fake_run)
    stitch_module.stitch_mp4(tmp_path / "frames", tmp_path / "render.mp4")

    assert [command[command.index("-c:v") + 1] for command in commands] == [
        "libx264",
        "h264_nvenc",
    ]
