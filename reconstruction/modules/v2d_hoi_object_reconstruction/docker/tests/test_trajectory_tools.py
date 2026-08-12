# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest

from v2d_hoi_object_reconstruction.docker import _trajectory_tools


def _write_result(call, result):
    output_dir = Path(call["outputs"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result))


def test_sfm_quality_check_runs_in_cpu_only_hoi_container(tmp_path, monkeypatch):
    calls = []

    def fake_run_in_container(**kwargs):
        calls.append(kwargs)
        _write_result(kwargs, {"passed": True})

    monkeypatch.setattr(_trajectory_tools, "run_in_container", fake_run_in_container)

    result_path = _trajectory_tools.run_sfm_quality_check(
        image="v2d_hoi_object_reconstruction",
        sfm_keyframes="/input/sfm.json",
        frames_meta="/input/frames.json",
        output_dir=tmp_path / "quality",
        config={"min_keyframes": 42},
        fail_on_error=True,
    )

    assert result_path == tmp_path / "quality" / "result.json"
    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.check_sfm_scan_quality",
        "inputs": {
            "sfm_keyframes": "/input/sfm.json",
            "frames_meta": "/input/frames.json",
        },
        "outputs": {"output_dir": str(tmp_path / "quality")},
        "extra_args": {
            "min_keyframes": 42,
            "min_angle_span_deg": 600.0,
            "max_backtracking_fraction": 0.25,
            "max_translation_step_m": 2.0,
            "max_rotation_step_deg": 90.0,
            "robust_step_sigma": 6.0,
            "large_step_floor_m": 0.25,
            "max_large_step_fraction": 0.25,
            "warn_only": False,
        },
        "gpus": False,
    }]


def test_sfm_quality_check_preserves_quality_failure(tmp_path, monkeypatch):
    def fake_run_in_container(**kwargs):
        _write_result(kwargs, {"passed": False})
        raise subprocess.CalledProcessError(2, ["docker", "run"])

    monkeypatch.setattr(_trajectory_tools, "run_in_container", fake_run_in_container)

    with pytest.raises(RuntimeError, match="scan quality check failed"):
        _trajectory_tools.run_sfm_quality_check(
            image="v2d_hoi_object_reconstruction",
            sfm_keyframes="/input/sfm.json",
            frames_meta="/input/frames.json",
            output_dir=tmp_path / "quality",
            config={},
            fail_on_error=True,
        )


def test_sfm_quality_check_identifies_container_failure(tmp_path, monkeypatch):
    output_dir = tmp_path / "quality"
    output_dir.mkdir()
    (output_dir / "result.json").write_text(json.dumps({"passed": True}))

    def fake_run_in_container(**_kwargs):
        raise subprocess.CalledProcessError(1, ["docker", "run"])

    monkeypatch.setattr(_trajectory_tools, "run_in_container", fake_run_in_container)

    with pytest.raises(RuntimeError, match="failed to execute in the HOI container"):
        _trajectory_tools.run_sfm_quality_check(
            image="v2d_hoi_object_reconstruction",
            sfm_keyframes="/input/sfm.json",
            frames_meta="/input/frames.json",
            output_dir=output_dir,
            config={},
            fail_on_error=True,
        )

    assert not (output_dir / "result.json").exists()


@pytest.mark.parametrize("detected_frame", [925, None])
def test_stage1_detection_runs_in_cpu_only_hoi_container(
    tmp_path, monkeypatch, detected_frame
):
    calls = []

    def fake_run_in_container(**kwargs):
        calls.append(kwargs)
        _write_result(kwargs, {"stage1_end_frame": detected_frame})

    monkeypatch.setattr(_trajectory_tools, "run_in_container", fake_run_in_container)

    result = _trajectory_tools.run_stage1_detection(
        image="v2d_hoi_object_reconstruction",
        sfm_keyframes="/input/sfm.json",
        frames_meta="/input/frames.json",
        output_dir=tmp_path / "stage1",
        buffer_deg=12.5,
    )

    assert result == detected_frame
    assert calls == [{
        "image": "v2d_hoi_object_reconstruction",
        "module": "v2d_hoi_object_reconstruction.lib.detect_stage1_end",
        "inputs": {
            "sfm_keyframes": "/input/sfm.json",
            "frames_meta": "/input/frames.json",
        },
        "outputs": {"output_dir": str(tmp_path / "stage1")},
        "extra_args": {"buffer_deg": 12.5},
        "gpus": False,
    }]
