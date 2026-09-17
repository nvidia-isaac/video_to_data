# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from v2d_hoi_object_reconstruction.lib import check_sfm_scan_quality as _quality


def _write_short_capture(tmp_path):
    frames_meta = {
        "camera_params_id_to_camera_params": {
            "left": {"sensor_meta_data": {"sensor_name": "front_stereo_camera_left"}},
            "right": {"sensor_meta_data": {"sensor_name": "front_stereo_camera_right"}},
        },
        "keyframes_metadata": [],
    }
    sfm_meta = {"keyframes_metadata": []}
    positions = [(1.0, 0.0, 0.0), (0.7, 0.7, 0.0), (0.0, 1.0, 0.0), (-0.7, 0.7, 0.0)]
    for index, (x, y, z) in enumerate(positions):
        timestamp = 1_000_000 + index
        for camera_id in ("left", "right"):
            frames_meta["keyframes_metadata"].append({
                "camera_params_id": camera_id,
                "synced_sample_id": index,
                "timestamp_microseconds": timestamp,
            })
        sfm_meta["keyframes_metadata"].append({
            "image_name": f"front_stereo_camera_left/{timestamp}.jpg",
            "timestamp_microseconds": timestamp,
            "camera_to_world": {
                "axis_angle": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0,
                    "angle_degrees": 0.0,
                },
                "translation": {"x": x, "y": y, "z": z},
            },
        })

    frames_path = tmp_path / "frames_meta.json"
    sfm_path = tmp_path / "sfm_frames_meta.json"
    frames_path.write_text(json.dumps(frames_meta))
    sfm_path.write_text(json.dumps(sfm_meta))
    return sfm_path, frames_path


def _args(tmp_path, capture_mode):
    sfm_path, frames_path = _write_short_capture(tmp_path)
    return SimpleNamespace(
        sfm_keyframes=sfm_path,
        frames_meta=frames_path,
        output_dir=tmp_path / capture_mode,
        capture_mode=capture_mode,
        min_keyframes=2,
        min_angle_span_deg=600.0,
        max_backtracking_fraction=0.25,
        max_translation_step_m=10.0,
        max_rotation_step_deg=180.0,
        robust_step_sigma=6.0,
        large_step_floor_m=10.0,
        max_large_step_fraction=1.0,
    )


def test_stationary_mode_skips_only_two_loop_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(_quality, "_plot_diagnostics", lambda *_args: None)

    stationary = _quality.check_quality(_args(tmp_path, "stationary"))
    two_stage = _quality.check_quality(_args(tmp_path, "two_stage"))

    assert stationary["passed"] is True
    assert stationary["capture_mode"] == "stationary"
    assert "two_loop_angle_span" not in {
        check["name"] for check in stationary["checks"]
    }
    assert two_stage["passed"] is False
    assert "two_loop_angle_span" in {
        check["name"] for check in two_stage["checks"]
    }
