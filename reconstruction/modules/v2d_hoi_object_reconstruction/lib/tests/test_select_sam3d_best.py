# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from v2d_hoi_object_reconstruction.lib.select_sam3d_best import (
    select_best_sam3d_frame,
)


def test_stationary_capture_ignores_stale_stage1_boundary(tmp_path):
    sam3d_dir = tmp_path / "sam3d"
    frame_dir = sam3d_dir / "000100"
    srt_dir = frame_dir / "srt"
    srt_dir.mkdir(parents=True)
    (sam3d_dir / "selected_frames.json").write_text('["000100"]')
    (sam3d_dir / "capture_contract.json").write_text(json.dumps({
        "capture_mode": "stationary",
        "stage1_end_frame": None,
    }))
    stale_stage_dir = tmp_path / "stage1_detect_debug"
    stale_stage_dir.mkdir()
    (stale_stage_dir / "result.json").write_text(json.dumps({
        "stage1_end_frame": 10,
    }))
    (srt_dir / "srt_result.json").write_text(json.dumps({
        "scale": 1.0,
        "total_loss": 0.1,
        "capture_mode": "stationary",
    }))
    (srt_dir / "output_scaled.glb").write_bytes(b"mesh")

    summary = select_best_sam3d_frame(tmp_path)

    assert summary["capture_mode"] == "stationary"
    assert summary["stage1_end_frame"] is None
    assert summary["ranked_frames"][0]["is_stage1_source"] is True
    assert "source_frame_after_stage1" not in summary["ranked_frames"][0]["warnings"]
