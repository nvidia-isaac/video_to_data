# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from v2d_hoi_object_reconstruction.lib import select_sam3d_frames as _selector


def test_stationary_selection_uses_view_direction_diversity(tmp_path, monkeypatch):
    masks_dir = tmp_path / "masks" / "0"
    masks_dir.mkdir(parents=True)
    for frame_id in range(4):
        (masks_dir / f"{frame_id:06d}.png").touch()

    monkeypatch.setattr(
        _selector,
        "_load_sfm_keyframes",
        lambda *_args: (
            np.array([0, 1, 2, 3]),
            np.zeros((4, 3)),
            np.array([
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
            ]),
        ),
    )
    areas = {"000000": 400, "000001": 100, "000002": 300, "000003": 200}
    monkeypatch.setattr(_selector, "_mask_area", lambda path: areas[path.stem])

    selected, report = _selector.select_frames_with_report(
        tmp_path,
        capture_mode="stationary",
        stationary_count=3,
    )

    assert selected == ["000000", "000002", "000003"]
    assert report == {
        "capture_mode": "stationary",
        "object_motion_assumption": "object_stationary_throughout",
        "object_motion_validation": "capture_procedure_contract",
        "selection_method": "camera_view_direction_farthest_point",
        "selected_frames": selected,
        "requested_view_count": 3,
    }


def test_two_stage_remains_the_default_selection_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _selector,
        "select_frames_by_angle_bins",
        lambda _job_dir, bin_deg: ["000010", "000020"],
    )

    selected, report = _selector.select_frames_with_report(tmp_path, bin_deg=45.0)

    assert selected == ["000010", "000020"]
    assert report["capture_mode"] == "two_stage"
    assert report["selection_method"] == "cumulative_orbit_angle_bins"
    assert report["angle_bin_degrees"] == 45.0
