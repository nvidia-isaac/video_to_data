# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from v2d.pipelines import run_mv_calibration as pipeline


def test_pipeline_uses_current_calibration_setup(monkeypatch) -> None:
    rosbag_call = {}
    calibration_call = {}
    monkeypatch.setattr(
        pipeline,
        "run_rosbag_to_edex",
        lambda **kwargs: rosbag_call.update(kwargs),
    )
    monkeypatch.setattr(
        pipeline,
        "run_calibrate_extrinsics",
        lambda **kwargs: calibration_call.update(kwargs),
    )

    pipeline.main("/input/bag", "/output/calibration", dev=True)

    assert rosbag_call == {
        "rosbag_path": "/input/bag",
        "output_dir": "/output/calibration/raw",
        "no_extrinsics": True,
        "dev": True,
    }
    assert calibration_call == {
        "camera_params_path": "/output/calibration/raw/edex",
        "rgb_dir": "/output/calibration/raw/images",
        "output_dir": "/output/calibration/extrinsics",
        "calibration_setup": "stereo4_6x10_100mm_marker",
        "dev": True,
    }


def test_pipeline_accepts_an_alternate_packaged_setup(monkeypatch) -> None:
    calibration_call = {}
    monkeypatch.setattr(pipeline, "run_rosbag_to_edex", lambda **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "run_calibrate_extrinsics",
        lambda **kwargs: calibration_call.update(kwargs),
    )

    pipeline.main(
        "/input/bag",
        "/output/calibration",
        calibration_setup="stereo4_6x10_22p58mm_marker",
    )

    assert (
        calibration_call["calibration_setup"]
        == "stereo4_6x10_22p58mm_marker"
    )


def test_cli_defaults_to_current_calibration_setup() -> None:
    args = pipeline._build_parser().parse_args([
        "--rosbag_path", "/input/bag",
        "--output_dir", "/output/calibration",
    ])

    assert args.calibration_setup == "stereo4_6x10_100mm_marker"
