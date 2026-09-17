# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline: extract calibration images from rosbag, run extrinsic calibration.

This pipeline is for calibration datasets containing chessboard images. The
output is an EDEX file with calibrated camera extrinsics for
``run_mv_hoi_reconstruction.py``.

Usage:
    python -m v2d.pipelines.run_mv_calibration \
        --rosbag_path /data/rosbags/2026-03-28_calibration \
        --output_dir /data/datasets/proc_2026-03-28_calibration
"""

import argparse
import os

from v2d.rosbag.docker.run_rosbag_to_edex import run_rosbag_to_edex
from v2d.mv.calibration.docker.run_calibrate_extrinsics import run_calibrate_extrinsics


DEFAULT_CALIBRATION_SETUP = "stereo4_6x10_100mm_marker"


def main(
    rosbag_path: str,
    output_dir: str,
    calibration_setup: str = DEFAULT_CALIBRATION_SETUP,
    dev: bool = False,
) -> None:
    raw_dir = os.path.join(output_dir, "raw")

    # Step 1: Extract images from rosbag
    run_rosbag_to_edex(
        rosbag_path=rosbag_path,
        output_dir=raw_dir,
        no_extrinsics=True,
        dev=dev,
    )

    # Step 2: Calibrate extrinsics
    run_calibrate_extrinsics(
        camera_params_path=os.path.join(raw_dir, "edex"),
        rgb_dir=os.path.join(raw_dir, "images"),
        output_dir=os.path.join(output_dir, "extrinsics"),
        calibration_setup=calibration_setup,
        dev=dev,
    )

    print("\n=== Calibration Complete ===")
    print(f"Extrinsics camera params: {os.path.join(output_dir, 'extrinsics', 'edex')}")
    print(
        "Reference this path as --calibration_camera_params_path "
        "in run_mv_hoi_reconstruction.py"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibration dataset pipeline")
    parser.add_argument("--rosbag_path", type=str, required=True,
                        help="Path to the calibration ROS bag")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory for calibration results")
    parser.add_argument(
        "--calibration_setup",
        type=str,
        default=DEFAULT_CALIBRATION_SETUP,
        help=(
            "Packaged calibration setup identifier (default: "
            f"{DEFAULT_CALIBRATION_SETUP})"
        ),
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Mount the checked-out module sources into both containers",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    main(
        rosbag_path=args.rosbag_path,
        output_dir=args.output_dir,
        calibration_setup=args.calibration_setup,
        dev=args.dev,
    )
