"""Pipeline: extract calibration images from rosbag, run extrinsic calibration.

This pipeline is for calibration datasets (containing chessboard images).
The output is an EDEX file with calibrated camera extrinsics that can be
referenced by data datasets via run_dataset_preprocessing.py.

Usage:
    python run_mv_calibration.py \
        --rosbag_path /data/rosbags/2026-03-28_calibration \
        --output_dir /data/datasets/proc_2026-03-28_calibration
"""

import argparse
import os

from v2d.rosbag.docker.run_rosbag_to_edex import run_rosbag_to_edex
from v2d.mv.calibration.docker.run_calibrate_extrinsics import run_calibrate_extrinsics


def main(
    rosbag_path: str,
    output_dir: str,
    dev: bool = False,
):
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
        image_dir=os.path.join(raw_dir, "images"),
        output_dir=os.path.join(output_dir, "extrinsics"),
        dev=dev,
    )

    print("\n=== Calibration Complete ===")
    print(f"Extrinsics camera params: {os.path.join(output_dir, 'extrinsics', 'edex')}")
    print(f"Reference this path as --extrinsics_camera_params_path in run_mv_hoi_reconstruction.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration dataset pipeline")
    parser.add_argument("--rosbag_path", type=str, required=True,
                        help="Path to the calibration ROS bag")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory for calibration results")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    main(
        rosbag_path=args.rosbag_path,
        output_dir=args.output_dir,
        dev=args.dev,
    )
