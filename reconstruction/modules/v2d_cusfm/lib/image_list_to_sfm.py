"""
Run pyCuSFM on a directory of images to produce camera poses.

Expects --input_dir to contain images and a frames_meta.json describing
camera calibration and timestamps (same format produced by the mapping pipeline).

The output frames_meta.json with camera poses is written to:
    <output_dir>/keyframes/frames_meta.json

Usage:
    python image_list_to_sfm.py \\
        --input_dir  /data/hoi_obj_FP/2026-02-18_..._cattle/left \\
        --output_dir /data/hoi_obj_FP/2026-02-18_..._cattle/sfm
"""

import argparse

from pycusfm.cusfm_runner import create_cusfm_runner


def main():
    parser = argparse.ArgumentParser(description="Run pyCuSFM on an image directory")
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing images and frames_meta.json")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for SfM results")
    parser.add_argument("--config_set", default="backpack",
                        help="Config preset: av, isaac, rgbd, backpack (default: backpack)")
    parser.add_argument("--feature_type", default=None,
                        help="Feature type (e.g. aliked_lightglue, superpoint_lightglue, sift)")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="Number of CPU threads (default: 1)")
    args = parser.parse_args()

    kwargs = dict(
        input_dir=args.input_dir,
        cusfm_base_dir=args.output_dir,
        num_threads=args.num_threads,
    )
    if args.feature_type:
        kwargs["feature_type"] = args.feature_type

    runner = create_cusfm_runner(config_set=args.config_set, **kwargs)
    runner.run_all()
    print(f"[cusfm] done — poses at {args.output_dir}/keyframes/frames_meta.json")


if __name__ == "__main__":
    main()
