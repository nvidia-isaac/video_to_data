"""CLI entry point for rosbag-to-EDEX extraction."""

import argparse
import logging
from pathlib import Path

import yaml

from v2d.rosbag.lib.config import Config
from v2d.rosbag.lib.edex_extraction import rosbag_to_edex, rosbag_extract_images


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract EDEX dataset from a ROS bag")
    parser.add_argument("--config_path", "-c", type=Path, required=True,
                        help="YAML config file")
    parser.add_argument("--rosbag_path", "-r", type=Path, required=True,
                        help="Path to rosbag directory")
    parser.add_argument("--output_path", "-o", type=Path, required=True,
                        help="Output directory for EDEX dataset")
    parser.add_argument("--no_extrinsics", action="store_true",
                        help="Extract images + intrinsics only (no TF extrinsics)")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config_dict = yaml.safe_load(f)

    config_dict["rosbag_path"] = str(args.rosbag_path)
    config_dict["output_path"] = str(args.output_path)

    config = Config(**config_dict)

    if args.no_extrinsics:
        rosbag_extract_images(config)
    else:
        rosbag_to_edex(config)


if __name__ == "__main__":
    main()
