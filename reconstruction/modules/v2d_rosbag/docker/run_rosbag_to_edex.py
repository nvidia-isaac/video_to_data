# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.rosbag.docker._config import IMAGE_NAME, MODULES_DIR

_DEFAULT_CONFIG = Path(__file__).parent.parent / "lib" / "configs" / "nova_hawk.yaml"


def run_rosbag_to_edex(
    rosbag_path: str,
    output_dir: str,
    config_path: str = str(_DEFAULT_CONFIG),
    no_extrinsics: bool = False,
    dev: bool = False,
) -> None:
    inputs = {
        "config_path": config_path,
        "rosbag_path": rosbag_path,
    }
    outputs = {"output_path": output_dir}

    extra_args = {
        "no_extrinsics": no_extrinsics or None,
    }

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.rosbag.lib.rosbag_to_edex",
        inputs=inputs,
        outputs=outputs,
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rosbag-to-EDEX extraction in Docker")
    parser.add_argument("--rosbag_path", "-r", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--config_path", "-c", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--no_extrinsics", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_rosbag_to_edex(
        config_path=args.config_path,
        rosbag_path=args.rosbag_path,
        output_dir=args.output_dir,
        no_extrinsics=args.no_extrinsics,
        dev=args.dev,
    )
