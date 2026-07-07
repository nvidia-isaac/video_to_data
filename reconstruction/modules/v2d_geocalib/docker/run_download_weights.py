# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.geocalib.docker._config import IMAGE_NAME, MODULES_DIR


def run_download_weights(
    output_dir: str,
    weights: str = "pinhole",
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.geocalib.lib.download_weights",
        inputs={},
        outputs={"output_dir": output_dir},
        extra_args={"weights": weights},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download GeoCalib checkpoint")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", choices=("pinhole", "distorted"), default="pinhole")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_download_weights(output_dir=args.output_dir, weights=args.weights, dev=args.dev)
