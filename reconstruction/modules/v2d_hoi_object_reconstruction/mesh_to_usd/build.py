# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import subprocess
from pathlib import Path

from validation_docker_runtime import VALIDATOR_IMAGE_NAME

IMAGE_NAME = os.environ.get("V2D_HOI_MESH_TO_USD_IMAGE", "v2d_hoi_mesh_to_usd")
_MODULE_DIR = Path(__file__).resolve().parent
_MODULES_DIR = _MODULE_DIR.parent.parent


def _build(tag: str, dockerfile: Path) -> None:
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            tag,
            "-f",
            str(dockerfile),
            str(_MODULES_DIR),
        ],
        check=True,
    )


def build(
    tag: str = IMAGE_NAME,
    validator_tag: str = VALIDATOR_IMAGE_NAME,
    *,
    target: str = "all",
) -> None:
    if target in {"all", "generator"}:
        _build(tag, _MODULE_DIR / "Dockerfile")
    if target in {"all", "validator"}:
        _build(validator_tag, _MODULE_DIR / "validator" / "Dockerfile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build mesh-to-USD workflow images.")
    parser.add_argument(
        "--target",
        choices=("all", "generator", "validator"),
        default="all",
    )
    parser.add_argument("--generator-image", default=IMAGE_NAME)
    parser.add_argument("--validator-image", default=VALIDATOR_IMAGE_NAME)
    args = parser.parse_args()
    build(args.generator_image, args.validator_image, target=args.target)
