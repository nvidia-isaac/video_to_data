# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os

IMAGE_NAME = "v2d_mesh"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
_DOCKERFILE = os.path.join(_CURRENT_DIR, "Dockerfile")


def build_docker_image() -> None:
    subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "-f", _DOCKERFILE, _MODULES_DIR],
        check=True,
    )


if __name__ == "__main__":
    build_docker_image()
