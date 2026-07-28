# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned E2FGVI image. This module never downloads checkpoints."""

from __future__ import annotations

import subprocess
from pathlib import Path

IMAGE_NAME = "v2d_e2fgvi"


def build_docker_image() -> None:
    docker_dir = Path(__file__).resolve().parent
    stage_root = docker_dir.parent
    subprocess.run(
        [
            "docker",
            "build",
            "--tag",
            IMAGE_NAME,
            "--file",
            str(docker_dir / "Dockerfile"),
            str(stage_root),
        ],
        check=True,
    )


if __name__ == "__main__":
    build_docker_image()
