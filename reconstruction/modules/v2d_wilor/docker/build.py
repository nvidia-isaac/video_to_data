# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import os
import subprocess

IMAGE_NAME = "v2d_wilor"

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.join(current_dir, "..")
root_dir = os.path.join(module_dir, "..")
dockerfile_path = os.path.join(current_dir, "Dockerfile")


def build_docker_image() -> None:
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, "-f", dockerfile_path, root_dir], check=True)


if __name__ == "__main__":
    build_docker_image()
