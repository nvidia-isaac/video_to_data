# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess

def build_docker_image() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    dockerfile = os.path.join(current_dir, "Dockerfile")
    subprocess.run(["docker", "build", "-t", "v2d_cari4d:latest", "-f", dockerfile, modules_dir], check=True)


if __name__ == "__main__":
    build_docker_image()
