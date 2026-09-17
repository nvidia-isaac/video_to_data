# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess

from v2d.cari4d.docker._config import IMAGE_NAME, MODULES_DIR


def run_tests() -> None:
    command = ["docker", "run", "--rm", "--runtime=nvidia", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp", "-v", f"{MODULES_DIR}:/workspace", "-v", "/workspace/v2d_foundation_pose/lib/FoundationPose/mycpp/build", "-v", "/workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/mycuda", "-w", "/workspace/v2d_cari4d"]
    subprocess.run(command + [IMAGE_NAME, "python", "-m", "pytest", "tests", "-v", "-o", "cache_dir=/tmp/pytest-cache"], check=True)


if __name__ == "__main__":
    run_tests()
