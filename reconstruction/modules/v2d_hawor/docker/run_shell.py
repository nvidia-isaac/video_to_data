# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess

from v2d.hawor.docker._config import IMAGE_NAME, MODULES_DIR


def run_shell(dev: bool = False) -> None:
    cmd = ["docker", "run", "--rm", "-it", "--runtime=nvidia", "--gpus", "all", "-e", "HOME=/tmp"]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [IMAGE_NAME, "bash"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_shell(dev=args.dev)
