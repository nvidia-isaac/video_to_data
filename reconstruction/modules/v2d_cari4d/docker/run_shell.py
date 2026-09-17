# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import subprocess

from v2d.cari4d.docker._config import IMAGE_NAME, MODULES_DIR


def run_shell(dev: bool = False) -> None:
    command = ["docker", "run", "-it", "--rm", "--runtime=nvidia", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}"]
    if dev:
        command += ["-v", f"{MODULES_DIR}:/workspace"]
    subprocess.run(command + [IMAGE_NAME, "bash"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a shell in the CARI4D container")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_shell(dev=args.dev)
