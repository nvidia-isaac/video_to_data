# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import subprocess
from pathlib import Path

IMAGE_NAME  = "v2d_bundlesdf"
_MODULES_DIR = str(Path(__file__).parents[2])  # reconstruction/modules/


def run_download(output_dir: str, dev: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{output_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d_bundlesdf.lib.download_weights",
        "--output_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BundleSDF (RoMA) weights")
    parser.add_argument("--output_dir", required=True, help="Root weights directory (e.g. data/weights)")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, dev=args.dev)
