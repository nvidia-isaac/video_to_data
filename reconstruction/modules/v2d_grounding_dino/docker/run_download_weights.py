# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import subprocess
import os
from v2d.grounding_dino.docker._config import IMAGE_NAME, MODULES_DIR


def run_download(output_dir: str, dev: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{output_dir}:/data/models",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.grounding_dino.lib.download_weights",
        "--output_dir", "/data/models",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download GroundingDINO checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoint")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, dev=args.dev)
