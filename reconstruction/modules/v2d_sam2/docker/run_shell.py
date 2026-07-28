# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
from v2d.sam2.docker._config import IMAGE_NAME, MODULES_DIR

def run_shell(dev: bool = False, gpu: int = 0) -> None:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative physical GPU index")
    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", f"device={gpu}",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "CUDA_VISIBLE_DEVICES=0",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [IMAGE_NAME, "bash"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run shell in v2d_sam2 container")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    parser.add_argument("--gpu", type=int, default=0, help="Physical host GPU index")
    args = parser.parse_args()
    run_shell(dev=args.dev, gpu=args.gpu)
