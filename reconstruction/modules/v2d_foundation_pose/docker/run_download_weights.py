# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_download(
    output_dir: str,
    backend: str = "nvidia_tensorrt",
    accept_nvidia_model_eula: bool = False,
    dev: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{output_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.foundation_pose.lib.download_weights",
        "--output_dir", "/data/weights",
        "--backend", backend,
    ]
    if accept_nvidia_model_eula:
        cmd.append("--accept_nvidia_model_eula")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FoundationPose weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for weights")
    parser.add_argument(
        "--backend", choices=("nvidia_tensorrt", "nvlabs_pytorch"),
        default="nvidia_tensorrt",
    )
    parser.add_argument("--accept_nvidia_model_eula", action="store_true")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_download(
        output_dir=args.output_dir,
        backend=args.backend,
        accept_nvidia_model_eula=args.accept_nvidia_model_eula,
        dev=args.dev,
    )
