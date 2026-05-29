# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import subprocess
import os
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR


def run_download(output_dir: str, dev: bool = False) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HF_HOME=/tmp/hf_cache",
        "-v", f"{output_dir}:/data/weights",
    ]
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(token_path):
            with open(token_path) as f:
                hf_token = f.read().strip()
    if hf_token:
        cmd += ["-e", f"HF_TOKEN={hf_token}"]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.sam3d_body.lib.download_weights",
        "--output_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SAM3D-Body and MoGe-2 weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for weights")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, dev=args.dev)
