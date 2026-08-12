# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
from v2d.sam3d.docker._config import IMAGE_NAME, MODULES_DIR

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
    run_env = os.environ.copy()
    # hf-xet can fail on this gated snapshot with
    # "Unable to parse string as hex hash value". Use Hugging Face's
    # supported HTTP fallback for a reliable, resumable download.
    run_env["HF_HUB_DISABLE_XET"] = "1"
    cmd += ["-e", "HF_HUB_DISABLE_XET"]
    if hf_token:
        # Pass the variable through Docker without embedding the secret in the
        # command line, where errors and process listings could expose it.
        run_env["HF_TOKEN"] = hf_token
        cmd += ["-e", "HF_TOKEN"]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.sam3d.lib.download_weights",
        "--output_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True, env=run_env)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download all SAM3D weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for all weights")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, dev=args.dev)
