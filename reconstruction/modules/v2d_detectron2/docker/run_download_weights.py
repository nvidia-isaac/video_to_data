# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
from v2d.detectron2.docker._config import IMAGE_NAME, MODULES_DIR


def run_download_weights(
    output_dir: str,
    model_sizes: list[str] | None = None,
    dev: bool = False,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{output_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [IMAGE_NAME, "python", "-m", "v2d.detectron2.lib.download_weights",
            "--output_dir", "/data/weights"]
    if model_sizes:
        cmd += ["--model_sizes"] + model_sizes

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ViTDet checkpoints")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_sizes", type=str, nargs="+", default=["b"], choices=["b", "l", "h"])
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_download_weights(args.output_dir, args.model_sizes, dev=args.dev)
