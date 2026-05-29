# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
import argparse
import subprocess


def download_weights(output_dir: str):
    subprocess.run([
        "hf", "download",
        "facebook/sam2.1-hiera-large",
        "--local-dir", output_dir,
    ], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SAM 2.1 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoint")
    args = parser.parse_args()
    download_weights(args.output_dir)
