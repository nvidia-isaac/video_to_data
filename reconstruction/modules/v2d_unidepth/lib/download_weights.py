# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os

def download_unidepth(output_dir: str | None = None) -> None:
    if output_dir is None:
        output_dir = os.environ.get("CHECKPOINT_DIR")
        if output_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
    print(output_dir)
    print(os.path.abspath(output_dir))
    subprocess.run(["hf", "download", "lpiccinelli/unidepth-v2-vitl14", "--local-dir", output_dir], check=True)
    subprocess.run(["echo", "UniDepth v2 checkpoint downloaded."], check=True)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Download UniDepth v2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoint")
    args = parser.parse_args()
    download_unidepth(output_dir=args.output_dir)