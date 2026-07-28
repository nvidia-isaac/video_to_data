# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-download GeoCalib weights into a torch hub cache directory."""
from __future__ import annotations

import argparse
import os


def download_geocalib(output_dir: str | None = None, weights: str = "pinhole") -> None:
    if weights not in {"pinhole", "distorted"}:
        raise ValueError("weights must be 'pinhole' or 'distorted'")
    if output_dir is None:
        output_dir = os.environ.get("CHECKPOINT_DIR")
        if output_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = output_dir

    from geocalib import GeoCalib

    print(f"Downloading GeoCalib weights={weights!r} into {output_dir}")
    GeoCalib(weights=weights)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GeoCalib checkpoint")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--weights", choices=("pinhole", "distorted"), default="pinhole")
    args = parser.parse_args()
    download_geocalib(output_dir=args.output_dir, weights=args.weights)

