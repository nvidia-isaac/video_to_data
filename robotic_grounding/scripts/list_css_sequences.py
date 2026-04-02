#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""List available sequences in CSS (NVIDIA PDX storage).

Connects to the PDX object store via the S3-compatible API and lists
sequence-level prefixes for each dataset stage (raw, loaded, processed).

Prerequisites:
  - boto3: pip install boto3
  - CSS credentials configured via environment variables:
      source scripts/setup_css_env.sh

Usage:
  python scripts/list_css_sequences.py
  python scripts/list_css_sequences.py --dataset taco
  python scripts/list_css_sequences.py --dataset arctic --stage loaded
  python scripts/list_css_sequences.py --pattern '.*screw.*'
"""

import argparse
import os
import re
import sys

import boto3
from botocore.config import Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENDPOINT_URL = os.environ.get("CSS_ENDPOINT_URL", "https://pdx.s8k.io")
ACCESS_KEY = os.environ.get("CSS_ACCESS_KEY", "")
SECRET_KEY = os.environ.get("CSS_SECRET_KEY", "")
REGION = os.environ.get("CSS_REGION", "us-east-1")

BUCKET = "datasets"
BASE_PREFIX = "v2d/human_motion_data"

DATASETS = ("taco", "arctic", "oakink2", "hot3d")

# Maps dataset + stage to the S3 prefix where sequence directories live.
STAGE_PREFIXES: dict[str, dict[str, str]] = {
    "taco": {
        "raw": f"{BASE_PREFIX}/taco/dataset/Hand_Poses/",
        "loaded": f"{BASE_PREFIX}/taco/taco_loaded/",
        "processed": f"{BASE_PREFIX}/taco/taco_processed/",
    },
    "arctic": {
        "raw": f"{BASE_PREFIX}/arctic/dataset/",
        "loaded": f"{BASE_PREFIX}/arctic/arctic_loaded/",
        "processed": f"{BASE_PREFIX}/arctic/arctic_processed/",
    },
    "oakink2": {
        "raw": f"{BASE_PREFIX}/oakink2/dataset/",
        "loaded": f"{BASE_PREFIX}/oakink2/oakink2_loaded/",
        "processed": f"{BASE_PREFIX}/oakink2/oakink2_processed/",
    },
    "hot3d": {
        "raw": f"{BASE_PREFIX}/hot3d/dataset/",
        "loaded": f"{BASE_PREFIX}/hot3d/hot3d_loaded/",
        "processed": f"{BASE_PREFIX}/hot3d/hot3d_processed/",
    },
}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def get_s3_client() -> "boto3.client":
    """Create an S3 client for CSS (PDX)."""
    if not ACCESS_KEY or not SECRET_KEY:
        print(
            "Error: Set CSS_ACCESS_KEY and CSS_SECRET_KEY environment variables.\n"
            "  source scripts/setup_css_env.sh",
            file=sys.stderr,
        )
        sys.exit(1)
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
        config=Config(connect_timeout=5),
    )


def list_prefixes(client: "boto3.client", prefix: str) -> list[str]:
    """List immediate sub-prefixes (directories) under *prefix*.

    Uses the S3 delimiter to get only one level of hierarchy.
    """
    names: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"][len(prefix) :].rstrip("/")
            if name:
                names.append(name)
    return sorted(names)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List available sequences on CSS (PDX storage).",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        default=None,
        help="Limit to a single dataset (default: all).",
    )
    parser.add_argument(
        "--stage",
        choices=("raw", "loaded", "processed"),
        default=None,
        help="Limit to a single pipeline stage (default: all).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Regex pattern to filter sequence names.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()
    client = get_s3_client()

    datasets = [args.dataset] if args.dataset else list(DATASETS)
    stages = [args.stage] if args.stage else ["raw", "loaded", "processed"]

    for dataset in datasets:
        for stage in stages:
            prefix = STAGE_PREFIXES[dataset][stage]
            names = list_prefixes(client, prefix)

            if args.pattern:
                regex = re.compile(args.pattern)
                names = [n for n in names if regex.search(n)]

            header = f"{dataset}/{stage}"
            if not names:
                print(f"{header}: (empty)")
                continue

            print(f"{header}: {len(names)} sequences")
            for name in names:
                print(f"  {name}")
            print()


if __name__ == "__main__":
    main()
