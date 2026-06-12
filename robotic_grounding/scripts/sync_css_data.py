#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download processed data from CSS (NVIDIA PDX storage) to local repo.

Syncs files from the remote S3-compatible bucket to the local assets directory,
skipping files that already exist locally with the same size.

By default all three outputs (loaded, processed, support_surfaces) are synced.
Use --loaded, --processed, or --support-surfaces to sync only specific outputs.

Prerequisites:
  - boto3: pip install boto3
  - CSS credentials configured via environment variables:
      source scripts/setup_css_env.sh

Usage:
  python scripts/sync_css_data.py --dataset taco
  python scripts/sync_css_data.py --dataset taco --component processed
  python scripts/sync_css_data.py --dataset all --component loaded
  python scripts/sync_css_data.py --dataset hot3d --component support_surfaces --dry-run
"""

import argparse
import os
import re
import sys
from pathlib import Path

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

# Dataset registry — single source of truth. Add new datasets there, not here.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "source" / "robotic_grounding"),
)
from robotic_grounding.retarget.dataset_registry import (  # noqa: E402
    get_all_dataset_names,
)

DATASETS = get_all_dataset_names()
STAGES = ("loaded", "processed", "support_surfaces")

LOCAL_ASSETS_DIR = (
    Path(__file__).resolve().parent.parent
    / "source"
    / "robotic_grounding"
    / "robotic_grounding"
    / "assets"
    / "human_motion_data"
)


def _remote_prefix(dataset: str, stage: str) -> str:
    """Return the S3 prefix for a given dataset and stage."""
    if stage == "support_surfaces":
        return f"{BASE_PREFIX}/{dataset}/support_surfaces/"
    return f"{BASE_PREFIX}/{dataset}/{dataset}_{stage}/"


def _local_dir(dataset: str, stage: str) -> Path:
    """Return the local directory for a given dataset and stage."""
    if stage == "support_surfaces":
        return LOCAL_ASSETS_DIR / dataset / "support_surfaces"
    return LOCAL_ASSETS_DIR / dataset / f"{dataset}_{stage}"


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
        config=Config(connect_timeout=10),
    )


def sync(
    client: "boto3.client",
    dataset: str,
    stage: str,
    pattern: str | None = None,
    dry_run: bool = False,
) -> None:
    """Download all objects under the remote prefix to the local directory.

    Skips files that already exist locally with the same size.
    When *pattern* is given, only files whose top-level directory (sequence ID)
    matches the regex are downloaded.
    """
    prefix = _remote_prefix(dataset, stage)
    local_root = _local_dir(dataset, stage)
    regex = re.compile(pattern) if pattern else None

    print(f"Syncing s3://{BUCKET}/{prefix} -> {local_root}")
    if regex:
        print(f"  filtering sequences: {pattern}")

    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    skipped = 0
    filtered = 0

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = key[len(prefix) :]
            if not rel_path:
                continue

            # Filter by sequence ID (first path component under the prefix)
            if regex:
                seq_id = rel_path.split("/")[0]
                if not regex.search(seq_id):
                    filtered += 1
                    continue

            local_path = local_root / rel_path
            remote_size = obj["Size"]

            if local_path.exists() and local_path.stat().st_size == remote_size:
                skipped += 1
                continue

            if dry_run:
                print(f"  [dry-run] would download: {rel_path} ({remote_size} bytes)")
                downloaded += 1
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  downloading: {rel_path} ({remote_size} bytes)")
            client.download_file(BUCKET, key, str(local_path))
            downloaded += 1

    summary = f"  done: {downloaded} downloaded, {skipped} skipped (already up to date)"
    if regex:
        summary += f", {filtered} filtered out"
    print(summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download processed data from CSS (PDX) to local repo.",
    )
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS, "all"],
        required=True,
        help="Dataset to sync, or 'all' for every dataset.",
    )
    parser.add_argument(
        "--component",
        choices=["all", "loaded", "processed", "support_surfaces"],
        default="all",
        help="Which output to sync: all (default), loaded, processed, or support_surfaces.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Regex pattern to filter sequence IDs (e.g., '.*screw.*').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List files that would be downloaded without downloading.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    client = get_s3_client()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    stages = list(STAGES) if args.component == "all" else [args.component]

    for dataset in datasets:
        for stage in stages:
            sync(client, dataset, stage, pattern=args.pattern, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
