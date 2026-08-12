# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a ready_for_processing marker in the HITL S3 batch folder.

Usage:
    python mark_ready.py --dataset sc_office_4exo_1 --batch batch_20260419
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config() -> dict:
    with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ready_for_processing marker for a HITL batch",
    )
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--batch", required=True, help="Batch name (e.g. batch_20260419)")
    parser.add_argument("--dry_run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(config['datasets'].keys())}")
        sys.exit(1)

    dataset_cfg = config["datasets"][args.dataset]
    base_path = dataset_cfg["hitl_s3_base"]
    batch_path = os.path.join(base_path, args.batch) + "/"
    marker_path = os.path.join(base_path, args.batch, "markers", "ready_for_processing")

    # List items in the batch folder
    ls_result = subprocess.run(
        ["aws", "s3", "ls", batch_path],
        capture_output=True, text=True,
    )
    if ls_result.returncode != 0:
        print(f"  ERROR listing batch: {ls_result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    lines = [l.strip() for l in ls_result.stdout.strip().splitlines() if l.strip()]
    datasets = [l.split()[-1].rstrip("/") for l in lines if l.strip().startswith("PRE")]
    files = [l for l in lines if not l.strip().startswith("PRE")]

    print(f"Batch: {args.batch}")
    print(f"  Datasets: {len(datasets)}")
    for d in datasets:
        print(f"    {d}")
    if files:
        print(f"  Files: {len(files)}")
        for f in files:
            print(f"    {f}")
    print()

    if not datasets and not files:
        print("  Batch folder is empty, aborting.")
        sys.exit(1)

    # Create marker
    payload = json.dumps({"created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    cmd = f"echo '{payload}' | aws s3 cp - {marker_path}"
    print(f"  {cmd}")

    if args.dry_run:
        print("  [dry-run] skipping")
        return

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    print(f"  Marker created: {marker_path}")


if __name__ == "__main__":
    main()
