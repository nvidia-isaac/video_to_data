#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pull a SUBSET of raw arctic sequences from CSS (PDX) for local testing.

Unlike sync_css_data.py (which pulls pipeline *outputs*), this downloads the raw
`dataset/{subject}/{seq}.mano.npy` + `.object.npy` files the arctic loader reads,
for a handful of sequences — enough to run a loader / equivalence test locally.

Prereqs:
  pip install boto3
  source scripts/setup_css_env.sh        # set CSS_ACCESS_KEY / CSS_SECRET_KEY

Usage:
  python scripts/pull_arctic_subset.py                       # 2 seqs, any
  python scripts/pull_arctic_subset.py --pattern 's01_box' --max 2
  python scripts/pull_arctic_subset.py --list                # just list, no download
"""
import argparse
import os
import re
import sys
from pathlib import Path

import boto3
from botocore.config import Config

BUCKET = "datasets"
RAW_PREFIX = "v2d/human_motion_data/arctic/dataset/"
DEST = (
    Path(__file__).resolve().parent.parent
    / "source/robotic_grounding/robotic_grounding/assets/human_motion_data/arctic/dataset"
)


def _client() -> "boto3.client":
    ak, sk = os.environ.get("CSS_ACCESS_KEY", ""), os.environ.get("CSS_SECRET_KEY", "")
    if not ak or not sk:
        sys.exit("Set CSS_ACCESS_KEY / CSS_SECRET_KEY: source scripts/setup_css_env.sh")
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("CSS_ENDPOINT_URL", "https://pdx.s8k.io"),
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name=os.environ.get("CSS_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )


def main() -> None:
    """Download (or list) a small subset of arctic raw .mano.npy sequences from CSS."""
    p = argparse.ArgumentParser()
    p.add_argument("--pattern", default=".*", help="regex over the .mano.npy key")
    p.add_argument("--max", type=int, default=2, help="max sequences to pull")
    p.add_argument("--list", action="store_true", help="list matches, don't download")
    args = p.parse_args()

    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    mano_keys = [
        obj["Key"]
        for page in paginator.paginate(Bucket=BUCKET, Prefix=RAW_PREFIX)
        for obj in page.get("Contents", [])
        if obj["Key"].endswith(".mano.npy")
        and "scissor" not in obj["Key"]
        and re.search(args.pattern, obj["Key"])
    ]
    mano_keys.sort()
    print(f"{len(mano_keys)} sequence(s) match; taking first {args.max}")
    for mk in mano_keys[: args.max]:
        seq = mk[len(RAW_PREFIX) : -len(".mano.npy")]
        print(f"  {seq}")
        if args.list:
            continue
        for key in (mk, mk[: -len(".mano.npy")] + ".object.npy"):
            dst = DEST / key[len(RAW_PREFIX) :]
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, key, str(dst))
            print(f"    -> {dst}")
    if not args.list:
        print(f"Done. Raw arctic under: {DEST}")


if __name__ == "__main__":
    main()
