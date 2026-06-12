#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Fetch a dataset's object assets (rigid URDFs + meshes) from CSS/swift.

The inverse of ``upload_object_assets.py``. Per-dataset object URDFs + meshes are
no longer committed to the repo (they are large and regenerable); they live on
CSS under::

    .../human_motion_data/<dataset>/object_assets/urdfs/<dataset>/
    .../human_motion_data/<dataset>/object_assets/meshes/<dataset>/

This downloads them back into the committed layout the pipeline expects::

    source/robotic_grounding/robotic_grounding/assets/urdfs/<dataset>/
    source/robotic_grounding/robotic_grounding/assets/meshes/<dataset>/

so the URDFs' ``../../meshes/...`` sibling refs resolve, generate_rigid_urdfs
skips (idempotent) and Isaac Sim can load the meshes for retarget / training /
replay. Run it once after a fresh clone (or in the image build) before using a
dataset locally.

Usage::

    source scripts/setup_css_env.sh   # or have aws creds for pdx.s8k.io
    python scripts/fetch_object_assets.py --dataset taco
    python scripts/fetch_object_assets.py --dataset all
    python scripts/fetch_object_assets.py --dataset taco --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ASSET_DIR = (
    Path(__file__).resolve().parents[1]
    / "source/robotic_grounding/robotic_grounding/assets"
)
# TODO(public-release): internal NVIDIA CSS endpoint + bucket — replace before open-sourcing.
ENDPOINT = "https://pdx.s8k.io"
BUCKET = "s3://datasets/v2d/human_motion_data"

# Datasets whose object assets are hosted as object_assets/ on CSS. h2o/grab/
# dexycb are intentionally absent — their meshes live inside the raw dataset and
# the loaders read them from there, so there is nothing to fetch here.
DATASETS = ("arctic", "taco", "oakink2", "hot3d")


def _sync(src: str, dst: Path, dry_run: bool) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws",
        "s3",
        "sync",
        src,
        str(dst),
        "--endpoint-url",
        ENDPOINT,
        "--region",
        "us-east-1",
    ]
    print("+", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _fetch_dataset(ds: str, dry_run: bool) -> None:
    for kind in ("urdfs", "meshes"):
        src = f"{BUCKET}/{ds}/object_assets/{kind}/{ds}/"
        dst = ASSET_DIR / kind / ds
        rc = _sync(src, dst, dry_run)
        if rc != 0:
            print(
                f"ERROR: failed to fetch {kind} for '{ds}' from {src} "
                f"(is it uploaded? see upload_object_assets.py)",
                file=sys.stderr,
            )
            sys.exit(rc)


def main() -> None:
    """Download the given dataset's object_assets (urdfs + meshes) from swift."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name, or 'all' for: {', '.join(DATASETS)}",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = DATASETS if args.dataset == "all" else (args.dataset,)
    if args.dataset != "all" and args.dataset not in DATASETS:
        print(
            f"WARNING: '{args.dataset}' is not a known object_assets dataset "
            f"({', '.join(DATASETS)}); trying anyway.",
            file=sys.stderr,
        )
    for ds in targets:
        _fetch_dataset(ds, args.dry_run)
    print(
        f"done: fetched object_assets for {', '.join(targets)}"
        f"{' (dry-run)' if args.dry_run else ''}"
    )


if __name__ == "__main__":
    main()
