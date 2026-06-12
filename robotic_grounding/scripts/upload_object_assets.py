#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upload a dataset's object assets (rigid URDFs + meshes) to CSS/swift.

The reconstruction ``v2d_<dataset>_load`` workflow fetches per-dataset object
assets from swift at runtime (the loader image stays lean and bakes nothing). It
expects them under::

    .../human_motion_data/<dataset>/object_assets/urdfs/<dataset>/
    .../human_motion_data/<dataset>/object_assets/meshes/<dataset>/

This walks the committed ``assets/{urdfs,meshes}/<dataset>`` trees and ``aws s3
sync``s them there, preserving the ``../../meshes/...`` sibling refs inside the
URDFs (urdfs/<dataset>/x.urdf -> ../../meshes/<dataset>/x.stl).

Only works for datasets whose meshes are committed (arctic, hot3d, taco,
oakink2). h2o/grab/dexycb keep object meshes inside the raw dataset on CSS and
are handled by the load workflow's mesh_dir pointing at the raw tree instead.

Usage::

    source scripts/setup_css_env.sh   # or have aws creds for pdx.s8k.io
    python scripts/upload_object_assets.py --dataset taco
    python scripts/upload_object_assets.py --dataset taco --dry-run
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
# Git-LFS pointer files are ~130 bytes of text beginning with this line; a sync
# of un-pulled pointers would silently upload stubs, so we refuse.
_LFS_MAGIC = b"version https://git-lfs"


# Geometry the loaders actually load as the object mesh. An un-pulled pointer
# here breaks the load, so we hard-fail. Other extensions (auxiliary .json,
# unused raw .ply scans like oakink2's) don't, so they're skipped — not blocked.
_PRIMARY_MESH_EXTS = {".obj", ".stl", ".glb"}


def _find_primary_stubs(root: Path) -> list[Path]:
    stubs = []
    for p in root.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in _PRIMARY_MESH_EXTS
            and p.stat().st_size < 300
        ):
            with open(p, "rb") as f:
                if f.read(len(_LFS_MAGIC)) == _LFS_MAGIC:
                    stubs.append(p)
    return stubs


def _sync(src: Path, dst: str, dry_run: bool, extra: list[str] | None = None) -> None:
    cmd = [
        "aws",
        "s3",
        "sync",
        str(src),
        dst,
        "--endpoint-url",
        ENDPOINT,
        "--region",
        "us-east-1",
    ] + (extra or [])
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    """Sync the given dataset's object_assets (urdfs + meshes) to swift."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ds = args.dataset

    any_uploaded = False
    for kind in ("urdfs", "meshes"):
        src = ASSET_DIR / kind / ds
        if not src.is_dir():
            print(f"WARNING: {src} missing — skipping {kind}/", file=sys.stderr)
            continue
        stubs = _find_primary_stubs(src)
        if stubs:
            print(
                f"ERROR: {len(stubs)} un-pulled Git-LFS pointer(s) for primary "
                f"geometry under {src} (e.g. {stubs[0]}). Run `git lfs pull` first.",
                file=sys.stderr,
            )
            sys.exit(1)
        dst = f"{BUCKET}/{ds}/object_assets/{kind}/{ds}/"
        # Skip un-pulled / unused .ply (e.g. oakink2 raw scans) — uploading 130B
        # LFS-pointer stubs would just litter the bucket; loaders read .obj/.stl/.glb.
        extra = ["--exclude", "*.ply"] if kind == "meshes" else []
        _sync(src, dst, args.dry_run, extra)
        any_uploaded = True

    if not any_uploaded:
        print(
            f"ERROR: no urdfs/ or meshes/ found for '{ds}' under {ASSET_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"done: object_assets for '{ds}'{' (dry-run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
