#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Materialize a versioned OSMO dataset bundle for local consumption.

The download destination defaults to ``HUMAN_MOTION_DATA_DIR/{dataset}``, which
is the layout used by training, validation, and visualization code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source" / "robotic_grounding"))

from robotic_grounding.retarget.bundle_paths import (  # noqa: E402
    get_dataset_bundle_root,
)
from robotic_grounding.retarget.dataset_registry import (  # noqa: E402
    get_all_dataset_names,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a pinned per-dataset OSMO bundle into the local data root."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=get_all_dataset_names(),
        help="Dataset bundle to materialize.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help=(
            "OSMO dataset name or name:version. Defaults to " "`v2d_{dataset}_bundle`."
        ),
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Optional OSMO dataset version. Ignored if --bundle already includes ':'.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination directory. Defaults to HUMAN_MOTION_DATA_DIR/{dataset}.",
    )
    parser.add_argument(
        "--regex",
        default=None,
        help="Optional regex passed through to `osmo dataset download --regex`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the osmo command without executing it.",
    )
    return parser.parse_args()


def _bundle_ref(dataset: str, bundle: str | None, version: str | None) -> str:
    ref = bundle or f"v2d_{dataset}_bundle"
    if version and ":" not in ref:
        ref = f"{ref}:{version}"
    return ref


def main() -> None:
    """Resolve dataset/version, then shell out to ``osmo dataset download``."""
    args = _parse_args()
    bundle_ref = _bundle_ref(args.dataset, args.bundle, args.version)
    dest = args.dest or get_dataset_bundle_root(args.dataset)

    cmd = ["osmo", "dataset", "download", bundle_ref, str(dest)]
    if args.regex:
        cmd.extend(["--regex", args.regex])

    print(" ".join(cmd))
    if args.dry_run:
        return

    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
