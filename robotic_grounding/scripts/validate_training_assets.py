#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

r"""Validate that all assets required for RL training exist on disk.

Run this BEFORE starting Isaac Sim to catch missing URDFs, meshes, or
malformed parquets early — rather than 45 seconds into GPU startup.

Usage::

    # Validate a single motion file
    python scripts/validate_training_assets.py \\
        --motion_file hot3d/hot3d_processed/P0001_4bf4e21a/sharpa_wave

    # Validate all processed sequences for a dataset
    python scripts/validate_training_assets.py --dataset hot3d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq

# Add source to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source" / "robotic_grounding"))

from robotic_grounding.assets import ASSET_DIR  # noqa: E402
from robotic_grounding.retarget.dataset_registry import (  # noqa: E402
    get_all_dataset_names,
    get_dataset_config,
)

HUMAN_MOTION_DATA_DIR = Path(ASSET_DIR) / "human_motion_data"


def validate_motion_file(motion_file: str) -> list[str]:
    """Validate a single motion file (parquet partition directory).

    Returns a list of error strings. Empty list means all assets are valid.
    """
    errors: list[str] = []

    # Resolve the path
    path = Path(motion_file)
    if not path.is_absolute():
        # Try as relative to HUMAN_MOTION_DATA_DIR
        parts = motion_file.strip("/").split("/")
        if len(parts) >= 4:
            dataset, dataset_retargeted, seq_id, robot = parts[:4]
            path = (
                HUMAN_MOTION_DATA_DIR
                / dataset
                / dataset_retargeted
                / f"sequence_id={seq_id}"
                / f"robot_name={robot}"
            )

    if not path.exists():
        errors.append(f"Motion file not found: {path}")
        return errors

    # Find parquet files
    parquet_files = list(path.glob("*.parquet")) if path.is_dir() else [path]
    if not parquet_files:
        errors.append(f"No parquet files in {path}")
        return errors

    # Read and validate
    try:
        data = pq.read_table(str(parquet_files[0])).to_pydict()
    except Exception as e:
        errors.append(f"Failed to read parquet {parquet_files[0]}: {e}")
        return errors

    # Check URDF paths
    urdf_paths = data.get("object_urdf_paths", [[]])[0] or []
    mesh_paths = data.get("object_mesh_paths", [[]])[0] or []

    for p in urdf_paths:
        if p and not Path(p).exists():
            errors.append(f"Missing URDF: {p}")

    for p in mesh_paths:
        if p and not Path(p).exists():
            errors.append(f"Missing mesh: {p}")

    # Note: objects without explicit urdf_paths may be resolved at training
    # time via the object registry or the mesh-derived URDF fallback in
    # SceneConfig._build_scene_objects. We only flag paths that ARE in the
    # parquet but point to missing files.

    return errors


def validate_dataset(dataset: str) -> dict[str, list[str]]:
    """Validate all processed sequences for a dataset.

    Returns a dict mapping sequence_id to list of errors (empty = valid).
    """
    config = get_dataset_config(dataset)
    processed_dir = (
        HUMAN_MOTION_DATA_DIR / config.name / f"{config.name}{config.processed_suffix}"
    )

    if not processed_dir.exists():
        return {"__dataset__": [f"Processed directory not found: {processed_dir}"]}

    results: dict[str, list[str]] = {}
    seq_dirs = sorted(processed_dir.glob("sequence_id=*/robot_name=*"))

    if not seq_dirs:
        return {"__dataset__": [f"No sequences found in {processed_dir}"]}

    for seq_dir in seq_dirs:
        seq_id = seq_dir.parent.name.replace("sequence_id=", "")
        errors = validate_motion_file(str(seq_dir))
        if errors:
            results[seq_id] = errors

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate training assets (URDFs, meshes, parquets)."
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        help="Validate a single motion file path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(get_all_dataset_names()) + ["all"],
        help="Validate all processed sequences for a dataset.",
    )
    args = parser.parse_args()

    if not args.motion_file and not args.dataset:
        parser.error("Provide --motion_file or --dataset")

    total_errors = 0

    if args.motion_file:
        errors = validate_motion_file(args.motion_file)
        if errors:
            print(f"FAIL: {args.motion_file}")
            for e in errors:
                print(f"  - {e}")
            total_errors += len(errors)
        else:
            print(f"OK: {args.motion_file}")

    if args.dataset:
        datasets = (
            list(get_all_dataset_names()) if args.dataset == "all" else [args.dataset]
        )
        for ds in datasets:
            results = validate_dataset(ds)
            if not results:
                print(f"OK: {ds} (all sequences valid)")
            else:
                for seq_id, errors in results.items():
                    print(f"FAIL: {ds}/{seq_id}")
                    for e in errors:
                        print(f"  - {e}")
                    total_errors += len(errors)

    if total_errors:
        print(f"\n{total_errors} error(s) found.")
        sys.exit(1)
    else:
        print("\nAll assets valid.")


if __name__ == "__main__":
    main()
