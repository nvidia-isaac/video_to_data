#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct support surfaces from object still-poses (CLI driver).

Library code lives in ``robotic_grounding.retarget.support_recon``. This
script just parses CLI args and dispatches to that module so the same
logic can be reused from the planner orchestrator.

Usage:
  1. Produce *_loaded data first: the reconstruction v2d_task_library_loader
     load workflow (hand+object datasets), or soma_to_g1.py --save (whole-body)
  2. python scripts/reconstruct_support_surfaces.py --input_dir ... [--sequence_id ID]
"""

import argparse
from pathlib import Path

from robotic_grounding.retarget import HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import (
    add_sequence_filter_args,
    filter_sequence_ids,
    list_sequence_ids,
)
from robotic_grounding.retarget.dataset_registry import (
    get_all_dataset_names,
    get_dataset_config,
)
from robotic_grounding.retarget.support_recon import (
    GROUND_Z_THRESHOLD,
    _detect_parquet_schema,
    reconstruct_support_for_sequence,
)

# soma_g1 is a *processed* whole-body schema, not a registered source
# dataset, so it's kept as a separate default outside the registry.
# ``PROCESSED_DATASET_NAMES`` is the canonical set of --dataset values
# that select ``DEFAULT_INPUT_DIR_G1`` instead of going through
# ``get_dataset_config``; keep argparse choices and the dispatch in
# ``main`` in sync by referencing this constant.
PROCESSED_DATASET_NAMES: tuple[str, ...] = ("soma_g1", "motion_v1")
DEFAULT_INPUT_DIR_G1 = HUMAN_MOTION_DATA_DIR / "whole_body" / "soma"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct support surfaces from object still-poses.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Parquet root (e.g. taco_loaded/mano_object_only or arctic_loaded/mano_object_only).",
    )
    parser.add_argument(
        "--dataset",
        choices=get_all_dataset_names() + PROCESSED_DATASET_NAMES,
        default="oakink2",
        help="Dataset for default input_dir when --input_dir not set.",
    )
    add_sequence_filter_args(parser)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list sequence IDs and exit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .usda path for support surfaces (default: <sequence_id>_support.usda).",
    )
    parser.add_argument(
        "--ground_threshold",
        type=float,
        default=GROUND_Z_THRESHOLD,
        help=(
            f"Disks with z <= this are on the ground plane and skipped "
            f"(default: {GROUND_Z_THRESHOLD}m)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Parse CLI args and reconstruct support surfaces for the dataset."""
    args = _parse_args()
    if args.input_dir:
        input_dir = args.input_dir
    elif args.dataset in PROCESSED_DATASET_NAMES:
        input_dir = DEFAULT_INPUT_DIR_G1
    else:
        config = get_dataset_config(args.dataset)
        input_dir = (
            HUMAN_MOTION_DATA_DIR / config.name / f"{config.name}{config.loaded_suffix}"
        )
    if not input_dir.is_dir():
        print(
            f"Input dir not found: {input_dir}. Run the loader first "
            "(e.g. via the reconstruction load workflow)."
        )
        return

    # HOT3D and OakInk2 objects often rest at floor level (z ≤ 0.05 m). Don't
    # filter those disks out — the sim ground plane can coexist with a support
    # surface there.  Only applies when the user hasn't explicitly overridden
    # --ground_threshold.
    if args.ground_threshold == GROUND_Z_THRESHOLD:
        is_hot3d = args.dataset == "hot3d" or "hot3d" in str(input_dir).lower()
        is_oakink2 = args.dataset == "oakink2" or "oakink2" in str(input_dir).lower()
        if is_hot3d or is_oakink2:
            args.ground_threshold = -0.02
            detected = "HOT3D" if is_hot3d else "OakInk2"
            print(
                f"{detected} detected: ground_threshold set to -0.01 (keeping floor-level disks)"
            )

    sequence_ids = list_sequence_ids(str(input_dir))
    if not sequence_ids:
        print(f"No sequences in {input_dir}")
        return

    if args.list:
        for sid in sequence_ids:
            print(sid)
        return

    ids_to_process = filter_sequence_ids(sequence_ids, args)
    print(f"Processing {len(ids_to_process)} of {len(sequence_ids)} sequence(s).")

    schema = _detect_parquet_schema(input_dir)
    print(f"Detected schema: {schema}")

    for sequence_id in ids_to_process:
        reconstruct_support_for_sequence(
            input_dir,
            sequence_id,
            args.output,
            schema=schema,
            ground_z_threshold=args.ground_threshold,
        )


if __name__ == "__main__":
    main()
