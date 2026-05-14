#!/usr/bin/env python3
"""Reconstruct support surfaces from object still-poses (CLI driver).

Library code lives in ``robotic_grounding.retarget.support_recon``. This
script just parses CLI args and dispatches to that module so the same
logic can be reused from the planner orchestrator.

Usage:
  1. Run retarget/loader first (e.g. taco_loader.py --save, nvhuman_to_g1.py --save)
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

# nvhuman_g1 is a *processed* whole-body schema, not a registered source
# dataset, so it's kept as a separate default outside the registry.
DEFAULT_INPUT_DIR_G1 = HUMAN_MOTION_DATA_DIR / "nvhuman_g1_processed"


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
        choices=get_all_dataset_names() + ("nvhuman_g1", "motion_v1"),
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
    elif args.dataset in {"nvhuman_g1", "motion_v1"}:
        input_dir = DEFAULT_INPUT_DIR_G1
    else:
        config = get_dataset_config(args.dataset)
        input_dir = (
            HUMAN_MOTION_DATA_DIR / config.name / f"{config.name}{config.loaded_suffix}"
        )
    if not input_dir.is_dir():
        print(
            f"Input dir not found: {input_dir}. Run the loader first "
            "(e.g. taco_loader.py --save)."
        )
        return

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
