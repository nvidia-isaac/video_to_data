# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manage dataset-scoped MV HOI sequence blacklist entries.

Examples:
    python blacklist.py --dataset sc_office_4exo_1 --sequence <name> --reason "bad capture"
    python blacklist.py --dataset sc_office_4exo_1 --sequence <name> --remove
    python blacklist.py --dataset sc_office_4exo_1 --list
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MV_HOI_DIR = os.path.dirname(SCRIPT_DIR)
try:
    from .db import (
        DB_PATH,
        TEST_DB_PATH,
        get_blacklisted_sequences,
        init_db,
        remove_blacklisted_sequence,
        upsert_blacklisted_sequence,
    )
except ImportError:  # Direct script execution.
    from db import (
        DB_PATH,
        TEST_DB_PATH,
        get_blacklisted_sequences,
        init_db,
        remove_blacklisted_sequence,
        upsert_blacklisted_sequence,
    )

def load_config() -> dict:
    with open(os.path.join(MV_HOI_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def _validate_dataset(dataset: str) -> None:
    config = load_config()
    if dataset not in config["datasets"]:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {list(config['datasets'].keys())}")
        sys.exit(1)


def _show_list(dataset: str) -> None:
    rows = get_blacklisted_sequences(dataset, db_path=DB_PATH)
    if not rows:
        print(f"No blacklisted sequences for {dataset}")
        return

    table_rows = [
        (
            row["sequence_name"], row["blacklisted_at"] or "",
            row.get("created_by") or "", row["reason"] or "",
        )
        for row in rows
    ]
    headers = ("Sequence", "Blacklisted At", "Created By", "Reason")
    widths = [
        max(len(h), *(len(r[i]) for r in table_rows))
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    header_line = fmt.format(*headers)
    print(header_line)
    print("-" * len(header_line))
    for row in table_rows:
        print(fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage dataset-scoped sequence blacklist entries",
    )
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--sequence", help="Sequence name")
    parser.add_argument("--reason", help="Optional blacklist reason")
    parser.add_argument("--created-by", default=os.environ.get("USER", "operator"))
    parser.add_argument("--test", action="store_true", help="Use the separate test database")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--remove", action="store_true", help="Remove blacklist entry")
    group.add_argument("--list", action="store_true", help="List blacklist entries")
    args = parser.parse_args()

    if not args.list and not args.sequence:
        parser.error("--sequence is required unless --list is used")
    if args.reason and (args.list or args.remove):
        parser.error("--reason can only be used when adding or updating an entry")

    _validate_dataset(args.dataset)
    global DB_PATH
    if args.test:
        DB_PATH = TEST_DB_PATH
    init_db(DB_PATH)

    if args.list:
        _show_list(args.dataset)
        return

    if args.remove:
        removed = remove_blacklisted_sequence(
            args.dataset, args.sequence, db_path=DB_PATH,
        )
        if removed:
            print(f"Removed {args.sequence} from blacklist for {args.dataset}")
        else:
            print(f"No blacklist entry for {args.dataset}/{args.sequence}")
        return

    upsert_blacklisted_sequence(
        args.dataset, args.sequence, reason=args.reason,
        created_by=args.created_by, db_path=DB_PATH,
    )
    suffix = f": {args.reason}" if args.reason else ""
    print(f"Blacklisted {args.sequence} for {args.dataset}{suffix}")


if __name__ == "__main__":
    main()
