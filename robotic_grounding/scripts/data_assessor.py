#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate retargeted sequence quality using auto-discovered checks.

Quality checks are Python files in ``scripts/data_quality_checks/`` that
define a ``check(data, **kwargs)`` function.  Adding a new check is as
simple as dropping a ``.py`` file there — no other files need to change.

Usage::

    # Report metrics for all sequences
    python scripts/data_assessor.py --input_dir <processed_dir>

    # Output CSV
    python scripts/data_assessor.py --input_dir <dir> --output_csv metrics.csv

    # Reject mode: write failed sequence IDs to file
    python scripts/data_assessor.py --input_dir <dir> --reject --output_reject rejected.txt

    # Override a check threshold
    python scripts/data_assessor.py --input_dir <dir> --set fitting_error.threshold=0.08

    # Disable a check
    python scripts/data_assessor.py --input_dir <dir> --disable fitting_error

    # Run only specific checks
    python scripts/data_assessor.py --input_dir <dir> --checks fitting_error,ik_task_error
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq

# Add the scripts directory to the path so data_quality_checks can be imported
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from data_quality_checks import discover_checks  # noqa: E402


def _parse_overrides(set_args: list[str]) -> dict[str, dict[str, float | str]]:
    """Parse ``--set check_name.param=value`` args into nested dict.

    Returns:
        ``{check_name: {param: value}}``
    """
    overrides: dict[str, dict[str, float | str]] = {}
    for arg in set_args or []:
        if "=" not in arg:
            print(f"Warning: ignoring malformed --set arg: {arg}", file=sys.stderr)
            continue
        key, val = arg.split("=", 1)
        parts = key.split(".", 1)
        if len(parts) != 2:
            print(
                f"Warning: --set must be check_name.param=value, got: {arg}",
                file=sys.stderr,
            )
            continue
        check_name, param = parts
        try:
            overrides.setdefault(check_name, {})[param] = float(val)
        except ValueError:
            overrides.setdefault(check_name, {})[param] = val
    return overrides


def _find_parquet_dirs(input_dir: Path) -> list[tuple[str, Path]]:
    """Find all sequence partition directories under input_dir.

    Returns:
        List of ``(sequence_id, partition_dir)`` tuples.
    """
    results: list[tuple[str, Path]] = []
    for seq_dir in sorted(input_dir.glob("sequence_id=*")):
        if not seq_dir.is_dir():
            continue
        seq_id = seq_dir.name.replace("sequence_id=", "")
        # Find first robot_name partition
        robot_dirs = list(seq_dir.glob("robot_name=*"))
        if robot_dirs:
            results.append((seq_id, robot_dirs[0]))
        else:
            # Parquet files directly in sequence_id dir
            results.append((seq_id, seq_dir))
    return results


def _load_parquet_data(parquet_dir: Path) -> dict | None:
    """Read a single partitioned Parquet directory into a dict."""
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        return None
    try:
        return pq.read_table(str(parquet_files[0])).to_pydict()
    except Exception as e:
        print(f"Warning: failed to read {parquet_files[0]}: {e}", file=sys.stderr)
        return None


def _run_check(
    check_fn: Callable[..., dict],
    data: dict,
    overrides: dict[str, float | str],
    seq_dir: Path | None = None,
) -> dict:
    """Run a single check with optional parameter overrides.

    Checks that declare a ``seq_dir`` parameter receive the sequence's
    partition directory — the ``robot_name=<robot>`` dir — so filesystem-aware
    checks (e.g. sentinel-based ones) can read sibling files.
    """
    sig = inspect.signature(check_fn)
    kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name == "data":
            continue
        if param_name == "seq_dir":
            kwargs[param_name] = seq_dir
            continue
        if param_name in overrides:
            kwargs[param_name] = type(param.default)(overrides[param_name])
        elif param.default is not inspect.Parameter.empty:
            kwargs[param_name] = param.default
    return check_fn(data, **kwargs)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate retargeted sequence quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing processed Parquet partitions.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Write per-sequence metrics to CSV.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Write per-sequence metrics to JSON.",
    )
    parser.add_argument(
        "--reject",
        action="store_true",
        help="Enable rejection mode.",
    )
    parser.add_argument(
        "--output_reject",
        type=Path,
        default=None,
        help="Write rejected sequence IDs to file (one per line).",
    )
    parser.add_argument(
        "--checks",
        type=str,
        default=None,
        help="Comma-separated list of checks to run (default: all).",
    )
    parser.add_argument(
        "--disable",
        type=str,
        default=None,
        help="Comma-separated list of checks to disable.",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="set_args",
        metavar="CHECK.PARAM=VALUE",
        help="Override a check parameter (e.g., fitting_error.threshold=0.08).",
    )
    parser.add_argument(
        "--sequence_id",
        type=str,
        default=None,
        help="Evaluate a single sequence.",
    )
    parser.add_argument(
        "--sequence_pattern",
        type=str,
        default=None,
        help="Regex pattern to filter sequences.",
    )
    args = parser.parse_args()

    # Discover checks
    all_checks = discover_checks()
    if not all_checks:
        print(
            "No quality checks found in scripts/data_quality_checks/", file=sys.stderr
        )
        sys.exit(1)

    # Filter checks
    if args.checks:
        selected = set(args.checks.split(","))
        all_checks = {k: v for k, v in all_checks.items() if k in selected}
    if args.disable:
        disabled = set(args.disable.split(","))
        all_checks = {k: v for k, v in all_checks.items() if k not in disabled}

    check_names = sorted(all_checks)
    overrides = _parse_overrides(args.set_args)

    print(f"Running {len(check_names)} checks: {', '.join(check_names)}")
    print(f"Input: {args.input_dir}\n")

    # Find sequences
    sequences = _find_parquet_dirs(args.input_dir)
    if not sequences:
        print(f"No sequences found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Apply sequence filters
    if args.sequence_id:
        sequences = [(sid, p) for sid, p in sequences if sid == args.sequence_id]
    if args.sequence_pattern:
        pat = re.compile(args.sequence_pattern)
        sequences = [(sid, p) for sid, p in sequences if pat.match(sid)]

    # Run checks
    results: list[dict] = []
    rejected: list[str] = []
    pass_counts = {name: 0 for name in check_names}
    total = len(sequences)

    for seq_id, parquet_dir in sequences:
        data = _load_parquet_data(parquet_dir)
        if data is None:
            print(f"  SKIP {seq_id}: could not read parquet")
            continue

        row: dict = {"sequence_id": seq_id}
        all_pass = True

        for check_name in check_names:
            check_fn = all_checks[check_name]
            check_overrides = overrides.get(check_name, {})
            result = _run_check(check_fn, data, check_overrides, seq_dir=parquet_dir)

            row[f"{check_name}_score"] = result["score"]
            row[f"{check_name}_pass"] = result["pass"]
            row[f"{check_name}_reason"] = result["reason"]

            if result["pass"]:
                pass_counts[check_name] += 1
            else:
                all_pass = False

        row["pass_all"] = all_pass
        results.append(row)

        if not all_pass:
            rejected.append(seq_id)

    # Console summary
    print(f"{'='*60}")
    print(f"Results: {total} sequences evaluated\n")

    print(f"{'Check':<25} {'Pass Rate':>12} {'Mean Score':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*12}")
    for name in check_names:
        pass_rate = pass_counts[name] / total if total > 0 else 0
        scores = [r[f"{name}_score"] for r in results]
        mean_score = sum(scores) / len(scores) if scores else 0
        print(f"{name:<25} {pass_rate:>11.1%} {mean_score:>12.4f}")

    passed = sum(1 for r in results if r["pass_all"])
    print(f"\nOverall: {passed}/{total} sequences pass all checks")

    if rejected:
        print(f"\nRejected ({len(rejected)}):")
        for seq_id in rejected[:20]:
            row = next(r for r in results if r["sequence_id"] == seq_id)
            fails = [n for n in check_names if not row[f"{n}_pass"]]
            print(f"  {seq_id}: failed {', '.join(fails)}")
        if len(rejected) > 20:
            print(f"  ... and {len(rejected) - 20} more")

    # Write CSV
    if args.output_csv:
        fieldnames = (
            ["sequence_id"]
            + [f"{n}_{s}" for n in check_names for s in ("score", "pass", "reason")]
            + ["pass_all"]
        )
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nCSV written to {args.output_csv}")

    # Write JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON written to {args.output_json}")

    # Write reject list
    if args.reject and args.output_reject:
        with open(args.output_reject, "w") as f:
            for seq_id in rejected:
                f.write(f"{seq_id}\n")
        print(
            f"Rejected sequences written to {args.output_reject} ({len(rejected)} IDs)"
        )

    # Exit with non-zero if any rejected (useful in CI)
    if args.reject and rejected:
        sys.exit(1)


if __name__ == "__main__":
    main()
