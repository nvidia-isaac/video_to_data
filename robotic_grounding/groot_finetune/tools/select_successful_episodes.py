#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select exactly N timeout-eligible joint rollouts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeInspection:
    """Validated metadata for one source episode."""

    path: Path
    source_success: bool
    frames: int
    time_series_keys: tuple[str, ...]


def inspect_episode(
    path: Path,
    expected_frames: int | None,
    timeout_termination: str,
) -> EpisodeInspection:
    """Validate one NPZ and infer its common time-series length."""
    with np.load(path, allow_pickle=False) as data:
        if "source_success" not in data.files:
            raise ValueError(f"{path}: missing scalar source_success marker")
        success_array = np.asarray(data["source_success"])
        if success_array.size != 1:
            raise ValueError(
                f"{path}: source_success must be scalar, got {success_array.shape}"
            )
        source_success = bool(success_array.item())
        if "termination_reasons" not in data.files:
            raise ValueError(f"{path}: missing termination_reasons")
        termination_reasons = tuple(
            str(reason) for reason in np.asarray(data["termination_reasons"])
        )
        if source_success != (timeout_termination in termination_reasons):
            raise ValueError(
                f"{path}: source_success does not match timeout termination "
                f"{timeout_termination!r} in {termination_reasons}"
            )

        lengths: dict[str, int] = {}
        for key in ("joint_pos", "action_target", "object_pose"):
            if key not in data.files:
                raise ValueError(f"{path}: missing required time series {key!r}")
            value = np.asarray(data[key])
            if value.shape[0] <= 0:
                raise ValueError(f"{path}: {key} has an empty time dimension")
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(f"{path}: {key} contains non-finite values")
            lengths[key] = int(value.shape[0])

    if not lengths:
        raise ValueError(f"{path}: no time-series arrays found")
    distinct_lengths = sorted(set(lengths.values()))
    if len(distinct_lengths) != 1:
        raise ValueError(f"{path}: inconsistent time-series lengths {lengths}")
    frames = distinct_lengths[0]
    if source_success and expected_frames is not None and frames != expected_frames:
        raise ValueError(f"{path}: expected {expected_frames} frames, found {frames}")
    return EpisodeInspection(path, source_success, frames, tuple(lengths))


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Export directory containing episode_*.npz; repeat in priority order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=positive_int, required=True)
    parser.add_argument("--expected-frames", type=positive_int)
    parser.add_argument(
        "--link",
        action="store_true",
        help="Hard-link selected episodes instead of copying them.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing selection after validating the new one.",
    )
    return parser.parse_args()


def _publish_selection(staging: Path, output: Path, *, replace: bool) -> None:
    """Publish a staged selection, rolling back if replacement cannot complete."""
    if not output.exists():
        staging.rename(output)
        return
    if not replace:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    backup = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.previous-", dir=str(output.parent))
    )
    backup.rmdir()
    output.rename(backup)
    try:
        staging.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    shutil.rmtree(backup)


def main() -> None:
    """Validate all sources, then atomically publish exactly the requested successes."""
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.replace:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    episode_paths: list[Path] = []
    input_manifests: list[dict] = []
    for input_dir in args.input:
        root = input_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {root}")
        paths = sorted(root.glob("episode_*.npz"))
        if not paths:
            raise FileNotFoundError(f"no episode_*.npz files under {root}")
        episode_paths.extend(paths)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing rollout manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format") != "joint_rollout"
            or manifest.get("schema_version") != 1
        ):
            raise ValueError(f"unsupported rollout manifest: {manifest_path}")
        input_manifests.append({"path": str(manifest_path), "manifest": manifest})

    contract_keys = (
        "schema_version",
        "embodiment_contract",
        "embodiment_contract_sha256",
        "fps",
        "joint_names",
        "object_names",
        "timeout_termination",
    )
    source_contracts = [
        {key: entry["manifest"].get(key) for key in contract_keys}
        for entry in input_manifests
    ]
    if any(contract != source_contracts[0] for contract in source_contracts[1:]):
        raise ValueError("input rollout manifests do not share one exact contract")
    timeout_termination = source_contracts[0]["timeout_termination"]
    if not isinstance(timeout_termination, str):
        raise ValueError(
            "input rollout manifests do not declare one timeout termination"
        )

    unique_paths = list(dict.fromkeys(path.resolve() for path in episode_paths))
    inspections = [
        inspect_episode(path, args.expected_frames, timeout_termination)
        for path in unique_paths
    ]
    successful = [inspection for inspection in inspections if inspection.source_success]
    if len(successful) < args.target:
        raise RuntimeError(
            f"only {len(successful)} timeout-eligible episodes available from "
            f"{len(inspections)} inputs; target is {args.target}"
        )
    selected = successful[: args.target]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    selected_entries = []
    try:
        for index, inspection in enumerate(selected):
            destination = staging / f"episode_{index:06d}.npz"
            if args.link:
                destination.hardlink_to(inspection.path)
            else:
                shutil.copy2(inspection.path, destination)
            selected_entries.append(
                {
                    "destination": destination.name,
                    "source": str(inspection.path),
                    "frames": inspection.frames,
                    "time_series_keys": list(inspection.time_series_keys),
                    "source_success": True,
                }
            )

        manifest = {
            "format": "joint_rollout_selection",
            **source_contracts[0],
            "input_directories": [str(path.resolve()) for path in args.input],
            "input_episode_count": len(inspections),
            "source_successful_episode_count": len(successful),
            "source_unsuccessful_episode_count": len(inspections) - len(successful),
            "selected_episode_count": len(selected),
            "expected_frames": args.expected_frames,
            "selection_order": "input argument order, then lexical episode filename",
            "selected": selected_entries,
            "source_manifests": input_manifests,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        _publish_selection(staging, output, replace=args.replace)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    rate = len(successful) / len(inspections)
    if not math.isfinite(rate):
        raise RuntimeError("non-finite eligibility rate")
    print(
        f"[SUMMARY] input={len(inspections)} timeout_eligible={len(successful)} "
        f"ineligible={len(inspections) - len(successful)} "
        f"selected={len(selected)} eligibility_rate={rate:.2%}"
    )
    print(f"[INFO] wrote selection: {output}")


if __name__ == "__main__":
    main()
