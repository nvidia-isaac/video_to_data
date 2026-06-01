# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate accuracy checks.

Reads chamfer-distance metrics from ``eval_chamfer_object`` and
``eval_chamfer_human``, plus the mask-containment result from
``check_object_mask``.  Writes a consolidated ``check_accuracy.json`` with
individual check results.

Exits non-zero if any check fails (chamfer above threshold, or
check_object_mask already failed), which marks the OSMO task as FAILED and
blocks the downstream HITL upload task.

Usage (inside container):
    python -m v2d.mv.postprocess.lib.check_accuracy \
        --chamfer_object_dir /data/eval_chamfer_object \
        --chamfer_human_dir  /data/eval_chamfer_human \
        --check_object_mask_dir /data/check_object_mask \
        --output_dir         /data/check_accuracy \
        --max_chamfer_object 30.0 \
        --max_chamfer_human  30.0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _find_metrics(directory: str) -> dict:
    for name in ("chamfer_metrics.json", "metrics.json"):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    jsons = sorted(glob.glob(os.path.join(directory, "*.json")))
    if jsons:
        with open(jsons[0]) as f:
            return json.load(f)
    return {}


def _load_check_decision(directory: str) -> dict:
    for name in ("check_object_mask.json", "decision.json"):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return {}


def _extract_chamfer_stats_mm(metrics: dict, label: str) -> tuple[float, float]:
    """Extract (median_mm, mean_mm) from chamfer metrics JSON."""
    combined = metrics.get("combined")
    if not isinstance(combined, dict):
        raise ValueError(f"{label} metrics missing 'combined' object")

    median_mm = combined.get("median_mm")
    mean_mm = combined.get("mean_mm")
    if not isinstance(median_mm, (int, float)):
        raise ValueError(f"{label} metrics missing numeric 'combined.median_mm'")
    if not isinstance(mean_mm, (int, float)):
        raise ValueError(f"{label} metrics missing numeric 'combined.mean_mm'")

    return float(median_mm), float(mean_mm)


def check_accuracy(
    chamfer_object_dir: str,
    chamfer_human_dir: str,
    check_object_mask_dir: str,
    output_dir: str,
    max_chamfer_object: float = 30.0,
    max_chamfer_human: float = 30.0,
) -> dict:
    obj_metrics = _find_metrics(chamfer_object_dir)
    human_metrics = _find_metrics(chamfer_human_dir)
    mask_decision = _load_check_decision(check_object_mask_dir)

    obj_chamfer_median, obj_chamfer_mean = _extract_chamfer_stats_mm(obj_metrics, "object")
    human_chamfer_median, human_chamfer_mean = _extract_chamfer_stats_mm(human_metrics, "human")
    obj_mask_containment = mask_decision.get("avg_containment", 0.0)
    obj_mask_status = mask_decision.get("status", "PASS")

    checks: dict[str, str] = {}
    reasons: list[str] = []

    if obj_chamfer_median > max_chamfer_object:
        checks["chamfer_object"] = "FAIL"
        reasons.append(
            f"object chamfer {obj_chamfer_median:.1f} > threshold {max_chamfer_object:.1f}"
        )
    else:
        checks["chamfer_object"] = "PASS"

    if human_chamfer_median > max_chamfer_human:
        checks["chamfer_human"] = "FAIL"
        reasons.append(
            f"human chamfer {human_chamfer_median:.1f} > threshold {max_chamfer_human:.1f}"
        )
    else:
        checks["chamfer_human"] = "PASS"

    checks["object_mask_containment"] = obj_mask_status
    if obj_mask_status == "FAIL":
        reasons.append(mask_decision.get("reason", "object mask containment below threshold"))

    status = "FAIL" if reasons else "PASS"

    decision: dict = {
        "status": status,
        "reason": "; ".join(reasons),
        "object_chamfer_median": obj_chamfer_median,
        "object_chamfer_mean": obj_chamfer_mean,
        "human_chamfer_median": human_chamfer_median,
        "human_chamfer_mean": human_chamfer_mean,
        "chamfer_metric": "combined.median_mm",
        "object_mask_containment": obj_mask_containment,
        "thresholds": {
            "max_chamfer_object": max_chamfer_object,
            "max_chamfer_human": max_chamfer_human,
            "min_object_mask_containment": mask_decision.get("threshold", 0.8),
        },
        "checks": checks,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "check_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(decision, f, indent=2)

    print(f"Accuracy: {status}")
    print(json.dumps(decision, indent=2))

    if status == "FAIL":
        sys.exit(1)

    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate accuracy checks: chamfer + mask containment"
    )
    parser.add_argument("--chamfer_object_dir", type=str, required=True)
    parser.add_argument("--chamfer_human_dir", type=str, required=True)
    parser.add_argument("--check_object_mask_dir", type=str, required=True,
                        help="check_object_mask output (contains check_object_mask.json)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_chamfer_object", type=float, default=30.0)
    parser.add_argument("--max_chamfer_human", type=float, default=30.0)
    args = parser.parse_args()

    check_accuracy(
        chamfer_object_dir=args.chamfer_object_dir,
        chamfer_human_dir=args.chamfer_human_dir,
        check_object_mask_dir=args.check_object_mask_dir,
        output_dir=args.output_dir,
        max_chamfer_object=args.max_chamfer_object,
        max_chamfer_human=args.max_chamfer_human,
    )
