# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build source-frame accuracy failure segments and enforce sequence limits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Callable


SILHOUETTE_CATEGORY = "Silhouette bounding box containment"
CHAMFER_CATEGORY = "Chamfer distance"
SUBJECTS = ("Human", "Object")
METRICS = (SILHOUETTE_CATEGORY, CHAMFER_CATEGORY)


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{label} metrics missing {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} metrics must be a JSON object")
    return payload


def _metric_path(directory: str, names: tuple[str, ...], label: str) -> Path:
    root = Path(directory)
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"{label} metrics missing under {root}")


def _load_mask_diagnostic(directory: str) -> dict:
    path = _metric_path(
        directory,
        ("check_object_mask.json", "decision.json"),
        "object mask diagnostic",
    )
    payload = _load_json(path, "object mask diagnostic")
    status = payload.get("status")
    if status not in {"PASS", "FAIL", "INCONCLUSIVE"}:
        raise ValueError(
            "object mask diagnostic status must be PASS, FAIL, or INCONCLUSIVE"
        )
    containment = payload.get("avg_containment")
    if containment is not None and (
        not isinstance(containment, (int, float)) or isinstance(containment, bool)
    ):
        raise ValueError("object mask diagnostic avg_containment must be numeric or null")
    return payload


def _load_silhouette_frames(
    directory: str,
    label: str,
) -> tuple[dict[str, list[tuple[int, float | None]]], int]:
    path = _metric_path(
        directory, ("silhouette_mask_metrics.json", "metrics.json"), label
    )
    payload = _load_json(path, label)
    frame_metrics = payload.get("frame_metrics")
    if not isinstance(frame_metrics, dict) or not frame_metrics:
        raise ValueError(f"{label} metrics missing nonempty frame_metrics object")

    per_camera: dict[str, list[tuple[int, float | None]]] = {}
    frame_counts: set[int] = set()
    for camera, rows in frame_metrics.items():
        if not isinstance(camera, str) or not isinstance(rows, list):
            raise ValueError(f"{label} frame_metrics entries must be camera lists")
        parsed: list[tuple[int, float | None]] = []
        for expected_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"{label} {camera} frame metric must be an object")
            frame_index = row.get("frame_idx")
            if (
                not isinstance(frame_index, int)
                or isinstance(frame_index, bool)
                or frame_index != expected_index
            ):
                raise ValueError(
                    f"{label} {camera} frame indices must be contiguous source indices"
                )
            if row.get("skipped", False):
                value = None
            else:
                value = row.get(
                    "filtered_sam2_bbox_in_padded_render_bbox_ratio"
                )
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(
                        f"{label} {camera} frame {frame_index} lacks bbox containment"
                    )
                value = float(value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"{label} {camera} frame {frame_index} containment is outside [0,1]"
                    )
            parsed.append((frame_index, value))
        frame_counts.add(len(parsed))
        per_camera[camera] = parsed
    if len(frame_counts) != 1:
        raise ValueError(f"{label} camera timelines differ in length")
    total_frames = next(iter(frame_counts))
    if total_frames <= 0:
        raise ValueError(f"{label} contains no source frames")
    return per_camera, total_frames


def _load_chamfer_frames(
    directory: str,
    label: str,
    total_frames: int,
) -> tuple[dict[str, list[tuple[int, float | None]]], float]:
    path = _metric_path(directory, ("chamfer_metrics.json", "metrics.json"), label)
    payload = _load_json(path, label)
    per_camera_payload = payload.get("per_camera")
    if not isinstance(per_camera_payload, dict) or not per_camera_payload:
        raise ValueError(f"{label} metrics missing nonempty per_camera object")

    per_camera: dict[str, list[tuple[int, float | None]]] = {}
    pooled: list[float] = []
    for camera, camera_payload in per_camera_payload.items():
        if not isinstance(camera, str) or not isinstance(camera_payload, dict):
            raise ValueError(f"{label} per_camera entries must be objects")
        rows = camera_payload.get("per_frame")
        if not isinstance(rows, list):
            raise ValueError(
                f"{label} {camera} missing source-indexed per_frame measurements"
            )
        parsed: list[tuple[int, float | None]] = []
        prior = -1
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{label} {camera} per_frame entries must be objects")
            frame_index = row.get("frame_index")
            distance = row.get("distance_mm")
            if (
                not isinstance(frame_index, int)
                or isinstance(frame_index, bool)
                or frame_index <= prior
                or not 0 <= frame_index < total_frames
            ):
                raise ValueError(
                    f"{label} {camera} frame indices must be strictly increasing source indices"
                )
            if (
                not isinstance(distance, (int, float))
                or isinstance(distance, bool)
                or not math.isfinite(float(distance))
                or float(distance) < 0.0
            ):
                raise ValueError(
                    f"{label} {camera} frame {frame_index} distance must be nonnegative"
                )
            parsed.append((frame_index, float(distance)))
            pooled.append(float(distance))
            prior = frame_index
        per_camera[camera] = parsed
    if not pooled:
        raise ValueError(f"{label} metrics contain no eligible camera frames")

    combined = payload.get("combined")
    if not isinstance(combined, dict):
        raise ValueError(f"{label} metrics missing 'combined' object")
    observed_median = combined.get("median_mm")
    observed_mean = combined.get("mean_mm")
    for name, value in (
        ("median_mm", observed_median), ("mean_mm", observed_mean),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{label} combined.{name} must be finite numeric")
    pooled_median = float(statistics.median(pooled))
    pooled_mean = float(statistics.fmean(pooled))
    if not math.isclose(
        float(observed_median), pooled_median, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError(f"{label} combined.median_mm is inconsistent")
    if not math.isclose(
        float(observed_mean), pooled_mean, rel_tol=0.0, abs_tol=1e-9,
    ):
        raise ValueError(f"{label} combined.mean_mm is inconsistent")
    return per_camera, pooled_median


def _qualifying_runs(
    rows: list[tuple[int, float | None]],
    *,
    fails: Callable[[float], bool],
    minimum_frames: int,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None

    def finish() -> None:
        nonlocal start, previous
        if start is not None and previous is not None:
            end = previous + 1
            if end - start >= minimum_frames:
                runs.append((start, end))
        start = None
        previous = None

    for frame_index, value in rows:
        failing = value is not None and fails(value)
        if not failing:
            finish()
            continue
        if previous is None or frame_index != previous + 1:
            finish()
            start = frame_index
        previous = frame_index
    finish()
    return runs


def _union_view_runs(
    per_camera: dict[str, list[tuple[int, int]]],
) -> list[tuple[int, int, list[str]]]:
    pending = sorted(
        (start, end, camera)
        for camera, runs in per_camera.items()
        for start, end in runs
    )
    merged: list[list[object]] = []
    for start, end, camera in pending:
        if not merged or start > int(merged[-1][1]):
            merged.append([start, end, {camera}])
            continue
        merged[-1][1] = max(int(merged[-1][1]), end)
        cameras = merged[-1][2]
        assert isinstance(cameras, set)
        cameras.add(camera)
    return [
        (int(start), int(end), sorted(cameras))
        for start, end, cameras in merged
    ]


def _segments_for_metric(
    *,
    subject: str,
    category: str,
    per_camera_rows: dict[str, list[tuple[int, float | None]]],
    fails: Callable[[float], bool],
    minimum_frames: int,
) -> tuple[list[dict], dict]:
    view_runs = {
        camera: _qualifying_runs(
            rows, fails=fails, minimum_frames=minimum_frames
        )
        for camera, rows in sorted(per_camera_rows.items())
    }
    view_runs = {camera: runs for camera, runs in view_runs.items() if runs}
    merged = _union_view_runs(view_runs)
    slug = (
        f"{subject}-{category}".lower().replace(" ", "-")
        .replace("bounding-box", "bbox")
    )
    segments = [
        {
            "id": f"accuracy-{slug}-{index:03d}",
            "start_frame": start,
            "end_frame": end,
            "failure_category": category,
            "reason": f"{subject} {category}: failed in views [{', '.join(cameras)}]",
        }
        for index, (start, end, cameras) in enumerate(merged, start=1)
    ]
    eligible = sum(
        value is not None for rows in per_camera_rows.values() for _, value in rows
    )
    return segments, {
        "status": "SEGMENTED" if segments else "PASS" if eligible else "INCONCLUSIVE",
        "eligible_camera_frames": eligible,
        "qualifying_view_runs": sum(len(runs) for runs in view_runs.values()),
        "failure_segments": len(segments),
        "per_camera_runs": {
            camera: [
                {"start_frame": start, "end_frame": end}
                for start, end in runs
            ]
            for camera, runs in view_runs.items()
        },
    }


def merged_interval_coverage(segments: list[dict]) -> int:
    intervals = sorted(
        (int(segment["start_frame"]), int(segment["end_frame"]))
        for segment in segments
    )
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def check_accuracy(
    chamfer_object_dir: str,
    chamfer_human_dir: str,
    check_object_mask_dir: str,
    silhouette_object_dir: str,
    output_dir: str,
    silhouette_human_dir: str | None = None,
    max_chamfer_object: float = 40.0,
    max_chamfer_human: float = 40.0,
    max_chamfer_segment_object: float = 50.0,
    max_chamfer_segment_human: float = 50.0,
    min_silhouette_bbox_containment: float = 0.8,
    min_failure_run_frames: int = 5,
    max_silhouette_failure_coverage: float = 0.5,
) -> dict:
    if silhouette_human_dir is None:
        raise ValueError("silhouette_human_dir is required for segment accuracy QC")
    if min_failure_run_frames < 1:
        raise ValueError("min_failure_run_frames must be positive")
    for name, value in (
        ("max_chamfer_object", max_chamfer_object),
        ("max_chamfer_human", max_chamfer_human),
        ("max_chamfer_segment_object", max_chamfer_segment_object),
        ("max_chamfer_segment_human", max_chamfer_segment_human),
    ):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.0 <= max_silhouette_failure_coverage <= 1.0:
        raise ValueError("max_silhouette_failure_coverage must be in [0,1]")
    if not 0.0 <= min_silhouette_bbox_containment <= 1.0:
        raise ValueError("min_silhouette_bbox_containment must be in [0,1]")

    silhouette_rows: dict[str, dict[str, list[tuple[int, float | None]]]] = {}
    frame_counts: dict[str, int] = {}
    for subject, directory in (
        ("Human", silhouette_human_dir),
        ("Object", silhouette_object_dir),
    ):
        rows, frame_count = _load_silhouette_frames(
            directory, f"{subject.lower()} silhouette"
        )
        silhouette_rows[subject] = rows
        frame_counts[subject] = frame_count
    if len(set(frame_counts.values())) != 1:
        raise ValueError(
            f"human and object silhouette timelines differ: {frame_counts}"
        )
    total_frames = next(iter(frame_counts.values()))

    human_chamfer_rows, human_chamfer_median = _load_chamfer_frames(
        chamfer_human_dir, "human chamfer", total_frames
    )
    object_chamfer_rows, object_chamfer_median = _load_chamfer_frames(
        chamfer_object_dir, "object chamfer", total_frames
    )
    chamfer_rows = {
        "Human": human_chamfer_rows,
        "Object": object_chamfer_rows,
    }
    chamfer_medians = {
        "Human": human_chamfer_median,
        "Object": object_chamfer_median,
    }
    mask_diagnostic = _load_mask_diagnostic(check_object_mask_dir)

    all_segments: list[dict] = []
    metric_details: dict[str, dict] = {}
    failed_accuracy_checks: list[str] = []
    failure_reasons: list[str] = []
    for subject in SUBJECTS:
        chamfer_gate_threshold = (
            max_chamfer_human if subject == "Human" else max_chamfer_object
        )
        chamfer_segment_threshold = (
            max_chamfer_segment_human
            if subject == "Human" else max_chamfer_segment_object
        )
        definitions = (
            (
                SILHOUETTE_CATEGORY,
                silhouette_rows[subject],
                lambda value, threshold=min_silhouette_bbox_containment: value
                < threshold,
            ),
            (
                CHAMFER_CATEGORY,
                chamfer_rows[subject],
                lambda value, threshold=chamfer_segment_threshold: value
                > threshold,
            ),
        )
        for category, rows, predicate in definitions:
            segments, details = _segments_for_metric(
                subject=subject,
                category=category,
                per_camera_rows=rows,
                fails=predicate,
                minimum_frames=min_failure_run_frames,
            )
            covered_frames = merged_interval_coverage(segments)
            coverage = covered_frames / total_frames
            details.update({
                "failure_coverage_frames": covered_frames,
                "failure_coverage": coverage,
            })
            if category == SILHOUETTE_CATEGORY:
                gate_failed = coverage > max_silhouette_failure_coverage
                check_name = (
                    f"{subject.lower()}_silhouette_bbox_containment"
                )
                details.update({
                    "gate_metric": "failure_coverage",
                    "gate_threshold": float(max_silhouette_failure_coverage),
                    "gate_status": "FAIL" if gate_failed else "PASS",
                })
                if gate_failed:
                    failed_accuracy_checks.append(check_name)
                    failure_reasons.append(
                        f"{subject} silhouette bbox containment coverage "
                        f"{coverage:.6f} > "
                        f"{max_silhouette_failure_coverage:.6f}"
                    )
            else:
                pooled_median = chamfer_medians[subject]
                gate_failed = pooled_median > chamfer_gate_threshold
                check_name = f"chamfer_{subject.lower()}"
                details.update({
                    "pooled_median_mm": pooled_median,
                    "gate_metric": "pooled_median_mm",
                    "gate_threshold_mm": float(chamfer_gate_threshold),
                    "segment_threshold_mm": float(chamfer_segment_threshold),
                    "gate_status": "FAIL" if gate_failed else "PASS",
                })
                if gate_failed:
                    failed_accuracy_checks.append(check_name)
                    failure_reasons.append(
                        f"{subject} Chamfer pooled median "
                        f"{pooled_median:.6f} mm > "
                        f"{chamfer_gate_threshold:.6f} mm"
                    )
            all_segments.extend(segments)
            metric_details[f"{subject.lower()}_{category.lower().replace(' ', '_')}"] = details

    all_segments.sort(
        key=lambda item: (
            int(item["start_frame"]),
            int(item["end_frame"]),
            str(item["reason"]),
        )
    )
    covered_frames = merged_interval_coverage(all_segments)
    coverage = covered_frames / total_frames
    status = "FAIL" if failed_accuracy_checks else "PASS"

    checks = {
        key: value["gate_status"] for key, value in metric_details.items()
    }
    checks["object_mask_containment_diagnostic"] = mask_diagnostic["status"]
    checks["accuracy_failure_limits"] = status
    decision = {
        "schema": "v2d.mv_hoi.check_accuracy.v3",
        "status": status,
        "reason": "; ".join(failure_reasons),
        "failed_accuracy_checks": failed_accuracy_checks,
        "total_frames": total_frames,
        "failure_segment_count": len(all_segments),
        "failure_coverage_frames": covered_frames,
        "failure_coverage": coverage,
        "failure_segments": all_segments,
        "metric_details": metric_details,
        "object_mask_containment_diagnostic": {
            "diagnostic_only": True,
            **mask_diagnostic,
        },
        "thresholds": {
            "max_chamfer_object_mm": float(max_chamfer_object),
            "max_chamfer_human_mm": float(max_chamfer_human),
            "max_chamfer_segment_object_mm": float(
                max_chamfer_segment_object
            ),
            "max_chamfer_segment_human_mm": float(
                max_chamfer_segment_human
            ),
            "min_silhouette_bbox_containment": float(
                min_silhouette_bbox_containment
            ),
            "min_failure_run_frames": int(min_failure_run_frames),
            "max_silhouette_failure_coverage": float(
                max_silhouette_failure_coverage
            ),
        },
        "checks": checks,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "check_accuracy.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    (output / "failure_segments.json").write_text(
        json.dumps(all_segments, indent=2, sort_keys=True) + "\n"
    )
    print(f"Accuracy: {status}")
    print(json.dumps(decision, indent=2, sort_keys=True))
    if status == "FAIL":
        sys.exit(1)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chamfer_object_dir", required=True)
    parser.add_argument("--chamfer_human_dir", required=True)
    parser.add_argument("--check_object_mask_dir", required=True)
    parser.add_argument("--silhouette_object_dir", required=True)
    parser.add_argument("--silhouette_human_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_chamfer_object", type=float, default=40.0)
    parser.add_argument("--max_chamfer_human", type=float, default=40.0)
    parser.add_argument(
        "--max_chamfer_segment_object", type=float, default=50.0
    )
    parser.add_argument(
        "--max_chamfer_segment_human", type=float, default=50.0
    )
    parser.add_argument(
        "--min_silhouette_bbox_containment", type=float, default=0.8
    )
    parser.add_argument("--min_failure_run_frames", type=int, default=5)
    parser.add_argument(
        "--max_silhouette_failure_coverage", type=float, default=0.5
    )
    args = parser.parse_args()
    check_accuracy(**vars(args))


if __name__ == "__main__":
    main()
