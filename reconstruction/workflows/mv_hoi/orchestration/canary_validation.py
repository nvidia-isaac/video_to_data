"""Validate persisted accuracy and pose-comparison canary evidence."""

from __future__ import annotations

import math
from typing import Any


def _merged_coverage(segments: list[dict[str, Any]]) -> int:
    intervals = sorted(
        (int(segment["start_frame"]), int(segment["end_frame"]))
        for segment in segments
    )
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _validate_accuracy_failure_evidence_v3(
    report: dict[str, Any],
    segments: object,
    *,
    expected_thresholds: dict[str, float | int],
) -> tuple[bool, str]:
    if report.get("status") != "FAIL":
        return False, "check_accuracy evidence is not a FAIL decision"
    report_segments = report.get("failure_segments")
    if not isinstance(report_segments, list) or report_segments != segments:
        return False, "failure_segments artifact differs from check_accuracy report"
    total_frames = report.get("total_frames")
    if not isinstance(total_frames, int) or total_frames <= 0:
        return False, "invalid total frame count"
    sort_keys: list[tuple[int, int, str]] = []
    for segment in report_segments:
        if not isinstance(segment, dict):
            return False, "failure segment is not an object"
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        category = segment.get("failure_category")
        reason = segment.get("reason")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > total_frames
        ):
            return False, "failure segment is not a valid half-open interval"
        if category not in {
            "Silhouette bounding box containment", "Chamfer distance",
        }:
            return False, "unexpected automatic failure category"
        if not isinstance(reason, str) or not reason.startswith(("Human ", "Object ")):
            return False, "unexpected automatic failure reason"
        sort_keys.append((start, end, reason))
    if sort_keys != sorted(sort_keys):
        return False, "failure segments are not deterministically ordered"

    covered = _merged_coverage(report_segments)
    coverage = covered / total_frames
    if report.get("failure_segment_count") != len(report_segments):
        return False, "failure segment count does not regenerate"
    if report.get("failure_coverage_frames") != covered:
        return False, "failure coverage frames do not regenerate"
    if not isinstance(report.get("failure_coverage"), (int, float)) or not math.isclose(
        float(report["failure_coverage"]), coverage,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        return False, "failure coverage does not regenerate"

    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict):
        return False, "accuracy thresholds are missing"
    for key, expected in expected_thresholds.items():
        observed = thresholds.get(key)
        if not isinstance(observed, (int, float)) or float(observed) != float(expected):
            return False, f"accuracy threshold mismatch: {key}"

    details = report.get("metric_details")
    if not isinstance(details, dict):
        return False, "metric details are missing"
    failures: list[str] = []
    reasons: list[str] = []
    definitions = (
        (
            "Human", "Silhouette bounding box containment",
            "human_silhouette_bounding_box_containment",
            "human_silhouette_bbox_containment",
        ),
        ("Human", "Chamfer distance", "human_chamfer_distance", "chamfer_human"),
        (
            "Object", "Silhouette bounding box containment",
            "object_silhouette_bounding_box_containment",
            "object_silhouette_bbox_containment",
        ),
        ("Object", "Chamfer distance", "object_chamfer_distance", "chamfer_object"),
    )
    for subject, category, detail_key, check_name in definitions:
        metric = details.get(detail_key)
        if not isinstance(metric, dict):
            return False, f"missing metric details: {detail_key}"
        metric_segments = [
            segment for segment in report_segments
            if segment.get("failure_category") == category
            and str(segment.get("reason") or "").startswith(f"{subject} ")
        ]
        metric_covered = _merged_coverage(metric_segments)
        metric_coverage = metric_covered / total_frames
        if metric.get("failure_segments") != len(metric_segments):
            return False, f"metric segment count does not regenerate: {detail_key}"
        if metric.get("failure_coverage_frames") != metric_covered:
            return False, f"metric coverage frames do not regenerate: {detail_key}"
        observed_coverage = metric.get("failure_coverage")
        if not isinstance(observed_coverage, (int, float)) or not math.isclose(
            float(observed_coverage), metric_coverage,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            return False, f"metric coverage does not regenerate: {detail_key}"

        if category == "Chamfer distance":
            median = metric.get("pooled_median_mm")
            limit_key = (
                "max_chamfer_human_mm" if subject == "Human"
                else "max_chamfer_object_mm"
            )
            segment_key = (
                "max_chamfer_segment_human_mm" if subject == "Human"
                else "max_chamfer_segment_object_mm"
            )
            if not isinstance(median, (int, float)) or not math.isfinite(float(median)):
                return False, f"invalid pooled median: {detail_key}"
            limit = float(expected_thresholds[limit_key])
            gate_failed = float(median) > limit
            if float(metric.get("gate_threshold_mm", -1)) != limit:
                return False, f"Chamfer gate threshold mismatch: {detail_key}"
            if float(metric.get("segment_threshold_mm", -1)) != float(
                expected_thresholds[segment_key]
            ):
                return False, f"Chamfer segment threshold mismatch: {detail_key}"
            if gate_failed:
                failures.append(check_name)
                reasons.append(
                    f"{subject} Chamfer pooled median {float(median):.6f} mm > "
                    f"{limit:.6f} mm"
                )
        else:
            limit = float(
                expected_thresholds["max_silhouette_failure_coverage"]
            )
            gate_failed = metric_coverage > limit
            if float(metric.get("gate_threshold", -1)) != limit:
                return False, f"silhouette gate threshold mismatch: {detail_key}"
            if gate_failed:
                failures.append(check_name)
                reasons.append(
                    f"{subject} silhouette bbox containment coverage "
                    f"{metric_coverage:.6f} > {limit:.6f}"
                )
        expected_status = "FAIL" if gate_failed else "PASS"
        if metric.get("gate_status") != expected_status:
            return False, f"metric gate status does not regenerate: {detail_key}"

    if report.get("failed_accuracy_checks") != failures:
        return False, "failed accuracy checks do not regenerate"
    if not failures:
        return False, "FAIL decision has no failed accuracy checks"
    if report.get("reason") != "; ".join(reasons):
        return False, "FAIL reason does not regenerate"
    return True, "independent metric decisions and segments regenerated"


def validate_accuracy_failure_evidence(
    report: dict[str, Any], segments: object, *,
    expected_thresholds: dict[str, float | int],
) -> tuple[bool, str]:
    """Validate a segment-limit failure independently from its exit status."""
    if report.get("schema") == "v2d.mv_hoi.check_accuracy.v3":
        return _validate_accuracy_failure_evidence_v3(
            report, segments, expected_thresholds=expected_thresholds,
        )
    if report.get("schema") != "v2d.mv_hoi.check_accuracy.v2":
        return False, "unexpected check_accuracy schema"
    if report.get("status") != "FAIL":
        return False, "check_accuracy evidence is not a FAIL decision"
    report_segments = report.get("failure_segments")
    if not isinstance(report_segments, list) or report_segments != segments:
        return False, "failure_segments artifact differs from check_accuracy report"
    total_frames = report.get("total_frames")
    if not isinstance(total_frames, int) or total_frames <= 0:
        return False, "invalid total frame count"
    normalized: list[tuple[int, int]] = []
    sort_keys: list[tuple[int, int, str]] = []
    for segment in report_segments:
        if not isinstance(segment, dict):
            return False, "failure segment is not an object"
        start = segment.get("start_frame")
        end = segment.get("end_frame")
        category = segment.get("failure_category")
        reason = segment.get("reason")
        if (
            not isinstance(start, int) or not isinstance(end, int)
            or start < 0 or end <= start or end > total_frames
        ):
            return False, "failure segment is not a valid half-open interval"
        if category not in {
            "Silhouette bounding box containment", "Chamfer distance",
        }:
            return False, "unexpected automatic failure category"
        if not isinstance(reason, str) or not reason.startswith(("Human ", "Object ")):
            return False, "unexpected automatic failure reason"
        normalized.append((start, end))
        sort_keys.append((start, end, reason))
    if sort_keys != sorted(sort_keys):
        return False, "failure segments are not deterministically ordered"
    merged: list[list[int]] = []
    for start, end in sorted(normalized):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(end - start for start, end in merged)
    coverage = covered / total_frames
    if report.get("failure_segment_count") != len(report_segments):
        return False, "failure segment count does not regenerate"
    if report.get("failure_coverage_frames") != covered:
        return False, "failure coverage frames do not regenerate"
    observed_coverage = report.get("failure_coverage")
    if not isinstance(observed_coverage, (int, float)) or not math.isclose(
        float(observed_coverage), coverage, rel_tol=0.0, abs_tol=1e-12,
    ):
        return False, "failure coverage does not regenerate"
    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict):
        return False, "accuracy thresholds are missing"
    for key, expected in expected_thresholds.items():
        observed = thresholds.get(key)
        if not isinstance(observed, (int, float)) or float(observed) != float(expected):
            return False, f"accuracy threshold mismatch: {key}"
    count_limit = int(expected_thresholds["max_failure_segments"])
    coverage_limit = float(expected_thresholds["max_failure_coverage"])
    count_exceeded = len(report_segments) > count_limit
    coverage_exceeded = coverage > coverage_limit
    if not (count_exceeded or coverage_exceeded):
        return False, "FAIL decision does not exceed a configured limit"
    reasons = []
    if count_exceeded:
        reasons.append(f"accuracy failure segments {len(report_segments)} > {count_limit}")
    if coverage_exceeded:
        reasons.append(
            f"accuracy failure coverage {coverage:.6f} > {coverage_limit:.6f}"
        )
    if report.get("reason") != "; ".join(reasons):
        return False, "FAIL reason does not regenerate"
    return True, "segment decision and limits regenerated"


def validate_foundation_comparison_evidence(
    report: dict[str, Any],
) -> tuple[bool, str]:
    if report.get("schema") != "v2d.mv_hoi.foundation_pose_comparison.v2":
        return False, "unexpected foundation comparison schema"
    if report.get("status") != "FAIL":
        return False, "foundation comparison evidence is not a FAIL decision"
    failures = report.get("failures")
    if not isinstance(failures, list) or not failures or not all(
        isinstance(item, str) and item for item in failures
    ):
        return False, "foundation comparison failure list is invalid"
    digest = report.get("tolerances_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False, "foundation comparison tolerance provenance is invalid"
    return True, "foundation comparison failure evidence validated"
