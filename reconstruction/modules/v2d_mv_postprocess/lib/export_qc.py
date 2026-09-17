"""Early post-trim QC decisions and compact rejection evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .interaction_trim import merged_interval_coverage, trim_failure_segments
except ImportError:  # Direct test/script execution from the lib directory.
    from interaction_trim import merged_interval_coverage, trim_failure_segments


REJECTION_SCHEMA = "v2d.mv_hoi.export_rejection.v1"
CONTAINMENT_CATEGORY = "Silhouette bounding box containment"


class ExportQCRejected(ValueError):
    """The retained export interval exceeds a configured QC gate."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        failed = [gate["name"] for gate in report["gates"] if gate["status"] == "FAIL"]
        super().__init__("post-trim QC rejected export: " + ", ".join(failed))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _subject_containment_segments(segments: list[dict], subject: str) -> list[dict]:
    prefix = subject.lower() + " "
    return [
        segment for segment in segments
        if segment.get("failure_category") == CONTAINMENT_CATEGORY
        and str(segment.get("reason") or "").lower().startswith(prefix)
    ]


def _coverage_gate(
    name: str, segments: list[dict], frame_count: int, threshold: float,
) -> dict:
    covered = merged_interval_coverage(segments)
    coverage = covered / frame_count
    return {
        "name": name,
        "metric": "retained_frame_coverage",
        "observed": coverage,
        "covered_frames": covered,
        "threshold": float(threshold),
        "comparison": ">",
        "status": "FAIL" if coverage > float(threshold) else "PASS",
    }


def evaluate_post_trim_qc(
    *, trim_manifest: dict, human_segments: list[dict], metric_segments: list[dict],
    max_failure_annotations: int, max_failure_coverage: float,
    max_silhouette_failure_coverage: float,
) -> dict:
    """Clip evidence, evaluate independent retained-interval gates, and summarize."""
    start = int(trim_manifest["export_source_start_frame"])
    end = int(trim_manifest["export_source_end_frame"])
    frame_count = int(trim_manifest["export_frame_count"])
    if frame_count <= 0 or end - start != frame_count:
        raise ValueError("Interaction trim manifest has an invalid retained interval")
    if max_failure_annotations < 0:
        raise ValueError("max_failure_annotations must be nonnegative")
    for value, name in (
        (max_failure_coverage, "max_failure_coverage"),
        (max_silhouette_failure_coverage, "max_silhouette_failure_coverage"),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    clipped_human = trim_failure_segments(
        human_segments, start_frame=start, end_frame=end,
    )
    clipped_metric = trim_failure_segments(
        metric_segments, start_frame=start, end_frame=end,
    )
    human_count = len(clipped_human)
    gates = [{
        "name": "human_qc_failure_count",
        "metric": "segment_count",
        "observed": human_count,
        "threshold": int(max_failure_annotations),
        "comparison": ">",
        "status": (
            "FAIL" if human_count > int(max_failure_annotations) else "PASS"
        ),
    }]
    gates.append(_coverage_gate(
        "human_qc_failure_coverage", clipped_human, frame_count,
        max_failure_coverage,
    ))
    containment: dict[str, list[dict]] = {}
    for subject in ("Human", "Object"):
        subject_segments = _subject_containment_segments(clipped_metric, subject)
        containment[subject.lower()] = subject_segments
        gates.append(_coverage_gate(
            f"{subject.lower()}_silhouette_bbox_containment_coverage",
            subject_segments, frame_count, max_silhouette_failure_coverage,
        ))
    return {
        "status": "REJECTED" if any(gate["status"] == "FAIL" for gate in gates) else "PASS",
        "trimmed_human_segments": clipped_human,
        "trimmed_metric_segments": clipped_metric,
        "trimmed_failure_segments": clipped_human + clipped_metric,
        "containment_segments": containment,
        "gates": gates,
    }


def build_rejection_report(
    *, decision: dict, trim_manifest: dict, human_segments: list[dict],
    metric_segments: list[dict], provenance: dict,
) -> dict:
    if decision.get("status") != "REJECTED":
        raise ValueError("A rejection report requires a rejected QC decision")
    report = {
        "schema": REJECTION_SCHEMA,
        "status": "REJECTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "post_trim_qc_limit",
        "trim": {
            "schema": trim_manifest.get("schema"),
            "decision_sha256": trim_manifest.get("decision_sha256"),
            "source_frame_count": int(trim_manifest["source_frame_count"]),
            "source_start_frame": int(trim_manifest["export_source_start_frame"]),
            "source_end_frame": int(trim_manifest["export_source_end_frame"]),
            "export_frame_count": int(trim_manifest["export_frame_count"]),
        },
        "evidence": {
            "human_source_sha256": _sha256_json(human_segments),
            "metric_source_sha256": _sha256_json(metric_segments),
            "human_segments": decision["trimmed_human_segments"],
            "metric_segments": decision["trimmed_metric_segments"],
        },
        "gates": decision["gates"],
        "provenance": provenance,
    }
    report["report_sha256"] = _sha256_json(report)
    return report


def validate_rejection_report(report: dict) -> dict:
    if report.get("schema") != REJECTION_SCHEMA or report.get("status") != "REJECTED":
        raise ValueError("Unsupported export rejection report")
    required = {"trim", "evidence", "gates", "provenance", "report_sha256"}
    if not required.issubset(report):
        raise ValueError("Export rejection report is incomplete")
    expected = dict(report)
    digest = expected.pop("report_sha256")
    if digest != _sha256_json(expected):
        raise ValueError("Export rejection report hash is inconsistent")
    trim = report["trim"]
    start, end, count = (
        int(trim["source_start_frame"]), int(trim["source_end_frame"]),
        int(trim["export_frame_count"]),
    )
    if start < 0 or end <= start or end - start != count:
        raise ValueError("Export rejection trim interval is inconsistent")
    gates = report["gates"]
    if not isinstance(gates, list) or not gates or not any(
        gate.get("status") == "FAIL" for gate in gates
    ):
        raise ValueError("Export rejection report has no failed gate")
    for gate in gates:
        if gate.get("comparison") != ">" or gate.get("status") not in {"PASS", "FAIL"}:
            raise ValueError("Export rejection gate is malformed")
        expected_status = (
            "FAIL" if float(gate["observed"]) > float(gate["threshold"]) else "PASS"
        )
        if gate["status"] != expected_status:
            raise ValueError("Export rejection gate decision is inconsistent")
    return report


def write_rejection_report(path: str | Path, report: dict) -> None:
    validate_rejection_report(report)
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
