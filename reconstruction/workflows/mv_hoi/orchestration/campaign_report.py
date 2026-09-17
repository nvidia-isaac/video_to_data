"""Generate a reconciled JSON/CSV report for one processing campaign."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

try:
    from . import db
    from .runtime import state_path
except ImportError:
    import db
    from runtime import state_path


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    pick = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {"p50": pick(0.5), "p95": pick(0.95), "max": ordered[-1]}


def build_report(campaign_name: str, *, db_path: str) -> tuple[dict, list[dict]]:
    campaign = db.campaign_progress(campaign_name, db_path=db_path)
    requests = db.list_stage_requests(campaign=campaign_name, db_path=db_path)
    rows = []
    for request in requests:
        summary = json.loads(request["result_summary_json"] or "{}")
        comparison = summary.get("pose_comparison", {})
        rows.append({
            "sequence": request["sequence_name"], "request_id": request["id"],
            "fulfilled_by_request_id": request["fulfilled_by_request_id"],
            "stage": request["stage"], "cohort": request["cohort"],
            "queue_priority": request.get("queue_priority", 0),
            "status": request["status"], "blocked_reason": request["blocked_reason"],
            "details": request["details"],
            "failure_category": summary.get("failure_category"),
            "failed_accuracy_checks": summary.get("failed_accuracy_checks"),
            "result_manifest_uri": request["result_manifest_uri"],
            "file_count": summary.get("file_count"),
            "total_bytes": summary.get("total_bytes"),
            "valid_coverage": comparison.get("valid_coverage"),
            "newly_invalid_fraction": comparison.get("newly_invalid_fraction"),
            "divergent_frame_fraction": comparison.get(
                "divergent_frame_fraction"
            ),
            "translation_p95_m": (comparison.get("translation_m") or {}).get("p95"),
            "rotation_p95_deg": (comparison.get("rotation_deg") or {}).get("p95"),
            "normalized_adds_p95": (comparison.get("normalized_adds") or {}).get("p95"),
        })
    unique_sequences = {row["sequence"] for row in rows}
    metrics = {}
    for key in (
        "valid_coverage", "newly_invalid_fraction", "divergent_frame_fraction",
        "translation_p95_m", "rotation_p95_deg", "normalized_adds_p95",
    ):
        metrics[key] = _percentiles([
            float(row[key]) for row in rows if row[key] is not None
        ])
    status_counts: dict[str, int] = {}
    accuracy_failure_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        for check in row.get("failed_accuracy_checks") or []:
            accuracy_failure_counts[check] = accuracy_failure_counts.get(check, 0) + 1
    report = {
        "schema": "v2d.mv_hoi.campaign_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign": {
            key: campaign[key] for key in (
                "id", "name", "campaign_type", "dataset", "status", "phase",
                "pipeline_version", "inventory_uri", "inventory_sha256",
                "configuration_uri", "configuration_sha256", "output_uri",
            )
        },
        "inventory_sequence_count": (
            campaign["expected_sequence_count"]
            if campaign["expected_sequence_count"] is not None
            else campaign["sequence_count"]
        ),
        "request_count": len(rows),
        "unique_sequence_count": len(unique_sequences),
        "committed_export_count": campaign["committed_export_count"],
        "request_status_counts": status_counts,
        "accuracy_failure_counts": accuracy_failure_counts,
        "metric_percentiles": metrics,
        "manual_reviews": {
            row["sequence"]: {
                "anonymized_videos": "PENDING", "pose_overlay": "PENDING",
                "reviewer": None, "reviewed_at": None,
            }
            for row in rows
            if row["cohort"] == "CANARY" and row["status"] == "SUCCEEDED"
        },
        "rows_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
        "rows": rows,
    }
    expected_count = (
        campaign["expected_sequence_count"]
        if campaign["expected_sequence_count"] is not None
        else campaign["sequence_count"]
    )
    if len(unique_sequences) != expected_count:
        raise ValueError("Campaign request sequences do not reconcile with frozen membership")
    return report, rows


def write_report(campaign_name: str, *, db_path: str, output_dir: Path) -> dict:
    report, rows = build_report(campaign_name, db_path=db_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{campaign_name}.json"
    csv_path = output_dir / f"{campaign_name}.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return {
        "json": str(json_path), "csv": str(csv_path),
        "json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "row_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=state_path("manifests") / "reports")
    args = parser.parse_args()
    print(json.dumps(write_report(
        args.campaign, db_path=args.db, output_dir=args.output_dir,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
