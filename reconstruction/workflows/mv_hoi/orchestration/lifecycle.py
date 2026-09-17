"""Pure campaign lifecycle resolution shared by query and status publishing."""

from __future__ import annotations

from collections import Counter
import json
import re


ACCURACY_CHECK_CATEGORIES = (
    "chamfer_object",
    "chamfer_human",
    "human_silhouette_bbox_containment",
    "object_silhouette_bbox_containment",
    "object_mask_containment",
    "object_silhouette_alignment",
)
RUNNING_REQUEST_STATUSES = frozenset(("RESERVED", "SUBMITTED", "RUNNING"))
LABEL_BLOCKERS = frozenset(("WAITING_LABELS", "WAITING_RELABEL"))
LIFECYCLE_BUCKET_ORDER = (
    "EXPORTED",
    "WAITING_EXPORT",
    "QC_FAILED",
    "WAITING_QC",
    "WAITING_LABELS",
    "MISSING_RAW_INPUT",
    "REVALIDATION_RUNNING",
    "REVALIDATION_PENDING",
    "REVALIDATION_UNKNOWN",
    "REVALIDATION_FAILED",
    "REVALIDATION_CANCELED",
    "REVALIDATION_BLOCKED",
    "REVALIDATION_SUCCEEDED",
    "EXPORT_RUNNING",
    "EXPORT_PENDING",
    "EXPORT_UNKNOWN",
    "EXPORT_FAILED",
    "EXPORT_CANCELED",
    "EXPORT_BLOCKED",
    "EXPORT_SUCCEEDED",
    "RECONSTRUCTION_RUNNING",
    "RECONSTRUCTION_PENDING",
    "RECONSTRUCTION_UNKNOWN",
    "RECONSTRUCTION_FAILED",
    "RECONSTRUCTION_CANCELED",
    "RECONSTRUCTION_BLOCKED",
    "RECONSTRUCTION_SUCCEEDED",
    "PREPROCESS_RUNNING",
    "PREPROCESS_PENDING",
    "PREPROCESS_UNKNOWN",
    "PREPROCESS_FAILED",
    "PREPROCESS_CANCELED",
    "PREPROCESS_BLOCKED",
    "PREPROCESS_SUCCEEDED",
    "CALIBRATION_RUNNING",
    "CALIBRATION_PENDING",
    "CALIBRATION_UNKNOWN",
    "CALIBRATION_FAILED",
    "CALIBRATION_CANCELED",
    "CALIBRATION_BLOCKED",
    "CALIBRATION_SUCCEEDED",
    "INCONSISTENT",
)


def lifecycle_bucket(row: dict) -> str:
    """Resolve one authoritative campaign request into one lifecycle bucket."""
    if row.get("export_run_status") == "SUCCEEDED":
        return "EXPORTED"

    stage = str(row.get("stage") or "unknown").upper()
    status = str(row.get("status") or "UNKNOWN").upper()
    if status == "BLOCKED":
        reason = str(row.get("blocked_reason") or "").upper()
        if reason in LABEL_BLOCKERS:
            return "WAITING_LABELS"
        if reason == "MISSING_RAW_INPUT":
            return "MISSING_RAW_INPUT"
        return f"{stage}_BLOCKED"
    if (
        stage == "REVALIDATION"
        and status in RUNNING_REQUEST_STATUSES
        and str(row.get("details") or "").startswith(
            "result_reconciliation_pending"
        )
    ):
        return "WAITING_EXPORT"
    if status in RUNNING_REQUEST_STATUSES:
        return f"{stage}_RUNNING"
    if status in ("PENDING", "UNKNOWN", "FAILED", "CANCELED"):
        return f"{stage}_{status}"
    if status != "SUCCEEDED":
        return "INCONSISTENT"

    required_run_status = {
        "CALIBRATION": row.get("calibration_run_status"),
        "PREPROCESS": row.get("preprocess_run_status"),
        "RECONSTRUCTION": row.get("reconstruction_run_status"),
    }.get(stage)
    if stage in ("CALIBRATION", "PREPROCESS"):
        return (
            f"{stage}_SUCCEEDED"
            if required_run_status == "SUCCEEDED"
            else "INCONSISTENT"
        )
    if stage == "RECONSTRUCTION":
        if required_run_status != "SUCCEEDED":
            return "INCONSISTENT"
        if row.get("qc_decision") == "PASS":
            return "WAITING_EXPORT"
        if row.get("qc_decision") == "FAIL":
            return "QC_FAILED"
        return "WAITING_QC"
    # Successful export and revalidation requests are not complete until their
    # request-bound export commit exists (checked above).
    return "INCONSISTENT"


def failure_categories(row: dict, bucket: str) -> list[str]:
    if bucket == "QC_FAILED":
        return ["qc_failed"]
    if not bucket.endswith("_FAILED"):
        return []
    try:
        summary = json.loads(row.get("result_summary_json") or "{}")
    except (TypeError, ValueError):
        summary = {}
    if not isinstance(summary, dict):
        summary = {}
    category = summary.get("failure_category")
    checks = summary.get("failed_accuracy_checks")
    if category == "accuracy_check_failed" and isinstance(checks, list):
        stable_checks = [
            str(check) for check in checks if str(check) in ACCURACY_CHECK_CATEGORIES
        ]
        if stable_checks:
            return [
                f"accuracy_check_failed:{check}" for check in sorted(set(stable_checks))
            ]
    if isinstance(category, str) and category.strip():
        return [category.strip()]

    details = str(row.get("details") or "")
    task = re.search(r"\btask_failed:\s*([A-Za-z0-9_-]+)", details)
    if task:
        return [f"task_failed:{task.group(1)}"]
    return ["uncategorized"]


def lifecycle_stage(bucket: str, request_stage: str | None) -> str:
    if bucket == "EXPORTED":
        return "export"
    if bucket == "WAITING_EXPORT":
        return "export"
    if bucket in ("WAITING_QC", "QC_FAILED"):
        return "qc"
    return str(request_stage or "unknown")


def resolve_lifecycle_row(row: dict) -> dict:
    """Add stable lifecycle and public request aliases to one snapshot row."""
    bucket = lifecycle_bucket(row)
    if bucket == "EXPORTED":
        details = row.get("export_run_details") or row.get("details")
        updated_at = row.get("export_run_updated_at") or row.get("updated_at")
    elif bucket in ("WAITING_EXPORT", "QC_FAILED"):
        if str(row.get("stage") or "").lower() == "revalidation":
            details = row.get("details")
            updated_at = row.get("updated_at")
        else:
            details = row.get("qc_details") or row.get("details")
            updated_at = row.get("qc_observed_at") or row.get("updated_at")
    elif bucket == "WAITING_QC":
        details = row.get("reconstruction_run_details") or row.get("details")
        updated_at = row.get("reconstruction_run_updated_at") or row.get("updated_at")
    else:
        details = row.get("blocked_reason") or row.get("details")
        updated_at = row.get("updated_at")
    categories = failure_categories(row, bucket)
    return {
        **row,
        "authoritative_request_id": row.get("id"),
        "request_stage": row.get("stage"),
        "request_status": row.get("status"),
        "lifecycle_status": bucket,
        "lifecycle_stage": lifecycle_stage(bucket, row.get("stage")),
        "lifecycle_details": details,
        "lifecycle_updated_at": updated_at,
        "failure_categories": ",".join(categories),
    }


def build_campaign_lifecycle_rows(snapshot: dict) -> list[dict]:
    return [
        resolve_lifecycle_row(row)
        for row in sorted(
            snapshot["sequences"],
            key=lambda item: str(item.get("sequence_name") or ""),
        )
    ]


def build_campaign_lifecycle_summary(
    snapshot: dict,
    *,
    rows: list[dict] | None = None,
) -> dict:
    rows = build_campaign_lifecycle_rows(snapshot) if rows is None else rows
    buckets: Counter[str] = Counter()
    authorizations: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for row in rows:
        bucket = row["lifecycle_status"]
        buckets[bucket] += 1
        if bucket == "EXPORTED":
            authorizations[
                str(row.get("export_authorization_type") or "UNKNOWN")
            ] += 1
        if str(row.get("request_status") or "").upper() == "BLOCKED":
            blockers[str(row.get("blocked_reason") or "UNKNOWN")] += 1
        for category in failure_categories(row, bucket):
            failures[category] += 1

    campaigns = snapshot["campaigns"]
    actual_by_campaign = {
        row["campaign_id"]: int(row["member_count"])
        for row in snapshot["campaign_memberships"]
    }
    campaign_memberships = []
    expected_known = True
    membership_ok = True
    expected_references = 0
    actual_references = 0
    for campaign in campaigns:
        actual = actual_by_campaign.get(campaign["id"], 0)
        expected = campaign.get("inventory_sequence_count")
        actual_references += actual
        if expected is None:
            expected_known = False
        else:
            expected = int(expected)
            expected_references += expected
            membership_ok = membership_ok and expected == actual
        campaign_memberships.append({
            **campaign,
            "actual_member_count": actual,
            "expected_member_count": expected,
        })

    distinct = len(rows)
    overlap = actual_references - distinct
    expected_distinct = (
        expected_references - overlap if expected_known else None
    )
    if expected_distinct is not None:
        membership_ok = membership_ok and expected_distinct == distinct
    reconciled = sum(buckets.values())
    return {
        "observed_at": snapshot["observed_at"],
        "campaigns": campaign_memberships,
        "distinct_membership": distinct,
        "actual_membership_references": actual_references,
        "expected_membership_references": (
            expected_references if expected_known else None
        ),
        "overlap": overlap,
        "expected_distinct_membership": expected_distinct,
        "buckets": dict(buckets),
        "export_authorizations": dict(authorizations),
        "blocked_reasons": dict(blockers),
        "failure_categories": dict(failures),
        "reconciled_total": reconciled,
        "membership_reconciles": membership_ok,
        "has_inconsistent": buckets.get("INCONSISTENT", 0) > 0,
    }


def lifecycle_membership_reconciles(summary: dict) -> bool:
    return (
        bool(summary["membership_reconciles"])
        and summary["reconciled_total"] == summary["distinct_membership"]
    )


def lifecycle_summary_is_healthy(summary: dict) -> bool:
    return (
        lifecycle_membership_reconciles(summary)
        and not summary["has_inconsistent"]
    )
