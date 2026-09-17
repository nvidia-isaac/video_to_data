# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Query workflow status, metrics, and aggregate summaries.

Single sequence:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --sequence <name>

Aggregate summary:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --summary

Sequence lifecycle across active campaigns:
    python query.py --dataset sc_office_4exo_1 --summary

Sequence lifecycle for one campaign:
    python query.py --dataset sc_office_4exo_1 --campaign <name> --summary

Latest row per sequence:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --latest

List all:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import subprocess
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MV_HOI_DIR = os.path.dirname(SCRIPT_DIR)
try:
    from .db import (
        DB_PATH,
        PIPELINES_TABLE,
        PIPELINES_TEST_TABLE,
        TEST_DB_PATH,
        apply_execution_observation,
        get_latest_workflow,
        get_blacklisted_sequence,
        get_campaign_lifecycle_snapshot,
        get_stage_run_by_request,
        get_stage_run,
        get_stage_request,
        get_workflow_execution,
        get_sequence_history,
        get_sequence_status,
        get_summary,
        get_workflows_by_export_id,
        init_db,
        list_current_stage_runs,
        list_stage_requests,
        list_stage_run_history,
        maybe_blacklist_repeated_failure,
        remove_blacklisted_sequence,
        replace_campaign_infrastructure_attempt,
        replace_campaign_request_infrastructure_attempt,
        set_campaign_execution_retry_state,
        upsert_blacklisted_sequence,
        update_stage_run,
        update_stage_request,
        update_workflow_execution,
    )
    from .config_utils import (
        EXPORT_PIPELINE,
        PREPROCESS_PIPELINE,
        RECON_PIPELINE,
    )
    from .campaign_inventory import CAMERAS_RGB, _client, _list
    from .runtime import require_submit_authority
    from .lifecycle import (
        ACCURACY_CHECK_CATEGORIES,
        LIFECYCLE_BUCKET_ORDER,
        build_campaign_lifecycle_summary,
        lifecycle_summary_is_healthy,
    )
    from .export_commit import (
        CandidateCleanupError,
        cleanup_rejected_export_candidate,
        cleanup_promoted_export_candidate,
        publish_remote_export_commit,
        read_remote_export_rejection,
        verify_remote_export_commit,
    )
    from .canary_validation import validate_accuracy_failure_evidence
except ImportError:  # Direct script execution.
    from db import (
        DB_PATH,
        PIPELINES_TABLE,
        PIPELINES_TEST_TABLE,
        TEST_DB_PATH,
        apply_execution_observation,
        get_latest_workflow,
        get_blacklisted_sequence,
        get_campaign_lifecycle_snapshot,
        get_stage_run_by_request,
        get_stage_run,
        get_stage_request,
        get_workflow_execution,
        get_sequence_history,
        get_sequence_status,
        get_summary,
        get_workflows_by_export_id,
        init_db,
        list_current_stage_runs,
        list_stage_requests,
        list_stage_run_history,
        maybe_blacklist_repeated_failure,
        remove_blacklisted_sequence,
        replace_campaign_infrastructure_attempt,
        replace_campaign_request_infrastructure_attempt,
        set_campaign_execution_retry_state,
        upsert_blacklisted_sequence,
        update_stage_run,
        update_stage_request,
        update_workflow_execution,
    )
    from config_utils import (
        EXPORT_PIPELINE,
        PREPROCESS_PIPELINE,
        RECON_PIPELINE,
    )
    from campaign_inventory import CAMERAS_RGB, _client, _list
    from runtime import require_submit_authority
    from lifecycle import (
        ACCURACY_CHECK_CATEGORIES,
        LIFECYCLE_BUCKET_ORDER,
        build_campaign_lifecycle_summary,
        lifecycle_summary_is_healthy,
    )
    from export_commit import (
        CandidateCleanupError,
        cleanup_rejected_export_candidate,
        cleanup_promoted_export_candidate,
        publish_remote_export_commit,
        read_remote_export_rejection,
        verify_remote_export_commit,
    )
    from canary_validation import validate_accuracy_failure_evidence

TABLE = PIPELINES_TABLE
DEFAULT_REFRESH_WORKERS = int(os.environ.get("MV_HOI_REFRESH_WORKERS", os.cpu_count() or 1))

PREPROCESS_REQUIRED_OUTPUTS = frozenset({
    "edex",
    "hoi_metadata.yaml",
    "object_mesh/output_aligned.glb",
    *(f"images/{camera}.h5" for camera in CAMERAS_RGB),
    *(f"videos/{camera}.mp4" for camera in CAMERAS_RGB),
})
AMBIGUOUS_SUBMIT_NOT_FOUND_GRACE_SECONDS = int(os.environ.get(
    "MV_HOI_AMBIGUOUS_SUBMIT_NOT_FOUND_GRACE_SECONDS", "600",
))
MAX_CAMPAIGN_INFRASTRUCTURE_RESTARTS = 2
CAMPAIGN_INFRASTRUCTURE_BACKOFF_SECONDS = (10 * 60, 30 * 60)
OSMO_TASK_LOG_TAIL_LINES = 200
OSMO_TASK_LOG_TIMEOUT_SECONDS = 30
FILE_NOT_FOUND_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"FileNotFoundError",
    r"\[Errno\s+2\]\s+No such file or directory",
    r"No such file or directory",
))
CSS_TRANSPORT_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"Upload exceeds quota",
    r"EntityTooLarge",
    r"RetriesExceededError:\s*Max Retries Exceeded",
    r"(?:HTTP\s*)?429\s+(?:Too Many Requests|Client Error)",
    r"Too Many Requests.*(?:CSS|Swift|S3|object)",
))
OSMO_CONTROL_FAILURE_EXIT_CODES = frozenset((2031,))
OSMO_INFRASTRUCTURE_WORKFLOW_STATUSES = frozenset(("FAILED_SERVER_ERROR",))
OSMO_INFRASTRUCTURE_TASK_STATUSES = frozenset((
    "FAILED_IMAGE_PULL",
    "FAILED_START_TIMEOUT",
))
def load_config() -> dict:
    with open(os.path.join(MV_HOI_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


# OSMO helpers (read-side)

def _osmo_error_text(result: subprocess.CompletedProcess) -> str:
    return "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
    )


def _json_dicts_in_text(text: str) -> list[dict]:
    decoder = json.JSONDecoder()
    payloads: list[dict] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _is_osmo_not_found_error(text: str) -> bool:
    lower = text.lower()
    for payload in _json_dicts_in_text(text):
        code = payload.get("code")
        message = str(payload.get("message", "")).lower()
        if str(code) == "400" and "workflow " in message and " is not found" in message:
            return True

    return (
        "server responded with status code 400" in lower
        and "workflow " in lower
        and " is not found" in lower
    )


def osmo_query(workflow_name: str, *, classify_logs: bool = True) -> dict:
    """Query OSMO workflow status via JSON output.

    Returns {"status": str, "tasks": {name: status}}. For query failures,
    includes error text and whether OSMO reported the workflow as missing.
    """
    cmd = [
        "osmo", "workflow", "query", workflow_name, "--verbose",
        "--format-type", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error = _osmo_error_text(result)
        return {
            "status": "UNKNOWN",
            "tasks": {},
            "error": error,
            "not_found": _is_osmo_not_found_error(error),
        }

    data = json.loads(result.stdout)
    status = data.get("status", "UNKNOWN")
    task_attempts: dict[str, list[dict]] = {}
    for group in data.get("groups", []):
        for task in group.get("tasks", []):
            detail = {
                key: task.get(key)
                for key in (
                    "name", "retry_id", "status", "failure_message", "exit_code",
                    "input_download_start_time", "input_download_end_time",
                    "output_upload_start_time",
                )
            }
            task_attempts.setdefault(task["name"], []).append(detail)

    tasks: dict[str, str] = {}
    task_details: dict[str, dict] = {}
    for name, attempts in task_attempts.items():
        attempts.sort(key=lambda item: int(item.get("retry_id") or 0))
        latest = attempts[-1]
        tasks[name] = latest["status"]
        task_details[name] = latest

    info = {
        "status": status,
        "tasks": tasks,
        "task_details": task_details,
        "task_attempts": task_attempts,
    }
    if classify_logs and str(status).startswith("FAILED"):
        _classify_file_not_found_failures(workflow_name, info)
    return info


def _failed_task_details(info: dict) -> list[dict]:
    return [
        detail
        for detail in info.get("task_details", {}).values()
        if (
            detail.get("status") == "FAILED"
            or detail.get("status") == "FAILED_BACKEND_ERROR"
            or detail.get("status") in OSMO_INFRASTRUCTURE_TASK_STATUSES
        )
    ]


def _task_is_retryable_infrastructure_failure(detail: dict, info: dict) -> bool:
    name = str(detail.get("name") or "")
    status = str(detail.get("status") or "")
    message = str(detail.get("failure_message") or "").lower()
    try:
        exit_code = int(detail.get("exit_code"))
    except (TypeError, ValueError):
        exit_code = None
    temporary_preprocess_handoff = (
        name == "face_detector"
        and exit_code == 75
        and info.get("tasks", {}).get("mv_preprocess") == "COMPLETED"
    )
    node_resource_rejection = (
        exit_code == 4000
        and "pod was rejected" in message
        and "node didn't have enough resource" in message
    )
    unhealthy_device_rejection = (
        status == "FAILED_BACKEND_ERROR"
        and exit_code == 3001
        and "pod was rejected" in message
        and "allocate failed due to no healthy devices present" in message
        and (
            "nvidia.com/mlnxnics" in message
            or "nvidia.com/gpu" in message
        )
    )
    return (
        exit_code in OSMO_CONTROL_FAILURE_EXIT_CODES
        or "osmo control failure" in message
        or temporary_preprocess_handoff
        or node_resource_rejection
        or unhealthy_device_rejection
        or status in OSMO_INFRASTRUCTURE_TASK_STATUSES
        or detail.get("retryable_failure_category") in {
            "file_not_found", "css_transport",
        }
    )


def _file_not_found_excerpt(text: str) -> str | None:
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in FILE_NOT_FOUND_PATTERNS):
            return line.strip()[:2048]
    return None


def _css_transport_excerpt(text: str) -> str | None:
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in CSS_TRANSPORT_PATTERNS):
            return line.strip()[:2048]
    return None


def _osmo_task_log_tail(
    workflow_name: str, detail: dict, *, error: bool,
) -> str | None:
    command = [
        "osmo", "workflow", "logs", workflow_name,
        "--task", str(detail.get("name") or ""),
        "--retry-id", str(int(detail.get("retry_id") or 0)),
    ]
    if error:
        command.append("--error")
    command.extend(["-n", str(OSMO_TASK_LOG_TAIL_LINES)])
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=OSMO_TASK_LOG_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _osmo_error_text(result) or None


def _classify_file_not_found_failures(workflow_name: str, info: dict) -> None:
    """Annotate unclassified root failures using bounded OSMO log evidence."""

    if is_retryable_infrastructure_failure(info):
        return
    for detail in _failed_task_details(info):
        if _task_is_retryable_infrastructure_failure(detail, info):
            continue
        for error in (True, False):
            log_tail = _osmo_task_log_tail(workflow_name, detail, error=error)
            if not log_tail:
                continue
            category = "file_not_found"
            excerpt = _file_not_found_excerpt(log_tail)
            if excerpt is None:
                category = "css_transport"
                excerpt = _css_transport_excerpt(log_tail)
            if excerpt is None:
                continue
            detail["retryable_failure_category"] = category
            detail["retryable_failure_evidence"] = {
                "task": str(detail.get("name") or ""),
                "source": "osmo_error_log" if error else "osmo_task_log",
                "excerpt": excerpt,
                "sha256": hashlib.sha256(log_tail.encode("utf-8")).hexdigest(),
            }
            break


def is_retryable_infrastructure_failure(info: dict) -> bool:
    """Return whether every root task failure was caused by infrastructure."""

    if info.get("preprocess_output_incomplete") is True:
        return True
    if info.get("missing_submission") is True:
        return True
    if info.get("export_output_incomplete") is True:
        return True
    if info.get("status") in OSMO_INFRASTRUCTURE_WORKFLOW_STATUSES:
        return True
    failed = _failed_task_details(info)
    return bool(failed) and all(
        _task_is_retryable_infrastructure_failure(detail, info)
        for detail in failed
    )


def export_task_failure_info(info: dict, task_names: list[str]) -> dict:
    """Return an infrastructure-classification view for one export sequence."""

    selected = {name for name in task_names if name}
    scoped = {
        "status": info.get("status", "UNKNOWN"),
        "tasks": {
            name: status for name, status in info.get("tasks", {}).items()
            if name in selected
        },
        "task_details": {
            name: dict(detail)
            for name, detail in info.get("task_details", {}).items()
            if name in selected or str(detail.get("name") or "") in selected
        },
    }
    if info.get("status") not in OSMO_INFRASTRUCTURE_WORKFLOW_STATUSES:
        scoped["status"] = "FAILED"
    return scoped


def infrastructure_restart_number(info: dict) -> int:
    """Return the one-based restart number implied by the latest failed retry."""

    failed = _failed_task_details(info)
    if not failed:
        return 1 if (
            info.get("preprocess_output_incomplete") is True
            or info.get("missing_submission") is True
            or info.get("export_output_incomplete") is True
            or info.get("status") in OSMO_INFRASTRUCTURE_WORKFLOW_STATUSES
        ) else 0
    return max(int(detail.get("retry_id") or 0) for detail in failed) + 1


def _request_infrastructure_failure_evidence(info: dict) -> dict:
    """Return the minimal per-request classifier evidence safe to persist."""

    if info.get("export_output_incomplete") is True:
        task = next(iter(info.get("tasks", {})), None)
        return {"classification": "export_output_incomplete", "task": task}
    for detail in _failed_task_details(info):
        category = detail.get("retryable_failure_category")
        evidence = detail.get("retryable_failure_evidence") or {}
        if category:
            return {
                "classification": category,
                "task": evidence.get("task") or detail.get("name"),
                "matched_excerpt": evidence.get("excerpt"),
                "log_sha256": evidence.get("sha256"),
            }
        status = str(detail.get("status") or "")
        if status in OSMO_INFRASTRUCTURE_TASK_STATUSES:
            return {
                "classification": status.lower(),
                "task": detail.get("name"),
            }
        if int(detail.get("exit_code") or 0) in OSMO_CONTROL_FAILURE_EXIT_CODES:
            return {
                "classification": "osmo_control_exit_code",
                "task": detail.get("name"),
            }
    return {"classification": "infrastructure", "task": None}


def handle_campaign_infrastructure_failure(
    *,
    workflow_id: str,
    execution_id: int,
    dataset: str,
    sequence_name: str,
    campaign_name: str | None,
    campaign_type: str | None,
    info: dict,
    retry_infrastructure: bool,
    db_path: str,
    failure_reason: str | None = None,
    query_payload: object | None = None,
    request_id: int | None = None,
) -> str | None:
    """Hold or replace a retryable campaign failure without blacklisting it.

    Returns ``None`` for a normal processing failure, ``RETRY_PENDING`` when
    submit authority is intentionally unavailable, ``RETRY_CREATED`` after a
    fresh request is created, or ``EXHAUSTED`` after the configured retry
    bound.
    """

    if campaign_type not in (
        "LEGACY_REVALIDATION", "BACKLOG_REPROCESSING", "REMEDIATION",
    ) or not is_retryable_infrastructure_failure(info):
        return None

    execution = get_workflow_execution(execution_id, db_path=db_path)
    if request_id is not None:
        request = get_stage_request(request_id, db_path=db_path)
        parameters = json.loads((request or {}).get("parameters_json") or "{}")
        restart_number = int(parameters.get("infrastructure_retry_attempt", 0)) + 1
    else:
        local_match = re.search(
            r"infrastructure_restart_attempts=(\d+)",
            str(execution.get("details") or "") if execution else "",
        )
        local_attempts = int(local_match.group(1)) if local_match else 0
        restart_number = max(
            infrastructure_restart_number(info),
            local_attempts + 1,
        )
    if restart_number > MAX_CAMPAIGN_INFRASTRUCTURE_RESTARTS:
        return "EXHAUSTED"

    detail = (
        "retryable_osmo_infrastructure_failure: "
        f"restart {restart_number}/{MAX_CAMPAIGN_INFRASTRUCTURE_RESTARTS}"
    )
    if failure_reason:
        detail += f": {failure_reason}"
    persisted_query_payload = info if query_payload is None else query_payload
    if not retry_infrastructure:
        if request_id is not None:
            return "RETRY_PENDING"
        set_campaign_execution_retry_state(
            execution_id, status="UNKNOWN", details=detail,
            query_payload=persisted_query_payload, db_path=db_path,
        )
        return "RETRY_PENDING"

    del workflow_id
    replacement_detail = (
        detail + f": infrastructure_retry_attempts={restart_number}"
    )
    if request_id is None:
        replacement = replace_campaign_infrastructure_attempt(
            execution_id, details=replacement_detail,
            query_payload=persisted_query_payload,
            max_attempts=MAX_CAMPAIGN_INFRASTRUCTURE_RESTARTS,
            backoff_seconds=CAMPAIGN_INFRASTRUCTURE_BACKOFF_SECONDS,
            db_path=db_path,
        )
    else:
        replacement = replace_campaign_request_infrastructure_attempt(
            request_id, execution_id, details=replacement_detail,
            query_payload=persisted_query_payload,
            failure_evidence=_request_infrastructure_failure_evidence(info),
            max_attempts=MAX_CAMPAIGN_INFRASTRUCTURE_RESTARTS,
            backoff_seconds=CAMPAIGN_INFRASTRUCTURE_BACKOFF_SECONDS,
            db_path=db_path,
        )
    if replacement is None:
        return "EXHAUSTED"
    if campaign_name:
        blacklist = get_blacklisted_sequence(
            dataset, sequence_name, db_path=db_path,
        )
        if (
            blacklist
            and blacklist.get("created_by") == f"campaign:{campaign_name}"
        ):
            remove_blacklisted_sequence(
                dataset, sequence_name, db_path=db_path,
            )
    return "RETRY_CREATED"


def osmo_cancel(workflow_name: str) -> bool:
    """Cancel a running OSMO workflow. Returns True on success."""
    require_submit_authority("cancel an OSMO workflow")
    cmd = ["osmo", "workflow", "cancel", workflow_name]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  osmo cancel failed: {result.stderr.strip()}")
        return False
    return True


def _failure_detail(info: dict) -> str:
    """Extract a human-readable failure detail from osmo_query result.

    Only reports root-cause FAILED tasks; FAILED_UPSTREAM/FAILED_CANCELED
    tasks are excluded since they are effects, not causes.
    """
    tasks = info.get("tasks", {})
    root = [t for t, s in tasks.items() if s == "FAILED"]
    if root:
        return "task_failed: " + ", ".join(sorted(root))
    return info.get("status", "failed").lower()


def categorize_accuracy_failure(report: dict) -> dict | None:
    """Return only stable accuracy-failure categories from a run report."""
    if (
        report.get("schema") == "v2d.mv_hoi.check_accuracy.v3"
        and report.get("status") == "FAIL"
    ):
        failed = report.get("failed_accuracy_checks")
        if not isinstance(failed, list) or not failed or not all(
            isinstance(name, str) and name for name in failed
        ):
            return None
        return {
            "failure_category": "accuracy_check_failed",
            "failed_accuracy_checks": failed,
            "accuracy_failure_segment_count": report.get(
                "failure_segment_count"
            ),
            "accuracy_failure_coverage": report.get("failure_coverage"),
            "reason": report.get("reason"),
        }
    if (
        report.get("schema") == "v2d.mv_hoi.check_accuracy.v2"
        and report.get("status") == "FAIL"
    ):
        return {
            "failure_category": "accuracy_limit",
            "accuracy_failure_segment_count": report.get("failure_segment_count"),
            "accuracy_failure_coverage": report.get("failure_coverage"),
            "reason": report.get("reason"),
        }
    checks = report.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    failed = [
        name for name in ACCURACY_CHECK_CATEGORIES
        if str(checks.get(name, "")).upper() == "FAIL"
    ]
    if not failed:
        reason = str(report.get("reason") or "").lower()
        reason_categories = (
            ("object silhouette", "object_silhouette_alignment"),
            ("object mask containment", "object_mask_containment"),
            ("object chamfer", "chamfer_object"),
            ("human chamfer", "chamfer_human"),
        )
        failed = [
            category for phrase, category in reason_categories if phrase in reason
        ]
    if not failed:
        return None
    return {
        "failure_category": "accuracy_check_failed",
        "failed_accuracy_checks": failed,
    }


def _read_accuracy_report(output_uri: str) -> dict:
    """Read one reconstruction accuracy report from its exact run output."""
    try:
        from .campaign_inventory import _client
    except ImportError:  # Direct script execution.
        from campaign_inventory import _client

    report_uri = (
        output_uri.rstrip("/") + "/check_accuracy/check_accuracy.json"
    )
    client, bucket, key = _client(report_uri)
    payload = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    if not isinstance(payload, dict):
        raise ValueError("Accuracy report must be a JSON object")
    segments_uri = output_uri.rstrip("/") + "/check_accuracy/failure_segments.json"
    segments_client, segments_bucket, segments_key = _client(segments_uri)
    segments = json.loads(
        segments_client.get_object(
            Bucket=segments_bucket, Key=segments_key,
        )["Body"].read()
    )
    payload["_failure_segments_artifact"] = segments
    return payload


def _accuracy_failure_summary(workflow: dict, info: dict) -> dict | None:
    if (
        workflow.get("pipeline_type") != RECON_PIPELINE
        or info.get("tasks", {}).get("check_accuracy") != "FAILED"
        or not workflow.get("output_uri")
    ):
        return None
    try:
        report = _read_accuracy_report(workflow["output_uri"])
        segments = report.pop("_failure_segments_artifact", None)
        summary = categorize_accuracy_failure(report)
        if report.get("schema") in {
            "v2d.mv_hoi.check_accuracy.v2",
            "v2d.mv_hoi.check_accuracy.v3",
        }:
            expected_thresholds = (
                {
                    "max_chamfer_object_mm": 40.0,
                    "max_chamfer_human_mm": 40.0,
                    "max_chamfer_segment_object_mm": 50.0,
                    "max_chamfer_segment_human_mm": 50.0,
                    "min_silhouette_bbox_containment": 0.8,
                    "min_failure_run_frames": 5,
                    "max_silhouette_failure_coverage": 0.5,
                }
                if report.get("schema") == "v2d.mv_hoi.check_accuracy.v3"
                else {
                    "max_chamfer_object_mm": 40.0,
                    "max_chamfer_human_mm": 40.0,
                    "min_silhouette_bbox_containment": 0.8,
                    "min_failure_run_frames": 5,
                    "max_failure_segments": 10,
                    "max_failure_coverage": 0.5,
                }
            )
            evidence_valid, evidence_detail = validate_accuracy_failure_evidence(
                report, segments,
                expected_thresholds=expected_thresholds,
            )
            if not evidence_valid:
                print(
                    "  invalid accuracy canary evidence for "
                    f"{workflow.get('sequence_name')}: {evidence_detail}"
                )
                return None
            if summary is not None:
                summary.update({
                    "evidence_valid": True,
                    "evidence_detail": evidence_detail,
                })
                if workflow.get("request_cohort") == "CANARY":
                    summary["canary_outcome"] = "EXPECTED_QC_DATA"
        return summary
    except Exception as exc:
        print(
            "  unable to categorize accuracy failure for "
            f"{workflow.get('sequence_name')}: {exc}"
        )
        return None


def _categorized_failure_detail(detail: str, summary: dict | None) -> str:
    if not summary:
        return detail
    checks = summary.get("failed_accuracy_checks")
    if isinstance(checks, list) and checks:
        return f"{detail}; accuracy_checks_failed: {', '.join(checks)}"
    reason = str(summary.get("reason") or "").strip()
    category = str(summary.get("failure_category") or "accuracy_failure")
    suffix = reason or category
    return f"{detail}; accuracy_failure: {suffix}"


def backfill_accuracy_failure_summaries(
    dataset: str, *, db_path: str = DB_PATH,
) -> int:
    """Enrich terminal accuracy failures that predate categorized ingestion."""
    enriched = 0
    requests = list_stage_requests(
        dataset=dataset, stage="reconstruction", status="FAILED",
        db_path=db_path,
    )
    for request in requests:
        if (
            request.get("result_summary_json")
            or "task_failed: check_accuracy" not in str(request.get("details") or "")
        ):
            continue
        run = get_stage_run_by_request(
            request["id"], stage=RECON_PIPELINE, db_path=db_path,
        )
        if not run or not run.get("output_uri"):
            continue
        summary = _accuracy_failure_summary(
            {
                **run,
                "pipeline_type": RECON_PIPELINE,
                "sequence_name": request["sequence_name"],
            },
            {"tasks": {"check_accuracy": "FAILED"}},
        )
        if not summary:
            continue
        detail = _categorized_failure_detail(
            "task_failed: check_accuracy", summary,
        )
        update_stage_request(
            request["id"], details=detail, result_summary=summary,
            db_path=db_path,
        )
        update_stage_run(
            run["stage_run_id"], details=detail, stage=RECON_PIPELINE,
            db_path=db_path,
        )
        if request.get("workflow_execution_id"):
            update_workflow_execution(
                request["workflow_execution_id"], details=detail,
                db_path=db_path,
            )
        blacklist = get_blacklisted_sequence(
            dataset, request["sequence_name"], db_path=db_path,
        )
        if (
            blacklist
            and request.get("campaign_name")
            and blacklist.get("created_by")
            == f"campaign:{request['campaign_name']}"
        ):
            upsert_blacklisted_sequence(
                dataset, request["sequence_name"], detail,
                created_by=blacklist["created_by"], db_path=db_path,
            )
        enriched += 1
    return enriched


def _maybe_auto_blacklist_repeated_failure(
    workflow: dict,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> None:
    """Blacklist a sequence after its two latest runs fail with same details."""
    del table
    if maybe_blacklist_repeated_failure(
        workflow["stage_run_id"], stage=workflow["pipeline_type"], db_path=db_path,
    ):
        print(
            f"Auto-blacklisted {workflow['dataset']}/{workflow['sequence_name']} "
            f"after 2 recent {workflow['pipeline_type']} failures"
        )


def _blacklist_campaign_failure(row: dict, detail: str, *, db_path: str) -> None:
    """Immediately retain a terminal campaign failure in the timeout box."""
    if row.get("request_campaign_type") not in (
        "LEGACY_REVALIDATION", "BACKLOG_REPROCESSING", "REMEDIATION",
    ):
        return
    upsert_blacklisted_sequence(
        row["dataset"], row["sequence_name"], detail,
        created_by=f"campaign:{row.get('request_campaign_name')}",
        db_path=db_path,
    )


def _query_waiting_workflow(workflow: dict) -> tuple[dict, dict]:
    """Return (workflow row, OSMO query info) for a WAITING_WF row."""
    osmo_id = workflow.get("osmo_workflow_id") or workflow["workflow_name"]
    return workflow, osmo_query(osmo_id)


def _query_export_workflow(export_id: str) -> tuple[str, dict]:
    """Return (export workflow id, OSMO query info)."""
    # Export task failures are independent within one shared workflow.  Defer
    # bounded log classification until each sequence task has been scoped.
    return export_id, osmo_query(export_id, classify_logs=False)


def _is_ambiguous_submit_placeholder(workflow: dict) -> bool:
    details = workflow.get("details") or ""
    return details.startswith((
        "submit_ambiguous:",
        "retryable_osmo_infrastructure_failure:",
    ))


def _ambiguous_submit_not_found_grace_expired(
    workflow: dict, *, now: datetime | None = None,
) -> bool:
    """Bound eventual-consistency handling for an ambiguous OSMO submit."""
    submitted_at = workflow.get("execution_submitted_at") or workflow.get("created_at")
    if not submitted_at:
        # A missing timestamp cannot justify reserving a request forever.
        return True
    try:
        submitted = datetime.fromisoformat(str(submitted_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    if submitted.tzinfo is None:
        submitted = submitted.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return (current - submitted).total_seconds() >= (
        AMBIGUOUS_SUBMIT_NOT_FOUND_GRACE_SECONDS
    )


def _query_waiting_workflows(
    workflows: list[dict],
    max_workers: int,
) -> list[tuple[dict, dict]]:
    """Query OSMO in parallel while leaving DB writes to the caller."""
    if not workflows:
        return []

    worker_count = max(1, min(max_workers, len(workflows)))
    if worker_count == 1:
        return [_query_waiting_workflow(wf) for wf in workflows]

    results: list[tuple[dict, dict]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_query_waiting_workflow, wf)
            for wf in workflows
        ]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed_count == len(futures) or completed_count % 10 == 0:
                print(f"  refreshed {completed_count}/{len(futures)} workflow(s)")
    return results


def _query_export_workflows(
    export_ids: list[str],
    max_workers: int,
) -> list[tuple[str, dict]]:
    """Query export OSMO workflows in parallel while DB writes stay serial."""
    if not export_ids:
        return []

    worker_count = max(1, min(max_workers, len(export_ids)))
    if worker_count == 1:
        return [_query_export_workflow(export_id) for export_id in export_ids]

    results: list[tuple[str, dict]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_query_export_workflow, export_id)
            for export_id in export_ids
        ]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed_count == len(futures) or completed_count % 10 == 0:
                print(
                    f"  refreshed {completed_count}/{len(futures)} "
                    "export workflow(s)"
                )
    return results


def validate_remote_preprocess_output(output_uri: str | None) -> dict:
    """Verify that a completed preprocess attempt was actually published.

    OSMO task execution can finish successfully even when its output uploader
    fails afterward.  Treat the remote CSS objects, rather than the task exit
    status alone, as the preprocess completion boundary.
    """
    if not output_uri:
        raise ValueError("preprocess run has no output URI")
    client, bucket, prefix = _client(output_uri, max_pool_connections=4)
    objects = _list(
        client, bucket, prefix, workers=1, finalize=False,
    )
    normalized_prefix = prefix.rstrip("/") + "/"
    paths = {
        item["key"][len(normalized_prefix):]: int(item.get("size", 0))
        for item in objects
        if item["key"].startswith(normalized_prefix)
    }
    missing = sorted(PREPROCESS_REQUIRED_OUTPUTS - paths.keys())
    empty = sorted(
        path for path in PREPROCESS_REQUIRED_OUTPUTS
        if path in paths and paths[path] <= 0
    )
    if missing or empty:
        parts = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if empty:
            parts.append("empty=" + ",".join(empty))
        raise ValueError(
            "remote preprocess output is incomplete: " + "; ".join(parts)
        )
    return {
        "required_file_count": len(PREPROCESS_REQUIRED_OUTPUTS),
        "remote_object_count": len(paths),
    }


def refresh_waiting(
    dataset: str,
    pipeline_type: str | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    max_workers: int = DEFAULT_REFRESH_WORKERS,
    retry_infrastructure: bool = False,
    skip_request_ids: set[int] | None = None,
) -> None:
    """Poll OSMO for WAITING_WF rows and advance or fail them."""
    workflows = list_current_stage_runs(
        dataset, stage=pipeline_type, status=["WAITING_WF", "UNKNOWN"],
        db_path=db_path, table=table,
    )
    if skip_request_ids:
        workflows = [
            workflow
            for workflow in workflows
            if workflow.get("request_id") not in skip_request_ids
        ]
    # Export attempts share workflows and have per-sequence task status.  They
    # are refreshed by refresh_waiting_exports(), not by whole-workflow status.
    workflows = [wf for wf in workflows if wf["pipeline_type"] != EXPORT_PIPELINE]
    if not workflows:
        print("Refreshing waiting workflow statuses: 0 WAITING_WF rows")
        return

    worker_count = max(1, min(max_workers, len(workflows)))
    print(
        f"Refreshing waiting workflow statuses: {len(workflows)} WAITING_WF "
        f"row(s) with {worker_count} worker(s)"
    )
    for wf, info in _query_waiting_workflows(workflows, max_workers=max_workers):
        wf_status = info["status"]
        retry_info = info
        retry_query_payload = None
        infrastructure_failure_reason = None
        if info.get("not_found") and _is_ambiguous_submit_placeholder(wf):
            if not _ambiguous_submit_not_found_grace_expired(wf):
                if wf.get("execution_id"):
                    update_workflow_execution(
                        wf["execution_id"], status="UNKNOWN",
                        details="submit_ambiguous_not_found_yet", query_payload=info,
                        db_path=db_path,
                    )
                # OSMO registration can be eventually consistent immediately
                # after a timed-out submission. Retain it only for a bounded
                # grace period, then recover it as infrastructure failure.
                continue
            retry_info = {
                **info,
                "missing_submission": True,
            }
            retry_query_payload = info
            infrastructure_failure_reason = (
                "osmo_submission_not_found_after_grace"
            )
            # This is a local lifecycle decision, not a synthesized OSMO
            # workflow status. Persist the raw UNKNOWN/not_found observation.
            wf_status = "FAILED"
        if (
            wf_status == "COMPLETED"
            and wf.get("pipeline_type") == PREPROCESS_PIPELINE
        ):
            try:
                validate_remote_preprocess_output(wf.get("output_uri"))
            except Exception as exc:
                infrastructure_failure_reason = "preprocess_output_incomplete"
                retry_info = {
                    **info,
                    "status": "FAILED",
                    "preprocess_output_incomplete": True,
                    "preprocess_output_validation_error": str(exc),
                }
                retry_query_payload = retry_info
                wf_status = "FAILED"
        if wf_status == "COMPLETED":
            if wf.get("execution_id"):
                apply_execution_observation(
                    wf["execution_id"], execution_status="SUCCEEDED",
                    details="workflow_completed", query_payload=info,
                    run_outcomes=[{
                        "run_id": wf["stage_run_id"],
                        "stage": wf["pipeline_type"], "status": "SUCCEEDED",
                    }], db_path=db_path,
                )
            else:
                update_stage_run(
                    wf["stage_run_id"], status="PASS", details="workflow_completed",
                    db_path=db_path,
                )
        elif wf_status.startswith("FAILED"):
            retry_result = None
            if wf.get("execution_id"):
                retry_result = handle_campaign_infrastructure_failure(
                    workflow_id=(
                        wf.get("osmo_workflow_id") or wf["workflow_name"]
                    ),
                    execution_id=wf["execution_id"],
                    dataset=wf["dataset"],
                    sequence_name=wf["sequence_name"],
                    campaign_name=wf.get("request_campaign_name"),
                    campaign_type=wf.get("request_campaign_type"),
                    info=retry_info,
                    retry_infrastructure=retry_infrastructure,
                    db_path=db_path,
                    failure_reason=infrastructure_failure_reason,
                    query_payload=retry_query_payload,
                )
            if retry_result in ("RETRY_PENDING", "RETRY_CREATED"):
                continue
            observed_failure_detail = (
                infrastructure_failure_reason or _failure_detail(info)
            )
            detail = (
                "osmo_infrastructure_retry_exhausted: " + observed_failure_detail
                if retry_result == "EXHAUSTED"
                else observed_failure_detail
            )
            result_summary = _accuracy_failure_summary(wf, info)
            detail = _categorized_failure_detail(detail, result_summary)
            if wf.get("execution_id"):
                request_outcomes = []
                if result_summary and wf.get("request_id"):
                    request_outcomes.append({
                        "request_id": wf["request_id"],
                        "status": "FAILED",
                        "details": detail,
                        "result_summary": result_summary,
                    })
                apply_execution_observation(
                    wf["execution_id"], execution_status="FAILED", details=detail,
                    query_payload=info, run_outcomes=[{
                        "run_id": wf["stage_run_id"],
                        "stage": wf["pipeline_type"], "status": "FAILED",
                    }], request_outcomes=request_outcomes, db_path=db_path,
                )
            else:
                update_stage_run(
                    wf["stage_run_id"], status="FAIL", details=detail, db_path=db_path,
                )
            _maybe_auto_blacklist_repeated_failure(wf, db_path=db_path, table=table)
            _blacklist_campaign_failure(wf, detail, db_path=db_path)
        elif wf.get("execution_id"):
            apply_execution_observation(
                wf["execution_id"], execution_status="RUNNING", query_payload=info,
                run_outcomes=[{
                    "run_id": wf["stage_run_id"],
                    "stage": wf["pipeline_type"], "status": "RUNNING",
                }], db_path=db_path,
            )


def _export_task_suffix(sequence_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", sequence_name).strip("_") or "sequence"
    digest = hashlib.sha1(sequence_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:50]}_{digest}"


def export_task_names(sequence_name: str) -> tuple[str, str]:
    suffix = _export_task_suffix(sequence_name)
    return f"export_{suffix}", f"copy_failure_segments_{suffix}"


def _status_failed(status: str | None) -> bool:
    return bool(status and status.startswith("FAILED"))


def _export_output_is_incomplete(error: BaseException) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in (
        "nosuchkey", "not found", "404", "key set differs",
        "no remote export commit", "incomplete remote export commit",
    ))


def _post_trim_rejection(
    row: dict, *, db_path: str = DB_PATH,
) -> tuple[dict, dict] | None:
    """Read and provenance-check one request-scoped early rejection."""
    parameters = json.loads(row.get("request_parameters_json") or "{}")
    candidate = parameters.get("candidate_uri")
    if not candidate:
        return None
    try:
        report = read_remote_export_rejection(candidate)
    except Exception:
        return None
    provenance = report.get("provenance") or {}
    reconstruction = get_stage_run(
        int(row["reconstruction_run_id"]), stage=RECON_PIPELINE,
        db_path=db_path,
    )
    reconstruction_request_id = (
        reconstruction.get("request_id") if reconstruction else None
    ) or row.get("reconstruction_run_id")
    expected = {
        "sequence_name": row["sequence_name"],
        "pipeline_version": row["pipeline_version"],
        # finalize_export's request_id is the reconstruction/metrics request;
        # the request-scoped candidate and export_run_id independently bind
        # this report to the export-stage request.
        "request_id": reconstruction_request_id,
        "reconstruction_run_id": row.get("reconstruction_run_id"),
        "export_run_id": row.get("stage_run_id"),
    }
    for key, value in expected.items():
        if value is not None and str(provenance.get(key)) != str(value):
            raise ValueError(
                f"Export rejection provenance mismatch for {key}"
            )
    return report, parameters


def waiting_export_rows(
    dataset: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> list[dict]:
    return list_current_stage_runs(
        dataset,
        stage=EXPORT_PIPELINE,
        status=["WAITING_WF", "UNKNOWN"],
        db_path=db_path,
        table=table,
    )


def refresh_waiting_exports(
    dataset: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    max_workers: int = DEFAULT_REFRESH_WORKERS,
    retry_infrastructure: bool = False,
) -> None:
    """Poll OSMO export workflows and update WAITING_EXPORT source rows."""
    rows = waiting_export_rows(dataset, db_path=db_path, table=table)
    if not rows:
        print("Refreshing export workflow statuses: 0 WAITING_EXPORT rows")
        return

    export_ids = sorted(
        {
            row.get("osmo_export_workflow_id")
            for row in rows
            if row.get("osmo_export_workflow_id")
        }
    )
    print(
        f"Refreshing export workflow statuses: {len(rows)} WAITING_EXPORT row(s) "
        f"across {len(export_ids)} export workflow(s) with "
        f"{max(1, min(max_workers, len(export_ids) or 1))} worker(s)"
    )

    for export_id, info in _query_export_workflows(
        export_ids,
        max_workers=max_workers,
    ):
        tasks = info.get("tasks", {})
        wf_status = info.get("status", "UNKNOWN")
        batch_rows = get_workflows_by_export_id(
            dataset,
            export_id,
            pipeline_type=EXPORT_PIPELINE,
            status=["WAITING_WF", "UNKNOWN"],
            db_path=db_path,
            table=table,
        )
        execution_status = (
            "SUCCEEDED" if wf_status == "COMPLETED"
            else "FAILED" if wf_status.startswith("FAILED")
            else "RUNNING"
        )
        execution_details = (
            _failure_detail(info) if execution_status == "FAILED"
            else "workflow_completed" if execution_status == "SUCCEEDED"
            else None
        )
        for row in batch_rows:
            export_task = row.get("workflow_task_name")
            copy_task = row.get("auxiliary_task_name")
            if not export_task:
                export_task, default_copy_task = export_task_names(row["sequence_name"])
                copy_task = copy_task or default_copy_task
            export_status = tasks.get(export_task)
            copy_status = tasks.get(copy_task) if copy_task else "COMPLETED"

            if export_status == "COMPLETED" and copy_status == "COMPLETED":
                if not copy_task:
                    parameters = json.loads(row.get("request_parameters_json") or "{}")
                    commit_url = parameters.get("candidate_uri") or row["output_uri"]
                    if str(row.get("details") or "").startswith(
                        "candidate_cleanup_pending"
                    ):
                        if row.get("request_campaign_type") in (
                            "BACKLOG_REPROCESSING", "REMEDIATION",
                        ):
                            continue
                        try:
                            cleanup_promoted_export_candidate(
                                parameters["candidate_uri"],
                                parameters.get("output_uri") or row["output_uri"],
                            )
                        except CandidateCleanupError:
                            continue
                        apply_execution_observation(
                            row["execution_id"], execution_status=execution_status,
                            details=execution_details, query_payload=info,
                            run_outcomes=[{
                                "run_id": row["stage_run_id"],
                                "stage": EXPORT_PIPELINE,
                                "status": "SUCCEEDED",
                                "details": "export_completed",
                            }], db_path=db_path,
                        )
                        continue
                    try:
                        verify_remote_export_commit(commit_url)
                    except Exception as exc:
                        retry_result = None
                        if _export_output_is_incomplete(exc):
                            retry_info = {
                                "status": "FAILED",
                                "export_output_incomplete": True,
                                "task_details": {},
                                "tasks": {export_task: "COMPLETED"},
                            }
                            retry_result = handle_campaign_infrastructure_failure(
                                workflow_id=export_id,
                                execution_id=row["execution_id"],
                                request_id=row.get("request_id"),
                                dataset=row["dataset"],
                                sequence_name=row["sequence_name"],
                                campaign_name=row.get("request_campaign_name"),
                                campaign_type=row.get("request_campaign_type"),
                                info=retry_info,
                                retry_infrastructure=retry_infrastructure,
                                failure_reason="completed_export_output_incomplete",
                                query_payload=info,
                                db_path=db_path,
                            )
                        if retry_result in ("RETRY_PENDING", "RETRY_CREATED"):
                            continue
                        detail = f"export_commit_invalid: {exc}"
                        if retry_result == "EXHAUSTED":
                            detail = "osmo_infrastructure_retry_exhausted: " + detail
                        apply_execution_observation(
                            row["execution_id"], execution_status=execution_status,
                            details=execution_details, query_payload=info,
                            run_outcomes=[{
                                "run_id": row["stage_run_id"],
                                "stage": EXPORT_PIPELINE,
                                "status": "FAILED", "details": detail,
                            }], db_path=db_path,
                        )
                        _maybe_auto_blacklist_repeated_failure(
                            row, db_path=db_path, table=table,
                        )
                        _blacklist_campaign_failure(row, detail, db_path=db_path)
                        continue
                    if row.get("request_campaign_type") in (
                        "BACKLOG_REPROCESSING", "REMEDIATION",
                    ):
                        apply_execution_observation(
                            row["execution_id"], execution_status=execution_status,
                            details=execution_details, query_payload=info,
                            run_outcomes=[{
                                "run_id": row["stage_run_id"],
                                "stage": EXPORT_PIPELINE,
                                "status": "RUNNING",
                                "details": "candidate_export_verified",
                            }], db_path=db_path,
                        )
                        continue
                    parameters = json.loads(
                        row.get("request_parameters_json") or "{}"
                    )
                    candidate_uri = parameters.get("candidate_uri")
                    output_uri = parameters.get("output_uri") or row["output_uri"]
                    if (
                        candidate_uri
                        and candidate_uri.rstrip("/") != output_uri.rstrip("/")
                    ):
                        try:
                            require_submit_authority(
                                "publish a verified request-isolated export"
                            )
                            publish_remote_export_commit(
                                candidate_uri, output_uri,
                            )
                        except CandidateCleanupError as exc:
                            apply_execution_observation(
                                row["execution_id"],
                                execution_status=execution_status,
                                details=execution_details,
                                query_payload=info,
                                run_outcomes=[{
                                    "run_id": row["stage_run_id"],
                                    "stage": EXPORT_PIPELINE,
                                    "status": "RUNNING",
                                    "details": f"candidate_cleanup_pending: {exc}",
                                }],
                                db_path=db_path,
                            )
                            continue
                        except Exception as exc:
                            detail = f"export_publish_failed: {exc}"
                            apply_execution_observation(
                                row["execution_id"],
                                execution_status=execution_status,
                                details=execution_details,
                                query_payload=info,
                                run_outcomes=[{
                                    "run_id": row["stage_run_id"],
                                    "stage": EXPORT_PIPELINE,
                                    "status": "FAILED",
                                    "details": detail,
                                }],
                                db_path=db_path,
                            )
                            _maybe_auto_blacklist_repeated_failure(
                                row, db_path=db_path, table=table,
                            )
                            continue
                apply_execution_observation(
                    row["execution_id"], execution_status=execution_status,
                    details=execution_details, query_payload=info,
                    run_outcomes=[{
                        "run_id": row["stage_run_id"], "stage": EXPORT_PIPELINE,
                        "status": "SUCCEEDED", "details": "export_completed",
                    }], db_path=db_path,
                )
                if row.get("request_campaign_type") in (
                    "BACKLOG_REPROCESSING", "REMEDIATION",
                ):
                    remove_blacklisted_sequence(
                        row["dataset"], row["sequence_name"], db_path=db_path,
                    )
                continue

            task_states = [(export_task, export_status)]
            if copy_task:
                task_states.append((copy_task, copy_status))
            failed_tasks = [
                name
                for name, status in task_states
                if _status_failed(status)
            ]
            if failed_tasks:
                try:
                    rejected = _post_trim_rejection(row, db_path=db_path)
                except Exception as exc:
                    rejected = None
                    print(
                        f"  {row['sequence_name']}: malformed export rejection: {exc}"
                    )
                if rejected is not None:
                    report, parameters = rejected
                    summary = {
                        "failure_category": "post_trim_qc_limit",
                        "canary_outcome": "EXPECTED_QC_DATA",
                        "export_rejection": report,
                        "export_rejection_sha256": report["report_sha256"],
                        "failed_gates": [
                            gate["name"] for gate in report["gates"]
                            if gate["status"] == "FAIL"
                        ],
                    }
                    detail = "expected_post_trim_qc_rejection: " + ", ".join(
                        summary["failed_gates"]
                    )
                    try:
                        cleanup = cleanup_rejected_export_candidate(
                            parameters["candidate_uri"],
                            expected_report=report,
                        )
                    except Exception as exc:
                        pending_detail = (
                            "post_trim_qc_rejection_cleanup_pending: " + str(exc)
                        )
                        apply_execution_observation(
                            row["execution_id"], execution_status=execution_status,
                            details=execution_details, query_payload=info,
                            run_outcomes=[{
                                "run_id": row["stage_run_id"],
                                "stage": EXPORT_PIPELINE,
                                "status": "UNKNOWN", "details": pending_detail,
                            }], request_outcomes=[{
                                "request_id": row["request_id"],
                                "status": "UNKNOWN", "details": pending_detail,
                                "result_summary": summary,
                            }], db_path=db_path,
                        )
                        continue
                    summary["staging_cleanup"] = cleanup
                    apply_execution_observation(
                        row["execution_id"], execution_status=execution_status,
                        details=execution_details, query_payload=info,
                        run_outcomes=[{
                            "run_id": row["stage_run_id"],
                            "stage": EXPORT_PIPELINE,
                            "status": "FAILED", "details": detail,
                        }], request_outcomes=[{
                            "request_id": row["request_id"],
                            "status": "FAILED", "details": detail,
                            "result_summary": summary,
                        }], db_path=db_path,
                    )
                    _blacklist_campaign_failure(row, detail, db_path=db_path)
                    continue
                detail = "task_failed: " + ", ".join(failed_tasks)
                scoped_info = export_task_failure_info(
                    info, [name for name, _status in task_states],
                )
                _classify_file_not_found_failures(export_id, scoped_info)
                retry_result = handle_campaign_infrastructure_failure(
                    workflow_id=export_id,
                    execution_id=row["execution_id"],
                    request_id=row.get("request_id"),
                    dataset=row["dataset"],
                    sequence_name=row["sequence_name"],
                    campaign_name=row.get("request_campaign_name"),
                    campaign_type=row.get("request_campaign_type"),
                    info=scoped_info,
                    retry_infrastructure=retry_infrastructure,
                    failure_reason=detail,
                    query_payload=info,
                    db_path=db_path,
                )
                if retry_result in ("RETRY_PENDING", "RETRY_CREATED"):
                    continue
                if retry_result == "EXHAUSTED":
                    detail = "osmo_infrastructure_retry_exhausted: " + detail
                apply_execution_observation(
                    row["execution_id"], execution_status=execution_status,
                    details=execution_details,
                    query_payload=info, run_outcomes=[{
                        "run_id": row["stage_run_id"], "stage": EXPORT_PIPELINE,
                        "status": "FAILED", "details": detail,
                    }], db_path=db_path,
                )
                _maybe_auto_blacklist_repeated_failure(
                    row, db_path=db_path, table=table,
                )
                _blacklist_campaign_failure(row, detail, db_path=db_path)
                continue

            if wf_status == "COMPLETED":
                missing = [
                    name
                    for name, status in task_states
                    if status != "COMPLETED"
                ]
                detail = "export_task_missing: " + ", ".join(missing)
                apply_execution_observation(
                    row["execution_id"], execution_status="SUCCEEDED",
                    query_payload=info, run_outcomes=[{
                        "run_id": row["stage_run_id"], "stage": EXPORT_PIPELINE,
                        "status": "FAILED", "details": detail,
                    }], db_path=db_path,
                )
                _maybe_auto_blacklist_repeated_failure(
                    row, db_path=db_path, table=table,
                )
                _blacklist_campaign_failure(row, detail, db_path=db_path)
            elif wf_status.startswith("FAILED"):
                scoped_info = export_task_failure_info(
                    info, [name for name, _status in task_states],
                )
                _classify_file_not_found_failures(export_id, scoped_info)
                retry_result = handle_campaign_infrastructure_failure(
                    workflow_id=export_id,
                    execution_id=row["execution_id"],
                    request_id=row.get("request_id"),
                    dataset=row["dataset"],
                    sequence_name=row["sequence_name"],
                    campaign_name=row.get("request_campaign_name"),
                    campaign_type=row.get("request_campaign_type"),
                    info=scoped_info,
                    retry_infrastructure=retry_infrastructure,
                    failure_reason=f"export_workflow_failed: {wf_status}",
                    query_payload=info,
                    db_path=db_path,
                )
                if retry_result in ("RETRY_PENDING", "RETRY_CREATED"):
                    continue
                detail = f"export_workflow_failed: {wf_status}"
                if retry_result == "EXHAUSTED":
                    detail = "osmo_infrastructure_retry_exhausted: " + detail
                apply_execution_observation(
                    row["execution_id"], execution_status="FAILED",
                    details=_failure_detail(info), query_payload=info,
                    run_outcomes=[{
                        "run_id": row["stage_run_id"], "stage": EXPORT_PIPELINE,
                        "status": "FAILED", "details": detail,
                    }], db_path=db_path,
                )
                _maybe_auto_blacklist_repeated_failure(
                    row, db_path=db_path, table=table,
                )
                _blacklist_campaign_failure(row, detail, db_path=db_path)


def refresh_workflow_states(
    dataset: str,
    pipeline_type: str | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    max_workers: int = DEFAULT_REFRESH_WORKERS,
    retry_infrastructure: bool = False,
    skip_request_ids: set[int] | None = None,
) -> None:
    """Refresh normal OSMO rows plus export rows that belong to reconstruction."""
    if pipeline_type != EXPORT_PIPELINE:
        refresh_waiting(
            dataset,
            pipeline_type=pipeline_type,
            db_path=db_path,
            table=table,
            max_workers=max_workers,
            retry_infrastructure=retry_infrastructure,
            skip_request_ids=skip_request_ids,
        )
    if pipeline_type in (None, RECON_PIPELINE, EXPORT_PIPELINE):
        refresh_waiting_exports(
            dataset,
            db_path=db_path,
            table=table,
            max_workers=max_workers,
            retry_infrastructure=retry_infrastructure,
        )
    if pipeline_type in (None, RECON_PIPELINE):
        enriched = backfill_accuracy_failure_summaries(
            dataset, db_path=db_path,
        )
        if enriched:
            noun = "summary" if enriched == 1 else "summaries"
            print(
                f"Enriched {enriched} reconstruction accuracy failure "
                f"{noun}"
            )


def show_sequence(dataset: str, sequence: str, pipeline_type: str) -> None:
    wf = get_latest_workflow(sequence, dataset, pipeline_type, db_path=DB_PATH,
                             table=TABLE)
    if not wf:
        print(f"No pipeline rows found for {sequence}")
        return

    print(f"Sequence:  {wf['sequence_name']}")
    print(f"Dataset:   {wf['dataset']}")
    print(f"Pipeline:  {wf['pipeline_type']}")
    print(f"Version:   {wf['pipeline_version']}")
    print(f"Workflow:  {wf['workflow_name']}")
    print(f"OSMO ID:   {wf.get('osmo_workflow_id', '')}")
    print(f"Export ID: {wf.get('osmo_export_workflow_id', '') or ''}")
    print(f"Status:    {wf['status']}")
    print(f"Details:   {wf['details']}")
    print(f"Created:   {wf['created_at']}")
    print(f"Updated:   {wf['updated_at']}")


def _stage_attempt_status(row: dict, stage: str) -> str:
    """Report only work belonging to *stage*, never a dependency's state."""
    request_status = row.get(f"{stage}_request_status")
    if request_status:
        return request_status
    run_status = row.get(f"{stage}_run_status")
    if run_status:
        return run_status
    return "NOT_STARTED"


def show_sequence_overview(dataset: str, sequence: str) -> None:
    rows = get_sequence_status(dataset, sequence, db_path=DB_PATH)
    if not rows:
        print(f"No sequence found for {dataset}/{sequence}")
        return
    row = rows[0]
    print(f"Sequence:       {row['sequence_name']}")
    print(f"Kind:           {row['sequence_kind']}")
    print(f"Blacklist:      {row.get('blacklist_reason') or ''}")
    print(
        f"Campaign:       {row.get('campaign_name') or ''} "
        f"{row.get('campaign_type') or ''} {row.get('campaign_cohort') or ''}"
    )
    print(
        f"Revalidation:   {row.get('revalidation_request_status') or ''} "
        f"({row.get('revalidation_request_blocked_reason') or ''})"
    )
    print(
        f"Overall:        {row.get('effective_status') or ''} "
        f"[{row.get('effective_stage') or ''}] "
        f"{row.get('effective_details') or ''}"
    )
    print(f"Calibration:    {_stage_attempt_status(row, 'calibration')} ({row.get('calibration_workflow') or ''})")
    print(f"Preprocessing:  {_stage_attempt_status(row, 'preprocess')} ({row.get('preprocess_workflow') or ''})")
    print(f"Reconstruction: {_stage_attempt_status(row, 'reconstruction')} ({row.get('reconstruction_workflow') or ''})")
    print(f"QC:             {row.get('qc_decision') or ''}")
    print(f"Export:         {_stage_attempt_status(row, 'export')} ({row.get('export_workflow') or ''})")
    print(f"Last activity:  {row.get('last_activity_at') or ''}")


def show_sequence_history(dataset: str, sequence: str) -> None:
    records = get_sequence_history(dataset, sequence, db_path=DB_PATH)
    if not records:
        print(f"No history found for {dataset}/{sequence}")
        return
    for record in records:
        timestamp = (
            record.get("created_at") or record.get("observed_at")
            or record.get("submitted_at") or ""
        )
        kind = record["record_type"]
        if kind == "stage_run":
            summary = f"{record['stage']} {record['status']}"
        elif kind == "qc_review":
            summary = f"QC {record['decision']} ({record['provider']})"
        elif kind == "stage_request":
            summary = f"request {record['stage']} {record['status']}"
        else:
            summary = f"execution {record['pipeline_stage']} {record['status']}"
        details = record.get("details") or ""
        print(f"{timestamp}  {kind:<20}  {summary:<42}  {details}")


def show_sequence_status_list(dataset: str) -> None:
    statuses = get_sequence_status(dataset, db_path=DB_PATH)
    if not statuses:
        print(f"No sequences found for {dataset}")
        return
    headers = (
        "Sequence", "Effective", "Stage", "Blacklist", "Preprocess",
        "Reconstruction", "QC", "Export",
    )
    rows = [
        (
            row["sequence_name"],
            row.get("effective_status") or "",
            row.get("effective_stage") or "",
            "YES" if row.get("blacklist_reason") else "",
            _stage_attempt_status(row, "preprocess"),
            _stage_attempt_status(row, "reconstruction"),
            row.get("qc_decision") or "",
            _stage_attempt_status(row, "export"),
        )
        for row in statuses
    ]
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print("-" * len(fmt.format(*headers)))
    for row in rows:
        print(fmt.format(*row))


def show_campaign_lifecycle_summary(
    dataset: str,
    *,
    campaign: int | str | None = None,
    db_path: str = DB_PATH,
) -> bool:
    snapshot = get_campaign_lifecycle_snapshot(
        dataset, campaign=campaign, db_path=db_path,
    )
    summary = build_campaign_lifecycle_summary(snapshot)
    scope = (
        f"campaign {campaign}"
        if campaign is not None
        else "all FROZEN/RUNNING campaigns"
    )
    print(f"=== Sequence lifecycle summary for {dataset} ({scope}) ===")
    print(f"DB observed at: {summary['observed_at']}")
    print(f"Campaigns selected: {len(summary['campaigns'])}")
    for item in summary["campaigns"]:
        expected = item["expected_member_count"]
        expected_label = "unknown" if expected is None else str(expected)
        print(
            f"  {item['name']} [{item['id']}] "
            f"{item['campaign_type']} {item['status']}/{item['phase']}: "
            f"{item['actual_member_count']}/{expected_label} members"
        )

    expected_distinct = summary["expected_distinct_membership"]
    expected_label = "unknown" if expected_distinct is None else str(expected_distinct)
    print(
        f"Distinct membership: {summary['distinct_membership']} "
        f"(expected: {expected_label})"
    )
    print(
        f"Membership references: {summary['actual_membership_references']} "
        f"(overlap: {summary['overlap']})"
    )
    print("Lifecycle:")
    emitted: set[str] = set()
    for bucket in LIFECYCLE_BUCKET_ORDER:
        if bucket in summary["buckets"]:
            print(f"  {bucket}: {summary['buckets'][bucket]}")
            emitted.add(bucket)
    for bucket in sorted(set(summary["buckets"]) - emitted):
        print(f"  {bucket}: {summary['buckets'][bucket]}")

    if summary["export_authorizations"]:
        print("EXPORTED authorization:")
        for authorization, count in sorted(summary["export_authorizations"].items()):
            print(f"  {authorization}: {count}")
    if summary["blocked_reasons"]:
        print("Blocked reasons:")
        for reason, count in sorted(
            summary["blocked_reasons"].items(), key=lambda item: (-item[1], item[0])
        ):
            print(f"  {reason}: {count}")
    if summary["failure_categories"]:
        print("Categorized failures:")
        for reason, count in sorted(
            summary["failure_categories"].items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {reason}: {count}")
    print(
        f"Reconciled lifecycle total: {summary['reconciled_total']}/"
        f"{summary['distinct_membership']}"
    )
    return lifecycle_summary_is_healthy(summary)


def show_summary(
    dataset: str,
    pipeline_type: str | None = None,
    latest_only: bool = False,
) -> None:
    if latest_only:
        workflows = list_current_stage_runs(
            dataset, stage=pipeline_type, db_path=DB_PATH, table=TABLE,
        )
        seen: set[tuple[str, str]] = set()
        counts: dict[str, int] = {}
        failure_reasons: dict[str, int] = {}
        for wf in workflows:
            key = (wf["sequence_name"], wf["pipeline_type"])
            if key in seen:
                continue
            seen.add(key)
            counts[wf["status"]] = counts.get(wf["status"], 0) + 1
            if wf["status"] == "FAIL":
                failure_reasons[wf["details"]] = failure_reasons.get(wf["details"], 0) + 1
        summary = {
            "counts": counts,
            "failure_reasons": dict(sorted(failure_reasons.items(), key=lambda kv: -kv[1])),
        }
    else:
        summary = get_summary(dataset, pipeline_type=pipeline_type, db_path=DB_PATH,
                              table=TABLE)

    total = sum(summary["counts"].values())
    pipeline_label = pipeline_type or "all pipelines"
    scope = "latest per sequence" if latest_only else "all rows"
    print(f"=== Summary for {dataset} ({pipeline_label}, {scope}) ===")
    print(f"Total pipeline rows: {total}")
    ordered_statuses = (
        "WAITING_WF", "WAITING_QC", "WAITING_EXPORT", "PASS", "FAIL", "SKIPPED",
    )
    for status in ordered_statuses:
        count = summary["counts"].get(status, 0)
        print(f"  {status}: {count}")
    for status, count in sorted(summary["counts"].items()):
        if status not in ordered_statuses:
            print(f"  {status}: {count}")

    if summary["failure_reasons"]:
        print("\nFailure reasons:")
        for reason, count in summary["failure_reasons"].items():
            print(f"  [{count}] {reason or '(no details)'}")


def show_list(
    dataset: str,
    pipeline_type: str | None = None,
    latest_only: bool = False,
) -> None:
    workflows = (
        list_current_stage_runs(
            dataset, stage=pipeline_type, db_path=DB_PATH, table=TABLE,
        )
        if latest_only
        else list_stage_run_history(
            dataset, stage=pipeline_type, db_path=DB_PATH, table=TABLE,
        )
    )
    if not workflows:
        print(f"No pipeline rows found for {dataset}")
        return

    if latest_only:
        seen: set[tuple[str, str]] = set()
        deduped = []
        for wf in workflows:
            key = (wf["sequence_name"], wf["pipeline_type"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(wf)
        workflows = deduped

    rows = [
        (
            wf["sequence_name"],
            wf["pipeline_type"],
            wf["status"],
            wf["pipeline_version"] or "?",
            wf["details"] or "",
        )
        for wf in workflows
    ]
    headers = ("Sequence", "Pipeline", "Status", "Ver", "Details")
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    header_line = fmt.format(*headers)
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(fmt.format(*r))


def main() -> int:
    parser = argparse.ArgumentParser(description="Query MV pipeline workflow status")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--pipeline", "--stage",
                        help="Pipeline type (e.g. mv_calibration, mv_hoi_reconstruction)")
    parser.add_argument("--sequence", help="Show details for a specific sequence")
    parser.add_argument("--summary", action="store_true", help="Show aggregate summary")
    parser.add_argument(
        "--campaign",
        help="Campaign name for the sequence lifecycle summary (closed campaigns allowed)",
    )
    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument("--latest", dest="latest", action="store_true", default=True,
                               help="Show current stage attempts (default)")
    history_group.add_argument("--all-runs", dest="latest", action="store_false",
                               help="Show/count complete attempt history")
    parser.add_argument(
        "--history", action="store_true",
        help="For --sequence, show stage runs, QC reviews, and executions chronologically",
    )
    parser.add_argument("--all-pipelines", "--all-stages", action="store_true",
                        help="Include all pipeline types in summary/list")
    parser.add_argument("--test", action="store_true",
                        help="Use pipelines_test table")
    parser.add_argument("--refresh-workers", type=int, default=DEFAULT_REFRESH_WORKERS,
                        help="Concurrent OSMO queries for workflow/export refresh "
                             f"(default: {DEFAULT_REFRESH_WORKERS}; env: "
                             "MV_HOI_REFRESH_WORKERS)")
    args = parser.parse_args()

    if args.campaign and (
        not args.summary or args.pipeline or args.all_pipelines or args.sequence
    ):
        parser.error(
            "--campaign is allowed only with the bare sequence lifecycle --summary"
        )

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        return 1

    global DB_PATH, TABLE
    if args.test:
        TABLE = PIPELINES_TEST_TABLE
        DB_PATH = TEST_DB_PATH

    if args.summary and not args.pipeline and not args.all_pipelines:
        try:
            healthy = show_campaign_lifecycle_summary(
                args.dataset, campaign=args.campaign, db_path=DB_PATH,
            )
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        return 0 if healthy else 1

    init_db(DB_PATH)

    pipeline_type = None if args.all_pipelines or not args.pipeline else args.pipeline

    refresh_workflow_states(args.dataset, pipeline_type=pipeline_type, db_path=DB_PATH,
                            table=TABLE, max_workers=args.refresh_workers)

    if args.sequence and args.history:
        show_sequence_history(args.dataset, args.sequence)
    elif args.sequence and args.pipeline:
        show_sequence(args.dataset, args.sequence, args.pipeline)
    elif args.sequence:
        show_sequence_overview(args.dataset, args.sequence)
    elif args.summary:
        show_summary(args.dataset, pipeline_type=pipeline_type, latest_only=args.latest)
    elif not args.pipeline and not args.all_pipelines:
        show_sequence_status_list(args.dataset)
    else:
        show_list(args.dataset, pipeline_type=pipeline_type, latest_only=args.latest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
