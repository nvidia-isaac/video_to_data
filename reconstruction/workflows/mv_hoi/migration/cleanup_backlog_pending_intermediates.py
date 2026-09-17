#!/usr/bin/env python3
"""Delete stale heavy outputs for undispatched backlog preprocessing requests.

The audit is read-only and records an exact object manifest.  ``--apply``
consumes that immutable audit without changing orchestration database state.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterable

from botocore.exceptions import ClientError


SCRIPT_DIR = Path(__file__).resolve().parent
MV_HOI_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MV_HOI_DIR))

from orchestration import cleanup_intermediates as cleanup, db  # noqa: E402
from orchestration.config_utils import load_config  # noqa: E402
from orchestration.runtime import (  # noqa: E402
    require_submit_authority,
    state_path,
)


SCHEMA = "v2d.mv_hoi.backlog_pending_cleanup_audit.v1"
APPLY_SCHEMA = "v2d.mv_hoi.backlog_pending_cleanup_apply.v1"
PENDING_SCHEMA = "v2d.mv_hoi.backlog_pending_cleanup_pending.v1"
COMPLETE_SCHEMA = "v2d.mv_hoi.backlog_pending_cleanup.v1"
QUOTA_SCHEMA = "v2d.mv_hoi.backlog_quota_retry_plan.v1"
PENDING_NAME = "backlog_quota_cleanup.pending.json"
COMPLETE_NAME = "backlog_quota_cleanup.json"
UNDISPATCHED = frozenset(("PENDING", "BLOCKED"))
WORKFLOW_OWNING = frozenset(("RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN"))
FIXED_PREFIXES = (
    ("rosbag_to_edex/", "rosbag_to_edex"),
    ("mv_preprocess/images/", "mv_preprocess_images"),
    ("face_detector/", "face_detector"),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_quota_plan(path: Path, campaign: dict) -> tuple[dict, str]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != QUOTA_SCHEMA:
        raise ValueError("unsupported quota retry plan schema")
    if (
        int(payload.get("campaign_id", -1)) != int(campaign["id"])
        or payload.get("campaign") != campaign["name"]
    ):
        raise ValueError("quota retry plan campaign identity does not match")
    retries = payload.get("preprocess_retries")
    if not isinstance(retries, list) or not retries:
        raise ValueError("quota retry plan has no preprocess retries")
    sequences = [item.get("sequence") for item in retries]
    retry_ids = [item.get("retry_request_id") for item in retries]
    if len(set(sequences)) != len(sequences) or len(set(retry_ids)) != len(retry_ids):
        raise ValueError("quota retry plan contains duplicate sequence or request IDs")
    return payload, _file_sha256(path)


def _database_snapshot(campaign: dict, *, db_path: str) -> dict:
    """Read request and run lineage without mutating or initializing the DB."""
    conn = db.get_connection(db_path)
    try:
        campaign_rows = [dict(row) for row in conn.execute(
            """SELECT sr.*, s.sequence_name, s.dataset,
                      pr.id AS preprocess_run_id,
                      pr.status AS preprocess_run_status,
                      pr.output_uri AS preprocess_output_uri
               FROM stage_requests sr
               JOIN sequences s ON s.id=sr.sequence_id
               LEFT JOIN preprocess_runs pr ON pr.request_id=sr.id
               WHERE sr.campaign_id=? AND sr.stage='preprocess'
               ORDER BY sr.created_at, sr.id""",
            (campaign["id"],),
        )]
        reconstruction_rows = [dict(row) for row in conn.execute(
            """SELECT sr.id AS request_id, sr.sequence_id, s.sequence_name,
                      sr.status AS request_status,
                      rr.id AS reconstruction_run_id,
                      rr.status AS reconstruction_run_status,
                      rr.output_uri, rr.pipeline_version, rr.details,
                      we.workflow_name
               FROM stage_requests sr
               JOIN sequences s ON s.id=sr.sequence_id
               LEFT JOIN reconstruction_runs rr ON rr.request_id=sr.id
               LEFT JOIN workflow_executions we ON we.id=rr.workflow_execution_id
               WHERE sr.campaign_id=? AND sr.stage='reconstruction'
               ORDER BY sr.created_at, sr.id""",
            (campaign["id"],),
        )]
        all_requests = [dict(row) for row in conn.execute(
            """SELECT sr.id, sr.sequence_id, s.sequence_name, sr.campaign_id,
                      sr.stage, sr.status, sr.created_at
               FROM stage_requests sr
               JOIN sequences s ON s.id=sr.sequence_id
               WHERE s.dataset=?
               ORDER BY sr.created_at, sr.id""",
            (campaign["dataset"],),
        )]
    finally:
        conn.close()
    return {
        "preprocess": campaign_rows,
        "reconstruction": reconstruction_rows,
        "all_requests": all_requests,
    }


def _select_candidates(campaign: dict, quota_plan: dict, snapshot: dict) -> list[dict]:
    by_sequence: dict[str, list[dict]] = defaultdict(list)
    by_id: dict[int, dict] = {}
    for row in snapshot["preprocess"]:
        by_sequence[row["sequence_name"]].append(row)
        by_id[int(row["id"])] = row

    quota_by_sequence = {
        item["sequence"]: item for item in quota_plan["preprocess_retries"]
    }
    all_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in snapshot["all_requests"]:
        all_by_sequence[row["sequence_name"]].append(row)

    reconstruction_by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in snapshot["reconstruction"]:
        reconstruction_by_sequence[row["sequence_name"]].append(row)

    candidates = []
    for sequence, requests in sorted(by_sequence.items()):
        latest = requests[-1]
        if latest["status"] not in UNDISPATCHED:
            continue
        global_rows = all_by_sequence.get(sequence, [])
        if any(row["status"] in WORKFLOW_OWNING for row in global_rows):
            continue
        if any(
            (row.get("created_at") or "", int(row["id"]))
            > (latest.get("created_at") or "", int(latest["id"]))
            for row in global_rows
        ):
            continue

        succeeded = [
            row for row in requests if row.get("preprocess_run_status") == "SUCCEEDED"
        ]
        quota = quota_by_sequence.get(sequence)
        quota_authorized = False
        quota_source = None
        if quota is not None:
            retry_id = int(quota["retry_request_id"])
            source_id = int(quota["source_request_id"])
            quota_source = by_id.get(source_id)
            expected_reason = f"missing_face_videos_retry_of_request_{source_id}"
            quota_authorized = (
                int(latest["id"]) == retry_id
                and latest.get("reason") == expected_reason
                and quota_source is not None
                and quota_source["sequence_name"] == sequence
            )
            if not quota_authorized:
                raise ValueError(
                    f"{sequence}: quota retry plan does not match database lineage"
                )
        if succeeded and not quota_authorized:
            continue

        reason = (
            "QUOTA_RETRY_MISSING_FACE_VIDEOS"
            if quota_authorized and succeeded
            else "NO_SUCCESSFUL_CAMPAIGN_PREPROCESS"
        )
        reconstruction_runs = []
        for row in reconstruction_by_sequence.get(sequence, []):
            if (
                row.get("reconstruction_run_id") is not None
                and row.get("reconstruction_run_status")
                in cleanup.TERMINAL_RUN_STATUSES
                and row.get("output_uri")
            ):
                reconstruction_runs.append({
                    key: row.get(key) for key in (
                        "request_id", "reconstruction_run_id",
                        "reconstruction_run_status", "output_uri",
                        "pipeline_version", "details", "workflow_name",
                    )
                })
        candidates.append({
            "sequence": sequence,
            "candidate_reason": reason,
            "request_id": int(latest["id"]),
            "request_status": latest["status"],
            "request_pipeline_version": latest["pipeline_version"],
            "request_created_at": latest.get("created_at"),
            "quota_retry": quota,
            "quota_source_request_status": (
                quota_source.get("status") if quota_source else None
            ),
            "successful_preprocess_request_ids": [int(row["id"]) for row in succeeded],
            "reconstruction_runs": reconstruction_runs,
        })
    return candidates


def _output_root(campaign: dict, config: dict):
    dataset_cfg = config["datasets"][campaign["dataset"]]
    pipeline = dataset_cfg["pipelines"][db.PREPROCESS_STAGE]
    uri = cleanup._join(
        dataset_cfg["swift_base"],
        pipeline.get("campaign_output_path", pipeline["output_path"]),
    )
    client, bucket, prefix = cleanup._client(uri)
    return client, bucket, prefix, uri


def _classify_deletion(relative: str, reconstruction_prefixes: Iterable[str]) -> str | None:
    for prefix, category in FIXED_PREFIXES:
        if relative.startswith(prefix):
            return category
    if any(relative.startswith(prefix) for prefix in reconstruction_prefixes):
        return "reconstruction"
    return None


def _list_with_backoff(client, bucket: str, prefix: str) -> list[dict]:
    for attempt in range(8):
        try:
            return cleanup._list(client, bucket, prefix)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if code not in ("429", "SlowDown", "Throttling") and status not in (429, 503):
                raise
            if attempt == 7:
                raise
            time.sleep(min(20.0, 2.0 ** attempt) + random.random())
    raise AssertionError("unreachable")


def _get_with_backoff(client, bucket: str, key: str) -> tuple[bytes, dict]:
    for attempt in range(8):
        try:
            return cleanup._get(client, bucket, key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if code not in ("429", "SlowDown", "Throttling") and status not in (429, 503):
                raise
            if attempt == 7:
                raise
            time.sleep(min(20.0, 2.0 ** attempt) + random.random())
    raise AssertionError("unreachable")


def _delete_with_backoff(client, bucket: str, keys: list[str]) -> None:
    for attempt in range(8):
        try:
            cleanup._delete(client, bucket, keys)
            return
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if code not in ("429", "SlowDown", "Throttling") and status not in (429, 503):
                raise
            if attempt == 7:
                raise
            time.sleep(min(20.0, 2.0 ** attempt) + random.random())


def _delete_object_with_backoff(client, bucket: str, key: str) -> None:
    for attempt in range(8):
        try:
            client.delete_object(Bucket=bucket, Key=key)
            return
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if code not in ("429", "SlowDown", "Throttling") and status not in (429, 503):
                raise
            if attempt == 7:
                raise
            time.sleep(min(20.0, 2.0 ** attempt) + random.random())


def _remove_pending_marker(client, bucket: str, key: str) -> None:
    _delete_object_with_backoff(client, bucket, key)
    for attempt in range(6):
        if _optional_json(client, bucket, key) is None:
            return
        if attempt < 5:
            time.sleep(min(8.0, 2.0 ** attempt) + random.random())
    raise RuntimeError(f"pending cleanup marker remains after deletion: {key}")


def _list_relevant_with_backoff(client, bucket: str, record: dict) -> list[dict]:
    """List only prefixes the apply transaction may delete or must protect."""
    prefixes = [
        cleanup._join(record["sequence_root"], "mv_preprocess"),
        cleanup._join(record["sequence_root"], "rosbag_to_edex"),
        cleanup._join(record["sequence_root"], "face_detector"),
        *(
            cleanup._join(record["sequence_root"], run["relative_prefix"])
            for run in record["reconstruction_runs"]
        ),
    ]
    objects = {}
    for prefix in dict.fromkeys(prefixes):
        for item in _list_with_backoff(client, bucket, prefix):
            objects[item["key"]] = item
    return sorted(objects.values(), key=lambda item: item["key"])


def _delete_and_verify(
    client, bucket: str, record: dict, keys: list[str],
) -> list[dict]:
    """Delete exact keys and tolerate temporarily stale CSS object listings."""
    audited_keys = {item["key"] for item in record["deletions"]}
    remaining = list(keys)
    for attempt in range(8):
        if remaining:
            _delete_with_backoff(client, bucket, remaining)
        current = _list_relevant_with_backoff(client, bucket, record)
        remaining = [item["key"] for item in current if item["key"] in audited_keys]
        if not remaining:
            return current
        if attempt == 7:
            break
        time.sleep(min(20.0, 2.0 ** attempt) + random.random())
    raise RuntimeError(
        f"{record['sequence']}: cleanup left {len(remaining)} object(s)"
    )


def _inspect_candidate(candidate: dict, *, client, bucket: str, output_root: str) -> dict:
    sequence_root = cleanup._join(output_root, candidate["sequence"])
    objects = _list_with_backoff(client, bucket, sequence_root)
    reconstruction_prefixes = []
    retained_runs = []
    for run in candidate["reconstruction_runs"]:
        _run_client, run_bucket, run_prefix = cleanup._client(run["output_uri"])
        expected = sequence_root.rstrip("/") + "/reconstruction_"
        if run_bucket != bucket:
            raise ValueError(f"{candidate['sequence']}: reconstruction bucket differs")
        if run_prefix.startswith(expected):
            relative_prefix = cleanup._relative(sequence_root, run_prefix) + "/"
            reconstruction_prefixes.append(relative_prefix)
            retained_runs.append({**run, "relative_prefix": relative_prefix})

    deletions = []
    retained = []
    for item in objects:
        relative = cleanup._relative(sequence_root, item["key"])
        category = _classify_deletion(relative, reconstruction_prefixes)
        if category and not cleanup._is_protected(relative):
            deletions.append({**item, "relative_path": relative, "category": category})
        else:
            retained.append({**item, "relative_path": relative})
    retained_nonmetric = [
        item for item in retained
        if not item["relative_path"].startswith("metrics/")
    ]
    protected = [
        item for item in retained_nonmetric
        if cleanup._is_protected(item["relative_path"])
    ]
    return {
        **candidate,
        "sequence_root": sequence_root,
        "storage_identity_sha256": cleanup._identity(objects),
        "retained_identity_sha256": cleanup._identity(retained_nonmetric),
        "protected_identity_sha256": cleanup._identity(protected),
        "object_count_before": len(objects),
        "bytes_before": sum(item["size"] for item in objects),
        "deletions": deletions,
        "deletion_object_count": len(deletions),
        "reclaimable_bytes": sum(item["size"] for item in deletions),
        "retained_object_count": len(retained),
        "retained_bytes": sum(item["size"] for item in retained),
        "protected_objects": protected,
        "reconstruction_runs": retained_runs,
    }


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {name: 0 for name in ("p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100")}
    ordered = sorted(values)
    result = {}
    for name, quantile in (
        ("p0", 0), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75),
        ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("p100", 1),
    ):
        result[name] = ordered[round((len(ordered) - 1) * quantile)]
    return result


def _human_bytes(value: int, *, binary: bool) -> str:
    unit = 1024 if binary else 1000
    names = ("B", "KiB", "MiB", "GiB", "TiB", "PiB") if binary else (
        "B", "kB", "MB", "GB", "TB", "PB",
    )
    amount = float(value)
    for name in names:
        if abs(amount) < unit or name == names[-1]:
            return f"{amount:.2f} {name}"
        amount /= unit
    raise AssertionError("unreachable")


def _storage_summary(records: list[dict]) -> dict:
    category = defaultdict(lambda: {"object_count": 0, "bytes": 0})
    reason = defaultdict(lambda: {"sequence_count": 0, "object_count": 0, "bytes": 0})
    for record in records:
        reason_row = reason[record["candidate_reason"]]
        reason_row["sequence_count"] += 1
        reason_row["object_count"] += record["deletion_object_count"]
        reason_row["bytes"] += record["reclaimable_bytes"]
        for item in record["deletions"]:
            category[item["category"]]["object_count"] += 1
            category[item["category"]]["bytes"] += item["size"]
    reclaimed = sum(record["reclaimable_bytes"] for record in records)
    retained = sum(record["retained_bytes"] for record in records)
    before = sum(record["bytes_before"] for record in records)
    for rows in (category, reason):
        for row in rows.values():
            row["decimal"] = _human_bytes(row["bytes"], binary=False)
            row["binary"] = _human_bytes(row["bytes"], binary=True)
    return {
        "candidate_sequence_count": len(records),
        "deletion_object_count": sum(record["deletion_object_count"] for record in records),
        "exact_reclaimable_logical_bytes": reclaimed,
        "exact_reclaimable_decimal": _human_bytes(reclaimed, binary=False),
        "exact_reclaimable_binary": _human_bytes(reclaimed, binary=True),
        "selected_bytes_before": before,
        "selected_bytes_before_decimal": _human_bytes(before, binary=False),
        "selected_bytes_before_binary": _human_bytes(before, binary=True),
        "expected_retained_logical_bytes": retained,
        "expected_retained_decimal": _human_bytes(retained, binary=False),
        "expected_retained_binary": _human_bytes(retained, binary=True),
        "by_category": dict(sorted(category.items())),
        "by_candidate_reason": dict(sorted(reason.items())),
        "per_sequence_reclaimable_byte_percentiles": _percentiles([
            record["reclaimable_bytes"] for record in records
        ]),
        "storage_accounting_note": (
            "Object sizes are exact logical bytes. CSS quota updates may be delayed "
            "and may include overhead not represented by object listings."
        ),
    }


def audit(
    campaign_name: str,
    *,
    expected_campaign_id: int,
    quota_retry_plan: Path,
    db_path: str,
    config: dict,
    max_workers: int = 4,
) -> dict:
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or int(campaign["id"]) != int(expected_campaign_id):
        raise ValueError("campaign name and --expected-campaign-id do not match")
    if campaign["campaign_type"] not in ("BACKLOG_REPROCESSING", "REMEDIATION"):
        raise ValueError("cleanup is limited to backlog/remediation campaigns")
    if campaign["status"] not in ("FROZEN", "RUNNING"):
        raise ValueError("campaign must be FROZEN or RUNNING")
    quota_plan, quota_sha = _load_quota_plan(quota_retry_plan, campaign)
    snapshot = _database_snapshot(campaign, db_path=db_path)
    candidates = _select_candidates(campaign, quota_plan, snapshot)
    client, bucket, output_root, output_root_uri = _output_root(campaign, config)
    records = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(
                _inspect_candidate, candidate, client=client, bucket=bucket,
                output_root=output_root,
            ): candidate["sequence"]
            for candidate in candidates
        }
        for completed, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"Enumerated {completed}/{len(futures)} cleanup candidate(s)",
                    file=sys.stderr,
                    flush=True,
                )
    records.sort(key=lambda item: item["sequence"])
    report = {
        "schema": SCHEMA,
        "generated_at": _now(),
        "campaign_id": campaign["id"],
        "campaign_name": campaign["name"],
        "campaign_status": campaign["status"],
        "campaign_pipeline_version": campaign["pipeline_version"],
        "dataset": campaign["dataset"],
        "output_root_uri": output_root_uri,
        "quota_retry_plan": str(quota_retry_plan.resolve()),
        "quota_retry_plan_sha256": quota_sha,
        "quota_retry_count": len(quota_plan["preprocess_retries"]),
        "storage": _storage_summary(records),
        "records": records,
    }
    report["audit_sha256"] = _payload_sha256(report)
    return report


def _validate_audit(report: dict) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("unsupported cleanup audit schema")
    claimed = report.get("audit_sha256")
    unsigned = {key: value for key, value in report.items() if key != "audit_sha256"}
    if claimed != _payload_sha256(unsigned):
        raise ValueError("cleanup audit SHA-256 does not match its contents")


def _optional_json(client, bucket: str, key: str) -> dict | None:
    try:
        payload, _response = _get_with_backoff(client, bucket, key)
    except KeyError:
        return None
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if code not in ("404", "NoSuchKey", "NotFound") and status != 404:
            raise
        return None
    return json.loads(payload)


def _put_json(client, bucket: str, key: str, payload: dict) -> None:
    client.put_object(
        Bucket=bucket, Key=key,
        Body=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        ContentType="application/json",
    )


def _verify_protected(current: list[dict], record: dict) -> None:
    protected = []
    for item in current:
        relative = cleanup._relative(record["sequence_root"], item["key"])
        if relative.startswith("metrics/"):
            continue
        if cleanup._is_protected(relative):
            protected.append(item)
    if cleanup._identity(protected) != record["protected_identity_sha256"]:
        raise RuntimeError(f"{record['sequence']}: protected storage changed")


def _verify_audited_storage(
    current: list[dict], record: dict, *, allow_missing_deletions: bool,
) -> None:
    """Validate exact deletion/protected objects while preserving unknown data."""
    current_by_key = {item["key"]: item for item in current}
    audited_deletions = {item["key"]: item for item in record["deletions"]}
    reconstruction_prefixes = [
        item["relative_prefix"] for item in record["reconstruction_runs"]
    ]
    for key, expected in audited_deletions.items():
        actual = current_by_key.get(key)
        if actual is None:
            if allow_missing_deletions:
                continue
            raise RuntimeError(f"{record['sequence']}: audited deletion key is missing")
        if (
            int(actual["size"]) != int(expected["size"])
            or actual.get("etag", "") != expected.get("etag", "")
        ):
            raise RuntimeError(f"{record['sequence']}: audited deletion key changed")
    for item in current:
        relative = cleanup._relative(record["sequence_root"], item["key"])
        category = _classify_deletion(relative, reconstruction_prefixes)
        if (
            category
            and not cleanup._is_protected(relative)
            and item["key"] not in audited_deletions
        ):
            raise RuntimeError(
                f"{record['sequence']}: new deletable object appeared after audit"
            )
    _verify_protected(current, record)


def _verify_retained(current: list[dict], record: dict) -> None:
    _verify_protected(current, record)


def _preserve_evidence(
    report: dict, record: dict, *, client, bucket: str,
) -> list[dict]:
    manifests = []
    deletion_by_key = {item["key"]: item for item in record["deletions"]}
    for run in record["reconstruction_runs"]:
        run_prefix = cleanup._join(record["sequence_root"], run["relative_prefix"])
        destination_root = cleanup._join(
            record["sequence_root"], "metrics", "reconstruction",
            str(run["request_id"]),
        )
        files = []
        missing = []
        for original, destination_name in cleanup.FALLBACK_EVIDENCE.items():
            source_key = cleanup._join(run_prefix, original)
            source = deletion_by_key.get(source_key)
            if source is None:
                missing.append(original)
                continue
            payload, response = _get_with_backoff(client, bucket, source_key)
            if len(payload) != int(source["size"]):
                raise RuntimeError(f"metric source size changed: {source_key}")
            destination = cleanup._join(destination_root, destination_name)
            client.put_object(Bucket=bucket, Key=destination, Body=payload)
            copied, copied_response = _get_with_backoff(client, bucket, destination)
            if copied != payload:
                raise RuntimeError(f"copied metric differs: {destination}")
            files.append({
                "path": destination_name,
                "original_key": source_key,
                "size": len(payload),
                "source_etag": cleanup._etag(response),
                "etag": cleanup._etag(copied_response),
                "sha256": cleanup._sha256(payload),
            })
        manifest = {
            "schema": cleanup.METRICS_SCHEMA,
            "created_at": _now(),
            "stage": "reconstruction",
            "request_id": run["request_id"],
            "campaign_id": report["campaign_id"],
            "campaign_name": report["campaign_name"],
            "reconstruction_run_id": run["reconstruction_run_id"],
            "workflow": run.get("workflow_name"),
            "pipeline_version": run.get("pipeline_version"),
            "source_uri": run["output_uri"],
            "categorized_failure_reason": cleanup.categorized_failure_reason(
                run.get("details")
            ),
            "files": files,
            "missing_optional_evidence": missing,
        }
        _put_json(
            client, bucket, cleanup._join(destination_root, "manifest.json"), manifest,
        )
        manifests.append(manifest)
    return manifests


def _is_complete_for_audit(complete: dict | None, report: dict) -> bool:
    return bool(
        complete
        and complete.get("schema") == COMPLETE_SCHEMA
        and complete.get("audit_sha256") == report["audit_sha256"]
        and complete.get("status") == "COMPLETE"
    )


def _already_complete_result(
    complete: dict, record: dict, current: list[dict], *, client, bucket: str,
) -> dict:
    _verify_retained(current, record)
    remaining = {item["key"] for item in current}
    if any(item["key"] in remaining for item in record["deletions"]):
        raise RuntimeError(f"{record['sequence']}: completed deletion keys reappeared")
    _remove_pending_marker(
        client, bucket, cleanup._join(record["sequence_root"], PENDING_NAME),
    )
    return {
        **complete,
        "status": "ALREADY_COMPLETE",
        "idempotent_noop": True,
        "actual_deleted_object_count_total": record["deletion_object_count"],
        "actual_reclaimed_logical_bytes_total": record["reclaimable_bytes"],
        "actual_deleted_object_count_this_run": 0,
        "actual_reclaimed_logical_bytes_this_run": 0,
    }


def _apply_record(report: dict, record: dict, *, client, bucket: str) -> dict:
    pending_key = cleanup._join(record["sequence_root"], PENDING_NAME)
    complete_key = cleanup._join(record["sequence_root"], COMPLETE_NAME)
    complete = _optional_json(client, bucket, complete_key)
    if _is_complete_for_audit(complete, report):
        current = _list_relevant_with_backoff(client, bucket, record)
        return _already_complete_result(
            complete, record, current, client=client, bucket=bucket,
        )

    current = _list_relevant_with_backoff(client, bucket, record)
    pending = _optional_json(client, bucket, pending_key)
    valid_pending = bool(
        pending and pending.get("audit_sha256") == report["audit_sha256"]
    )
    current_keys = {item["key"] for item in current}
    missing_deletions = [
        item for item in record["deletions"] if item["key"] not in current_keys
    ]
    if missing_deletions and not valid_pending:
        # CSS may briefly return 404 for a completion marker written by a
        # concurrently finishing worker just before this invocation began.
        for attempt in range(4):
            complete = _optional_json(client, bucket, complete_key)
            if _is_complete_for_audit(complete, report):
                return _already_complete_result(
                    complete, record, current, client=client, bucket=bucket,
                )
            if attempt < 3:
                time.sleep(0.5 * (2.0 ** attempt))
    _verify_audited_storage(
        current, record, allow_missing_deletions=valid_pending,
    )

    pending_payload = {
        "schema": PENDING_SCHEMA,
        "status": "PENDING",
        "created_at": pending.get("created_at") if pending else _now(),
        "audit_sha256": report["audit_sha256"],
        "campaign_id": report["campaign_id"],
        "sequence": record["sequence"],
        "request_id": record["request_id"],
        "expected_reclaimable_bytes": record["reclaimable_bytes"],
    }
    _put_json(client, bucket, pending_key, pending_payload)
    metrics = _preserve_evidence(report, record, client=client, bucket=bucket)

    current_before_delete = {item["key"]: item for item in current}
    remaining_deletions = [
        item for item in record["deletions"] if item["key"] in current_before_delete
    ]
    current_after = _delete_and_verify(
        client, bucket, record, [item["key"] for item in remaining_deletions],
    )
    _verify_retained(current_after, record)
    result = {
        "schema": COMPLETE_SCHEMA,
        "status": "COMPLETE",
        "completed_at": _now(),
        "audit_sha256": report["audit_sha256"],
        "campaign_id": report["campaign_id"],
        "campaign_name": report["campaign_name"],
        "sequence": record["sequence"],
        "request_id": record["request_id"],
        "candidate_reason": record["candidate_reason"],
        "expected_deleted_object_count": record["deletion_object_count"],
        "actual_deleted_object_count_total": record["deletion_object_count"],
        "actual_deleted_object_count_this_run": len(remaining_deletions),
        "expected_reclaimed_logical_bytes": record["reclaimable_bytes"],
        "actual_reclaimed_logical_bytes_total": record["reclaimable_bytes"],
        "actual_reclaimed_logical_bytes_this_run": sum(
            item["size"] for item in remaining_deletions
        ),
        "deleted_keys_sha256": _payload_sha256([
            item["key"] for item in record["deletions"]
        ]),
        "protected_identity_sha256": record["protected_identity_sha256"],
        "retained_attempt_metrics": metrics,
    }
    _put_json(client, bucket, complete_key, result)
    _remove_pending_marker(client, bucket, pending_key)
    return result


def apply_audit(
    report: dict,
    *,
    quota_retry_plan: Path,
    db_path: str,
    config: dict,
    sequences: set[str] | None = None,
    max_workers: int = 4,
) -> dict:
    _validate_audit(report)
    campaign = db.get_campaign(report["campaign_name"], db_path=db_path)
    if not campaign or int(campaign["id"]) != int(report["campaign_id"]):
        raise ValueError("campaign identity changed after audit")
    quota_plan, quota_sha = _load_quota_plan(quota_retry_plan, campaign)
    if quota_sha != report["quota_retry_plan_sha256"]:
        raise ValueError("quota retry plan changed after audit")
    current = _select_candidates(
        campaign, quota_plan, _database_snapshot(campaign, db_path=db_path),
    )
    current_identity = {
        item["sequence"]: (
            item["request_id"], item["request_status"], item["candidate_reason"]
        ) for item in current
    }
    audited_identity = {
        item["sequence"]: (
            item["request_id"], item["request_status"], item["candidate_reason"]
        ) for item in report["records"]
    }
    if current_identity != audited_identity:
        raise RuntimeError("campaign candidate state changed after audit")
    selected = [
        item for item in report["records"]
        if sequences is None or item["sequence"] in sequences
    ]
    if sequences is not None and {item["sequence"] for item in selected} != sequences:
        missing = sorted(sequences - {item["sequence"] for item in selected})
        raise ValueError(f"requested canary sequence is absent from audit: {missing}")
    client, bucket, _root_prefix, _uri = _output_root(campaign, config)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(
                _apply_record, report, record, client=client, bucket=bucket,
            ): record["sequence"]
            for record in selected
        }
        try:
            for completed_count, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if completed_count % 25 == 0 or completed_count == len(selected):
                    print(
                        f"Applied {completed_count}/{len(selected)} cleanup candidate(s)",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception:
            for future in futures:
                future.cancel()
            raise
    results.sort(key=lambda item: item["sequence"])
    completed = [item for item in results if item["status"] == "COMPLETE"]
    already = [item for item in results if item["status"] == "ALREADY_COMPLETE"]
    expected = sum(record["reclaimable_bytes"] for record in selected)
    verified = sum(item["actual_reclaimed_logical_bytes_total"] for item in results)
    actual_this_run = sum(
        item["actual_reclaimed_logical_bytes_this_run"] for item in results
    )
    return {
        "schema": APPLY_SCHEMA,
        "completed_at": _now(),
        "audit_sha256": report["audit_sha256"],
        "campaign_id": report["campaign_id"],
        "selected_sequence_count": len(selected),
        "completed_sequence_count": len(completed),
        "already_complete_sequence_count": len(already),
        "expected_reclaimed_logical_bytes": expected,
        "verified_reclaimed_logical_bytes": verified,
        "actual_reclaimed_logical_bytes_this_run": actual_this_run,
        "variance_bytes": verified - expected,
        "results": results,
    }


def _apply_result_path(report_path: Path, *, canary: bool) -> Path:
    suffix = ".canary.apply.json" if canary else ".apply.json"
    return report_path.with_name(report_path.stem + suffix)


def _summary(report: dict, *, report_path: Path, applied: dict | None = None) -> dict:
    result = {
        "report": str(report_path),
        "audit_sha256": report["audit_sha256"],
        "campaign_id": report["campaign_id"],
        **report["storage"],
        "applied": applied is not None,
    }
    if applied:
        result["apply"] = {
            key: value for key, value in applied.items() if key != "results"
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--expected-campaign-id", type=int, required=True)
    parser.add_argument("--quota-retry-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--expected-candidate-count", type=int)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config = load_config(MV_HOI_DIR)
    if not args.apply:
        if args.sequence:
            parser.error("--sequence is only valid with --apply")
        report = audit(
            args.campaign,
            expected_campaign_id=args.expected_campaign_id,
            quota_retry_plan=args.quota_retry_plan,
            db_path=args.db,
            config=config,
            max_workers=args.max_workers,
        )
        if (
            args.expected_candidate_count is not None
            and report["storage"]["candidate_sequence_count"]
            != args.expected_candidate_count
        ):
            raise RuntimeError(
                "candidate count differs from --expected-candidate-count: "
                f"{report['storage']['candidate_sequence_count']} != "
                f"{args.expected_candidate_count}"
            )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(_summary(report, report_path=args.report), indent=2, sort_keys=True))
        return

    require_submit_authority("delete stale pending backlog intermediates")
    report = json.loads(args.report.read_text())
    if (
        args.expected_candidate_count is not None
        and report["storage"]["candidate_sequence_count"]
        != args.expected_candidate_count
    ):
        raise RuntimeError("audit candidate count differs from expected count")
    if (
        report.get("campaign_name") != args.campaign
        or int(report.get("campaign_id", -1)) != args.expected_campaign_id
    ):
        raise ValueError("audit campaign identity does not match command")
    lock_path = state_path("locks") / "mv_hoi_campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("campaign orchestration lock is already held") from exc
        applied = apply_audit(
            report,
            quota_retry_plan=args.quota_retry_plan,
            db_path=args.db,
            config=config,
            sequences=set(args.sequence) or None,
            max_workers=args.max_workers,
        )
    result_path = _apply_result_path(args.report, canary=bool(args.sequence))
    result_path.write_text(json.dumps(applied, indent=2, sort_keys=True) + "\n")
    summary = _summary(report, report_path=args.report, applied=applied)
    summary["apply_report"] = str(result_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
