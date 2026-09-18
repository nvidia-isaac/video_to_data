"""Enqueue, dispatch, and reconcile frozen MV-HOI processing campaigns."""

from __future__ import annotations

try:
    from .registry_versions import image_registry
except ImportError:
    from registry_versions import image_registry

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Iterable

from botocore.exceptions import ClientError, ReadTimeoutError, ResponseStreamingError

try:
    from . import cleanup_intermediates, db, export as export_controller, submit
    from .campaign_inventory import _client, _list, _parse_swift_url
    from .config_utils import (
        PREPROCESS_PIPELINE, RECON_PIPELINE, REVALIDATION_PIPELINE,
        REVALIDATION_EXPORT_RETRY_WORKFLOW, REVALIDATION_WORKFLOW,
        get_backlog_promotion_settings, get_cleanup_settings,
        get_workflow_cfg, load_config,
    )
    from .query import (
        _ambiguous_submit_not_found_grace_expired,
        _failure_detail,
        _classify_file_not_found_failures,
        _is_ambiguous_submit_placeholder,
        handle_campaign_infrastructure_failure,
        export_task_failure_info,
        export_task_names,
        is_retryable_infrastructure_failure,
        osmo_query,
        refresh_workflow_states,
    )
    from .pool_selection import PoolSelector, load_active_counts
    from .runtime import MV_HOI_DIR, require_submit_authority
    from .export_commit import (
        CandidateCleanupError,
        cleanup_rejected_export_candidate,
        cleanup_promoted_export_candidate,
        read_remote_export_rejection,
    )
    from .canary_validation import (
        validate_accuracy_failure_evidence,
        validate_foundation_comparison_evidence,
    )
except ImportError:
    import cleanup_intermediates, db, export as export_controller, submit
    from campaign_inventory import _client, _list, _parse_swift_url
    from config_utils import (
        PREPROCESS_PIPELINE, RECON_PIPELINE, REVALIDATION_PIPELINE,
        REVALIDATION_EXPORT_RETRY_WORKFLOW, REVALIDATION_WORKFLOW,
        get_backlog_promotion_settings, get_cleanup_settings,
        get_workflow_cfg, load_config,
    )
    from query import (
        _ambiguous_submit_not_found_grace_expired,
        _failure_detail,
        _classify_file_not_found_failures,
        _is_ambiguous_submit_placeholder,
        handle_campaign_infrastructure_failure,
        export_task_failure_info,
        export_task_names,
        is_retryable_infrastructure_failure,
        osmo_query,
        refresh_workflow_states,
    )
    from pool_selection import PoolSelector, load_active_counts
    from runtime import MV_HOI_DIR, require_submit_authority
    from export_commit import (
        CandidateCleanupError,
        cleanup_rejected_export_candidate,
        cleanup_promoted_export_candidate,
        read_remote_export_rejection,
    )
    from canary_validation import (
        validate_accuracy_failure_evidence,
        validate_foundation_comparison_evidence,
    )


ACTIVE = ("RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN")
IDENTITY_CONFIRMATION_READS = 2
LIST_THROTTLE_RETRIES = 6
COMMIT_VISIBILITY_RETRIES = 6
COMMIT_VISIBILITY_BASE_DELAY_SECONDS = 1.0
PUBLISH_COPY_WORKERS = 64
COMMIT_HASH_WORKERS = 64
COMMIT_HASH_READ_RETRIES = 4
COMMIT_HASH_RANGE_BYTES = 64 * 1024**2
SINGLE_COPY_MAX_BYTES = 5 * 1024**3
MULTIPART_COPY_PART_BYTES = 512 * 1024**2
REVALIDATION_QC_REPAIR_ROOT = "_recovery/20260801_revalidation_human_qc"
REVALIDATION_RESULT_PENDING = "result_reconciliation_pending"
_NON_DATA_PREREQUISITE_PREFIXES = (
    "no current successful preprocessing run",
    "current preprocessing run has no output URI",
    "current preprocessing run does not match campaign lineage",
    "current preprocessing run is outside the canonical campaign prefix",
    "current preprocessing output uses an unexpected bucket",
)
DEFAULT_CAMPAIGN_QUERY_WORKERS = int(
    os.environ.get("MV_HOI_CAMPAIGN_QUERY_WORKERS", "16")
)
DEFAULT_CAMPAIGN_SUBMIT_WORKERS = int(
    os.environ.get("MV_HOI_CAMPAIGN_SUBMIT_WORKERS", "8")
)
DEFAULT_REVALIDATION_PUBLISH_WORKERS = int(
    os.environ.get("MV_HOI_REVALIDATION_PUBLISH_WORKERS", "4")
)


def _retry_is_ready(request: dict, *, now: datetime | None = None) -> bool:
    try:
        parameters = json.loads(request.get("parameters_json") or "{}")
    except (TypeError, ValueError):
        return False
    value = parameters.get("retry_not_before")
    if not value:
        return True
    try:
        not_before = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    if not_before.tzinfo is None:
        not_before = not_before.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) >= not_before


def _blacklist_campaign_failure(
    request: dict, campaign: dict, details: str, *, db_path: str,
) -> None:
    """Put a terminal data/revalidation failure in the shared timeout box."""
    normalized = " ".join(str(details).split())
    db.upsert_blacklisted_sequence(
        campaign["dataset"], request["sequence_name"],
        reason=f"{request['stage']}_failed: {normalized}",
        created_by=f"campaign:{campaign['name']}", db_path=db_path,
    )


def _resolve_blacklist_after_committed_export(
    request: dict, campaign: dict, *, db_path: str,
) -> bool:
    """Clear the active timeout box only after publication has committed."""
    return db.remove_blacklisted_sequence(
        campaign["dataset"], request["sequence_name"], db_path=db_path,
    )


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _join_url(*parts: str) -> str:
    root, *rest = parts
    suffix = "/".join(part.strip("/") for part in rest if part)
    return root.rstrip("/") + ("/" + suffix if suffix else "")


def _revalidation_repair_marker_url(
    request: dict, campaign: dict, dataset_cfg: dict,
) -> str:
    return _join_url(
        dataset_cfg["swift_base"], REVALIDATION_QC_REPAIR_ROOT,
        campaign["name"], f"request_{int(request['id'])}", "pending.json",
    )


def _revalidation_repair_active(
    request: dict, campaign: dict, dataset_cfg: dict,
) -> bool:
    if not dataset_cfg.get("swift_base"):
        return False
    client, bucket, key = _client(
        _revalidation_repair_marker_url(request, campaign, dataset_cfg)
    )
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        response["Body"].close()
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"NoSuchKey", "404"} or status == 404:
            return False
        raise


def load_frozen_inventory(campaign: dict, path: str | Path) -> dict:
    path = Path(path)
    actual = _sha256_file(path)
    if actual != campaign["inventory_sha256"]:
        raise ValueError(
            f"Inventory SHA-256 mismatch: campaign={campaign['inventory_sha256']}, file={actual}"
        )
    inventory = json.loads(path.read_text())
    if inventory.get("dataset") != campaign["dataset"]:
        raise ValueError("Inventory dataset does not match campaign")
    if int(inventory.get("sequence_count", -1)) != len(inventory.get("sequences", [])):
        raise ValueError("Inventory sequence_count does not reconcile")
    names = [record["sequence"] for record in inventory["sequences"]]
    if len(names) != len(set(names)):
        raise ValueError("Inventory contains duplicate sequence names")
    return inventory


def enqueue_campaign(campaign_name: str, inventory_path: str | Path, *, db_path: str) -> dict:
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or campaign["status"] not in ("FROZEN", "RUNNING"):
        raise ValueError("Campaign must be frozen before enqueue")
    inventory = load_frozen_inventory(campaign, inventory_path)
    campaign = db.set_campaign_inventory_count(
        campaign["id"], int(inventory["sequence_count"]), db_path=db_path,
    )
    expected_kind = {
        "LEGACY_REVALIDATION": "legacy_revalidation",
        "BACKLOG_REPROCESSING": "backlog_reprocessing",
        "REMEDIATION": None,
    }[campaign["campaign_type"]]
    if expected_kind and inventory.get("kind") != expected_kind:
        raise ValueError("Inventory kind does not match campaign type")
    created = 0
    existing = 0
    blocked = 0
    pending_inserts: list[dict] = []
    current = {
        row["sequence_name"]: row
        for row in db.list_stage_requests(campaign=campaign_name, db_path=db_path)
    }
    for record_index, record in enumerate(inventory["sequences"]):
        queue_priority = record.get("queue_priority", 0)
        if (
            isinstance(queue_priority, bool)
            or not isinstance(queue_priority, int)
            or queue_priority < 0
        ):
            raise ValueError(
                f"Inventory sequence {record['sequence']} has invalid queue_priority"
            )
        if record["sequence"] in current:
            existing += 1
            continue
        if campaign["campaign_type"] == "LEGACY_REVALIDATION":
            stage = "revalidation"
            cohort = record["cohort"]
            request_status = "PENDING"
            blocker = None
        else:
            cohort = record.get("cohort")
            if cohort is None:
                cohort = (
                    "CANARY"
                    if campaign["phase"] == "CANARY" and record_index < 20
                    else "BULK"
                )
            recommended = record.get("recommended_stage")
            if recommended not in ("preprocess", "reconstruction"):
                raise ValueError(
                    f"Backlog sequence {record['sequence']} must start at preprocessing "
                    "or reconstruction; "
                    f"found recommended_stage={recommended!r}"
                )
            stage = recommended
            if stage == "preprocess":
                if record.get("raw_input_objects"):
                    request_status, blocker = "PENDING", None
                else:
                    request_status, blocker = "BLOCKED", "MISSING_RAW_INPUT"
            else:
                preprocess = db.get_latest_successful_stage_run(
                    record["sequence"], campaign["dataset"], PREPROCESS_PIPELINE,
                    db_path=db_path,
                )
                expected_preprocess_id = record.get(
                    "validated_preprocess_run_id", record.get("preprocess_run_id")
                )
                if (
                    not preprocess
                    or expected_preprocess_id is None
                    or int(preprocess["stage_run_id"]) != int(expected_preprocess_id)
                ):
                    raise ValueError(
                        f"Backlog sequence {record['sequence']} must start at "
                        "preprocessing unless reconstruction reuse matches its "
                        "validated preprocess run"
                    )
                request_status, blocker = "PENDING", None
        inventory_uri = campaign["inventory_uri"]
        if campaign["campaign_type"] == "LEGACY_REVALIDATION":
            inventory_uri = _join_url(
                inventory_uri.rsplit("/", 1)[0],
                "revalidation_records", f"{record['sequence']}.json",
            )
        parameters = {
            "route": record.get("route"),
            "recommended_stage": record.get("recommended_stage"),
            "output_uri": _join_url(campaign["output_uri"], record["sequence"]),
            "configuration_uri": campaign["configuration_uri"],
            "configuration_sha256": campaign["configuration_sha256"],
            "inventory_uri": inventory_uri,
        }
        if campaign["campaign_type"] == "LEGACY_REVALIDATION":
            parameters["work_output_layout"] = "request_scoped_v1"
        elif stage == "reconstruction":
            parameters.update({
                "preprocess_request_id": preprocess.get("request_id"),
                "preprocess_run_id": int(preprocess["stage_run_id"]),
                "preprocess_output_uri": preprocess["output_uri"],
                "preprocess_reused": True,
            })
        pending_inserts.append({
            "sequence_name": record["sequence"], "dataset": campaign["dataset"],
            "stage": stage, "pipeline_version": campaign["pipeline_version"],
            "trigger": "MIGRATION", "requested_by": campaign["created_by"],
            "reason": campaign["campaign_type"].lower(), "cohort": cohort,
            "queue_priority": queue_priority,
            "parameters": parameters, "source_manifest": record,
            "status": request_status, "blocked_reason": blocker,
        })
        created += 1
        blocked += int(request_status == "BLOCKED")
    db.create_campaign_stage_requests_batch(
        campaign["id"], pending_inserts, db_path=db_path,
    )
    total = len(inventory["sequences"])
    if created + existing != total:
        raise RuntimeError("Campaign enqueue count did not reconcile")
    return {"inventory": total, "created": created, "existing": existing, "blocked": blocked}


def _object_map(client, bucket: str, root: str) -> dict[str, dict]:
    root = root.rstrip("/")
    for attempt in range(LIST_THROTTLE_RETRIES):
        try:
            objects = _list(client, bucket, root)
            return {
                item["key"][len(root) + 1:]: {
                    "size": int(item["size"]), "etag": item.get("etag") or "",
                }
                for item in objects
                if item["key"].startswith(root + "/")
            }
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if status != 429 and code not in {"429", "TooManyRequests"}:
                raise
            if attempt + 1 == LIST_THROTTLE_RETRIES:
                raise
            # Swift can throttle a large, otherwise read-only prefix listing.
            # Retry the snapshot with bounded backoff rather than falling back
            # to thousands of individual HEAD requests.
            time.sleep(min(2 ** attempt, 16))
    raise AssertionError("unreachable")


def _head_identity(client, bucket: str, key: str) -> dict[str, object]:
    """Read a CSS object identity, tolerating Swift's broken HEAD response."""
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        response_metadata = exc.response.get("ResponseMetadata", {})
        error = exc.response.get("Error", {})
        if response_metadata.get("HTTPStatusCode") != 400 and str(
            error.get("Code", "")
        ) not in {"400", "BadRequest"}:
            raise
        response = client.get_object(Bucket=bucket, Key=key, Range="bytes=0-0")
        content_range = str(response.get("ContentRange", ""))
        match = re.fullmatch(r"bytes\s+\d+-\d+/(\d+)", content_range)
        if match is None:
            raise ValueError(
                f"Invalid ranged GET Content-Range for {key}"
            ) from exc
        body = response.get("Body")
        if body is not None:
            body.read(1)
            close = getattr(body, "close", None)
            if close is not None:
                close()
        return {
            "size": int(match.group(1)),
            "etag": str(response.get("ETag", "")).strip('"'),
        }
    return {
        "size": int(response["ContentLength"]),
        "etag": str(response.get("ETag", "")).strip('"'),
    }


def _matches_frozen_identity(
    client, bucket: str, key: str, expected: dict[str, object],
) -> bool:
    """Confirm a mismatched Swift identity before treating it as source drift.

    CSS occasionally returns a stale successful HEAD response even though an
    immediate ranged GET/HEAD observes the frozen object. Callers accept an
    initial match directly. An initial mismatch is accepted only when two fresh
    reads both reproduce the expected size and ETag; any disagreement remains a
    hard failure.
    """
    return all(
        _head_identity(client, bucket, key) == expected
        for _ in range(IDENTITY_CONFIRMATION_READS)
    )


def validate_frozen_objects(request: dict, dataset_cfg: dict, client=None) -> None:
    manifest = json.loads(request["source_manifest_json"])
    if client is None:
        client, bucket, base = _client(dataset_cfg["swift_base"])
    else:
        _, bucket, base = _parse_swift_url(dataset_cfg["swift_base"])
    pipeline = dataset_cfg["pipelines"][REVALIDATION_PIPELINE]
    for field, directory in (
        ("data_output_objects", pipeline["input_path"]),
        ("data_export_objects", pipeline["legacy_export_path"]),
    ):
        expected = {
            item["relative_path"]: {
                "size": int(item["size"]), "etag": item.get("etag") or "",
            }
            for item in manifest.get(field, [])
        }
        if not expected:
            raise ValueError(f"Frozen manifest contains no {field}")
        root = f"{base}/{directory}/{request['sequence_name']}"
        # A prefix listing already returns the size and ETag for every object.
        # Compare that snapshot first so legacy per-frame exports do not issue
        # thousands of serial HEAD requests.  Preserve the stricter repeated
        # ranged-GET/HEAD confirmation for the uncommon mismatch case because
        # CSS can transiently serve stale object metadata.
        actual_objects = _object_map(client, bucket, root)
        missing: list[str] = []
        changed: list[str] = []
        for relative_path, identity in expected.items():
            key = f"{root.rstrip('/')}/{relative_path}"
            actual = actual_objects.get(relative_path)
            if actual is None:
                missing.append(relative_path)
                continue
            if actual != identity and not _matches_frozen_identity(
                client, bucket, key, identity,
            ):
                changed.append(relative_path)
        if missing or changed:
            raise ValueError(
                f"Frozen {directory} changed: missing={missing[:5]}, "
                f"changed={changed[:5]}"
            )


def validate_frozen_raw_input(request: dict, dataset_cfg: dict, client=None) -> None:
    manifest = json.loads(request["source_manifest_json"] or "{}")
    expected = {
        item["relative_path"]: {
            "size": int(item["size"]), "etag": item.get("etag") or "",
        }
        for item in manifest.get("raw_input_objects", [])
    }
    if not expected:
        raise ValueError("Frozen backlog entry has no raw input objects")
    if client is None:
        client, bucket, base = _client(dataset_cfg["swift_base"])
    else:
        _, bucket, base = _parse_swift_url(dataset_cfg["swift_base"])
    raw_path = dataset_cfg["pipelines"][PREPROCESS_PIPELINE]["input_path"]
    root = f"{base}/{raw_path}/{request['sequence_name']}"
    actual = _object_map(client, bucket, root)
    if actual != expected:
        raise ValueError("Frozen raw recording identity changed after cutover")


def _workflow_name(version: str, request_id: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"v2d_revalidation_{version.replace('.', '-')}_{request_id}_{stamp}"


def _legacy_pose_provenance(request: dict, *, db_path: str) -> tuple[str, str]:
    """Resolve the frozen legacy reconstruction version and its object frame."""
    manifest = json.loads(request.get("source_manifest_json") or "{}")
    version = manifest.get("reconstruction_pipeline_version")
    run_id = manifest.get("reconstruction_run_id")
    if run_id is not None:
        run = db.get_stage_run(
            int(run_id), stage=db.RECONSTRUCTION_STAGE, db_path=db_path,
        )
        if run is None:
            raise ValueError(f"Frozen reconstruction run {run_id} no longer exists")
        run_version = run.get("pipeline_version")
        if version is not None and version != run_version:
            raise ValueError(
                "Frozen reconstruction pipeline version differs from the database"
            )
        version = run_version
    if not isinstance(version, str):
        raise ValueError("Revalidation source lacks a reconstruction pipeline version")
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"Invalid legacy reconstruction version: {version!r}")
    pose_frame = (
        "original"
        if tuple(int(value) for value in match.groups()) < (1, 4, 0)
        else "aligned"
    )
    return version, pose_frame


def _result_destination(request: dict, campaign: dict, dataset_cfg: dict) -> str:
    """Stage every result until the controller revalidates frozen CSS ETags."""
    pipeline = dataset_cfg["pipelines"][REVALIDATION_PIPELINE]
    parameters = json.loads(request.get("parameters_json") or "{}")
    request_scope = (
        [f"request_{request['id']}"]
        if parameters.get("work_output_layout") == "request_scoped_v1"
        else []
    )
    return _join_url(
        dataset_cfg["swift_base"], pipeline["work_output_path"],
        campaign["name"], request["sequence_name"], *request_scope,
        "candidate_export",
    )


def _read_remote_json(uri: str) -> tuple[object, str]:
    client, bucket, key = _client(uri)
    raw = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _expected_revalidation_canary_failure(
    request: dict, campaign: dict, dataset_cfg: dict, info: dict, *, db_path: str,
) -> dict | None:
    """Return validated expected-QC evidence, never a status-only guess."""
    failed_tasks = sorted(
        task for task, status in (info.get("tasks") or {}).items()
        if status == "FAILED"
    )
    if len(failed_tasks) != 1:
        return None
    failed_task = failed_tasks[0]
    values = _revalidation_values(
        request, campaign, dataset_cfg, db_path=db_path,
    )
    work_output_url = values["work_output_url"].rstrip("/")
    try:
        if failed_task == "check_accuracy":
            report_uri = work_output_url + "/check_accuracy/check_accuracy.json"
            segments_uri = work_output_url + "/check_accuracy/failure_segments.json"
            report, report_hash = _read_remote_json(report_uri)
            segments, segments_hash = _read_remote_json(segments_uri)
            if not isinstance(report, dict):
                return None
            valid, evidence_detail = validate_accuracy_failure_evidence(
                report, segments,
                expected_thresholds={
                    "max_chamfer_object_mm": float(values["max_chamfer_object"]),
                    "max_chamfer_human_mm": float(values["max_chamfer_human"]),
                    "max_chamfer_segment_object_mm": float(
                        values["max_chamfer_segment_object"]
                    ),
                    "max_chamfer_segment_human_mm": float(
                        values["max_chamfer_segment_human"]
                    ),
                    "min_silhouette_bbox_containment": float(
                        values["min_silhouette_bbox_containment"]
                    ),
                    "min_failure_run_frames": int(
                        values["min_accuracy_failure_run_frames"]
                    ),
                    "max_silhouette_failure_coverage": float(
                        values["max_silhouette_failure_coverage"]
                    ),
                },
            )
            if not valid:
                return None
            return {
                "canary_outcome": "EXPECTED_QC_DATA",
                "failure_category": "accuracy_limit",
                "failed_task": failed_task,
                "evidence_valid": True,
                "evidence_detail": evidence_detail,
                "evidence_uri": report_uri,
                "evidence_sha256": report_hash,
                "failure_segments_uri": segments_uri,
                "failure_segments_sha256": segments_hash,
                "failure_segment_count": report["failure_segment_count"],
                "failure_coverage": report["failure_coverage"],
                "failed_accuracy_checks": report["failed_accuracy_checks"],
                "reason": report["reason"],
            }
        if failed_task == "compare_foundation_pose":
            report_uri = (
                work_output_url
                + "/compare_foundation_pose/foundation_pose_comparison.json"
            )
            report, report_hash = _read_remote_json(report_uri)
            if not isinstance(report, dict):
                return None
            valid, evidence_detail = validate_foundation_comparison_evidence(report)
            if not valid:
                return None
            return {
                "canary_outcome": "EXPECTED_QC_DATA",
                "failure_category": "compare_foundation_pose",
                "failed_task": failed_task,
                "evidence_valid": True,
                "evidence_detail": evidence_detail,
                "evidence_uri": report_uri,
                "evidence_sha256": report_hash,
                "failures": report["failures"],
            }
    except Exception as exc:
        print(
            "  unable to validate expected revalidation canary failure for "
            f"{request['sequence_name']}: {exc}"
        )
    return None


def _revalidation_post_trim_rejection(
    request: dict, campaign: dict, dataset_cfg: dict,
) -> tuple[dict, str] | None:
    candidate = _result_destination(request, campaign, dataset_cfg)
    try:
        report = read_remote_export_rejection(candidate)
    except Exception:
        return None
    provenance = report.get("provenance") or {}
    expected = {
        "request_id": request["id"],
        "sequence_name": request["sequence_name"],
        "pipeline_version": request["pipeline_version"],
        "campaign_name": campaign["name"],
        "configuration_sha256": campaign["configuration_sha256"],
    }
    for key, value in expected.items():
        if str(provenance.get(key)) != str(value):
            raise ValueError(f"Revalidation rejection provenance mismatch for {key}")
    return report, candidate


def _published_destination(request: dict, campaign: dict) -> str:
    return _join_url(campaign["output_uri"], request["sequence_name"])


def _revalidation_values(
    request: dict, campaign: dict, dataset_cfg: dict, *, db_path: str,
) -> dict[str, str]:
    pipeline = dataset_cfg["pipelines"][REVALIDATION_PIPELINE]
    root = dataset_cfg["swift_base"]
    sequence = request["sequence_name"]
    workflow_cfg = get_workflow_cfg(
        dataset_cfg, REVALIDATION_PIPELINE, REVALIDATION_WORKFLOW,
    )
    thresholds = workflow_cfg.get("qc_thresholds", {})
    if "mv_hoi_export" in dataset_cfg.get("pipelines", {}):
        (
            max_failure_annotations,
            max_failure_coverage,
            max_silhouette_failure_coverage,
        ) = export_controller.export_qc_thresholds(dataset_cfg)
    else:
        max_failure_annotations = 10
        max_failure_coverage = 0.5
        max_silhouette_failure_coverage = 0.5
    parameters = json.loads(request.get("parameters_json") or "{}")
    request_scope = (
        [f"request_{request['id']}"]
        if parameters.get("work_output_layout") == "request_scoped_v1"
        else []
    )
    legacy_pipeline_version, legacy_pose_frame = _legacy_pose_provenance(
        request, db_path=db_path,
    )
    values = {
        "workflow_name": _workflow_name(request["pipeline_version"], request["id"]),
        "image_tag": request["pipeline_version"],
        "image_registry": image_registry(dataset_cfg.get("image_registry")),
        "sequence_name": sequence,
        "request_id": str(request["id"]),
        "campaign_name": campaign["name"],
        "source_url": _join_url(root, pipeline["input_path"], sequence),
        "legacy_export_url": _join_url(root, pipeline["legacy_export_path"], sequence),
        "inventory_url": parameters.get("inventory_uri") or campaign["inventory_uri"],
        "configuration_url": campaign["configuration_uri"],
        "source_manifest_sha256": request["source_manifest_sha256"],
        "configuration_sha256": campaign["configuration_sha256"],
        "legacy_pipeline_version": legacy_pipeline_version,
        "legacy_pose_frame": legacy_pose_frame,
        "weights_base_url": dataset_cfg["weights_base_url"],
        "work_output_url": _join_url(
            root, pipeline["work_output_path"], campaign["name"], sequence,
            *request_scope,
        ),
        "destination_url": _result_destination(request, campaign, dataset_cfg),
        "max_failure_annotations": str(max_failure_annotations),
        "max_failure_coverage": str(max_failure_coverage),
        "max_post_trim_silhouette_failure_coverage": str(
            max_silhouette_failure_coverage
        ),
    }
    for key, default in (
        ("max_chamfer_object", 40.0), ("max_chamfer_human", 40.0),
        ("max_chamfer_segment_object", 50.0),
        ("max_chamfer_segment_human", 50.0),
        ("min_object_silhouette_bbox_containment", 0.8),
        ("min_object_segment_pixels", 10),
        ("min_object_bbox_component_pixels", 3),
        ("min_object_bbox_component_fraction_of_largest", 0.001),
        ("object_silhouette_render_bbox_padding_pixels", 8),
        ("max_object_silhouette_bad_frame_fraction", 0.05),
        ("min_silhouette_bbox_containment", 0.8),
        ("min_silhouette_mask_pixels", 10),
        ("min_silhouette_bbox_component_pixels", 3),
        ("min_silhouette_bbox_component_fraction_of_largest", 0.001),
        ("silhouette_render_bbox_padding_pixels", 8),
        ("silhouette_render_bbox_padding_fraction", 0.1),
        ("min_accuracy_failure_run_frames", 5),
        ("max_silhouette_failure_coverage", 0.5),
    ):
        values[key] = str(thresholds.get(key, default))
    values["object_silhouette_render_bbox_padding_pixels"] = str(
        thresholds.get(
            "silhouette_render_bbox_padding_pixels",
            thresholds.get("object_silhouette_render_bbox_padding_pixels", 8),
        )
    )
    return values


def prepare_destination(
    request: dict, campaign: dict, *, destination: str | None = None,
    clean_incomplete: bool = False,
) -> None:
    destination = destination or _join_url(campaign["output_uri"], request["sequence_name"])
    client, bucket, prefix = _client(destination)
    objects = _list(client, bucket, prefix)
    if not objects:
        return
    keys = {item["key"] for item in objects}
    commit_key = prefix.rstrip("/") + "/commit.json"
    if commit_key in keys:
        try:
            commit, _ = _read_commit(destination, verify_hashes=False)
        except Exception as exc:
            commit = None
            invalid_reason = str(exc)
        else:
            if (
                int(commit.get("request_id", -1)) == request["id"]
                and commit.get("source_manifest_sha256") == request["source_manifest_sha256"]
                and commit.get("configuration_sha256") == campaign["configuration_sha256"]
            ):
                raise ValueError("Destination already has a valid commit for this request")
            invalid_reason = "existing commit belongs to different provenance"
    else:
        invalid_reason = "destination has payloads but no commit.json"
    if not clean_incomplete:
        raise ValueError(
            f"Incomplete/conflicting destination ({invalid_reason}); "
            "pass --clean-incomplete-destination after inspection"
        )
    require_submit_authority("delete an incomplete revalidation destination")
    for offset in range(0, len(objects), 1000):
        batch = objects[offset:offset + 1000]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": item["key"]} for item in batch], "Quiet": True},
        )


def matching_destination_commit(
    request: dict, campaign: dict, destination: str,
    *, verify_payload_hashes: bool = False,
) -> tuple[dict, str] | None:
    """Return an idempotent committed result, or reject conflicting payloads."""
    client, bucket, prefix = _client(destination)
    existing = _object_map(client, bucket, prefix)
    if not existing or "commit.json" not in existing:
        return None
    commit, manifest_hash = _read_commit(
        destination, verify_hashes=verify_payload_hashes,
    )
    if (
        int(commit.get("request_id", -1)) != request["id"]
        or commit.get("source_manifest_sha256") != request["source_manifest_sha256"]
        or commit.get("configuration_sha256") != campaign["configuration_sha256"]
    ):
        raise ValueError("Existing destination commit has different provenance")
    return commit, manifest_hash


def dispatch_revalidation(
    campaign_name: str, *, db_path: str, config: dict, limit: int | None = None,
    dry_run: bool = False, clean_incomplete_destination: bool = False,
    reuse_work_output_url: str | None = None,
    request_ids: set[int] | None = None,
    pool: str | None = None,
    pool_selector: PoolSelector | None = None,
    submit_workers: int = DEFAULT_CAMPAIGN_SUBMIT_WORKERS,
) -> list[dict]:
    if submit_workers < 1:
        raise ValueError("Campaign submission workers must be positive")
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or campaign["campaign_type"] != "LEGACY_REVALIDATION":
        raise ValueError("Revalidation dispatch requires a legacy revalidation campaign")
    dataset_cfg = config["datasets"][campaign["dataset"]]
    maximum = int(
        dataset_cfg["pipelines"][REVALIDATION_PIPELINE].get(
            "max_concurrent", 20,
        )
    )
    active_count = db.count_active_stage_executions(
        "revalidation", campaign=campaign_name, db_path=db_path,
    )
    capacity = max(0, maximum - active_count)
    if limit is not None:
        capacity = min(capacity, limit)
    pending = db.list_stage_requests(
        campaign=campaign_name, stage="revalidation", status="PENDING", db_path=db_path,
    )
    pending = [item for item in pending if _retry_is_ready(item)]
    if request_ids is not None:
        pending = [item for item in pending if int(item["id"]) in request_ids]
    # Explicit policy retries repair an already-attempted campaign member and
    # should not wait behind the untouched bulk tail. Stable sorting preserves
    # FIFO ordering within the retry and ordinary groups.
    candidates = sorted(
        pending,
        key=lambda item: not str(item.get("reason") or "").startswith(
            "policy_retry_of_request_"
        ),
    )[:capacity]
    allowed_retry_root = _join_url(
        dataset_cfg["swift_base"],
        dataset_cfg["pipelines"][REVALIDATION_PIPELINE]["work_output_path"],
    ).rstrip("/") + "/"
    selector = pool_selector or PoolSelector.collect(
        dataset_cfg,
        active_counts=load_active_counts(
            dataset_cfg, lambda: db.list_workflow_executions(
                status=("SUBMITTING", "RUNNING", "UNKNOWN"), db_path=db_path,
            ),
        ),
    )
    dispatch_started = time.perf_counter()
    first_submission_logged = False
    submission_log_lock = threading.Lock()

    def log_submission_start(request: dict) -> None:
        nonlocal first_submission_logged
        with submission_log_lock:
            if first_submission_logged:
                return
            first_submission_logged = True
            print(
                "campaign_submission first stage=revalidation "
                f"request_id={request['id']} "
                f"dispatch_elapsed={time.perf_counter() - dispatch_started:.3f}s",
                flush=True,
            )

    def submit_prepared(item: tuple[dict, int, str, dict, object]) -> dict:
        request, execution_id, template, values, pool_decision = item
        log_submission_start(request)
        try:
            workflow_id = submit.osmo_submit(
                template, pool_decision.pool, values,
            )
        except submit.AmbiguousSubmitError as exc:
            details = pool_decision.detail(f"submit_ambiguous: {exc}")
            db.apply_execution_observation(
                execution_id, execution_status="UNKNOWN",
                osmo_workflow_id=values["workflow_name"] + "-1", details=details,
                request_outcomes=[{
                    "request_id": request["id"], "status": "UNKNOWN",
                }], db_path=db_path,
            )
            return {"request_id": request["id"], "status": "UNKNOWN"}
        except Exception as exc:
            details = pool_decision.detail(f"submit_failed: {exc}")
            db.apply_execution_observation(
                execution_id, execution_status="FAILED", details=details,
                request_outcomes=[{
                    "request_id": request["id"], "status": "FAILED",
                }], db_path=db_path,
            )
            return {
                "request_id": request["id"], "status": "FAILED",
                "details": details,
            }
        else:
            db.apply_execution_observation(
                execution_id, execution_status="RUNNING",
                osmo_workflow_id=workflow_id,
                details=pool_decision.detail("workflow_running"),
                request_outcomes=[{
                    "request_id": request["id"], "status": "RUNNING",
                }], db_path=db_path,
            )
            return {
                "request_id": request["id"], "status": "RUNNING",
                "workflow_id": workflow_id,
            }

    results_by_index: dict[int, dict] = {}
    futures: list[tuple[int, object]] = []
    # Reservation and pool debit remain serial.  Each immutable submission is
    # handed to a worker immediately, so later CSS preflights do not delay the
    # first OSMO launch.
    with ThreadPoolExecutor(max_workers=submit_workers) as executor:
        for index, request in enumerate(candidates):
            parameters = json.loads(request.get("parameters_json") or "{}")
            candidate_retry_url = (
                reuse_work_output_url or parameters.get("retry_work_output_url")
            )
            normalized_retry_url = (
                str(candidate_retry_url).rstrip("/")
                if candidate_retry_url else None
            )
            if normalized_retry_url and (
                not normalized_retry_url.startswith(allowed_retry_root)
                or ".." in normalized_retry_url.split("/")
            ):
                raise ValueError(
                    "Revalidation export retry inputs must come from the configured "
                    "revalidation work-output prefix"
                )
            workflow_key = (
                REVALIDATION_EXPORT_RETRY_WORKFLOW
                if normalized_retry_url
                else REVALIDATION_WORKFLOW
            )
            template = get_workflow_cfg(
                dataset_cfg, REVALIDATION_PIPELINE, workflow_key,
            )["workflow_yaml"]
            workload = (
                "mv_hoi_revalidation_export_retry"
                if workflow_key == REVALIDATION_EXPORT_RETRY_WORKFLOW
                else REVALIDATION_PIPELINE
            )
            pool_decision = selector.choose(
                workload,
                workflow_path=MV_HOI_DIR / template,
                override=pool,
            )
            values = _revalidation_values(
                request, campaign, dataset_cfg, db_path=db_path,
            )
            if normalized_retry_url:
                values["retry_work_output_url"] = normalized_retry_url
            try:
                validate_frozen_objects(request, dataset_cfg)
            except Exception as exc:
                if dry_run:
                    results_by_index[index] = {
                        "request_id": request["id"],
                        "status": "DRY_RUN_FAILED", "details": str(exc),
                    }
                    continue
                db.update_stage_request(
                    request["id"], status="FAILED",
                    details=str(exc), db_path=db_path,
                )
                _blacklist_campaign_failure(
                    request, campaign, str(exc), db_path=db_path,
                )
                results_by_index[index] = {
                    "request_id": request["id"], "status": "FAILED",
                    "details": str(exc),
                }
                continue
            try:
                prepare_destination(
                    request, campaign, destination=values["destination_url"],
                    clean_incomplete=clean_incomplete_destination and not dry_run,
                )
                if request["cohort"] == "BULK":
                    prepare_destination(
                        request, campaign,
                        destination=_published_destination(request, campaign),
                        clean_incomplete=(
                            clean_incomplete_destination and not dry_run
                        ),
                    )
            except Exception as exc:
                if dry_run:
                    results_by_index[index] = {
                        "request_id": request["id"],
                        "status": "DRY_RUN_FAILED", "details": str(exc),
                    }
                    continue
                db.update_stage_request(
                    request["id"], status="FAILED",
                    details=str(exc), db_path=db_path,
                )
                _blacklist_campaign_failure(
                    request, campaign, str(exc), db_path=db_path,
                )
                results_by_index[index] = {
                    "request_id": request["id"], "status": "FAILED",
                    "details": str(exc),
                }
                continue
            if dry_run:
                results_by_index[index] = {
                    "request_id": request["id"], "status": "DRY_RUN",
                    "values": values,
                }
                continue
            reservation = db.reserve_request_with_execution(
                request["id"],
                reserved_by=os.environ.get("HOSTNAME", "campaign-dispatcher"),
                workflow_name=values["workflow_name"],
                pipeline_version=request["pipeline_version"],
                workflow_spec_path=template, pool=pool_decision.pool,
                details=pool_decision.detail("submit_reserved"),
                db_path=db_path,
            )
            if not reservation:
                continue
            futures.append((
                index,
                executor.submit(submit_prepared, (
                    request, reservation["execution_id"], template, values,
                    pool_decision,
                )),
            ))
        for index, future in futures:
            results_by_index[index] = future.result()
    return [results_by_index[index] for index in sorted(results_by_index)]


def _read_commit(url: str, *, verify_hashes: bool = False) -> tuple[dict, str]:
    client, bucket, prefix = _client(url)
    commit_key = prefix.rstrip("/") + "/commit.json"
    for attempt in range(COMMIT_VISIBILITY_RETRIES):
        try:
            payload = client.get_object(Bucket=bucket, Key=commit_key)["Body"].read()
            break
        except ClientError as exc:
            error = exc.response.get("Error", {})
            if (
                error.get("Code") not in ("NoSuchKey", "404")
                or attempt + 1 >= COMMIT_VISIBILITY_RETRIES
            ):
                raise
            time.sleep(COMMIT_VISIBILITY_BASE_DELAY_SECONDS * (2 ** attempt))
    commit = json.loads(payload)
    expected = {item["path"]: item for item in commit.get("files", [])}
    actual_objects: dict[str, dict] = {}
    for attempt in range(COMMIT_VISIBILITY_RETRIES):
        actual_objects = _object_map(client, bucket, prefix)
        actual_objects.pop("commit.json", None)
        try:
            if set(expected) != set(actual_objects):
                raise ValueError("Committed export key set does not match destination")
            for key, item in expected.items():
                if int(item["size"]) != actual_objects[key]["size"]:
                    raise ValueError(f"Committed export size differs for {key}")
        except ValueError:
            if attempt + 1 >= COMMIT_VISIBILITY_RETRIES:
                raise
            time.sleep(COMMIT_VISIBILITY_BASE_DELAY_SECONDS * (2 ** attempt))
            continue
        break
    if verify_hashes:
        # Only hash after the destination listing and sizes have converged.
        # A commit is uploaded last, but CSS list results can briefly lag it.
        def verify_object_hash(entry: tuple[str, dict]) -> None:
            key, item = entry
            object_key = f"{prefix.rstrip('/')}/{key}"
            size = int(item["size"])
            digest = hashlib.sha256()
            for first_byte in range(0, size, COMMIT_HASH_RANGE_BYTES):
                last_byte = min(size, first_byte + COMMIT_HASH_RANGE_BYTES) - 1
                expected_bytes = last_byte - first_byte + 1
                for attempt in range(COMMIT_HASH_READ_RETRIES):
                    body = None
                    tentative = digest.copy()
                    observed_bytes = 0
                    try:
                        body = client.get_object(
                            Bucket=bucket,
                            Key=object_key,
                            Range=f"bytes={first_byte}-{last_byte}",
                        )["Body"]
                        for chunk in iter(lambda: body.read(4 * 1024 * 1024), b""):
                            observed_bytes += len(chunk)
                            tentative.update(chunk)
                        if observed_bytes != expected_bytes:
                            raise ResponseStreamingError(
                                error=RuntimeError(
                                    f"short range read: expected {expected_bytes}, "
                                    f"observed {observed_bytes}"
                                ),
                            )
                    except (ReadTimeoutError, ResponseStreamingError):
                        if attempt + 1 >= COMMIT_HASH_READ_RETRIES:
                            raise
                        time.sleep(
                            COMMIT_VISIBILITY_BASE_DELAY_SECONDS * (2 ** attempt)
                        )
                        continue
                    finally:
                        if body is not None:
                            body.close()
                    digest = tentative
                    break
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Committed export SHA-256 differs for {key}")

        with ThreadPoolExecutor(max_workers=COMMIT_HASH_WORKERS) as executor:
            list(executor.map(verify_object_hash, expected.items()))
    return commit, _sha256_bytes(payload)


def _handle_revalidation_observation(
    request: dict,
    execution: dict,
    info: dict,
    *,
    campaign: dict,
    dataset_cfg: dict,
    db_path: str,
    verify_payload_hashes: bool,
    retry_infrastructure: bool,
    recover_publication_failures: bool,
    defer_bulk_publication: bool,
    publication_copy_workers: int,
) -> dict:
    """Apply one OSMO observation and, when requested, publish its result."""
    workflow_id = execution.get("osmo_workflow_id") or execution["workflow_name"]
    status = info.get("status", "UNKNOWN")
    retry_info = info
    retry_query_payload = None
    infrastructure_failure_reason = None
    if (
        info.get("not_found")
        and _is_ambiguous_submit_placeholder(execution)
        and _ambiguous_submit_not_found_grace_expired({
            "details": execution.get("details"),
            "execution_submitted_at": execution.get("submitted_at"),
        })
    ):
        retry_info = {**info, "missing_submission": True}
        retry_query_payload = info
        infrastructure_failure_reason = (
            "osmo_submission_not_found_after_grace"
        )
        status = "FAILED"
    if status == "COMPLETED":
        if defer_bulk_publication and request["cohort"] != "CANARY":
            detail = REVALIDATION_RESULT_PENDING
            if _revalidation_repair_active(request, campaign, dataset_cfg):
                detail += ": repair_active"
            db.apply_execution_observation(
                execution["id"], execution_status="SUCCEEDED", details=detail,
                query_payload=info, request_outcomes=[{
                    "request_id": request["id"], "status": "RUNNING",
                    "details": detail,
                }], db_path=db_path,
            )
            return {"request_id": request["id"], "status": "WAITING_EXPORT"}
        if _revalidation_repair_active(request, campaign, dataset_cfg):
            return {"request_id": request["id"], "status": "REPAIR_ACTIVE"}
        destination = _result_destination(request, campaign, dataset_cfg)
        published_destination = _published_destination(request, campaign)
        cleanup_pending = str(request.get("details") or "").startswith(
            "candidate_cleanup_pending"
        )
        try:
            validate_frozen_objects(request, dataset_cfg)
            if cleanup_pending:
                commit, manifest_hash = _read_commit(
                    published_destination, verify_hashes=verify_payload_hashes,
                )
                cleanup_promoted_export_candidate(
                    destination, published_destination, expected_commit=commit,
                )
            else:
                commit, manifest_hash = _read_commit(
                    destination, verify_hashes=verify_payload_hashes,
                )
            if (
                commit.get("complete") is not True
                or int(commit.get("request_id", -1)) != request["id"]
                or commit.get("source_manifest_sha256")
                != request["source_manifest_sha256"]
                or commit.get("configuration_sha256")
                != campaign["configuration_sha256"]
            ):
                raise ValueError("Destination commit provenance does not match request")
        except CandidateCleanupError as exc:
            detail = f"candidate_cleanup_pending: {exc}"
            db.apply_execution_observation(
                execution["id"], execution_status="SUCCEEDED", details=detail,
                query_payload=info, request_outcomes=[{
                    "request_id": request["id"], "status": "RUNNING",
                    "details": detail,
                }], db_path=db_path,
            )
            return {
                "request_id": request["id"], "status": "CLEANUP_PENDING",
                "details": detail,
            }
        except Exception as exc:
            detail = f"result_validation_failed: {exc}"
            db.apply_execution_observation(
                execution["id"],
                execution_status=(
                    "SUCCEEDED" if execution["status"] == "SUCCEEDED" else "FAILED"
                ),
                details=detail, query_payload=info, request_outcomes=[{
                    "request_id": request["id"], "status": "FAILED",
                }], db_path=db_path,
            )
            _blacklist_campaign_failure(request, campaign, detail, db_path=db_path)
            return {"request_id": request["id"], "status": "FAILED", "details": detail}
        manifest_uri = destination.rstrip("/") + "/commit.json"
        summary = {
            "file_count": commit["file_count"],
            "total_bytes": commit["total_bytes"],
            "pose_comparison_status": commit["pose_comparison_status"],
            "accuracy_status": commit["accuracy_status"],
            "silhouette_status": commit["silhouette_status"],
            "pose_comparison": commit.get("pose_comparison_summary", {}),
        }
        reconstruction = db.get_latest_successful_stage_run(
            request["sequence_name"], campaign["dataset"], RECON_PIPELINE,
            db_path=db_path,
        )
        if not reconstruction:
            detail = "result_validation_failed: legacy reconstruction is no longer current"
            db.apply_execution_observation(
                execution["id"],
                execution_status=(
                    "SUCCEEDED" if execution["status"] == "SUCCEEDED" else "FAILED"
                ),
                details=detail, query_payload=info, request_outcomes=[{
                    "request_id": request["id"], "status": "FAILED",
                }], db_path=db_path,
            )
            _blacklist_campaign_failure(request, campaign, detail, db_path=db_path)
            return {"request_id": request["id"], "status": "FAILED", "details": detail}
        if request["cohort"] == "CANARY":
            db.apply_execution_observation(
                execution["id"], execution_status="SUCCEEDED",
                details="canary_result_verified", query_payload=info,
                request_outcomes=[{
                    "request_id": request["id"], "status": "SUCCEEDED",
                    "result_manifest_uri": manifest_uri,
                    "result_manifest_sha256": manifest_hash,
                    "result_summary": summary,
                }], db_path=db_path,
            )
            return {"request_id": request["id"], "status": "SUCCEEDED"}
        try:
            require_submit_authority("publish a verified revalidation export")
            existing = (
                (commit, manifest_hash)
                if cleanup_pending
                else matching_destination_commit(
                    request, campaign, published_destination,
                    verify_payload_hashes=verify_payload_hashes,
                )
            )
            if existing is None:
                published_commit, published_hash = _publish_committed_prefix(
                    destination, published_destination,
                    verify_payload_hashes=verify_payload_hashes,
                    copy_workers=publication_copy_workers,
                )
            else:
                published_commit, published_hash = existing
            if published_commit != commit or published_hash != manifest_hash:
                raise ValueError("Published export differs from verified candidate")
            committed_export = db.commit_revalidation_export(
                request["id"],
                reconstruction_run_id=reconstruction["stage_run_id"],
                result_manifest_uri=published_destination.rstrip("/") + "/commit.json",
                result_manifest_sha256=published_hash,
                result_summary=summary,
                source_uri=reconstruction["output_uri"],
                output_uri=published_destination,
                query_payload=info,
                recover_publication_failure=recover_publication_failures,
                db_path=db_path,
            )
            _resolve_blacklist_after_committed_export(
                request, campaign, db_path=db_path,
            )
        except CandidateCleanupError as exc:
            detail = f"candidate_cleanup_pending: {exc}"
            db.apply_execution_observation(
                execution["id"], execution_status="SUCCEEDED", details=detail,
                query_payload=info, request_outcomes=[{
                    "request_id": request["id"], "status": "RUNNING",
                    "details": detail,
                }], db_path=db_path,
            )
            return {
                "request_id": request["id"], "status": "CLEANUP_PENDING",
                "details": detail,
            }
        except Exception as exc:
            detail = f"export_run_commit_failed: {exc}"
            db.apply_execution_observation(
                execution["id"],
                execution_status=(
                    "SUCCEEDED" if execution["status"] == "SUCCEEDED" else "FAILED"
                ),
                details=detail, query_payload=info, request_outcomes=[{
                    "request_id": request["id"], "status": "FAILED",
                }], db_path=db_path,
            )
            _blacklist_campaign_failure(request, campaign, detail, db_path=db_path)
            return {"request_id": request["id"], "status": "FAILED", "details": detail}
        return {
            "request_id": request["id"], "status": "SUCCEEDED",
            "export_run_id": committed_export["stage_run_id"],
        }
    if status.startswith("FAILED"):
        try:
            rejected = _revalidation_post_trim_rejection(
                request, campaign, dataset_cfg,
            )
        except Exception as exc:
            rejected = None
            print(
                f"  {request['sequence_name']}: malformed revalidation "
                f"export rejection: {exc}"
            )
        if rejected is not None:
            report, candidate = rejected
            failed_gates = [
                gate["name"] for gate in report["gates"]
                if gate["status"] == "FAIL"
            ]
            detail = "expected_post_trim_qc_rejection: " + ", ".join(
                failed_gates
            )
            summary = {
                "failure_category": "post_trim_qc_limit",
                "canary_outcome": "EXPECTED_QC_DATA",
                "export_rejection": report,
                "export_rejection_sha256": report["report_sha256"],
                "failed_gates": failed_gates,
            }
            try:
                summary["staging_cleanup"] = cleanup_rejected_export_candidate(
                    candidate, expected_report=report,
                )
            except Exception as exc:
                pending = "post_trim_qc_rejection_cleanup_pending: " + str(exc)
                db.apply_execution_observation(
                    execution["id"], execution_status="FAILED", details=pending,
                    query_payload=info, request_outcomes=[{
                        "request_id": request["id"], "status": "UNKNOWN",
                        "details": pending, "result_summary": summary,
                    }], db_path=db_path,
                )
                return {
                    "request_id": request["id"], "status": "CLEANUP_PENDING",
                    "details": pending,
                }
            db.apply_execution_observation(
                execution["id"], execution_status="FAILED", details=detail,
                query_payload=info, request_outcomes=[{
                    "request_id": request["id"], "status": "FAILED",
                    "details": detail, "result_summary": summary,
                }], db_path=db_path,
            )
            _blacklist_campaign_failure(request, campaign, detail, db_path=db_path)
            return {
                "request_id": request["id"], "status": "EXPECTED_QC_DATA",
                "details": detail,
            }
        retry_result = handle_campaign_infrastructure_failure(
            workflow_id=workflow_id,
            execution_id=execution["id"],
            dataset=request["dataset"],
            sequence_name=request["sequence_name"],
            campaign_name=request["campaign_name"],
            campaign_type=request["campaign_type"],
            info=retry_info,
            retry_infrastructure=retry_infrastructure,
            db_path=db_path,
            failure_reason=infrastructure_failure_reason,
            query_payload=retry_query_payload,
        )
        if retry_result in ("RETRY_PENDING", "RETRY_CREATED"):
            return {"request_id": request["id"], "status": retry_result}
        if request["cohort"] == "CANARY" and retry_result is None:
            expected_summary = _expected_revalidation_canary_failure(
                request, campaign, dataset_cfg, info, db_path=db_path,
            )
            if expected_summary is not None:
                detail = (
                    "expected_canary_qc_data_failure: "
                    + str(expected_summary["failed_task"])
                )
                db.apply_execution_observation(
                    execution["id"], execution_status="FAILED", details=detail,
                    query_payload=info, request_outcomes=[{
                        "request_id": request["id"], "status": "FAILED",
                        "details": detail,
                        "result_summary": expected_summary,
                    }], db_path=db_path,
                )
                _blacklist_campaign_failure(
                    request, campaign, detail, db_path=db_path,
                )
                return {
                    "request_id": request["id"],
                    "status": "EXPECTED_QC_DATA", "details": detail,
                }
        observed_failure_detail = (
            infrastructure_failure_reason or _failure_detail(info)
        )
        detail = (
            "osmo_infrastructure_retry_exhausted: " + observed_failure_detail
            if retry_result == "EXHAUSTED"
            else observed_failure_detail
        )
        db.apply_execution_observation(
            execution["id"], execution_status="FAILED", details=detail,
            query_payload=info, request_outcomes=[{
                "request_id": request["id"], "status": "FAILED",
            }], db_path=db_path,
        )
        _blacklist_campaign_failure(request, campaign, detail, db_path=db_path)
        return {"request_id": request["id"], "status": "FAILED", "details": detail}
    normalized = (
        "UNKNOWN"
        if status == "UNKNOWN" or info.get("not_found")
        else "RUNNING"
    )
    db.apply_execution_observation(
        execution["id"], execution_status=normalized, query_payload=info,
        request_outcomes=[{
            "request_id": request["id"], "status": normalized,
        }], db_path=db_path,
    )
    return {"request_id": request["id"], "status": normalized}


def reconcile_revalidation(
    campaign_name: str,
    *,
    db_path: str,
    verify_payload_hashes: bool = False,
    retry_infrastructure: bool = False,
    recover_publication_failures: bool = False,
    skip_request_ids: set[int] | None = None,
    query_workers: int = DEFAULT_CAMPAIGN_QUERY_WORKERS,
    publish_workers: int = DEFAULT_REVALIDATION_PUBLISH_WORKERS,
    defer_bulk_publication: bool = False,
    pending_publication_only: bool = False,
) -> list[dict]:
    """Observe revalidation work and optionally publish durable pending results."""
    if query_workers < 1 or publish_workers < 1:
        raise ValueError("Campaign query and publication workers must be positive")
    effective_publish_workers = min(publish_workers, PUBLISH_COPY_WORKERS)
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign:
        raise ValueError(f"Unknown campaign: {campaign_name}")
    config = load_config(MV_HOI_DIR)
    dataset_cfg = config["datasets"][campaign["dataset"]]
    if pending_publication_only:
        requests = [
            request for request in db.list_stage_requests(
                campaign=campaign_name, stage="revalidation", status="RUNNING",
                db_path=db_path,
            )
            if str(request.get("details") or "").startswith(
                REVALIDATION_RESULT_PENDING
            )
        ]
    else:
        requests = db.list_stage_requests(
            campaign=campaign_name, stage="revalidation",
            status=("SUBMITTED", "RUNNING", "UNKNOWN"), db_path=db_path,
        )
        requests = [
            request for request in requests
            if not str(request.get("details") or "").startswith(
                REVALIDATION_RESULT_PENDING
            )
        ]
    if recover_publication_failures and (pending_publication_only or not defer_bulk_publication):
        requests.extend(
            request
            for request in db.list_stage_requests(
                campaign=campaign_name, stage="revalidation", status="FAILED",
                db_path=db_path,
            )
            if str(request.get("details") or "").startswith(
                "export_run_commit_failed:"
            )
        )
    if skip_request_ids:
        requests = [
            request for request in requests
            if request["id"] not in skip_request_ids
        ]
    rows: list[tuple[dict, dict]] = []
    for request in requests:
        if request["workflow_execution_id"] is None:
            continue
        rows.append((
            request,
            db.get_workflow_execution(
                request["workflow_execution_id"], db_path=db_path,
            ),
        ))

    if pending_publication_only:
        def publish(row: tuple[dict, dict]) -> dict:
            request, execution = row
            info = json.loads(execution.get("last_query_payload_json") or "{}")
            info["status"] = "COMPLETED"
            return _handle_revalidation_observation(
                request, execution, info, campaign=campaign,
                dataset_cfg=dataset_cfg, db_path=db_path,
                verify_payload_hashes=verify_payload_hashes,
                retry_infrastructure=retry_infrastructure,
                recover_publication_failures=recover_publication_failures,
                defer_bulk_publication=False,
                publication_copy_workers=max(
                    1, PUBLISH_COPY_WORKERS // effective_publish_workers,
                ),
            )

        with ThreadPoolExecutor(
            max_workers=min(effective_publish_workers, max(1, len(rows))),
        ) as executor:
            return list(executor.map(publish, rows))

    def query(row: tuple[dict, dict]) -> tuple[dict, dict, dict]:
        request, execution = row
        workflow_id = execution.get("osmo_workflow_id") or execution["workflow_name"]
        return request, execution, osmo_query(workflow_id)

    with ThreadPoolExecutor(
        max_workers=min(query_workers, max(1, len(rows))),
    ) as executor:
        observations = list(executor.map(query, rows))
    return [
        _handle_revalidation_observation(
            request, execution, info, campaign=campaign,
            dataset_cfg=dataset_cfg, db_path=db_path,
            verify_payload_hashes=verify_payload_hashes,
            retry_infrastructure=retry_infrastructure,
            recover_publication_failures=recover_publication_failures,
            defer_bulk_publication=defer_bulk_publication,
            publication_copy_workers=PUBLISH_COPY_WORKERS,
        )
        for request, execution, info in observations
    ]


def recover_terminal_campaign_infrastructure_failures(
    campaign_name: str,
    *,
    db_path: str,
    query_workers: int = DEFAULT_CAMPAIGN_QUERY_WORKERS,
    request_ids: set[int] | None = None,
) -> list[dict]:
    """Restart terminal campaign requests that failed in OSMO control.

    Older observations may not contain task exit metadata, so terminal rows are
    queried once from OSMO. Non-infrastructure failures remain untouched.
    """

    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign:
        raise ValueError(f"Unknown campaign: {campaign_name}")
    if campaign["status"] not in {"FROZEN", "RUNNING"}:
        return []
    campaign_requests = db.list_stage_requests(
        campaign=campaign_name, db_path=db_path,
    )
    if request_ids is not None:
        known_ids = {int(request["id"]) for request in campaign_requests}
        missing = sorted(request_ids - known_ids)
        if missing:
            raise ValueError(
                f"Requests do not belong to campaign {campaign_name}: {missing}"
            )
        campaign_requests = [
            request for request in campaign_requests
            if int(request["id"]) in request_ids
        ]
    active_requests = db.list_stage_requests(
        dataset=campaign["dataset"],
        status=("PENDING", "BLOCKED", "RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN"),
        db_path=db_path,
    )
    active_sequence_stages = {
        (int(request["sequence_id"]), request["stage"])
        for request in active_requests
    }
    replaced_request_ids = {
        int(match.group(1))
        for request in campaign_requests
        if (
            match := re.fullmatch(
                r"infrastructure_retry_of_request_(\d+)",
                str(request.get("reason") or ""),
            )
        )
    }
    requests = [
        request for request in campaign_requests
        if request["status"] == "FAILED"
        and request["id"] not in replaced_request_ids
        and (int(request["sequence_id"]), request["stage"])
        not in active_sequence_stages
    ]
    if query_workers < 1:
        raise ValueError("Campaign query workers must be positive")
    candidates: list[tuple[dict, dict, dict]] = []
    for request in requests:
        if not request["workflow_execution_id"]:
            continue
        execution = db.get_workflow_execution(
            request["workflow_execution_id"], db_path=db_path,
        )
        stored = json.loads(execution.get("last_query_payload_json") or "{}")
        stored_for_request = stored
        if request["stage"] == "export":
            export_run = db.get_stage_run_by_request(
                request["id"], stage=db.EXPORT_STAGE, db_path=db_path,
            )
            task_name = (
                export_run.get("workflow_task_name") if export_run else None
            ) or export_task_names(request["sequence_name"])[0]
            stored_for_request = export_task_failure_info(stored, [task_name])
            if (
                request_ids is None
                and stored_for_request.get("tasks", {}).get(task_name)
                != "COMPLETED"
                and not is_retryable_infrastructure_failure(stored_for_request)
            ):
                continue
        if (
            request_ids is None
            and "task_details" in stored_for_request
            and not is_retryable_infrastructure_failure(stored_for_request)
            and request["stage"] != "export"
        ):
            continue
        candidates.append((request, execution, stored))

    executions: dict[int, dict] = {
        int(execution["id"]): execution for _request, execution, _stored in candidates
    }

    def query(execution: dict) -> tuple[int, dict]:
        workflow_id = execution.get("osmo_workflow_id") or execution["workflow_name"]
        info = osmo_query(workflow_id, classify_logs=False)
        return int(execution["id"]), info

    with ThreadPoolExecutor(
        max_workers=min(query_workers, max(1, len(executions))),
    ) as executor:
        observations_by_execution = dict(executor.map(query, executions.values()))
    observations = [
        (request, execution, stored, observations_by_execution[int(execution["id"])])
        for request, execution, stored in candidates
    ]

    def scope_observation(
        observation: tuple[dict, dict, dict, dict],
    ) -> tuple[dict, dict, dict, dict, dict, dict | None, str | None]:
        request, execution, stored, info = observation
        workflow_id = execution.get("osmo_workflow_id") or execution["workflow_name"]
        scoped_info = info
        export_run = None
        task_name = None
        if request["stage"] == "export":
            export_run = db.get_stage_run_by_request(
                request["id"], stage=db.EXPORT_STAGE, db_path=db_path,
            )
            task_name = (
                export_run.get("workflow_task_name") if export_run else None
            ) or export_task_names(request["sequence_name"])[0]
            scoped_info = export_task_failure_info(info, [task_name])
        if str(scoped_info.get("status") or "").startswith("FAILED"):
            _classify_file_not_found_failures(workflow_id, scoped_info)
        return (
            request, execution, stored, info, scoped_info, export_run, task_name,
        )

    # Log fallback is bounded but can take up to 30 seconds per task.  Keep it
    # read-only and parallel so one batched export does not serialize recovery
    # across all of its independently failed sequence tasks.
    with ThreadPoolExecutor(
        max_workers=min(query_workers, max(1, len(observations))),
    ) as executor:
        scoped_observations = list(executor.map(scope_observation, observations))

    def resolve_completed_export(
        observation: tuple[dict, dict, dict, dict, dict, dict | None, str | None],
    ) -> tuple[int, dict | None]:
        request, _execution, _stored, info, _scoped, export_run, task_name = observation
        if (
            request["stage"] != "export"
            or task_name is None
            or info.get("tasks", {}).get(task_name) != "COMPLETED"
        ):
            return int(request["id"]), None
        # A prior commit verification already proved that this request wrote no
        # durable candidate/final commit.  Preserve that evidence and avoid
        # repeating a slow object-store miss during targeted recovery.
        request_detail = str(request.get("details") or "")
        if (
            request_detail.startswith("export_commit_invalid:")
            and "NoSuchKey" in request_detail
        ):
            return int(request["id"]), {"status": "MISSING"}
        parameters = json.loads(request.get("parameters_json") or "{}")
        candidate = parameters.get("candidate_uri")
        destination = parameters.get("output_uri") or (
            export_run.get("output_uri") if export_run else None
        )
        if not candidate or not destination or export_run is None:
            return int(request["id"]), {"status": "MISSING"}
        try:
            durable = _read_commit(destination, verify_hashes=False)
            cleanup_promoted_export_candidate(
                candidate, destination, expected_commit=durable[0],
            )
        except CandidateCleanupError as exc:
            return int(request["id"]), {
                "status": "CLEANUP_PENDING", "details": str(exc),
            }
        except Exception:
            try:
                durable = _publish_committed_prefix(
                    candidate, destination, verify_payload_hashes=False,
                )
            except CandidateCleanupError as exc:
                return int(request["id"]), {
                    "status": "CLEANUP_PENDING", "details": str(exc),
                }
            except Exception:
                return int(request["id"]), {"status": "MISSING"}
        commit, manifest_hash = durable
        if (
            commit.get("sequence_name") != request["sequence_name"]
            or commit.get("pipeline_version") != request["pipeline_version"]
            or int(commit.get("reconstruction_run_id", -1))
            != int(export_run["reconstruction_run_id"])
        ):
            return int(request["id"]), {"status": "MISSING"}
        return int(request["id"]), {
            "status": "DURABLE",
            "commit": commit,
            "manifest_hash": manifest_hash,
            "destination": destination,
        }

    # Candidate/final verification is also read-heavy and independent by
    # request.  Resolve it concurrently, then perform DB transitions below in
    # stable request order.
    with ThreadPoolExecutor(
        max_workers=min(query_workers, max(1, len(scoped_observations))),
    ) as executor:
        completed_export_results = dict(
            executor.map(resolve_completed_export, scoped_observations)
        )

    results: list[dict] = []
    for (
        request, execution, stored, info, scoped_info, export_run, task_name,
    ) in scoped_observations:
        workflow_id = execution.get("osmo_workflow_id") or execution["workflow_name"]
        if request["stage"] == "export":
            assert task_name is not None
            if info.get("tasks", {}).get(task_name) == "COMPLETED":
                artifact = completed_export_results.get(int(request["id"])) or {
                    "status": "MISSING",
                }
                if artifact["status"] == "CLEANUP_PENDING":
                    detail = f"candidate_cleanup_pending: {artifact['details']}"
                    assert export_run is not None
                    db.reopen_failed_export_for_candidate_cleanup(
                        request["id"], details=detail,
                        db_path=db_path,
                    )
                    results.append({
                        "request_id": request["id"],
                        "status": "CLEANUP_PENDING",
                        "details": artifact["details"],
                    })
                    continue
                if artifact["status"] == "DURABLE":
                    commit = artifact["commit"]
                    destination = artifact["destination"]
                    db.reconcile_failed_export_success(
                        request["id"],
                        result_manifest_uri=destination.rstrip("/") + "/commit.json",
                        result_manifest_sha256=artifact["manifest_hash"],
                        result_summary={
                            "file_count": commit.get("file_count"),
                            "total_bytes": commit.get("total_bytes"),
                            "export_validation": commit.get("export_validation"),
                        },
                        details="export_reconciled_from_durable_commit",
                        db_path=db_path,
                    )
                    _resolve_blacklist_after_committed_export(
                        request, campaign, db_path=db_path,
                    )
                    results.append({
                        "request_id": request["id"],
                        "status": "DURABLE_EXPORT_RECONCILED",
                    })
                    continue
                scoped_info = {
                    "status": "FAILED",
                    "tasks": {task_name: "COMPLETED"},
                    "task_details": {},
                    "export_output_incomplete": True,
                }
        if (
            info.get("status") == "RUNNING"
            and is_retryable_infrastructure_failure(
                export_task_failure_info(stored, [task_name])
                if request["stage"] == "export" else stored
            )
        ):
            if request["stage"] == "export":
                continue
            db.set_campaign_execution_retry_state(
                execution["id"], status="RUNNING",
                details="retryable_osmo_infrastructure_failure: remote restart recovered",
                query_payload=stored, db_path=db_path,
            )
            results.append({
                "request_id": request["id"],
                "status": "REMOTE_RUNNING_RECOVERED",
            })
            continue
        if (
            not str(info.get("status", "")).startswith("FAILED")
            and not scoped_info.get("export_output_incomplete")
            and not any(
                str(status).startswith("FAILED")
                for status in scoped_info.get("tasks", {}).values()
            )
        ):
            continue
        if not is_retryable_infrastructure_failure(scoped_info):
            db.update_workflow_execution(
                execution["id"], status="FAILED",
                query_payload=info, db_path=db_path,
            )
            continue
        retry_result = handle_campaign_infrastructure_failure(
            workflow_id=workflow_id,
            execution_id=execution["id"],
            dataset=request["dataset"],
            sequence_name=request["sequence_name"],
            campaign_name=request["campaign_name"],
            campaign_type=request["campaign_type"],
            info=scoped_info,
            retry_infrastructure=True,
            request_id=request["id"] if request["stage"] == "export" else None,
            query_payload=info,
            db_path=db_path,
        )
        if retry_result:
            results.append({
                "request_id": request["id"], "status": retry_result,
            })
    return results


def _publish_committed_prefix(
    source: str, destination: str, *, verify_payload_hashes: bool = False,
    copy_workers: int = PUBLISH_COPY_WORKERS,
) -> tuple[dict, str]:
    """Resumably copy a verified result, preserving commit.json as the last write."""
    commit, commit_hash = _read_commit(source, verify_hashes=verify_payload_hashes)
    source_client, source_bucket, source_prefix = _client(source)
    destination_client, destination_bucket, destination_prefix = _client(destination)
    del source_client  # The destination client can copy from the same CSS account.

    destination_root = destination_prefix.rstrip("/")
    relative_existing = _object_map(
        destination_client, destination_bucket, destination_root,
    )
    if "commit.json" in relative_existing:
        copied, copied_hash = _read_commit(
            destination, verify_hashes=verify_payload_hashes,
        )
        if copied_hash != commit_hash or copied != commit:
            raise ValueError(
                "Published canary commit differs from its verified staged result"
            )
        return copied, copied_hash

    expected = {item["path"]: item for item in commit["files"]}
    unexpected = sorted(set(relative_existing) - set(expected))
    if unexpected:
        raise ValueError(
            "Incomplete destination contains unexpected keys: "
            + ", ".join(unexpected[:5])
        )
    for relative, observed in relative_existing.items():
        if int(observed["size"]) != int(expected[relative]["size"]):
            raise ValueError(
                f"Incomplete destination size differs for {relative}"
            )

    pending = [
        item for item in commit["files"]
        if item["path"] not in relative_existing
    ]

    def copy_object(*, relative: str, size: int) -> None:
        destination_key = f"{destination_prefix.rstrip('/')}/{relative}"
        copy_source = {
            "Bucket": source_bucket,
            "Key": f"{source_prefix.rstrip('/')}/{relative}",
        }
        if size <= SINGLE_COPY_MAX_BYTES:
            destination_client.copy_object(
                Bucket=destination_bucket,
                Key=destination_key,
                CopySource=copy_source,
            )
            return

        upload = destination_client.create_multipart_upload(
            Bucket=destination_bucket, Key=destination_key,
        )
        upload_id = upload["UploadId"]
        parts = []
        try:
            for part_number, first_byte in enumerate(
                range(0, size, MULTIPART_COPY_PART_BYTES), start=1,
            ):
                last_byte = min(size, first_byte + MULTIPART_COPY_PART_BYTES) - 1
                response = destination_client.upload_part_copy(
                    Bucket=destination_bucket,
                    Key=destination_key,
                    PartNumber=part_number,
                    UploadId=upload_id,
                    CopySource=copy_source,
                    CopySourceRange=f"bytes={first_byte}-{last_byte}",
                )
                parts.append({
                    "PartNumber": part_number,
                    "ETag": response["CopyPartResult"]["ETag"],
                })
            destination_client.complete_multipart_upload(
                Bucket=destination_bucket,
                Key=destination_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            destination_client.abort_multipart_upload(
                Bucket=destination_bucket, Key=destination_key, UploadId=upload_id,
            )
            raise

    def copy_payload(item: dict) -> None:
        copy_object(
            relative=item["path"], size=int(item["size"]),
        )

    if copy_workers < 1:
        raise ValueError("Publication copy workers must be positive")
    with ThreadPoolExecutor(
        max_workers=min(copy_workers, max(1, len(pending))),
    ) as executor:
        list(executor.map(copy_payload, pending))

    copy_object(relative="commit.json", size=0)
    copied, copied_hash = _read_commit(
        destination, verify_hashes=verify_payload_hashes,
    )
    if copied_hash != commit_hash or copied != commit:
        raise ValueError("Published canary commit differs from its verified staged result")
    return copied, copied_hash


def publish_backlog_exports(
    campaign_name: str, *, db_path: str, verify_payload_hashes: bool = False,
    max_per_cycle: int = 20, workers: int = 4,
) -> list[dict]:
    """Publish verified backlog candidates and commit their export runs."""
    if max_per_cycle < 1 or workers < 1:
        raise ValueError("Backlog publication limit and workers must be positive")
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or campaign["campaign_type"] not in (
        "BACKLOG_REPROCESSING", "REMEDIATION",
    ):
        raise ValueError("Backlog publication requires a backlog/remediation campaign")
    if campaign["phase"] == "CANARY":
        return []
    requests = db.list_stage_requests(
        campaign=campaign_name, stage="export", status="RUNNING", db_path=db_path,
    )
    eligible: list[tuple[dict, dict]] = []
    for request in requests:
        run = db.get_stage_run_by_request(
            request["id"], stage=db.EXPORT_STAGE, db_path=db_path,
        )
        if run and run.get("details") == "candidate_export_verified":
            eligible.append((request, run))
    eligible.sort(key=lambda item: (
        item[0].get("created_at") or "", int(item[0]["id"]),
    ))
    selected = eligible[:max_per_cycle]
    effective_workers = min(workers, max(1, len(selected)))

    def publish(item: tuple[dict, dict]) -> dict:
        request, run = item
        parameters = json.loads(request.get("parameters_json") or "{}")
        candidate = parameters.get("candidate_uri")
        destination = parameters.get("output_uri")
        if not candidate or not destination:
            detail = "candidate_publish_failed: export request lacks candidate/output URI"
            db.update_stage_run(
                run["stage_run_id"], status="FAILED", details=detail,
                stage=db.EXPORT_STAGE, db_path=db_path,
            )
            db.update_stage_request(
                request["id"], status="FAILED", details=detail, db_path=db_path,
            )
            _blacklist_campaign_failure(request, campaign, detail, db_path=db_path)
            return {"request_id": request["id"], "status": "FAILED", "details": detail}
        expected_destination = _join_url(campaign["output_uri"], request["sequence_name"])
        if destination.rstrip("/") != expected_destination.rstrip("/"):
            detail = "candidate_publish_failed: destination is outside campaign output"
            db.update_stage_run(
                run["stage_run_id"], status="FAILED", details=detail,
                stage=db.EXPORT_STAGE, db_path=db_path,
            )
            db.update_stage_request(
                request["id"], status="FAILED", details=detail, db_path=db_path,
            )
            _blacklist_campaign_failure(request, campaign, detail, db_path=db_path)
            return {"request_id": request["id"], "status": "FAILED", "details": detail}
        try:
            commit, manifest_hash = _read_commit(
                candidate, verify_hashes=verify_payload_hashes,
            )
            if (
                commit.get("schema") not in {
                    "v2d.mv_hoi.export_commit.v1",
                    "v2d.mv_hoi.export_commit.v2",
                    "v2d.mv_hoi.export_commit.v3",
                }
                or commit.get("complete") is not True
                or commit.get("sequence_name") != request["sequence_name"]
                or commit.get("pipeline_version") != request["pipeline_version"]
                or int(commit.get("reconstruction_run_id", -1))
                != int(run["reconstruction_run_id"])
            ):
                raise ValueError("Candidate export commit provenance is inconsistent")
            client, bucket, prefix = _client(destination)
            if _list(client, bucket, prefix):
                published, published_hash = _read_commit(
                    destination, verify_hashes=verify_payload_hashes,
                )
            else:
                require_submit_authority("publish a verified backlog export")
                published, published_hash = _publish_committed_prefix(
                    candidate, destination,
                    verify_payload_hashes=verify_payload_hashes,
                    copy_workers=max(1, PUBLISH_COPY_WORKERS // effective_workers),
                )
            if published != commit or published_hash != manifest_hash:
                raise ValueError("Published backlog export differs from candidate")
            # One OSMO export workflow may own many per-sequence runs. A
            # committed candidate terminalizes only this run; OSMO refresh is
            # the sole authority for the shared workflow execution status.
            db.update_stage_run(
                run["stage_run_id"], stage=db.EXPORT_STAGE,
                status="SUCCEEDED", details="export_committed", db_path=db_path,
            )
            db.update_stage_request(
                request["id"], status="SUCCEEDED",
                result_manifest_uri=destination.rstrip("/") + "/commit.json",
                result_manifest_sha256=published_hash,
                result_summary={
                    "file_count": commit.get("file_count"),
                    "total_bytes": commit.get("total_bytes"),
                    "export_validation": commit.get("export_validation"),
                }, db_path=db_path,
            )
            cleared = _resolve_blacklist_after_committed_export(
                request, campaign, db_path=db_path,
            )
            return {
                "request_id": request["id"], "status": "SUCCEEDED",
                "export_run_id": run["stage_run_id"],
                "blacklist_cleared": cleared,
            }
        except Exception as exc:
            detail = f"candidate_publish_failed: {exc}"
            db.update_stage_run(
                run["stage_run_id"], status="FAILED", details=detail,
                stage=db.EXPORT_STAGE, db_path=db_path,
            )
            db.update_stage_request(
                request["id"], status="FAILED", details=detail, db_path=db_path,
            )
            _blacklist_campaign_failure(request, campaign, detail, db_path=db_path)
            return {"request_id": request["id"], "status": "FAILED", "details": detail}

    if not selected:
        return []
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        return list(executor.map(publish, selected))


def cleanup_promoted_staging(
    campaign_name: str, *, db_path: str, config: dict, apply: bool = False,
    workers: int = DEFAULT_CAMPAIGN_QUERY_WORKERS,
) -> dict:
    """Clean only DB-linked staging prefixes with successful final exports."""

    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if campaign is None:
        raise ValueError(f"Unknown campaign: {campaign_name}")
    if workers < 1:
        raise ValueError("Staging cleanup workers must be positive")
    dataset_cfg = config["datasets"][campaign["dataset"]]
    request_stage = (
        "revalidation"
        if campaign["campaign_type"] == "LEGACY_REVALIDATION"
        else "export"
    )
    requests = db.list_stage_requests(
        campaign=campaign_name, stage=request_stage,
        status="SUCCEEDED", db_path=db_path,
    )

    def inspect(request: dict) -> dict | None:
        run = db.get_stage_run_by_request(
            request["id"], stage=db.EXPORT_STAGE, db_path=db_path,
        )
        if run is None or run.get("run_status") != "SUCCEEDED":
            return None
        if request["stage"] == "export":
            parameters = json.loads(request.get("parameters_json") or "{}")
            candidate = parameters.get("candidate_uri")
        elif request["stage"] == "revalidation":
            candidate = _result_destination(request, campaign, dataset_cfg)
        else:
            return None
        destination = run.get("output_uri")
        if not candidate or not destination or candidate.rstrip("/") == destination.rstrip("/"):
            return None
        client, bucket, prefix = _client(candidate)
        objects = _list(client, bucket, prefix)
        if not objects:
            return None
        verify_remote = _read_commit(destination, verify_hashes=False)
        result = {
            "request_id": request["id"],
            "export_run_id": run["stage_run_id"],
            "sequence_name": request["sequence_name"],
            "candidate_uri": candidate,
            "destination_uri": destination,
            "object_count": len(objects),
            "total_bytes": sum(int(item["size"]) for item in objects),
            "final_manifest_sha256": verify_remote[1],
        }
        if apply:
            try:
                result.update(
                    cleanup_promoted_export_candidate(candidate, destination)
                )
                result["status"] = "CLEANED"
            except CandidateCleanupError as exc:
                result["status"] = "CLEANUP_PENDING"
                result["details"] = str(exc)
        else:
            result["status"] = "WOULD_CLEAN"
        return result

    with ThreadPoolExecutor(
        max_workers=min(workers, max(1, len(requests))),
    ) as executor:
        candidates = [
            result for result in executor.map(inspect, requests)
            if result is not None
        ]
    candidates.sort(key=lambda item: int(item["request_id"]))
    return {
        "campaign": campaign_name,
        "apply": apply,
        "candidate_count": len(candidates),
        "object_count": sum(item["object_count"] for item in candidates),
        "total_bytes": sum(item["total_bytes"] for item in candidates),
        "cleaned_count": sum(
            item["status"] == "CLEANED" for item in candidates
        ),
        "cleanup_pending_count": sum(
            item["status"] == "CLEANUP_PENDING" for item in candidates
        ),
        "reclaimed_bytes": sum(
            int(item.get("reclaimed_bytes") or 0) for item in candidates
        ),
        "candidates": candidates,
    }


def publish_canary_results(
    campaign_name: str, *, db_path: str, config: dict,
    verify_payload_hashes: bool = False, if_approved: bool = False,
) -> list[dict]:
    """Publish approved canary results without rerunning FoundationPose."""
    require_submit_authority("publish approved canary exports")
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or campaign["campaign_type"] != "LEGACY_REVALIDATION":
        raise ValueError("Canary publication requires a legacy revalidation campaign")
    if campaign["phase"] != "BULK" or not campaign["canary_approved_by"]:
        if if_approved:
            return []
        raise ValueError("Canary results cannot publish before recorded approval")
    dataset_cfg = config["datasets"][campaign["dataset"]]
    requests = db.list_stage_requests(
        campaign=campaign_name, stage="revalidation", status="SUCCEEDED",
        db_path=db_path,
    )
    results: list[dict] = []
    for request in (item for item in requests if item["cohort"] == "CANARY"):
        if _revalidation_repair_active(request, campaign, dataset_cfg):
            results.append({
                "request_id": request["id"], "status": "REPAIR_ACTIVE",
            })
            continue
        if db.get_stage_run_by_request(
            request["id"], stage=db.EXPORT_STAGE, db_path=db_path,
        ):
            results.append({"request_id": request["id"], "status": "ALREADY_PUBLISHED"})
            continue
        validate_frozen_objects(request, dataset_cfg)
        source = str(request["result_manifest_uri"]).rsplit("/", 1)[0]
        destination = _join_url(campaign["output_uri"], request["sequence_name"])
        commit, manifest_hash = _publish_committed_prefix(
            source, destination, verify_payload_hashes=verify_payload_hashes,
        )
        if (
            int(commit.get("request_id", -1)) != request["id"]
            or commit.get("source_manifest_sha256") != request["source_manifest_sha256"]
            or commit.get("configuration_sha256") != campaign["configuration_sha256"]
        ):
            raise ValueError("Staged canary commit provenance does not match its request")
        reconstruction = db.get_latest_successful_stage_run(
            request["sequence_name"], campaign["dataset"], RECON_PIPELINE,
            db_path=db_path,
        )
        if not reconstruction:
            raise ValueError("Canary lost its current legacy reconstruction dependency")
        committed_export = db.commit_revalidation_export(
            request["id"], reconstruction_run_id=reconstruction["stage_run_id"],
            result_manifest_uri=destination.rstrip("/") + "/commit.json",
            result_manifest_sha256=manifest_hash,
            result_summary=json.loads(request["result_summary_json"] or "{}"),
            source_uri=reconstruction["output_uri"], output_uri=destination,
            details="approved_canary_published", db_path=db_path,
        )
        _resolve_blacklist_after_committed_export(
            request, campaign, db_path=db_path,
        )
        results.append({
            "request_id": request["id"], "status": "PUBLISHED",
            "export_run_id": committed_export["stage_run_id"],
        })
    return results


def dispatch_backlog(
    campaign_name: str, *, db_path: str, config: dict,
    preprocess_limit: int | None = None,
    reconstruction_limit: int | None = None,
    force_blacklist: bool = False, dry_run: bool = False,
    requested_stage: str | None = None,
    sequences: Iterable[str] | None = None,
    pool: str | None = None,
    pool_selector: PoolSelector | None = None,
    submit_workers: int = DEFAULT_CAMPAIGN_SUBMIT_WORKERS,
) -> list[dict]:
    if submit_workers < 1:
        raise ValueError("Campaign submission workers must be positive")
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or campaign["campaign_type"] not in ("BACKLOG_REPROCESSING", "REMEDIATION"):
        raise ValueError("Backlog dispatch requires a backlog or remediation campaign")
    dataset_cfg = config["datasets"][campaign["dataset"]]
    pending = db.list_stage_requests(
        campaign=campaign_name, status="PENDING", db_path=db_path,
    )
    pending = [item for item in pending if _retry_is_ready(item)]
    selected_sequences = set(sequences or ())
    if selected_sequences:
        campaign_sequences = {
            item["sequence_name"]
            for item in db.list_stage_requests(
                campaign=campaign_name, db_path=db_path,
            )
        }
        unknown = selected_sequences - campaign_sequences
        if unknown:
            raise ValueError(
                "Requested backlog sequence(s) are not campaign members: "
                + ", ".join(sorted(unknown))
            )
        pending = [
            item for item in pending
            if item["sequence_name"] in selected_sequences
        ]
    stage_pipelines = (
        ("preprocess", PREPROCESS_PIPELINE, preprocess_limit),
        ("reconstruction", RECON_PIPELINE, reconstruction_limit),
    )
    candidates: list[dict] = []
    for stage, pipeline, submission_limit in stage_pipelines:
        if requested_stage is not None and stage != requested_stage:
            continue
        pipeline_config = dataset_cfg["pipelines"][pipeline]
        maximum = int(
            pipeline_config.get(
                "campaign_max_concurrent",
                pipeline_config.get("max_concurrent", 20),
            )
        )
        if submission_limit is not None and submission_limit < 0:
            raise ValueError(f"{stage} dispatch limit cannot be negative")
        active_count = db.count_active_stage_executions(
            stage, db_path=db_path,
        )
        capacity = max(0, maximum - active_count)
        if submission_limit is not None:
            capacity = min(capacity, submission_limit)
        stage_candidates = [
            item for item in pending if item["stage"] == stage
        ]
        stage_candidates.sort(key=lambda item: (
            -int(item.get("queue_priority") or 0),
            item.get("created_at") or "",
            int(item["id"]),
        ))
        stage_candidates = stage_candidates[:capacity]
        candidates.extend(stage_candidates)
    submit.DB_PATH = db_path
    selector = pool_selector or PoolSelector.collect(
        dataset_cfg,
        active_counts=load_active_counts(
            dataset_cfg, lambda: db.list_workflow_executions(
                status=("SUBMITTING", "RUNNING", "UNKNOWN"), db_path=db_path,
            ),
        ),
    )
    dispatch_started = time.perf_counter()
    first_submission_logged = False
    submission_log_lock = threading.Lock()

    def log_submission_start(request: dict) -> None:
        nonlocal first_submission_logged
        with submission_log_lock:
            if first_submission_logged:
                return
            first_submission_logged = True
            print(
                f"campaign_submission first stage={request['stage']} "
                f"request_id={request['id']} "
                f"dispatch_elapsed={time.perf_counter() - dispatch_started:.3f}s",
                flush=True,
            )

    def submit_request(request: dict) -> dict:
        pipeline = {
            "preprocess": PREPROCESS_PIPELINE,
            "reconstruction": RECON_PIPELINE,
        }.get(request["stage"])
        if not pipeline:
            return {
                "request_id": request["id"], "workflow_name": None,
                "status": "SKIPPED",
            }
        if pipeline == PREPROCESS_PIPELINE:
            try:
                validate_frozen_raw_input(request, dataset_cfg)
            except Exception as exc:
                if dry_run:
                    return {
                        "request_id": request["id"], "workflow_name": None,
                        "status": "DRY_RUN_BLOCKED", "details": str(exc),
                    }
                db.update_stage_request(
                    request["id"], status="BLOCKED", blocked_reason="SOURCE_CHANGED",
                    details=str(exc), db_path=db_path,
                )
                return {
                    "request_id": request["id"], "workflow_name": None,
                    "status": "BLOCKED", "details": str(exc),
                }
        log_submission_start(request)
        outcome = submit.submit_sequence(
            request["sequence_name"], campaign["dataset"], dataset_cfg, pipeline,
            # Frozen backlog/remediation membership is the audited authorization
            # to retry a blacklisted sequence.  Keep the blacklist row until a
            # verified export commits, but do not let it suppress campaign work.
            force=True, dry_run=dry_run,
            pipeline_version=campaign["pipeline_version"], trigger="MIGRATION",
            request_id=request["id"],
            pool=pool,
            pool_selector=selector,
        )
        if (
            outcome is not None
            and getattr(outcome, "prereq_skipped", False)
            and request.get("cohort") == "CANARY"
            and not dry_run
        ):
            # A frozen canary may intentionally exercise a source-data
            # outcome.  Normal campaign members remain BLOCKED so corrected
            # prerequisites can be picked up later, but a canary observation
            # must be terminal before it can participate in automated paired
            # acceptance.  Preserve the exact prerequisite evidence written
            # by submit_sequence and classify it explicitly without creating
            # a fake workflow execution.
            observed = db.get_stage_request(request["id"], db_path=db_path)
            reason = str(
                observed.get("blocked_reason")
                or observed.get("details")
                or "source-data prerequisite failed"
            )
            if reason.startswith(_NON_DATA_PREREQUISITE_PREFIXES):
                # These are campaign/runtime lineage invariants, not properties
                # of the source sequence.  Keep the request blocked so the
                # implementation-remediation loop cannot accidentally approve
                # an orchestration defect as an expected data outcome.
                return {
                    "request_id": request["id"],
                    "workflow_name": None,
                    "status": "IMPLEMENTATION_BLOCKED",
                    "details": reason,
                }
            detail = f"expected_canary_source_data_failure: {reason}"
            db.update_stage_request(
                request["id"], status="FAILED", details=detail,
                result_summary={
                    "canary_outcome": "EXPECTED_QC_DATA",
                    "failure_category": "source_data",
                    "reason": reason,
                    "evidence_valid": True,
                },
                db_path=db_path,
            )
            return {
                "request_id": request["id"],
                "workflow_name": None,
                "status": "EXPECTED_QC_DATA",
                "details": reason,
            }
        return {
            "request_id": request["id"],
            "workflow_name": outcome.workflow_name if outcome else None,
        }

    with ThreadPoolExecutor(
        max_workers=min(submit_workers, max(1, len(candidates))),
    ) as executor:
        return list(executor.map(submit_request, candidates))


def run_campaign_cycle(
    *, revalidation_campaign: str | None, backlog_campaign: str | None,
    db_path: str, config: dict, verify_payload_hashes: bool = False,
    pool: str | None = None,
    query_workers: int = DEFAULT_CAMPAIGN_QUERY_WORKERS,
    submit_workers: int = DEFAULT_CAMPAIGN_SUBMIT_WORKERS,
    publish_workers: int = DEFAULT_REVALIDATION_PUBLISH_WORKERS,
) -> dict:
    """Refresh, advance, publish, and replenish both frozen campaigns once."""
    if min(query_workers, submit_workers, publish_workers) < 1:
        raise ValueError("Campaign worker counts must be positive")
    cycle_started = time.perf_counter()
    results: dict[str, object] = {}

    def phase(name: str, operation):
        started = time.perf_counter()
        print(
            f"campaign_phase {name} start "
            f"cycle_elapsed={started - cycle_started:.3f}s",
            flush=True,
        )
        value = operation()
        count = len(value) if isinstance(value, (list, tuple, dict)) else None
        suffix = f" count={count}" if count is not None else ""
        print(
            f"campaign_phase {name} done elapsed={time.perf_counter() - started:.3f}s"
            f"{suffix}",
            flush=True,
        )
        return value

    def selector_for_dispatch(dataset: str) -> PoolSelector:
        """Take one fresh pool snapshot for each independent dispatch pass."""
        dataset_cfg = config["datasets"][dataset]
        return PoolSelector.collect(
            dataset_cfg,
            active_counts=load_active_counts(
                dataset_cfg, lambda: db.list_workflow_executions(
                    status=("SUBMITTING", "RUNNING", "UNKNOWN"),
                    db_path=db_path,
                ),
            ),
        )

    def enqueue_cleanup_exports(
        settings: dict | None, outcomes: Iterable[dict], *, campaign_name: str,
    ) -> list[dict]:
        if not settings:
            return []
        enqueued = []
        for export_run_id in dict.fromkeys(
            int(item["export_run_id"])
            for item in outcomes if item.get("export_run_id") is not None
        ):
            enqueued.append(db.enqueue_intermediate_cleanup(
                export_run_id, source="AUTOMATIC", db_path=db_path,
            ))
        enqueued.extend(db.enqueue_missing_intermediate_cleanups(
            campaign_name,
            completed_after=settings["completed_after"],
            source="AUTOMATIC", db_path=db_path,
        ))
        return enqueued
    revalidation = None
    revalidation_cleanup = None
    if revalidation_campaign:
        revalidation = db.get_campaign(revalidation_campaign, db_path=db_path)
        if not revalidation:
            raise ValueError(f"Unknown campaign: {revalidation_campaign}")
        revalidation_phase = revalidation.get("phase", "BULK")
        revalidation_retries = phase(
            "revalidation_infrastructure_recovery",
            lambda: recover_terminal_campaign_infrastructure_failures(
                revalidation_campaign, db_path=db_path,
                query_workers=query_workers,
            ),
        )
        results["revalidation_infrastructure_retries"] = revalidation_retries
        revalidation_skip = {
            int(result["request_id"])
            for result in revalidation_retries
            if result["status"] in ("RETRY_CREATED", "REMOTE_RUNNING_RECOVERED")
        }
        results["revalidation_reconcile"] = phase(
            "revalidation_observe",
            lambda: reconcile_revalidation(
                revalidation_campaign, db_path=db_path,
                verify_payload_hashes=verify_payload_hashes,
                retry_infrastructure=True,
                recover_publication_failures=True,
                skip_request_ids=revalidation_skip,
                query_workers=query_workers,
                publish_workers=publish_workers,
                defer_bulk_publication=(revalidation_phase == "BULK"),
            ),
        )
        if revalidation_phase == "CANARY":
            results["revalidation_canary_publish"] = phase(
                "revalidation_canary_publish",
                lambda: publish_canary_results(
                    revalidation_campaign, db_path=db_path, config=config,
                    verify_payload_hashes=verify_payload_hashes,
                    if_approved=True,
                ),
            )
        revalidation_cleanup = get_cleanup_settings(
            config, revalidation["dataset"], revalidation_campaign,
        )
        results["revalidation_dispatch"] = phase(
            "revalidation_dispatch",
            lambda: dispatch_revalidation(
                revalidation_campaign, db_path=db_path, config=config, pool=pool,
                pool_selector=selector_for_dispatch(revalidation["dataset"]),
                submit_workers=submit_workers,
            ),
        )
    backlog_cleanup = None
    if backlog_campaign:
        campaign = db.get_campaign(backlog_campaign, db_path=db_path)
        if not campaign:
            raise ValueError(f"Unknown campaign: {backlog_campaign}")
        dataset = campaign["dataset"]
        backlog_promotion = get_backlog_promotion_settings(config, dataset)
        backlog_retries = phase(
            "backlog_infrastructure_recovery",
            lambda: recover_terminal_campaign_infrastructure_failures(
                backlog_campaign, db_path=db_path,
                query_workers=query_workers,
            ),
        )
        results["backlog_infrastructure_retries"] = backlog_retries
        backlog_skip = {
            int(result["request_id"])
            for result in backlog_retries
            if result["status"] in ("RETRY_CREATED", "REMOTE_RUNNING_RECOVERED")
        }
        phase(
            "backlog_refresh",
            lambda: refresh_workflow_states(
                dataset, pipeline_type=None, db_path=db_path,
                max_workers=query_workers,
                retry_infrastructure=True,
                skip_request_ids=backlog_skip,
            ),
        )
        backlog_cleanup = get_cleanup_settings(config, dataset, backlog_campaign)
        results["backlog_advance"] = phase(
            "backlog_advance",
            lambda: advance_backlog(
                backlog_campaign, db_path=db_path, config=config,
                query_workers=query_workers,
            ),
        )
        results["backlog_dispatch"] = phase(
            "backlog_dispatch",
            lambda: dispatch_backlog(
                backlog_campaign, db_path=db_path, config=config, pool=pool,
                pool_selector=selector_for_dispatch(dataset),
                submit_workers=submit_workers,
            ),
        )
        results["backlog_publish"] = phase(
            "backlog_publish",
            lambda: publish_backlog_exports(
                backlog_campaign, db_path=db_path,
                verify_payload_hashes=verify_payload_hashes,
                max_per_cycle=backlog_promotion["limit"],
                workers=backlog_promotion["workers"],
            ),
        )
        results["backlog_cleanup_enqueued"] = enqueue_cleanup_exports(
            backlog_cleanup, results["backlog_publish"],
            campaign_name=backlog_campaign,
        )
        dataset_cfg = config["datasets"][dataset]
        export_controller.DB_PATH = db_path
        phase(
            "backlog_export_schedule",
            lambda: export_controller.run_export(
                dataset, dataset_cfg, db_path=db_path,
                campaign=backlog_campaign, pool=pool,
                pool_selector=selector_for_dispatch(dataset),
                refresh=False,
            ),
        )
        results["backlog_export"] = "checked"
    if revalidation_campaign and revalidation:
        results["revalidation_publish"] = phase(
            "revalidation_publish",
            lambda: reconcile_revalidation(
                revalidation_campaign, db_path=db_path,
                verify_payload_hashes=verify_payload_hashes,
                recover_publication_failures=True,
                query_workers=query_workers,
                publish_workers=publish_workers,
                pending_publication_only=True,
            ),
        )
        if revalidation_phase != "CANARY":
            results["revalidation_canary_publish"] = phase(
                "revalidation_canary_publish",
                lambda: publish_canary_results(
                    revalidation_campaign, db_path=db_path, config=config,
                    verify_payload_hashes=verify_payload_hashes,
                    if_approved=True,
                ),
            )
        results["revalidation_cleanup_enqueued"] = enqueue_cleanup_exports(
            revalidation_cleanup,
            list(results["revalidation_publish"])
            + list(results.get("revalidation_canary_publish") or ()),
            campaign_name=revalidation_campaign,
        )
        if revalidation_cleanup and revalidation_cleanup["mode"] == "inline":
            results["revalidation_cleanup"] = phase(
                "revalidation_cleanup",
                lambda: cleanup_intermediates.run_cleanup_jobs(
                    revalidation["dataset"], campaign=revalidation_campaign,
                    db_path=db_path, apply=True,
                    limit=revalidation_cleanup["limit"],
                    workers=revalidation_cleanup["workers"],
                ),
            )
        elif revalidation_cleanup:
            results["revalidation_cleanup"] = "deferred_to_async_worker"
    if (backlog_campaign and backlog_cleanup
            and backlog_cleanup["mode"] == "inline"):
        results["backlog_cleanup"] = phase(
            "backlog_cleanup",
            lambda: cleanup_intermediates.run_cleanup_jobs(
                dataset, campaign=backlog_campaign, db_path=db_path, apply=True,
                limit=backlog_cleanup["limit"],
                workers=backlog_cleanup["workers"],
            ),
        )
    elif backlog_campaign and backlog_cleanup:
        results["backlog_cleanup"] = "deferred_to_async_worker"
    print(
        f"campaign_cycle done elapsed={time.perf_counter() - cycle_started:.3f}s",
        flush=True,
    )
    return results


def _bbox_objects(objects: Iterable[dict]) -> list[dict]:
    prefix = "mv_preprocess/labeled_bboxes/"
    return sorted(
        [
            {
                "relative_path": item["relative_path"],
                "size": int(item["size"]),
                "etag": item.get("etag") or "",
                "sha256": item.get("sha256"),
                **(
                    {"schema_valid": item["schema_valid"]}
                    if "schema_valid" in item else {}
                ),
                **(
                    {"schema_error": item["schema_error"]}
                    if "schema_error" in item else {}
                ),
                **(
                    {"detection_count": item["detection_count"]}
                    if "detection_count" in item else {}
                ),
            }
            for item in objects
            if item["relative_path"].startswith(prefix)
            and item["relative_path"].endswith(".json")
        ],
        key=lambda item: item["relative_path"],
    )


def _canonical_preprocess_output_url(
    dataset_cfg: dict, sequence_name: str,
) -> str:
    pipeline = dataset_cfg["pipelines"][PREPROCESS_PIPELINE]
    output_path = pipeline.get("campaign_output_path") or pipeline.get("output_path")
    if not output_path:
        raise ValueError("Preprocessing must define a canonical output path")
    if (
        pipeline.get("output_path")
        and str(output_path).strip("/") != str(pipeline["output_path"]).strip("/")
    ):
        raise ValueError("Preprocess campaign and mainline output paths must match")
    return _join_url(dataset_cfg["swift_base"], output_path, sequence_name)


def _inspect_bbox_payload(payload: bytes) -> dict:
    """Validate the storage-independent portion of one manual bbox file."""
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"schema_valid": False, "schema_error": f"invalid JSON: {exc}"}
    if not isinstance(decoded, dict) or not decoded:
        return {
            "schema_valid": False,
            "schema_error": "bbox payload must be a nonempty frame mapping",
        }
    detections_seen = 0
    try:
        for frame_stem, detections in decoded.items():
            frame_index = int(frame_stem)
            if frame_index < 0 or not isinstance(detections, list):
                raise ValueError("frame keys must be nonnegative and map to lists")
            for detection in detections:
                if not isinstance(detection, dict):
                    raise ValueError("detections must be objects")
                box = detection.get("box")
                if not isinstance(box, dict):
                    raise ValueError("each detection must contain a box object")
                x0, y0, x1, y1 = (
                    float(box[name]) for name in ("x0", "y0", "x1", "y1")
                )
                if not (0 <= x0 < x1 and 0 <= y0 < y1):
                    raise ValueError("bbox coordinates must be ordered and nonnegative")
                detections_seen += 1
    except (KeyError, TypeError, ValueError) as exc:
        return {"schema_valid": False, "schema_error": str(exc)}
    if detections_seen == 0:
        return {
            "schema_valid": False,
            "schema_error": "bbox payload contains no detections",
        }
    return {"schema_valid": True, "detection_count": detections_seen}


def _current_sequence_objects(
    dataset_cfg: dict, request: dict, client=None,
) -> list[dict]:
    output_uri = _canonical_preprocess_output_url(
        dataset_cfg, request["sequence_name"],
    )
    if client is None:
        client, bucket, root = _client(output_uri)
    else:
        _, bucket, root = _parse_swift_url(output_uri)
    objects = []
    bbox_root = f"{root.rstrip('/')}/mv_preprocess/labeled_bboxes"
    for item in _list(client, bucket, bbox_root):
        if not item["key"].startswith(root.rstrip("/") + "/"):
            continue
        relative_path = item["key"][len(root.rstrip('/')) + 1:]
        record = {**item, "relative_path": relative_path}
        if (
            relative_path.startswith("mv_preprocess/labeled_bboxes/")
            and relative_path.endswith(".json")
        ):
            body = client.get_object(Bucket=bucket, Key=item["key"])["Body"].read()
            record["sha256"] = hashlib.sha256(body).hexdigest()
            record.update(_inspect_bbox_payload(body))
        objects.append(record)
    marker_key = f"{root.rstrip('/')}/mv_preprocess/object_bbox_source.txt"
    try:
        response = client.get_object(Bucket=bucket, Key=marker_key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in (
            "404", "NoSuchKey", "NotFound",
        ):
            raise
    else:
        body = response["Body"].read()
        objects.append({
            "key": marker_key,
            "relative_path": "mv_preprocess/object_bbox_source.txt",
            "size": len(body),
            "etag": str(response.get("ETag") or "").strip('"'),
            "sha256": hashlib.sha256(body).hexdigest(),
            "content": body.decode("utf-8").strip(),
        })
    return objects


def _restore_transferred_manual_bbox_marker(
    dataset_cfg: dict, request: dict, current_objects: list[dict],
) -> tuple[list[dict], dict | None]:
    """Restore only a proven marker removed by transferred fresh preprocess.

    The human-authored bbox payloads themselves are never copied or changed.
    Every one must match the immutable pre-transfer inventory by path, size,
    and ETag and pass the current schema validator before the marker is written.
    """
    if any(
        item.get("relative_path") == "mv_preprocess/object_bbox_source.txt"
        for item in current_objects
    ):
        return current_objects, None
    try:
        frozen = json.loads(request.get("source_manifest_json") or "{}")
    except (TypeError, ValueError):
        return current_objects, None
    transfer = frozen.get("membership_transfer")
    lineage = frozen.get("historical_preprocess_lineage")
    if not (
        request.get("queue_priority") == 100
        and request.get("stage") == "preprocess"
        and frozen.get("route") == "reprocessing_missing_revalidation_input"
        and isinstance(transfer, dict)
        and transfer.get("source_campaign") == "legacy_revalidation"
        and frozen.get("labeled_bboxes_ready") is True
        and isinstance(lineage, dict)
        and lineage.get("reuse_permitted") is False
    ):
        return current_objects, None
    required = {
        "back_stereo_camera_left", "front_stereo_camera_left",
        "left_stereo_camera_left", "right_stereo_camera_left",
    }
    frozen_bboxes = {
        Path(item.get("relative_path") or "").stem: item
        for item in frozen.get("data_output_objects") or []
        if (
            str(item.get("relative_path") or "").startswith(
                "mv_preprocess/labeled_bboxes/"
            )
            and str(item.get("relative_path") or "").endswith(".json")
        )
    }
    current_bboxes = {
        Path(item.get("relative_path") or "").stem: item
        for item in current_objects
        if (
            str(item.get("relative_path") or "").startswith(
                "mv_preprocess/labeled_bboxes/"
            )
            and str(item.get("relative_path") or "").endswith(".json")
        )
    }
    if set(frozen_bboxes) != required or set(current_bboxes) != required:
        return current_objects, None
    for camera in sorted(required):
        expected = frozen_bboxes[camera]
        observed = current_bboxes[camera]
        if (
            observed.get("schema_valid") is not True
            or int(observed.get("size") or -1) != int(expected.get("size") or -2)
            or str(observed.get("etag") or "") != str(expected.get("etag") or "")
        ):
            return current_objects, None

    output_uri = _canonical_preprocess_output_url(
        dataset_cfg, request["sequence_name"],
    )
    client, bucket, root = _client(output_uri)
    key = f"{root.rstrip('/')}/mv_preprocess/object_bbox_source.txt"
    body = (submit.OBJECT_BBOX_SOURCE_MANUAL + "\n").encode()
    require_submit_authority(
        "restore frozen manual bbox provenance after fresh preprocessing"
    )
    response = client.put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="text/plain",
    )
    marker = {
        "key": key,
        "relative_path": "mv_preprocess/object_bbox_source.txt",
        "size": len(body),
        "etag": str(response.get("ETag") or "").strip('"'),
        "sha256": hashlib.sha256(body).hexdigest(),
        "content": submit.OBJECT_BBOX_SOURCE_MANUAL,
    }
    evidence = {
        "schema": "v2d.mv_hoi.transferred_bbox_marker_restoration.v1",
        "preprocess_request_id": request["id"],
        "historical_preprocess_run_id": lineage.get("preprocess_run_id"),
        "reuse_permitted": False,
        "marker_uri": output_uri.rstrip("/")
        + "/mv_preprocess/object_bbox_source.txt",
        "marker_sha256": marker["sha256"],
        "frozen_bbox_identities": [
            {
                "relative_path": frozen_bboxes[camera]["relative_path"],
                "size": frozen_bboxes[camera]["size"],
                "etag": frozen_bboxes[camera]["etag"],
            }
            for camera in sorted(required)
        ],
    }
    return [*current_objects, marker], evidence


def advance_backlog(
    campaign_name: str, *, db_path: str, config: dict,
    query_workers: int = DEFAULT_CAMPAIGN_QUERY_WORKERS,
) -> list[dict]:
    """Advance successful preprocessing into label validation/reconstruction intent."""
    if query_workers < 1:
        raise ValueError("Campaign query workers must be positive")
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or campaign["campaign_type"] not in ("BACKLOG_REPROCESSING", "REMEDIATION"):
        raise ValueError("Backlog advancement requires a backlog/remediation campaign")
    dataset_cfg = config["datasets"][campaign["dataset"]]
    requests = db.list_campaign_advancement_requests(
        campaign_name, db_path=db_path,
    )
    by_sequence: dict[str, list[dict]] = {}
    for request in requests:
        by_sequence.setdefault(request["sequence_name"], []).append(request)
    current_preprocess_runs = {
        item["sequence_name"]: item
        for item in db.list_current_successful_preprocess_lineage(
            campaign["dataset"], db_path=db_path,
        )
    }
    label_storage = threading.local()
    candidates: list[tuple[str, dict, dict | None, int, bool]] = []
    for sequence, sequence_requests in by_sequence.items():
        preprocess = next(
            (item for item in reversed(sequence_requests)
             if item["stage"] == "preprocess" and item["status"] == "SUCCEEDED"),
            None,
        )
        reconstruction_requests = [
            item for item in sequence_requests if item["stage"] == "reconstruction"
        ]
        if not preprocess and not reconstruction_requests:
            continue
        matching_reconstruction_requests = []
        if preprocess:
            for item in reconstruction_requests:
                try:
                    parameters = json.loads(item.get("parameters_json") or "{}")
                except (TypeError, ValueError):
                    parameters = {}
                if parameters.get("preprocess_request_id") == preprocess["id"]:
                    matching_reconstruction_requests.append(item)
        current_request = (
            matching_reconstruction_requests[-1]
            if matching_reconstruction_requests else None
        )
        if current_request and current_request["status"] not in ("BLOCKED",):
            continue
        if current_request is None and any(
            item["status"] in ACTIVE for item in reconstruction_requests
        ):
            # A retry recovery must explicitly retire any reconstruction intent
            # tied to the superseded preprocess lineage before a replacement is
            # created. Never race two reconstruction requests for one sequence.
            continue
        if preprocess is not None:
            candidates.append((
                sequence, preprocess, current_request, int(preprocess["id"]), True,
            ))
            continue

        # Reconstruction-first inventory members reuse a validated current
        # preprocess run whose request may belong to an earlier campaign.  A
        # blocked reconstruction request must remain eligible for label
        # refresh without requiring a synthetic preprocess request in the
        # current campaign.  Validate every persisted lineage field against
        # the current successful run before trusting that reuse.
        current_request = reconstruction_requests[-1]
        if current_request["status"] != "BLOCKED":
            continue
        try:
            parameters = json.loads(current_request.get("parameters_json") or "{}")
            expected_request_id = int(parameters["preprocess_request_id"])
            expected_run_id = int(parameters["preprocess_run_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        current_run = current_preprocess_runs.get(sequence)
        if not (
            parameters.get("preprocess_reused") is True
            and current_run
            and int(current_run["stage_run_id"]) == expected_run_id
            and int(current_run.get("request_id") or -1) == expected_request_id
            and str(current_run.get("output_uri") or "").rstrip("/")
            == str(parameters.get("preprocess_output_uri") or "").rstrip("/")
        ):
            continue
        candidates.append((
            sequence, current_request, current_request,
            expected_request_id, False,
        ))

    def inspect_labels(
        candidate: tuple[str, dict, dict | None, int, bool],
    ) -> dict:
        (
            sequence, storage_request, current_request,
            preprocess_request_id, allow_marker_restoration,
        ) = candidate
        if not hasattr(label_storage, "client"):
            label_storage.client = _client(dataset_cfg["swift_base"])[0]
        current_objects = _current_sequence_objects(
            dataset_cfg, storage_request, label_storage.client,
        )
        marker_restoration = None
        if allow_marker_restoration:
            restoration_request = storage_request
            if (
                storage_request.get("queue_priority") == 100
                and not storage_request.get("source_manifest_json")
                and not any(
                    item.get("relative_path")
                    == "mv_preprocess/object_bbox_source.txt"
                    for item in current_objects
                )
            ):
                restoration_request = db.get_stage_request(
                    storage_request["id"], db_path=db_path,
                ) or storage_request
            current_objects, marker_restoration = (
                _restore_transferred_manual_bbox_marker(
                    dataset_cfg, restoration_request, current_objects,
                )
            )
        current_bboxes = _bbox_objects(current_objects)
        marker = next(
            (
                item.get("content")
                for item in current_objects
                if item["relative_path"] == "mv_preprocess/object_bbox_source.txt"
            ),
            None,
        )
        camera_names = {
            Path(item["relative_path"]).stem for item in current_bboxes
        }
        required = {
            "back_stereo_camera_left", "front_stereo_camera_left",
            "left_stereo_camera_left", "right_stereo_camera_left",
        }
        if marker != submit.OBJECT_BBOX_SOURCE_MANUAL:
            request_status, blocker = "BLOCKED", "WAITING_LABELS"
            details = (
                "manual bbox provenance marker is required"
                if marker is None
                else f"untrusted bbox provenance marker: {marker}"
            )
        elif not required.issubset(camera_names):
            request_status, blocker = "BLOCKED", "WAITING_LABELS"
            details = "four left-camera labeled bbox JSON files are required"
        elif invalid := [
            item for item in current_bboxes
            if Path(item["relative_path"]).stem in required
            and item.get("schema_valid") is False
        ]:
            request_status, blocker = "BLOCKED", "WAITING_LABELS"
            details = (
                "invalid manual bbox schema: "
                + "; ".join(
                    f"{Path(item['relative_path']).name}: {item['schema_error']}"
                    for item in invalid
                )
            )
        else:
            request_status, blocker, details = "PENDING", None, "labels_ready"
        return {
            "sequence": sequence,
            "preprocess": storage_request,
            "preprocess_request_id": preprocess_request_id,
            "current_request": current_request,
            "status": request_status,
            "blocked_reason": blocker,
            "details": details,
            "manifest": {
                "preprocess_request_id": preprocess_request_id,
                "labeled_bboxes": current_bboxes,
                "camera_views": sorted(camera_names),
                "object_bbox_source": marker,
                "bbox_preview_required": True,
                "marker_restoration": marker_restoration,
            },
        }

    with ThreadPoolExecutor(
        max_workers=min(query_workers, max(1, len(candidates))),
    ) as executor:
        observations = list(executor.map(inspect_labels, candidates))

    # Storage reads are independent and concurrent; database transitions stay
    # ordered and per-request transactional so the result matches serial
    # advancement exactly.
    results = []
    for observation in observations:
        sequence = observation["sequence"]
        preprocess = observation["preprocess"]
        preprocess_request_id = observation["preprocess_request_id"]
        current_request = observation["current_request"]
        request_status = observation["status"]
        blocker = observation["blocked_reason"]
        details = observation["details"]
        manifest = observation["manifest"]
        if current_request:
            try:
                stored_manifest = json.loads(
                    current_request.get("source_manifest_json") or "{}"
                )
            except (TypeError, ValueError):
                stored_manifest = {}
            unchanged = (
                current_request["status"] == request_status
                and current_request.get("blocked_reason") == blocker
                and current_request.get("details") == details
                and _canonical(stored_manifest) == _canonical(manifest)
            )
            if not unchanged:
                db.update_stage_request(
                    current_request["id"], status=request_status,
                    blocked_reason=blocker, details=details,
                    source_manifest=manifest, db_path=db_path,
                )
            request_id = current_request["id"]
        else:
            created = db.create_stage_request(
                sequence_name=sequence, dataset=campaign["dataset"],
                stage="reconstruction", pipeline_version=campaign["pipeline_version"],
                trigger="MIGRATION", requested_by=campaign["created_by"],
                reason="backlog_after_preprocess", campaign=campaign["id"],
                cohort=preprocess.get("cohort") or "BULK",
                queue_priority=int(preprocess.get("queue_priority") or 0),
                parameters={
                    "preprocess_request_id": preprocess_request_id,
                    "preprocess_output_uri": _canonical_preprocess_output_url(
                        dataset_cfg, sequence,
                    ),
                },
                source_manifest=manifest, status=request_status,
                blocked_reason=blocker, db_path=db_path,
            )
            request_id = created["id"]
            db.update_stage_request(
                request_id, details=details, db_path=db_path,
            )
        results.append({
            "sequence": sequence, "request_id": request_id,
            "status": request_status, "blocked_reason": blocker,
        })
    return results


def build_failed_remediation_inventory(
    campaign_name: str, *, db_path: str,
) -> dict:
    """Freeze unresolved failed campaign members for a new remediation campaign."""
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign:
        raise ValueError(f"Unknown campaign: {campaign_name}")
    requests = db.list_stage_requests(campaign=campaign_name, db_path=db_path)
    by_sequence: dict[str, list[dict]] = {}
    for request in requests:
        by_sequence.setdefault(request["sequence_name"], []).append(request)
    current_status = {
        row["sequence_name"]: row
        for row in db.get_sequence_status(campaign["dataset"], db_path=db_path)
    }
    records: list[dict] = []
    resolved: list[str] = []
    for sequence, sequence_requests in sorted(by_sequence.items()):
        terminal_failures = [
            item for item in sequence_requests
            if item["status"] in ("FAILED", "CANCELED")
        ]
        if not terminal_failures:
            continue
        failure = terminal_failures[-1]
        status = current_status.get(sequence, {})
        if (
            status.get("latest_request_id") != failure["id"]
            or status.get("latest_request_status") not in ("FAILED", "CANCELED")
        ):
            resolved.append(sequence)
            continue
        source_record = None
        for item in sequence_requests:
            candidate = json.loads(item["source_manifest_json"] or "{}")
            if any(
                key in candidate for key in (
                    "data_output_objects", "data_export_objects", "raw_input_objects",
                )
            ):
                source_record = candidate
                break
        if source_record is None:
            raise ValueError(
                f"{sequence}: failed campaign member has no frozen source inventory record"
            )
        records.append({
            **source_record,
            "route": "remediation",
            "recommended_stage": "preprocess",
            "source_campaign_id": campaign["id"],
            "source_campaign_name": campaign["name"],
            "failed_request_id": failure["id"],
            "failed_stage": failure["stage"],
            "failed_status": failure["status"],
            "failed_details": failure["details"],
        })
    return {
        "schema": "v2d.mv_hoi.campaign_inventory.v1",
        "kind": "remediation",
        "dataset": campaign["dataset"],
        "source_campaign": campaign["name"],
        "source_campaign_inventory_sha256": campaign["inventory_sha256"],
        "cutover_at": datetime.now(timezone.utc).isoformat(),
        "sequence_count": len(records),
        "resolved_sequence_count": len(resolved),
        "resolved_sequences": resolved,
        "sequences": records,
    }


def write_failed_remediation_inventory(
    campaign_name: str, output_path: Path, *, db_path: str,
) -> dict:
    inventory = build_failed_remediation_inventory(campaign_name, db_path=db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(output_path)
    return {
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "sequence_count": inventory["sequence_count"],
        "resolved_sequence_count": inventory["resolved_sequence_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--config", type=Path, default=MV_HOI_DIR / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("campaign")
    enqueue.add_argument("--inventory", type=Path, required=True)
    dispatch = sub.add_parser("dispatch-revalidation")
    dispatch.add_argument("campaign")
    dispatch.add_argument("--limit", type=int)
    dispatch.add_argument("--pool", help="Explicit configured OSMO pool override")
    dispatch.add_argument("--dry-run", action="store_true")
    dispatch.add_argument("--clean-incomplete-destination", action="store_true")
    dispatch.add_argument(
        "--request-id", dest="request_ids", type=int, action="append",
        help="Dispatch only this exact pending request; repeat to select several.",
    )
    dispatch.add_argument(
        "--reuse-work-output-url",
        help=(
            "Run only the export retry using successful task outputs from this "
            "revalidation work-output URL"
        ),
    )
    reconcile = sub.add_parser("reconcile-revalidation")
    reconcile.add_argument("campaign")
    reconcile.add_argument("--verify-payload-hashes", action="store_true")
    reconcile.add_argument("--retry-infrastructure", action="store_true")
    reconcile.add_argument(
        "--recover-publication-failures", action="store_true",
        help=(
            "Revalidate and recover FAILED requests whose only recorded failure "
            "was committing an already-completed export"
        ),
    )
    publish_canary = sub.add_parser("publish-canary")
    publish_canary.add_argument("campaign")
    publish_canary.add_argument("--verify-payload-hashes", action="store_true")
    publish_canary.add_argument("--if-approved", action="store_true")
    backlog = sub.add_parser("dispatch-backlog")
    backlog.add_argument("campaign")
    backlog.add_argument("--preprocess-limit", type=int)
    backlog.add_argument("--reconstruction-limit", type=int)
    backlog.add_argument("--force-blacklist", action="store_true")
    backlog.add_argument("--dry-run", action="store_true")
    backlog.add_argument("--pool", help="Explicit configured OSMO pool override")
    backlog.add_argument(
        "--stage", dest="requested_stage",
        choices=("preprocess", "reconstruction"),
        help="Dispatch only one backlog stage; campaign cycles omit this filter.",
    )
    backlog.add_argument(
        "--sequence", dest="sequences", action="append",
        help=(
            "Dispatch only this exact campaign member; repeat for a scoped canary."
        ),
    )
    advance = sub.add_parser("advance-backlog")
    advance.add_argument("campaign")
    failed_inventory = sub.add_parser("failed-inventory")
    failed_inventory.add_argument("campaign")
    failed_inventory.add_argument("--output", type=Path, required=True)
    retry_infrastructure = sub.add_parser("retry-infrastructure")
    retry_infrastructure.add_argument("campaign")
    retry_infrastructure.add_argument(
        "--request-id", dest="request_ids", type=int, action="append",
        help="Retry only this exact failed request; repeat to select several.",
    )
    cleanup_staging = sub.add_parser("cleanup-promoted-staging")
    cleanup_staging.add_argument("campaign")
    cleanup_staging.add_argument("--apply", action="store_true")
    cleanup_staging.add_argument("--summary-only", action="store_true")
    cleanup_staging.add_argument(
        "--workers", type=int, default=DEFAULT_CAMPAIGN_QUERY_WORKERS,
    )
    cycle = sub.add_parser("cycle")
    cycle.add_argument("--revalidation-campaign")
    cycle.add_argument("--backlog-campaign")
    cycle.add_argument("--verify-payload-hashes", action="store_true")
    cycle.add_argument("--pool", help="Explicit configured OSMO pool override")
    cycle.add_argument(
        "--query-workers", type=int, default=DEFAULT_CAMPAIGN_QUERY_WORKERS,
    )
    cycle.add_argument(
        "--submit-workers", type=int, default=DEFAULT_CAMPAIGN_SUBMIT_WORKERS,
    )
    cycle.add_argument(
        "--publish-workers", type=int,
        default=DEFAULT_REVALIDATION_PUBLISH_WORKERS,
    )
    args = parser.parse_args()
    db.init_db(args.db)
    config = load_config(args.config.parent)
    if args.command == "enqueue":
        result = enqueue_campaign(args.campaign, args.inventory, db_path=args.db)
    elif args.command == "dispatch-revalidation":
        if not args.dry_run:
            require_submit_authority("dispatch campaign workflows")
        result = dispatch_revalidation(
            args.campaign, db_path=args.db, config=config, limit=args.limit,
            dry_run=args.dry_run,
            clean_incomplete_destination=args.clean_incomplete_destination,
            reuse_work_output_url=args.reuse_work_output_url,
            request_ids=set(args.request_ids) if args.request_ids else None,
            pool=args.pool,
        )
    elif args.command == "reconcile-revalidation":
        if args.retry_infrastructure:
            require_submit_authority("retry OSMO infrastructure failures")
        result = reconcile_revalidation(
            args.campaign, db_path=args.db,
            verify_payload_hashes=args.verify_payload_hashes,
            retry_infrastructure=args.retry_infrastructure,
            recover_publication_failures=args.recover_publication_failures,
        )
    elif args.command == "publish-canary":
        result = publish_canary_results(
            args.campaign, db_path=args.db, config=config,
            verify_payload_hashes=args.verify_payload_hashes,
            if_approved=args.if_approved,
        )
    elif args.command == "dispatch-backlog":
        if not args.dry_run:
            require_submit_authority("dispatch backlog workflows")
        result = dispatch_backlog(
            args.campaign, db_path=args.db, config=config,
            preprocess_limit=args.preprocess_limit,
            reconstruction_limit=args.reconstruction_limit,
            force_blacklist=args.force_blacklist, dry_run=args.dry_run,
            requested_stage=args.requested_stage,
            sequences=args.sequences,
            pool=args.pool,
        )
    elif args.command == "advance-backlog":
        result = advance_backlog(args.campaign, db_path=args.db, config=config)
    elif args.command == "failed-inventory":
        result = write_failed_remediation_inventory(
            args.campaign, args.output, db_path=args.db,
        )
    elif args.command == "retry-infrastructure":
        require_submit_authority("retry OSMO infrastructure failures")
        result = recover_terminal_campaign_infrastructure_failures(
            args.campaign, db_path=args.db,
            request_ids=set(args.request_ids) if args.request_ids else None,
        )
    elif args.command == "cleanup-promoted-staging":
        if args.apply:
            require_submit_authority("clean verified promoted export staging")
        result = cleanup_promoted_staging(
            args.campaign, db_path=args.db, config=config, apply=args.apply,
            workers=args.workers,
        )
        if args.summary_only:
            result = {key: value for key, value in result.items() if key != "candidates"}
    else:
        require_submit_authority("run the campaign scheduler cycle")
        result = run_campaign_cycle(
            revalidation_campaign=args.revalidation_campaign,
            backlog_campaign=args.backlog_campaign,
            db_path=args.db, config=config,
            verify_payload_hashes=args.verify_payload_hashes,
            pool=args.pool,
            query_workers=args.query_workers,
            submit_workers=args.submit_workers,
            publish_workers=args.publish_workers,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
