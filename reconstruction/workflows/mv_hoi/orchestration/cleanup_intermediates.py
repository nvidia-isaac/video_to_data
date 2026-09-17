#!/usr/bin/env python3
"""Preserve evidence and delete DB-resolved heavy MV-HOI intermediates.

The command is read-only unless ``--apply`` is supplied.  Exactly one of
``--sequence``, ``--campaign``, or ``--export-run`` is required.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import PurePosixPath
import re
import time
from typing import Iterable

from botocore.exceptions import ClientError

try:
    from . import db
    from .campaign_inventory import _client
    from .config_utils import get_cleanup_settings, load_config
    from .export_commit import (
        SUPPORTED_SCHEMAS,
        CandidateCleanupError,
        cleanup_promoted_export_candidate,
        verify_remote_export_commit,
    )
    from .runtime import MV_HOI_DIR, require_submit_authority
except ImportError:
    import db
    from campaign_inventory import _client
    from config_utils import get_cleanup_settings, load_config
    from export_commit import (
        SUPPORTED_SCHEMAS,
        CandidateCleanupError,
        cleanup_promoted_export_candidate,
        verify_remote_export_commit,
    )
    from runtime import MV_HOI_DIR, require_submit_authority


CLEANUP_SCHEMA = "v2d.mv_hoi.intermediate_cleanup.v1"
METRICS_SCHEMA = "v2d.mv_hoi.retained_metrics.v1"
PENDING_NAME = "intermediate_cleanup.pending.json"
COMPLETE_NAME = "intermediate_cleanup.json"
ACTIVE_REQUEST_STATUSES = frozenset(("RESERVED", "SUBMITTED", "RUNNING", "UNKNOWN"))
TERMINAL_RUN_STATUSES = frozenset(("SUCCEEDED", "FAILED", "CANCELED", "SKIPPED"))

PROTECTED_EXACT = frozenset((
    "mv_preprocess/edex",
    "mv_preprocess/frame_metadata.jsonl",
    "mv_preprocess/hoi_metadata.yaml",
    "mv_preprocess/prompt.txt",
    "mv_preprocess/object_bbox_source.txt",
))
PROTECTED_PREFIXES = (
    "mv_preprocess/object_mesh/",
    "mv_preprocess/videos/",
    "mv_preprocess/labeled_bboxes/",
    "metrics/",
)
METRIC_VISUALIZATION_DIRS = frozenset((
    ("eval_chamfer_object", "chamfer_vis"),
    ("eval_chamfer_human", "chamfer_vis"),
    ("eval_silhouette_mask_object", "silhouette_mask_vis"),
    ("eval_silhouette_mask_human", "silhouette_mask_vis"),
))
CRITICAL_PROTECTED_EXACT = frozenset((
    "mv_preprocess/object_bbox_source.txt",
))
CRITICAL_PROTECTED_PREFIXES = (
    "mv_preprocess/labeled_bboxes/",
)
LEGACY_RECONSTRUCTION_PREFIXES = frozenset((
    "foundation_stereo", "grounding_dino", "sam2_object_masks", "sam2_human_masks",
    "check_object_mask", "foundation_pose", "sam3d_body", "export_soma",
    "estimate_ground_plane", "eval_chamfer_object", "eval_chamfer_human",
    "eval_silhouette_mask_object", "eval_silhouette_mask_human",
    "render_hoi_overlay", "check_accuracy", "upload_hitl",
))
FALLBACK_EVIDENCE = {
    "check_accuracy/check_accuracy.json": "accuracy/check_accuracy.json",
    "eval_chamfer_object/chamfer_metrics.json": "chamfer/object.json",
    "eval_chamfer_human/chamfer_metrics.json": "chamfer/human.json",
    "eval_silhouette_mask_object/silhouette_mask_metrics.json": "silhouette/object.json",
    "eval_silhouette_mask_human/silhouette_mask_metrics.json": "silhouette/human.json",
    "check_object_mask/check_object_mask.json": "object_mask/check_object_mask.json",
    "compare_foundation_pose/foundation_pose_comparison.json": (
        "foundation_pose/foundation_pose_comparison.json"
    ),
    "checkpoint_manifest.json": "checkpoint/manifest.json",
}


class RetryableCleanupBlock(ValueError):
    """A cleanup precondition that may become true without operator repair."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _join(prefix: str, *parts: str) -> str:
    return "/".join((prefix.rstrip("/"), *(part.strip("/") for part in parts)))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _etag(item: dict) -> str:
    return str(item.get("ETag") or item.get("etag") or "").strip('"')


def _list(client, bucket: str, prefix: str) -> list[dict]:
    objects: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        objects.extend({
            "key": item["Key"],
            "size": int(item.get("Size", 0)),
            "etag": _etag(item),
        } for item in page.get("Contents", []))
    return sorted(objects, key=lambda item: item["key"])


def _get(client, bucket: str, key: str) -> tuple[bytes, dict]:
    response = client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read(), response


def _identity(items: Iterable[dict]) -> str:
    records = [{
        "key": item["key"], "size": int(item["size"]), "etag": item.get("etag", ""),
        **({"sha256": item["sha256"]} if item.get("sha256") else {}),
    } for item in sorted(items, key=lambda value: value["key"])]
    return _sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode())


def _is_critical_protected(relative: str) -> bool:
    return (
        relative in CRITICAL_PROTECTED_EXACT
        or relative.startswith(CRITICAL_PROTECTED_PREFIXES)
    )


def _is_metric_visualization(relative: str) -> bool:
    """Return whether a reconstruction MP4 is retained for metric review."""

    parts = PurePosixPath(relative).parts
    if parts and parts[0].startswith("reconstruction_"):
        parts = parts[1:]
    return (
        len(parts) >= 3
        and (parts[0], parts[1]) in METRIC_VISUALIZATION_DIRS
        and parts[-1].lower().endswith(".mp4")
    )


def _protect_objects(
    client, bucket: str, root: str, items: Iterable[dict],
) -> list[dict]:
    """Snapshot protected identities without downloading large retained data."""
    records = []
    for item in sorted(items, key=lambda value: value["key"]):
        record = dict(item)
        if _is_critical_protected(_relative(root, item["key"])):
            payload, _response = _get(client, bucket, item["key"])
            if len(payload) != int(item["size"]):
                raise RuntimeError(f"protected object size changed: {item['key']}")
            record["sha256"] = _sha256(payload)
        records.append(record)
    return records


def _relative(root: str, key: str) -> str:
    token = root.rstrip("/") + "/"
    if not key.startswith(token):
        raise ValueError(f"Object {key!r} is outside {root!r}")
    return key[len(token):]


def _is_protected(relative: str) -> bool:
    return (
        relative in PROTECTED_EXACT
        or relative.startswith(PROTECTED_PREFIXES)
        or _is_metric_visualization(relative)
    )


def categorized_failure_reason(details: str | None) -> str | None:
    value = str(details or "")
    accuracy = re.search(r"accuracy_checks_failed:\s*([^;]+)", value)
    if accuracy:
        return "accuracy_check_failed:" + ",".join(
            sorted(part.strip() for part in accuracy.group(1).split(",") if part.strip())
        )
    task = re.search(r"\btask_failed:\s*([A-Za-z0-9_-]+)", value)
    if task:
        return f"task_failed:{task.group(1)}"
    lowered = value.lower()
    if "upload" in lowered and "quota" in lowered:
        return "infrastructure:upload_quota"
    if "retryable_osmo_infrastructure_failure" in lowered:
        return "infrastructure:osmo"
    return None


def _validate_export_commit(commit: dict, *, sequence: str, export: dict) -> None:
    if commit.get("schema") not in SUPPORTED_SCHEMAS or commit.get("complete") is not True:
        raise ValueError("unsupported or incomplete export commit")
    if commit.get("sequence_name") not in (None, sequence):
        raise ValueError("export commit sequence differs from the database")
    committed_reconstruction = commit.get("reconstruction_run_id")
    if committed_reconstruction is not None:
        if int(committed_reconstruction) != int(export["reconstruction_run_id"]):
            raise ValueError("export commit reconstruction lineage differs from the database")
    elif export.get("authorization_type") == "REVALIDATION":
        if int(commit.get("request_id", -1)) != int(export.get("request_id") or -2):
            raise ValueError("revalidation commit request lineage differs from the database")
    else:
        raise ValueError("export commit lacks reconstruction lineage")
    validation = commit.get("export_validation") or {}
    counts = validation.get("camera_counts") or {}
    if validation.get("schema") != "v2d.mv_hoi.export_validation.v2":
        raise ValueError("export commit lacks supported FFV1 validation evidence")
    expected = {"images": 8, "images_anonymized": 8, "depth": 4}
    if any(int(counts.get(name, -1)) != count for name, count in expected.items()):
        raise ValueError("export commit does not validate all required FFV1 camera pairs")
    committed_paths = {str(item.get("path")) for item in commit.get("files", [])}
    required_suffixes = (
        "/failure_segments/source_failure_segments.json",
        "/failure_segments/export_failure_segments.json",
    )
    if (
        "failure_segments.json" not in committed_paths
        or any(
            not any(path.endswith(suffix) for path in committed_paths)
            for suffix in required_suffixes
        )
    ):
        raise ValueError("export commit lacks required human-QC failure-segment evidence")


def _candidate_export_runs(
    dataset: str,
    *,
    sequence: str | None,
    campaign: str | None,
    export_run: int | None,
    db_path: str,
) -> list[dict]:
    if export_run is not None:
        run = db.get_stage_run(export_run, stage=db.EXPORT_STAGE, db_path=db_path)
        if not run or run["dataset"] != dataset:
            raise ValueError(f"Unknown export run {export_run} for {dataset}")
        return [run]
    snapshot = db.get_campaign_lifecycle_snapshot(
        dataset, campaign=campaign, db_path=db_path,
    )
    selected = []
    for row in snapshot["sequences"]:
        if sequence is not None and row["sequence_name"] != sequence:
            continue
        if row.get("export_run_id") is not None:
            run = db.get_stage_run(
                int(row["export_run_id"]), stage=db.EXPORT_STAGE, db_path=db_path,
            )
            if run:
                selected.append(run)
    if sequence is not None and not selected:
        raise ValueError(f"{sequence}: no active-campaign export lineage")
    return selected


def _lineage(export: dict, *, db_path: str) -> dict:
    if export["run_status"] != "SUCCEEDED":
        raise ValueError("export run is not SUCCEEDED")
    reconstruction_id = export.get("reconstruction_run_id")
    if reconstruction_id is None:
        raise ValueError("export has no reconstruction lineage")
    reconstruction = db.get_stage_run(
        int(reconstruction_id), stage=db.RECONSTRUCTION_STAGE, db_path=db_path,
    )
    if not reconstruction or reconstruction["run_status"] != "SUCCEEDED":
        raise ValueError("export reconstruction lineage is not successful")
    preprocess_id = reconstruction.get("preprocess_run_id")
    if preprocess_id is None:
        raise ValueError("reconstruction has no preprocess lineage")
    preprocess = db.get_stage_run(
        int(preprocess_id), stage=db.PREPROCESS_STAGE, db_path=db_path,
    )
    if not preprocess or preprocess["run_status"] != "SUCCEEDED":
        raise ValueError("export preprocess lineage is not successful")
    request = (
        db.get_stage_request(int(export["request_id"]), db_path=db_path)
        if export.get("request_id") is not None else None
    )
    authorization = export.get("authorization_type")
    if authorization == "QC" and export.get("qc_review_id") is None:
        raise ValueError("QC-authorized export has no QC review")
    if authorization == "REVALIDATION" and (
        request is None or request.get("stage") != "revalidation"
    ):
        raise ValueError("revalidation export is not bound to a revalidation request")
    if authorization not in ("QC", "REVALIDATION", "MANUAL_OVERRIDE"):
        raise ValueError(f"unsupported cleanup authorization: {authorization}")
    return {
        "export": export,
        "request": request,
        "reconstruction": reconstruction,
        "preprocess": preprocess,
    }


def _reject_active_work(lineage: dict, *, db_path: str) -> None:
    export = lineage["export"]
    active = db.list_stage_requests(
        dataset=export["dataset"], status=ACTIVE_REQUEST_STATUSES, db_path=db_path,
    )
    conflicts = [
        item for item in active
        if item["sequence_name"] == export["sequence_name"]
        and item["stage"] in ("preprocess", "reconstruction", "revalidation")
    ]
    if conflicts:
        raise RetryableCleanupBlock(
            "active newer processing request(s): "
            + ", ".join(f"{item['stage']}:{item['id']}" for item in conflicts)
        )


def _canonical_root(lineage: dict) -> tuple[object, str, str]:
    uri = str(lineage["preprocess"].get("output_uri") or "").rstrip("/")
    if not uri:
        raise ValueError("preprocess lineage has no output URI")
    return _client(uri)


def _metric_sources(
    lineage: dict, commit: dict, export_client, export_bucket: str, export_prefix: str,
) -> tuple[str, int, list[dict]]:
    export = lineage["export"]
    authorization = export["authorization_type"]
    stage = "revalidation" if authorization == "REVALIDATION" else "reconstruction"
    request = lineage["request"]
    request_id = int(
        request["id"] if stage == "revalidation" and request
        else lineage["reconstruction"].get("request_id")
        or lineage["reconstruction"]["stage_run_id"]
    )
    root = f"metrics/{stage}/{request_id}/"
    committed = {
        item["path"]: item for item in commit.get("files", [])
        if item["path"].startswith(root)
    }
    if committed:
        sources = [{
            "client": export_client,
            "bucket": export_bucket,
            "key": _join(export_prefix, path),
            "relative": path[len(root):],
            "size": int(item["size"]),
            "sha256": item.get("sha256"),
            "original_key": _join(export_prefix, path),
        } for path, item in sorted(committed.items()) if not path.endswith("/manifest.json")]
        return stage, request_id, sources

    legacy = {
        item["path"]: item for item in commit.get("files", [])
        if item["path"].startswith("revalidation/") and item["path"].endswith(".json")
    }
    if legacy:
        return stage, request_id, [{
            "client": export_client, "bucket": export_bucket,
            "key": _join(export_prefix, path),
            "relative": f"legacy/{PurePosixPath(path).name}",
            "size": int(item["size"]), "sha256": item.get("sha256"),
            "original_key": _join(export_prefix, path),
        } for path, item in sorted(legacy.items())]

    source_uri = str(lineage["reconstruction"].get("output_uri") or "")
    source_client, source_bucket, source_prefix = _client(source_uri)
    objects = _list(source_client, source_bucket, source_prefix)
    by_relative = {_relative(source_prefix, item["key"]): item for item in objects}
    sources = []
    for original, retained in FALLBACK_EVIDENCE.items():
        item = by_relative.get(original)
        if item:
            sources.append({
                "client": source_client, "bucket": source_bucket,
                "key": item["key"], "relative": retained,
                "size": item["size"], "sha256": None,
                "original_key": item["key"],
            })
    if not sources:
        raise ValueError("no export-bundled or historical reconstruction evidence found")
    return stage, request_id, sources


def _copy_metrics(
    lineage: dict, commit: dict, *, apply: bool,
    export_client, export_bucket: str, export_prefix: str,
    canonical_client, canonical_bucket: str, canonical_prefix: str,
) -> dict:
    stage, request_id, sources = _metric_sources(
        lineage, commit, export_client, export_bucket, export_prefix,
    )
    destination_root = _join(canonical_prefix, "metrics", stage, str(request_id))
    records = []
    for source in sources:
        destination = _join(destination_root, source["relative"])
        payload, response = _get(source["client"], source["bucket"], source["key"])
        digest = _sha256(payload)
        if source.get("sha256") and digest != source["sha256"]:
            raise ValueError(f"metric source hash mismatch: {source['key']}")
        if len(payload) != int(source["size"]):
            raise ValueError(f"metric source size mismatch: {source['key']}")
        if apply:
            canonical_client.put_object(Bucket=canonical_bucket, Key=destination, Body=payload)
            copied, copied_response = _get(canonical_client, canonical_bucket, destination)
            if copied != payload:
                raise RuntimeError(f"copied metric differs: {destination}")
            destination_etag = _etag(copied_response)
        else:
            destination_etag = None
        records.append({
            "path": source["relative"], "original_key": source["original_key"],
            "size": len(payload), "source_etag": _etag(response),
            "etag": destination_etag, "sha256": digest,
        })
    export = lineage["export"]
    reconstruction = lineage["reconstruction"]
    preprocess = lineage["preprocess"]
    request = lineage["request"]
    manifest = {
        "schema": METRICS_SCHEMA, "created_at": _now(), "stage": stage,
        "request_id": request_id, "campaign_id": request.get("campaign_id") if request else None,
        "campaign_name": request.get("campaign_name") if request else None,
        "preprocess_run_id": preprocess["stage_run_id"],
        "reconstruction_run_id": reconstruction["stage_run_id"],
        "qc_review_id": export.get("qc_review_id"), "export_run_id": export["stage_run_id"],
        "workflow_versions": {
            "preprocess": preprocess.get("pipeline_version"),
            "reconstruction": reconstruction.get("pipeline_version"),
            "export": export.get("pipeline_version"),
        },
        "workflows": {
            "preprocess": preprocess.get("workflow_name"),
            "reconstruction": reconstruction.get("workflow_name"),
            "export": export.get("workflow_name"),
        },
        "source_uris": {
            "preprocess": preprocess.get("output_uri"),
            "reconstruction": reconstruction.get("output_uri"),
            "export": export.get("output_uri"),
        },
        "configuration_sha256": commit.get("configuration_sha256"),
        "label_sha256": reconstruction.get("labeled_bboxes_sha256"),
        "categorized_failure_reason": categorized_failure_reason(
            reconstruction.get("details")
        ),
        "files": records,
    }
    if apply:
        canonical_client.put_object(
            Bucket=canonical_bucket, Key=_join(destination_root, "manifest.json"),
            Body=(json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            ContentType="application/json",
        )
    return manifest


def _deletion_objects(
    lineage: dict, *, db_path: str, canonical_prefix: str,
    canonical_objects: list[dict] | None = None,
) -> list[dict]:
    export = lineage["export"]
    client, bucket, _ = _canonical_root(lineage)
    selected: dict[str, dict] = {}
    canonical_objects = (
        canonical_objects
        if canonical_objects is not None else _list(client, bucket, canonical_prefix)
    )
    for item in canonical_objects:
        relative = _relative(canonical_prefix, item["key"])
        if relative.startswith(("rosbag_to_edex/", "mv_preprocess/images/", "face_detector/")):
            selected[item["key"]] = item

    histories = db.list_stage_run_history(
        export["dataset"], stage=db.RECONSTRUCTION_STAGE,
        sequence_name=export["sequence_name"], db_path=db_path,
    )
    authoritative_reconstruction_id = int(
        lineage["reconstruction"]["stage_run_id"]
    )
    for run in histories:
        if run["run_status"] not in TERMINAL_RUN_STATUSES or not run.get("output_uri"):
            continue
        run_client, run_bucket, run_prefix = _client(str(run["output_uri"]))
        if run_bucket != bucket:
            if int(run["stage_run_id"]) == authoritative_reconstruction_id:
                raise ValueError(
                    "authoritative reconstruction moved outside canonical CSS bucket"
                )
            continue
        if run_prefix == canonical_prefix:
            for item in canonical_objects:
                relative = _relative(canonical_prefix, item["key"])
                if (
                    relative.split("/", 1)[0] in LEGACY_RECONSTRUCTION_PREFIXES
                    and not _is_metric_visualization(relative)
                ):
                    selected[item["key"]] = item
        elif run_prefix.startswith(canonical_prefix.rstrip("/") + "/reconstruction_"):
            for item in _list(run_client, run_bucket, run_prefix):
                relative = _relative(canonical_prefix, item["key"])
                if not _is_metric_visualization(relative):
                    selected[item["key"]] = item
        else:
            if int(run["stage_run_id"]) == authoritative_reconstruction_id:
                raise ValueError(
                    f"unrecognized authoritative reconstruction output layout: "
                    f"{run['output_uri']}"
                )
            # Historical request-scoped roots (notably retired data_output_2)
            # are immutable lineage but are outside this canonical cleanup.
            continue
    unsafe = [
        item["key"] for item in selected.values()
        if _is_protected(_relative(canonical_prefix, item["key"]))
    ]
    if unsafe:
        raise ValueError(f"cleanup deletion intersects protected paths: {unsafe[:3]}")
    return sorted(selected.values(), key=lambda item: item["key"])


def _protected_nonmetric(root: str, objects: Iterable[dict]) -> list[dict]:
    return [
        item for item in objects
        if _is_protected(_relative(root, item["key"]))
        and not _relative(root, item["key"]).startswith("metrics/")
    ]


def _verify_db_export_manifest(
    lineage: dict, commit: dict, commit_payload: bytes,
) -> str:
    request = lineage["request"]
    export = lineage["export"]
    if request is None:
        raise ValueError("campaign export has no request lineage")
    expected_uri = str(export.get("output_uri") or "").rstrip("/") + "/commit.json"
    if str(request.get("result_manifest_uri") or "").rstrip("/") != expected_uri:
        raise ValueError("request and export disagree on the committed manifest URI")
    digest = _sha256(commit_payload)
    if request.get("result_manifest_sha256") != digest:
        raise ValueError("request and export disagree on the committed manifest SHA-256")
    if json.loads(commit_payload) != commit:
        raise ValueError("verified export commit differs from committed manifest bytes")
    return digest


def _protected_matches(expected: Iterable[dict], current: Iterable[dict]) -> bool:
    by_key = {item["key"]: item for item in expected}
    observed = {item["key"]: item for item in current}
    if set(by_key) != set(observed):
        return False
    return all(
        int(observed[key]["size"]) == int(item["size"])
        and observed[key].get("etag", "") == item.get("etag", "")
        and (
            not observed[key].get("sha256")
            or observed[key].get("sha256") == item.get("sha256")
        )
        for key, item in by_key.items()
    )


def _delete(client, bucket: str, keys: list[str]) -> None:
    for offset in range(0, len(keys), 1000):
        response = client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": key} for key in keys[offset:offset + 1000]], "Quiet": True},
        )
        if response.get("Errors"):
            raise RuntimeError(f"object deletion failed: {response['Errors'][:3]}")


def _join_url(*parts: object) -> str:
    return "/".join(str(part).strip("/") for part in parts if str(part))


def _staging_candidate_uri(lineage: dict) -> str | None:
    request = lineage.get("request")
    if request is None:
        return None
    parameters = json.loads(request.get("parameters_json") or "{}")
    explicit = parameters.get("candidate_uri")
    if explicit:
        return str(explicit).rstrip("/")
    if request.get("stage") != "revalidation":
        return None
    config = load_config(MV_HOI_DIR)
    dataset_cfg = config["datasets"][lineage["export"]["dataset"]]
    pipeline = dataset_cfg["pipelines"]["mv_hoi_revalidation"]
    scoped = (
        [f"request_{request['id']}"]
        if parameters.get("work_output_layout") == "request_scoped_v1" else []
    )
    return _join_url(
        dataset_cfg["swift_base"], pipeline["work_output_path"],
        request["campaign_name"], request["sequence_name"], *scoped,
        "candidate_export",
    )


def cleanup_staging_phase(export: dict, *, db_path: str, apply: bool) -> dict:
    """Verify and remove the exact request-scoped staging prefix."""
    lineage = _lineage(export, db_path=db_path)
    candidate = _staging_candidate_uri(lineage)
    destination = str(export.get("output_uri") or "").rstrip("/")
    if not candidate or candidate == destination:
        return {"status": "NOT_APPLICABLE", "candidate_uri": candidate}
    final_commit = verify_remote_export_commit(destination)
    if not apply:
        client, bucket, prefix = _client(candidate)
        objects = _list(client, bucket, prefix)
        return {
            "status": "WOULD_CLEAN" if objects else "ALREADY_COMPLETE",
            "candidate_uri": candidate,
            "deleted_object_count": len(objects),
            "reclaimed_bytes": sum(int(item["size"]) for item in objects),
        }
    result = cleanup_promoted_export_candidate(
        candidate, destination, expected_commit=final_commit,
    )
    return {"status": "COMPLETE", "candidate_uri": candidate, **result}


def cleanup_export(
    export: dict, *, db_path: str, apply: bool,
    expected_report: dict | None = None,
) -> dict:
    lineage = _lineage(export, db_path=db_path)
    _reject_active_work(lineage, db_path=db_path)
    sequence = export["sequence_name"]
    canonical_client, canonical_bucket, canonical_prefix = _canonical_root(lineage)
    complete_key = _join(canonical_prefix, COMPLETE_NAME)
    try:
        prior_payload, _prior_response = _get(
            canonical_client, canonical_bucket, complete_key,
        )
    except (ClientError, KeyError):
        prior = None
    else:
        prior = json.loads(prior_payload)
    if (
        prior and prior.get("schema") == CLEANUP_SCHEMA
        and prior.get("status") == "COMPLETE"
        and int(prior.get("export_run_id", -1)) == int(export["stage_run_id"])
    ):
        current_objects = _list(
            canonical_client, canonical_bucket, canonical_prefix,
        )
        current = _protect_objects(
            canonical_client, canonical_bucket, canonical_prefix,
            _protected_nonmetric(canonical_prefix, current_objects),
        )
        expected = [
            item for item in prior.get("protected_paths", [])
            if not _relative(canonical_prefix, item["key"]).startswith("metrics/")
        ]
        if not _protected_matches(expected, current):
            raise RuntimeError("completed cleanup no longer reconciles protected inputs")
        return {
            **prior, "status": "ALREADY_COMPLETE", "idempotent_noop": True,
            "manifest_uri": str(lineage["preprocess"]["output_uri"]).rstrip("/")
            + "/" + COMPLETE_NAME,
            "manifest_sha256": _sha256(prior_payload),
        }
    export_client, export_bucket, export_prefix = _client(str(export["output_uri"]))
    commit = verify_remote_export_commit(str(export["output_uri"]))
    _validate_export_commit(commit, sequence=sequence, export=export)
    commit_payload, _commit_response = _get(
        export_client, export_bucket, _join(export_prefix, "commit.json"),
    )
    export_manifest_sha256 = _verify_db_export_manifest(
        lineage, commit, commit_payload,
    )
    canonical_objects = _list(
        canonical_client, canonical_bucket, canonical_prefix,
    )
    protected_before = _protect_objects(
        canonical_client, canonical_bucket, canonical_prefix,
        _protected_nonmetric(canonical_prefix, canonical_objects),
    )
    metrics = _copy_metrics(
        lineage, commit, apply=apply,
        export_client=export_client, export_bucket=export_bucket,
        export_prefix=export_prefix, canonical_client=canonical_client,
        canonical_bucket=canonical_bucket, canonical_prefix=canonical_prefix,
    )
    deletions = _deletion_objects(
        lineage, db_path=db_path, canonical_prefix=canonical_prefix,
        canonical_objects=canonical_objects,
    )
    report = {
        "schema": CLEANUP_SCHEMA, "status": "PENDING" if apply else "DRY_RUN",
        "created_at": _now(), "dataset": export["dataset"], "sequence_name": sequence,
        "export_run_id": export["stage_run_id"],
        "reconstruction_run_id": lineage["reconstruction"]["stage_run_id"],
        "preprocess_run_id": lineage["preprocess"]["stage_run_id"],
        "export_uri": export["output_uri"], "canonical_output_uri": lineage["preprocess"]["output_uri"],
        "export_commit_schema": commit["schema"],
        "export_manifest_sha256": export_manifest_sha256,
        "metrics_manifest": metrics,
        "protected_before_sha256": _identity(protected_before),
        "protected_paths": protected_before,
        "deleted_keys": deletions,
        "deleted_object_count": len(deletions),
        "reclaimed_bytes": sum(item["size"] for item in deletions),
    }
    if expected_report is not None:
        expected_deletions = {
            (item["key"], int(item["size"]), item.get("etag"))
            for item in expected_report.get("deleted_keys", [])
        }
        current_deletions = {
            (item["key"], int(item["size"]), item.get("etag"))
            for item in report["deleted_keys"]
        }
        if (
            int(expected_report.get("export_run_id", -1))
            != int(report["export_run_id"])
            or expected_report.get("export_manifest_sha256")
            != report["export_manifest_sha256"]
            or expected_report.get("protected_before_sha256")
            != report["protected_before_sha256"]
            # An interrupted cleanup may have removed only part of the exact
            # audited deletion set.  Accept that resumable state, but never a
            # new key or a key whose size/ETag changed after the audit.
            or not current_deletions.issubset(expected_deletions)
            or int(report["reclaimed_bytes"])
            != sum(item[1] for item in current_deletions)
        ):
            raise ValueError("cleanup inputs drifted from the immutable backfill audit")
    if not apply:
        return report

    pending_key = _join(canonical_prefix, PENDING_NAME)
    try:
        pending_payload, _pending_response = _get(
            canonical_client, canonical_bucket, pending_key,
        )
    except (ClientError, KeyError):
        pending = None
    else:
        pending = json.loads(pending_payload)
    if pending is not None:
        pending_differs = (
            pending.get("schema") != CLEANUP_SCHEMA
            or pending.get("status") != "PENDING"
            or int(pending.get("export_run_id", -1)) != int(export["stage_run_id"])
            or pending.get("export_manifest_sha256") != export_manifest_sha256
        )
        if pending_differs:
            if expected_report is None:
                raise RuntimeError("cleanup pending marker belongs to different lineage")
            # A current immutable backfill audit may safely supersede a stale
            # marker written by the pre-queue implementation. The report was
            # already compared against current storage immediately above.
            pending_payload = (
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            ).encode()
            canonical_client.put_object(
                Bucket=canonical_bucket, Key=pending_key, Body=pending_payload,
                ContentType="application/json",
            )
        else:
            if not _protected_matches(
                pending.get("protected_paths", []), protected_before,
            ):
                raise RuntimeError("protected canonical inputs changed since cleanup began")
            report = pending
            deletions = list(report.get("deleted_keys", []))
    else:
        pending_payload = (
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        ).encode()
        canonical_client.put_object(
            Bucket=canonical_bucket, Key=pending_key, Body=pending_payload,
            ContentType="application/json",
        )
    # Recheck the export and protected identity after writing metrics/pending.
    current_commit = verify_remote_export_commit(str(export["output_uri"]))
    _validate_export_commit(current_commit, sequence=sequence, export=export)
    current_commit_payload, _response = _get(
        export_client, export_bucket, _join(export_prefix, "commit.json"),
    )
    if _verify_db_export_manifest(
        lineage, current_commit, current_commit_payload,
    ) != export_manifest_sha256:
        raise RuntimeError("committed export changed during cleanup")
    current_objects = _list(
        canonical_client, canonical_bucket, canonical_prefix,
    )
    protected_now = _protect_objects(
        canonical_client, canonical_bucket, canonical_prefix,
        _protected_nonmetric(canonical_prefix, current_objects),
    )
    protected_original = list(report["protected_paths"])
    if not _protected_matches(protected_original, protected_now):
        raise RuntimeError("protected canonical inputs changed during cleanup")
    current_by_key = {item["key"]: item for item in current_objects}
    changed_deletions = [
        item["key"] for item in deletions
        if item["key"] in current_by_key and (
            int(current_by_key[item["key"]]["size"]) != int(item["size"])
            or current_by_key[item["key"]].get("etag") != item.get("etag")
        )
    ]
    if changed_deletions:
        raise RuntimeError(
            f"cleanup deletion inputs changed: {changed_deletions[:3]}"
        )
    remaining_deletions = [
        item["key"] for item in deletions if item["key"] in current_by_key
    ]
    _delete(canonical_client, canonical_bucket, remaining_deletions)
    after_objects = []
    undeleted = []
    for verification_attempt in range(4):
        after_objects = _list(canonical_client, canonical_bucket, canonical_prefix)
        remaining = {item["key"] for item in after_objects}
        undeleted = [item["key"] for item in deletions if item["key"] in remaining]
        if not undeleted:
            break
        if verification_attempt == 3:
            break
        _delete(canonical_client, canonical_bucket, undeleted)
        time.sleep(0.5 * (verification_attempt + 1))
    if undeleted:
        raise RuntimeError(f"cleanup left {len(undeleted)} enumerated object(s)")
    protected_after = _protect_objects(
        canonical_client, canonical_bucket, canonical_prefix,
        _protected_nonmetric(canonical_prefix, after_objects),
    )
    if not _protected_matches(protected_original, protected_after):
        raise RuntimeError("protected canonical inputs changed during deletion")
    report["status"] = "COMPLETE"
    report["completed_at"] = _now()
    report["protected_after_sha256"] = _identity(protected_after)
    complete_payload = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode()
    canonical_client.put_object(
        Bucket=canonical_bucket, Key=complete_key, Body=complete_payload,
        ContentType="application/json",
    )
    canonical_client.delete_object(Bucket=canonical_bucket, Key=pending_key)
    return {
        **report,
        "manifest_uri": str(lineage["preprocess"]["output_uri"]).rstrip("/")
        + "/" + COMPLETE_NAME,
        "manifest_sha256": _sha256(complete_payload),
    }


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def run_cleanup(
    dataset: str, *, sequence: str | None = None, campaign: str | None = None,
    export_run: int | None = None, db_path: str = db.DB_PATH, apply: bool = False,
    limit: int | None = None, workers: int = 1,
    completed_after: str | None = None,
) -> list[dict]:
    if limit is not None and limit < 0:
        raise ValueError("cleanup limit cannot be negative")
    if workers < 1:
        raise ValueError("cleanup workers must be positive")
    if campaign is not None:
        if completed_after and apply:
            db.enqueue_missing_intermediate_cleanups(
                campaign, completed_after=completed_after,
                source="AUTOMATIC", db_path=db_path,
            )
        return run_cleanup_jobs(
            dataset, campaign=campaign, db_path=db_path, apply=apply,
            limit=limit, workers=workers,
        )

    candidates = _candidate_export_runs(
        dataset, sequence=sequence, campaign=None,
        export_run=export_run, db_path=db_path,
    )
    if completed_after:
        cutoff = _parse_timestamp(completed_after)
        candidates = [
            export for export in candidates
            if export.get("completed_at")
            and _parse_timestamp(str(export["completed_at"])) >= cutoff
        ]
    if limit is not None:
        candidates = candidates[:limit]
    if not apply:
        return [_execute_cleanup(export, dataset=dataset, db_path=db_path, apply=False)
                for export in candidates]
    jobs = []
    for export in candidates:
        jobs.append(db.enqueue_intermediate_cleanup(
            int(export["stage_run_id"]), source="MANUAL", db_path=db_path,
        ))
    if jobs and all(job["status"] == "SUCCEEDED" for job in jobs):
        return [
            _execute_cleanup(export, dataset=dataset, db_path=db_path, apply=True)
            for export in candidates
        ]
    return run_cleanup_jobs(
        dataset, export_run_ids=[int(item["stage_run_id"]) for item in candidates],
        db_path=db_path, apply=True, limit=limit, workers=workers,
    )


def _execute_cleanup(
    export: dict, *, dataset: str, db_path: str, apply: bool,
    expected_report: dict | None = None,
) -> dict:
    try:
        return cleanup_export(
            export, db_path=db_path, apply=apply,
            expected_report=expected_report,
        )
    except Exception as exc:
        return {
            "dataset": dataset, "sequence_name": export.get("sequence_name"),
            "export_run_id": export.get("stage_run_id"), "status": "REFUSED",
            "reason": str(exc), "retryable": _is_retryable_cleanup_failure(exc),
        }


def _is_retryable_cleanup_failure(exc: BaseException) -> bool:
    """Recognize transient cleanup transport and database failures."""
    pending: list[BaseException] = [exc]
    observed: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in observed:
            continue
        observed.add(id(current))
        if isinstance(current, (RetryableCleanupBlock, CandidateCleanupError)):
            return True
        if isinstance(current, IncompleteRead):
            return True
        error_type = type(current)
        module = error_type.__module__.lower()
        name = error_type.__name__.lower()
        message = str(current).lower()
        if isinstance(current, ClientError):
            response = current.response or {}
            error = response.get("Error") or {}
            metadata = response.get("ResponseMetadata") or {}
            code = str(error.get("Code") or "").lower()
            status = metadata.get("HTTPStatusCode")
            if status == 429 or code in {
                "429", "toomanyrequests", "slowdown", "throttling",
                "throttlingexception", "requestlimitexceeded",
            }:
                return True
        if name in {"responsestreamingerror", "incompleteread"}:
            return True
        if any(pattern in message for pattern in (
            "incompleteread",
            "response stream",
            "response ended prematurely",
            "connection broken",
            "too many requests",
            "request limit exceeded",
            "requestlimitexceeded",
            "slowdown",
            "http 429",
            "status code: 429",
        )):
            return True
        if module.startswith("psycopg") and (
            "timeout" in name or "connection timeout expired" in message
        ):
            return True
        if module.startswith("sqlalchemy") and (
            "psycopg.errors.connectiontimeout" in message
            or "connection timeout expired" in message
        ):
            return True
        for nested in (
            getattr(current, "orig", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(nested, BaseException):
                pending.append(nested)
    return False


def run_cleanup_jobs(
    dataset: str,
    *,
    campaign: int | str | None = None,
    export_run_ids: Iterable[int] | None = None,
    db_path: str = db.DB_PATH,
    apply: bool = False,
    limit: int | None = None,
    workers: int = 1,
    reserved_by: str | None = None,
    expected_reports: dict[int, dict] | None = None,
) -> list[dict]:
    """Process bounded DB-selected cleanup jobs without campaign-wide CSS scans."""
    if limit is not None and limit < 0:
        raise ValueError("cleanup limit cannot be negative")
    if workers < 1:
        raise ValueError("cleanup workers must be positive")
    requested_ids = list(dict.fromkeys(int(value) for value in (export_run_ids or ())))
    effective_limit = limit if limit is not None else max(1, len(requested_ids) or 1000)
    if apply:
        jobs = db.reserve_intermediate_cleanup_jobs(
            reserved_by=reserved_by or f"cleanup:{os.getpid()}",
            limit=effective_limit, campaign=campaign,
            export_run_ids=requested_ids or None, db_path=db_path,
        )
    else:
        jobs = db.list_intermediate_cleanup_jobs(
            campaign=campaign, status=("PENDING", "RUNNING", "BLOCKED"),
            db_path=db_path,
        )
        if requested_ids:
            selected = set(requested_ids)
            jobs = [item for item in jobs if int(item["export_run_id"]) in selected]
        jobs = jobs[:effective_limit]
    jobs = [item for item in jobs if item["dataset"] == dataset]

    def execute(job: dict) -> dict:
        progress: dict = {}
        try:
            export = db.get_stage_run(
                int(job["export_run_id"]), stage=db.EXPORT_STAGE, db_path=db_path,
            )
            if export is None:
                result = {
                    "dataset": dataset, "sequence_name": job.get("sequence_name"),
                    "export_run_id": job["export_run_id"], "status": "REFUSED",
                    "reason": "cleanup export run no longer exists", "retryable": False,
                }
            else:
                try:
                    progress = json.loads(job.get("details") or "{}")
                except (TypeError, ValueError):
                    progress = {}
                staging = progress.get("staging") or {}
                if apply and staging.get("status") not in (
                    "COMPLETE", "NOT_APPLICABLE",
                ):
                    staging = cleanup_staging_phase(
                        export, db_path=db_path, apply=apply,
                    )
                    progress["staging"] = staging
                    db.update_intermediate_cleanup_job_progress(
                        int(job["id"]), details=progress, db_path=db_path,
                    )
                result = _execute_cleanup(
                    export, dataset=dataset, db_path=db_path, apply=apply,
                    expected_report=(expected_reports or {}).get(
                        int(job["export_run_id"])
                    ),
                )
                result["staging_cleanup"] = staging
        except Exception as exc:
            result = {
                "dataset": dataset, "sequence_name": job.get("sequence_name"),
                "export_run_id": job["export_run_id"], "status": "REFUSED",
                "reason": str(exc),
                "retryable": _is_retryable_cleanup_failure(exc),
            }
        result["cleanup_job_id"] = job["id"]
        if not apply:
            return result
        if result["status"] in ("COMPLETE", "ALREADY_COMPLETE"):
            db.finish_intermediate_cleanup_job(
                int(job["id"]), status="SUCCEEDED",
                details=("idempotent_noop" if result["status"] == "ALREADY_COMPLETE"
                         else "cleanup_complete"),
                manifest_uri=result.get("manifest_uri"),
                manifest_sha256=result.get("manifest_sha256"),
                protected_manifest_sha256=result.get("protected_before_sha256"),
                deleted_object_count=result.get("deleted_object_count"),
                reclaimed_bytes=result.get("reclaimed_bytes"), db_path=db_path,
            )
        elif result.get("retryable"):
            progress["data_output_error"] = result.get("reason")
            db.finish_intermediate_cleanup_job(
                int(job["id"]), status="BLOCKED",
                details=json.dumps(progress, sort_keys=True),
                retry_after_seconds=3600, db_path=db_path,
            )
        else:
            progress["data_output_error"] = result.get("reason")
            db.finish_intermediate_cleanup_job(
                int(job["id"]), status="FAILED",
                details=json.dumps(progress, sort_keys=True),
                db_path=db_path,
            )
        return result

    if workers == 1 or len(jobs) < 2:
        return [execute(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        return list(executor.map(execute, jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--dataset")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--sequence")
    scope.add_argument("--campaign")
    scope.add_argument("--export-run", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--configured-asynchronous",
        action="store_true",
        help="Use the campaign's allowlisted asynchronous cleanup configuration",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print bounded per-job results instead of full deletion manifests",
    )
    parser.add_argument(
        "--completed-after",
        help="Only select exports completed at or after this ISO-8601 timestamp",
    )
    args = parser.parse_args()
    if args.configured_asynchronous:
        if not args.campaign:
            parser.error("--configured-asynchronous requires --campaign")
        campaign = db.get_campaign(args.campaign, db_path=args.db)
        if campaign is None:
            parser.error(f"unknown campaign: {args.campaign}")
        args.dataset = campaign["dataset"]
        settings = get_cleanup_settings(
            load_config(MV_HOI_DIR), args.dataset, args.campaign,
        )
        if not settings:
            parser.error("campaign is not allowlisted for automatic cleanup")
        if settings["mode"] != "asynchronous":
            parser.error("campaign cleanup mode is not asynchronous")
        args.limit = settings["limit"]
        args.workers = settings["workers"]
        args.completed_after = settings["completed_after"]
    elif not args.dataset:
        parser.error("--dataset is required unless --configured-asynchronous is used")
    if args.apply:
        require_submit_authority("delete committed-export intermediates")
    results = run_cleanup(
        args.dataset, sequence=args.sequence, campaign=args.campaign,
        export_run=args.export_run, db_path=args.db, apply=args.apply,
        limit=args.limit, workers=args.workers,
        completed_after=args.completed_after,
    )
    output = results
    if args.summary:
        output = [
            {
                key: item.get(key)
                for key in (
                    "cleanup_job_id", "sequence_name", "export_run_id", "status",
                    "deleted_object_count", "reclaimed_bytes", "reason",
                )
                if item.get(key) is not None
            }
            for item in results
        ]
    print(json.dumps(output, indent=2, sort_keys=True, default=str))
    if any(item["status"] == "REFUSED" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
