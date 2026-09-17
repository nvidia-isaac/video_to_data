#!/usr/bin/env python3
"""Delete DB-provenanced legacy flat reconstruction outputs for a backlog.

Historical reconstruction workflows wrote task outputs directly below the
canonical sequence root.  This migration is deliberately separate from the
request-scoped pending-intermediate cleanup and never changes database state.
Audit shards contain the exact keys, sizes, and ETags authorized for deletion.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from botocore.exceptions import ClientError


SCRIPT_DIR = Path(__file__).resolve().parent
MV_HOI_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MV_HOI_DIR))

from migration import cleanup_backlog_pending_intermediates as pending  # noqa: E402
from orchestration import cleanup_intermediates as cleanup, db  # noqa: E402
from orchestration.config_utils import load_config  # noqa: E402
from orchestration.runtime import require_submit_authority, state_path  # noqa: E402


SCHEMA = "v2d.mv_hoi.backlog_legacy_flat_cleanup_audit.v1"
SHARD_SCHEMA = "v2d.mv_hoi.backlog_legacy_flat_cleanup_shard.v1"
APPLY_SCHEMA = "v2d.mv_hoi.backlog_legacy_flat_cleanup_apply.v1"
PENDING_SCHEMA = "v2d.mv_hoi.backlog_legacy_flat_cleanup_pending.v1"
COMPLETE_SCHEMA = "v2d.mv_hoi.backlog_legacy_flat_cleanup.v1"
PENDING_NAME = "backlog_legacy_flat_cleanup.pending.json"
COMPLETE_NAME = "backlog_legacy_flat_cleanup.json"
TERMINAL = frozenset(("SUCCEEDED", "FAILED", "CANCELED", "SKIPPED"))
LEGACY_PREFIXES = tuple(sorted(cleanup.LEGACY_RECONSTRUCTION_PREFIXES))
EVIDENCE_PREFIXES = frozenset((
    "check_accuracy", "check_object_mask", "estimate_ground_plane",
    "eval_chamfer_object", "eval_chamfer_human",
    "eval_silhouette_mask_object", "eval_silhouette_mask_human",
))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _load_source_audit(path: Path, campaign_name: str, campaign_id: int) -> dict:
    report = json.loads(path.read_text())
    pending._validate_audit(report)
    if (
        report.get("campaign_name") != campaign_name
        or int(report.get("campaign_id", -1)) != int(campaign_id)
    ):
        raise ValueError("source pending-cleanup audit campaign identity differs")
    return report


def _legacy_snapshot(dataset: str, *, db_path: str) -> list[dict]:
    conn = db.get_connection(db_path)
    try:
        return [dict(row) for row in conn.execute(
            """SELECT s.sequence_name, rr.id AS reconstruction_run_id,
                      rr.status AS run_status, rr.pipeline_version,
                      rr.legacy_source_table, rr.legacy_source_id,
                      rr.request_id, rr.created_at, we.workflow_name
               FROM reconstruction_runs rr
               JOIN sequences s ON s.id=rr.sequence_id
               LEFT JOIN workflow_executions we ON we.id=rr.workflow_execution_id
               WHERE s.dataset=? AND rr.status IN ('SUCCEEDED','FAILED','CANCELED','SKIPPED')
                 AND rr.legacy_source_table IS NOT NULL
               ORDER BY rr.created_at, rr.id""",
            (dataset,),
        )]
    finally:
        conn.close()


def _current_candidate_identity(
    campaign: dict, quota_path: Path, *, db_path: str,
) -> dict[str, tuple[int, str, str]]:
    quota, _quota_sha = pending._load_quota_plan(quota_path, campaign)
    selected = pending._select_candidates(
        campaign, quota, pending._database_snapshot(campaign, db_path=db_path),
    )
    return {
        row["sequence"]: (
            int(row["request_id"]), row["request_status"], row["candidate_reason"],
        )
        for row in selected
    }


def _source_candidate_identity(source: dict) -> dict[str, tuple[int, str, str]]:
    return {
        row["sequence"]: (
            int(row["request_id"]), row["request_status"], row["candidate_reason"],
        )
        for row in source["records"]
    }


def _legacy_by_sequence(source: dict, rows: list[dict]) -> dict[str, list[dict]]:
    members = {row["sequence"] for row in source["records"]}
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["sequence_name"] in members and row["run_status"] in TERMINAL:
            result[row["sequence_name"]].append(row)
    return dict(result)


def _output_root(campaign: dict, config: dict):
    dataset_cfg = config["datasets"][campaign["dataset"]]
    pipeline = dataset_cfg["pipelines"][db.PREPROCESS_STAGE]
    uri = cleanup._join(
        dataset_cfg["swift_base"],
        pipeline.get("campaign_output_path", pipeline["output_path"]),
    )
    client, bucket, prefix = cleanup._client(uri)
    return client, bucket, prefix, uri


def _list_legacy(client, bucket: str, sequence_root: str) -> list[dict]:
    objects = {}
    for name in LEGACY_PREFIXES:
        for item in pending._list_with_backoff(
            client, bucket, cleanup._join(sequence_root, name),
        ):
            objects[item["key"]] = item
    return sorted(objects.values(), key=lambda item: item["key"])


def _list_protected(client, bucket: str, sequence_root: str) -> list[dict]:
    return [
        item for item in pending._list_with_backoff(
            client, bucket, cleanup._join(sequence_root, "mv_preprocess"),
        )
        if cleanup._is_protected(cleanup._relative(sequence_root, item["key"]))
    ]


def _head_optional(client, bucket: str, key: str) -> dict | None:
    for attempt in range(8):
        try:
            response = client.head_object(Bucket=bucket, Key=key)
            last_modified = response.get("LastModified")
            return {
                "key": key,
                "size": int(response.get("ContentLength", 0)),
                "etag": cleanup._etag(response),
                "last_modified": (
                    last_modified.isoformat()
                    if hasattr(last_modified, "isoformat") else last_modified
                ),
            }
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                return None
            if code not in ("429", "SlowDown", "Throttling") and status not in (429, 503):
                raise
            if attempt == 7:
                raise
            time.sleep(min(20.0, 2.0 ** attempt) + random.random())
    raise AssertionError("unreachable")


def _existing_after_head(client, bucket: str, objects: list[dict]) -> list[dict]:
    """Return LIST entries that an authoritative HEAD still observes."""
    if not objects:
        return []
    existing = []
    with ThreadPoolExecutor(max_workers=min(16, len(objects))) as executor:
        futures = {
            executor.submit(_head_optional, client, bucket, item["key"]): item
            for item in objects
        }
        for future in as_completed(futures):
            if future.result() is not None:
                existing.append(futures[future])
    return sorted(existing, key=lambda item: item["key"])


def _is_evidence(relative: str) -> bool:
    if relative in cleanup.FALLBACK_EVIDENCE:
        return True
    path = Path(relative)
    top = relative.split("/", 1)[0]
    return (
        (top in EVIDENCE_PREFIXES and path.suffix.lower() == ".json")
        or ("manifest" in path.name.lower() and path.suffix.lower() == ".json")
    )


def _shard_path(manifest_dir: Path, sequence: str) -> Path:
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    return manifest_dir / digest[:2] / f"{sequence}.json.gz"


def _write_shard(path: Path, payload: dict) -> str:
    raw = _canonical(payload)
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as stream:
        stream.write(raw)
    temporary.replace(path)
    return digest


def _read_shard(path: Path, expected_sha256: str) -> dict:
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"cleanup shard SHA-256 differs: {path}")
    payload = json.loads(raw)
    if payload.get("schema") != SHARD_SCHEMA:
        raise ValueError(f"unsupported cleanup shard: {path}")
    return payload


def _inspect(
    sequence: str, runs: list[dict], *, client, bucket: str, output_root: str,
    manifest_dir: Path,
) -> dict:
    sequence_root = cleanup._join(output_root, sequence)
    objects = _list_legacy(client, bucket, sequence_root)
    protected = _list_protected(client, bucket, sequence_root)
    by_prefix: dict[str, dict[str, int]] = defaultdict(
        lambda: {"object_count": 0, "bytes": 0},
    )
    for item in objects:
        prefix = cleanup._relative(sequence_root, item["key"]).split("/", 1)[0]
        by_prefix[prefix]["object_count"] += 1
        by_prefix[prefix]["bytes"] += int(item["size"])
    shard = {
        "schema": SHARD_SCHEMA,
        "sequence": sequence,
        "sequence_root": sequence_root,
        "legacy_runs": runs,
        "objects": objects,
        "protected_objects": protected,
        "protected_identity_sha256": cleanup._identity(protected),
        "evidence_keys": [
            item["key"] for item in objects
            if _is_evidence(cleanup._relative(sequence_root, item["key"]))
        ],
    }
    path = _shard_path(manifest_dir, sequence)
    shard_sha = _write_shard(path, shard)
    return {
        "sequence": sequence,
        "legacy_run_count": len(runs),
        "legacy_run_ids": [int(run["reconstruction_run_id"]) for run in runs],
        "latest_legacy_run_id": int(runs[-1]["reconstruction_run_id"]),
        "object_count": len(objects),
        "logical_bytes": sum(int(item["size"]) for item in objects),
        "evidence_object_count": len(shard["evidence_keys"]),
        "by_prefix": dict(sorted(by_prefix.items())),
        "shard": str(path.resolve()),
        "shard_sha256": shard_sha,
    }


def audit(
    campaign_name: str, *, expected_campaign_id: int, quota_retry_plan: Path,
    source_audit_path: Path, report_path: Path, manifest_dir: Path,
    db_path: str, config: dict, max_workers: int,
) -> dict:
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or int(campaign["id"]) != int(expected_campaign_id):
        raise ValueError("campaign name and --expected-campaign-id do not match")
    source = _load_source_audit(source_audit_path, campaign_name, expected_campaign_id)
    if _current_candidate_identity(campaign, quota_retry_plan, db_path=db_path) != (
        _source_candidate_identity(source)
    ):
        raise RuntimeError("campaign candidates changed after the source cleanup audit")
    lineage = _legacy_by_sequence(
        source, _legacy_snapshot(campaign["dataset"], db_path=db_path),
    )
    client, bucket, output_root, output_uri = _output_root(campaign, config)
    records = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(
                _inspect, sequence, runs, client=client, bucket=bucket,
                output_root=output_root, manifest_dir=manifest_dir,
            ): sequence
            for sequence, runs in sorted(lineage.items())
        }
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"Audited {index}/{len(futures)} legacy sequence(s)", flush=True)
    records.sort(key=lambda item: item["sequence"])
    prefix_summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"object_count": 0, "bytes": 0},
    )
    for record in records:
        for name, values in record["by_prefix"].items():
            prefix_summary[name]["object_count"] += values["object_count"]
            prefix_summary[name]["bytes"] += values["bytes"]
    reclaimed = sum(item["logical_bytes"] for item in records)
    report = {
        "schema": SCHEMA,
        "generated_at": _now(),
        "campaign_id": campaign["id"],
        "campaign_name": campaign["name"],
        "campaign_status": campaign["status"],
        "dataset": campaign["dataset"],
        "output_root_uri": output_uri,
        "source_audit": str(source_audit_path.resolve()),
        "source_audit_sha256": source["audit_sha256"],
        "quota_retry_plan": str(quota_retry_plan.resolve()),
        "quota_retry_plan_sha256": pending._file_sha256(quota_retry_plan),
        "manifest_dir": str(manifest_dir.resolve()),
        "source_candidate_count": len(source["records"]),
        "legacy_lineage_sequence_count": len(records),
        "sequences_with_objects": sum(item["object_count"] > 0 for item in records),
        "object_count": sum(item["object_count"] for item in records),
        "logical_bytes": reclaimed,
        "decimal": pending._human_bytes(reclaimed, binary=False),
        "binary": pending._human_bytes(reclaimed, binary=True),
        "by_prefix": dict(sorted(prefix_summary.items())),
        "per_sequence_byte_percentiles": pending._percentiles([
            item["logical_bytes"] for item in records
        ]),
        "records": records,
        "storage_accounting_note": (
            "Object sizes are exact logical bytes; CSS quota accounting may lag."
        ),
    }
    report["audit_sha256"] = _sha256(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _validate_report(report: dict) -> None:
    if report.get("schema") != SCHEMA:
        raise ValueError("unsupported legacy flat cleanup audit")
    claimed = report.get("audit_sha256")
    unsigned = {key: value for key, value in report.items() if key != "audit_sha256"}
    if claimed != _sha256(unsigned):
        raise ValueError("legacy flat cleanup audit SHA-256 differs")


def _verify_protected(current: list[dict], shard: dict) -> None:
    if cleanup._identity(current) != shard["protected_identity_sha256"]:
        raise RuntimeError(f"{shard['sequence']}: protected storage changed")


def _copy_evidence(
    report: dict, shard: dict, *, client, bucket: str, allow_existing: bool = False,
) -> dict:
    if not shard["evidence_keys"]:
        return {"files": [], "manifest": None}
    by_key = {item["key"]: item for item in shard["objects"]}
    latest = shard["legacy_runs"][-1]
    destination_root = cleanup._join(
        shard["sequence_root"], "metrics", "reconstruction",
        str(latest["reconstruction_run_id"]), "legacy_flat",
    )
    files = []
    for key in shard["evidence_keys"]:
        expected = by_key[key]
        relative = cleanup._relative(shard["sequence_root"], key)
        destination = cleanup._join(destination_root, relative)
        source_etag = expected["etag"]
        try:
            payload, response = pending._get_with_backoff(client, bucket, key)
        except (KeyError, ClientError) as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if not allow_existing or (
                not isinstance(exc, KeyError)
                and code not in ("404", "NoSuchKey", "NotFound")
            ):
                raise
            copied, copied_response = pending._get_with_backoff(
                client, bucket, destination,
            )
            if (
                len(copied) != int(expected["size"])
                or cleanup._etag(copied_response) != expected["etag"]
            ):
                raise RuntimeError(
                    f"{shard['sequence']}: retained evidence differs from audit"
                )
            payload = copied
        else:
            if (
                len(payload) != int(expected["size"])
                or cleanup._etag(response) != expected["etag"]
            ):
                raise RuntimeError(f"{shard['sequence']}: evidence changed after audit")
            client.put_object(Bucket=bucket, Key=destination, Body=payload)
            copied, copied_response = pending._get_with_backoff(
                client, bucket, destination,
            )
        if copied != payload:
            raise RuntimeError(f"copied legacy evidence differs: {destination}")
        files.append({
            "relative_path": relative,
            "original_key": key,
            "destination_key": destination,
            "size": len(payload),
            "source_etag": source_etag,
            "etag": cleanup._etag(copied_response),
            "sha256": cleanup._sha256(payload),
        })
    manifest = {
        "schema": cleanup.METRICS_SCHEMA,
        "created_at": _now(),
        "stage": "reconstruction",
        "request_id": latest.get("request_id"),
        "campaign_id": report["campaign_id"],
        "campaign_name": report["campaign_name"],
        "reconstruction_run_id": latest["reconstruction_run_id"],
        "workflow": latest.get("workflow_name"),
        "pipeline_version": latest.get("pipeline_version"),
        "source_uri": cleanup._join(report["output_root_uri"], shard["sequence"]),
        "legacy_runs": shard["legacy_runs"],
        "files": files,
    }
    manifest_key = cleanup._join(destination_root, "manifest.json")
    pending._put_json(client, bucket, manifest_key, manifest)
    return {"files": files, "manifest": manifest_key}


def _delete_and_verify(
    client, bucket: str, shard: dict, remaining: list[dict],
) -> list[dict]:
    audited = {item["key"] for item in shard["objects"]}
    keys = [item["key"] for item in remaining]
    empty_observations = 0
    for attempt in range(8):
        if keys:
            pending._delete_with_backoff(client, bucket, keys)
        current = _list_legacy(client, bucket, shard["sequence_root"])
        unexpected = sorted(item["key"] for item in current if item["key"] not in audited)
        if unexpected:
            raise RuntimeError(f"{shard['sequence']}: new legacy objects appeared")
        keys = [item["key"] for item in current]
        if not keys:
            empty_observations += 1
            if empty_observations >= 2:
                return current
        else:
            empty_observations = 0
        if attempt < 7:
            time.sleep(min(20.0, 2.0 ** attempt) + random.random())
    raise RuntimeError(f"{shard['sequence']}: legacy cleanup left {len(keys)} objects")


def _complete_result(complete: dict, record: dict) -> dict:
    return {
        **complete,
        "status": "ALREADY_COMPLETE",
        "idempotent_noop": True,
        "actual_deleted_object_count_total": record["object_count"],
        "actual_reclaimed_logical_bytes_total": complete.get(
            "actual_reclaimed_logical_bytes_total", record["logical_bytes"],
        ),
        "actual_deleted_object_count_this_run": 0,
        "actual_reclaimed_logical_bytes_this_run": 0,
    }


def _apply_record(report: dict, record: dict, *, client, bucket: str) -> dict:
    shard = _read_shard(Path(record["shard"]), record["shard_sha256"])
    if shard["sequence"] != record["sequence"]:
        raise ValueError("cleanup shard sequence differs from audit index")
    pending_key = cleanup._join(shard["sequence_root"], PENDING_NAME)
    complete_key = cleanup._join(shard["sequence_root"], COMPLETE_NAME)
    complete = pending._optional_json(client, bucket, complete_key)
    current = _list_legacy(client, bucket, shard["sequence_root"])
    protected = _list_protected(client, bucket, shard["sequence_root"])
    if (
        complete and complete.get("schema") == COMPLETE_SCHEMA
        and complete.get("audit_sha256") == report["audit_sha256"]
        and complete.get("status") == "COMPLETE"
    ):
        stale_listing_phantoms = 0
        if current:
            # A completed deletion was already verified by two empty LIST
            # observations. If the gateway later resurfaces stale entries,
            # distinguish that from a real rewrite with authoritative HEADs.
            existing = _existing_after_head(client, bucket, current)
            if existing:
                raise RuntimeError(f"{record['sequence']}: legacy objects reappeared")
            stale_listing_phantoms = len(current)
        _verify_protected(protected, shard)
        pending._remove_pending_marker(client, bucket, pending_key)
        result = _complete_result(complete, record)
        if stale_listing_phantoms:
            result["post_complete_stale_list_phantom_count"] = stale_listing_phantoms
        return result

    in_progress = pending._optional_json(client, bucket, pending_key)
    valid_pending = bool(
        in_progress and in_progress.get("audit_sha256") == report["audit_sha256"]
    )
    current_by_key = {item["key"]: item for item in current}
    audited_by_key = {item["key"]: item for item in shard["objects"]}
    # A CSS listing can briefly surface an object version which disappears on
    # the next consistent read.  Never delete an unaudited key: require it to
    # persist across repeated listings before treating it as genuine drift.
    unexpected = sorted(set(current_by_key) - set(audited_by_key))
    for attempt in range(4):
        if not unexpected:
            break
        time.sleep(min(4.0, 2.0 ** attempt) + random.random())
        current_by_key = {
            item["key"]: item
            for item in _list_legacy(client, bucket, shard["sequence_root"])
        }
        unexpected = sorted(set(current_by_key) - set(audited_by_key))
    if unexpected:
        raise RuntimeError(
            f"{record['sequence']}: {len(unexpected)} new legacy object(s) appeared: "
            + ", ".join(unexpected[:3])
        )
    # A fresh apply must prove that every audited key still exists.  CSS list
    # responses can transiently omit an object, so confirm omissions with HEAD.
    # During a valid resume, missing keys are expected: a previous worker may
    # already have deleted them after writing the pending marker.
    if not valid_pending:
        for key in sorted(set(audited_by_key) - set(current_by_key)):
            headed = _head_optional(client, bucket, key)
            if headed is not None:
                current_by_key[key] = headed
    identity_reconciliations = []
    for key, expected in audited_by_key.items():
        actual = current_by_key.get(key)
        if actual is None:
            if valid_pending:
                continue
            # LIST may also return a phantom entry that HEAD says is already
            # absent. Confirm twice, retain the discrepancy in the completion
            # record, and exclude its listed bytes from actual reclamation.
            headed = _head_optional(client, bucket, key)
            time.sleep(0.1)
            confirmed = _head_optional(client, bucket, key)
            if headed is None and confirmed is None:
                identity_reconciliations.append({
                    "key": key,
                    "listed_size": expected["size"],
                    "listed_etag": expected.get("etag"),
                    "head_size": 0,
                    "head_etag": None,
                    "last_modified": None,
                    "reason": "stale_list_phantom_confirmed_missing_by_head",
                })
                continue
            raise RuntimeError(f"{record['sequence']}: audited legacy key is missing")
        if actual["size"] != expected["size"] or actual.get("etag") != expected.get("etag"):
            # Some Swift/S3 gateways return stale Size/ETag values from LIST
            # while HEAD consistently exposes the current, older object.  This
            # is not post-audit drift when two direct observations agree and
            # LastModified predates the audit.  Record the reconciliation so
            # reclaimed-byte accounting uses the authoritative HEAD identity.
            headed = _head_optional(client, bucket, key)
            time.sleep(0.1)
            confirmed = _head_optional(client, bucket, key)
            audit_time = report.get("generated_at")
            stable_head = bool(
                headed is not None and confirmed is not None
                and headed["size"] == confirmed["size"]
                and headed.get("etag") == confirmed.get("etag")
            )
            old_object = bool(
                audit_time and headed and headed.get("last_modified")
                and str(headed["last_modified"]) <= str(audit_time)
            )
            if not stable_head or not old_object:
                raise RuntimeError(f"{record['sequence']}: audited legacy key changed")
            current_by_key[key] = headed
            identity_reconciliations.append({
                "key": key,
                "listed_size": expected["size"],
                "listed_etag": expected.get("etag"),
                "head_size": headed["size"],
                "head_etag": headed.get("etag"),
                "last_modified": headed.get("last_modified"),
                "reason": "stale_list_metadata_confirmed_by_stable_head",
            })
    _verify_protected(protected, shard)
    pending._put_json(client, bucket, pending_key, {
        "schema": PENDING_SCHEMA,
        "status": "PENDING",
        "created_at": in_progress.get("created_at") if in_progress else _now(),
        "audit_sha256": report["audit_sha256"],
        "campaign_id": report["campaign_id"],
        "sequence": record["sequence"],
        "expected_object_count": record["object_count"],
        "expected_reclaimable_bytes": record["logical_bytes"],
    })
    evidence = _copy_evidence(
        report, shard, client=client, bucket=bucket,
        allow_existing=valid_pending,
    )
    remaining = [
        current_by_key[item["key"]]
        for item in shard["objects"] if item["key"] in current_by_key
    ]
    _delete_and_verify(client, bucket, shard, remaining)
    _verify_protected(
        _list_protected(client, bucket, shard["sequence_root"]), shard,
    )
    result = {
        "schema": COMPLETE_SCHEMA,
        "status": "COMPLETE",
        "completed_at": _now(),
        "audit_sha256": report["audit_sha256"],
        "campaign_id": report["campaign_id"],
        "campaign_name": report["campaign_name"],
        "sequence": record["sequence"],
        "legacy_run_ids": record["legacy_run_ids"],
        "expected_deleted_object_count": record["object_count"],
        "actual_deleted_object_count_total": record["object_count"],
        "actual_deleted_object_count_this_run": len(remaining),
        "expected_reclaimed_logical_bytes": record["logical_bytes"],
        "actual_reclaimed_logical_bytes_total": (
            record["logical_bytes"]
            + sum(
                item["head_size"] - item["listed_size"]
                for item in identity_reconciliations
            )
        ),
        "actual_reclaimed_logical_bytes_this_run": sum(item["size"] for item in remaining),
        "deleted_keys_sha256": _sha256([item["key"] for item in shard["objects"]]),
        "protected_identity_sha256": shard["protected_identity_sha256"],
        "retained_evidence": evidence,
        "identity_reconciliations": identity_reconciliations,
    }
    pending._put_json(client, bucket, complete_key, result)
    pending._remove_pending_marker(client, bucket, pending_key)
    return result


def apply_audit(
    report: dict, *, quota_retry_plan: Path, source_audit_path: Path,
    db_path: str, config: dict, sequences: set[str] | None, max_workers: int,
) -> dict:
    _validate_report(report)
    campaign = db.get_campaign(report["campaign_name"], db_path=db_path)
    if not campaign or int(campaign["id"]) != int(report["campaign_id"]):
        raise ValueError("campaign identity changed after legacy flat audit")
    source = _load_source_audit(
        source_audit_path, report["campaign_name"], report["campaign_id"],
    )
    if source["audit_sha256"] != report["source_audit_sha256"]:
        raise ValueError("source audit differs from legacy flat audit")
    if pending._file_sha256(quota_retry_plan) != report["quota_retry_plan_sha256"]:
        raise ValueError("quota retry plan differs from legacy flat audit")
    if _current_candidate_identity(campaign, quota_retry_plan, db_path=db_path) != (
        _source_candidate_identity(source)
    ):
        raise RuntimeError("campaign candidates changed after legacy flat audit")
    current_lineage = _legacy_by_sequence(
        source, _legacy_snapshot(campaign["dataset"], db_path=db_path),
    )
    audited_lineage = {
        record["sequence"]: record["legacy_run_ids"] for record in report["records"]
    }
    if {
        sequence: [int(run["reconstruction_run_id"]) for run in runs]
        for sequence, runs in current_lineage.items()
    } != audited_lineage:
        raise RuntimeError("legacy reconstruction lineage changed after audit")
    selected = [
        record for record in report["records"]
        if (
            (sequences is not None and record["sequence"] in sequences)
            or (sequences is None and record["object_count"] > 0)
        )
    ]
    if sequences is not None and {item["sequence"] for item in selected} != sequences:
        raise ValueError("requested canary sequence is absent from legacy flat audit")
    client, bucket, _prefix, _uri = _output_root(campaign, config)
    results = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(_apply_record, report, record, client=client, bucket=bucket): (
                record["sequence"]
            )
            for record in selected
        }
        try:
            for index, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if index % 25 == 0 or index == len(futures):
                    print(f"Applied {index}/{len(futures)} legacy sequence(s)", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    results.sort(key=lambda item: item["sequence"])
    expected = sum(record["logical_bytes"] for record in selected)
    verified = sum(item["actual_reclaimed_logical_bytes_total"] for item in results)
    return {
        "schema": APPLY_SCHEMA,
        "completed_at": _now(),
        "audit_sha256": report["audit_sha256"],
        "campaign_id": report["campaign_id"],
        "selected_sequence_count": len(selected),
        "completed_sequence_count": sum(item["status"] == "COMPLETE" for item in results),
        "already_complete_sequence_count": sum(
            item["status"] == "ALREADY_COMPLETE" for item in results
        ),
        "expected_reclaimed_logical_bytes": expected,
        "verified_reclaimed_logical_bytes": verified,
        "actual_reclaimed_logical_bytes_this_run": sum(
            item["actual_reclaimed_logical_bytes_this_run"] for item in results
        ),
        "variance_bytes": verified - expected,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--expected-campaign-id", type=int, required=True)
    parser.add_argument("--quota-retry-plan", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--expected-object-count", type=int)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config = load_config(MV_HOI_DIR)
    lock_path = state_path("locks") / "mv_hoi_campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("campaign orchestration lock is already held") from exc
        if not args.apply:
            if args.sequence:
                parser.error("--sequence is only valid with --apply")
            report = audit(
                args.campaign, expected_campaign_id=args.expected_campaign_id,
                quota_retry_plan=args.quota_retry_plan,
                source_audit_path=args.source_audit, report_path=args.report,
                manifest_dir=args.manifest_dir, db_path=args.db, config=config,
                max_workers=args.max_workers,
            )
            result = None
        else:
            require_submit_authority("delete legacy flat reconstruction intermediates")
            report = json.loads(args.report.read_text())
            if args.expected_object_count is not None and report["object_count"] != (
                args.expected_object_count
            ):
                raise RuntimeError("legacy flat object count differs from expectation")
            result = apply_audit(
                report, quota_retry_plan=args.quota_retry_plan,
                source_audit_path=args.source_audit, db_path=args.db, config=config,
                sequences=set(args.sequence) or None, max_workers=args.max_workers,
            )
    if args.expected_object_count is not None and report["object_count"] != (
        args.expected_object_count
    ):
        raise RuntimeError("legacy flat object count differs from expectation")
    apply_path = args.report.with_name(
        args.report.stem + (".canary.apply.json" if args.sequence else ".apply.json")
    )
    if result is not None:
        apply_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report": str(args.report),
        "audit_sha256": report["audit_sha256"],
        "campaign_id": report["campaign_id"],
        "legacy_lineage_sequence_count": report["legacy_lineage_sequence_count"],
        "sequences_with_objects": report["sequences_with_objects"],
        "object_count": report["object_count"],
        "logical_bytes": report["logical_bytes"],
        "decimal": report["decimal"],
        "binary": report["binary"],
        "applied": result is not None,
        "apply_report": str(apply_path) if result is not None else None,
        "apply": ({key: value for key, value in result.items() if key != "results"}
                  if result is not None else None),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
