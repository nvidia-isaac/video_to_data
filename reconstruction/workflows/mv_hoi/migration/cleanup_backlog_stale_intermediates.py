#!/usr/bin/env python3
"""Delete stale canonical intermediates and requeue exact backlog members.

This recovery is separate from post-export cleanup.  It selects only campaign
members that do not already have a successful campaign preprocessing attempt,
preserves available manual assets and reconstruction evidence, and requires an
exact campaign ID plus ``--apply`` before changing storage or request state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
MV_HOI_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(MV_HOI_DIR))

from orchestration import cleanup_intermediates as cleanup, db  # noqa: E402
from orchestration.config_utils import load_config  # noqa: E402
from orchestration.runtime import require_submit_authority, state_path  # noqa: E402


SCHEMA = "v2d.mv_hoi.backlog_stale_intermediate_recovery.v1"


def _requests_by_sequence(campaign: dict, *, db_path: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for request in db.list_stage_requests(campaign=campaign["name"], db_path=db_path):
        result.setdefault(request["sequence_name"], []).append(request)
    return result


def audit(
    campaign_name: str, *, expected_campaign_id: int, db_path: str, config: dict,
) -> dict:
    campaign = db.get_campaign(campaign_name, db_path=db_path)
    if not campaign or int(campaign["id"]) != int(expected_campaign_id):
        raise ValueError("campaign name and --expected-campaign-id do not match")
    if campaign["campaign_type"] not in ("BACKLOG_REPROCESSING", "REMEDIATION"):
        raise ValueError("stale cleanup is limited to backlog/remediation campaigns")
    requests = _requests_by_sequence(campaign, db_path=db_path)
    active = [
        request for rows in requests.values() for request in rows
        if request["status"] in cleanup.ACTIVE_REQUEST_STATUSES
    ]
    if active:
        raise RuntimeError(
            f"campaign still owns {len(active)} active/ambiguous request(s); "
            "cancel and reconcile them first"
        )
    dataset_cfg = config["datasets"][campaign["dataset"]]
    output_root_uri = cleanup._join(
        dataset_cfg["swift_base"],
        dataset_cfg["pipelines"]["mv_preprocess"].get(
            "campaign_output_path",
            dataset_cfg["pipelines"]["mv_preprocess"]["output_path"],
        ),
    )
    client, bucket, output_root = cleanup._client(output_root_uri)
    records = []
    for sequence, rows in sorted(requests.items()):
        preprocess_requests = [item for item in rows if item["stage"] == "preprocess"]
        successful = []
        for request in preprocess_requests:
            run = db.get_stage_run_by_request(
                request["id"], stage=db.PREPROCESS_STAGE, db_path=db_path,
            )
            if run and run["run_status"] == "SUCCEEDED":
                successful.append(run)
        reconstruction_requests = [
            item for item in rows if item["stage"] == "reconstruction"
        ]
        successful_reconstruction = []
        for request in reconstruction_requests:
            run = db.get_stage_run_by_request(
                request["id"], stage=db.RECONSTRUCTION_STAGE, db_path=db_path,
            )
            if run and run["run_status"] == "SUCCEEDED":
                successful_reconstruction.append(run)
        if successful and successful_reconstruction:
            records.append({
                "sequence": sequence, "status": "PRESERVE_COMPLETED_RECONSTRUCTION",
                "successful_preprocess_run_id": successful[-1]["stage_run_id"],
                "successful_reconstruction_run_id": (
                    successful_reconstruction[-1]["stage_run_id"]
                ),
            })
            continue
        if successful:
            retryable_reconstruction = [
                item for item in reconstruction_requests
                if item["status"] in ("FAILED", "CANCELED")
            ]
            if not retryable_reconstruction:
                records.append({
                    "sequence": sequence, "status": "PRESERVE_COMPLETED_PREPROCESS",
                    "successful_preprocess_run_id": successful[-1]["stage_run_id"],
                })
                continue
            recovery_stage = "reconstruction"
            prior = retryable_reconstruction[-1]
        else:
            recovery_stage = "preprocess"
            prior = preprocess_requests[-1] if preprocess_requests else None
        sequence_root = cleanup._join(output_root, sequence)
        objects = cleanup._list(client, bucket, sequence_root)
        reconstruction_prefixes = []
        for request in (item for item in rows if item["stage"] == "reconstruction"):
            run = db.get_stage_run_by_request(
                request["id"], stage=db.RECONSTRUCTION_STAGE, db_path=db_path,
            )
            if (
                run and run["run_status"] in cleanup.TERMINAL_RUN_STATUSES
                and run.get("output_uri")
            ):
                _run_client, run_bucket, run_prefix = cleanup._client(run["output_uri"])
                if run_bucket != bucket or not run_prefix.startswith(sequence_root + "/"):
                    raise ValueError(
                        f"{sequence}: reconstruction run has unexpected output URI"
                    )
                reconstruction_prefixes.append(run_prefix.rstrip("/") + "/")
        deletions = []
        for item in objects:
            relative = cleanup._relative(sequence_root, item["key"])
            fixed_heavy = recovery_stage == "preprocess" and relative.startswith((
                "rosbag_to_edex/", "mv_preprocess/images/", "face_detector/",
            ))
            resolved_reconstruction = any(
                item["key"].startswith(prefix) for prefix in reconstruction_prefixes
            )
            if (fixed_heavy or resolved_reconstruction) and not cleanup._is_protected(relative):
                deletions.append(item)
        records.append({
            "sequence": sequence,
            "status": f"RESET_TO_{recovery_stage.upper()}",
            "recovery_stage": recovery_stage,
            "prior_request_id": prior["id"] if prior else None,
            "prior_parameters_json": prior.get("parameters_json") if prior else None,
            "prior_source_manifest_json": prior.get("source_manifest_json") if prior else None,
            "deletions": deletions,
            "storage_identity_sha256": cleanup._identity(objects),
        })
    reset = [item for item in records if item["status"].startswith("RESET_TO_")]
    return {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign["id"], "campaign_name": campaign["name"],
        "dataset": campaign["dataset"], "pipeline_version": campaign["pipeline_version"],
        "member_count": len(records), "reset_count": len(reset),
        "preserved_completed_preprocess_count": len(records) - len(reset),
        "object_count": sum(len(item.get("deletions", [])) for item in reset),
        "reclaimed_bytes": sum(
            obj["size"] for item in reset for obj in item.get("deletions", [])
        ),
        "records": records,
    }


def _preserve_attempt_evidence(
    report: dict, record: dict, *, client, bucket: str, sequence_root: str,
    db_path: str,
) -> list[dict]:
    retained = []
    requests = db.list_stage_requests(
        campaign=report["campaign_name"], stage="reconstruction", db_path=db_path,
    )
    for request in (item for item in requests if item["sequence_name"] == record["sequence"]):
        run = db.get_stage_run_by_request(
            request["id"], stage=db.RECONSTRUCTION_STAGE, db_path=db_path,
        )
        if not run or not run.get("output_uri"):
            continue
        source_client, source_bucket, source_prefix = cleanup._client(run["output_uri"])
        by_relative = {
            cleanup._relative(source_prefix, item["key"]): item
            for item in cleanup._list(source_client, source_bucket, source_prefix)
        }
        files = []
        destination_root = cleanup._join(
            sequence_root, "metrics", "reconstruction", str(request["id"]),
        )
        for original, destination_name in cleanup.FALLBACK_EVIDENCE.items():
            item = by_relative.get(original)
            if not item:
                continue
            payload, response = cleanup._get(source_client, source_bucket, item["key"])
            key = cleanup._join(destination_root, destination_name)
            client.put_object(Bucket=bucket, Key=key, Body=payload)
            files.append({
                "path": destination_name, "original_key": item["key"],
                "size": len(payload), "etag": cleanup._etag(response),
                "sha256": cleanup._sha256(payload),
            })
        manifest = {
            "schema": cleanup.METRICS_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "reconstruction", "request_id": request["id"],
            "campaign_id": report["campaign_id"], "campaign_name": report["campaign_name"],
            "reconstruction_run_id": run["stage_run_id"],
            "workflow": run.get("workflow_name"), "pipeline_version": run.get("pipeline_version"),
            "source_uri": run["output_uri"],
            "categorized_failure_reason": cleanup.categorized_failure_reason(
                run.get("details")
            ),
            "files": files,
        }
        client.put_object(
            Bucket=bucket, Key=cleanup._join(destination_root, "manifest.json"),
            Body=(json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            ContentType="application/json",
        )
        retained.append(manifest)
    return retained


def apply(report: dict, *, db_path: str, config: dict) -> dict:
    dataset_cfg = config["datasets"][report["dataset"]]
    output_root_uri = cleanup._join(
        dataset_cfg["swift_base"],
        dataset_cfg["pipelines"]["mv_preprocess"].get(
            "campaign_output_path",
            dataset_cfg["pipelines"]["mv_preprocess"]["output_path"],
        ),
    )
    client, bucket, output_root = cleanup._client(output_root_uri)
    deleted = 0
    retained_attempts = 0
    created_requests = 0
    for record in report["records"]:
        if not record["status"].startswith("RESET_TO_"):
            continue
        sequence_root = cleanup._join(output_root, record["sequence"])
        current = cleanup._list(client, bucket, sequence_root)
        if cleanup._identity(current) != record["storage_identity_sha256"]:
            raise RuntimeError(f"{record['sequence']}: storage changed after audit")
        retained_attempts += len(_preserve_attempt_evidence(
            report, record, client=client, bucket=bucket,
            sequence_root=sequence_root, db_path=db_path,
        ))
        cleanup._delete(
            client, bucket, [item["key"] for item in record["deletions"]],
        )
        deleted += len(record["deletions"])
        recovery_stage = record["recovery_stage"]
        existing_pending = [
            item for item in db.list_stage_requests(
                campaign=report["campaign_name"], stage=recovery_stage, db_path=db_path,
            )
            if item["sequence_name"] == record["sequence"] and item["status"] == "PENDING"
        ]
        if existing_pending:
            continue
        prior_parameters = json.loads(record.get("prior_parameters_json") or "{}")
        for key in (
            "processing_output_layout", "processing_output_path",
            "retry_work_output_url", "prior_request_id", "prior_execution_id",
        ):
            prior_parameters.pop(key, None)
        prior_manifest = json.loads(record.get("prior_source_manifest_json") or "{}")
        db.create_stage_request(
            sequence_name=record["sequence"], dataset=report["dataset"],
            stage=recovery_stage, pipeline_version=report["pipeline_version"],
            trigger="MIGRATION", requested_by="backlog_stale_recovery",
            reason=(
                "stale_intermediate_recovery_of_request_"
                + str(record.get("prior_request_id") or "none")
            ),
            campaign=report["campaign_id"], cohort="BULK",
            parameters=prior_parameters, source_manifest=prior_manifest,
            status="PENDING", db_path=db_path,
        )
        created_requests += 1
    return {
        "deleted_objects": deleted, "retained_attempt_metrics": retained_attempts,
        "created_retry_requests": created_requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign")
    parser.add_argument("--expected-campaign-id", type=int, required=True)
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    config = load_config(MV_HOI_DIR)
    report = audit(
        args.campaign, expected_campaign_id=args.expected_campaign_id,
        db_path=args.db, config=config,
    )
    if args.apply:
        require_submit_authority("delete stale backlog intermediates and requeue preprocessing")
        report["apply_result"] = apply(report, db_path=args.db, config=config)
        report["applied_at"] = datetime.now(timezone.utc).isoformat()
    report_path = args.report or state_path("manifests") / (
        f"backlog_stale_cleanup_{report['campaign_id']}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report": str(report_path), "campaign_id": report["campaign_id"],
        "member_count": report["member_count"], "reset_count": report["reset_count"],
        "object_count": report["object_count"], "reclaimed_bytes": report["reclaimed_bytes"],
        "applied": args.apply, "apply_result": report.get("apply_result"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
