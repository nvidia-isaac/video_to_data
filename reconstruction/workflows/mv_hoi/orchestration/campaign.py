"""Manage frozen MV-HOI processing campaigns and durable requests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from . import db
except ImportError:
    import db


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _validate_manual_canary_reviews(path: Path, campaign: dict, db_path: str) -> None:
    payload = json.loads(path.read_text())
    reviews = payload.get("manual_reviews") or {}
    canary = db.list_stage_requests(
        campaign=campaign["id"], stage="revalidation", db_path=db_path,
    )
    expected = {
        row["sequence_name"] for row in canary
        if row["cohort"] == "CANARY" and row["status"] == "SUCCEEDED"
    }
    if set(reviews) != expected:
        raise ValueError("Canary report manual-review sequence set does not match the canary")
    for sequence, review in reviews.items():
        if review.get("anonymized_videos") not in ("PASS", "APPROVED"):
            raise ValueError(f"{sequence}: anonymized videos are not approved")
        if review.get("pose_overlay") not in ("PASS", "APPROVED"):
            raise ValueError(f"{sequence}: pose overlay is not approved")
        if not review.get("reviewer") or not review.get("reviewed_at"):
            raise ValueError(f"{sequence}: reviewer and reviewed_at are required")


def _validate_completion_report(path: Path, campaign: dict, db_path: str) -> None:
    payload = json.loads(path.read_text())
    progress = db.campaign_progress(campaign["id"], db_path=db_path)
    identity = payload.get("campaign") or {}
    if identity.get("id") != campaign["id"] or identity.get("inventory_sha256") != campaign["inventory_sha256"]:
        raise ValueError("Completion report campaign identity does not match")
    expected_count = (
        progress["expected_sequence_count"]
        if progress["expected_sequence_count"] is not None
        else progress["sequence_count"]
    )
    if int(payload.get("inventory_sequence_count", -1)) != expected_count:
        raise ValueError("Completion report frozen inventory count does not reconcile")
    if int(payload.get("unique_sequence_count", -1)) != expected_count:
        raise ValueError("Completion report sequence count does not reconcile")
    if int(payload.get("committed_export_count", -1)) != progress["committed_export_count"]:
        raise ValueError("Completion report committed-export count does not reconcile")


def _validate_automated_canary_report(path: Path, campaign: dict) -> None:
    payload = json.loads(path.read_text())
    identity = payload.get("campaign") or {}
    if (
        payload.get("schema") != "v2d.mv_hoi.accuracy_segment_canary_report.v1"
        or payload.get("approval_type") != "AUTOMATED"
        or payload.get("status") != "PASS"
        or identity.get("id") != campaign["id"]
        or identity.get("name") != campaign["name"]
        or identity.get("inventory_sha256") != campaign["inventory_sha256"]
        or identity.get("configuration_sha256") != campaign["configuration_sha256"]
    ):
        raise ValueError("Automated canary report identity or acceptance status is invalid")


def _validate_paired_automated_canary_report(
    path: Path, campaigns: list[dict],
) -> None:
    payload = json.loads(path.read_text())
    identities = payload.get("campaigns") or []
    expected = {
        (item["id"], item["name"], item["inventory_sha256"], item["configuration_sha256"])
        for item in campaigns
    }
    observed = {
        (
            item.get("id"), item.get("name"), item.get("inventory_sha256"),
            item.get("configuration_sha256"),
        )
        for item in identities if isinstance(item, dict)
    }
    if (
        payload.get("schema") != "v2d.mv_hoi.combined_canary_report.v1"
        or payload.get("approval_type") != "AUTOMATED"
        or payload.get("approved_by") != "accuracy-segment-rollout"
        or payload.get("status") != "PASS"
        or observed != expected
        or (payload.get("symmetry_performance") or {}).get("valid") is not True
        or (payload.get("trim_export") or {}).get("revalidation", {}).get("status")
        != "PASS"
        or (payload.get("trim_export") or {}).get("reprocessing", {}).get("status")
        != "PASS"
    ):
        raise ValueError("Paired automated canary report identity or acceptance is invalid")


def _verified_reconciled_revalidation_outputs(
    source_campaign: dict, *, db_path: str,
) -> dict[int, dict]:
    """Verify remote-completed v1.6.41 candidates before export-only adoption."""
    try:
        from .campaign_controller import _read_commit, _revalidation_values
        from .config_utils import load_config
    except ImportError:
        from campaign_controller import _read_commit, _revalidation_values
        from config_utils import load_config

    config = load_config(Path(__file__).resolve().parents[1])
    dataset_cfg = config["datasets"][source_campaign["dataset"]]
    verified: dict[int, dict] = {}
    for request in db.list_stage_requests(
        campaign=source_campaign["id"], stage="revalidation", db_path=db_path,
    ):
        if not (
            request["status"] == "CANCELED"
            and str(request.get("details") or "").startswith(
                "invalid_generation_retired:remote_completed"
            )
            and request.get("workflow_execution_id")
        ):
            continue
        execution = db.get_workflow_execution(
            int(request["workflow_execution_id"]), db_path=db_path,
        )
        query_payload = json.loads(
            (execution or {}).get("last_query_payload_json") or "{}"
        )
        if not (
            execution
            and execution["status"] == "SUCCEEDED"
            and query_payload.get("status") == "COMPLETED"
            and query_payload.get("tasks")
            and all(
                status == "COMPLETED"
                for status in query_payload["tasks"].values()
            )
        ):
            raise ValueError(
                f"Unverified remote completion for request {request['id']}"
            )
        values = _revalidation_values(
            request, source_campaign, dataset_cfg, db_path=db_path,
        )
        candidate_uri = values["work_output_url"].rstrip("/") + "/candidate_export"
        commit, commit_sha256 = _read_commit(candidate_uri, verify_hashes=True)
        if not (
            commit.get("complete") is True
            and commit.get("campaign_name") == source_campaign["name"]
            and commit.get("sequence_name") == request["sequence_name"]
            and commit.get("pipeline_version") == source_campaign["pipeline_version"]
            and commit.get("configuration_sha256")
            == source_campaign["configuration_sha256"]
            and commit.get("source_manifest_sha256")
            == request["source_manifest_sha256"]
        ):
            raise ValueError(
                f"Candidate provenance differs for request {request['id']}"
            )
        verified[int(request["id"])] = {
            "workflow_execution_id": int(execution["id"]),
            "workflow_id": execution["osmo_workflow_id"],
            "candidate_commit_uri": candidate_uri + "/commit.json",
            "candidate_commit_sha256": commit_sha256,
            "commit_schema": commit.get("schema"),
            "commit_file_count": commit.get("file_count"),
        }
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=db.DB_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--name", required=True)
    create.add_argument("--type", required=True, choices=db._CAMPAIGN_TYPES)
    create.add_argument("--dataset", required=True)
    create.add_argument("--version", required=True)
    create.add_argument("--output-uri", required=True)
    create.add_argument("--created-by", default=os.environ.get("USER", "operator"))
    create.add_argument(
        "--canary", action="store_true",
        help="Start this campaign in CANARY phase (revalidation does so by default)",
    )

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("campaign")
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--inventory-uri", required=True)
    freeze.add_argument("--configuration", type=Path, required=True)
    freeze.add_argument("--configuration-uri", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("campaign", nargs="?")
    status.add_argument("--dataset")

    approve = subparsers.add_parser("approve-canary")
    approve.add_argument("campaign")
    approve.add_argument("--report", type=Path, required=True)
    approve.add_argument("--report-uri", required=True)
    approve.add_argument("--approved-by", default=os.environ.get("USER", "operator"))
    approve.add_argument("--automated", action="store_true")

    approve_pair = subparsers.add_parser("approve-paired-canaries")
    approve_pair.add_argument("campaigns", nargs=2)
    approve_pair.add_argument("--report", type=Path, required=True)
    approve_pair.add_argument("--report-uri", required=True)
    approve_pair.add_argument(
        "--approved-by", default="accuracy-segment-rollout",
    )

    finish = subparsers.add_parser("finish")
    finish.add_argument("campaign")
    finish.add_argument("--report", type=Path, required=True)
    cancel = subparsers.add_parser("cancel")
    cancel.add_argument("campaign")

    requests = subparsers.add_parser("requests")
    requests.add_argument("campaign")
    requests.add_argument("--stage", choices=db._REQUEST_STAGES)
    requests.add_argument("--status", action="append", choices=db._REQUEST_STATUSES)
    adopt = subparsers.add_parser("adopt-result")
    adopt.add_argument(
        "--target-request",
        type=int,
        required=True,
        help="Frozen campaign request to mark fulfilled",
    )
    adopt.add_argument(
        "--source-request",
        type=int,
        required=True,
        help="Newer request with a committed REVALIDATION export",
    )
    adopt_revalidation = subparsers.add_parser(
        "adopt-revalidation-export-policy",
        help="Reuse terminal revalidation evidence and rerun only candidate export",
    )
    adopt_reconstruction = subparsers.add_parser(
        "adopt-reconstruction-export-policy",
        help="Reuse verified reconstruction lineage for an export-only patch",
    )
    adopt_preprocess = subparsers.add_parser(
        "adopt-preprocess-patch",
        help="Reuse validated transferred preprocessing in a patch replacement",
    )
    for adoption in (adopt_revalidation, adopt_reconstruction, adopt_preprocess):
        adoption.add_argument("source_campaign")
        adoption.add_argument("replacement_campaign")
        adoption.add_argument(
            "--adopted-by", default=os.environ.get("USER", "operator"),
        )
    adopt_revalidation.add_argument(
        "--verify-reconciled-remote-completions", action="store_true",
        help="Verify canceled remote-completed candidates and adopt export-only inputs",
    )
    successor = subparsers.add_parser(
        "successor",
        help="Create a new-version revalidation campaign without rebuilding inventory",
    )
    successor.add_argument("source_campaign")
    successor.add_argument("--name", required=True)
    successor.add_argument("--version", required=True)
    successor.add_argument(
        "--created-by", default=os.environ.get("USER", "operator"),
    )
    set_version = subparsers.add_parser(
        "set-version",
        help="Advance an active campaign and its undispatched requests in place",
    )
    set_version.add_argument("campaign")
    set_version.add_argument("--version", required=True)
    set_version.add_argument("--message", default="")
    set_version.add_argument(
        "--apply", action="store_true",
        help="Apply the update; without this flag only show campaign status",
    )

    args = parser.parse_args()
    db.init_db(args.db)
    if args.command == "create":
        _print(db.create_campaign(
            name=args.name, campaign_type=args.type, dataset=args.dataset,
            pipeline_version=args.version, output_uri=args.output_uri,
            created_by=args.created_by,
            canary=(True if args.canary else None), db_path=args.db,
        ))
    elif args.command == "freeze":
        inventory_payload = json.loads(args.inventory.read_text())
        _print(db.freeze_campaign(
            args.campaign,
            inventory_uri=args.inventory_uri,
            inventory_sha256=_sha256_file(args.inventory),
            configuration_uri=args.configuration_uri,
            configuration_sha256=_sha256_file(args.configuration),
            inventory_sequence_count=int(inventory_payload["sequence_count"]),
            db_path=args.db,
        ))
    elif args.command == "status":
        if args.campaign:
            _print(db.campaign_progress(args.campaign, db_path=args.db))
        else:
            _print(db.list_campaigns(dataset=args.dataset, db_path=args.db))
    elif args.command == "approve-canary":
        campaign = db.get_campaign(args.campaign, db_path=args.db)
        if campaign is None:
            raise ValueError(f"Unknown campaign: {args.campaign}")
        if args.automated:
            _validate_automated_canary_report(args.report, campaign)
            if args.approved_by != "accuracy-segment-rollout":
                raise ValueError(
                    "Automated approval must use approved_by=accuracy-segment-rollout"
                )
        else:
            _validate_manual_canary_reviews(args.report, campaign, args.db)
        _print(db.approve_campaign_canary(
            args.campaign, report_uri=args.report_uri,
            report_sha256=_sha256_file(args.report),
            approved_by=args.approved_by, db_path=args.db,
        ))
    elif args.command == "approve-paired-canaries":
        if args.approved_by != "accuracy-segment-rollout":
            raise ValueError(
                "Paired automated approval must use "
                "approved_by=accuracy-segment-rollout"
            )
        campaigns = [
            db.get_campaign(name, db_path=args.db) for name in args.campaigns
        ]
        if any(item is None for item in campaigns):
            raise ValueError("Unknown paired campaign")
        _validate_paired_automated_canary_report(args.report, campaigns)
        report_sha256 = _sha256_file(args.report)
        _print(db.approve_paired_campaign_canaries([
            {
                "campaign": item["name"],
                "report_uri": args.report_uri,
                "report_sha256": report_sha256,
                "approved_by": args.approved_by,
            }
            for item in campaigns
        ], db_path=args.db))
    elif args.command == "finish":
        campaign = db.get_campaign(args.campaign, db_path=args.db)
        if campaign is None:
            raise ValueError(f"Unknown campaign: {args.campaign}")
        _validate_completion_report(args.report, campaign, args.db)
        _print(db.finish_campaign(args.campaign, db_path=args.db))
    elif args.command == "cancel":
        _print(db.cancel_campaign(args.campaign, db_path=args.db))
    elif args.command == "adopt-result":
        _print(db.adopt_request_fulfillment(
            args.target_request,
            args.source_request,
            db_path=args.db,
        ))
    elif args.command == "adopt-revalidation-export-policy":
        source = db.get_campaign(args.source_campaign, db_path=args.db)
        if source is None:
            raise ValueError(f"Unknown source campaign: {args.source_campaign}")
        verified = (
            _verified_reconciled_revalidation_outputs(source, db_path=args.db)
            if args.verify_reconciled_remote_completions else None
        )
        _print(db.adopt_valid_replacement_revalidation_results(
            args.source_campaign, args.replacement_campaign,
            adopted_by=args.adopted_by, verified_work_outputs=verified,
            db_path=args.db,
        ))
    elif args.command == "adopt-reconstruction-export-policy":
        _print(db.adopt_valid_replacement_reconstruction_results(
            args.source_campaign, args.replacement_campaign,
            adopted_by=args.adopted_by, db_path=args.db,
        ))
    elif args.command == "adopt-preprocess-patch":
        _print(db.adopt_valid_replacement_preprocess_results(
            args.source_campaign, args.replacement_campaign,
            adopted_by=args.adopted_by, db_path=args.db,
        ))
    elif args.command == "successor":
        _print(db.create_revalidation_successor(
            args.source_campaign,
            name=args.name,
            pipeline_version=args.version,
            created_by=args.created_by,
            db_path=args.db,
        ))
    elif args.command == "set-version":
        campaign = db.get_campaign(args.campaign, db_path=args.db)
        if campaign is None:
            raise ValueError(f"Unknown campaign: {args.campaign}")
        if not args.apply:
            _print({
                "apply": False,
                "campaign": campaign,
                "requested_pipeline_version": args.version,
                "requests": db.campaign_progress(args.campaign, db_path=args.db),
            })
        else:
            _print(db.update_campaign_pipeline_version(
                args.campaign,
                args.version,
                message=args.message,
                db_path=args.db,
            ))
    else:
        _print(db.list_stage_requests(
            campaign=args.campaign, stage=args.stage, status=args.status,
            db_path=args.db,
        ))


if __name__ == "__main__":
    main()
