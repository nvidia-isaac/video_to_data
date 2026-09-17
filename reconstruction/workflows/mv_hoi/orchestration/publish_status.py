"""Publish MV HOI lifecycle or pipeline-attempt status to Google Sheets.

The default mode publishes one DB-only active-campaign lifecycle snapshot to
the data and summary tabs. Explicit pipeline modes retain the legacy
attempt-oriented publisher and refresh behavior. Google-specific code stays at
the edge so row generation and summary aggregation remain independently
testable.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MV_HOI_DIR = SCRIPT_DIR.parent
try:
    from .config_utils import (
        CALIBRATION_PIPELINE,
        EXPORT_PIPELINE,
        PREPROCESS_PIPELINE,
        RECON_PIPELINE,
    )
    from . import db as workflow_db
    from .query import DEFAULT_REFRESH_WORKERS, refresh_workflow_states
    from .lifecycle import (
        LIFECYCLE_BUCKET_ORDER,
        build_campaign_lifecycle_rows,
        build_campaign_lifecycle_summary,
        lifecycle_membership_reconciles,
    )
except ImportError:  # Direct script execution.
    from config_utils import (
        CALIBRATION_PIPELINE,
        EXPORT_PIPELINE,
        PREPROCESS_PIPELINE,
        RECON_PIPELINE,
    )
    import db as workflow_db
    from query import DEFAULT_REFRESH_WORKERS, refresh_workflow_states
    from lifecycle import (
        LIFECYCLE_BUCKET_ORDER,
        build_campaign_lifecycle_rows,
        build_campaign_lifecycle_summary,
        lifecycle_membership_reconciles,
    )

DB_PATH = workflow_db.DB_PATH
TABLE = workflow_db.PIPELINES_TABLE

SEQUENCE_STAGES = (PREPROCESS_PIPELINE, RECON_PIPELINE, EXPORT_PIPELINE)
ORDERED_STAGES = (CALIBRATION_PIPELINE, *SEQUENCE_STAGES)

STATUS_HEADERS = (
    "dataset",
    "sequence_name",
    "stage",
    "status",
    "details",
    "stage_run_id",
    "attempt",
    "trigger",
    "target_id",
    "upstream_run_ids",
    "pipeline_version",
    "workflow_name",
    "osmo_workflow_id",
    "execution_id",
    "workflow_task_name",
    "auxiliary_task_name",
    "created_at",
    "updated_at",
)
DATA_SEQUENCE_STATUS_HEADERS = (
    "dataset",
    "sequence_name",
    "calibration_sequence_name",
    "blacklist_reason",
    "blacklisted_by",
    "blacklisted_at",
    "lifecycle_status",
    "lifecycle_stage",
    "lifecycle_details",
    "lifecycle_updated_at",
    "campaign_id",
    "campaign_name",
    "campaign_type",
    "campaign_status",
    "campaign_phase",
    "cohort",
    "authoritative_request_id",
    "fulfillment_request_id",
    "effective_request_id",
    "request_stage",
    "request_status",
    "blocked_reason",
    "pipeline_version",
    "preprocess_run_id",
    "preprocess_run_status",
    "preprocess_details",
    "preprocess_workflow",
    "reconstruction_run_id",
    "reconstruction_run_status",
    "reconstruction_details",
    "reconstruction_workflow",
    "qc_review_id",
    "qc_decision",
    "export_run_id",
    "export_run_status",
    "export_authorization",
    "export_workflow",
    "failure_categories",
)
SEQUENCE_STATUS_HEADERS = DATA_SEQUENCE_STATUS_HEADERS

CALIBRATION_SEQUENCE_STATUS_HEADERS = (
    "dataset",
    "sequence_name",
    "blacklist_reason",
    "blacklisted_by",
    "blacklisted_at",
    "effective_status",
    "effective_details",
    "effective_updated_at",
    "calibration_run_id",
    "calibration_status",
    "calibration_run_status",
    "calibration_version",
    "calibration_details",
    "calibration_workflow",
    "last_activity_at",
)
ORDERED_STATUSES = (
    "WAITING_WF",
    "WAITING_QC",
    "WAITING_EXPORT",
    "PASS",
    "FAIL",
    "SKIPPED",
)
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"


def load_config() -> dict:
    with open(MV_HOI_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _stage(workflow: dict) -> str:
    """Return the normalized stage name, accepting legacy flattened rows."""
    return _text(workflow.get("stage") or workflow.get("pipeline_type"))


def _stage_sort_key(stage: str) -> tuple[int, str]:
    try:
        return ORDERED_STAGES.index(stage), stage
    except ValueError:
        return len(ORDERED_STAGES), stage


def _status_value(workflow: dict, header: str) -> Any:
    if header == "stage":
        return _stage(workflow)
    if header == "stage_run_id":
        return workflow.get("stage_run_id", workflow.get("run_id", workflow.get("id")))
    if header == "execution_id":
        return workflow.get("execution_id", workflow.get("osmo_workflow_id"))
    return workflow.get(header)


def latest_workflows(workflows: list[dict]) -> list[dict]:
    """Deduplicate newest-first rows by sequence/stage, then sort for display."""
    seen: set[tuple[str, str]] = set()
    latest = []
    for workflow in workflows:
        key = (workflow["sequence_name"], _stage(workflow))
        if key in seen:
            continue
        seen.add(key)
        latest.append(workflow)
    return sorted(
        latest,
        key=lambda workflow: (
            workflow["sequence_name"],
            _stage_sort_key(_stage(workflow)),
        ),
    )


def workflow_to_status_row(workflow: dict) -> list[str]:
    return [_text(_status_value(workflow, header)) for header in STATUS_HEADERS]


def status_values(workflows: list[dict]) -> list[list[str]]:
    if not workflows or "preprocess_status" in workflows[0]:
        return [list(SEQUENCE_STATUS_HEADERS)] + [
            [_text(row.get(header)) for header in SEQUENCE_STATUS_HEADERS]
            for row in workflows
        ]
    return [list(STATUS_HEADERS)] + [
        workflow_to_status_row(workflow) for workflow in workflows
    ]


def lifecycle_status_values(rows: list[dict]) -> list[list[str]]:
    aliases = {
        "preprocess_details": "preprocess_run_details",
        "reconstruction_details": "reconstruction_run_details",
        "export_authorization": "export_authorization_type",
    }
    return [list(DATA_SEQUENCE_STATUS_HEADERS)] + [
        [
            _text(row.get(aliases.get(header, header)))
            for header in DATA_SEQUENCE_STATUS_HEADERS
        ]
        for row in sorted(rows, key=lambda item: item["sequence_name"])
    ]


def calibration_status_values(rows: list[dict]) -> list[list[str]]:
    return [list(CALIBRATION_SEQUENCE_STATUS_HEADERS)] + [
        [_text(row.get(header)) for header in CALIBRATION_SEQUENCE_STATUS_HEADERS]
        for row in sorted(rows, key=lambda item: item["sequence_name"])
    ]


def lifecycle_summary_values(summary: dict, *, dataset: str) -> list[list[str]]:
    expected = summary["expected_distinct_membership"]
    integrity = (
        "MEMBERSHIP_MISMATCH"
        if not lifecycle_membership_reconciles(summary)
        else "INCONSISTENT"
        if summary["has_inconsistent"]
        else "OK"
    )
    rows = [
        ["DB observed at", _text(summary["observed_at"])],
        ["Dataset", dataset],
        ["Campaign scope", "FROZEN/RUNNING"],
        ["Campaigns selected", str(len(summary["campaigns"]))],
        ["Distinct membership", str(summary["distinct_membership"])],
        ["Expected distinct membership", "unknown" if expected is None else str(expected)],
        ["Membership references", str(summary["actual_membership_references"])],
        ["Overlap", str(summary["overlap"])],
        ["Reconciled lifecycle total", str(summary["reconciled_total"])],
        ["Integrity", integrity],
        [],
        [
            "Campaign ID", "Campaign", "Type", "Status", "Phase",
            "Actual members", "Expected members",
        ],
    ]
    for campaign in summary["campaigns"]:
        campaign_expected = campaign["expected_member_count"]
        rows.append([
            str(campaign["id"]),
            _text(campaign["name"]),
            _text(campaign["campaign_type"]),
            _text(campaign["status"]),
            _text(campaign["phase"]),
            str(campaign["actual_member_count"]),
            "unknown" if campaign_expected is None else str(campaign_expected),
        ])

    rows.extend([[], ["Lifecycle status", "Count"]])
    emitted: set[str] = set()
    for bucket in LIFECYCLE_BUCKET_ORDER:
        if bucket in summary["buckets"]:
            rows.append([bucket, str(summary["buckets"][bucket])])
            emitted.add(bucket)
    for bucket in sorted(set(summary["buckets"]) - emitted):
        rows.append([bucket, str(summary["buckets"][bucket])])

    for title, values in (
        ("EXPORTED authorization", summary["export_authorizations"]),
        ("Blocked reason", summary["blocked_reasons"]),
        ("Categorized failure", summary["failure_categories"]),
    ):
        rows.extend([[], [title, "Count"]])
        if values:
            rows.extend(
                [name, str(count)]
                for name, count in sorted(
                    values.items(), key=lambda item: (-item[1], item[0]),
                )
            )
        else:
            rows.append(["(none)", "0"])
    return rows


def _detail_counts(workflows: list[dict], status: str) -> list[tuple[str, int]]:
    counts = Counter(
        _text(workflow.get("details")) or "(no details)"
        for workflow in workflows
        if workflow.get("status") == status
    )
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def summary_values(
    workflows: list[dict],
    *,
    dataset: str,
    pipeline_scope: str,
    generated_at: datetime | None = None,
) -> list[list[str]]:
    generated_at = generated_at or datetime.now().astimezone()
    if workflows and "preprocess_status" in workflows[0]:
        flattened = []
        for row in workflows:
            for stage in ("calibration", "preprocess", "reconstruction", "export"):
                flattened.append({
                    "stage": stage,
                    "status": row[f"{stage}_status"],
                    "details": row.get(f"{stage}_details"),
                })
        rows = summary_values(
            flattened, dataset=dataset, pipeline_scope=pipeline_scope,
            generated_at=generated_at,
        )
        rows.insert(4, ["Blacklisted sequences", str(sum(
            bool(row.get("blacklist_reason")) for row in workflows
        ))])
        return rows
    counts = Counter(
        (_stage(workflow), _text(workflow.get("status"))) for workflow in workflows
    )
    rows: list[list[str]] = [
        ["Generated at", generated_at.isoformat(timespec="seconds")],
        ["Dataset", dataset],
        ["Pipeline scope", pipeline_scope],
        ["Published rows", str(len(workflows))],
        [],
        ["Stage", "Status", "Count"],
    ]
    present_stages = {_stage(workflow) for workflow in workflows}
    stages = [stage for stage in ORDERED_STAGES if stage in present_stages]
    stages.extend(sorted(present_stages - set(stages)))
    for stage in stages:
        for status in ORDERED_STATUSES:
            rows.append([stage, status, str(counts.get((stage, status), 0))])
        for (count_stage, status), count in sorted(counts.items()):
            if count_stage == stage and status not in ORDERED_STATUSES:
                rows.append([stage, status, str(count)])

    for status in ("FAIL", "SKIPPED"):
        rows.extend([[], ["Stage", f"Top {status} details", "Count"]])
        found = False
        for stage in stages:
            stage_workflows = [
                workflow for workflow in workflows if _stage(workflow) == stage
            ]
            for detail, count in _detail_counts(stage_workflows, status)[:20]:
                found = True
                rows.append([stage, detail, str(count)])
        if not found:
            rows.append(["(none)", "(none)", "0"])
    return rows


def load_status_workflows(
    *,
    dataset: str,
    pipeline_type: str | None,
    latest: bool,
    stages: tuple[str, ...] | None = None,
    db_path: str | Path = DB_PATH,
    table: str = TABLE,
) -> list[dict]:
    if latest and pipeline_type is None and stages is None:
        return workflow_db.get_sequence_status(dataset, db_path=str(db_path))
    requested_stage = None if stages else pipeline_type
    list_current = getattr(workflow_db, "list_current_stage_runs", None)
    if latest and list_current is not None:
        workflows = list_current(
            dataset,
            stage=requested_stage,
            db_path=str(db_path),
            table=table,
        )
    else:
        workflows = workflow_db.get_workflows_by_dataset(
            dataset,
            pipeline_type=requested_stage,
            db_path=str(db_path),
            table=table,
        )
    if stages is not None:
        workflows = [workflow for workflow in workflows if _stage(workflow) in stages]
    if latest:
        return latest_workflows(workflows)
    return sorted(
        workflows,
        key=lambda workflow: (
            workflow["sequence_name"],
            _stage_sort_key(_stage(workflow)),
            workflow.get("created_at") or "",
            workflow.get("id") or 0,
        ),
    )


def _quote_sheet_name(title: str) -> str:
    return "'" + title.replace("'", "''") + "'"


def build_sheets_service(credentials_path: str):
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Google Sheets publishing requires google-api-python-client and "
            "google-auth. Install workflows/mv_hoi/requirements.txt."
        ) from exc

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=[SHEETS_SCOPE],
    )
    return build("sheets", "v4", credentials=credentials, cache_discovery=False)


def existing_sheet_titles(service, spreadsheet_id: str) -> set[str]:
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return {
        sheet.get("properties", {}).get("title", "")
        for sheet in spreadsheet.get("sheets", [])
    }


def ensure_worksheet(service, spreadsheet_id: str, title: str) -> None:
    if title in existing_sheet_titles(service, spreadsheet_id):
        return
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": title}}}]},
    ).execute()


def clear_and_write_values(
    service,
    spreadsheet_id: str,
    worksheet: str,
    values: list[list[str]],
) -> None:
    sheet_ref = _quote_sheet_name(worksheet)
    service.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_ref}!A:AZ",
        body={},
    ).execute()
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"{sheet_ref}!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def publish_to_sheets(
    service,
    *,
    spreadsheet_id: str,
    status_worksheet: str,
    summary_worksheet: str,
    status_rows: list[list[str]],
    summary_rows: list[list[str]],
) -> None:
    for worksheet in (status_worksheet, summary_worksheet):
        ensure_worksheet(service, spreadsheet_id, worksheet)
    clear_and_write_values(service, spreadsheet_id, status_worksheet, status_rows)
    clear_and_write_values(service, spreadsheet_id, summary_worksheet, summary_rows)


def publish_current_status_to_sheets(
    service, *, spreadsheet_id: str, summary_worksheet: str,
    data_worksheet: str, calibration_worksheet: str,
    summary_rows: list[list[str]], data_rows: list[list[str]],
    calibration_rows: list[list[str]],
) -> None:
    for worksheet in (summary_worksheet, data_worksheet, calibration_worksheet):
        ensure_worksheet(service, spreadsheet_id, worksheet)
    clear_and_write_values(service, spreadsheet_id, data_worksheet, data_rows)
    clear_and_write_values(
        service, spreadsheet_id, calibration_worksheet, calibration_rows,
    )
    # The summary is the publication marker and is written only after both
    # detailed tabs have completed.
    clear_and_write_values(service, spreadsheet_id, summary_worksheet, summary_rows)


def resolve_publish_settings(
    *,
    spreadsheet_id: str | None,
    dry_run: bool,
) -> tuple[str, str | None]:
    resolved_spreadsheet_id = spreadsheet_id or os.environ.get(
        "MV_HOI_STATUS_SPREADSHEET_ID", ""
    )
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if dry_run:
        return resolved_spreadsheet_id or "(unset)", credentials_path

    if not resolved_spreadsheet_id:
        raise RuntimeError(
            "Missing spreadsheet ID. Pass --spreadsheet-id or set "
            "MV_HOI_STATUS_SPREADSHEET_ID."
        )
    if not credentials_path:
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_APPLICATION_CREDENTIALS."
        )
    path = Path(credentials_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"Google credentials file does not exist: {path}")
    return resolved_spreadsheet_id, str(path)


def _print_dry_run(
    *,
    spreadsheet_id: str,
    status_worksheet: str,
    summary_worksheet: str,
    workflows: list[dict],
) -> None:
    if workflows and "preprocess_status" in workflows[0]:
        print("Dry run: Google Sheets was not updated.")
        print(f"Spreadsheet ID: {spreadsheet_id}")
        print(f"Status worksheet: {status_worksheet}")
        print(f"Summary worksheet: {summary_worksheet}")
        print(f"Published sequence rows: {len(workflows)}")
        print(f"Blacklisted: {sum(bool(row.get('blacklist_reason')) for row in workflows)}")
        return
    counts = Counter(
        (_stage(workflow), _text(workflow.get("status"))) for workflow in workflows
    )
    print("Dry run: Google Sheets was not updated.")
    print(f"Spreadsheet ID: {spreadsheet_id}")
    print(f"Status worksheet: {status_worksheet}")
    print(f"Summary worksheet: {summary_worksheet}")
    print(f"Published rows: {len(workflows)}")
    for (stage, status), count in sorted(counts.items()):
        print(f"  {stage} / {status}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Publish active-campaign lifecycle status (default) or explicit "
            "pipeline-attempt status to Google Sheets"
        ),
    )
    parser.add_argument("--dataset", required=True)
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument("--pipeline", "--stage")
    scope_group.add_argument("--all-pipelines", "--all-stages", action="store_true")
    scope_group.add_argument(
        "--sequence-stages",
        action="store_true",
        help="Publish preprocess, reconstruction, and export stages together",
    )
    latest_group = parser.add_mutually_exclusive_group()
    latest_group.add_argument("--latest", dest="latest", action="store_true", default=True)
    latest_group.add_argument("--all-rows", dest="latest", action="store_false")
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument("--refresh", dest="refresh", action="store_true")
    refresh_group.add_argument("--no-refresh", dest="refresh", action="store_false")
    parser.set_defaults(refresh=None)
    parser.add_argument("--spreadsheet-id")
    parser.add_argument("--status-worksheet", default="latest_status")
    parser.add_argument("--summary-worksheet", default="summary")
    parser.add_argument("--data-worksheet", default="data_sequence_status")
    parser.add_argument(
        "--calibration-worksheet", default="calibration_sequence_status",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--refresh-workers",
        type=int,
        default=DEFAULT_REFRESH_WORKERS,
        help=f"Concurrent OSMO refresh workers (default: {DEFAULT_REFRESH_WORKERS})",
    )
    args = parser.parse_args()

    config = load_config()
    if args.dataset not in config["datasets"]:
        raise SystemExit(f"Unknown dataset: {args.dataset}")
    dataset_cfg = config["datasets"][args.dataset]
    if (
        not args.all_pipelines
        and not args.sequence_stages
        and args.pipeline
        and args.pipeline not in dataset_cfg.get("pipelines", {})
        and args.pipeline != EXPORT_PIPELINE
    ):
        raise SystemExit(f"Unknown pipeline: {args.pipeline}")

    table = (
        workflow_db.PIPELINES_TEST_TABLE
        if args.test
        else workflow_db.PIPELINES_TABLE
    )
    db_path = workflow_db.TEST_DB_PATH if args.test else workflow_db.DB_PATH
    stages = SEQUENCE_STAGES if args.sequence_stages else None
    pipeline_type = None if args.all_pipelines or stages else args.pipeline
    current_mode = not (
        args.pipeline or args.all_pipelines or args.sequence_stages or not args.latest
    )
    if args.all_pipelines:
        pipeline_scope = "all stages"
    elif stages:
        pipeline_scope = "sequence stages"
    elif args.pipeline:
        pipeline_scope = args.pipeline
    else:
        pipeline_scope = "sequence status"
    spreadsheet_id, credentials_path = resolve_publish_settings(
        spreadsheet_id=args.spreadsheet_id,
        dry_run=args.dry_run,
    )

    should_refresh = args.refresh if args.refresh is not None else not current_mode
    if not current_mode:
        workflow_db.init_db(db_path)
    if should_refresh:
        refresh_workflow_states(
            args.dataset,
            pipeline_type=pipeline_type,
            db_path=db_path,
            table=table,
            max_workers=args.refresh_workers,
        )

    if current_mode:
        snapshot = workflow_db.get_campaign_lifecycle_snapshot(
            args.dataset, db_path=db_path,
        )
        data = build_campaign_lifecycle_rows(snapshot)
        summary = build_campaign_lifecycle_summary(snapshot, rows=data)
        if not lifecycle_membership_reconciles(summary):
            raise RuntimeError(
                "Refusing to publish: campaign lifecycle membership does not reconcile "
                f"({summary['reconciled_total']}/"
                f"{summary['distinct_membership']})"
            )
        calibration = workflow_db.get_sequence_status(
            args.dataset, db_path=db_path, sequence_kind="calibration",
        )
        data_values = lifecycle_status_values(data)
        calibration_values = calibration_status_values(calibration)
        summary_rows = lifecycle_summary_values(summary, dataset=args.dataset)
        if args.dry_run:
            print("Dry run: Google Sheets was not updated.")
            print(f"Spreadsheet ID: {spreadsheet_id}")
            print(f"Summary worksheet: {args.summary_worksheet}")
            print(f"Data worksheet: {args.data_worksheet}")
            print(f"Calibration worksheet: {args.calibration_worksheet}")
            print(f"Published data sequence rows: {len(data)}")
            print(f"Published calibration sequence rows: {len(calibration)}")
            print(
                "Reconciled lifecycle total: "
                f"{summary['reconciled_total']}/"
                f"{summary['distinct_membership']}"
            )
            print(f"Lifecycle overlap: {summary['overlap']}")
            print(
                "Lifecycle inconsistencies: "
                f"{summary['buckets'].get('INCONSISTENT', 0)}"
            )
            return
        assert credentials_path is not None
        service = build_sheets_service(credentials_path)
        publish_current_status_to_sheets(
            service, spreadsheet_id=spreadsheet_id,
            summary_worksheet=args.summary_worksheet,
            data_worksheet=args.data_worksheet,
            calibration_worksheet=args.calibration_worksheet,
            summary_rows=summary_rows, data_rows=data_values,
            calibration_rows=calibration_values,
        )
        print(
            f"Published {len(data)} data and {len(calibration)} calibration "
            f"sequence row(s) to {spreadsheet_id}"
        )
        return

    workflows = load_status_workflows(
        dataset=args.dataset,
        pipeline_type=pipeline_type,
        latest=args.latest,
        stages=stages,
        db_path=db_path,
        table=table,
    )
    status_rows = status_values(workflows)
    summary_rows = summary_values(
        workflows,
        dataset=args.dataset,
        pipeline_scope=pipeline_scope,
    )

    if args.dry_run:
        _print_dry_run(
            spreadsheet_id=spreadsheet_id,
            status_worksheet=args.status_worksheet,
            summary_worksheet=args.summary_worksheet,
            workflows=workflows,
        )
        return

    assert credentials_path is not None
    service = build_sheets_service(credentials_path)
    publish_to_sheets(
        service,
        spreadsheet_id=spreadsheet_id,
        status_worksheet=args.status_worksheet,
        summary_worksheet=args.summary_worksheet,
        status_rows=status_rows,
        summary_rows=summary_rows,
    )
    print(
        f"Published {len(workflows)} status row(s) to "
        f"{spreadsheet_id}: {args.status_worksheet}, {args.summary_worksheet}"
    )


if __name__ == "__main__":
    main()
