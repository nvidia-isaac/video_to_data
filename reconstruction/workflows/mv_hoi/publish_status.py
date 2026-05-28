"""Publish MV HOI pipeline status to Google Sheets.

The publisher keeps Google-specific code at the edge: status row generation and
summary aggregation are pure helpers so dry-runs and tests do not require
Google client libraries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_utils import CALIBRATION_PIPELINE, RECON_PIPELINE
from db import (
    PIPELINES_TABLE,
    PIPELINES_TEST_TABLE,
    get_workflows_by_dataset,
    init_db,
)
from query import DEFAULT_REFRESH_WORKERS, refresh_workflow_states

DB_PATH = SCRIPT_DIR / "processing.db"
TABLE = PIPELINES_TABLE

STATUS_HEADERS = (
    "dataset",
    "sequence_name",
    "pipeline_type",
    "status",
    "details",
    "pipeline_version",
    "workflow_name",
    "osmo_workflow_id",
    "osmo_export_workflow_id",
    "created_at",
    "updated_at",
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
    with open(SCRIPT_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def latest_workflows(workflows: list[dict]) -> list[dict]:
    """Deduplicate newest-first rows by sequence/pipeline, then sort for display."""
    seen: set[tuple[str, str]] = set()
    latest = []
    for workflow in workflows:
        key = (workflow["sequence_name"], workflow["pipeline_type"])
        if key in seen:
            continue
        seen.add(key)
        latest.append(workflow)
    return sorted(
        latest,
        key=lambda workflow: (workflow["sequence_name"], workflow["pipeline_type"]),
    )


def workflow_to_status_row(workflow: dict) -> list[str]:
    return [_text(workflow.get(header)) for header in STATUS_HEADERS]


def status_values(workflows: list[dict]) -> list[list[str]]:
    return [list(STATUS_HEADERS)] + [
        workflow_to_status_row(workflow) for workflow in workflows
    ]


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
    counts = Counter(_text(workflow.get("status")) for workflow in workflows)
    rows: list[list[str]] = [
        ["Generated at", generated_at.isoformat(timespec="seconds")],
        ["Dataset", dataset],
        ["Pipeline scope", pipeline_scope],
        ["Published rows", str(len(workflows))],
        [],
        ["Status", "Count"],
    ]
    for status in ORDERED_STATUSES:
        rows.append([status, str(counts.get(status, 0))])
    for status, count in sorted(counts.items()):
        if status not in ORDERED_STATUSES:
            rows.append([status, str(count)])

    rows.extend([[], ["Top FAIL details", "Count"]])
    fail_details = _detail_counts(workflows, "FAIL")
    rows.extend([[detail, str(count)] for detail, count in fail_details[:20]])
    if not fail_details:
        rows.append(["(none)", "0"])

    rows.extend([[], ["Top SKIPPED details", "Count"]])
    skipped_details = _detail_counts(workflows, "SKIPPED")
    rows.extend([[detail, str(count)] for detail, count in skipped_details[:20]])
    if not skipped_details:
        rows.append(["(none)", "0"])
    return rows


def load_status_workflows(
    *,
    dataset: str,
    pipeline_type: str | None,
    latest: bool,
    db_path: str | Path = DB_PATH,
    table: str = TABLE,
) -> list[dict]:
    workflows = get_workflows_by_dataset(
        dataset,
        pipeline_type=pipeline_type,
        db_path=str(db_path),
        table=table,
    )
    if latest:
        return latest_workflows(workflows)
    return sorted(
        workflows,
        key=lambda workflow: (
            workflow["sequence_name"],
            workflow["pipeline_type"],
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
        range=f"{sheet_ref}!A:Z",
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
    counts = Counter(_text(workflow.get("status")) for workflow in workflows)
    print("Dry run: Google Sheets was not updated.")
    print(f"Spreadsheet ID: {spreadsheet_id}")
    print(f"Status worksheet: {status_worksheet}")
    print(f"Summary worksheet: {summary_worksheet}")
    print(f"Published rows: {len(workflows)}")
    for status in ORDERED_STATUSES:
        print(f"  {status}: {counts.get(status, 0)}")
    for status, count in sorted(counts.items()):
        if status not in ORDERED_STATUSES:
            print(f"  {status}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish MV HOI pipeline status to Google Sheets",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pipeline", default=RECON_PIPELINE)
    parser.add_argument("--all-pipelines", action="store_true")
    latest_group = parser.add_mutually_exclusive_group()
    latest_group.add_argument("--latest", dest="latest", action="store_true", default=True)
    latest_group.add_argument("--all-rows", dest="latest", action="store_false")
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument("--refresh", dest="refresh", action="store_true", default=True)
    refresh_group.add_argument("--no-refresh", dest="refresh", action="store_false")
    parser.add_argument("--spreadsheet-id")
    parser.add_argument("--status-worksheet", default="latest_status")
    parser.add_argument("--summary-worksheet", default="summary")
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
    if not args.all_pipelines and args.pipeline not in dataset_cfg.get("pipelines", {}):
        raise SystemExit(f"Unknown pipeline: {args.pipeline}")

    table = PIPELINES_TEST_TABLE if args.test else PIPELINES_TABLE
    pipeline_type = None if args.all_pipelines else args.pipeline
    pipeline_scope = "all pipelines" if args.all_pipelines else args.pipeline
    spreadsheet_id, credentials_path = resolve_publish_settings(
        spreadsheet_id=args.spreadsheet_id,
        dry_run=args.dry_run,
    )

    init_db(str(DB_PATH))
    if args.refresh:
        refresh_workflow_states(
            args.dataset,
            pipeline_type=pipeline_type,
            db_path=str(DB_PATH),
            table=table,
            max_workers=args.refresh_workers,
        )

    workflows = load_status_workflows(
        dataset=args.dataset,
        pipeline_type=pipeline_type,
        latest=args.latest,
        db_path=DB_PATH,
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
