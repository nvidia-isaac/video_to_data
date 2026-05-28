"""Query workflow status, metrics, and aggregate summaries.

Single sequence:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --sequence <name>

Aggregate summary:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --summary

Latest row per sequence:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --latest

List all:
    python query.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import subprocess
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from db import (
    PIPELINES_TABLE,
    PIPELINES_TEST_TABLE,
    get_blacklisted_sequence,
    get_latest_workflow,
    get_recent_workflows_for_sequence,
    get_summary,
    get_workflows_by_export_id,
    get_workflows_by_dataset,
    init_db,
    update_workflow,
    upsert_blacklisted_sequence,
)
from config_utils import CALIBRATION_PIPELINE, RECON_PIPELINE

DB_PATH = os.path.join(SCRIPT_DIR, "processing.db")
TABLE = PIPELINES_TABLE
DEFAULT_REFRESH_WORKERS = int(os.environ.get("MV_HOI_REFRESH_WORKERS", os.cpu_count() or 1))


def load_config() -> dict:
    with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


# OSMO helpers (read-side)

def osmo_query(workflow_name: str) -> dict:
    """Query OSMO workflow status via JSON output.

    Returns {"status": str, "tasks": {name: status}}.
    """
    cmd = [
        "osmo", "workflow", "query", workflow_name, "--format-type", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"status": "UNKNOWN", "tasks": {}}

    data = json.loads(result.stdout)
    status = data.get("status", "UNKNOWN")
    tasks: dict[str, str] = {}
    for group in data.get("groups", []):
        for task in group.get("tasks", []):
            tasks[task["name"]] = task["status"]

    return {"status": status, "tasks": tasks}


def osmo_cancel(workflow_name: str) -> bool:
    """Cancel a running OSMO workflow. Returns True on success."""
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


def _maybe_auto_blacklist_repeated_failure(
    workflow: dict,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> None:
    """Blacklist a sequence after its two latest runs fail with same details."""
    recent = get_recent_workflows_for_sequence(
        workflow["sequence_name"],
        workflow["dataset"],
        pipeline_type=workflow["pipeline_type"],
        limit=2,
        db_path=db_path,
        table=table,
    )
    if len(recent) < 2:
        return
    if any(row["status"] != "FAIL" for row in recent):
        return

    reason = recent[0].get("details") or ""
    if not reason or any((row.get("details") or "") != reason for row in recent):
        return

    existing = get_blacklisted_sequence(
        workflow["dataset"], workflow["sequence_name"], db_path=db_path,
    )
    if existing:
        return

    upsert_blacklisted_sequence(
        workflow["dataset"], workflow["sequence_name"], reason=reason, db_path=db_path,
    )
    print(
        f"Auto-blacklisted {workflow['dataset']}/{workflow['sequence_name']} "
        f"after 2 recent {workflow['pipeline_type']} failures: {reason}"
    )


def _query_waiting_workflow(workflow: dict) -> tuple[dict, dict]:
    """Return (workflow row, OSMO query info) for a WAITING_WF row."""
    osmo_id = workflow.get("osmo_workflow_id") or workflow["workflow_name"]
    return workflow, osmo_query(osmo_id)


def _query_export_workflow(export_id: str) -> tuple[str, dict]:
    """Return (export workflow id, OSMO query info)."""
    return export_id, osmo_query(export_id)


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


def refresh_waiting(
    dataset: str,
    pipeline_type: str | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    max_workers: int = DEFAULT_REFRESH_WORKERS,
) -> None:
    """Poll OSMO for WAITING_WF rows and advance or fail them."""
    workflows = get_workflows_by_dataset(
        dataset, pipeline_type=pipeline_type, status="WAITING_WF",
        db_path=db_path, table=table,
    )
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
        if wf_status == "COMPLETED":
            completed_status = (
                "PASS" if wf["pipeline_type"] == CALIBRATION_PIPELINE else "WAITING_QC"
            )
            update_workflow(wf["workflow_name"], status=completed_status,
                           details="workflow_completed", db_path=db_path,
                           table=table)
        elif wf_status.startswith("FAILED"):
            detail = _failure_detail(info)
            update_workflow(wf["workflow_name"], status="FAIL",
                           details=detail, db_path=db_path, table=table)
            _maybe_auto_blacklist_repeated_failure(wf, db_path=db_path, table=table)


def _export_task_suffix(sequence_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", sequence_name).strip("_") or "sequence"
    digest = hashlib.sha1(sequence_name.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:50]}_{digest}"


def export_task_names(sequence_name: str) -> tuple[str, str]:
    suffix = _export_task_suffix(sequence_name)
    return f"export_{suffix}", f"copy_failure_segments_{suffix}"


def _status_failed(status: str | None) -> bool:
    return bool(status and status.startswith("FAILED"))


def waiting_export_rows(
    dataset: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
) -> list[dict]:
    return get_workflows_by_dataset(
        dataset,
        pipeline_type=RECON_PIPELINE,
        status="WAITING_EXPORT",
        db_path=db_path,
        table=table,
    )


def refresh_waiting_exports(
    dataset: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    max_workers: int = DEFAULT_REFRESH_WORKERS,
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
            pipeline_type=RECON_PIPELINE,
            status="WAITING_EXPORT",
            db_path=db_path,
            table=table,
        )
        for row in batch_rows:
            export_task, copy_task = export_task_names(row["sequence_name"])
            export_status = tasks.get(export_task)
            copy_status = tasks.get(copy_task)

            if export_status == "COMPLETED" and copy_status == "COMPLETED":
                update_workflow(
                    row["workflow_name"],
                    status="PASS",
                    details="export_completed",
                    db_path=db_path,
                    table=table,
                )
                continue

            failed_tasks = [
                name
                for name, status in (
                    (export_task, export_status),
                    (copy_task, copy_status),
                )
                if _status_failed(status)
            ]
            if failed_tasks:
                update_workflow(
                    row["workflow_name"],
                    status="FAIL",
                    details="task_failed: " + ", ".join(failed_tasks),
                    db_path=db_path,
                    table=table,
                )
                continue

            if wf_status == "COMPLETED":
                missing = [
                    name
                    for name, status in (
                        (export_task, export_status),
                        (copy_task, copy_status),
                    )
                    if status != "COMPLETED"
                ]
                update_workflow(
                    row["workflow_name"],
                    status="FAIL",
                    details="export_task_missing: " + ", ".join(missing),
                    db_path=db_path,
                    table=table,
                )
            elif wf_status.startswith("FAILED"):
                update_workflow(
                    row["workflow_name"],
                    status="FAIL",
                    details=f"export_workflow_failed: {wf_status}",
                    db_path=db_path,
                    table=table,
                )


def refresh_workflow_states(
    dataset: str,
    pipeline_type: str | None = None,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    max_workers: int = DEFAULT_REFRESH_WORKERS,
) -> None:
    """Refresh normal OSMO rows plus export rows that belong to reconstruction."""
    refresh_waiting(
        dataset,
        pipeline_type=pipeline_type,
        db_path=db_path,
        table=table,
        max_workers=max_workers,
    )
    if pipeline_type in (None, RECON_PIPELINE):
        refresh_waiting_exports(
            dataset,
            db_path=db_path,
            table=table,
            max_workers=max_workers,
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


def show_summary(
    dataset: str,
    pipeline_type: str | None = None,
    latest_only: bool = False,
) -> None:
    if latest_only:
        workflows = get_workflows_by_dataset(
            dataset, pipeline_type=pipeline_type, db_path=DB_PATH, table=TABLE,
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
    workflows = get_workflows_by_dataset(
        dataset, pipeline_type=pipeline_type, db_path=DB_PATH, table=TABLE,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Query MV pipeline workflow status")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--pipeline", required=True,
                        help="Pipeline type (e.g. mv_calibration, mv_hoi_reconstruction)")
    parser.add_argument("--sequence", help="Show details for a specific sequence")
    parser.add_argument("--summary", action="store_true", help="Show aggregate summary")
    parser.add_argument("--latest", action="store_true",
                        help="Show only the latest workflow per sequence")
    parser.add_argument("--all-pipelines", action="store_true",
                        help="Include all pipeline types in summary/list")
    parser.add_argument("--test", action="store_true",
                        help="Use pipelines_test table")
    parser.add_argument("--refresh-workers", type=int, default=DEFAULT_REFRESH_WORKERS,
                        help="Concurrent OSMO queries for workflow/export refresh "
                             f"(default: {DEFAULT_REFRESH_WORKERS}; env: "
                             "MV_HOI_REFRESH_WORKERS)")
    args = parser.parse_args()

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    global TABLE
    if args.test:
        TABLE = PIPELINES_TEST_TABLE

    init_db(DB_PATH)

    pipeline_type = None if args.all_pipelines else args.pipeline

    refresh_workflow_states(args.dataset, pipeline_type=pipeline_type, db_path=DB_PATH,
                            table=TABLE, max_workers=args.refresh_workers)

    if args.sequence:
        show_sequence(args.dataset, args.sequence, args.pipeline)
    elif args.summary:
        show_summary(args.dataset, pipeline_type=pipeline_type, latest_only=args.latest)
    else:
        show_list(args.dataset, pipeline_type=pipeline_type, latest_only=args.latest)


if __name__ == "__main__":
    main()
