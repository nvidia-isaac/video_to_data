# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
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
import json
import os
import subprocess
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from db import (
    get_blacklisted_sequence,
    get_latest_workflow,
    get_recent_workflows_for_sequence,
    get_summary,
    get_workflows_by_dataset,
    init_db,
    update_workflow,
    upsert_blacklisted_sequence,
)

DB_PATH = os.path.join(SCRIPT_DIR, "processing.db")
TABLE = "workflows"


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
    table: str = "workflows",
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


def refresh_waiting(
    dataset: str,
    pipeline_type: str | None = None,
    db_path: str = DB_PATH,
    table: str = "workflows",
) -> None:
    """Poll OSMO for WAITING_WF rows and advance or fail them."""
    workflows = get_workflows_by_dataset(
        dataset, pipeline_type=pipeline_type, status="WAITING_WF",
        db_path=db_path, table=table,
    )
    for wf in workflows:
        osmo_id = wf.get("osmo_workflow_id") or wf["workflow_name"]
        info = osmo_query(osmo_id)
        wf_status = info["status"]
        if wf_status == "COMPLETED":
            update_workflow(wf["workflow_name"], status="WAITING_QC",
                           details="workflow_completed", db_path=db_path,
                           table=table)
        elif wf_status.startswith("FAILED"):
            detail = _failure_detail(info)
            update_workflow(wf["workflow_name"], status="FAIL",
                           details=detail, db_path=db_path, table=table)
            _maybe_auto_blacklist_repeated_failure(wf, db_path=db_path, table=table)


def show_sequence(dataset: str, sequence: str, pipeline_type: str) -> None:
    wf = get_latest_workflow(sequence, dataset, pipeline_type, db_path=DB_PATH,
                             table=TABLE)
    if not wf:
        print(f"No workflows found for {sequence}")
        return

    print(f"Sequence:  {wf['sequence_name']}")
    print(f"Dataset:   {wf['dataset']}")
    print(f"Pipeline:  {wf['pipeline_type']}")
    print(f"Version:   {wf['pipeline_version']}")
    print(f"Workflow:  {wf['workflow_name']}")
    print(f"OSMO ID:   {wf.get('osmo_workflow_id', '')}")
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
    print(f"Total workflows: {total}")
    for status in ("WAITING_WF", "WAITING_QC", "PASS", "FAIL"):
        count = summary["counts"].get(status, 0)
        print(f"  {status}: {count}")
    for status, count in sorted(summary["counts"].items()):
        if status not in ("WAITING_WF", "WAITING_QC", "PASS", "FAIL"):
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
        print(f"No workflows found for {dataset}")
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
                        help="Use workflows_test table")
    args = parser.parse_args()

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    global TABLE
    if args.test:
        TABLE = "workflows_test"

    init_db(DB_PATH)

    pipeline_type = None if args.all_pipelines else args.pipeline

    refresh_waiting(args.dataset, pipeline_type=pipeline_type, db_path=DB_PATH,
                    table=TABLE)

    if args.sequence:
        show_sequence(args.dataset, args.sequence, args.pipeline)
    elif args.summary:
        show_summary(args.dataset, pipeline_type=pipeline_type, latest_only=args.latest)
    else:
        show_list(args.dataset, pipeline_type=pipeline_type, latest_only=args.latest)


if __name__ == "__main__":
    main()
