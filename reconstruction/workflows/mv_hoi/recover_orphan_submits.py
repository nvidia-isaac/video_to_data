"""Recover OSMO workflows accepted after a submit CLI timeout.

The recovery flow parses submit.log, confirms each candidate with
`osmo workflow list`, and optionally inserts missing rows into processing.db.
Dry-run is the default; pass --apply to write confirmed single-match rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from db import PIPELINES_TABLE, PIPELINES_TEST_TABLE, get_connection, get_workflow, init_db
from query import _failure_detail

DB_PATH = SCRIPT_DIR / "processing.db"
DEFAULT_LOG_PATH = SCRIPT_DIR / "logs" / "submit.log"
TABLE = PIPELINES_TABLE


@dataclass(frozen=True)
class SubmitCandidate:
    sequence_name: str
    pipeline_type: str
    pipeline_version: str
    workflow_name: str
    pool: str
    line_no: int


@dataclass(frozen=True)
class RemoteWorkflow:
    name: str
    status: str
    submit_time: str | None
    raw: dict


@dataclass(frozen=True)
class RecoveryAction:
    candidate: SubmitCandidate
    action: str
    remote: RemoteWorkflow | None = None
    local_status: str | None = None
    details: str | None = None
    message: str = ""


_SUBMIT_RE = re.compile(
    r"Submitting\s+(?P<pipeline>\S+)\s+for\s+"
    r"(?P<sequence>\S+)\s+\((?P<version>[^)]+)\)\.\.\."
)
_WORKFLOW_TS_RE = re.compile(r"_(?P<stamp>\d{8}_\d{6})$")


def load_config() -> dict:
    with open(SCRIPT_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def _parse_time_arg(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) == 10:
        text += "T00:00:00"
    elif "_" in text and "T" not in text:
        text = text.replace("_", "T")
    dt = datetime.fromisoformat(text)
    return dt.replace(tzinfo=None)


def _workflow_timestamp(workflow_name: str) -> datetime | None:
    match = _WORKFLOW_TS_RE.search(workflow_name)
    if not match:
        return None
    return datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M%S")


def _within_time_window(
    workflow_name: str,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    timestamp = _workflow_timestamp(workflow_name)
    if timestamp is None:
        return True
    if since is not None and timestamp < since:
        return False
    if until is not None and timestamp >= until:
        return False
    return True


def _cmd_set_vars_and_pool(cmd: str) -> tuple[dict[str, str], str | None]:
    tokens = shlex.split(cmd)
    set_vars: dict[str, str] = {}
    pool: str | None = None
    try:
        set_index = tokens.index("--set") + 1
    except ValueError:
        set_index = -1

    index = set_index
    while index > 0 and index < len(tokens):
        token = tokens[index]
        if token == "--pool":
            if index + 1 < len(tokens):
                pool = tokens[index + 1]
            break
        if token.startswith("--"):
            break
        key, sep, value = token.partition("=")
        if sep:
            set_vars[key] = value
        index += 1

    if pool is None and "--pool" in tokens:
        pool_index = tokens.index("--pool")
        if pool_index + 1 < len(tokens):
            pool = tokens[pool_index + 1]
    return set_vars, pool


def parse_submit_log(
    log_path: str | Path,
    *,
    pipeline_type: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[SubmitCandidate]:
    candidates: list[SubmitCandidate] = []
    pending: tuple[str, str, str, int] | None = None

    with open(log_path) as f:
        for line_no, line in enumerate(f, start=1):
            submit_match = _SUBMIT_RE.search(line)
            if submit_match:
                pending = (
                    submit_match.group("pipeline"),
                    submit_match.group("sequence"),
                    submit_match.group("version"),
                    line_no,
                )
                continue

            if pending is None or "CMD: osmo workflow submit " not in line:
                continue

            cmd = line.split("CMD:", 1)[1].strip()
            set_vars, pool = _cmd_set_vars_and_pool(cmd)
            workflow_name = set_vars.get("workflow_name")
            pending_pipeline, sequence_name, version, submit_line_no = pending
            pending = None

            if pending_pipeline != pipeline_type:
                continue
            if not workflow_name or not pool:
                continue
            if not _within_time_window(workflow_name, since, until):
                continue
            candidates.append(
                SubmitCandidate(
                    sequence_name=sequence_name,
                    pipeline_type=pending_pipeline,
                    pipeline_version=version,
                    workflow_name=workflow_name,
                    pool=pool,
                    line_no=submit_line_no,
                )
            )

    deduped: dict[str, SubmitCandidate] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.workflow_name, candidate)
    return list(deduped.values())


def _extract_workflows(data: object) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("workflows", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def osmo_list_by_name(
    workflow_name: str,
    pool: str,
    *,
    count: int = 10,
) -> list[RemoteWorkflow]:
    cmd = [
        "osmo",
        "workflow",
        "list",
        "--name",
        workflow_name,
        "--pool",
        pool,
        "--format-type",
        "json",
        "--count",
        str(count),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(message or f"osmo workflow list failed: {result.returncode}")

    data = json.loads(result.stdout)
    remotes = []
    for item in _extract_workflows(data):
        name = item.get("name")
        if not name:
            continue
        remotes.append(
            RemoteWorkflow(
                name=name,
                status=item.get("status", "UNKNOWN"),
                submit_time=item.get("submit_time"),
                raw=item,
            )
        )
    return remotes


def osmo_query_for_failure(workflow_name: str) -> dict:
    cmd = ["osmo", "workflow", "query", workflow_name, "--format-type", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"status": "FAILED", "tasks": {}}

    data = json.loads(result.stdout)
    tasks: dict[str, str] = {}
    for group in data.get("groups", []):
        for task in group.get("tasks", []):
            tasks[task["name"]] = task["status"]
    return {"status": data.get("status", "FAILED"), "tasks": tasks}


def _matches_candidate(candidate: SubmitCandidate, remote: RemoteWorkflow) -> bool:
    if remote.name == f"{candidate.workflow_name}-1":
        return True
    prefix = f"{candidate.workflow_name}-"
    suffix = remote.name.removeprefix(prefix)
    return remote.name.startswith(prefix) and suffix.isdigit()


def _sqlite_timestamp(value: str | None) -> str:
    if not value:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return value.replace("T", " ", 1)


def local_status_and_details(remote: RemoteWorkflow) -> tuple[str, str]:
    status = remote.status.upper()
    if status == "COMPLETED":
        return "WAITING_QC", "workflow_completed"
    if status.startswith("FAILED"):
        return "FAIL", _failure_detail(osmo_query_for_failure(remote.name))
    if status in {"CANCELED", "CANCELLED"}:
        return "FAIL", status.lower()
    return "WAITING_WF", "workflow_running"


def insert_recovered_workflow(
    candidate: SubmitCandidate,
    remote: RemoteWorkflow,
    *,
    dataset: str,
    status: str,
    details: str,
    db_path: str | Path = DB_PATH,
    table: str = TABLE,
) -> None:
    timestamp = _sqlite_timestamp(remote.submit_time)
    conn = get_connection(str(db_path))
    try:
        conn.execute(
            f"""INSERT INTO {table}
               (sequence_name, dataset, pipeline_type, pipeline_version,
                workflow_name, osmo_workflow_id, status, details,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                candidate.sequence_name,
                dataset,
                candidate.pipeline_type,
                candidate.pipeline_version,
                candidate.workflow_name,
                remote.name,
                status,
                details,
                timestamp,
                timestamp,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def plan_recovery(
    candidates: list[SubmitCandidate],
    *,
    dataset: str,
    db_path: str | Path = DB_PATH,
    table: str = TABLE,
    count: int = 10,
) -> list[RecoveryAction]:
    actions: list[RecoveryAction] = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"Checking {index}/{len(candidates)} {candidate.workflow_name}", flush=True)
        if get_workflow(candidate.workflow_name, db_path=str(db_path), table=table):
            actions.append(RecoveryAction(candidate, "ALREADY_TRACKED"))
            continue

        try:
            remotes = osmo_list_by_name(candidate.workflow_name, candidate.pool, count=count)
        except Exception as exc:
            actions.append(RecoveryAction(candidate, "QUERY_ERROR", message=str(exc)))
            continue

        matches = [remote for remote in remotes if _matches_candidate(candidate, remote)]
        if not matches:
            actions.append(RecoveryAction(candidate, "NOT_FOUND"))
            continue
        if len(matches) > 1:
            actions.append(
                RecoveryAction(
                    candidate,
                    "MULTIPLE_MATCHES",
                    message=", ".join(remote.name for remote in matches),
                )
            )
            continue

        remote = matches[0]
        status, details = local_status_and_details(remote)
        actions.append(
            RecoveryAction(
                candidate,
                "BACKFILL",
                remote=remote,
                local_status=status,
                details=details,
            )
        )
    return actions


def apply_recovery(
    actions: list[RecoveryAction],
    *,
    dataset: str,
    db_path: str | Path = DB_PATH,
    table: str = TABLE,
) -> int:
    inserted = 0
    for action in actions:
        if action.action != "BACKFILL" or action.remote is None:
            continue
        insert_recovered_workflow(
            action.candidate,
            action.remote,
            dataset=dataset,
            status=action.local_status or "WAITING_WF",
            details=action.details or "workflow_running",
            db_path=db_path,
            table=table,
        )
        inserted += 1
    return inserted


def print_summary(actions: list[RecoveryAction]) -> None:
    counts: dict[str, int] = {}
    for action in actions:
        counts[action.action] = counts.get(action.action, 0) + 1
        suffix = ""
        if action.remote:
            suffix = (
                f" -> {action.remote.name} remote={action.remote.status} "
                f"local={action.local_status} details={action.details}"
            )
        elif action.message:
            suffix = f" ({action.message})"
        print(
            f"{action.action}: {action.candidate.workflow_name} "
            f"{action.candidate.sequence_name}{suffix}"
        )

    print("\nSummary:")
    for action, count in sorted(counts.items()):
        print(f"  {action}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--since", help="Workflow-name timestamp lower bound, inclusive")
    parser.add_argument("--until", help="Workflow-name timestamp upper bound, exclusive")
    parser.add_argument("--db-path", default=str(DB_PATH))
    parser.add_argument(
        "--table",
        choices=[PIPELINES_TABLE, PIPELINES_TEST_TABLE],
        default=TABLE,
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="Write confirmed rows")
    args = parser.parse_args()

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    dataset_cfg = config["datasets"][args.dataset]
    if args.pipeline not in dataset_cfg.get("pipelines", {}):
        print(f"Unknown pipeline: {args.pipeline}", file=sys.stderr)
        sys.exit(1)

    init_db(args.db_path)
    candidates = parse_submit_log(
        _resolve_path(args.log_path),
        pipeline_type=args.pipeline,
        since=_parse_time_arg(args.since),
        until=_parse_time_arg(args.until),
    )
    print(f"Parsed {len(candidates)} submit candidate(s)", flush=True)
    actions = plan_recovery(
        candidates,
        dataset=args.dataset,
        db_path=args.db_path,
        table=args.table,
        count=args.count,
    )
    print_summary(actions)

    if args.apply:
        inserted = apply_recovery(
            actions,
            dataset=args.dataset,
            db_path=args.db_path,
            table=args.table,
        )
        print(f"\nInserted {inserted} recovered workflow row(s)")
    else:
        print("\nDry run only; pass --apply to insert confirmed rows.")


if __name__ == "__main__":
    main()
