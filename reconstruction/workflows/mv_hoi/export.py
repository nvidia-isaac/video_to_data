"""Submit batched MV HOI export workflows after human QC.

Auto mode scans the local workflow DB for latest mv_hoi_reconstruction rows in
WAITING_QC, takes the oldest sequence-name batch, checks the Kratos/Databricks
table for completed human QC, applies failure thresholds, and submits one OSMO
workflow containing up to the configured export batch size.
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
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from db import (
    PIPELINES_TABLE,
    PIPELINES_TEST_TABLE,
    get_latest_workflow,
    get_workflows_by_dataset,
    init_db,
    update_workflow,
)
from config_utils import (
    EXPORT_WORKFLOW,
    RECON_PIPELINE,
    apply_test_mode,
    get_pipeline_export_path,
    get_pipeline_output_path,
    get_workflow_cfg,
    load_config as _load_config,
)
from query import (
    DEFAULT_REFRESH_WORKERS,
    export_task_names,
    refresh_workflow_states,
    waiting_export_rows,
)


DB_PATH = str(SCRIPT_DIR / "processing.db")
TABLE = PIPELINES_TABLE

MAX_FAILURE_ANNOTATIONS = 5
MAX_FAILURE_COVERAGE = 0.30
DEFAULT_BATCH_SIZE = 30
GENERATED_DIR = SCRIPT_DIR / "osmo" / "generated"
EXPORT_IMAGE = "nvcr.io/nvstaging/isaac-amr/mv_hoi_mv_postprocess:{{image_tag}}"

_VALID_TABLE_RE = re.compile(r"^[A-Za-z0-9_.]+$")
_AMBIGUOUS_SUBMIT_MARKERS = (
    "read timed out",
    "cannot connect to osmo service",
    "httpsconnectionpool",
    "connectionerror",
    "connection aborted",
    "max retries exceeded",
    "timed out",
)


@dataclass(frozen=True)
class PreparedExport:
    workflow: dict
    failure_segments: list[dict]
    source_url: str
    export_url: str
    task_suffix: str


def load_config() -> dict:
    return _load_config(SCRIPT_DIR)


def _apply_test_mode(dataset_cfg: dict) -> None:
    apply_test_mode(dataset_cfg)


def _parse_swift_url(url: str) -> tuple[str, str, str]:
    stripped = url.rstrip("/").replace("swift://", "")
    parts = stripped.split("/", 3)
    endpoint = f"https://{parts[0]}"
    bucket = parts[2] if len(parts) > 2 else ""
    prefix = parts[3] if len(parts) > 3 else ""
    return endpoint, bucket, prefix


def get_s3_client(swift_url: str):
    import boto3

    endpoint, bucket, prefix = _parse_swift_url(swift_url)
    access_key = os.environ.get("CSS_ACCESS_KEY", "")
    secret_key = os.environ.get("CSS_SECRET_KEY", "")
    if not access_key or not secret_key:
        print(
            "Error: Set CSS_ACCESS_KEY and CSS_SECRET_KEY environment variables.\n"
            "  source reconstruction/scripts/setup_css_env.sh",
            file=sys.stderr,
        )
        sys.exit(1)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return client, bucket, prefix


def _s3_text(client, bucket: str, key: str) -> str | None:
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    body = resp["Body"].read()
    if isinstance(body, bytes):
        return body.decode("utf-8")
    return str(body)


def frame_count_from_metadata_text(text: str | None) -> int | None:
    if not text:
        return None
    try:
        meta = yaml.safe_load(text) or {}
        frame_count = int(meta["frame_count"])
    except (KeyError, TypeError, ValueError, yaml.YAMLError):
        return None
    return frame_count if frame_count > 0 else None


def frame_count_from_edex_text(text: str | None) -> int | None:
    if not text:
        return None
    try:
        edex = json.loads(text)
        header = edex[0] if isinstance(edex, list) and edex else edex
        frame_start = int(header["frame_start"])
        frame_end = int(header["frame_end"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    frame_count = frame_end - frame_start
    return frame_count if frame_count > 0 else None


def frame_count_from_sources(
    root_metadata: str | None,
    preprocess_metadata: str | None,
    preprocess_edex: str | None,
) -> int | None:
    return (
        frame_count_from_metadata_text(root_metadata)
        or frame_count_from_metadata_text(preprocess_metadata)
        or frame_count_from_edex_text(preprocess_edex)
    )


def resolve_frame_count(
    client,
    bucket: str,
    base_prefix: str,
    dataset_cfg: dict,
    sequence_name: str,
) -> int | None:
    seq_prefix = (
        f"{base_prefix}/{get_pipeline_output_path(dataset_cfg, RECON_PIPELINE)}"
        f"/{sequence_name}"
    ).strip("/")
    return frame_count_from_sources(
        _s3_text(client, bucket, f"{seq_prefix}/hoi_metadata.yaml"),
        _s3_text(client, bucket, f"{seq_prefix}/mv_preprocess/hoi_metadata.yaml"),
        _s3_text(client, bucket, f"{seq_prefix}/mv_preprocess/edex"),
    )


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a frame number")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    raise ValueError(f"not an integer: {value!r}")


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return False
        if text[0] in "[{":
            try:
                return _nonempty(json.loads(text))
            except json.JSONDecodeError:
                pass
        return True
    if isinstance(value, dict):
        return any(_nonempty(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_nonempty(item) for item in value)
    return bool(value)


def normalize_failure_annotations(
    annotations: list[dict],
    frame_count: int,
) -> tuple[list[dict], str | None]:
    failure_segments: list[dict] = []
    for ann in annotations:
        if not _nonempty(ann.get("failure_category")):
            continue
        try:
            start_frame = _coerce_int(ann["start_frame"]) - 1
            end_frame = _coerce_int(ann["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            return [], f"invalid_failure_annotation: {exc}"

        if start_frame < 0 or end_frame <= start_frame or end_frame > frame_count:
            return [], (
                "invalid_failure_annotation: "
                f"start_frame={start_frame}, end_frame={end_frame}, "
                f"frame_count={frame_count}"
            )

        failure_segments.append(
            {
                "id": ann.get("id"),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "failure_category": ann.get("failure_category"),
                "reason": ann.get("reason"),
            }
        )
    return failure_segments, None


def merged_interval_coverage(segments: list[dict]) -> int:
    intervals = sorted(
        (int(seg["start_frame"]), int(seg["end_frame"])) for seg in segments
    )
    if not intervals:
        return 0

    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def export_qc_thresholds(dataset_cfg: dict) -> tuple[int, float]:
    export_cfg = get_workflow_cfg(dataset_cfg, RECON_PIPELINE, EXPORT_WORKFLOW)
    max_failure_annotations = int(
        export_cfg.get("max_failure_annotations", MAX_FAILURE_ANNOTATIONS)
    )
    max_failure_coverage = float(
        export_cfg.get("max_failure_coverage", MAX_FAILURE_COVERAGE)
    )
    return max_failure_annotations, max_failure_coverage


def qc_failure_reason(
    failure_segments: list[dict],
    frame_count: int,
    *,
    max_failure_annotations: int = MAX_FAILURE_ANNOTATIONS,
    max_failure_coverage: float = MAX_FAILURE_COVERAGE,
) -> str | None:
    if len(failure_segments) > max_failure_annotations:
        return (
            f"qc_fail: failure_annotations>{max_failure_annotations} "
            f"({len(failure_segments)})"
        )

    covered = merged_interval_coverage(failure_segments)
    if covered / frame_count > max_failure_coverage:
        pct = covered / frame_count
        return (
            f"qc_fail: failure_coverage>{max_failure_coverage:.0%} "
            f"({covered}/{frame_count}={pct:.1%})"
        )
    return None


def _row_get(row: Any, name: str, index: int) -> Any:
    if hasattr(row, name):
        return getattr(row, name)
    if isinstance(row, dict):
        return row[name]
    return row[index]


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class KratosAnnotations(dict):
    """Completed annotation rows plus per-item Kratos status diagnostics."""

    def __init__(
        self,
        *args,
        statuses: dict[str, str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.statuses = statuses or {}


def query_completed_kratos_annotations(
    kratos_table: str,
    item_names: list[str],
) -> dict[str, list[dict]]:
    if not item_names:
        return KratosAnnotations()
    if not _VALID_TABLE_RE.match(kratos_table):
        raise ValueError(f"Invalid Databricks table name: {kratos_table!r}")

    try:
        from databricks import sql
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "databricks-sql-connector is required. Install it and source "
            "reconstruction/scripts/setup_databricks_env.sh."
        ) from exc

    host = os.environ.get("DATABRICKS_SERVER_HOSTNAME")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not host or not http_path or not token:
        raise RuntimeError(
            "Set DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, and "
            "DATABRICKS_TOKEN before running export.py."
        )

    names = ", ".join(_sql_literal(name) for name in item_names)
    query = f"""
        SELECT item_name, item_status, `table` AS table_json
        FROM {kratos_table}
        WHERE item_name IN ({names})
    """

    result = KratosAnnotations()
    with sql.connect(
        server_hostname=host,
        http_path=http_path,
        access_token=token,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            for row in cursor.fetchall():
                item_name = _row_get(row, "item_name", 0)
                item_status = str(_row_get(row, "item_status", 1) or "")
                result.statuses.setdefault(item_name, item_status)
                if not item_status.startswith("Completed"):
                    continue
                if item_name in result:
                    continue
                result.statuses[item_name] = item_status
                table_json = _row_get(row, "table_json", 2)
                table = json.loads(table_json)
                result[item_name] = table.get("rows") or []
    return result


def _latest_rows_by_sequence(workflows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    latest: list[dict] = []
    for workflow in workflows:
        sequence = workflow["sequence_name"]
        if sequence in seen:
            continue
        seen.add(sequence)
        latest.append(workflow)
    return latest


def is_recheckable_qc_fail(workflow: dict) -> bool:
    return (
        workflow.get("status") == "FAIL"
        and (workflow.get("details") or "").startswith("qc_fail:")
    )


def waiting_qc_candidates(
    dataset: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    include_qc_fail: bool = False,
) -> list[dict]:
    workflows = get_workflows_by_dataset(
        dataset,
        pipeline_type=RECON_PIPELINE,
        db_path=db_path,
        table=table,
    )
    candidates = [
        workflow
        for workflow in _latest_rows_by_sequence(workflows)
        if workflow["status"] == "WAITING_QC"
        or (include_qc_fail and is_recheckable_qc_fail(workflow))
    ]
    return sorted(
        candidates,
        key=lambda workflow: (
            workflow["sequence_name"],
            workflow.get("created_at") or "",
            workflow.get("id") or 0,
        ),
    )


def waiting_qc_candidate_for_sequence(
    dataset: str,
    sequence: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    include_qc_fail: bool = False,
) -> list[dict]:
    workflow = get_latest_workflow(
        sequence,
        dataset,
        RECON_PIPELINE,
        db_path=db_path,
        table=table,
    )
    if not workflow:
        print(f"No reconstruction pipeline row found for sequence: {sequence}")
        return []
    if workflow["status"] == "WAITING_QC":
        return [workflow]
    if include_qc_fail and is_recheckable_qc_fail(workflow):
        return [workflow]

    expected = (
        "WAITING_QC or FAIL with qc_fail details"
        if include_qc_fail
        else "WAITING_QC"
    )
    print(
        f"Latest reconstruction row for {sequence} is {workflow['status']}; "
        f"expected {expected}."
    )
    return []


def _normalize_time_arg(s: str) -> str:
    """Normalize a user time arg to `YYYY-MM-DD_HH-MM-SS`.

    This mirrors submit.py: bare dates expand to midnight, and are used with
    inclusive start / exclusive end filtering.
    """
    if len(s) == 10:
        return s + "_00-00-00"
    if len(s) == 19:
        return s
    raise ValueError(
        f"Time must be YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS: {s!r}"
    )


def _filter_workflows_by_time(
    workflows: list[dict],
    start_time: str | None,
    end_time: str | None,
) -> list[dict]:
    """Keep workflows whose sequence timestamp prefix is in [start, end)."""
    if not start_time and not end_time:
        return workflows
    lo = _normalize_time_arg(start_time) if start_time else None
    hi = _normalize_time_arg(end_time) if end_time else None
    kept: list[dict] = []
    for workflow in workflows:
        sequence = workflow["sequence_name"]
        prefix = sequence[:19]
        if len(prefix) < 19 or prefix[4] != "-" or prefix[10] != "_":
            continue
        if lo and prefix < lo:
            continue
        if hi and prefix >= hi:
            continue
        kept.append(workflow)
    return kept


def task_suffix(sequence_name: str) -> str:
    export_task, _ = export_task_names(sequence_name)
    return export_task.removeprefix("export_")


def generate_export_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"v2d_mv_hoi_export_{now.strftime('%Y%m%d_%H%M%S')}"


def osmo_export_workflow_id(export_name: str) -> str:
    return f"{export_name}-1"


def _render_tasks(items: list[PreparedExport]) -> str:
    parts: list[str] = []
    for item in items:
        export_task, copy_task = export_task_names(item.workflow["sequence_name"])
        parts.append(
            f"""  - name: {export_task}
    image: {EXPORT_IMAGE}
    resource: cpu_small
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    inputs:
    - url: {item.source_url}
    outputs:
    - url: {item.export_url}/
    files:
    - path: /tmp/entry.sh
      contents: |-
        set -ex
        python -m v2d.mv.postprocess.lib.export_sequence \\
          --source_dir {{{{input:0}}}} \\
          --output_dir {{{{output}}}}

  - name: {copy_task}
    image: {EXPORT_IMAGE}
    resource: cpu_small
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    outputs:
    - url: {item.export_url}/
    files:
    - localpath: failure_segments/{item.task_suffix}.json
      path: /tmp/failure_segments.json
    - path: /tmp/entry.sh
      contents: |-
        set -ex
        mkdir -p "{{{{output}}}}"
        cp /tmp/failure_segments.json "{{{{output}}}}/failure_segments.json"
"""
        )
    return "\n".join(parts)


def render_batch_workflow(template_text: str, items: list[PreparedExport]) -> str:
    return template_text.replace("__TASKS__", _render_tasks(items).rstrip())


def write_generated_workflow(
    export_name: str,
    template_path: Path,
    items: list[PreparedExport],
) -> Path:
    export_dir = GENERATED_DIR / export_name
    segments_dir = export_dir / "failure_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        segment_path = segments_dir / f"{item.task_suffix}.json"
        segment_path.write_text(json.dumps(item.failure_segments, indent=2) + "\n")

    workflow_path = export_dir / "workflow.yaml"
    workflow_path.write_text(render_batch_workflow(template_path.read_text(), items))
    return workflow_path


def _submit_error_text(error: subprocess.CalledProcessError) -> str:
    return "\n".join(part for part in (error.output, error.stderr) if part)


def _is_ambiguous_submit_error(error: subprocess.CalledProcessError) -> bool:
    if error.returncode == 10:
        return True
    text = _submit_error_text(error).lower()
    return any(marker in text for marker in _AMBIGUOUS_SUBMIT_MARKERS)


def osmo_submit(
    workflow_yaml: Path,
    pool: str,
    set_vars: dict[str, str],
    *,
    dry_run: bool = False,
) -> str:
    cmd = ["osmo", "workflow", "submit", str(workflow_yaml)]
    if set_vars:
        cmd.append("--set")
        cmd.extend(f"{key}={value}" for key, value in set_vars.items())
    cmd.extend(["--pool", pool])
    print(f"  CMD: {shlex.join(cmd)}")
    if dry_run:
        print("  [dry-run] skipping osmo submit")
        return set_vars.get("workflow_name", "dry-run")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error = subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
        if _is_ambiguous_submit_error(error):
            print("  WARNING: OSMO submit had an ambiguous transient failure")
        raise error

    stdout = result.stdout.strip()
    print(f"  OSMO: {stdout}")
    for line in stdout.splitlines():
        if line.strip().startswith("Workflow ID"):
            return line.split("-", 1)[1].strip()
    return set_vars.get("workflow_name", stdout)


def _swift_url(*parts: str) -> str:
    first, *rest = parts
    url = first.rstrip("/")
    suffix = "/".join(part.strip("/") for part in rest if part)
    return f"{url}/{suffix}" if suffix else url


def _mark_fail_if_changed(
    workflow: dict,
    details: str,
    db_path: str,
    table: str,
) -> None:
    if workflow.get("status") == "FAIL" and (workflow.get("details") or "") == details:
        return
    update_workflow(
        workflow["workflow_name"],
        status="FAIL",
        details=details,
        db_path=db_path,
        table=table,
    )


def _kratos_not_completed_detail(
    item_name: str,
    kratos_status_by_item: dict[str, str],
) -> str:
    status = kratos_status_by_item.get(item_name)
    if status is None:
        return "no Kratos row"
    return f"Kratos status={status!r}"


def _not_completed_kratos_message(
    workflow: dict,
    kratos_status_by_item: dict[str, str],
) -> str:
    sequence = workflow["sequence_name"]
    item_name = f"{workflow['workflow_name']}.json"
    return (
        f"  {sequence} ({item_name}): "
        f"{_kratos_not_completed_detail(item_name, kratos_status_by_item)}; "
        "staying WAITING_QC"
    )


def _print_not_completed_kratos_rows(
    workflows: list[dict],
    kratos_status_by_item: dict[str, str],
) -> None:
    for workflow in workflows:
        print(_not_completed_kratos_message(workflow, kratos_status_by_item))


def prepare_exports(
    candidates: list[dict],
    annotations_by_item: dict[str, list[dict]],
    dataset_cfg: dict,
    frame_count_lookup,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    dry_run: bool = False,
    limit: int | None = None,
    kratos_status_by_item: dict[str, str] | None = None,
) -> list[PreparedExport]:
    accepted: list[PreparedExport] = []
    swift_base = dataset_cfg["swift_base"]
    max_failure_annotations, max_failure_coverage = export_qc_thresholds(dataset_cfg)
    kratos_status_by_item = kratos_status_by_item or {}
    reject_counts = {
        "kratos_not_completed": 0,
        "missing_frame_count": 0,
        "invalid_annotation": 0,
        "qc_fail": 0,
    }
    checked = 0

    for workflow in candidates:
        if limit is not None and len(accepted) >= limit:
            break
        checked += 1
        sequence = workflow["sequence_name"]
        item_name = f"{workflow['workflow_name']}.json"
        annotations = annotations_by_item.get(item_name)
        if annotations is None:
            reject_counts["kratos_not_completed"] += 1
            print(_not_completed_kratos_message(workflow, kratos_status_by_item))
            continue

        frame_count = frame_count_lookup(sequence)
        if frame_count is None:
            reject_counts["missing_frame_count"] += 1
            print(f"  {sequence}: no frame_count found; staying WAITING_QC")
            continue

        failure_segments, invalid_reason = normalize_failure_annotations(
            annotations,
            frame_count,
        )
        if invalid_reason:
            reject_counts["invalid_annotation"] += 1
            if not dry_run:
                _mark_fail_if_changed(workflow, invalid_reason, db_path, table)
            print(f"  {sequence}: {invalid_reason}")
            continue

        failure_reason = qc_failure_reason(
            failure_segments,
            frame_count,
            max_failure_annotations=max_failure_annotations,
            max_failure_coverage=max_failure_coverage,
        )
        if failure_reason:
            reject_counts["qc_fail"] += 1
            if not dry_run:
                _mark_fail_if_changed(workflow, failure_reason, db_path, table)
            print(f"  {sequence}: {failure_reason}")
            continue

        suffix = task_suffix(sequence)
        accepted.append(
            PreparedExport(
                workflow=workflow,
                failure_segments=failure_segments,
                source_url=_swift_url(
                    swift_base,
                    get_pipeline_output_path(dataset_cfg, RECON_PIPELINE),
                    sequence,
                ),
                export_url=_swift_url(
                    swift_base,
                    get_pipeline_export_path(dataset_cfg, RECON_PIPELINE),
                    sequence,
                ),
                task_suffix=suffix,
            )
        )
    if candidates:
        rejected = sum(reject_counts.values())
        print(
            "Export QC summary: "
            f"accepted={len(accepted)}, rejected_or_waiting={rejected}, "
            f"checked={checked}, "
            f"kratos_not_completed={reject_counts['kratos_not_completed']}, "
            f"missing_frame_count={reject_counts['missing_frame_count']}, "
            f"invalid_annotation={reject_counts['invalid_annotation']}, "
            f"qc_fail={reject_counts['qc_fail']}"
        )
    return accepted


def submit_batch(
    items: list[PreparedExport],
    dataset_cfg: dict,
    workflow_yaml: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    dry_run: bool = False,
) -> str | None:
    if not items:
        return None

    export_name = generate_export_id()
    export_id = osmo_export_workflow_id(export_name)
    template_path = SCRIPT_DIR / workflow_yaml
    generated_yaml = write_generated_workflow(export_name, template_path, items)
    osmo_id = osmo_submit(
        generated_yaml,
        dataset_cfg["osmo_pool"],
        {"workflow_name": export_name},
        dry_run=dry_run,
    )

    if dry_run:
        print(f"  [dry-run] would stamp {len(items)} row(s) with {export_id}")
        return export_id

    for item in items:
        update_workflow(
            item.workflow["workflow_name"],
            status="WAITING_EXPORT",
            details=f"export_running: {export_id}",
            osmo_export_workflow_id=export_id,
            db_path=db_path,
            table=table,
        )
    print(f"Submitted export workflow {export_id} ({osmo_id}) for {len(items)} sequence(s)")
    return export_id


def print_export_sequences(items: list[PreparedExport]) -> None:
    print(f"Exporting {len(items)} sequence(s):")
    for item in items:
        print(f"  {item.workflow['sequence_name']}")


def run_export(
    dataset_name: str,
    dataset_cfg: dict,
    *,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    dry_run: bool = False,
    refresh_workers: int = DEFAULT_REFRESH_WORKERS,
    sequence: str | None = None,
    ignore_qc_fail: bool = False,
    start_time: str | None = None,
    end_time: str | None = None,
) -> None:
    refresh_workflow_states(
        dataset_name,
        pipeline_type=RECON_PIPELINE,
        db_path=db_path,
        table=table,
        max_workers=refresh_workers,
    )

    active = waiting_export_rows(dataset_name, db_path=db_path, table=table)
    if active:
        export_ids = sorted(
            {
                row.get("osmo_export_workflow_id") or "(missing export id)"
                for row in active
            }
        )
        print(
            "Active export workflow still running; not submitting a new batch: "
            + ", ".join(export_ids)
        )
        return

    if sequence:
        candidates = waiting_qc_candidate_for_sequence(
            dataset_name,
            sequence,
            db_path=db_path,
            table=table,
            include_qc_fail=ignore_qc_fail,
        )
    else:
        candidates = waiting_qc_candidates(
            dataset_name,
            db_path=db_path,
            table=table,
            include_qc_fail=ignore_qc_fail,
        )
        if start_time or end_time:
            candidates = _filter_workflows_by_time(candidates, start_time, end_time)
            bounds = f"[{start_time or '-inf'}, {end_time or '+inf'}]"
            print(f"Filtered to {len(candidates)} export candidate(s) in time range {bounds}")
    if not candidates:
        if not sequence:
            print("No WAITING_QC reconstruction rows ready for Kratos export checks.")
        return

    export_cfg = get_workflow_cfg(dataset_cfg, RECON_PIPELINE, EXPORT_WORKFLOW)
    batch_size = int(export_cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    kratos_table = export_cfg["kratos_table"]
    batch_candidates = candidates
    if not sequence:
        print(
            f"Checking {len(batch_candidates)} WAITING_QC candidate(s) "
            f"oldest-first to fill up to {batch_size} export(s)"
        )
    item_names = [f"{workflow['workflow_name']}.json" for workflow in batch_candidates]
    annotations_by_item = query_completed_kratos_annotations(kratos_table, item_names)
    kratos_status_by_item = getattr(annotations_by_item, "statuses", {})
    print(
        f"Found completed Kratos rows for {len(annotations_by_item)} "
        f"of {len(item_names)} candidate item(s)"
    )
    if not annotations_by_item:
        if sequence:
            print(
                f"No completed Kratos row found for {sequence} "
                f"({item_names[0]})."
            )
        else:
            print("No completed Kratos rows found for checked WAITING_QC candidates.")
        _print_not_completed_kratos_rows(batch_candidates, kratos_status_by_item)
        return

    s3, bucket, base_prefix = get_s3_client(dataset_cfg["swift_base"])

    def _frame_count_lookup(sequence: str) -> int | None:
        return resolve_frame_count(s3, bucket, base_prefix, dataset_cfg, sequence)

    prepared = prepare_exports(
        batch_candidates,
        annotations_by_item,
        dataset_cfg,
        _frame_count_lookup,
        db_path=db_path,
        table=table,
        dry_run=dry_run,
        limit=None if sequence else batch_size,
        kratos_status_by_item=kratos_status_by_item,
    )

    if not prepared:
        print("No sequences passed export QC checks for this run.")
        return

    print_export_sequences(prepared)

    workflow_yaml = export_cfg["workflow_yaml"]
    submit_batch(
        prepared,
        dataset_cfg,
        workflow_yaml,
        db_path=db_path,
        table=table,
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit batched MV HOI export workflows")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--sequence",
                        help="Export only this exact sequence if its latest row is WAITING_QC")
    parser.add_argument("--ignore-qc-fail", action="store_true",
                        help="Recheck latest FAIL rows whose details start with qc_fail:")
    parser.add_argument("--start_time",
                        help="Batch mode: only include sequences with timestamp >= this "
                             "(YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS, inclusive)")
    parser.add_argument("--end_time",
                        help="Batch mode: only include sequences with timestamp < this "
                             "(YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS, exclusive; "
                             "a bare date excludes that entire day)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate workflow files and print submit command without running OSMO")
    parser.add_argument("--test", action="store_true",
                        help="Use pipelines_test table and _test output paths")
    parser.add_argument("--refresh-workers", type=int, default=DEFAULT_REFRESH_WORKERS,
                        help="Concurrent OSMO queries for reconstruction WAITING_WF refresh")
    args = parser.parse_args()

    global TABLE
    if args.test:
        TABLE = PIPELINES_TEST_TABLE

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(config['datasets'].keys())}")
        sys.exit(1)

    dataset_cfg = config["datasets"][args.dataset]
    try:
        get_workflow_cfg(dataset_cfg, RECON_PIPELINE, EXPORT_WORKFLOW)
    except KeyError:
        print(f"Dataset {args.dataset} has no {EXPORT_WORKFLOW} workflow config")
        sys.exit(1)
    if args.test:
        _apply_test_mode(dataset_cfg)

    init_db(DB_PATH)
    run_export(
        args.dataset,
        dataset_cfg,
        db_path=DB_PATH,
        table=TABLE,
        dry_run=args.dry_run,
        refresh_workers=args.refresh_workers,
        sequence=args.sequence,
        ignore_qc_fail=args.ignore_qc_fail,
        start_time=args.start_time,
        end_time=args.end_time,
    )


if __name__ == "__main__":
    main()
