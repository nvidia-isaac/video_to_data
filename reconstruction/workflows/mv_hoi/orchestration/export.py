"""Submit batched MV HOI export workflows after human QC.

Auto mode scans the local workflow DB for latest mv_hoi_reconstruction rows in
WAITING_QC, takes the oldest sequence-name batch, checks the Kratos DRS
table for completed human QC, applies failure thresholds, and submits one OSMO
workflow containing up to the configured export batch size.
"""

from __future__ import annotations

try:
    from .registry_versions import image_registry
    from .storage import parse_storage_url, s3_client_kwargs
except ImportError:  # Direct script execution.
    from registry_versions import image_registry
    from storage import parse_storage_url, s3_client_kwargs

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MV_HOI_DIR = SCRIPT_DIR.parent
try:
    from .db import (
        DB_PATH,
        attach_request_execution,
        EXPORT_STAGE,
        PIPELINES_TABLE,
        PIPELINES_TEST_TABLE,
        TEST_DB_PATH,
        apply_execution_observation,
        create_stage_run,
        create_stage_request,
        create_workflow_execution,
        get_blacklisted_sequence,
        get_campaign,
        get_latest_qc_review,
        get_qc_review,
        get_latest_workflow,
        get_stage_run,
        get_stage_run_by_request,
        get_stage_request,
        init_db,
        list_current_stage_runs,
        list_workflow_executions,
        list_stage_requests,
        maybe_blacklist_repeated_failure,
        reserve_stage_request,
        record_qc_review,
        update_stage_run,
        update_stage_request,
        update_workflow_execution,
    )
    from .config_utils import (
        EXPORT_CONFIG_PIPELINE,
        EXPORT_WORKFLOW,
        RECON_PIPELINE,
        apply_test_mode,
        get_pipeline_export_path,
        get_pipeline_input_path,
        get_workflow_cfg,
        load_config as _load_config,
    )
    from .query import (
        DEFAULT_REFRESH_WORKERS,
        export_task_names,
        refresh_workflow_states,
        waiting_export_rows,
    )
    from .pool_selection import (
        PoolSelector,
        load_active_counts,
        osmo_submission_priority,
    )
    from .runtime import generated_dir, require_submit_authority
except ImportError:  # Direct script execution.
    from db import (
        DB_PATH,
        attach_request_execution,
        EXPORT_STAGE,
        PIPELINES_TABLE,
        PIPELINES_TEST_TABLE,
        TEST_DB_PATH,
        apply_execution_observation,
        create_stage_run,
        create_stage_request,
        create_workflow_execution,
        get_blacklisted_sequence,
        get_campaign,
        get_latest_qc_review,
        get_qc_review,
        get_latest_workflow,
        get_stage_run,
        get_stage_run_by_request,
        get_stage_request,
        init_db,
        list_current_stage_runs,
        list_workflow_executions,
        list_stage_requests,
        maybe_blacklist_repeated_failure,
        reserve_stage_request,
        record_qc_review,
        update_stage_run,
        update_stage_request,
        update_workflow_execution,
    )
    from config_utils import (
        EXPORT_CONFIG_PIPELINE,
        EXPORT_WORKFLOW,
        RECON_PIPELINE,
        apply_test_mode,
        get_pipeline_export_path,
        get_pipeline_input_path,
        get_workflow_cfg,
        load_config as _load_config,
    )
    from query import (
        DEFAULT_REFRESH_WORKERS,
        export_task_names,
        refresh_workflow_states,
        waiting_export_rows,
    )
    from pool_selection import PoolSelector, load_active_counts, osmo_submission_priority
    from runtime import generated_dir, require_submit_authority


TABLE = PIPELINES_TABLE

MAX_FAILURE_ANNOTATIONS = 5
MAX_FAILURE_COVERAGE = 0.30
DEFAULT_BATCH_SIZE = 30
GENERATED_DIR = generated_dir()
EXPORT_IMAGE = "{{image_registry}}/mv_hoi_mv_postprocess:{{image_tag}}"
DEFAULT_KRATOS_STATUS_TABLE = None
DEFAULT_KRATOS_PROJECT_ID = None

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

_QC_QUERY_UNAVAILABLE_MARKERS = (
    "401",
    "403",
    "access token",
    "authentication",
    "connection aborted",
    "connection refused",
    "connection reset",
    "credential",
    "dns",
    "forbidden",
    "host is unreachable",
    "httpconnectionpool",
    "httpsconnectionpool",
    "max retries exceeded",
    "name resolution",
    "network is unreachable",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "unauthorized",
)
_QC_QUERY_UNAVAILABLE_EXCEPTION_NAMES = {
    "ConnectionError",
    "InterfaceError",
    "KratosAuthenticationError",
    "KratosConfigurationError",
    "MaxRetryDurationError",
    "NonRecoverableNetworkError",
    "OperationalError",
    "RequestError",
    "Timeout",
    "UnsafeToRetryError",
}
_QC_QUERY_DEFECT_MARKERS = (
    "catalog_not_found",
    "column_not_found",
    "insufficient_permissions",
    "parse_exception",
    "syntax_error",
    "table_or_view_not_found",
)


class QCQueryUnavailableError(RuntimeError):
    """The external human-QC query service cannot currently be reached."""


def _is_qc_query_unavailable_error(exc: BaseException) -> bool:
    """Separate transient/auth failures from query/schema implementation errors."""
    detail = " ".join(str(exc).lower().split())
    if any(marker in detail for marker in _QC_QUERY_DEFECT_MARKERS):
        return False
    if type(exc).__name__ in _QC_QUERY_UNAVAILABLE_EXCEPTION_NAMES:
        return True
    return any(marker in detail for marker in _QC_QUERY_UNAVAILABLE_MARKERS)


def _qc_query_error_detail(exc: BaseException) -> str:
    detail = " ".join(str(exc).split())
    if len(detail) > 500:
        detail = detail[:497] + "..."
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _qc_query_enabled() -> bool:
    return os.environ.get("MV_HOI_QC_QUERY_ENABLED", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _qc_query_poll_interval_seconds() -> float:
    try:
        interval = float(
            os.environ.get("MV_HOI_QC_QUERY_POLL_INTERVAL_SECONDS", "10")
        )
    except ValueError as exc:
        raise ValueError(
            "MV_HOI_QC_QUERY_POLL_INTERVAL_SECONDS must be numeric"
        ) from exc
    if interval <= 0:
        raise ValueError(
            "MV_HOI_QC_QUERY_POLL_INTERVAL_SECONDS must be positive"
        )
    return interval


def _qc_query_max_polls() -> int:
    try:
        max_polls = int(os.environ.get("MV_HOI_QC_QUERY_MAX_POLLS", "60"))
    except ValueError as exc:
        raise ValueError("MV_HOI_QC_QUERY_MAX_POLLS must be an integer") from exc
    if max_polls <= 0:
        raise ValueError("MV_HOI_QC_QUERY_MAX_POLLS must be positive")
    return max_polls


@dataclass(frozen=True)
class PreparedExport:
    workflow: dict
    failure_segments: list[dict]
    source_url: str
    export_url: str
    task_suffix: str
    qc_review_id: int | None = None
    authorization_type: str = "QC"
    override_reason: str | None = None
    requested_by: str | None = None
    publish_url: str | None = None
    preprocess_source_url: str | None = None
    interaction_trim_enabled: bool = True
    interaction_distance_threshold_m: float = 0.10
    interaction_pre_contact_padding_seconds: float = 3.0
    interaction_post_contact_padding_seconds: float = 3.0
    interaction_window_frames: int = 7
    interaction_required_under_threshold_frames: int = 5
    max_failure_annotations: int = 10
    max_failure_coverage: float = 0.5
    max_silhouette_failure_coverage: float = 0.5
    metrics_request_id: int | None = None
    metrics_campaign_name: str | None = None
    export_run_id: int | None = None
    checkpoint_manifest_url: str | None = None
    label_sha256: str | None = None
    configuration_sha256: str | None = None
    existing_request_id: int | None = None


def load_config() -> dict:
    return _load_config(MV_HOI_DIR)


def _apply_test_mode(dataset_cfg: dict) -> None:
    apply_test_mode(dataset_cfg)


def _parse_swift_url(url: str) -> tuple[str | None, str, str]:
    """Compatibility alias accepting S3, Swift and bare bucket paths."""
    return parse_storage_url(url)


def get_s3_client(swift_url: str):
    import boto3

    endpoint, bucket, prefix = _parse_swift_url(swift_url)
    client = boto3.client("s3", **s3_client_kwargs(endpoint))
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
    preprocess_metadata: str | None,
    preprocess_edex: str | None,
) -> int | None:
    return (
        frame_count_from_metadata_text(preprocess_metadata)
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
        f"{base_prefix}/{get_pipeline_input_path(dataset_cfg, EXPORT_CONFIG_PIPELINE)}"
        f"/{sequence_name}"
    ).strip("/")
    return frame_count_from_sources(
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


def export_qc_thresholds(dataset_cfg: dict) -> tuple[int, float, float]:
    export_cfg = get_workflow_cfg(
        dataset_cfg, EXPORT_CONFIG_PIPELINE, EXPORT_WORKFLOW
    )
    max_failure_annotations = int(
        export_cfg.get("max_failure_annotations", MAX_FAILURE_ANNOTATIONS)
    )
    max_failure_coverage = float(
        export_cfg.get("max_failure_coverage", MAX_FAILURE_COVERAGE)
    )
    max_silhouette_failure_coverage = float(
        export_cfg.get("max_silhouette_failure_coverage", 0.5)
    )
    return (
        max_failure_annotations,
        max_failure_coverage,
        max_silhouette_failure_coverage,
    )


def interaction_trim_settings(dataset_cfg: dict) -> dict[str, Any]:
    export_cfg = get_workflow_cfg(
        dataset_cfg, EXPORT_CONFIG_PIPELINE, EXPORT_WORKFLOW,
    )
    trim = export_cfg.get("interaction_trim", {})
    return {
        "interaction_trim_enabled": bool(trim.get("enabled", True)),
        "interaction_distance_threshold_m": float(
            trim.get("distance_threshold_m", 0.10)
        ),
        "interaction_pre_contact_padding_seconds": float(
            trim.get("pre_contact_padding_seconds", 3.0)
        ),
        "interaction_post_contact_padding_seconds": float(
            trim.get("post_contact_padding_seconds", 3.0)
        ),
        "interaction_window_frames": int(trim.get("window_frames", 7)),
        "interaction_required_under_threshold_frames": int(
            trim.get("required_under_threshold_frames", 5)
        ),
    }


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _kratos_drs_settings() -> tuple[str, str, str]:
    profile = os.environ.get("KRATOS_PROFILE", "production").strip() or "production"
    namespace = os.environ.get("KRATOS_NAMESPACE", "").strip()
    warehouse_id = os.environ.get("KRATOS_DRS_WAREHOUSE_ID", "").strip()
    profile_prefix = profile.upper().replace("-", "_")
    client_id_name = f"{profile_prefix}_KRATOS_CLI_SSA_CLIENT_ID"
    client_secret_name = f"{profile_prefix}_KRATOS_CLI_SSA_CLIENT_SECRET"
    missing = [
        name
        for name, value in (
            ("KRATOS_NAMESPACE", namespace),
            ("KRATOS_DRS_WAREHOUSE_ID", warehouse_id),
            (client_id_name, os.environ.get(client_id_name, "").strip()),
            (client_secret_name, os.environ.get(client_secret_name, "").strip()),
        )
        if not value
    ]
    if missing:
        raise QCQueryUnavailableError(
            "Set the Kratos DRS environment before running export.py; missing: "
            + ", ".join(missing)
        )
    return profile, namespace, warehouse_id


def _normalize_kratos_drs_response(response: Any) -> tuple[int, dict[str, Any]]:
    if isinstance(response, dict):
        envelope = response
    else:
        status_code = getattr(response, "status_code", None)
        payload = response.json()
        if isinstance(payload, dict) and "statusCode" in payload:
            envelope = payload
        else:
            envelope = {"statusCode": status_code, "result": payload}
    try:
        status_code = int(envelope["statusCode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Malformed Kratos DRS response status: {envelope!r}"
        ) from exc
    result = envelope.get("result")
    if result is None:
        result = {}
    if not isinstance(result, dict):
        raise RuntimeError(f"Malformed Kratos DRS result: {result!r}")
    return status_code, result


def _kratos_drs_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = result.get("data")
    if rows is None:
        return []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Malformed Kratos DRS data rows: {rows!r}")
    return rows


def _kratos_drs_total_pages(result: dict[str, Any]) -> int | None:
    metadata = result.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Malformed Kratos DRS metadata: {metadata!r}")
    if metadata.get("resultTruncated") is True:
        raise RuntimeError("Kratos DRS result was truncated")
    value = metadata.get("totalPages")
    if value is None:
        return None
    try:
        total_pages = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Malformed Kratos DRS totalPages: {value!r}"
        ) from exc
    if total_pages < 0:
        raise RuntimeError(f"Malformed Kratos DRS totalPages: {total_pages}")
    return total_pages


def _is_kratos_page_out_of_bounds(exc: BaseException) -> bool:
    return "requested page is out of bounds" in str(exc).lower()


def _execute_kratos_drs_json_query(query: str) -> list[dict[str, Any]]:
    try:
        from kratos.drs_jobs import execute_drs_adhoc_job, get_drs_job
    except ModuleNotFoundError as exc:
        raise QCQueryUnavailableError(
            "The optional kratos-cli integration is required for Kratos QC queries. "
            "Install the compatible client from your Kratos provider and configure "
            "its credentials; see workflows/mv_hoi/README.md. Exports remain waiting for QC."
        ) from exc

    profile, namespace, warehouse_id = _kratos_drs_settings()
    poll_interval = _qc_query_poll_interval_seconds()
    max_polls = _qc_query_max_polls()
    common_kwargs = {
        "profile": profile,
        "namespace": namespace,
        "auth_type": "service",
    }
    try:
        response = execute_drs_adhoc_job(
            warehouse_id=warehouse_id,
            query=query,
            file_format="json",
            **common_kwargs,
        )
        status_code, result = _normalize_kratos_drs_response(response)
        job_id = result.get("jobId")
        if status_code == 202:
            if not job_id:
                raise RuntimeError(
                    "Malformed Kratos DRS 202 response: missing jobId"
                )
            for _attempt in range(max_polls):
                time.sleep(poll_interval)
                response = get_drs_job(
                    job_id=str(job_id), page=1, page_size=100,
                    **common_kwargs,
                )
                status_code, result = _normalize_kratos_drs_response(response)
                if status_code == 200:
                    break
                if status_code != 202:
                    raise RuntimeError(
                        f"Unexpected Kratos DRS poll status {status_code}: {result!r}"
                    )
            else:
                raise QCQueryUnavailableError(
                    f"Kratos DRS job {job_id} did not complete after "
                    f"{max_polls} poll(s)"
                )
        elif status_code != 200:
            raise RuntimeError(
                f"Unexpected Kratos DRS submission status {status_code}: {result!r}"
            )

        rows = _kratos_drs_rows(result)
        total_pages = _kratos_drs_total_pages(result)
        job_id = result.get("jobId") or job_id
        if not job_id:
            return rows
        if total_pages is not None and total_pages <= 1:
            return rows
        if total_pages is None and len(rows) < 100:
            return rows
        page = 2
        while total_pages is None or page <= total_pages:
            try:
                response = get_drs_job(
                    job_id=str(job_id), page=page, page_size=100,
                    **common_kwargs,
                )
            except Exception as exc:
                if total_pages is None and _is_kratos_page_out_of_bounds(exc):
                    break
                raise
            page_status, page_result = _normalize_kratos_drs_response(response)
            if page_status != 200:
                raise RuntimeError(
                    f"Unexpected Kratos DRS page status {page_status}: "
                    f"{page_result!r}"
                )
            page_rows = _kratos_drs_rows(page_result)
            if not page_rows:
                break
            rows.extend(page_rows)
            page += 1
        return rows
    except QCQueryUnavailableError:
        raise
    except Exception as exc:
        if not _is_qc_query_unavailable_error(exc):
            raise
        raise QCQueryUnavailableError(
            "Kratos QC query is temporarily unavailable ("
            + _qc_query_error_detail(exc)
            + ")"
        ) from exc


def _metric_status_to(row: dict[str, Any]) -> str:
    value = row.get("status_to")
    if value:
        return str(value)
    raise RuntimeError("Could not find status_to in Kratos status metrics row.")


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
    kratos_status_table: str | None = DEFAULT_KRATOS_STATUS_TABLE,
    kratos_project_id: int | None = DEFAULT_KRATOS_PROJECT_ID,
) -> dict[str, list[dict]]:
    """Return annotation rows for items whose latest Kratos status is Completed."""
    if not item_names:
        return KratosAnnotations()
    missing = [name for name, value in (
        ("kratos_table", kratos_table),
        ("kratos_status_table", kratos_status_table),
        ("kratos_project_id", kratos_project_id),
    ) if not value]
    if missing:
        raise QCQueryUnavailableError(
            "Configure the optional Kratos integration before querying human QC; missing: "
            + ", ".join(missing) + ". Exports remain waiting for QC."
        )
    if isinstance(kratos_project_id, bool) or int(kratos_project_id) < 1:
        raise ValueError("kratos_project_id must be a positive integer")
    if not _VALID_TABLE_RE.match(kratos_table):
        raise ValueError(f"Invalid Kratos table name: {kratos_table!r}")
    if not _VALID_TABLE_RE.match(kratos_status_table):
        raise ValueError(f"Invalid Kratos table name: {kratos_status_table!r}")

    names = ", ".join(_sql_literal(name) for name in item_names)
    status_query = f"""
        SELECT *
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY item_name
                       ORDER BY event_datetime DESC
                   ) AS status_rank
            FROM {kratos_status_table}
            WHERE date_partition = (
                SELECT MAX(date_partition)
                FROM {kratos_status_table}
            )
            AND project_id = {int(kratos_project_id)}
            AND item_name IN ({names})
        ) latest_status
        WHERE status_rank = 1
        ORDER BY item_name
    """
    annotations_query = f"""
        SELECT item_name, `table` AS table_json
        FROM {kratos_table}
        WHERE item_name IN ({names})
    """

    result = KratosAnnotations()
    annotation_rows_by_item: dict[str, list[dict]] = {}
    for metric in _execute_kratos_drs_json_query(status_query):
        item_name = str(metric.get("item_name") or "")
        if not item_name:
            continue
        result.statuses[item_name] = _metric_status_to(metric)

    for row in _execute_kratos_drs_json_query(annotations_query):
        item_name = str(row.get("item_name") or "")
        if not item_name or item_name in annotation_rows_by_item:
            continue
        table_json = row.get("table_json")
        if table_json is None:
            continue
        table = json.loads(table_json) if isinstance(table_json, str) else table_json
        if table is None:
            continue
        if not isinstance(table, dict):
            raise RuntimeError(
                f"Malformed Kratos annotation table for {item_name}: "
                f"expected an object, got {type(table).__name__}"
            )
        annotation_rows = table.get("rows") or []
        if not isinstance(annotation_rows, list):
            raise RuntimeError(
                f"Malformed Kratos annotation rows for {item_name}: "
                f"expected a list, got {type(annotation_rows).__name__}"
            )
        annotation_rows_by_item[item_name] = annotation_rows

    for item_name in item_names:
        item_status = result.statuses.get(item_name, "")
        if not item_status.startswith("Completed"):
            continue
        if item_name not in annotation_rows_by_item:
            result.statuses[item_name] = (
                f"{item_status}; no annotation row in {kratos_table}"
            )
            continue
        result[item_name] = annotation_rows_by_item[item_name]
    return result


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
    campaign: str | None = None,
) -> list[dict]:
    workflows = list_current_stage_runs(
        dataset,
        stage=RECON_PIPELINE,
        db_path=db_path,
        table=table,
    )
    candidates = [
        workflow
        for workflow in workflows
        if (
            campaign is not None
            or not get_blacklisted_sequence(
                workflow["dataset"], workflow["sequence_name"], db_path=db_path,
            )
        )
        and (
            campaign is None
            or workflow.get("request_campaign_name") == campaign
            and workflow.get("request_campaign_type") in (
                "BACKLOG_REPROCESSING", "REMEDIATION",
            )
        )
        and (
            workflow["status"] == "WAITING_QC"
            or (include_qc_fail and is_recheckable_qc_fail(workflow))
        )
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
    include_pass: bool = False,
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
    if include_pass and workflow["status"] == "PASS":
        return [workflow]

    expected = (
        "WAITING_QC, QC FAIL, or PASS"
        if include_pass
        else ("WAITING_QC or FAIL with qc_fail details" if include_qc_fail else "WAITING_QC")
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
    return f"v2d_mv_hoi_export_{now.strftime('%Y%m%d_%H%M%S_%f')}"


def osmo_export_workflow_id(export_name: str) -> str:
    return f"{export_name}-1"


def _render_tasks(items: list[PreparedExport]) -> str:
    parts: list[str] = []
    for item in items:
        export_task, _ = export_task_names(item.workflow["sequence_name"])
        preprocess_input = ""
        preprocess_argument = ""
        disable_trim_argument = (
            ""
            if item.interaction_trim_enabled
            else " \\\n          --disable-interaction-trim"
        )
        if item.preprocess_source_url:
            preprocess_input = f"\n    - url: {item.preprocess_source_url}"
            preprocess_argument = (
                " \\\n          --preprocess-source-dir {{input:1}}"
            )
        checkpoint_input = ""
        checkpoint_argument = ""
        if item.checkpoint_manifest_url:
            checkpoint_index = 2 if item.preprocess_source_url else 1
            checkpoint_input = f"\n    - url: {item.checkpoint_manifest_url}"
            checkpoint_argument = (
                " \\" + "\n          --checkpoint-manifest-path "
                + f"{{{{input:{checkpoint_index}}}}}/manifest.json"
            )
        metrics_request_id = (
            item.metrics_request_id or item.workflow["stage_run_id"]
        )
        campaign_argument = ""
        if item.metrics_campaign_name:
            campaign_argument = (
                " \\" + "\n          --campaign-name "
                + f'"{item.metrics_campaign_name}"'
            )
        qc_argument = ""
        if item.qc_review_id is not None:
            qc_argument = " \\" + f"\n          --qc-review-id {item.qc_review_id}"
        export_run_argument = ""
        if item.export_run_id is not None:
            export_run_argument = (
                " \\" + f"\n          --export-run-id {item.export_run_id}"
            )
        provenance_arguments = ""
        if item.label_sha256:
            provenance_arguments += (
                " \\" + f"\n          --label-sha256 {item.label_sha256}"
            )
        if item.configuration_sha256:
            provenance_arguments += (
                " \\" + "\n          --configuration-sha256 "
                + item.configuration_sha256
            )
        parts.append(
            f"""  - name: {export_task}
    image: "{EXPORT_IMAGE}"
    resource: cpu_export
    command: [/bin/bash]
    args: [/tmp/entry.sh]
    inputs:
    - url: {item.source_url}{preprocess_input}{checkpoint_input}
    outputs:
    - url: {item.export_url}/
    files:
    - localpath: failure_segments/{item.task_suffix}.json
      path: /tmp/failure_segments.json
    - path: /tmp/entry.sh
      contents: |-
        set -ex
        python -m v2d.mv.postprocess.lib.finalize_export \\
          --source-dir {{{{input:0}}}} \\
          --failure-segments-path /tmp/failure_segments.json \\
          --output-dir {{{{output}}}} \\
          --sequence-name "{item.workflow['sequence_name']}" \\
          --pipeline-version "{item.workflow['pipeline_version']}" \\
          --authorization-type "{item.authorization_type}" \\
          --reconstruction-run-id {item.workflow['stage_run_id']} \\
          --request-id {metrics_request_id}{campaign_argument}{qc_argument}{export_run_argument}{checkpoint_argument}{provenance_arguments} \\
          --interaction-distance-threshold-m {item.interaction_distance_threshold_m} \\
          --interaction-pre-contact-padding-seconds {item.interaction_pre_contact_padding_seconds} \\
          --interaction-post-contact-padding-seconds {item.interaction_post_contact_padding_seconds} \\
          --interaction-window-frames {item.interaction_window_frames} \\
          --interaction-required-under-threshold-frames {item.interaction_required_under_threshold_frames} \\
          --max-failure-annotations {item.max_failure_annotations} \\
          --max-failure-coverage {item.max_failure_coverage} \\
          --max-silhouette-failure-coverage {item.max_silhouette_failure_coverage} \\
          --rebuild-incomplete{preprocess_argument}{disable_trim_argument}
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
    priority = osmo_submission_priority(pool)
    if priority:
        cmd.extend(["--priority", priority])
    print(f"  CMD: {shlex.join(cmd)}")
    if dry_run:
        print("  [dry-run] skipping osmo submit")
        return set_vars.get("workflow_name", "dry-run")

    require_submit_authority("submit an export workflow")

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


def _processing_source_urls(
    workflow: dict, dataset_cfg: dict, *, db_path: str,
) -> tuple[str, str | None]:
    """Resolve exact reconstruction and preprocessing lineage prefixes."""
    fallback = _swift_url(
        dataset_cfg["swift_base"],
        get_pipeline_input_path(dataset_cfg, EXPORT_CONFIG_PIPELINE),
        workflow["sequence_name"],
    )
    source = str(workflow.get("output_uri") or fallback).rstrip("/")
    preprocess_source = None
    preprocess_run_id = workflow.get("preprocess_run_id")
    if preprocess_run_id is not None:
        preprocess = get_stage_run(
            int(preprocess_run_id), stage="mv_preprocess", db_path=db_path,
        )
        if preprocess and preprocess.get("output_uri"):
            candidate = str(preprocess["output_uri"]).rstrip("/")
            if candidate != source:
                preprocess_source = candidate
    return source, preprocess_source


def _mark_fail_if_changed(
    workflow: dict,
    details: str,
    db_path: str,
    table: str,
) -> None:
    if workflow.get("status") == "FAIL" and (workflow.get("details") or "") == details:
        return
    qc_status = "FAIL" if details.startswith("qc_fail:") else "INVALID"
    latest_review = get_latest_qc_review(workflow["stage_run_id"], db_path=db_path)
    if not latest_review or latest_review["status"] != qc_status or latest_review.get("details") != details:
        record_qc_review(
            workflow["stage_run_id"], qc_status, details=details,
            source="kratos", external_review_id=f"{workflow['workflow_name']}.json",
            db_path=db_path,
        )
    update_stage_run(
        workflow["stage_run_id"], details=details, db_path=db_path,
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
    (
        max_failure_annotations,
        max_failure_coverage,
        max_silhouette_failure_coverage,
    ) = export_qc_thresholds(dataset_cfg)
    trim_settings = interaction_trim_settings(dataset_cfg)
    kratos_status_by_item = kratos_status_by_item or {}
    reject_counts = {
        "kratos_not_completed": 0,
        "missing_frame_count": 0,
        "invalid_annotation": 0,
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

        suffix = task_suffix(sequence)
        qc_review_id = None
        if not dry_run and workflow.get("stage_run_id") is not None:
            qc_review_id = record_qc_review(
                workflow["stage_run_id"],
                "PASS",
                details="export_qc_thresholds_deferred_until_after_trim",
                source="kratos",
                external_review_id=item_name,
                external_status=kratos_status_by_item.get(item_name),
                failure_annotation_count=len(failure_segments),
                failure_coverage=merged_interval_coverage(failure_segments) / frame_count,
                failure_segments=failure_segments,
                thresholds={
                    "max_failure_annotations": max_failure_annotations,
                    "max_failure_coverage": max_failure_coverage,
                },
                raw_payload=annotations,
                db_path=db_path,
            )
        source_url, preprocess_source_url = _processing_source_urls(
            workflow, dataset_cfg, db_path=db_path,
        )
        campaign_row = (
            get_campaign(int(workflow["request_campaign_id"]), db_path=db_path)
            if workflow.get("request_campaign_id") is not None else None
        )
        export_workflow = dict(workflow)
        if campaign_row is not None:
            # Campaigns may be advanced in place to a newer release while their
            # already-successful reconstruction lineage remains immutable.  The
            # export is new work, so use the campaign's current release rather
            # than inheriting the historical reconstruction image version.
            export_workflow["pipeline_version"] = campaign_row["pipeline_version"]
        accepted.append(
            PreparedExport(
                workflow=export_workflow,
                failure_segments=failure_segments,
                source_url=source_url,
                export_url=(
                    _swift_url(campaign_row["output_uri"], sequence)
                    if campaign_row is not None
                    else _swift_url(
                        swift_base,
                        get_pipeline_export_path(dataset_cfg),
                        sequence,
                    )
                ),
                task_suffix=suffix,
                qc_review_id=qc_review_id,
                authorization_type="QC",
                preprocess_source_url=preprocess_source_url,
                max_failure_annotations=max_failure_annotations,
                max_failure_coverage=max_failure_coverage,
                max_silhouette_failure_coverage=(
                    max_silhouette_failure_coverage
                ),
                metrics_request_id=workflow.get("request_id"),
                metrics_campaign_name=workflow.get("request_campaign_name"),
                checkpoint_manifest_url=(
                    _swift_url(
                        dataset_cfg["weights_base_url"], "foundation_pose",
                        "nvidia_tensorrt", "deployable_v1.0",
                    )
                    if dataset_cfg.get("weights_base_url") else None
                ),
                label_sha256=workflow.get("labeled_bboxes_sha256"),
                configuration_sha256=(
                    campaign_row.get("configuration_sha256") if campaign_row else None
                ),
                **trim_settings,
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
            f"invalid_annotation={reject_counts['invalid_annotation']}"
        )
    accepted.sort(key=lambda item: (
        -int(item.workflow.get("request_queue_priority") or 0),
        item.workflow.get("request_created_at") or "",
        int(
            item.workflow.get("request_id")
            or item.workflow.get("stage_run_id")
            or 0
        ),
    ))
    return accepted


def submit_batch(
    items: list[PreparedExport],
    dataset_cfg: dict,
    workflow_yaml: str,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    dry_run: bool = False,
    trigger: str = "automatic",
    requested_by: str | None = None,
    pool: str | None = None,
    pool_selector: PoolSelector | None = None,
) -> str | None:
    if not items:
        return None

    registry = image_registry(dataset_cfg.get("image_registry"))
    export_name = generate_export_id()
    export_id = osmo_export_workflow_id(export_name)
    export_pipeline = dataset_cfg["pipelines"][EXPORT_CONFIG_PIPELINE]
    template_path = MV_HOI_DIR / workflow_yaml
    selector = pool_selector or PoolSelector.collect(
        dataset_cfg,
        active_counts=load_active_counts(
            dataset_cfg, lambda: list_workflow_executions(
                status=("SUBMITTING", "RUNNING", "UNKNOWN"), db_path=db_path,
            ),
        ),
    )
    pool_decision = selector.choose(
        EXPORT_CONFIG_PIPELINE,
        workflow_text=render_batch_workflow(template_path.read_text(), items),
        override=pool,
    )
    if dry_run:
        dry_run_items = [
            replace(
                item,
                publish_url=item.export_url,
                export_url=_swift_url(
                    dataset_cfg["swift_base"],
                    export_pipeline.get("work_output_path", "_export_work"),
                    item.workflow.get("request_campaign_name") or "mainline",
                    item.workflow["sequence_name"],
                    f"request_dry_run_{item.task_suffix}",
                ),
            )
            for item in items
        ]
        generated_yaml = write_generated_workflow(
            export_name, template_path, dry_run_items,
        )
        osmo_submit(
            generated_yaml, pool_decision.pool,
            {"workflow_name": export_name, "image_registry": registry,
             "image_tag": items[0].workflow["pipeline_version"]}, dry_run=True,
        )
        print(f"  [dry-run] would create {len(items)} export stage run(s) for {export_id}")
        return export_id

    generated_yaml = GENERATED_DIR / export_name / "workflow.yaml"
    execution_id = create_workflow_execution(
        workflow_name=export_name,
        pipeline_type=EXPORT_STAGE,
        pipeline_version=items[0].workflow.get("pipeline_version"),
        status="SUBMITTING",
        details=pool_decision.detail("submit_reserved"),
        workflow_spec_path=str(generated_yaml),
        pool=pool_decision.pool,
        db_path=db_path,
    )
    reserved_runs: list[dict] = []
    reserved_requests: list[dict] = []
    request_scoped_items: list[PreparedExport] = []
    for item in items:
        export_task, _ = export_task_names(item.workflow["sequence_name"])
        try:
            pipeline_version = item.workflow.get("pipeline_version")
            if not pipeline_version:
                raise ValueError(
                    f"{item.workflow['sequence_name']}: export requires a pipeline version"
                )
            if item.existing_request_id is not None:
                request = get_stage_request(
                    item.existing_request_id, db_path=db_path,
                )
                if request is None or request["status"] != "PENDING":
                    raise ValueError("Export retry request is no longer pending")
                if (
                    request["stage"] != "export"
                    or request["sequence_name"] != item.workflow["sequence_name"]
                    or request["pipeline_version"] != pipeline_version
                ):
                    raise ValueError("Export retry request lineage is inconsistent")
            else:
                request = create_stage_request(
                    sequence_name=item.workflow["sequence_name"],
                    dataset=item.workflow["dataset"],
                    stage="export",
                    pipeline_version=pipeline_version,
                    trigger=trigger,
                    requested_by=item.requested_by or requested_by,
                    reason=item.override_reason,
                    campaign=item.workflow.get("request_campaign_id"),
                    cohort=item.workflow.get("request_cohort"),
                    queue_priority=int(
                        item.workflow.get("request_queue_priority") or 0
                    ),
                    parameters={
                        "authorization_type": item.authorization_type,
                        "source_uri": item.source_url,
                        "output_uri": item.export_url,
                    },
                    source_manifest={
                        "reconstruction_run_id": item.workflow["stage_run_id"],
                    },
                    db_path=db_path,
                )
            reserved_requests.append(request)
            candidate_url = _swift_url(
                dataset_cfg["swift_base"],
                export_pipeline.get("work_output_path", "_export_work"),
                item.workflow.get("request_campaign_name") or "mainline",
                item.workflow["sequence_name"],
                f"request_{request['id']}",
            )
            request_parameters = {
                **json.loads(request.get("parameters_json") or "{}"),
                "authorization_type": item.authorization_type,
                "source_uri": item.source_url,
                "preprocess_source_uri": item.preprocess_source_url,
                "candidate_uri": candidate_url,
                "output_uri": item.export_url,
                "output_layout": "request_scoped_v1",
            }
            request = update_stage_request(
                request["id"], parameters=request_parameters, db_path=db_path,
            )
            if reserve_stage_request(
                request["id"],
                reserved_by=os.environ.get("HOSTNAME", os.environ.get("USER", "submit-host")),
                force_blacklist=trigger == "manual",
                db_path=db_path,
            ) is None:
                raise RuntimeError(
                    f"Could not reserve export request {request['id']}"
                )
            export_run = create_stage_run(
                sequence_name=item.workflow["sequence_name"],
                dataset=item.workflow["dataset"],
                stage=EXPORT_STAGE,
                pipeline_version=item.workflow.get("pipeline_version"),
                status="SUBMITTING",
                details="submit_reserved",
                trigger=trigger,
                execution_id=execution_id,
                workflow_task_name=export_task,
                auxiliary_task_name=None,
                reconstruction_run_id=item.workflow["stage_run_id"],
                qc_review_id=item.qc_review_id,
                authorization_type=item.authorization_type,
                override_reason=item.override_reason,
                requested_by=item.requested_by or requested_by,
                source_uri=item.source_url,
                output_uri=item.export_url,
                request_id=request["id"],
                set_current=False,
                environment="test" if table == PIPELINES_TEST_TABLE else "prod",
                db_path=db_path,
            )
            reserved_runs.append(export_run)
            attach_request_execution(
                request["id"], execution_id, status="SUBMITTED", db_path=db_path,
            )
            request_scoped_items.append(
                replace(
                    item,
                    publish_url=item.export_url,
                    export_url=candidate_url,
                    export_run_id=export_run["stage_run_id"],
                )
            )
        except Exception:
            for run in reserved_runs:
                update_stage_run(
                    run["stage_run_id"], status="CANCELED",
                    details="batch_reservation_failed", stage=EXPORT_STAGE,
                    db_path=db_path,
                )
            for request in reserved_requests:
                update_stage_request(
                    request["id"], status="CANCELED",
                    details="batch_reservation_failed", db_path=db_path,
                )
            update_workflow_execution(
                execution_id, status="CANCELED", details="batch_reservation_failed",
                db_path=db_path,
            )
            raise

    items = request_scoped_items
    generated_yaml = write_generated_workflow(export_name, template_path, items)
    try:
        osmo_id = osmo_submit(
            generated_yaml, pool_decision.pool,
            {
                "workflow_name": export_name,
                "image_tag": items[0].workflow["pipeline_version"],
                "image_registry": registry,
            }, dry_run=False,
        )
    except subprocess.CalledProcessError as exc:
        ambiguous = _is_ambiguous_submit_error(exc)
        osmo_id = export_id if ambiguous else None
        execution_status = "UNKNOWN" if ambiguous else "FAILED"
        submit_details = (
            f"submit_ambiguous: {_submit_error_text(exc).strip()}"
            if ambiguous
            else f"submit_failed: {_submit_error_text(exc).strip()}"
        )
        details = pool_decision.detail(submit_details)
        apply_execution_observation(
            execution_id, execution_status=execution_status, details=details,
            osmo_workflow_id=osmo_id,
            run_outcomes=[{
                "run_id": run["stage_run_id"], "stage": EXPORT_STAGE,
                "status": execution_status, "details": details,
                "set_current": ambiguous,
            } for run in reserved_runs], db_path=db_path,
        )
        for run in reserved_runs:
            if not ambiguous:
                maybe_blacklist_repeated_failure(
                    run["stage_run_id"], stage=EXPORT_STAGE, db_path=db_path,
                )
        if ambiguous:
            print(
                f"Ambiguous export submission retained as {export_id}; "
                "it will be refreshed instead of resubmitted"
            )
            return export_id
        raise

    apply_execution_observation(
        execution_id, execution_status="RUNNING",
        details=pool_decision.detail("export_running"),
        osmo_workflow_id=osmo_id,
        run_outcomes=[{
            "run_id": run["stage_run_id"], "stage": EXPORT_STAGE,
            "status": "RUNNING", "details": f"export_running: {osmo_id}",
            "set_current": True,
        } for run in reserved_runs], db_path=db_path,
    )
    print(f"Submitted export workflow {export_id} ({osmo_id}) for {len(items)} sequence(s)")
    return osmo_id


def print_export_sequences(items: list[PreparedExport]) -> None:
    print(f"Exporting {len(items)} sequence(s):")
    for item in items:
        print(f"  {item.workflow['sequence_name']}")


def prepare_manual_override_exports(
    sequence_names: list[str],
    dataset_name: str,
    dataset_cfg: dict,
    *,
    bypass_qc: bool,
    reason: str | None,
    force: bool,
    requested_by: str,
    db_path: str,
    table: str,
) -> list[PreparedExport]:
    """Preflight an all-or-nothing manual batch from current reconstructions."""
    if bypass_qc and not reason:
        raise ValueError("--reason is required with --bypass-qc")
    errors: list[str] = []
    prepared: list[PreparedExport] = []
    swift_base = dataset_cfg["swift_base"]
    (
        max_failure_annotations,
        max_failure_coverage,
        max_silhouette_failure_coverage,
    ) = export_qc_thresholds(dataset_cfg)
    trim_settings = interaction_trim_settings(dataset_cfg)
    for sequence in sequence_names:
        blacklist = get_blacklisted_sequence(dataset_name, sequence, db_path=db_path)
        if blacklist and not force:
            errors.append(f"{sequence}: blacklisted ({blacklist.get('reason') or 'no reason'})")
            continue
        reconstruction = get_latest_workflow(
            sequence, dataset_name, RECON_PIPELINE, db_path=db_path, table=table,
        )
        if not reconstruction:
            errors.append(f"{sequence}: no current reconstruction run")
            continue
        if reconstruction.get("run_status") != "SUCCEEDED":
            errors.append(
                f"{sequence}: current reconstruction artifacts are not successful "
                f"({reconstruction.get('run_status')})"
            )
            continue
        review = get_latest_qc_review(
            reconstruction["stage_run_id"], db_path=db_path,
        )
        if review and review["status"] == "PASS":
            authorization_type = "QC"
            qc_review_id = review["id"]
            override_reason = None
            try:
                failure_segments = json.loads(review.get("failure_segments_json") or "[]")
            except json.JSONDecodeError:
                failure_segments = []
        elif bypass_qc:
            authorization_type = "MANUAL_OVERRIDE"
            qc_review_id = None
            override_reason = reason
            failure_segments = []
        else:
            state = review["status"] if review else "missing"
            errors.append(f"{sequence}: passing QC required (current QC={state})")
            continue
        source_url, preprocess_source_url = _processing_source_urls(
            reconstruction, dataset_cfg, db_path=db_path,
        )
        export_workflow = dict(reconstruction)
        campaign_id = reconstruction.get("request_campaign_id")
        campaign = None
        if campaign_id is not None:
            campaign = get_campaign(int(campaign_id), db_path=db_path)
            if campaign is not None:
                export_workflow["pipeline_version"] = campaign["pipeline_version"]
        prepared.append(PreparedExport(
            workflow=export_workflow,
            failure_segments=failure_segments,
            source_url=source_url,
            export_url=(
                _swift_url(campaign["output_uri"], sequence)
                if campaign is not None
                else _swift_url(
                    swift_base,
                    get_pipeline_export_path(dataset_cfg),
                    sequence,
                )
            ),
            task_suffix=task_suffix(sequence),
            qc_review_id=qc_review_id,
            authorization_type=authorization_type,
            override_reason=override_reason,
            requested_by=requested_by,
            preprocess_source_url=preprocess_source_url,
            max_failure_annotations=max_failure_annotations,
            max_failure_coverage=max_failure_coverage,
            max_silhouette_failure_coverage=max_silhouette_failure_coverage,
            **trim_settings,
        ))
    if errors:
        raise ValueError("Manual export preflight failed:\n  " + "\n  ".join(errors))
    return prepared


def _pending_export_retry_is_ready(request: dict) -> bool:
    parameters = json.loads(request.get("parameters_json") or "{}")
    value = parameters.get("retry_not_before")
    if not value:
        return True
    not_before = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if not_before.tzinfo:
        not_before = not_before.astimezone(timezone.utc).replace(tzinfo=None)
    return datetime.utcnow() >= not_before


def prepare_pending_export_retries(
    campaign_name: str,
    dataset_cfg: dict,
    *,
    db_path: str,
) -> list[PreparedExport]:
    """Rebuild export inputs from immutable failed-attempt lineage."""

    campaign = get_campaign(campaign_name, db_path=db_path)
    if campaign is None:
        raise ValueError(f"Unknown campaign: {campaign_name}")
    (
        max_failure_annotations,
        max_failure_coverage,
        max_silhouette_failure_coverage,
    ) = export_qc_thresholds(dataset_cfg)
    trim_settings = interaction_trim_settings(dataset_cfg)
    prepared: list[PreparedExport] = []
    requests = list_stage_requests(
        campaign=campaign_name, stage="export", status="PENDING", db_path=db_path,
    )
    for request in requests:
        if not str(request.get("reason") or "").startswith(
            "infrastructure_retry_of_request_"
        ) or not _pending_export_retry_is_ready(request):
            continue
        parameters = json.loads(request.get("parameters_json") or "{}")
        source_manifest = json.loads(request.get("source_manifest_json") or "{}")
        reconstruction_run_id = int(source_manifest.get("reconstruction_run_id") or 0)
        reconstruction = get_stage_run(
            reconstruction_run_id, stage=RECON_PIPELINE, db_path=db_path,
        )
        if reconstruction is None or reconstruction.get("run_status") != "SUCCEEDED":
            raise ValueError(
                f"Export retry {request['id']} lost its successful reconstruction"
            )
        prior_request_id = int(parameters.get("prior_request_id") or 0)
        prior_run = get_stage_run_by_request(
            prior_request_id, stage=EXPORT_STAGE, db_path=db_path,
        )
        if prior_run is None or int(prior_run["reconstruction_run_id"]) != reconstruction_run_id:
            raise ValueError(f"Export retry {request['id']} has inconsistent prior lineage")
        qc_review_id = prior_run.get("qc_review_id")
        failure_segments: list[dict] = []
        if prior_run.get("authorization_type") == "QC":
            review = get_qc_review(int(qc_review_id or 0), db_path=db_path)
            if (
                review is None
                or review.get("status") != "PASS"
                or int(review["reconstruction_run_id"]) != reconstruction_run_id
            ):
                raise ValueError(f"Export retry {request['id']} lost its bound QC review")
            failure_segments = json.loads(review.get("failure_segments_json") or "[]")
        workflow = dict(reconstruction)
        workflow.update({
            "pipeline_version": request["pipeline_version"],
            "request_campaign_id": request["campaign_id"],
            "request_campaign_name": request["campaign_name"],
            "request_campaign_type": request["campaign_type"],
            "request_cohort": request.get("cohort"),
            "request_queue_priority": request.get("queue_priority") or 0,
        })
        prepared.append(PreparedExport(
            workflow=workflow,
            failure_segments=failure_segments,
            source_url=parameters["source_uri"],
            preprocess_source_url=parameters.get("preprocess_source_uri"),
            export_url=parameters["output_uri"],
            task_suffix=task_suffix(request["sequence_name"]),
            qc_review_id=qc_review_id,
            authorization_type=prior_run["authorization_type"],
            override_reason=prior_run.get("override_reason"),
            requested_by=request.get("requested_by"),
            max_failure_annotations=max_failure_annotations,
            max_failure_coverage=max_failure_coverage,
            max_silhouette_failure_coverage=max_silhouette_failure_coverage,
            metrics_request_id=reconstruction.get("request_id"),
            metrics_campaign_name=request["campaign_name"],
            checkpoint_manifest_url=(
                _swift_url(
                    dataset_cfg["weights_base_url"], "foundation_pose",
                    "nvidia_tensorrt", "deployable_v1.0",
                )
                if dataset_cfg.get("weights_base_url") else None
            ),
            label_sha256=reconstruction.get("labeled_bboxes_sha256"),
            configuration_sha256=campaign.get("configuration_sha256"),
            existing_request_id=int(request["id"]),
            **trim_settings,
        ))
    prepared.sort(key=lambda item: (
        -int(item.workflow.get("request_queue_priority") or 0),
        int(item.existing_request_id or 0),
    ))
    return prepared


def submit_prepared_batches(
    items: list[PreparedExport],
    dataset_cfg: dict,
    *,
    db_path: str,
    table: str,
    dry_run: bool,
    trigger: str,
    requested_by: str,
    pool: str | None = None,
    pool_selector: PoolSelector | None = None,
) -> None:
    export_cfg = get_workflow_cfg(
        dataset_cfg, EXPORT_CONFIG_PIPELINE, EXPORT_WORKFLOW
    )
    batch_size = int(export_cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    selector = pool_selector or PoolSelector.collect(
        dataset_cfg,
        active_counts=load_active_counts(
            dataset_cfg, lambda: list_workflow_executions(
                status=("SUBMITTING", "RUNNING", "UNKNOWN"), db_path=db_path,
            ),
        ),
    )
    for start in range(0, len(items), batch_size):
        submit_batch(
            items[start:start + batch_size],
            dataset_cfg,
            export_cfg["workflow_yaml"],
            db_path=db_path,
            table=table,
            dry_run=dry_run,
            trigger=trigger,
            requested_by=requested_by,
            pool=pool,
            pool_selector=selector,
        )


def run_export(
    dataset_name: str,
    dataset_cfg: dict,
    *,
    db_path: str = DB_PATH,
    table: str = PIPELINES_TABLE,
    dry_run: bool = False,
    refresh_workers: int = DEFAULT_REFRESH_WORKERS,
    sequence: str | None = None,
    sequences: list[str] | None = None,
    ignore_qc_fail: bool = False,
    rerun: bool = False,
    bypass_qc: bool = False,
    reason: str | None = None,
    force: bool = False,
    pool: str | None = None,
    requested_by: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    campaign: str | None = None,
    pool_selector: PoolSelector | None = None,
    refresh: bool = True,
) -> None:
    manual_sequences = list(dict.fromkeys(
        (sequences or []) + ([sequence] if sequence else [])
    ))
    requested_by = requested_by or os.environ.get("USER", "unknown")
    if refresh:
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

    if campaign and not manual_sequences:
        retry_items = prepare_pending_export_retries(
            campaign, dataset_cfg, db_path=db_path,
        )
        if retry_items:
            print(f"Dispatching {len(retry_items)} eligible export retry request(s)")
            print_export_sequences(retry_items)
            submit_prepared_batches(
                retry_items, dataset_cfg, db_path=db_path, table=table,
                dry_run=dry_run, trigger="automatic", requested_by=requested_by,
                pool=pool, pool_selector=pool_selector,
            )
            return

    locally_authorized: list[PreparedExport] = []
    manual_sequences_to_query = list(manual_sequences)
    if manual_sequences:
        if bypass_qc:
            try:
                locally_authorized = prepare_manual_override_exports(
                    manual_sequences, dataset_name, dataset_cfg,
                    bypass_qc=True, reason=reason, force=force,
                    requested_by=requested_by, db_path=db_path, table=table,
                )
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return
            manual_sequences_to_query = []
        else:
            manual_sequences_to_query = []
            for manual_sequence in manual_sequences:
                try:
                    locally_authorized.extend(prepare_manual_override_exports(
                        [manual_sequence], dataset_name, dataset_cfg,
                        bypass_qc=False, reason=None, force=force,
                        requested_by=requested_by, db_path=db_path, table=table,
                    ))
                except ValueError as exc:
                    if "passing QC required" in str(exc):
                        manual_sequences_to_query.append(manual_sequence)
                    else:
                        print(str(exc), file=sys.stderr)
                        return
        if locally_authorized:
            if manual_sequences_to_query:
                print(
                    f"Using stored passing QC for {len(locally_authorized)} sequence(s); "
                    f"checking {len(manual_sequences_to_query)} remaining sequence(s)"
                )
            else:
                print_export_sequences(locally_authorized)
                submit_prepared_batches(
                    locally_authorized, dataset_cfg, db_path=db_path, table=table,
                    dry_run=dry_run, trigger="manual", requested_by=requested_by,
                    pool=pool,
                    pool_selector=pool_selector,
                )
                return

    if manual_sequences:
        candidates = []
        for manual_sequence in manual_sequences_to_query:
            if get_blacklisted_sequence(dataset_name, manual_sequence, db_path=db_path) and not force:
                print(
                    f"Manual export preflight failed: {manual_sequence} is blacklisted; "
                    "use --force for a one-time bypass",
                    file=sys.stderr,
                )
                return
            rows = waiting_qc_candidate_for_sequence(
                dataset_name,
                manual_sequence,
                db_path=db_path,
                table=table,
                include_qc_fail=ignore_qc_fail,
                include_pass=False,
            )
            if not rows:
                print("Manual export preflight failed; nothing was submitted", file=sys.stderr)
                return
            candidates.extend(rows)
    else:
        candidates = waiting_qc_candidates(
            dataset_name,
            db_path=db_path,
            table=table,
            include_qc_fail=ignore_qc_fail,
            campaign=campaign,
        )
        if start_time or end_time:
            candidates = _filter_workflows_by_time(candidates, start_time, end_time)
            bounds = f"[{start_time or '-inf'}, {end_time or '+inf'}]"
            print(f"Filtered to {len(candidates)} export candidate(s) in time range {bounds}")

    if not candidates:
        if locally_authorized:
            print_export_sequences(locally_authorized)
            submit_prepared_batches(
                locally_authorized, dataset_cfg, db_path=db_path, table=table,
                dry_run=dry_run, trigger="manual", requested_by=requested_by,
                pool=pool,
                pool_selector=pool_selector,
            )
            return
        if not manual_sequences:
            print("No WAITING_QC reconstruction rows ready for Kratos export checks.")
        return

    export_cfg = get_workflow_cfg(
        dataset_cfg, EXPORT_CONFIG_PIPELINE, EXPORT_WORKFLOW
    )
    batch_size = int(export_cfg.get("batch_size", DEFAULT_BATCH_SIZE))
    if not _qc_query_enabled():
        print(
            "QC query disabled by MV_HOI_QC_QUERY_ENABLED; deferring "
            f"{len(candidates)} reconstruction(s) in WAITING_QC and submitting "
            "no export workflow"
        )
        return
    kratos_table = export_cfg.get("kratos_table", "")
    kratos_status_kwargs = {}
    if "kratos_status_table" in export_cfg:
        kratos_status_kwargs["kratos_status_table"] = export_cfg["kratos_status_table"]
    if "kratos_project_id" in export_cfg:
        kratos_status_kwargs["kratos_project_id"] = export_cfg["kratos_project_id"]
    batch_candidates = candidates
    if not manual_sequences:
        print(
            f"Checking {len(batch_candidates)} WAITING_QC candidate(s) "
            f"oldest-first to fill up to {batch_size} export(s)"
        )
    item_names = [f"{workflow['workflow_name']}.json" for workflow in batch_candidates]
    try:
        annotations_by_item = query_completed_kratos_annotations(
            kratos_table,
            item_names,
            **kratos_status_kwargs,
        )
    except QCQueryUnavailableError as exc:
        print(
            "QC query unavailable; deferring "
            f"{len(batch_candidates)} reconstruction(s) in WAITING_QC and "
            f"submitting no export workflow: {exc}"
        )
        return
    kratos_status_by_item = getattr(annotations_by_item, "statuses", {})
    print(
        f"Found completed Kratos rows for {len(annotations_by_item)} "
        f"of {len(item_names)} candidate item(s)"
    )
    if not annotations_by_item:
        if manual_sequences:
            print(
                "No completed Kratos rows found for the requested manual batch."
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
        limit=None if manual_sequences else batch_size,
        kratos_status_by_item=kratos_status_by_item,
    )

    if manual_sequences:
        prepared = locally_authorized + prepared
        prepared_names = {item.workflow["sequence_name"] for item in prepared}
        missing = [name for name in manual_sequences if name not in prepared_names]
        if missing:
            print(
                "Manual export preflight failed; no sequences were submitted. "
                "Missing authorization for: " + ", ".join(missing),
                file=sys.stderr,
            )
            return

    if not prepared:
        print("No sequences passed export QC checks for this run.")
        return

    print_export_sequences(prepared)

    submit_prepared_batches(
        prepared,
        dataset_cfg,
        db_path=db_path,
        table=table,
        dry_run=dry_run,
        trigger="manual" if manual_sequences else "automatic",
        requested_by=requested_by,
        pool=pool,
        pool_selector=pool_selector,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit batched MV HOI export workflows")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument(
        "--sequence", action="append",
        help="Manual export sequence; repeat for a batch",
    )
    parser.add_argument(
        "--sequence-file", type=Path,
        help="Text file containing one manual export sequence per line",
    )
    parser.add_argument("--ignore-qc-fail", action="store_true",
                        help="Recheck latest FAIL rows whose details start with qc_fail:")
    parser.add_argument(
        "--rerun", action="store_true",
        help="Deprecated compatibility flag; manual exports may always create a new attempt",
    )
    parser.add_argument("--bypass-qc", action="store_true",
                        help="Allow manual export without passing QC")
    parser.add_argument("--reason", help="Required audit reason with --bypass-qc")
    parser.add_argument("--force", action="store_true",
                        help="One-time manual bypass for blacklisted sequences")
    parser.add_argument("--requested-by", help="Operator identity (default: $USER)")
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
    parser.add_argument(
        "--campaign",
        help="Only export QC-passing reconstructions from this backlog/remediation campaign",
    )
    parser.add_argument(
        "--pool",
        help="Explicit configured OSMO pool override (otherwise select by capacity)",
    )
    parser.add_argument("--image-registry", help="Container registry/namespace (or V2D_IMAGE_REGISTRY)")
    args = parser.parse_args()

    global DB_PATH, TABLE
    if args.test:
        TABLE = PIPELINES_TEST_TABLE
        DB_PATH = TEST_DB_PATH

    manual_sequences = list(args.sequence or [])
    if args.sequence_file:
        manual_sequences.extend(
            line.strip() for line in args.sequence_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    manual_sequences = list(dict.fromkeys(manual_sequences))
    if args.bypass_qc and not manual_sequences:
        parser.error("--bypass-qc requires --sequence or --sequence-file")
    if args.bypass_qc and not args.reason:
        parser.error("--reason is required with --bypass-qc")

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(config['datasets'].keys())}")
        sys.exit(1)

    dataset_cfg = config["datasets"][args.dataset]
    if args.image_registry:
        dataset_cfg["image_registry"] = args.image_registry
    try:
        from .config_utils import validate_deployment_config
    except ImportError:
        from config_utils import validate_deployment_config
    try:
        validate_deployment_config(dataset_cfg, EXPORT_CONFIG_PIPELINE)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        get_workflow_cfg(dataset_cfg, EXPORT_CONFIG_PIPELINE, EXPORT_WORKFLOW)
    except KeyError:
        print(
            f"Dataset {args.dataset} has no {EXPORT_CONFIG_PIPELINE}."
            f"workflows.{EXPORT_WORKFLOW} config"
        )
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
        sequences=manual_sequences,
        ignore_qc_fail=args.ignore_qc_fail,
        rerun=args.rerun,
        bypass_qc=args.bypass_qc,
        reason=args.reason,
        force=args.force,
        requested_by=args.requested_by,
        start_time=args.start_time,
        end_time=args.end_time,
        campaign=args.campaign,
        pool=args.pool,
    )


if __name__ == "__main__":
    main()
