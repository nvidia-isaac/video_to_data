import sqlite3
import sys
import types
import json
from datetime import datetime
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import db, query
from orchestration import export as mv_export


def _db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "processing.db")
    db.init_db(path)
    db.insert_version("1.0.0", db_path=path)
    return path


def _workflow(
    sequence: str,
    workflow_name: str,
    status: str = "WAITING_QC",
    export_id: str | None = None,
) -> dict:
    return {
        "sequence_name": sequence,
        "dataset": "dataset_a",
        "pipeline_type": query.RECON_PIPELINE,
        "pipeline_version": "1.0.0",
        "workflow_name": workflow_name,
        "osmo_workflow_id": f"{workflow_name}-1",
        "osmo_export_workflow_id": export_id,
        "status": status,
        "details": "",
    }


def _insert_completed_reconstruction(
    db_path: str,
    *,
    sequence_name: str,
    workflow_name: str,
    qc_status: str = "PENDING",
    details: str = "workflow_completed",
) -> dict:
    """Create the normalized equivalent of a legacy completed recon row."""
    run = db.insert_workflow(
        sequence_name=sequence_name,
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name=workflow_name,
        status="PASS",
        details=details,
        trigger="migration",
        db_path=db_path,
    )
    db.record_qc_review(
        run["stage_run_id"], qc_status, details=details, source="test",
        db_path=db_path,
    )
    return run


def _export_dataset_cfg(**export_overrides) -> dict:
    export_cfg = {
        "workflow_yaml": "osmo/mv_hoi_export.yaml",
        "batch_size": 30,
        "kratos_table": "catalog.schema.annotations",
    }
    export_cfg.update(export_overrides)
    return {
        "swift_base": "swift://host/AUTH/container/root",
        "pipelines": {
            query.RECON_PIPELINE: {
                "input_path": "data",
                "output_path": "data_output",
                "workflows": {
                    "reconstruction": {
                        "workflow_yaml": "osmo/mv_hoi_reconstruction.yaml",
                        "hitl_s3_base": "s3://bucket/path",
                    },
                },
            },
            "mv_hoi_export": {
                "input_path": "data_output",
                "work_output_path": "_export_work",
                "output_path": "data_export_2",
                "workflows": {"export": export_cfg},
            },
        },
        "osmo_pool": "pool",
    }


def _frozen_backlog_campaign(db_path: str) -> dict:
    campaign = db.create_campaign(
        name="backlog", campaign_type="BACKLOG_REPROCESSING",
        dataset="dataset_a", pipeline_version="1.0.0",
        output_uri="swift://host/AUTH/container/root/data_export_2",
        created_by="test", db_path=db_path,
    )
    return db.freeze_campaign(
        campaign["id"],
        inventory_uri="swift://host/AUTH/container/inventory.json",
        inventory_sha256="a" * 64,
        configuration_uri="swift://host/AUTH/container/config.json",
        configuration_sha256="b" * 64,
        db_path=db_path,
    )


def _attach_reconstruction_to_campaign(
    db_path: str, reconstruction: dict, campaign: dict,
) -> dict:
    request = db.create_stage_request(
        sequence_name=reconstruction["sequence_name"], dataset="dataset_a",
        stage="reconstruction", pipeline_version="1.0.0",
        campaign=campaign["id"], cohort="BULK", db_path=db_path,
    )
    db.update_stage_request(request["id"], status="SUCCEEDED", db_path=db_path)
    connection = db.get_connection(db_path)
    try:
        connection.execute(
            "UPDATE reconstruction_runs SET request_id=? WHERE id=?",
            (request["id"], reconstruction["stage_run_id"]),
        )
        connection.commit()
    finally:
        connection.close()
    return request


def _stub_export_io(monkeypatch, frame_count: int = 100) -> None:
    monkeypatch.setattr(
        mv_export,
        "get_s3_client",
        lambda _swift_base: (object(), "bucket", "base"),
    )
    monkeypatch.setattr(
        mv_export,
        "resolve_frame_count",
        lambda *_args, **_kwargs: frame_count,
    )


def test_export_is_a_separate_stage_with_a_shared_execution(tmp_path):
    db_path = _db_path(tmp_path)
    reconstruction = _insert_completed_reconstruction(
        db_path, sequence_name="seq_a", workflow_name="wf_a",
    )
    export_id = "v2d_mv_hoi_export_20260513_010203-1"
    execution_id = db.create_workflow_execution(
        workflow_name="v2d_mv_hoi_export_20260513_010203",
        osmo_workflow_id=export_id,
        pipeline_type=db.EXPORT_STAGE,
        status="WAITING_EXPORT",
        db_path=db_path,
    )
    export_run = db.create_stage_run(
        sequence_name="seq_a", dataset="dataset_a", stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0", status="WAITING_WF",
        execution_id=execution_id,
        reconstruction_run_id=reconstruction["stage_run_id"],
        authorization_type="MANUAL_OVERRIDE", requested_by="test",
        override_reason="unit test", db_path=db_path,
    )

    assert db.get_workflow("wf_a", db_path=db_path)["osmo_export_workflow_id"] is None
    assert export_run["osmo_export_workflow_id"] == export_id
    assert export_run["upstream_run_ids"] == [reconstruction["stage_run_id"]]


def test_init_db_refuses_in_place_legacy_pipeline_migration(tmp_path):
    db_path = str(tmp_path / "processing.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE pipelines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence_name TEXT NOT NULL,
            dataset TEXT NOT NULL,
            pipeline_type TEXT NOT NULL,
            pipeline_version TEXT,
            workflow_name TEXT UNIQUE NOT NULL,
            osmo_workflow_id TEXT,
            status TEXT NOT NULL DEFAULT 'WAITING_WF',
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO pipelines
            (sequence_name, dataset, pipeline_type, pipeline_version,
             workflow_name, osmo_workflow_id, status, details)
        VALUES
            ('seq_a', 'dataset_a', 'mv_hoi_reconstruction', '1.0.0',
             'wf_a', 'wf_a-1', 'WAITING_QC', 'workflow_completed');
        """
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="Use migrate_processing_db.py"):
        db.init_db(db_path)

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT sequence_name, status FROM pipelines").fetchone() == (
        "seq_a", "WAITING_QC"
    )
    conn.close()


def test_init_db_refuses_ambiguous_legacy_and_new_tables(tmp_path):
    db_path = str(tmp_path / "processing.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE workflows (id INTEGER PRIMARY KEY);
        CREATE TABLE pipelines (id INTEGER PRIMARY KEY);
        """
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="Legacy pipeline tables detected"):
        db.init_db(db_path)


def test_generate_export_id_includes_microseconds():
    export_id = mv_export.generate_export_id(
        datetime(2026, 5, 26, 15, 30, 12, 123456)
    )

    assert export_id == "v2d_mv_hoi_export_20260526_153012_123456"


def _set_kratos_drs_env(monkeypatch):
    monkeypatch.setenv("KRATOS_PROFILE", "production")
    monkeypatch.setenv("KRATOS_NAMESPACE", "llmdf-team")
    monkeypatch.setenv("KRATOS_DRS_WAREHOUSE_ID", "drs-warehouse")
    monkeypatch.setenv("PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID", "client-id")
    monkeypatch.setenv("PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET", "client-secret")


def _install_kratos_drs_module(monkeypatch, *, execute, get):
    package = types.ModuleType("kratos")
    package.__path__ = []
    drs_jobs = types.ModuleType("kratos.drs_jobs")
    drs_jobs.execute_drs_adhoc_job = execute
    drs_jobs.get_drs_job = get
    package.drs_jobs = drs_jobs
    monkeypatch.setitem(sys.modules, "kratos", package)
    monkeypatch.setitem(sys.modules, "kratos.drs_jobs", drs_jobs)


def test_query_completed_kratos_annotations_uses_status_metrics_source_of_truth(
    monkeypatch,
):
    _set_kratos_drs_env(monkeypatch)
    submissions = []
    responses = iter([
        {
            "statusCode": 200,
            "result": {
                "jobId": "status-job",
                "data": [
                    {"item_name": "wf_qc.json", "status_to": "QC_Complete"},
                    {"item_name": "wf_done.json", "status_to": "Completed"},
                ],
            },
        },
        {
            "statusCode": 200,
            "result": {
                "jobId": "annotation-job",
                "data": [
                    {
                        "item_name": "wf_qc.json",
                        "table_json": '{"rows": [{"id": "qc"}]}',
                    },
                    {
                        "item_name": "wf_done.json",
                        "table_json": '{"rows": [{"id": "done"}]}',
                    },
                ],
            },
        },
    ])

    def _execute(**kwargs):
        submissions.append(kwargs)
        return next(responses)

    page_calls = []

    def _get(**kwargs):
        page_calls.append(kwargs)
        return {"statusCode": 200, "result": {"jobId": kwargs["job_id"], "data": []}}

    _install_kratos_drs_module(monkeypatch, execute=_execute, get=_get)

    result = mv_export.query_completed_kratos_annotations(
        "catalog.schema.annotations",
        ["wf_qc.json", "wf_done.json"],
    kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
    )

    assert "wf_qc.json" not in result
    assert result["wf_done.json"] == [{"id": "done"}]
    assert result.statuses["wf_qc.json"] == "QC_Complete"
    assert result.statuses["wf_done.json"] == "Completed"
    status_query = submissions[0]["query"]
    annotations_query = submissions[1]["query"]
    assert "catalog.schema.status_events" in status_query
    assert "SELECT MAX(date_partition)" in status_query
    assert "project_id = 42" in status_query
    assert "PARTITION BY item_name" in status_query
    assert "ORDER BY event_datetime DESC" in status_query
    assert "WHERE status_rank = 1" in status_query
    assert "catalog.schema.annotations" in annotations_query
    assert "SELECT item_name, `table` AS table_json" in annotations_query
    assert all(call["warehouse_id"] == "drs-warehouse" for call in submissions)
    assert all(call["namespace"] == "llmdf-team" for call in submissions)
    assert all(call["profile"] == "production" for call in submissions)
    assert all(call["auth_type"] == "service" for call in submissions)
    assert page_calls == []


def test_execute_kratos_drs_query_polls_and_fetches_remaining_pages(monkeypatch):
    _set_kratos_drs_env(monkeypatch)
    monkeypatch.setenv("MV_HOI_QC_QUERY_POLL_INTERVAL_SECONDS", "0.01")
    monkeypatch.setattr(mv_export.time, "sleep", lambda _seconds: None)
    calls = []
    responses = iter([
        {"statusCode": 202, "result": {"jobId": "job-1", "status": "processing"}},
        {
            "statusCode": 200,
            "result": {
                "jobId": "job-1",
                "data": [{"row": 1}],
                "metadata": {"totalPages": 2, "resultTruncated": False},
            },
        },
        {"statusCode": 200, "result": {"jobId": "job-1", "data": [{"row": 2}]}},
    ])

    def _get(**kwargs):
        calls.append(kwargs)
        return next(responses)

    _install_kratos_drs_module(
        monkeypatch,
        execute=lambda **_kwargs: next(responses),
        get=_get,
    )

    assert mv_export._execute_kratos_drs_json_query("SELECT 1") == [
        {"row": 1}, {"row": 2},
    ]
    assert [call["page"] for call in calls] == [1, 2]
    assert all(call["page_size"] == 100 for call in calls)


def test_normalize_kratos_drs_response_accepts_http_response_shape():
    response = types.SimpleNamespace(
        status_code=200,
        json=lambda: {"jobId": "job-1", "data": [{"row": 1}]},
    )

    assert mv_export._normalize_kratos_drs_response(response) == (
        200, {"jobId": "job-1", "data": [{"row": 1}]},
    )


def test_execute_kratos_drs_query_accepts_out_of_bounds_as_legacy_page_end(
    monkeypatch,
):
    _set_kratos_drs_env(monkeypatch)
    rows = [{"row": index} for index in range(100)]

    def _get(**_kwargs):
        raise RuntimeError("The requested page is out of bounds")

    _install_kratos_drs_module(
        monkeypatch,
        execute=lambda **_kwargs: {
            "statusCode": 200,
            "result": {"jobId": "job-1", "data": rows},
        },
        get=_get,
    )

    assert mv_export._execute_kratos_drs_json_query("SELECT 1") == rows


def test_execute_kratos_drs_query_rejects_truncated_result(monkeypatch):
    _set_kratos_drs_env(monkeypatch)
    _install_kratos_drs_module(
        monkeypatch,
        execute=lambda **_kwargs: {
            "statusCode": 200,
            "result": {
                "jobId": "job-1",
                "data": [],
                "metadata": {"totalPages": 1, "resultTruncated": True},
            },
        },
        get=lambda **_kwargs: pytest.fail("get_drs_job must not be called"),
    )

    with pytest.raises(RuntimeError, match="result was truncated"):
        mv_export._execute_kratos_drs_json_query("SELECT 1")


def test_execute_kratos_drs_query_rejects_202_without_job_id(monkeypatch):
    _set_kratos_drs_env(monkeypatch)
    _install_kratos_drs_module(
        monkeypatch,
        execute=lambda **_kwargs: {"statusCode": 202, "result": {}},
        get=lambda **_kwargs: pytest.fail("get_drs_job must not be called"),
    )

    with pytest.raises(RuntimeError, match="missing jobId"):
        mv_export._execute_kratos_drs_json_query("SELECT 1")


def test_execute_kratos_drs_query_defers_poll_exhaustion(monkeypatch):
    _set_kratos_drs_env(monkeypatch)
    monkeypatch.setenv("MV_HOI_QC_QUERY_MAX_POLLS", "2")
    monkeypatch.setattr(mv_export.time, "sleep", lambda _seconds: None)
    processing = {
        "statusCode": 202,
        "result": {"jobId": "job-1", "status": "processing"},
    }
    _install_kratos_drs_module(
        monkeypatch,
        execute=lambda **_kwargs: processing,
        get=lambda **_kwargs: processing,
    )

    with pytest.raises(mv_export.QCQueryUnavailableError, match="2 poll"):
        mv_export._execute_kratos_drs_json_query("SELECT 1")


def test_query_completed_kratos_annotations_defers_when_credentials_missing(
    monkeypatch,
):
    for name in (
        "KRATOS_NAMESPACE",
        "KRATOS_DRS_WAREHOUSE_ID",
        "PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID",
        "PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)

    _install_kratos_drs_module(
        monkeypatch,
        execute=lambda **_kwargs: pytest.fail("missing credentials must prevent query"),
        get=lambda **_kwargs: pytest.fail("missing credentials must prevent polling"),
    )
    with pytest.raises(mv_export.QCQueryUnavailableError, match="KRATOS_NAMESPACE"):
        mv_export.query_completed_kratos_annotations(
            "catalog.schema.annotations", ["wf.json"],
            kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
        )


def test_query_completed_kratos_annotations_defers_invalid_token(monkeypatch):
    _set_kratos_drs_env(monkeypatch)

    def _execute(**_kwargs):
        raise RuntimeError("HTTP 401 Unauthorized: invalid access token")

    _install_kratos_drs_module(
        monkeypatch, execute=_execute,
        get=lambda **_kwargs: pytest.fail("get_drs_job must not be called"),
    )

    with pytest.raises(
        mv_export.QCQueryUnavailableError, match="temporarily unavailable",
    ):
        mv_export.query_completed_kratos_annotations(
            "catalog.schema.annotations", ["wf.json"],
            kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
        )


def test_query_completed_kratos_annotations_does_not_hide_query_defect(monkeypatch):
    _set_kratos_drs_env(monkeypatch)

    def _execute(**_kwargs):
        raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND: annotations")

    _install_kratos_drs_module(
        monkeypatch, execute=_execute,
        get=lambda **_kwargs: pytest.fail("get_drs_job must not be called"),
    )

    with pytest.raises(RuntimeError, match="TABLE_OR_VIEW_NOT_FOUND"):
        mv_export.query_completed_kratos_annotations(
            "catalog.schema.annotations", ["wf.json"],
            kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
        )


def test_query_completed_kratos_annotations_does_not_hide_permission_defect(
    monkeypatch,
):
    _set_kratos_drs_env(monkeypatch)

    def _execute(**_kwargs):
        raise RuntimeError(
            "HTTP 403 Forbidden: INSUFFICIENT_PERMISSIONS for catalog"
        )

    _install_kratos_drs_module(
        monkeypatch, execute=_execute,
        get=lambda **_kwargs: pytest.fail("get_drs_job must not be called"),
    )

    with pytest.raises(RuntimeError, match="INSUFFICIENT_PERMISSIONS"):
        mv_export.query_completed_kratos_annotations(
            "catalog.schema.annotations", ["wf.json"],
            kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
        )


def test_query_completed_kratos_annotations_refuses_null_annotation(monkeypatch):
    responses = iter([
        [{"item_name": "wf.json", "status_to": "Completed"}],
        [{"item_name": "wf.json", "table_json": "null"}],
    ])
    monkeypatch.setattr(
        mv_export, "_execute_kratos_drs_json_query", lambda _query: next(responses),
    )

    result = mv_export.query_completed_kratos_annotations(
        "catalog.schema.annotations", ["wf.json"],
    kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
    )

    assert result == {}
    assert result.statuses["wf.json"] == (
        "Completed; no annotation row in catalog.schema.annotations"
    )


def test_query_completed_kratos_annotations_keeps_first_duplicate_row(monkeypatch):
    responses = iter([
        [{"item_name": "wf.json", "status_to": "Completed"}],
        [
            {"item_name": "wf.json", "table_json": '{"rows": [{"id": "first"}]}'},
            {"item_name": "wf.json", "table_json": '{"rows": [{"id": "second"}]}'},
        ],
    ])
    monkeypatch.setattr(
        mv_export, "_execute_kratos_drs_json_query", lambda _query: next(responses),
    )

    result = mv_export.query_completed_kratos_annotations(
        "catalog.schema.annotations", ["wf.json"],
    kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
    )

    assert result["wf.json"] == [{"id": "first"}]


def test_query_completed_kratos_annotations_rejects_malformed_annotation(monkeypatch):
    responses = iter([
        [{"item_name": "wf.json", "status_to": "Completed"}],
        [{"item_name": "wf.json", "table_json": "[]"}],
    ])
    monkeypatch.setattr(
        mv_export, "_execute_kratos_drs_json_query", lambda _query: next(responses),
    )

    with pytest.raises(RuntimeError, match="expected an object"):
        mv_export.query_completed_kratos_annotations(
            "catalog.schema.annotations", ["wf.json"],
            kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
        )


def test_query_completed_kratos_annotations_empty_input_needs_no_client(monkeypatch):
    monkeypatch.setattr(
        mv_export,
        "_execute_kratos_drs_json_query",
        lambda _query: pytest.fail("empty input must not query Kratos"),
    )

    assert mv_export.query_completed_kratos_annotations(
        "catalog.schema.annotations", [],
    kratos_status_table="catalog.schema.status_events", kratos_project_id=42,
    ) == {}


def test_normalize_failure_annotations_filters_and_converts_to_half_open():
    annotations = [
        {
            "id": "fail-1",
            "start_frame": 1,
            "end_frame": 10,
            "failure_category": "bad_pose",
            "reason": "drift",
            "extra": "ignored",
        },
        {
            "id": "pass-row",
            "start_frame": 5,
            "end_frame": 8,
            "failure_category": "",
            "reason": "not counted",
        },
        {
            "id": "empty-list",
            "start_frame": 7,
            "end_frame": 9,
            "failure_category": [],
            "reason": "not counted",
        },
        {
            "id": "empty-list-string",
            "start_frame": 8,
            "end_frame": 10,
            "failure_category": "[]",
            "reason": "not counted",
        },
    ]

    segments, error = mv_export.normalize_failure_annotations(annotations, frame_count=20)

    assert error is None
    assert segments == [
        {
            "id": "fail-1",
            "start_frame": 0,
            "end_frame": 10,
            "failure_category": "bad_pose",
            "reason": "drift",
        }
    ]


def test_normalize_failure_annotations_keeps_nonempty_category_lists():
    segments, error = mv_export.normalize_failure_annotations(
        [
            {
                "id": "fail-list",
                "start_frame": 95,
                "end_frame": 105,
                "failure_category": ["bad_pose"],
                "reason": "drag segment",
            }
        ],
        frame_count=220,
    )

    assert error is None
    assert segments == [
        {
            "id": "fail-list",
            "start_frame": 94,
            "end_frame": 105,
            "failure_category": ["bad_pose"],
            "reason": "drag segment",
        }
    ]


def test_normalize_failure_annotations_accepts_last_frame_closed_range():
    segments, error = mv_export.normalize_failure_annotations(
        [
            {
                "id": "last-frame",
                "start_frame": 999,
                "end_frame": 999,
                "failure_category": "bad_pose",
            }
        ],
        frame_count=999,
    )

    assert error is None
    assert segments == [
        {
            "id": "last-frame",
            "start_frame": 998,
            "end_frame": 999,
            "failure_category": "bad_pose",
            "reason": None,
        }
    ]


def test_normalize_failure_annotations_rejects_invalid_range():
    segments, error = mv_export.normalize_failure_annotations(
        [
            {
                "id": "bad",
                "start_frame": 0,
                "end_frame": 4,
                "failure_category": "bad_pose",
            }
        ],
        frame_count=20,
    )

    assert segments == []
    assert error.startswith("invalid_failure_annotation")


def test_normalize_failure_annotations_rejects_end_past_frame_count():
    segments, error = mv_export.normalize_failure_annotations(
        [
            {
                "id": "bad",
                "start_frame": 999,
                "end_frame": 1000,
                "failure_category": "bad_pose",
            }
        ],
        frame_count=999,
    )

    assert segments == []
    assert error == (
        "invalid_failure_annotation: start_frame=998, end_frame=1000, "
        "frame_count=999"
    )


def test_frame_count_discovery_order_and_edex_fallback():
    assert mv_export.frame_count_from_sources(
        "frame_count: 456\n",
        '[{"frame_start": 0, "frame_end": 789}]',
    ) == 456
    assert mv_export.frame_count_from_sources(
        None,
        '[{"version": "0.9", "frame_start": 7, "frame_end": 39}]',
    ) == 32


def test_waiting_qc_candidates_use_latest_row_per_sequence(tmp_path):
    db_path = _db_path(tmp_path)
    _insert_completed_reconstruction(
        db_path, sequence_name="seq_a", workflow_name="wf_old",
    )
    _insert_completed_reconstruction(
        db_path, sequence_name="seq_a", workflow_name="wf_new", qc_status="PASS",
    )
    _insert_completed_reconstruction(
        db_path, sequence_name="2026-03-02_later", workflow_name="wf_later",
    )
    _insert_completed_reconstruction(
        db_path, sequence_name="2026-02-01_older", workflow_name="wf_older",
    )

    candidates = mv_export.waiting_qc_candidates(
        "dataset_a",
        db_path=db_path,
    )

    assert [row["workflow_name"] for row in candidates] == ["wf_older", "wf_later"]


def test_campaign_export_candidates_are_membership_filtered_and_ignore_old_blacklist(
    tmp_path,
):
    db_path = _db_path(tmp_path)
    campaign = _frozen_backlog_campaign(db_path)
    included = _insert_completed_reconstruction(
        db_path, sequence_name="included", workflow_name="wf_included",
    )
    _attach_reconstruction_to_campaign(db_path, included, campaign)
    _insert_completed_reconstruction(
        db_path, sequence_name="unrelated", workflow_name="wf_unrelated",
    )
    db.upsert_blacklisted_sequence(
        "dataset_a", "included", reason="legacy failure", db_path=db_path,
    )

    candidates = mv_export.waiting_qc_candidates(
        "dataset_a", db_path=db_path, campaign="backlog",
    )

    assert [row["sequence_name"] for row in candidates] == ["included"]


def test_filter_workflows_by_time_uses_sequence_prefix_and_exclusive_end():
    workflows = [
        _workflow("2026-04-21_23-59-59_before", "wf_before"),
        _workflow("2026-04-22_00-00-00_start", "wf_start"),
        _workflow("2026-04-23_12-34-56_middle", "wf_middle"),
        _workflow("2026-04-24_23-59-59_end_day", "wf_end_day"),
        _workflow("2026-04-25_00-00-00_after", "wf_after"),
        _workflow("not_a_timestamp", "wf_invalid"),
    ]

    kept = mv_export._filter_workflows_by_time(
        workflows,
        start_time="2026-04-22",
        end_time="2026-04-25",
    )

    assert [row["workflow_name"] for row in kept] == [
        "wf_start",
        "wf_middle",
        "wf_end_day",
    ]


def test_prepare_exports_defers_qc_thresholds_until_after_trim(tmp_path):
    db_path = _db_path(tmp_path)
    for workflow_name, sequence in (("wf_bad", "seq_bad"), ("wf_ok", "seq_ok")):
        _insert_completed_reconstruction(
            db_path, sequence_name=sequence, workflow_name=workflow_name,
        )

    candidates = mv_export.waiting_qc_candidates("dataset_a", db_path=db_path)
    annotations = {
        "wf_bad.json": [
            {
                "id": str(i),
                "start_frame": i + 1,
                "end_frame": i + 1,
                "failure_category": "bad_pose",
            }
            for i in range(6)
        ],
        "wf_ok.json": [
            {
                "id": "ok",
                "start_frame": 1,
                "end_frame": 2,
                "failure_category": "bad_pose",
            }
        ],
    }

    prepared = mv_export.prepare_exports(
        candidates,
        annotations,
        _export_dataset_cfg(),
        lambda _sequence: 100,
        db_path=db_path,
    )

    assert [item.workflow["workflow_name"] for item in prepared] == [
        "wf_bad", "wf_ok",
    ]
    assert db.get_workflow("wf_bad", db_path=db_path)["status"] == "PASS"
    assert db.get_blacklisted_sequence(
        "dataset_a", "seq_bad", db_path=db_path,
    ) is None
    assert db.get_blacklisted_sequence(
        "dataset_a", "seq_ok", db_path=db_path,
    ) is None


def test_campaign_export_uses_current_campaign_version_without_rewriting_reconstruction(
    tmp_path,
):
    db_path = _db_path(tmp_path)
    reconstruction = _insert_completed_reconstruction(
        db_path, sequence_name="seq_a", workflow_name="wf_a",
    )
    campaign = _frozen_backlog_campaign(db_path)
    _attach_reconstruction_to_campaign(db_path, reconstruction, campaign)
    db.insert_version("1.0.1", db_path=db_path)
    connection = db.get_connection(db_path)
    try:
        connection.execute(
            "UPDATE processing_campaigns SET pipeline_version=? WHERE id=?",
            ("1.0.1", campaign["id"]),
        )
        connection.commit()
    finally:
        connection.close()

    candidates = mv_export.waiting_qc_candidates("dataset_a", db_path=db_path)
    prepared = mv_export.prepare_exports(
        candidates,
        {"wf_a.json": []},
        _export_dataset_cfg(),
        lambda _sequence: 100,
        db_path=db_path,
        dry_run=True,
    )

    assert prepared[0].workflow["pipeline_version"] == "1.0.1"
    assert db.get_workflow("wf_a", db_path=db_path)["pipeline_version"] == "1.0.0"


def test_manual_campaign_reexport_uses_current_campaign_version(tmp_path):
    db_path = _db_path(tmp_path)
    reconstruction = _insert_completed_reconstruction(
        db_path, sequence_name="seq_a", workflow_name="wf_a", qc_status="PASS",
    )
    campaign = _frozen_backlog_campaign(db_path)
    _attach_reconstruction_to_campaign(db_path, reconstruction, campaign)
    db.insert_version("1.0.1", db_path=db_path)
    connection = db.get_connection(db_path)
    try:
        connection.execute(
            "UPDATE processing_campaigns SET pipeline_version=? WHERE id=?",
            ("1.0.1", campaign["id"]),
        )
        connection.commit()
    finally:
        connection.close()

    prepared = mv_export.prepare_manual_override_exports(
        ["seq_a"], "dataset_a", _export_dataset_cfg(),
        bypass_qc=False, reason=None, force=False, requested_by="test",
        db_path=db_path, table=db.PIPELINES_TABLE,
    )

    assert prepared[0].workflow["pipeline_version"] == "1.0.1"
    assert db.get_workflow("wf_a", db_path=db_path)["pipeline_version"] == "1.0.0"


def test_prepare_exports_preserves_existing_blacklist_reason(tmp_path):
    db_path = _db_path(tmp_path)
    _insert_completed_reconstruction(
        db_path, sequence_name="seq_bad", workflow_name="wf_bad",
    )
    db.upsert_blacklisted_sequence(
        "dataset_a", "seq_bad", reason="operator hold", db_path=db_path,
    )

    candidates = mv_export.waiting_qc_candidates("dataset_a", db_path=db_path)
    annotations = {
        "wf_bad.json": [
            {
                "id": str(i),
                "start_frame": i + 1,
                "end_frame": i + 1,
                "failure_category": "bad_pose",
            }
            for i in range(6)
        ]
    }

    prepared = mv_export.prepare_exports(
        candidates,
        annotations,
        _export_dataset_cfg(),
        lambda _sequence: 100,
        db_path=db_path,
    )

    assert prepared == []
    assert db.get_workflow("wf_bad", db_path=db_path)["status"] == "WAITING_QC"
    assert db.get_blacklisted_sequence(
        "dataset_a", "seq_bad", db_path=db_path,
    )["reason"] == "operator hold"


def test_prepare_exports_does_not_blacklist_non_qc_blockers(tmp_path):
    db_path = _db_path(tmp_path)
    for workflow_name, sequence in (
        ("wf_missing_frame_count", "seq_missing_frame_count"),
        ("wf_invalid_annotation", "seq_invalid_annotation"),
    ):
        _insert_completed_reconstruction(
            db_path, sequence_name=sequence, workflow_name=workflow_name,
        )

    candidates = mv_export.waiting_qc_candidates("dataset_a", db_path=db_path)
    annotations = {
        "wf_missing_frame_count.json": [
            {
                "id": "missing",
                "start_frame": 1,
                "end_frame": 1,
                "failure_category": "bad_pose",
            }
        ],
        "wf_invalid_annotation.json": [
            {
                "id": "invalid",
                "start_frame": 0,
                "end_frame": 1,
                "failure_category": "bad_pose",
            }
        ],
    }

    prepared = mv_export.prepare_exports(
        candidates,
        annotations,
        _export_dataset_cfg(),
        lambda sequence: None if sequence == "seq_missing_frame_count" else 100,
        db_path=db_path,
    )

    assert prepared == []
    assert db.get_workflow(
        "wf_missing_frame_count", db_path=db_path,
    )["status"] == "WAITING_QC"
    assert db.get_workflow(
        "wf_invalid_annotation", db_path=db_path,
    )["status"] == "FAIL"
    assert db.get_blacklisted_sequence(
        "dataset_a", "seq_missing_frame_count", db_path=db_path,
    ) is None
    assert db.get_blacklisted_sequence(
        "dataset_a", "seq_invalid_annotation", db_path=db_path,
    ) is None


def test_prepare_exports_uses_configured_failure_annotation_limit(tmp_path):
    db_path = _db_path(tmp_path)
    _insert_completed_reconstruction(
        db_path, sequence_name="seq_ok", workflow_name="wf_ok",
    )

    candidates = mv_export.waiting_qc_candidates("dataset_a", db_path=db_path)
    annotations = {
        "wf_ok.json": [
            {
                "id": str(i),
                "start_frame": i + 1,
                "end_frame": i + 1,
                "failure_category": "bad_pose",
            }
            for i in range(6)
        ]
    }

    prepared = mv_export.prepare_exports(
        candidates,
        annotations,
        _export_dataset_cfg(max_failure_annotations=6),
        lambda _sequence: 100,
        db_path=db_path,
    )

    assert [item.workflow["workflow_name"] for item in prepared] == ["wf_ok"]


def test_prepare_exports_prints_missing_kratos_rejection(capsys):
    candidates = [
        _workflow("seq_missing", "wf_missing"),
        _workflow("seq_ok", "wf_ok"),
    ]
    annotations = {
        "wf_ok.json": [
            {
                "id": "ok",
                "start_frame": 1,
                "end_frame": 2,
                "failure_category": "bad_pose",
            }
        ]
    }

    prepared = mv_export.prepare_exports(
        candidates,
        annotations,
        _export_dataset_cfg(),
        lambda _sequence: 100,
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert [item.workflow["workflow_name"] for item in prepared] == ["wf_ok"]
    assert (
        "seq_missing (wf_missing.json): no Kratos row; staying WAITING_QC"
    ) in output
    assert "Export QC summary: accepted=1, rejected_or_waiting=1" in output
    assert "kratos_not_completed=1" in output


def test_prepare_exports_prints_actual_kratos_status(capsys):
    candidates = [
        _workflow("seq_pending", "wf_pending"),
    ]

    prepared = mv_export.prepare_exports(
        candidates,
        {},
        _export_dataset_cfg(),
        lambda _sequence: 100,
        dry_run=True,
        kratos_status_by_item={"wf_pending.json": "Pending review"},
    )

    output = capsys.readouterr().out
    assert prepared == []
    assert (
        "seq_pending (wf_pending.json): Kratos status='Pending review'; "
        "staying WAITING_QC"
    ) in output


def test_prepare_exports_respects_batch_limit():
    candidates = [
        _workflow(f"seq_{i}", f"wf_{i}")
        for i in range(3)
    ]
    annotations = {
        f"wf_{i}.json": [
            {
                "id": str(i),
                "start_frame": 1,
                "end_frame": 2,
                "failure_category": "bad_pose",
            }
        ]
        for i in range(3)
    }

    prepared = mv_export.prepare_exports(
        candidates,
        annotations,
        _export_dataset_cfg(),
        lambda _sequence: 100,
        dry_run=True,
        limit=2,
    )

    assert [item.workflow["workflow_name"] for item in prepared] == ["wf_0", "wf_1"]


def test_prepare_exports_uses_independent_export_pipeline_paths():
    cfg = _export_dataset_cfg()
    cfg["pipelines"]["mv_hoi_export"]["input_path"] = "reconstruction_ready"
    cfg["pipelines"]["mv_hoi_export"]["output_path"] = "published_exports"

    prepared = mv_export.prepare_exports(
        [_workflow("seq_a", "wf_a")],
        {
            "wf_a.json": [{
                "id": "ok",
                "start_frame": 1,
                "end_frame": 2,
                "failure_category": "bad_pose",
            }],
        },
        cfg,
        lambda _sequence: 100,
        dry_run=True,
    )

    assert prepared[0].source_url.endswith("/reconstruction_ready/seq_a")
    assert prepared[0].export_url.endswith("/published_exports/seq_a")


def test_submit_batch_uses_unsuffixed_name_and_stores_osmo_id(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    reconstruction = _insert_completed_reconstruction(
        db_path, sequence_name="seq_export", workflow_name="wf_export",
    )
    item = mv_export.PreparedExport(
        workflow=reconstruction,
        failure_segments=[],
        source_url="swift://host/AUTH/container/data_output/seq_export",
        export_url="swift://host/AUTH/container/data_export_2/seq_export",
        task_suffix=mv_export.task_suffix("seq_export"),
        qc_review_id=db.record_qc_review(
            reconstruction["stage_run_id"], "PASS", source="kratos",
            external_review_id="wf_export.json", db_path=db_path,
        ),
    )
    submit_calls = []

    monkeypatch.setattr(
        mv_export,
        "generate_export_id",
        lambda: "v2d_mv_hoi_export_20260513_163035",
    )
    monkeypatch.setattr(
        mv_export,
        "write_generated_workflow",
        lambda *_args, **_kwargs: tmp_path / "workflow.yaml",
    )

    def _osmo_submit(_yaml, _pool, set_vars, *, dry_run=False):
        submit_calls.append(set_vars)
        return "v2d_mv_hoi_export_20260513_163035-1"

    monkeypatch.setattr(mv_export, "osmo_submit", _osmo_submit)

    export_id = mv_export.submit_batch(
        [item],
        _export_dataset_cfg(),
        "osmo/mv_hoi_export.yaml",
        db_path=db_path,
    )

    assert submit_calls == [
        {
            "workflow_name": "v2d_mv_hoi_export_20260513_163035",
            "image_tag": "1.0.0",
            "image_registry": "registry.example.com/test-pipelines",
        }
    ]
    assert export_id == "v2d_mv_hoi_export_20260513_163035-1"
    reconstruction = db.get_workflow("wf_export", db_path=db_path)
    assert reconstruction["status"] == "PASS"
    assert reconstruction["execution_status"] == "PASS"
    export_run = db.get_current_stage_run(
        "seq_export", "dataset_a", db.EXPORT_STAGE, db_path=db_path,
    )
    assert export_run["status"] == "WAITING_WF"
    assert export_run["osmo_export_workflow_id"] == (
        "v2d_mv_hoi_export_20260513_163035-1"
    )
    assert export_run["pool"] == "pool"
    assert "pool_selection=" in export_run["execution_details"]
    request = db.get_stage_request(export_run["request_id"], db_path=db_path)
    parameters = json.loads(request["parameters_json"])
    assert parameters["candidate_uri"].endswith(
        f"/_export_work/mainline/seq_export/request_{request['id']}"
    )
    assert parameters["output_uri"].endswith("/data_export_2/seq_export")
    assert parameters["output_layout"] == "request_scoped_v1"


def test_campaign_export_stages_to_unique_candidate_prefix(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    campaign = _frozen_backlog_campaign(db_path)
    reconstruction = _insert_completed_reconstruction(
        db_path, sequence_name="seq_export", workflow_name="wf_export",
        qc_status="PASS",
    )
    _attach_reconstruction_to_campaign(db_path, reconstruction, campaign)
    reconstruction = db.get_current_stage_run(
        "seq_export", "dataset_a", db.RECONSTRUCTION_STAGE, db_path=db_path,
    )
    item = mv_export.PreparedExport(
        workflow=reconstruction, failure_segments=[],
        source_url="swift://host/AUTH/container/root/data_output/seq_export",
        export_url="swift://host/AUTH/container/root/data_export_2/seq_export",
        task_suffix=mv_export.task_suffix("seq_export"),
        qc_review_id=db.get_latest_qc_review(
            reconstruction["stage_run_id"], db_path=db_path,
        )["id"],
    )
    captured = {}
    monkeypatch.setattr(
        mv_export, "generate_export_id",
        lambda: "v2d_mv_hoi_export_20260720_120000",
    )

    def _write(_name, _template, items):
        captured["items"] = items
        return tmp_path / "workflow.yaml"

    monkeypatch.setattr(mv_export, "write_generated_workflow", _write)
    monkeypatch.setattr(
        mv_export, "osmo_submit", lambda *_args, **_kwargs: "export-1",
    )

    mv_export.submit_batch(
        [item], _export_dataset_cfg(), "osmo/mv_hoi_export.yaml",
        db_path=db_path,
    )

    request = db.list_stage_requests(
        campaign="backlog", stage="export", db_path=db_path,
    )[0]
    staged = captured["items"][0]
    assert staged.export_url.endswith(
        f"/_export_work/backlog/seq_export/request_{request['id']}"
    )
    assert staged.publish_url.endswith("/data_export_2/seq_export")
    parameters = __import__("json").loads(request["parameters_json"])
    assert parameters["candidate_uri"] == staged.export_url
    assert parameters["output_uri"] == staged.publish_url

    export_task, _ = query.export_task_names("seq_export")
    monkeypatch.setattr(
        query, "osmo_query",
        lambda _id, **_kwargs: {
            "status": "COMPLETED", "tasks": {export_task: "COMPLETED"},
        },
    )
    verified = []
    monkeypatch.setattr(
        query, "verify_remote_export_commit",
        lambda url: verified.append(url) or {"complete": True},
    )
    db.upsert_blacklisted_sequence(
        "dataset_a", "seq_export", reason="legacy failure", db_path=db_path,
    )

    query.refresh_waiting_exports("dataset_a", db_path=db_path)

    refreshed_request = db.get_stage_request(request["id"], db_path=db_path)
    refreshed_run = db.get_stage_run_by_request(
        request["id"], stage=db.EXPORT_STAGE, db_path=db_path,
    )
    assert verified == [staged.export_url]
    assert refreshed_request["status"] == "RUNNING"
    assert refreshed_run["run_status"] == "RUNNING"
    assert refreshed_run["details"] == "candidate_export_verified"
    assert db.get_blacklisted_sequence(
        "dataset_a", "seq_export", db_path=db_path,
    ) is not None


def test_run_export_queries_waiting_qc_candidates_oldest_first(monkeypatch):
    queried_items = []
    candidates = [
        _workflow("2026-02-01_old", "wf_old"),
        _workflow("2026-02-02_mid", "wf_mid"),
        _workflow("2026-03-20_new", "wf_new"),
    ]

    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        mv_export,
        "waiting_qc_candidates",
        lambda *args, **kwargs: candidates,
    )

    def _query_completed(_table, item_names):
        queried_items.extend(item_names)
        return {}

    monkeypatch.setattr(
        mv_export,
        "query_completed_kratos_annotations",
        _query_completed,
    )

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(batch_size=2),
    )

    assert queried_items == ["wf_old.json", "wf_mid.json", "wf_new.json"]


def test_run_export_defers_unavailable_qc_without_submitting(monkeypatch, capsys):
    candidates = [_workflow("seq_waiting", "wf_waiting")]
    submitted = []
    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *a, **k: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        mv_export, "waiting_qc_candidates", lambda *a, **k: candidates,
    )
    monkeypatch.setattr(
        mv_export,
        "query_completed_kratos_annotations",
        lambda *a, **k: (_ for _ in ()).throw(
            mv_export.QCQueryUnavailableError("invalid access token")
        ),
    )
    monkeypatch.setattr(
        mv_export, "submit_prepared_batches",
        lambda *a, **k: submitted.append((a, k)),
    )

    mv_export.run_export("dataset_a", _export_dataset_cfg())

    assert submitted == []
    assert (
        "deferring 1 reconstruction(s) in WAITING_QC and submitting no export workflow"
        in capsys.readouterr().out
    )


def test_run_export_skips_qc_client_when_temporarily_disabled(monkeypatch, capsys):
    candidates = [_workflow("seq_waiting", "wf_waiting")]
    queried = []
    monkeypatch.setenv("MV_HOI_QC_QUERY_ENABLED", "0")
    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *a, **k: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        mv_export, "waiting_qc_candidates", lambda *a, **k: candidates,
    )
    monkeypatch.setattr(
        mv_export, "query_completed_kratos_annotations",
        lambda *a, **k: queried.append((a, k)),
    )

    mv_export.run_export("dataset_a", _export_dataset_cfg())

    assert queried == []
    assert "QC query disabled" in capsys.readouterr().out


def test_run_export_filters_by_sequence_time_before_batching(monkeypatch):
    queried_items = []
    candidates = [
        _workflow("2026-04-21_23-59-59_before", "wf_before"),
        _workflow("2026-04-22_00-00-00_start", "wf_start"),
        _workflow("2026-04-23_12-00-00_middle", "wf_middle"),
    ]

    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        mv_export,
        "waiting_qc_candidates",
        lambda *args, **kwargs: candidates,
    )

    def _query_completed(_table, item_names):
        queried_items.extend(item_names)
        return {}

    monkeypatch.setattr(
        mv_export,
        "query_completed_kratos_annotations",
        _query_completed,
    )

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(batch_size=1),
        start_time="2026-04-22",
        end_time="2026-04-24",
    )

    assert queried_items == ["wf_start.json", "wf_middle.json"]


def test_run_export_sequence_mode_queries_only_selected_sequence(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    for workflow_name, sequence in (
        ("wf_target", "seq_target"),
        ("wf_other", "seq_other"),
    ):
        _insert_completed_reconstruction(
            db_path, sequence_name=sequence, workflow_name=workflow_name,
        )

    queried_items = []
    submitted_sequences = []
    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    _stub_export_io(monkeypatch)

    def _query_completed(_table, item_names):
        queried_items.extend(item_names)
        return {"wf_target.json": []}

    def _submit_batch(items, *args, **kwargs):
        submitted_sequences.extend(item.workflow["sequence_name"] for item in items)
        return "export_id"

    monkeypatch.setattr(mv_export, "query_completed_kratos_annotations", _query_completed)
    monkeypatch.setattr(mv_export, "submit_batch", _submit_batch)

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(batch_size=30),
        db_path=db_path,
        sequence="seq_target",
    )

    assert queried_items == ["wf_target.json"]
    assert submitted_sequences == ["seq_target"]


def test_run_export_sequence_mode_submits_threshold_check_after_trim(
    monkeypatch, tmp_path,
):
    db_path = _db_path(tmp_path)
    for workflow_name, sequence in (
        ("wf_bad", "seq_bad"),
        ("wf_other", "seq_other"),
    ):
        _insert_completed_reconstruction(
            db_path, sequence_name=sequence, workflow_name=workflow_name,
        )

    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    _stub_export_io(monkeypatch)
    monkeypatch.setattr(
        mv_export,
        "query_completed_kratos_annotations",
        lambda _table, _item_names: {
            "wf_bad.json": [
                {
                    "id": str(i),
                    "start_frame": i + 1,
                    "end_frame": i + 1,
                    "failure_category": "bad_pose",
                }
                for i in range(6)
            ]
        },
    )
    submitted_sequences = []
    monkeypatch.setattr(
        mv_export,
        "submit_batch",
        lambda items, *args, **kwargs: (
            submitted_sequences.extend(
                item.workflow["sequence_name"] for item in items
            )
            or "export_id"
        ),
    )

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(),
        db_path=db_path,
        sequence="seq_bad",
    )

    assert submitted_sequences == ["seq_bad"]
    assert db.get_workflow("wf_bad", db_path=db_path)["status"] == "PASS"
    assert db.get_workflow("wf_other", db_path=db_path)["status"] == "WAITING_QC"


def test_run_export_sequence_mode_rechecks_qc_fail_when_requested(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    _insert_completed_reconstruction(
        db_path, sequence_name="seq_retry", workflow_name="wf_retry",
        qc_status="FAIL",
        details="qc_fail: failure_annotations>2 (3)",
    )

    queried_items = []
    submitted_sequences = []
    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    _stub_export_io(monkeypatch)

    def _query_completed(_table, item_names):
        queried_items.extend(item_names)
        return {"wf_retry.json": []}

    def _submit_batch(items, *args, **kwargs):
        submitted_sequences.extend(item.workflow["sequence_name"] for item in items)
        return "export_id"

    monkeypatch.setattr(mv_export, "query_completed_kratos_annotations", _query_completed)
    monkeypatch.setattr(mv_export, "submit_batch", _submit_batch)

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(),
        db_path=db_path,
        sequence="seq_retry",
        ignore_qc_fail=True,
        force=True,
    )

    assert queried_items == ["wf_retry.json"]
    assert submitted_sequences == ["seq_retry"]


def test_run_export_rechecked_qc_fail_is_deferred_to_trimmed_export(
    monkeypatch, tmp_path,
):
    db_path = _db_path(tmp_path)
    _insert_completed_reconstruction(
        db_path, sequence_name="seq_retry", workflow_name="wf_retry",
        qc_status="FAIL",
        details="qc_fail: previous threshold",
    )
    db.insert_workflow(
        sequence_name="seq_unrelated",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_unrelated",
        status="FAIL",
        details="task_failed: reconstruction",
        trigger="migration",
        db_path=db_path,
    )

    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    _stub_export_io(monkeypatch)
    monkeypatch.setattr(
        mv_export,
        "query_completed_kratos_annotations",
        lambda _table, _item_names: {
            "wf_retry.json": [
                {
                    "id": str(i),
                    "start_frame": i + 1,
                    "end_frame": i + 1,
                    "failure_category": "bad_pose",
                }
                for i in range(6)
            ]
        },
    )
    submitted_sequences = []
    monkeypatch.setattr(
        mv_export,
        "submit_batch",
        lambda items, *args, **kwargs: (
            submitted_sequences.extend(
                item.workflow["sequence_name"] for item in items
            )
            or "export_id"
        ),
    )

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(),
        db_path=db_path,
        sequence="seq_retry",
        ignore_qc_fail=True,
        force=True,
    )

    assert submitted_sequences == ["seq_retry"]
    retry = db.get_workflow("wf_retry", db_path=db_path)
    assert retry["status"] == "PASS"
    assert retry["details"] == "qc_fail: previous threshold"
    assert (
        db.get_workflow("wf_unrelated", db_path=db_path)["details"]
        == "task_failed: reconstruction"
    )


@pytest.mark.parametrize(
    "details",
    [
        "invalid_failure_annotation: start_frame=-1, end_frame=4, frame_count=20",
        "task_failed: export_seq",
    ],
)
def test_run_export_sequence_mode_does_not_recheck_unrelated_failures(
    monkeypatch,
    tmp_path,
    details,
):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_failed",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_failed",
        status="FAIL",
        details=details,
        trigger="migration",
        db_path=db_path,
    )

    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        mv_export,
        "query_completed_kratos_annotations",
        lambda *args, **kwargs: pytest.fail("Kratos should not be queried"),
    )

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(),
        db_path=db_path,
        sequence="seq_failed",
        ignore_qc_fail=True,
    )

    assert db.get_workflow("wf_failed", db_path=db_path)["details"] == details


def test_run_export_sequence_mode_reexports_using_stored_passing_qc(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    _insert_completed_reconstruction(
        db_path, sequence_name="seq_done", workflow_name="wf_done",
        qc_status="PASS",
    )

    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        mv_export,
        "query_completed_kratos_annotations",
        lambda *args, **kwargs: pytest.fail("Kratos should not be queried"),
    )
    submitted = []
    monkeypatch.setattr(
        mv_export, "submit_batch",
        lambda items, *_args, **_kwargs: submitted.extend(items) or "export-id",
    )

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(),
        db_path=db_path,
        sequence="seq_done",
    )
    assert [item.workflow["sequence_name"] for item in submitted] == ["seq_done"]

    assert db.get_workflow("wf_done", db_path=db_path)["status"] == "PASS"


def test_run_export_batch_mode_rechecks_qc_fail_and_respects_batch_size(
    monkeypatch,
    tmp_path,
    capsys,
):
    db_path = _db_path(tmp_path)
    rows = [
        ("2026-01_wait", "wf_wait", "WAITING_QC", ""),
        ("2026-02_qc_fail", "wf_qc_fail", "FAIL", "qc_fail: failure_annotations>2 (3)"),
        ("2026-03_wait", "wf_wait_2", "WAITING_QC", ""),
        ("2026-03_invalid", "wf_invalid", "FAIL", "invalid_failure_annotation: bad"),
        ("2026-04_other_fail", "wf_other_fail", "FAIL", "task_failed: reconstruction"),
    ]
    for sequence, workflow_name, status, details in rows:
        db.insert_workflow(
            sequence_name=sequence,
            dataset="dataset_a",
            pipeline_type=query.RECON_PIPELINE,
            pipeline_version="1.0.0",
            workflow_name=workflow_name,
            status=status,
            details=details,
            trigger="migration",
            db_path=db_path,
        )

    queried_items = []
    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])

    def _query_completed(_table, item_names):
        queried_items.extend(item_names)
        return {}

    monkeypatch.setattr(mv_export, "query_completed_kratos_annotations", _query_completed)

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(batch_size=2),
        db_path=db_path,
        ignore_qc_fail=True,
    )

    output = capsys.readouterr().out
    assert queried_items == ["wf_wait.json", "wf_qc_fail.json", "wf_wait_2.json"]
    assert (
        "Checking 3 WAITING_QC candidate(s) oldest-first to fill up to "
        "2 export(s)"
    ) in output
    assert "Found completed Kratos rows for 0 of 3 candidate item(s)" in output


def test_prepare_exports_rechecks_legacy_qc_fail_only_after_trim(
    monkeypatch, tmp_path,
):
    db_path = _db_path(tmp_path)
    workflow = _workflow("seq_fail", "wf_fail", status="FAIL")
    workflow["details"] = "qc_fail: failure_annotations>5 (6)"
    annotations = {
        "wf_fail.json": [
            {
                "id": str(i),
                "start_frame": i + 1,
                "end_frame": i + 1,
                "failure_category": "bad_pose",
            }
            for i in range(6)
        ]
    }
    for helper in ("record_qc_review", "update_stage_run"):
        monkeypatch.setattr(
            mv_export,
            helper,
            lambda *args, **kwargs: pytest.fail(
                "unchanged qc_fail should not update DB"
            ),
        )

    prepared = mv_export.prepare_exports(
        [workflow],
        annotations,
        _export_dataset_cfg(),
        lambda _sequence: 100,
        db_path=db_path,
    )

    assert [item.workflow["workflow_name"] for item in prepared] == ["wf_fail"]


def test_run_export_prints_sequences_selected_for_batch(monkeypatch, capsys):
    candidates = [
        _workflow("seq_a", "wf_a"),
        _workflow("seq_b", "wf_b"),
    ]
    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(mv_export, "waiting_qc_candidates", lambda *args, **kwargs: candidates)
    _stub_export_io(monkeypatch)
    monkeypatch.setattr(
        mv_export,
        "query_completed_kratos_annotations",
        lambda _table, _item_names: {"wf_a.json": [], "wf_b.json": []},
    )
    monkeypatch.setattr(mv_export, "submit_batch", lambda *args, **kwargs: "export_id")

    mv_export.run_export("dataset_a", _export_dataset_cfg())

    output = capsys.readouterr().out
    assert "Exporting 2 sequence(s):" in output
    assert "  seq_a" in output
    assert "  seq_b" in output


def test_run_export_can_reuse_an_already_refreshed_snapshot(monkeypatch):
    monkeypatch.setattr(
        mv_export, "refresh_workflow_states",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("campaign cycle must not refresh export state twice")
        ),
    )
    monkeypatch.setattr(
        mv_export, "waiting_export_rows",
        lambda *_args, **_kwargs: [{
            "osmo_export_workflow_id": "already-running",
        }],
    )

    mv_export.run_export(
        "dataset_a", _export_dataset_cfg(), refresh=False,
    )


def test_run_export_fills_batch_past_non_completed_kratos_rows(monkeypatch):
    candidates = [
        _workflow("seq_pending", "wf_pending"),
        _workflow("seq_a", "wf_a"),
        _workflow("seq_b", "wf_b"),
        _workflow("seq_c", "wf_c"),
    ]
    queried_items = []
    submitted_sequences = []

    monkeypatch.setattr(mv_export, "refresh_workflow_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(mv_export, "waiting_export_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(mv_export, "waiting_qc_candidates", lambda *args, **kwargs: candidates)
    _stub_export_io(monkeypatch)

    def _query_completed(_table, item_names):
        queried_items.extend(item_names)
        return {
            "wf_a.json": [],
            "wf_b.json": [],
            "wf_c.json": [],
        }

    def _submit_batch(items, *args, **kwargs):
        submitted_sequences.extend(item.workflow["sequence_name"] for item in items)
        return "export_id"

    monkeypatch.setattr(mv_export, "query_completed_kratos_annotations", _query_completed)
    monkeypatch.setattr(mv_export, "submit_batch", _submit_batch)

    mv_export.run_export("dataset_a", _export_dataset_cfg(batch_size=2))

    assert queried_items == [
        "wf_pending.json",
        "wf_a.json",
        "wf_b.json",
        "wf_c.json",
    ]
    assert submitted_sequences == ["seq_a", "seq_b"]


def test_prepare_pending_export_retry_reuses_bound_reconstruction_and_qc(monkeypatch):
    request = {
        "id": 22, "status": "PENDING", "stage": "export",
        "sequence_name": "seq", "campaign_id": 9,
        "campaign_name": "campaign", "campaign_type": "BACKLOG_REPROCESSING",
        "cohort": "BULK", "queue_priority": 100, "pipeline_version": "1.6.45",
        "requested_by": "pytest", "reason": "infrastructure_retry_of_request_11",
        "parameters_json": json.dumps({
            "prior_request_id": 11,
            "source_uri": "swift://host/account/bucket/source",
            "preprocess_source_uri": "swift://host/account/bucket/preprocess",
            "output_uri": "swift://host/account/bucket/data_export_3/seq",
        }),
        "source_manifest_json": json.dumps({"reconstruction_run_id": 7}),
    }
    reconstruction = {
        "stage_run_id": 7, "run_status": "SUCCEEDED", "dataset": "dataset_a",
        "sequence_name": "seq", "request_id": 6,
        "labeled_bboxes_sha256": "a" * 64,
    }
    monkeypatch.setattr(mv_export, "list_stage_requests", lambda **_kwargs: [request])
    monkeypatch.setattr(
        mv_export, "get_campaign", lambda *_args, **_kwargs: {
            "configuration_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(mv_export, "get_stage_run", lambda *_args, **_kwargs: reconstruction)
    monkeypatch.setattr(
        mv_export, "get_stage_run_by_request", lambda *_args, **_kwargs: {
            "reconstruction_run_id": 7, "qc_review_id": 5,
            "authorization_type": "QC", "override_reason": None,
        },
    )
    monkeypatch.setattr(
        mv_export, "get_qc_review", lambda *_args, **_kwargs: {
            "id": 5, "status": "PASS", "reconstruction_run_id": 7,
            "failure_segments_json": json.dumps([{"start_frame": 1, "end_frame": 6}]),
        },
    )

    prepared = mv_export.prepare_pending_export_retries(
        "campaign", _export_dataset_cfg(), db_path="unused",
    )

    assert len(prepared) == 1
    assert prepared[0].existing_request_id == 22
    assert prepared[0].qc_review_id == 5
    assert prepared[0].workflow["stage_run_id"] == 7
    assert prepared[0].failure_segments == [{"start_frame": 1, "end_frame": 6}]


def test_refresh_waiting_exports_maps_task_statuses(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    export_id = "v2d_mv_hoi_export_20260513_010203-1"
    execution_id = db.create_workflow_execution(
        workflow_name="v2d_mv_hoi_export_20260513_010203",
        osmo_workflow_id=export_id, pipeline_type=db.EXPORT_STAGE,
        status="WAITING_WF", db_path=db_path,
    )
    for workflow_name, sequence in (("wf_done", "seq_done"), ("wf_failed", "seq_failed")):
        reconstruction = _insert_completed_reconstruction(
            db_path, sequence_name=sequence, workflow_name=workflow_name,
            qc_status="PASS",
        )
        export_task, copy_task = query.export_task_names(sequence)
        review = db.get_latest_qc_review(
            reconstruction["stage_run_id"], db_path=db_path,
        )
        db.create_stage_run(
            sequence_name=sequence, dataset="dataset_a", stage=db.EXPORT_STAGE,
            pipeline_version="1.0.0", status="WAITING_WF",
            execution_id=execution_id,
            reconstruction_run_id=reconstruction["stage_run_id"],
            qc_review_id=review["id"], authorization_type="QC",
            workflow_task_name=export_task, auxiliary_task_name=copy_task,
            db_path=db_path,
        )

    done_export, done_copy = query.export_task_names("seq_done")
    failed_export, failed_copy = query.export_task_names("seq_failed")
    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _export_id, **_kwargs: {
            "status": "FAILED",
            "tasks": {
                done_export: "COMPLETED",
                done_copy: "COMPLETED",
                failed_export: "FAILED",
                failed_copy: "FAILED_UPSTREAM",
            },
        },
    )

    query.refresh_waiting_exports("dataset_a", db_path=db_path)

    assert db.get_current_stage_run(
        "seq_done", "dataset_a", db.EXPORT_STAGE, db_path=db_path,
    )["status"] == "PASS"
    failed = db.get_current_stage_run(
        "seq_failed", "dataset_a", db.EXPORT_STAGE, db_path=db_path,
    )
    assert failed["status"] == "FAIL"
    assert "task_failed" in failed["details"]


def test_post_trim_rejection_uses_reconstruction_request_provenance(monkeypatch):
    report = {
        "provenance": {
            "sequence_name": "seq_rejected",
            "pipeline_version": "1.6.46",
            "request_id": 101,
            "reconstruction_run_id": 202,
            "export_run_id": 303,
        },
    }
    monkeypatch.setattr(
        query, "read_remote_export_rejection", lambda _candidate: report,
    )
    monkeypatch.setattr(
        query, "get_stage_run",
        lambda *_args, **_kwargs: {"request_id": 101},
    )
    row = {
        "sequence_name": "seq_rejected",
        "pipeline_version": "1.6.46",
        "request_id": 404,
        "reconstruction_run_id": 202,
        "stage_run_id": 303,
        "request_parameters_json": json.dumps({
            "candidate_uri": "swift://host/bucket/work/request_404",
        }),
    }

    observed, parameters = query._post_trim_rejection(row, db_path="unused")

    assert observed is report
    assert parameters["candidate_uri"].endswith("/request_404")


def test_query_export_workflows_parallelizes_osmo_queries(monkeypatch):
    queried = []

    def _fake_osmo_query(export_id, **_kwargs):
        queried.append(export_id)
        return {"status": "COMPLETED", "tasks": {export_id: "COMPLETED"}}

    monkeypatch.setattr(query, "osmo_query", _fake_osmo_query)

    results = query._query_export_workflows(
        ["export_a", "export_b", "export_c"],
        max_workers=2,
    )

    assert sorted(queried) == ["export_a", "export_b", "export_c"]
    assert sorted(export_id for export_id, _info in results) == [
        "export_a",
        "export_b",
        "export_c",
    ]


def test_refresh_workflow_states_includes_export_refresh_for_reconstruction(monkeypatch):
    calls = []
    monkeypatch.setattr(
        query,
        "refresh_waiting",
        lambda *args, **kwargs: calls.append(("waiting", args, kwargs)),
    )
    monkeypatch.setattr(
        query,
        "refresh_waiting_exports",
        lambda *args, **kwargs: calls.append(("export", args, kwargs)),
    )
    monkeypatch.setattr(
        query,
        "backfill_accuracy_failure_summaries",
        lambda *args, **kwargs: calls.append(("accuracy", args, kwargs)) or 0,
    )

    query.refresh_workflow_states(
        "dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        db_path="db.sqlite",
        table="pipelines_test",
        max_workers=7,
    )

    assert [call[0] for call in calls] == ["waiting", "export", "accuracy"]
    assert calls[0][2]["pipeline_type"] == query.RECON_PIPELINE
    assert calls[1][2]["db_path"] == "db.sqlite"
    assert calls[1][2]["max_workers"] == 7


def test_render_batch_workflow_injects_localpath():
    workflow = _workflow("2026-01-01_00-00-00_seq", "wf_seq")
    workflow["stage_run_id"] = 17
    item = mv_export.PreparedExport(
        workflow=workflow,
        failure_segments=[],
        source_url="swift://host/AUTH/container/data_output/seq",
        export_url="swift://host/AUTH/container/data_export_2/seq",
        task_suffix=mv_export.task_suffix("2026-01-01_00-00-00_seq"),
        metrics_request_id=16,
        metrics_campaign_name="campaign",
        export_run_id=18,
        checkpoint_manifest_url="swift://host/AUTH/container/weights/foundation_pose",
        label_sha256="a" * 64,
        configuration_sha256="b" * 64,
    )

    rendered = mv_export.render_batch_workflow(
        "workflow:\n  name: {{workflow_name}}\n  tasks:\n__TASKS__\n",
        [item],
    )

    assert "localpath: failure_segments/" in rendered
    assert "python -m v2d.mv.postprocess.lib.finalize_export" in rendered
    assert "--failure-segments-path /tmp/failure_segments.json" in rendered
    assert "--authorization-type \"QC\"" in rendered
    assert "--request-id 16" in rendered
    assert '--campaign-name "campaign"' in rendered
    assert "--export-run-id 18" in rendered
    assert "--checkpoint-manifest-path {{input:1}}/manifest.json" in rendered
    assert f"--label-sha256 {'a' * 64}" in rendered
    assert f"--configuration-sha256 {'b' * 64}" in rendered
    assert "--interaction-distance-threshold-m 0.1" in rendered
    assert "--interaction-pre-contact-padding-seconds 3.0" in rendered
    assert "--interaction-post-contact-padding-seconds 3.0" in rendered
    assert "--interaction-window-frames 7" in rendered
    assert "--interaction-required-under-threshold-frames 5" in rendered
    assert "copy_failure_segments_" not in rendered
    assert "resource: cpu_export" in rendered


def test_export_workflow_allocates_sixteen_cpus_per_export_task():
    template = (
        Path(__file__).parents[1] / "osmo" / "mv_hoi_export.yaml"
    ).read_text()
    assert "cpu_export:\n      cpu: 16" in template


def test_render_batch_workflow_mounts_split_preprocess_lineage():
    workflow = _workflow("sequence", "wf_seq")
    workflow["stage_run_id"] = 17
    item = mv_export.PreparedExport(
        workflow=workflow,
        failure_segments=[],
        source_url="swift://host/AUTH/container/data_output_2/sequence/request_2",
        preprocess_source_url=(
            "swift://host/AUTH/container/data_output_2/sequence/request_1"
        ),
        export_url="swift://host/AUTH/container/_export_work/sequence/request_3",
        task_suffix=mv_export.task_suffix("sequence"),
    )

    rendered = mv_export.render_batch_workflow(
        "workflow:\n  tasks:\n__TASKS__\n", [item],
    )

    assert item.preprocess_source_url in rendered
    assert "--preprocess-source-dir {{input:1}}" in rendered


def test_refresh_single_task_export_requires_remote_commit(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    reconstruction = _insert_completed_reconstruction(
        db_path, sequence_name="seq", workflow_name="wf", qc_status="PASS",
    )
    execution_id = db.create_workflow_execution(
        workflow_name="export", osmo_workflow_id="export-1",
        pipeline_type=db.EXPORT_STAGE, status="RUNNING", db_path=db_path,
    )
    export_task, _ = query.export_task_names("seq")
    review = db.get_latest_qc_review(
        reconstruction["stage_run_id"], db_path=db_path,
    )
    run = db.create_stage_run(
        sequence_name="seq", dataset="dataset_a", stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0", status="RUNNING", execution_id=execution_id,
        reconstruction_run_id=reconstruction["stage_run_id"],
        qc_review_id=review["id"], authorization_type="QC",
        workflow_task_name=export_task,
        auxiliary_task_name=None, output_uri="swift://host/AUTH/bucket/export/seq",
        db_path=db_path,
    )
    monkeypatch.setattr(
        query, "osmo_query",
        lambda _id, **_kwargs: {
            "status": "COMPLETED", "tasks": {export_task: "COMPLETED"},
        },
    )
    monkeypatch.setattr(
        query, "verify_remote_export_commit", lambda _url: {"complete": True},
    )

    query.refresh_waiting_exports("dataset_a", db_path=db_path)

    assert db.get_stage_run(
        run["stage_run_id"], stage=db.EXPORT_STAGE, db_path=db_path,
    )["run_status"] == "SUCCEEDED"


def test_refresh_publishes_request_isolated_mainline_export(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    reconstruction = _insert_completed_reconstruction(
        db_path, sequence_name="seq", workflow_name="wf", qc_status="PASS",
    )
    candidate = "swift://host/AUTH/bucket/_export_work/mainline/seq/request_3"
    destination = "swift://host/AUTH/bucket/data_export_2/seq"
    request = db.create_stage_request(
        sequence_name="seq", dataset="dataset_a", stage="export",
        pipeline_version="1.0.0",
        parameters={
            "candidate_uri": candidate,
            "output_uri": destination,
            "output_layout": "request_scoped_v1",
        },
        db_path=db_path,
    )
    execution_id = db.create_workflow_execution(
        workflow_name="export", osmo_workflow_id="export-1",
        pipeline_type=db.EXPORT_STAGE, status="RUNNING", db_path=db_path,
    )
    assert db.reserve_stage_request(
        request["id"], reserved_by="test", db_path=db_path,
    )
    db.attach_request_execution(
        request["id"], execution_id, status="RUNNING", db_path=db_path,
    )
    export_task, _ = query.export_task_names("seq")
    run = db.create_stage_run(
        sequence_name="seq", dataset="dataset_a", stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0", status="RUNNING", execution_id=execution_id,
        reconstruction_run_id=reconstruction["stage_run_id"],
        qc_review_id=db.get_latest_qc_review(
            reconstruction["stage_run_id"], db_path=db_path,
        )["id"],
        authorization_type="QC", workflow_task_name=export_task,
        auxiliary_task_name=None, output_uri=destination,
        request_id=request["id"], db_path=db_path,
    )
    monkeypatch.setattr(
        query, "osmo_query",
        lambda _id, **_kwargs: {
            "status": "COMPLETED", "tasks": {export_task: "COMPLETED"},
        },
    )
    monkeypatch.setattr(
        query, "verify_remote_export_commit", lambda url: {"url": url},
    )
    monkeypatch.setattr(query, "require_submit_authority", lambda _action: None)
    published = []
    monkeypatch.setattr(
        query, "publish_remote_export_commit",
        lambda source, target: published.append((source, target)),
    )

    query.refresh_waiting_exports("dataset_a", db_path=db_path)

    assert published == [(candidate, destination)]
    assert db.get_stage_run(
        run["stage_run_id"], stage=db.EXPORT_STAGE, db_path=db_path,
    )["run_status"] == "SUCCEEDED"
