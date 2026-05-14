import sqlite3
import sys
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import db
import export as mv_export
import query


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
                "export_path": "data_export",
                "workflows": {
                    "reconstruction": {
                        "workflow_yaml": "osmo/mv_hoi_reconstruction.yaml",
                        "hitl_s3_base": "s3://bucket/path",
                    },
                    "export": export_cfg,
                },
            }
        },
        "osmo_pool": "pool",
    }


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


def test_init_db_adds_export_workflow_id_and_update_helper(tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_a",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_a",
        db_path=db_path,
    )

    db.update_workflow(
        "wf_a",
        status="WAITING_EXPORT",
        osmo_export_workflow_id="v2d_mv_hoi_export_20260513_010203-1",
        db_path=db_path,
    )

    row = db.get_workflow("wf_a", db_path=db_path)
    assert row["osmo_export_workflow_id"] == "v2d_mv_hoi_export_20260513_010203-1"

    conn = sqlite3.connect(db_path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(pipelines)")}
    conn.close()
    assert "osmo_export_workflow_id" in columns


def test_init_db_migrates_legacy_workflow_tables(tmp_path):
    db_path = str(tmp_path / "processing.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE workflows (
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
        CREATE TABLE workflows_test (
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
        INSERT INTO workflows
            (sequence_name, dataset, pipeline_type, pipeline_version,
             workflow_name, osmo_workflow_id, status, details)
        VALUES
            ('seq_a', 'dataset_a', 'mv_hoi_reconstruction', '1.0.0',
             'wf_a', 'wf_a-1', 'WAITING_QC', 'workflow_completed');
        """
    )
    conn.commit()
    conn.close()

    db.init_db(db_path)

    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    row = conn.execute("SELECT sequence_name, status FROM pipelines").fetchone()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(pipelines)")}
    conn.close()

    assert "pipelines" in tables
    assert "pipelines_test" in tables
    assert "workflows" not in tables
    assert "workflows_test" not in tables
    assert row == ("seq_a", "WAITING_QC")
    assert "osmo_export_workflow_id" in columns


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

    with pytest.raises(RuntimeError, match="Ambiguous DB migration"):
        db.init_db(db_path)


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
        frame_count=200,
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


def test_qc_failure_gates_count_and_union_coverage():
    too_many = [
        {"start_frame": i * 2, "end_frame": i * 2 + 1}
        for i in range(mv_export.MAX_FAILURE_ANNOTATIONS + 1)
    ]
    assert mv_export.qc_failure_reason(too_many, frame_count=100).startswith(
        "qc_fail: failure_annotations"
    )

    overlapping = [
        {"start_frame": 0, "end_frame": 20},
        {"start_frame": 10, "end_frame": 25},
        {"start_frame": 60, "end_frame": 70},
    ]
    assert mv_export.merged_interval_coverage(overlapping) == 35
    assert mv_export.qc_failure_reason(overlapping, frame_count=100).startswith(
        "qc_fail: failure_coverage"
    )

    exactly_thirty = [
        {"start_frame": 0, "end_frame": 10},
        {"start_frame": 20, "end_frame": 40},
    ]
    assert mv_export.qc_failure_reason(exactly_thirty, frame_count=100) is None


def test_qc_failure_gates_accept_configured_thresholds():
    six_rows = [
        {"start_frame": i * 2, "end_frame": i * 2 + 1}
        for i in range(6)
    ]
    assert (
        mv_export.qc_failure_reason(
            six_rows,
            frame_count=100,
            max_failure_annotations=6,
        )
        is None
    )

    thirty_five_percent = [{"start_frame": 0, "end_frame": 35}]
    assert (
        mv_export.qc_failure_reason(
            thirty_five_percent,
            frame_count=100,
            max_failure_coverage=0.35,
        )
        is None
    )
    assert mv_export.qc_failure_reason(
        thirty_five_percent,
        frame_count=100,
        max_failure_coverage=0.34,
    ).startswith("qc_fail: failure_coverage>34%")


def test_frame_count_discovery_order_and_edex_fallback():
    assert mv_export.frame_count_from_sources(
        "frame_count: 123\n",
        "frame_count: 456\n",
        '[{"frame_start": 0, "frame_end": 789}]',
    ) == 123
    assert mv_export.frame_count_from_sources(
        None,
        "frame_count: 456\n",
        '[{"frame_start": 0, "frame_end": 789}]',
    ) == 456
    assert mv_export.frame_count_from_sources(
        None,
        None,
        '[{"version": "0.9", "frame_start": 7, "frame_end": 39}]',
    ) == 32


def test_waiting_qc_candidates_use_latest_row_per_sequence(tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_a",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_old",
        status="WAITING_QC",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="seq_a",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_new",
        status="PASS",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="2026-03-02_later",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_later",
        status="WAITING_QC",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="2026-02-01_older",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_older",
        status="WAITING_QC",
        db_path=db_path,
    )

    candidates = mv_export.waiting_qc_candidates(
        "dataset_a",
        db_path=db_path,
    )

    assert [row["workflow_name"] for row in candidates] == ["wf_older", "wf_later"]


def test_prepare_exports_marks_failed_qc_rows_and_keeps_valid_rows(tmp_path):
    db_path = _db_path(tmp_path)
    for workflow_name, sequence in (("wf_bad", "seq_bad"), ("wf_ok", "seq_ok")):
        db.insert_workflow(
            sequence_name=sequence,
            dataset="dataset_a",
            pipeline_type=query.RECON_PIPELINE,
            pipeline_version="1.0.0",
            workflow_name=workflow_name,
            status="WAITING_QC",
            db_path=db_path,
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

    assert [item.workflow["workflow_name"] for item in prepared] == ["wf_ok"]
    assert db.get_workflow("wf_bad", db_path=db_path)["status"] == "FAIL"


def test_prepare_exports_uses_configured_failure_annotation_limit(tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_ok",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_ok",
        status="WAITING_QC",
        db_path=db_path,
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


def test_submit_batch_uses_unsuffixed_name_and_stores_osmo_id(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_export",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_export",
        status="WAITING_QC",
        db_path=db_path,
    )
    item = mv_export.PreparedExport(
        workflow=_workflow("seq_export", "wf_export"),
        failure_segments=[],
        source_url="swift://host/AUTH/container/data_output/seq_export",
        export_url="swift://host/AUTH/container/data_export/seq_export",
        task_suffix=mv_export.task_suffix("seq_export"),
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
        {"workflow_name": "v2d_mv_hoi_export_20260513_163035"}
    ]
    assert export_id == "v2d_mv_hoi_export_20260513_163035-1"
    row = db.get_workflow("wf_export", db_path=db_path)
    assert row["status"] == "WAITING_EXPORT"
    assert row["osmo_export_workflow_id"] == "v2d_mv_hoi_export_20260513_163035-1"


def test_run_export_checks_only_oldest_waiting_qc_batch(monkeypatch):
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

    assert queried_items == ["wf_old.json", "wf_mid.json"]


def test_run_export_sequence_mode_queries_only_selected_sequence(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    for workflow_name, sequence in (
        ("wf_target", "seq_target"),
        ("wf_other", "seq_other"),
    ):
        db.insert_workflow(
            sequence_name=sequence,
            dataset="dataset_a",
            pipeline_type=query.RECON_PIPELINE,
            pipeline_version="1.0.0",
            workflow_name=workflow_name,
            status="WAITING_QC",
            db_path=db_path,
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


def test_run_export_sequence_mode_marks_only_selected_qc_failure(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    for workflow_name, sequence in (
        ("wf_bad", "seq_bad"),
        ("wf_other", "seq_other"),
    ):
        db.insert_workflow(
            sequence_name=sequence,
            dataset="dataset_a",
            pipeline_type=query.RECON_PIPELINE,
            pipeline_version="1.0.0",
            workflow_name=workflow_name,
            status="WAITING_QC",
            db_path=db_path,
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
    monkeypatch.setattr(
        mv_export,
        "submit_batch",
        lambda *args, **kwargs: pytest.fail("submit_batch should not be called"),
    )

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(),
        db_path=db_path,
        sequence="seq_bad",
    )

    assert db.get_workflow("wf_bad", db_path=db_path)["status"] == "FAIL"
    assert db.get_workflow("wf_other", db_path=db_path)["status"] == "WAITING_QC"


def test_run_export_sequence_mode_rechecks_qc_fail_when_requested(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_retry",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_retry",
        status="FAIL",
        details="qc_fail: failure_annotations>2 (3)",
        db_path=db_path,
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
    )

    assert queried_items == ["wf_retry.json"]
    assert submitted_sequences == ["seq_retry"]


def test_run_export_rechecked_qc_fail_can_remain_failed(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_retry",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_retry",
        status="FAIL",
        details="qc_fail: previous threshold",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="seq_unrelated",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_unrelated",
        status="FAIL",
        details="task_failed: reconstruction",
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
    monkeypatch.setattr(
        mv_export,
        "submit_batch",
        lambda *args, **kwargs: pytest.fail("submit_batch should not be called"),
    )

    mv_export.run_export(
        "dataset_a",
        _export_dataset_cfg(),
        db_path=db_path,
        sequence="seq_retry",
        ignore_qc_fail=True,
    )

    retry = db.get_workflow("wf_retry", db_path=db_path)
    assert retry["status"] == "FAIL"
    assert retry["details"] == "qc_fail: failure_annotations>5 (6)"
    assert (
        db.get_workflow("wf_unrelated", db_path=db_path)["details"]
        == "task_failed: reconstruction"
    )


@pytest.mark.parametrize(
    "details",
    [
        "invalid_failure_annotation: start_frame=0, end_frame=4, frame_count=20",
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


def test_run_export_sequence_mode_ignores_non_waiting_qc_latest_row(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_done",
        dataset="dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name="wf_done",
        status="PASS",
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
        sequence="seq_done",
    )

    assert db.get_workflow("wf_done", db_path=db_path)["status"] == "PASS"


def test_run_export_batch_mode_rechecks_qc_fail_and_respects_batch_size(
    monkeypatch,
    tmp_path,
):
    db_path = _db_path(tmp_path)
    rows = [
        ("2026-01_wait", "wf_wait", "WAITING_QC", ""),
        ("2026-02_qc_fail", "wf_qc_fail", "FAIL", "qc_fail: failure_annotations>2 (3)"),
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

    assert queried_items == ["wf_wait.json", "wf_qc_fail.json"]


def test_prepare_exports_leaves_same_qc_fail_details_unchanged(monkeypatch):
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
    monkeypatch.setattr(
        mv_export,
        "update_workflow",
        lambda *args, **kwargs: pytest.fail("unchanged qc_fail should not update DB"),
    )

    prepared = mv_export.prepare_exports(
        [workflow],
        annotations,
        _export_dataset_cfg(),
        lambda _sequence: 100,
    )

    assert prepared == []


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


def test_refresh_waiting_exports_maps_task_statuses(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    export_id = "v2d_mv_hoi_export_20260513_010203-1"
    for workflow_name, sequence in (("wf_done", "seq_done"), ("wf_failed", "seq_failed")):
        db.insert_workflow(
            sequence_name=sequence,
            dataset="dataset_a",
            pipeline_type=query.RECON_PIPELINE,
            pipeline_version="1.0.0",
            workflow_name=workflow_name,
            status="WAITING_EXPORT",
            db_path=db_path,
        )
        db.update_workflow(
            workflow_name,
            osmo_export_workflow_id=export_id,
            db_path=db_path,
        )

    done_export, done_copy = query.export_task_names("seq_done")
    failed_export, failed_copy = query.export_task_names("seq_failed")
    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _export_id: {
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

    assert db.get_workflow("wf_done", db_path=db_path)["status"] == "PASS"
    failed = db.get_workflow("wf_failed", db_path=db_path)
    assert failed["status"] == "FAIL"
    assert "task_failed" in failed["details"]


def test_query_export_workflows_parallelizes_osmo_queries(monkeypatch):
    queried = []

    def _fake_osmo_query(export_id):
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

    query.refresh_workflow_states(
        "dataset_a",
        pipeline_type=query.RECON_PIPELINE,
        db_path="db.sqlite",
        table="pipelines_test",
        max_workers=7,
    )

    assert [call[0] for call in calls] == ["waiting", "export"]
    assert calls[0][2]["pipeline_type"] == query.RECON_PIPELINE
    assert calls[1][2]["db_path"] == "db.sqlite"
    assert calls[1][2]["max_workers"] == 7


def test_render_batch_workflow_injects_localpath():
    item = mv_export.PreparedExport(
        workflow=_workflow("2026-01-01_00-00-00_seq", "wf_seq"),
        failure_segments=[],
        source_url="swift://host/AUTH/container/data_output/seq",
        export_url="swift://host/AUTH/container/data_export/seq",
        task_suffix=mv_export.task_suffix("2026-01-01_00-00-00_seq"),
    )

    rendered = mv_export.render_batch_workflow(
        "workflow:\n  name: {{workflow_name}}\n  tasks:\n__TASKS__\n",
        [item],
    )

    assert "localpath: failure_segments/" in rendered
    assert "python -m v2d.mv.postprocess.lib.export_sequence" in rendered
