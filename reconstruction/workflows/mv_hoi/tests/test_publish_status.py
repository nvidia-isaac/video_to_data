import sys
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import db, publish_status


def _workflow(
    sequence: str,
    workflow_name: str,
    status: str,
    *,
    pipeline_type: str = "mv_hoi_reconstruction",
    details: str = "",
    created_at: str = "2026-05-01 00:00:00",
    stage_run_id: int = 10,
    attempt: int = 1,
    trigger: str = "auto",
    upstream_run_ids: str = "",
) -> dict:
    return {
        "dataset": "dataset_a",
        "sequence_name": sequence,
        "pipeline_type": pipeline_type,
        "status": status,
        "details": details,
        "pipeline_version": "1.0.0",
        "stage": pipeline_type,
        "stage_run_id": stage_run_id,
        "attempt": attempt,
        "trigger": trigger,
        "target_id": 5,
        "upstream_run_ids": upstream_run_ids,
        "workflow_name": workflow_name,
        "osmo_workflow_id": f"{workflow_name}-1",
        "execution_id": f"{workflow_name}-1",
        "workflow_task_name": "main",
        "auxiliary_task_name": "",
        "created_at": created_at,
        "updated_at": created_at,
    }


def _db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "processing.db")
    db.init_db(path)
    db.insert_version("1.0.0", db_path=path)
    return path


def test_latest_workflows_dedupes_newest_first_and_sorts_for_display():
    workflows = [
        _workflow("seq_b", "wf_new", "SKIPPED", details="skipped: no mesh"),
        _workflow("seq_a", "wf_pass", "PASS"),
        _workflow("seq_b", "wf_old", "FAIL", details="task_failed: old"),
    ]

    latest = publish_status.latest_workflows(workflows)

    assert [workflow["workflow_name"] for workflow in latest] == [
        "wf_pass",
        "wf_new",
    ]
    assert {workflow["status"] for workflow in latest} == {"PASS", "SKIPPED"}


def test_status_values_use_stable_headers_and_ordering():
    workflow = _workflow(
        "seq_a",
        "wf_a",
        "WAITING_QC",
        details="workflow_completed",
    )

    values = publish_status.status_values([workflow])

    assert values[0] == list(publish_status.STATUS_HEADERS)
    assert values[1] == [
        "dataset_a",
        "seq_a",
        "mv_hoi_reconstruction",
        "WAITING_QC",
        "workflow_completed",
        "10",
        "1",
        "auto",
        "5",
        "",
        "1.0.0",
        "wf_a",
        "wf_a-1",
        "wf_a-1",
        "main",
        "",
        "2026-05-01 00:00:00",
        "2026-05-01 00:00:00",
    ]


def test_status_values_accept_flattened_normalized_run_aliases():
    workflow = _workflow("seq_a", "wf_a", "RUNNING")
    workflow.pop("stage_run_id")
    workflow["run_id"] = 42
    workflow["upstream_run_ids"] = "7,9"

    values = publish_status.status_values([workflow])

    assert "osmo_export_workflow_id" not in values[0]
    assert values[1][values[0].index("stage_run_id")] == "42"
    assert values[1][values[0].index("upstream_run_ids")] == "7,9"


def test_summary_values_counts_statuses_and_top_reasons():
    workflows = [
        _workflow("seq_a", "wf_a", "FAIL", details="task_failed: export"),
        _workflow("seq_b", "wf_b", "FAIL", details="task_failed: export"),
        _workflow("seq_c", "wf_c", "SKIPPED", details="skipped: no mesh"),
        _workflow("seq_d", "wf_d", "PASS"),
    ]

    values = publish_status.summary_values(
        workflows,
        dataset="dataset_a",
        pipeline_scope="all pipelines",
    )

    assert ["Dataset", "dataset_a"] in values
    assert ["Pipeline scope", "all pipelines"] in values
    assert ["Published rows", "4"] in values
    assert ["Stage", "Status", "Count"] in values
    assert ["mv_hoi_reconstruction", "PASS", "1"] in values
    assert ["mv_hoi_reconstruction", "FAIL", "2"] in values
    assert ["mv_hoi_reconstruction", "SKIPPED", "1"] in values
    assert ["mv_hoi_reconstruction", "task_failed: export", "2"] in values
    assert ["mv_hoi_reconstruction", "skipped: no mesh", "1"] in values


def test_summary_values_keeps_stage_status_counts_separate():
    workflows = [
        _workflow(
            "seq_a",
            "preprocess",
            "PASS",
            pipeline_type="mv_preprocess",
        ),
        _workflow("seq_a", "reconstruction", "PASS"),
        _workflow(
            "seq_a",
            "export",
            "FAIL",
            pipeline_type="mv_export",
            details="upload_failed",
        ),
    ]

    values = publish_status.summary_values(
        workflows,
        dataset="dataset_a",
        pipeline_scope="sequence stages",
    )

    assert ["mv_preprocess", "PASS", "1"] in values
    assert ["mv_hoi_reconstruction", "PASS", "1"] in values
    assert ["mv_export", "FAIL", "1"] in values
    assert ["mv_export", "upload_failed", "1"] in values


def test_resolve_publish_settings_allows_missing_env_in_dry_run(monkeypatch):
    monkeypatch.delenv("MV_HOI_STATUS_SPREADSHEET_ID", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    spreadsheet_id, credentials_path = publish_status.resolve_publish_settings(
        spreadsheet_id=None,
        dry_run=True,
    )

    assert spreadsheet_id == "(unset)"
    assert credentials_path is None


def test_resolve_publish_settings_requires_spreadsheet_id_and_credentials(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("MV_HOI_STATUS_SPREADSHEET_ID", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    with pytest.raises(RuntimeError, match="Missing spreadsheet ID"):
        publish_status.resolve_publish_settings(
            spreadsheet_id=None,
            dry_run=False,
        )

    monkeypatch.setenv("MV_HOI_STATUS_SPREADSHEET_ID", "sheet-id")
    with pytest.raises(RuntimeError, match="Missing Google credentials"):
        publish_status.resolve_publish_settings(
            spreadsheet_id=None,
            dry_run=False,
        )

    missing = tmp_path / "missing.json"
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(missing))
    with pytest.raises(RuntimeError, match="does not exist"):
        publish_status.resolve_publish_settings(
            spreadsheet_id=None,
            dry_run=False,
        )


class _FakeRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class _FakeValues:
    def __init__(self, service):
        self.service = service

    def clear(self, **kwargs):
        self.service.calls.append(("clear", kwargs))
        return _FakeRequest({})

    def update(self, **kwargs):
        self.service.calls.append(("update", kwargs))
        return _FakeRequest({})


class _FakeSpreadsheets:
    def __init__(self, service):
        self.service = service

    def get(self, **kwargs):
        self.service.calls.append(("get", kwargs))
        return _FakeRequest(
            {
                "sheets": [
                    {"properties": {"title": title}}
                    for title in sorted(self.service.titles)
                ]
            }
        )

    def batchUpdate(self, **kwargs):
        self.service.calls.append(("batchUpdate", kwargs))
        for request in kwargs["body"]["requests"]:
            title = request["addSheet"]["properties"]["title"]
            self.service.titles.add(title)
        return _FakeRequest({})

    def values(self):
        return _FakeValues(self.service)


class _FakeSheetsService:
    def __init__(self, titles):
        self.titles = set(titles)
        self.calls = []

    def spreadsheets(self):
        return _FakeSpreadsheets(self)


def test_publish_to_sheets_creates_missing_tabs_and_rewrites_values():
    service = _FakeSheetsService({"latest_status"})

    publish_status.publish_to_sheets(
        service,
        spreadsheet_id="sheet-id",
        status_worksheet="latest_status",
        summary_worksheet="summary",
        status_rows=[["h"], ["v"]],
        summary_rows=[["metric", "value"]],
    )

    assert "summary" in service.titles
    assert any(call[0] == "batchUpdate" for call in service.calls)
    clear_ranges = [
        call[1]["range"] for call in service.calls if call[0] == "clear"
    ]
    assert clear_ranges == ["'latest_status'!A:AZ", "'summary'!A:AZ"]
    update_bodies = [
        call[1]["body"] for call in service.calls if call[0] == "update"
    ]
    assert update_bodies == [
        {"values": [["h"], ["v"]]},
        {"values": [["metric", "value"]]},
    ]


def test_load_status_workflows_can_use_separate_test_database(tmp_path):
    db_path = _db_path(tmp_path)
    test_db_path = str(tmp_path / "processing_test_v2.db")
    db.init_db(test_db_path)
    db.ensure_version_cached("1.0.0", db_path=test_db_path)
    db.insert_workflow(
        sequence_name="seq_prod",
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.0.0",
        workflow_name="wf_prod",
        status="PASS",
        trigger="migration",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="seq_test",
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.0.0",
        workflow_name="wf_test",
        status="SKIPPED",
        trigger="migration",
        db_path=test_db_path,
    )

    workflows = publish_status.load_status_workflows(
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        latest=True,
        db_path=test_db_path,
    )

    assert [workflow["sequence_name"] for workflow in workflows] == ["seq_test"]


def test_load_status_workflows_uses_current_stage_runs_and_filters_sequence_stages(
    monkeypatch, tmp_path,
):
    calls = []
    current = [
        _workflow(
            "seq_a",
            "calibration",
            "PASS",
            pipeline_type="mv_calibration",
        ),
        _workflow(
            "seq_a",
            "preprocess",
            "PASS",
            pipeline_type="mv_preprocess",
        ),
        _workflow("seq_a", "reconstruction", "RUNNING"),
        _workflow(
            "seq_a",
            "export",
            "BLOCKED",
            pipeline_type="mv_export",
        ),
    ]

    def _list_current(dataset, *, stage, db_path, table):
        calls.append((dataset, stage, db_path, table))
        return current

    monkeypatch.setattr(
        publish_status.workflow_db,
        "list_current_stage_runs",
        _list_current,
        raising=False,
    )

    workflows = publish_status.load_status_workflows(
        dataset="dataset_a",
        pipeline_type=None,
        latest=True,
        stages=publish_status.SEQUENCE_STAGES,
        db_path=tmp_path / "status.db",
    )

    assert calls == [(
        "dataset_a", None, str(tmp_path / "status.db"), db.PIPELINES_TABLE,
    )]
    assert [workflow["stage"] for workflow in workflows] == [
        "mv_preprocess",
        "mv_hoi_reconstruction",
        "mv_export",
    ]


def test_main_test_mode_reads_separate_test_database(monkeypatch, tmp_path, capsys):
    db_path = str(tmp_path / "processing_test_v2.db")
    db.init_db(db_path)
    db.ensure_version_cached("1.0.0", db_path=db_path)
    campaign = db.create_campaign(
        name="test-campaign",
        campaign_type="BACKLOG_REPROCESSING",
        dataset="sc_office_4exo_1",
        pipeline_version="1.0.0",
        output_uri="swift://example/data_export_2",
        created_by="test",
        db_path=db_path,
    )
    campaign = db.freeze_campaign(
        campaign["id"],
        inventory_uri="swift://example/inventory.json",
        inventory_sha256="a" * 64,
        inventory_sequence_count=1,
        configuration_uri="swift://example/configuration.json",
        configuration_sha256="b" * 64,
        db_path=db_path,
    )
    db.create_stage_request(
        sequence_name="seq_test",
        dataset="sc_office_4exo_1",
        stage="preprocess",
        pipeline_version="1.0.0",
        campaign=campaign["id"],
        cohort="BULK",
        db_path=db_path,
    )
    monkeypatch.setattr(publish_status.workflow_db, "TEST_DB_PATH", db_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_status.py",
            "--dataset",
            "sc_office_4exo_1",
            "--test",
            "--no-refresh",
            "--dry-run",
        ],
    )

    publish_status.main()

    output = capsys.readouterr().out
    assert "Published data sequence rows: 1" in output
    assert "Published calibration sequence rows: 0" in output
    assert "Reconciled lifecycle total: 1/1" in output


def test_lifecycle_tabs_share_one_reconciled_population():
    snapshot = {
        "observed_at": "2026-07-29T00:00:00.000Z",
        "campaigns": [{
            "id": 10,
            "name": "campaign",
            "campaign_type": "BACKLOG_REPROCESSING",
            "status": "RUNNING",
            "phase": "BULK",
            "inventory_sequence_count": 2,
        }],
        "campaign_memberships": [{"campaign_id": 10, "member_count": 2}],
        "sequences": [
            {
                "id": 101,
                "dataset": "dataset",
                "sequence_name": "pending",
                "campaign_id": 10,
                "campaign_name": "campaign",
                "campaign_type": "BACKLOG_REPROCESSING",
                "campaign_status": "RUNNING",
                "campaign_phase": "BULK",
                "cohort": "BULK",
                "stage": "preprocess",
                "status": "PENDING",
                "pipeline_version": "1.0.0",
            },
            {
                "id": 102,
                "dataset": "dataset",
                "sequence_name": "exported",
                "campaign_id": 10,
                "campaign_name": "campaign",
                "campaign_type": "BACKLOG_REPROCESSING",
                "campaign_status": "RUNNING",
                "campaign_phase": "BULK",
                "cohort": "BULK",
                "stage": "export",
                "status": "SUCCEEDED",
                "pipeline_version": "1.0.0",
                "export_run_id": 502,
                "export_run_status": "SUCCEEDED",
                "export_authorization_type": "QC",
                "export_workflow": "export-workflow",
            },
        ],
    }

    rows = publish_status.build_campaign_lifecycle_rows(snapshot)
    summary = publish_status.build_campaign_lifecycle_summary(snapshot)
    data_values = publish_status.lifecycle_status_values(rows)
    summary_values = publish_status.lifecycle_summary_values(
        summary, dataset="dataset",
    )

    assert len(data_values) - 1 == 2
    assert sum(summary["buckets"].values()) == len(data_values) - 1
    assert summary["buckets"] == {
        "PREPROCESS_PENDING": 1,
        "EXPORTED": 1,
    }
    assert ["Distinct membership", "2"] in summary_values
    assert ["Reconciled lifecycle total", "2"] in summary_values
    assert ["Integrity", "OK"] in summary_values
    assert ["EXPORTED", "1"] in summary_values
    assert ["PREPROCESS_PENDING", "1"] in summary_values
    headers = data_values[0]
    exported = next(
        row for row in data_values[1:]
        if row[headers.index("sequence_name")] == "exported"
    )
    assert exported[headers.index("lifecycle_status")] == "EXPORTED"
    assert exported[headers.index("export_authorization")] == "QC"
    assert exported[headers.index("export_workflow")] == "export-workflow"


def test_default_publisher_preserves_postgresql_url_and_is_read_only(
    monkeypatch, capsys,
):
    database_url = (
        "postgresql+psycopg://operator@example.test:5432/mvhoi?sslmode=require"
    )
    snapshot = {
        "observed_at": "2026-07-29T00:00:00.000Z",
        "campaigns": [],
        "campaign_memberships": [],
        "sequences": [],
    }
    seen = {}
    monkeypatch.setattr(
        publish_status,
        "load_config",
        lambda: {"datasets": {"dataset": {"pipelines": {}}}},
    )
    monkeypatch.setattr(publish_status.workflow_db, "DB_PATH", database_url)
    monkeypatch.setattr(
        publish_status.workflow_db,
        "get_campaign_lifecycle_snapshot",
        lambda dataset, db_path: seen.update(
            snapshot_dataset=dataset, snapshot_db_path=db_path,
        ) or snapshot,
    )

    def _calibration(dataset, *, db_path, sequence_kind):
        seen.update(
            calibration_dataset=dataset,
            calibration_db_path=db_path,
            sequence_kind=sequence_kind,
        )
        return []

    monkeypatch.setattr(
        publish_status.workflow_db, "get_sequence_status", _calibration,
    )
    monkeypatch.setattr(
        publish_status.workflow_db,
        "init_db",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default lifecycle publishing must not initialize the DB")
        ),
    )
    monkeypatch.setattr(
        publish_status,
        "refresh_workflow_states",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("default lifecycle publishing must not refresh OSMO")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["publish_status.py", "--dataset", "dataset", "--dry-run"],
    )

    publish_status.main()

    assert seen == {
        "snapshot_dataset": "dataset",
        "snapshot_db_path": database_url,
        "calibration_dataset": "dataset",
        "calibration_db_path": database_url,
        "sequence_kind": "calibration",
    }
    assert "Published data sequence rows: 0" in capsys.readouterr().out


def test_lifecycle_publication_writes_summary_last():
    service = _FakeSheetsService({
        "summary", "data_sequence_status", "calibration_sequence_status",
    })

    publish_status.publish_current_status_to_sheets(
        service,
        spreadsheet_id="sheet-id",
        summary_worksheet="summary",
        data_worksheet="data_sequence_status",
        calibration_worksheet="calibration_sequence_status",
        summary_rows=[["summary"]],
        data_rows=[["data"]],
        calibration_rows=[["calibration"]],
    )

    update_ranges = [
        call[1]["range"] for call in service.calls if call[0] == "update"
    ]
    assert update_ranges == [
        "'data_sequence_status'!A1",
        "'calibration_sequence_status'!A1",
        "'summary'!A1",
    ]


def test_default_publisher_refuses_unreconciled_membership(monkeypatch):
    snapshot = {
        "observed_at": "2026-07-29T00:00:00.000Z",
        "campaigns": [{
            "id": 10,
            "name": "campaign",
            "campaign_type": "BACKLOG_REPROCESSING",
            "status": "RUNNING",
            "phase": "BULK",
            "inventory_sequence_count": 2,
        }],
        "campaign_memberships": [{"campaign_id": 10, "member_count": 1}],
        "sequences": [{
            "id": 101,
            "dataset": "dataset",
            "sequence_name": "only-member",
            "campaign_id": 10,
            "campaign_name": "campaign",
            "campaign_type": "BACKLOG_REPROCESSING",
            "campaign_status": "RUNNING",
            "campaign_phase": "BULK",
            "stage": "preprocess",
            "status": "PENDING",
        }],
    }
    monkeypatch.setattr(
        publish_status,
        "load_config",
        lambda: {"datasets": {"dataset": {"pipelines": {}}}},
    )
    monkeypatch.setattr(
        publish_status.workflow_db,
        "get_campaign_lifecycle_snapshot",
        lambda dataset, db_path: snapshot,
    )
    monkeypatch.setattr(
        publish_status.workflow_db,
        "get_sequence_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("calibration must not be read after failed reconciliation")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["publish_status.py", "--dataset", "dataset", "--dry-run"],
    )

    with pytest.raises(RuntimeError, match="does not reconcile"):
        publish_status.main()
