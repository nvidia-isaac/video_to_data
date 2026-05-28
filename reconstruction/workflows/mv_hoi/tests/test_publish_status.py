import sys
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import db
import publish_status


def _workflow(
    sequence: str,
    workflow_name: str,
    status: str,
    *,
    pipeline_type: str = "mv_hoi_reconstruction",
    details: str = "",
    created_at: str = "2026-05-01 00:00:00",
) -> dict:
    return {
        "dataset": "dataset_a",
        "sequence_name": sequence,
        "pipeline_type": pipeline_type,
        "status": status,
        "details": details,
        "pipeline_version": "1.0.0",
        "workflow_name": workflow_name,
        "osmo_workflow_id": f"{workflow_name}-1",
        "osmo_export_workflow_id": "",
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
        "1.0.0",
        "wf_a",
        "wf_a-1",
        "",
        "2026-05-01 00:00:00",
        "2026-05-01 00:00:00",
    ]


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
    assert ["PASS", "1"] in values
    assert ["FAIL", "2"] in values
    assert ["SKIPPED", "1"] in values
    assert ["task_failed: export", "2"] in values
    assert ["skipped: no mesh", "1"] in values


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
    assert clear_ranges == ["'latest_status'!A:Z", "'summary'!A:Z"]
    update_bodies = [
        call[1]["body"] for call in service.calls if call[0] == "update"
    ]
    assert update_bodies == [
        {"values": [["h"], ["v"]]},
        {"values": [["metric", "value"]]},
    ]


def test_load_status_workflows_can_use_test_table(tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_prod",
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.0.0",
        workflow_name="wf_prod",
        status="PASS",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="seq_test",
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.0.0",
        workflow_name="wf_test",
        status="SKIPPED",
        db_path=db_path,
        table=db.PIPELINES_TEST_TABLE,
    )

    workflows = publish_status.load_status_workflows(
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        latest=True,
        db_path=db_path,
        table=db.PIPELINES_TEST_TABLE,
    )

    assert [workflow["workflow_name"] for workflow in workflows] == ["wf_test"]


def test_main_test_mode_reads_pipelines_test_table(monkeypatch, tmp_path, capsys):
    db_path = _db_path(tmp_path)
    db.insert_workflow(
        sequence_name="seq_test",
        dataset="sc_office_4exo_1",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.0.0",
        workflow_name="wf_test",
        status="SKIPPED",
        db_path=db_path,
        table=db.PIPELINES_TEST_TABLE,
    )
    monkeypatch.setattr(publish_status, "DB_PATH", Path(db_path))
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
    assert "Published rows: 1" in output
    assert "  SKIPPED: 1" in output
