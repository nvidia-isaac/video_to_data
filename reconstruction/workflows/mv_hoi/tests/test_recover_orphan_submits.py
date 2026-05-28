import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import db
import recover_orphan_submits as recover


def _db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "processing.db")
    db.init_db(path)
    db.insert_version("1.4.14", db_path=path)
    return path


def _candidate(name: str = "v2d_mv_hoi_reconstruction_1-4-14_20260508_002505"):
    return recover.SubmitCandidate(
        sequence_name="2026-03-19_16-33-29_hatchet_examine_04",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.4.14",
        workflow_name=name,
        pool="isaac-dev-h100-01",
        line_no=10,
    )


def _remote(
    name: str = "v2d_mv_hoi_reconstruction_1-4-14_20260508_002505-1",
    status: str = "COMPLETED",
):
    return recover.RemoteWorkflow(
        name=name,
        status=status,
        submit_time="2026-05-08T07:30:40.062340",
        raw={"name": name, "status": status},
    )


def test_parse_submit_log_extracts_candidate(tmp_path):
    log_path = tmp_path / "submit.log"
    log_path.write_text(
        "  Submitting mv_hoi_reconstruction for seq_a (1.4.14)...\n"
        "  CMD: osmo workflow submit workflow.yaml --set "
        'workflow_name="v2d_mv_hoi_reconstruction_1-4-14_20260508_002505" '
        'rosbag_url="swift://example/seq_a/" --pool isaac-dev-h100-01\n',
    )

    candidates = recover.parse_submit_log(
        log_path,
        pipeline_type="mv_hoi_reconstruction",
        since=recover._parse_time_arg("2026-05-08T00:00:00"),
        until=recover._parse_time_arg("2026-05-08T01:00:00"),
    )

    assert candidates == [
        recover.SubmitCandidate(
            sequence_name="seq_a",
            pipeline_type="mv_hoi_reconstruction",
            pipeline_version="1.4.14",
            workflow_name="v2d_mv_hoi_reconstruction_1-4-14_20260508_002505",
            pool="isaac-dev-h100-01",
            line_no=1,
        )
    ]


def test_parse_submit_log_extracts_microsecond_timestamp_candidate(tmp_path):
    log_path = tmp_path / "submit.log"
    workflow_name = "v2d_mv_hoi_reconstruction_1-4-14_20260508_002505_123456"
    log_path.write_text(
        "  Submitting mv_hoi_reconstruction for seq_a (1.4.14)...\n"
        "  CMD: osmo workflow submit workflow.yaml --set "
        f'workflow_name="{workflow_name}" '
        'rosbag_url="swift://example/seq_a/" --pool isaac-dev-h100-01\n',
    )

    assert recover._workflow_timestamp(workflow_name).strftime(
        "%Y-%m-%dT%H:%M:%S"
    ) == "2026-05-08T00:25:05"

    candidates = recover.parse_submit_log(
        log_path,
        pipeline_type="mv_hoi_reconstruction",
        since=recover._parse_time_arg("2026-05-08T00:25:05"),
        until=recover._parse_time_arg("2026-05-08T00:25:06"),
    )

    assert [candidate.workflow_name for candidate in candidates] == [workflow_name]


def test_completed_recovery_uses_normal_completed_details(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    candidate = _candidate()

    monkeypatch.setattr(
        recover,
        "osmo_list_by_name",
        lambda *_args, **_kwargs: [_remote(status="COMPLETED")],
    )

    actions = recover.plan_recovery(
        [candidate],
        dataset="sc_office_4exo_1",
        db_path=db_path,
    )
    assert actions[0].local_status == "WAITING_QC"
    assert actions[0].details == "workflow_completed"

    assert recover.apply_recovery(
        actions,
        dataset="sc_office_4exo_1",
        db_path=db_path,
    ) == 1
    row = db.get_workflow(candidate.workflow_name, db_path=db_path)
    assert row["status"] == "WAITING_QC"
    assert row["details"] == "workflow_completed"
    assert "backfilled" not in row["details"]
    assert row["osmo_workflow_id"] == f"{candidate.workflow_name}-1"
    assert row["created_at"] == "2026-05-08 07:30:40.062340"


def test_failed_recovery_uses_normal_failure_details(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    candidate = _candidate()

    monkeypatch.setattr(
        recover,
        "osmo_list_by_name",
        lambda *_args, **_kwargs: [_remote(status="FAILED")],
    )
    monkeypatch.setattr(
        recover,
        "osmo_query_for_failure",
        lambda _workflow_name: {
            "status": "FAILED",
            "tasks": {
                "postprocess": "FAILED",
                "upload": "FAILED_UPSTREAM",
            },
        },
    )

    actions = recover.plan_recovery(
        [candidate],
        dataset="sc_office_4exo_1",
        db_path=db_path,
    )
    assert actions[0].local_status == "FAIL"
    assert actions[0].details == "task_failed: postprocess"

    recover.apply_recovery(
        actions,
        dataset="sc_office_4exo_1",
        db_path=db_path,
    )
    row = db.get_workflow(candidate.workflow_name, db_path=db_path)
    assert row["status"] == "FAIL"
    assert row["details"] == "task_failed: postprocess"
    assert "backfilled" not in row["details"]


def test_existing_and_ambiguous_remote_matches_are_not_inserted(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    tracked = _candidate("v2d_mv_hoi_reconstruction_1-4-14_20260508_003014")
    db.insert_workflow(
        sequence_name=tracked.sequence_name,
        dataset="sc_office_4exo_1",
        pipeline_type=tracked.pipeline_type,
        pipeline_version=tracked.pipeline_version,
        workflow_name=tracked.workflow_name,
        status="WAITING_WF",
        db_path=db_path,
    )

    missing = _candidate("v2d_mv_hoi_reconstruction_1-4-14_20260508_003116")
    multiple = _candidate("v2d_mv_hoi_reconstruction_1-4-14_20260508_003218")

    def fake_list(workflow_name, *_args, **_kwargs):
        if workflow_name == missing.workflow_name:
            return []
        return [
            _remote(name=f"{workflow_name}-1"),
            _remote(name=f"{workflow_name}-2"),
        ]

    monkeypatch.setattr(recover, "osmo_list_by_name", fake_list)

    actions = recover.plan_recovery(
        [tracked, missing, multiple],
        dataset="sc_office_4exo_1",
        db_path=db_path,
    )

    assert [action.action for action in actions] == [
        "ALREADY_TRACKED",
        "NOT_FOUND",
        "MULTIPLE_MATCHES",
    ]
    assert recover.apply_recovery(
        actions,
        dataset="sc_office_4exo_1",
        db_path=db_path,
    ) == 0
