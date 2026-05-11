import subprocess
import sys
import types
from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

try:
    import boto3  # noqa: F401
except ModuleNotFoundError:
    sys.modules["boto3"] = types.SimpleNamespace(client=lambda *args, **kwargs: None)

import db
import query
import submit


def _called_process_error(
    returncode: int,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(
        returncode,
        "osmo workflow submit",
        output=stdout,
        stderr=stderr,
    )


def _db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "processing.db")
    db.init_db(path)
    return path


def _dataset_cfg() -> dict:
    return {
        "swift_base": "swift://host/AUTH_account/container/root",
        "calibration_path": "calibration",
        "calibration_output_path": "calibration_output",
        "data_path": "data",
        "data_output_path": "data_output",
        "mesh_base": "swift://host/AUTH_account/container/mesh",
        "pipelines": {
            "mv_calibration": {"workflow_yaml": "osmo/mv_calibration.yaml"},
            "mv_hoi_reconstruction": {
                "workflow_yaml": "osmo/mv_hoi_reconstruction.yaml",
            },
        },
        "osmo_pool": "pool",
        "max_concurrent": 10,
        "hitl_s3_base": "s3://bucket/path",
    }


def test_blacklist_helpers_are_dataset_scoped_and_removable(tmp_path):
    db_path = _db_path(tmp_path)

    db.upsert_blacklisted_sequence(
        "dataset_a", "shared_sequence", reason="bad capture", db_path=db_path,
    )

    assert db.is_sequence_blacklisted(
        "dataset_a", "shared_sequence", db_path=db_path,
    )
    assert not db.is_sequence_blacklisted(
        "dataset_b", "shared_sequence", db_path=db_path,
    )

    first_entry = db.get_blacklisted_sequence(
        "dataset_a", "shared_sequence", db_path=db_path,
    )
    assert first_entry["blacklisted_at"]

    db.upsert_blacklisted_sequence(
        "dataset_a", "shared_sequence", reason="updated reason", db_path=db_path,
    )
    updated_entry = db.get_blacklisted_sequence(
        "dataset_a", "shared_sequence", db_path=db_path,
    )
    assert updated_entry["reason"] == "updated reason"
    assert updated_entry["blacklisted_at"] == first_entry["blacklisted_at"]
    assert db.get_blacklisted_sequences("dataset_a", db_path=db_path) == [
        {
            "dataset": "dataset_a",
            "sequence_name": "shared_sequence",
            "reason": "updated reason",
            "blacklisted_at": first_entry["blacklisted_at"],
        }
    ]

    assert db.remove_blacklisted_sequence(
        "dataset_a", "shared_sequence", db_path=db_path,
    )
    assert not db.is_sequence_blacklisted(
        "dataset_a", "shared_sequence", db_path=db_path,
    )
    assert not db.remove_blacklisted_sequence(
        "dataset_a", "shared_sequence", db_path=db_path,
    )


def test_submit_error_classification_marks_timeouts_ambiguous():
    error = _called_process_error(
        10,
        stdout=(
            "Error message:\n"
            "Cannot connect to OSMO service, with error:\n"
            "HTTPSConnectionPool(host='us-west-2-aws.osmo.nvidia.com'): "
            "Read timed out. (read timeout=60)\n"
            "Error code: 1\n"
        ),
    )

    assert submit._is_ambiguous_submit_error(error)
    assert "Cannot connect to OSMO service" in submit._short_submit_error(error)


def test_submit_error_classification_ignores_validation_failure():
    error = _called_process_error(
        2,
        stderr="invalid choice: --bad-flag",
    )

    assert not submit._is_ambiguous_submit_error(error)


def test_submit_sequence_skips_blacklisted_manual_sequence(
    monkeypatch, tmp_path, capsys,
):
    db_path = _db_path(tmp_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "blocked_sequence", reason="bad capture", db_path=db_path,
    )
    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: pytest.fail("Swift should not be touched"),
    )

    result = submit.submit_sequence(
        "blocked_sequence",
        "dataset_a",
        _dataset_cfg(),
        "mv_calibration",
    )

    assert result is None
    assert "blacklisted for dataset_a: bad capture" in capsys.readouterr().out


def test_submit_sequence_force_removes_blacklist(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "blocked_sequence", reason="bad capture", db_path=db_path,
    )
    inserted = []

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_args: "wf-force")
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "osmo_submit",
        lambda *_args, **_kwargs: "osmo-force",
    )
    monkeypatch.setattr(
        submit,
        "insert_workflow",
        lambda **kwargs: inserted.append(kwargs),
    )

    result = submit.submit_sequence(
        "blocked_sequence",
        "dataset_a",
        _dataset_cfg(),
        "mv_calibration",
        force=True,
    )

    assert result.workflow_name == "wf-force"
    assert not result.ambiguous
    assert inserted[0]["sequence_name"] == "blocked_sequence"
    assert db.get_blacklisted_sequence(
        "dataset_a", "blocked_sequence", db_path=db_path,
    ) is None


def test_submit_sequence_force_dry_run_preserves_blacklist(
    monkeypatch, tmp_path, capsys,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "blocked_sequence", reason="bad capture", db_path=db_path,
    )

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_args: "wf-force")
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )

    result = submit.submit_sequence(
        "blocked_sequence",
        "dataset_a",
        _dataset_cfg(),
        "mv_calibration",
        force=True,
        dry_run=True,
    )

    assert result.workflow_name == "wf-force"
    assert not result.ambiguous
    assert db.get_blacklisted_sequence(
        "dataset_a", "blocked_sequence", db_path=db_path,
    ) is not None
    assert (
        "[dry-run] would remove blacklist entry for blocked_sequence due to --force"
        in capsys.readouterr().out
    )


def test_submit_sequence_ambiguous_submit_records_waiting_placeholder(
    monkeypatch, tmp_path,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    error = _called_process_error(
        10,
        stdout=(
            "Error message:\n"
            "Cannot connect to OSMO service, with error:\n"
            "HTTPSConnectionPool: Read timed out. (read timeout=60)\n"
            "Error code: 1\n"
        ),
    )

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_args: "wf-ambiguous")
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "osmo_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            submit.AmbiguousSubmitError(error)
        ),
    )

    result = submit.submit_sequence(
        "maybe_submitted",
        "dataset_a",
        _dataset_cfg(),
        "mv_calibration",
    )

    assert result.workflow_name == "wf-ambiguous"
    assert result.ambiguous
    row = db.get_workflow("wf-ambiguous", db_path=db_path)
    assert row["sequence_name"] == "maybe_submitted"
    assert row["status"] == "WAITING_WF"
    assert row["osmo_workflow_id"] == "wf-ambiguous-1"
    assert row["details"].startswith("submit_ambiguous: exit 10:")


def test_submit_sequence_ambiguous_force_preserves_blacklist(
    monkeypatch, tmp_path,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "blocked_sequence", reason="bad capture", db_path=db_path,
    )
    error = _called_process_error(10, stdout="Read timed out. (read timeout=60)")

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_args: "wf-ambiguous")
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "osmo_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            submit.AmbiguousSubmitError(error)
        ),
    )

    result = submit.submit_sequence(
        "blocked_sequence",
        "dataset_a",
        _dataset_cfg(),
        "mv_calibration",
        force=True,
    )

    assert result.ambiguous
    assert db.get_blacklisted_sequence(
        "dataset_a", "blocked_sequence", db_path=db_path,
    ) is not None


def test_submit_sequence_non_ambiguous_submit_error_records_nothing(
    monkeypatch, tmp_path,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    error = _called_process_error(2, stderr="invalid choice: --bad-flag")

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_args: "wf-failed")
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "osmo_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    result = submit.submit_sequence(
        "bad_submit",
        "dataset_a",
        _dataset_cfg(),
        "mv_calibration",
    )

    assert result is None
    assert db.get_workflow("wf-failed", db_path=db_path) is None


def test_auto_submit_filters_blacklist_by_active_dataset(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "shared_sequence", reason="bad for a", db_path=db_path,
    )
    db.upsert_blacklisted_sequence(
        "dataset_b", "blocked_b", reason="bad for b", db_path=db_path,
    )
    sequences = []
    submitted = []

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "list_sequences",
        lambda *_args, **_kwargs: list(sequences),
    )
    monkeypatch.setattr(
        submit,
        "submit_sequence",
        lambda seq, dataset, *_args, **kwargs: (
            submitted.append((dataset, seq, kwargs["force"])) or f"wf-{seq}"
        ),
    )

    sequences[:] = ["shared_sequence", "open_sequence"]
    submit.auto_submit("dataset_a", _dataset_cfg(), "mv_calibration")
    assert submitted == [("dataset_a", "open_sequence", False)]

    submitted.clear()
    sequences[:] = ["shared_sequence", "blocked_b"]
    submit.auto_submit("dataset_b", _dataset_cfg(), "mv_calibration")
    assert submitted == [("dataset_b", "shared_sequence", False)]


def test_auto_submit_force_bypasses_blacklist(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "blocked_sequence", reason="bad capture", db_path=db_path,
    )
    submitted = []

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "list_sequences",
        lambda *_args, **_kwargs: ["blocked_sequence"],
    )
    monkeypatch.setattr(
        submit,
        "submit_sequence",
        lambda seq, dataset, *_args, **kwargs: (
            submitted.append((dataset, seq, kwargs["force"])) or f"wf-{seq}"
        ),
    )

    submit.auto_submit(
        "dataset_a", _dataset_cfg(), "mv_calibration", force=True,
    )

    assert submitted == [("dataset_a", "blocked_sequence", True)]


def test_auto_submit_stops_after_ambiguous_submit(monkeypatch, tmp_path, capsys):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    workflow_names = iter(["wf-first", "wf-second"])
    attempted = []
    error = _called_process_error(10, stdout="Read timed out. (read timeout=60)")

    def fake_osmo_submit(_workflow_yaml, _pool, set_vars, **_kwargs):
        attempted.append(set_vars["workflow_name"])
        raise submit.AmbiguousSubmitError(error)

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(
        submit,
        "_generate_workflow_name",
        lambda *_args: next(workflow_names),
    )
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "list_sequences",
        lambda *_args, **_kwargs: ["first_sequence", "second_sequence"],
    )
    monkeypatch.setattr(submit, "osmo_submit", fake_osmo_submit)

    submit.auto_submit("dataset_a", _dataset_cfg(), "mv_calibration")

    assert attempted == ["wf-first"]
    assert db.get_workflow("wf-first", db_path=db_path)["status"] == "WAITING_WF"
    assert db.get_workflow("wf-second", db_path=db_path) is None
    assert (
        "Ambiguous OSMO submit recorded; stopping auto-submit"
        in capsys.readouterr().out
    )


def test_auto_submit_force_removes_blacklist_for_submitted_sequence(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.upsert_blacklisted_sequence(
        "dataset_a", "blocked_sequence", reason="bad capture", db_path=db_path,
    )

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_args: "wf-force")
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "list_sequences",
        lambda *_args, **_kwargs: ["blocked_sequence"],
    )
    monkeypatch.setattr(
        submit,
        "osmo_submit",
        lambda *_args, **_kwargs: "osmo-force",
    )

    submit.auto_submit(
        "dataset_a", _dataset_cfg(), "mv_calibration", force=True,
    )

    assert db.get_blacklisted_sequence(
        "dataset_a", "blocked_sequence", db_path=db_path,
    ) is None
    latest = db.get_latest_workflow(
        "blocked_sequence", "dataset_a", "mv_calibration", db_path=db_path,
    )
    assert latest["workflow_name"] == "wf-force"


class _FakeMeshClient:
    def __init__(self, keys):
        self.keys = keys

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys):
        contents = [
            {"Key": key}
            for key in self.keys
            if key.startswith(Prefix)
        ][:MaxKeys]
        return {"Contents": contents}


def test_resolve_mesh_url_requires_aligned_mesh():
    client = _FakeMeshClient([
        "mesh/mug/einstar/output.glb",
        "mesh/mug/bundlesdf/output_aligned.glb",
    ])

    assert submit.resolve_mesh_url(
        client,
        "container",
        "mesh",
        "mug",
        "swift://host/AUTH_account/container/mesh",
    ) == "swift://host/AUTH_account/container/mesh/mug/bundlesdf/"


def test_resolve_mesh_url_ignores_output_aligned_prefix_without_exact_file():
    client = _FakeMeshClient([
        "mesh/mug/einstar/output_aligned.glb.bak",
        "mesh/mug/bundlesdf/output.glb",
    ])

    assert submit.resolve_mesh_url(
        client,
        "container",
        "mesh",
        "mug",
        "swift://host/AUTH_account/container/mesh",
    ) is None


def test_refresh_waiting_auto_blacklists_repeated_failure(
    monkeypatch, tmp_path, capsys,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.insert_workflow(
        sequence_name="sequence_a",
        dataset="dataset_a",
        pipeline_type="mv_calibration",
        pipeline_version="1.0.0",
        workflow_name="wf-old",
        status="FAIL",
        details="task_failed: solve_calibration",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="sequence_a",
        dataset="dataset_a",
        pipeline_type="mv_calibration",
        pipeline_version="1.0.0",
        workflow_name="wf-new",
        osmo_workflow_id="osmo-new",
        status="WAITING_WF",
        db_path=db_path,
    )
    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _workflow_id: {
            "status": "FAILED",
            "tasks": {"solve_calibration": "FAILED"},
        },
    )

    query.refresh_waiting(
        "dataset_a", pipeline_type="mv_calibration", db_path=db_path,
    )

    entry = db.get_blacklisted_sequence(
        "dataset_a", "sequence_a", db_path=db_path,
    )
    assert entry["reason"] == "task_failed: solve_calibration"
    assert (
        "Auto-blacklisted dataset_a/sequence_a after 2 recent "
        "mv_calibration failures: task_failed: solve_calibration"
    ) in capsys.readouterr().out


def test_refresh_waiting_does_not_blacklist_different_failure_details(
    monkeypatch, tmp_path, capsys,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.insert_workflow(
        sequence_name="sequence_a",
        dataset="dataset_a",
        pipeline_type="mv_calibration",
        pipeline_version="1.0.0",
        workflow_name="wf-old",
        status="FAIL",
        details="task_failed: collect_frames",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="sequence_a",
        dataset="dataset_a",
        pipeline_type="mv_calibration",
        pipeline_version="1.0.0",
        workflow_name="wf-new",
        osmo_workflow_id="osmo-new",
        status="WAITING_WF",
        db_path=db_path,
    )
    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _workflow_id: {
            "status": "FAILED",
            "tasks": {"solve_calibration": "FAILED"},
        },
    )

    query.refresh_waiting(
        "dataset_a", pipeline_type="mv_calibration", db_path=db_path,
    )

    assert db.get_blacklisted_sequence(
        "dataset_a", "sequence_a", db_path=db_path,
    ) is None
    assert "Auto-blacklisted" not in capsys.readouterr().out


def test_refresh_waiting_updates_multiple_workflows(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.insert_workflow(
        sequence_name="sequence_done",
        dataset="dataset_a",
        pipeline_type="mv_calibration",
        pipeline_version="1.0.0",
        workflow_name="wf-done",
        osmo_workflow_id="osmo-done",
        status="WAITING_WF",
        db_path=db_path,
    )
    db.insert_workflow(
        sequence_name="sequence_failed",
        dataset="dataset_a",
        pipeline_type="mv_calibration",
        pipeline_version="1.0.0",
        workflow_name="wf-failed",
        osmo_workflow_id="osmo-failed",
        status="WAITING_WF",
        db_path=db_path,
    )

    def fake_osmo_query(workflow_id):
        if workflow_id == "osmo-done":
            return {"status": "COMPLETED", "tasks": {}}
        return {"status": "FAILED", "tasks": {"solve_calibration": "FAILED"}}

    monkeypatch.setattr(query, "osmo_query", fake_osmo_query)

    query.refresh_waiting(
        "dataset_a", pipeline_type="mv_calibration", db_path=db_path,
        max_workers=2,
    )

    assert db.get_workflow("wf-done", db_path=db_path)["status"] == "WAITING_QC"
    failed = db.get_workflow("wf-failed", db_path=db_path)
    assert failed["status"] == "FAIL"
    assert failed["details"] == "task_failed: solve_calibration"
