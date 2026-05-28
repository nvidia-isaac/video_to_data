import io
import subprocess
import sys
import types
from datetime import datetime
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


class _BodyClient:
    def __init__(self, body: bytes):
        self.body = body
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        return {"Body": io.BytesIO(self.body)}


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
        "mesh_base": "swift://host/AUTH_account/container/mesh",
        "pipelines": {
            "mv_calibration": {
                "input_path": "calibration",
                "output_path": "calibration_output",
                "max_concurrent": 10,
                "workflows": {
                    "calibration": {"workflow_yaml": "osmo/mv_calibration.yaml"},
                },
            },
            "mv_hoi_reconstruction": {
                "input_path": "data",
                "output_path": "data_output",
                "export_path": "data_export",
                "max_concurrent": 10,
                "workflows": {
                    "reconstruction": {
                        "workflow_yaml": "osmo/mv_hoi_reconstruction.yaml",
                        "qc_thresholds": {},
                        "hitl_s3_base": "s3://bucket/path",
                    },
                    "export": {
                        "workflow_yaml": "osmo/mv_hoi_export.yaml",
                        "batch_size": 30,
                        "kratos_table": "catalog.schema.annotations",
                    },
                },
            },
        },
        "osmo_pool": "pool",
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


def test_get_hoi_metadata_parses_yaml_body():
    client = _BodyClient(
        b"calib_seq_name: 2026-01-01_calibration\n"
        b"object:\n"
        b"  id: toy_car\n"
    )

    metadata = submit.get_hoi_metadata(
        client,
        "recordings",
        "v2d/multiview/sc_office_4exo_1/data/seq_a",
    )

    assert metadata == {
        "calib_seq_name": "2026-01-01_calibration",
        "object": {"id": "toy_car"},
    }
    assert client.calls == [
        {
            "Bucket": "recordings",
            "Key": "v2d/multiview/sc_office_4exo_1/data/seq_a/hoi_metadata.yaml",
        }
    ]


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


def test_generate_workflow_name_includes_microseconds(monkeypatch):
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 5, 26, 15, 30, 12, 123456)

    monkeypatch.setattr(submit, "datetime", _FixedDatetime)

    assert (
        submit._generate_workflow_name("mv_hoi_reconstruction", "1.4.20")
        == "v2d_mv_hoi_reconstruction_1-4-20_20260526_153012_123456"
    )


@pytest.mark.parametrize(
    ("reason", "metadata", "calibration_exists", "mesh_url"),
    [
        ("no hoi_metadata.yaml", None, True, "swift://mesh/box/"),
        ("no calib_seq_name in hoi_metadata", {}, True, "swift://mesh/box/"),
        (
            "calibration not found for calib_a",
            {"calib_seq_name": "calib_a", "object_id": "box"},
            False,
            "swift://mesh/box/",
        ),
        (
            "no object_id in hoi_metadata",
            {"calib_seq_name": "calib_a"},
            True,
            "swift://mesh/box/",
        ),
        (
            "no mesh for object box",
            {"calib_seq_name": "calib_a", "object_id": "box"},
            True,
            None,
        ),
    ],
)
def test_submit_sequence_records_reconstruction_prereq_skips(
    monkeypatch, tmp_path, reason, metadata, calibration_exists, mesh_url,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_args: "wf-skipped")
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "get_hoi_metadata",
        lambda *_args, **_kwargs: metadata,
    )
    monkeypatch.setattr(
        submit,
        "path_exists",
        lambda *_args, **_kwargs: calibration_exists,
    )
    monkeypatch.setattr(
        submit,
        "resolve_mesh_url",
        lambda *_args, **_kwargs: mesh_url,
    )
    monkeypatch.setattr(
        submit,
        "osmo_submit",
        lambda *_args, **_kwargs: pytest.fail("OSMO should not be submitted"),
    )

    result = submit.submit_sequence(
        "sequence_a",
        "dataset_a",
        _dataset_cfg(),
        "mv_hoi_reconstruction",
    )

    assert result.prereq_skipped
    assert result.workflow_name == "wf-skipped"
    row = db.get_workflow("wf-skipped", db_path=db_path)
    assert row["sequence_name"] == "sequence_a"
    assert row["status"] == "SKIPPED"
    assert row["details"] == f"skipped: {reason}"
    assert not row["osmo_workflow_id"]


def test_submit_sequence_does_not_duplicate_same_skip_reason(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    workflow_names = iter(["wf-skipped-1", "wf-skipped-2"])

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
    monkeypatch.setattr(submit, "get_hoi_metadata", lambda *_args, **_kwargs: None)

    for _ in range(2):
        result = submit.submit_sequence(
            "sequence_a",
            "dataset_a",
            _dataset_cfg(),
            "mv_hoi_reconstruction",
        )
        assert result.prereq_skipped

    rows = db.get_workflows_by_dataset(
        "dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        db_path=db_path,
    )
    assert [row["workflow_name"] for row in rows] == ["wf-skipped-1"]
    assert rows[0]["details"] == "skipped: no hoi_metadata.yaml"


def test_submit_sequence_records_new_skip_when_reason_changes(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    workflow_names = iter(["wf-skipped-1", "wf-skipped-2"])
    metadata_by_call = iter([None, {}])

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
        "get_hoi_metadata",
        lambda *_args, **_kwargs: next(metadata_by_call),
    )

    for _ in range(2):
        result = submit.submit_sequence(
            "sequence_a",
            "dataset_a",
            _dataset_cfg(),
            "mv_hoi_reconstruction",
        )
        assert result.prereq_skipped

    rows = db.get_workflows_by_dataset(
        "dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        db_path=db_path,
    )
    assert [row["workflow_name"] for row in rows] == [
        "wf-skipped-2",
        "wf-skipped-1",
    ]
    assert rows[0]["details"] == "skipped: no calib_seq_name in hoi_metadata"
    assert rows[1]["details"] == "skipped: no hoi_metadata.yaml"


def test_submit_sequence_submits_after_previous_skip_when_prereqs_pass(
    monkeypatch, tmp_path,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.insert_workflow(
        sequence_name="sequence_a",
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.0.0",
        workflow_name="wf-skipped",
        status="SKIPPED",
        details="skipped: no mesh for object box",
        db_path=db_path,
    )

    monkeypatch.setattr(submit, "DB_PATH", db_path)
    monkeypatch.setattr(submit, "_generate_workflow_name", lambda *_args: "wf-submit")
    monkeypatch.setattr(
        submit,
        "get_s3_client",
        lambda *_args, **_kwargs: (object(), "bucket", "root"),
    )
    monkeypatch.setattr(
        submit,
        "get_hoi_metadata",
        lambda *_args, **_kwargs: {"calib_seq_name": "calib_a", "object_id": "box"},
    )
    monkeypatch.setattr(submit, "path_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        submit,
        "resolve_mesh_url",
        lambda *_args, **_kwargs: "swift://mesh/box/",
    )
    monkeypatch.setattr(submit, "osmo_submit", lambda *_args, **_kwargs: "osmo-submit")

    result = submit.submit_sequence(
        "sequence_a",
        "dataset_a",
        _dataset_cfg(),
        "mv_hoi_reconstruction",
    )

    assert result.workflow_name == "wf-submit"
    latest = db.get_latest_workflow(
        "sequence_a",
        "dataset_a",
        "mv_hoi_reconstruction",
        db_path=db_path,
    )
    assert latest["workflow_name"] == "wf-submit"
    assert latest["status"] == "WAITING_WF"


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


def test_auto_submit_retries_latest_skipped_sequence(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.insert_workflow(
        sequence_name="skipped_sequence",
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.0.0",
        workflow_name="wf-skipped",
        status="SKIPPED",
        details="skipped: no mesh for object box",
        db_path=db_path,
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
        lambda *_args, **_kwargs: ["skipped_sequence"],
    )
    monkeypatch.setattr(
        submit,
        "submit_sequence",
        lambda seq, dataset, *_args, **kwargs: (
            submitted.append((dataset, seq, kwargs["force"])) or f"wf-{seq}"
        ),
    )

    submit.auto_submit("dataset_a", _dataset_cfg(), "mv_hoi_reconstruction")

    assert submitted == [("dataset_a", "skipped_sequence", False)]


def test_auto_submit_counts_prereq_skips_without_per_sequence_logs(
    monkeypatch, tmp_path, capsys,
):
    db_path = _db_path(tmp_path)
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
        lambda *_args, **_kwargs: ["missing_prereq"],
    )

    def _submit_sequence(seq, dataset, *_args, **kwargs):
        assert kwargs["log_prereq_skips"] is False
        submitted.append(seq)
        return submit.SubmitResult("wf-skipped", prereq_skipped=True)

    monkeypatch.setattr(submit, "submit_sequence", _submit_sequence)

    submit.auto_submit("dataset_a", _dataset_cfg(), "mv_hoi_reconstruction")

    output = capsys.readouterr().out
    assert submitted == ["missing_prereq"]
    assert "missing_prereq:" not in output
    assert (
        "Skipped 1 sequence(s) due to unmet prerequisites "
        "(recorded as SKIPPED)."
    ) in output
    assert "Submitted 0 new workflow(s)" in output


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

    assert db.get_workflow("wf-done", db_path=db_path)["status"] == "PASS"
    failed = db.get_workflow("wf-failed", db_path=db_path)
    assert failed["status"] == "FAIL"
    assert failed["details"] == "task_failed: solve_calibration"


def test_refresh_waiting_keeps_reconstruction_completed_in_waiting_qc(
    monkeypatch, tmp_path,
):
    db_path = _db_path(tmp_path)
    db.insert_version("1.0.0", db_path=db_path)
    db.insert_workflow(
        sequence_name="sequence_done",
        dataset="dataset_a",
        pipeline_type="mv_hoi_reconstruction",
        pipeline_version="1.0.0",
        workflow_name="wf-recon-done",
        osmo_workflow_id="osmo-recon-done",
        status="WAITING_WF",
        db_path=db_path,
    )

    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _workflow_id: {"status": "COMPLETED", "tasks": {}},
    )

    query.refresh_waiting(
        "dataset_a", pipeline_type="mv_hoi_reconstruction", db_path=db_path,
    )

    row = db.get_workflow("wf-recon-done", db_path=db_path)
    assert row["status"] == "WAITING_QC"
    assert row["details"] == "workflow_completed"
