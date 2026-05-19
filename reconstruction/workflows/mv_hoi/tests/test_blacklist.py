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


def test_submit_sequence_force_bypasses_blacklist(monkeypatch, tmp_path):
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

    assert result == "wf-force"
    assert inserted[0]["sequence_name"] == "blocked_sequence"


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
