import sys
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

import db
import query
import revoke_misaligned_qc_exports as revoke


def _db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "processing.db")
    db.init_db(path)
    db.insert_version("1.0.0", db_path=path)
    return path


def _dataset_cfg() -> dict:
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
                    },
                    "export": {
                        "workflow_yaml": "osmo/mv_hoi_export.yaml",
                        "kratos_table": "catalog.schema.annotations",
                        "max_failure_annotations": 6,
                        "max_failure_coverage": 0.5,
                    },
                },
            }
        },
        "osmo_pool": "pool",
    }


def _config() -> dict:
    return {"datasets": {"sc_office_4exo_1": _dataset_cfg()}}


def _insert_workflow(
    db_path: str,
    sequence: str,
    workflow_name: str,
    *,
    status: str = "PASS",
    details: str = "export_completed",
) -> None:
    db.insert_workflow(
        sequence_name=sequence,
        dataset="sc_office_4exo_1",
        pipeline_type=query.RECON_PIPELINE,
        pipeline_version="1.0.0",
        workflow_name=workflow_name,
        status=status,
        details=details,
        db_path=db_path,
    )


def _coverage_annotations(start_frame=1, end_frame=60) -> list[dict]:
    return [
        {
            "id": "ann",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "failure_category": "bad_pose",
        }
    ]


class _FakePaginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, *, Bucket, Prefix):
        assert Bucket == self.client.bucket
        keys = sorted(key for key in self.client.objects if key.startswith(Prefix))
        if not keys:
            return [{}]
        return [{"Contents": [{"Key": key} for key in keys]}]


class _FakeSwiftClient:
    bucket = "bucket"

    def __init__(self, objects):
        self.objects = set(objects)
        self.deleted_batches = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self)

    def delete_objects(self, *, Bucket, Delete):
        assert Bucket == self.bucket
        keys = [obj["Key"] for obj in Delete["Objects"]]
        self.deleted_batches.append(keys)
        for key in keys:
            self.objects.remove(key)


def _patch_runtime(monkeypatch, client, annotations_by_item, frame_count=100):
    monkeypatch.setattr(revoke, "load_config", _config)
    monkeypatch.setattr(
        revoke,
        "get_s3_client",
        lambda _swift_base: (client, client.bucket, "root"),
    )
    monkeypatch.setattr(
        revoke,
        "query_completed_kratos_annotations",
        lambda *_args, **_kwargs: annotations_by_item,
    )
    monkeypatch.setattr(
        revoke,
        "resolve_frame_count",
        lambda *_args, **_kwargs: frame_count,
    )


def test_dry_run_does_not_delete_update_or_blacklist(monkeypatch, tmp_path):
    db_path = _db_path(tmp_path)
    _insert_workflow(db_path, "seq_bad", "wf_bad")
    client = _FakeSwiftClient(
        [
            "root/data_export/seq_bad/edex",
            "root/data_export/seq_bad/nested/file.txt",
        ]
    )
    _patch_runtime(
        monkeypatch,
        client,
        {"wf_bad.json": _coverage_annotations()},
    )

    results = revoke.run_revoke(
        apply=False,
        db_path=db_path,
        sequences=("seq_bad",),
    )

    assert results[0].eligible
    assert results[0].failure_reason == (
        "qc_fail: failure_coverage>50% (60/100=60.0%)"
    )
    assert client.deleted_batches == []
    assert db.get_workflow("wf_bad", db_path=db_path)["status"] == "PASS"
    assert db.get_blacklisted_sequence(
        "sc_office_4exo_1", "seq_bad", db_path=db_path,
    ) is None


def test_apply_deletes_only_exact_export_prefix_and_marks_fail(
    monkeypatch,
    tmp_path,
):
    db_path = _db_path(tmp_path)
    _insert_workflow(db_path, "seq_bad", "wf_bad")
    db.upsert_blacklisted_sequence(
        "sc_office_4exo_1",
        "seq_bad",
        reason="old reason",
        db_path=db_path,
    )
    client = _FakeSwiftClient(
        [
            "root/data_export/seq_bad/edex",
            "root/data_export/seq_bad/failure_segments.json",
            "root/data_export/seq_bad_nested/keep.txt",
            "root/data_export/other_seq/keep.txt",
        ]
    )
    _patch_runtime(
        monkeypatch,
        client,
        {"wf_bad.json": _coverage_annotations()},
    )

    results = revoke.run_revoke(
        apply=True,
        db_path=db_path,
        sequences=("seq_bad",),
    )

    reason = "qc_fail: failure_coverage>50% (60/100=60.0%)"
    assert results[0].deleted_count == 2
    assert client.objects == {
        "root/data_export/seq_bad_nested/keep.txt",
        "root/data_export/other_seq/keep.txt",
    }
    assert db.get_workflow("wf_bad", db_path=db_path)["status"] == "FAIL"
    assert db.get_workflow("wf_bad", db_path=db_path)["details"] == reason
    assert db.get_blacklisted_sequence(
        "sc_office_4exo_1", "seq_bad", db_path=db_path,
    )["reason"] == reason


def test_apply_skips_non_pass_rows_unless_already_matching_fail(
    monkeypatch,
    tmp_path,
):
    db_path = _db_path(tmp_path)
    matching_reason = "qc_fail: failure_coverage>50% (60/100=60.0%)"
    _insert_workflow(
        db_path,
        "seq_waiting",
        "wf_waiting",
        status="WAITING_QC",
        details="workflow_completed",
    )
    _insert_workflow(
        db_path,
        "seq_failed",
        "wf_failed",
        status="FAIL",
        details=matching_reason,
    )
    client = _FakeSwiftClient(
        [
            "root/data_export/seq_waiting/file.txt",
            "root/data_export/seq_failed/file.txt",
        ]
    )
    _patch_runtime(
        monkeypatch,
        client,
        {
            "wf_waiting.json": _coverage_annotations(),
            "wf_failed.json": _coverage_annotations(),
        },
    )

    results = revoke.run_revoke(
        apply=True,
        db_path=db_path,
        sequences=("seq_waiting", "seq_failed"),
    )

    result_by_sequence = {result.sequence_name: result for result in results}
    assert not result_by_sequence["seq_waiting"].eligible
    assert result_by_sequence["seq_failed"].eligible
    assert client.objects == {"root/data_export/seq_waiting/file.txt"}
    assert db.get_workflow("wf_waiting", db_path=db_path)["status"] == "WAITING_QC"
    assert db.get_blacklisted_sequence(
        "sc_office_4exo_1", "seq_waiting", db_path=db_path,
    ) is None
    assert db.get_workflow("wf_failed", db_path=db_path)["status"] == "FAIL"
    assert db.get_blacklisted_sequence(
        "sc_office_4exo_1", "seq_failed", db_path=db_path,
    )["reason"] == matching_reason


def test_apply_rejects_sequences_that_do_not_exceed_coverage_gate(
    monkeypatch,
    tmp_path,
):
    db_path = _db_path(tmp_path)
    _insert_workflow(db_path, "seq_ok", "wf_ok")
    client = _FakeSwiftClient(["root/data_export/seq_ok/file.txt"])
    _patch_runtime(
        monkeypatch,
        client,
        {"wf_ok.json": _coverage_annotations(start_frame=1, end_frame=25)},
    )

    results = revoke.run_revoke(
        apply=True,
        db_path=db_path,
        sequences=("seq_ok",),
    )

    assert not results[0].eligible
    assert results[0].skip_reason == "corrected QC annotations do not fail export gate"
    assert client.objects == {"root/data_export/seq_ok/file.txt"}
    assert db.get_workflow("wf_ok", db_path=db_path)["status"] == "PASS"
    assert db.get_blacklisted_sequence(
        "sc_office_4exo_1", "seq_ok", db_path=db_path,
    ) is None
