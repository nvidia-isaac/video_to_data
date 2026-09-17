import io
from http.client import IncompleteRead
import json
from pathlib import Path
import sys

import pytest
from botocore.exceptions import ClientError


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import cleanup_intermediates as cleanup


class MemoryClient:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.gets = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        client = self

        class Paginator:
            def paginate(self, *, Bucket, Prefix):
                assert Bucket == "bucket"
                yield {"Contents": [
                    {"Key": key, "Size": len(value), "ETag": f'"etag-{len(value)}"'}
                    for key, value in sorted(client.objects.items())
                    if key.startswith(Prefix)
                ]}

        return Paginator()

    def get_object(self, *, Bucket, Key):
        assert Bucket == "bucket"
        self.gets.append(Key)
        if Key not in self.objects:
            raise KeyError(Key)
        value = self.objects[Key]
        return {"Body": io.BytesIO(value), "ETag": f'"etag-{len(value)}"'}

    def put_object(self, *, Bucket, Key, Body, **_kwargs):
        assert Bucket == "bucket"
        self.objects[Key] = Body

    def delete_objects(self, *, Bucket, Delete):
        assert Bucket == "bucket"
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
        return {}

    def delete_object(self, *, Bucket, Key):
        assert Bucket == "bucket"
        self.objects.pop(Key, None)


class EventuallyConsistentDeleteClient(MemoryClient):
    def __init__(self, objects):
        super().__init__(objects)
        self.delete_calls = 0

    def delete_objects(self, *, Bucket, Delete):
        self.delete_calls += 1
        if self.delete_calls == 1:
            Delete = {"Objects": Delete["Objects"][1:], "Quiet": True}
        return super().delete_objects(Bucket=Bucket, Delete=Delete)


def _lineage_rows():
    export = {
        "stage_run_id": 30, "id": 30, "request_id": 300,
        "dataset": "dataset", "sequence_name": "sequence",
        "run_status": "SUCCEEDED", "authorization_type": "QC",
        "qc_review_id": 12, "reconstruction_run_id": 20,
        "pipeline_version": "1.2.3", "workflow_name": "export-workflow",
        "output_uri": "swift://host/account/bucket/data_export_2/sequence",
    }
    reconstruction = {
        "stage_run_id": 20, "id": 20, "request_id": 200,
        "run_status": "SUCCEEDED", "preprocess_run_id": 10,
        "pipeline_version": "1.2.3", "workflow_name": "recon-workflow",
        "output_uri": "swift://host/account/bucket/data_output/sequence/reconstruction_200",
        "labeled_bboxes_sha256": "labels-sha",
    }
    preprocess = {
        "stage_run_id": 10, "id": 10, "request_id": 100,
        "run_status": "SUCCEEDED", "pipeline_version": "1.2.3",
        "workflow_name": "preprocess-workflow",
        "output_uri": "swift://host/account/bucket/data_output/sequence",
    }
    request = {
        "id": 300, "stage": "export", "campaign_id": 4,
        "campaign_name": "campaign",
    }
    return export, reconstruction, preprocess, request


def _commit():
    metric_payload = b'{"status":"PASS"}'
    segments = b"[]"
    return {
        "schema": "v2d.mv_hoi.export_commit.v2", "complete": True,
        "sequence_name": "sequence", "reconstruction_run_id": 20,
        "export_validation": {
            "schema": "v2d.mv_hoi.export_validation.v2",
            "camera_counts": {"images": 8, "images_anonymized": 8, "depth": 4},
        },
        "files": [
            {
                "path": "metrics/reconstruction/200/accuracy/check_accuracy.json",
                "size": len(metric_payload), "sha256": cleanup._sha256(metric_payload),
            },
            {
                "path": "failure_segments.json", "size": len(segments),
                "sha256": cleanup._sha256(segments),
            },
            {
                "path": (
                    "metrics/reconstruction/200/failure_segments/"
                    "source_failure_segments.json"
                ),
                "size": len(segments), "sha256": cleanup._sha256(segments),
            },
            {
                "path": (
                    "metrics/reconstruction/200/failure_segments/"
                    "export_failure_segments.json"
                ),
                "size": len(segments), "sha256": cleanup._sha256(segments),
            },
        ],
    }


def _install_lineage(monkeypatch, client):
    export, reconstruction, preprocess, request = _lineage_rows()
    commit = _commit()
    commit_payload = json.dumps(commit, sort_keys=True).encode()
    export_root = "data_export_2/sequence"
    client.objects[f"{export_root}/commit.json"] = commit_payload
    client.objects[f"{export_root}/failure_segments.json"] = b"[]"
    for name in ("source_failure_segments.json", "export_failure_segments.json"):
        client.objects[
            f"{export_root}/metrics/reconstruction/200/failure_segments/{name}"
        ] = b"[]"
    request.update({
        "result_manifest_uri": (
            "swift://host/account/bucket/data_export_2/sequence/commit.json"
        ),
        "result_manifest_sha256": cleanup._sha256(commit_payload),
    })

    def get_stage_run(run_id, *, stage, db_path):
        del db_path
        return {
            (cleanup.db.RECONSTRUCTION_STAGE, 20): reconstruction,
            (cleanup.db.PREPROCESS_STAGE, 10): preprocess,
        }.get((stage, run_id))

    monkeypatch.setattr(cleanup.db, "get_stage_run", get_stage_run)
    monkeypatch.setattr(cleanup.db, "get_stage_request", lambda *_a, **_k: request)
    monkeypatch.setattr(cleanup.db, "list_stage_requests", lambda **_k: [])
    monkeypatch.setattr(
        cleanup.db, "list_stage_run_history", lambda *_a, **_k: [
            {
                **reconstruction, "stage_run_id": 19, "id": 19,
                "output_uri": (
                    "swift://host/account/bucket/data_output_2/sequence/request_19"
                ),
            },
            reconstruction,
        ],
    )
    monkeypatch.setattr(
        cleanup, "_client",
        lambda url: (client, "bucket", url.split("/bucket/", 1)[1].rstrip("/")),
    )
    monkeypatch.setattr(cleanup, "verify_remote_export_commit", lambda _url: commit)
    return export


def test_cleanup_preserves_canonical_assets_and_is_idempotent(tmp_path, monkeypatch):
    del tmp_path
    root = "data_output/sequence"
    metric_payload = b'{"status":"PASS"}'
    protected = {
        f"{root}/mv_preprocess/edex": b"edex",
        f"{root}/mv_preprocess/hoi_metadata.yaml": b"metadata",
        f"{root}/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes\n",
        f"{root}/mv_preprocess/labeled_bboxes/back_stereo_camera_left.json": b"{}",
        f"{root}/mv_preprocess/videos/back_stereo_camera_left.mp4": b"video",
        f"{root}/mv_preprocess/object_mesh/output_aligned.glb": b"mesh",
        (
            f"{root}/reconstruction_200/eval_chamfer_object/chamfer_vis/"
            "tiled_chamfer.mp4"
        ): b"object-chamfer-visualization",
        (
            f"{root}/reconstruction_200/eval_chamfer_human/chamfer_vis/"
            "front_stereo_camera_left.mp4"
        ): b"human-chamfer-visualization",
        (
            f"{root}/reconstruction_200/eval_silhouette_mask_object/"
            "silhouette_mask_vis/tiled_silhouette_mask.mp4"
        ): b"object-silhouette-visualization",
        (
            f"{root}/reconstruction_200/eval_silhouette_mask_human/"
            "silhouette_mask_vis/front_stereo_camera_left.mp4"
        ): b"human-silhouette-visualization",
    }
    heavy = {
        f"{root}/rosbag_to_edex/raw.bin": b"raw" * 10,
        f"{root}/mv_preprocess/images/camera.h5": b"rgb" * 10,
        f"{root}/face_detector/images/camera.h5": b"anon" * 10,
        f"{root}/reconstruction_200/foundation_pose/poses.npy": b"poses" * 10,
    }
    export_metric = (
        "data_export_2/sequence/metrics/reconstruction/200/"
        "accuracy/check_accuracy.json"
    )
    client = MemoryClient({**protected, **heavy, export_metric: metric_payload})
    export = _install_lineage(monkeypatch, client)

    report = cleanup.cleanup_export(export, db_path="db", apply=True)
    assert report["status"] == "COMPLETE"
    assert report["deleted_object_count"] == len(heavy)
    assert report["reclaimed_bytes"] == sum(map(len, heavy.values()))
    assert all(client.objects[key] == value for key, value in protected.items())
    assert not set(heavy) & set(client.objects)
    assert client.objects[
        f"{root}/metrics/reconstruction/200/accuracy/check_accuracy.json"
    ] == metric_payload
    assert f"{root}/{cleanup.COMPLETE_NAME}" in client.objects
    assert f"{root}/{cleanup.PENDING_NAME}" not in client.objects
    assert f"{root}/mv_preprocess/object_bbox_source.txt" in client.gets
    assert f"{root}/mv_preprocess/labeled_bboxes/back_stereo_camera_left.json" in client.gets
    assert f"{root}/mv_preprocess/videos/back_stereo_camera_left.mp4" not in client.gets
    assert f"{root}/mv_preprocess/object_mesh/output_aligned.glb" not in client.gets

    second = cleanup.cleanup_export(export, db_path="db", apply=True)
    assert second["status"] == "ALREADY_COMPLETE"
    assert second["idempotent_noop"] is True
    assert second["deleted_object_count"] == len(heavy)


def test_cleanup_preserves_legacy_flat_metric_visualizations(monkeypatch):
    root = "data_output/sequence"
    visualization = (
        f"{root}/eval_silhouette_mask_object/silhouette_mask_vis/"
        "tiled_silhouette_mask.mp4"
    )
    metric_report = f"{root}/eval_silhouette_mask_object/silhouette_mask_metrics.json"
    client = MemoryClient({visualization: b"visualization", metric_report: b"{}"})
    monkeypatch.setattr(
        cleanup, "_client",
        lambda url: (client, "bucket", url.split("/bucket/", 1)[1].rstrip("/")),
    )
    monkeypatch.setattr(
        cleanup.db, "list_stage_run_history", lambda *_args, **_kwargs: [{
            "stage_run_id": 20,
            "run_status": "SUCCEEDED",
            "output_uri": "swift://host/account/bucket/data_output/sequence",
        }],
    )
    lineage = {
        "export": {"dataset": "dataset", "sequence_name": "sequence"},
        "reconstruction": {"stage_run_id": 20},
        "preprocess": {
            "output_uri": "swift://host/account/bucket/data_output/sequence",
        },
    }

    deletions = cleanup._deletion_objects(
        lineage, db_path="db", canonical_prefix=root,
    )

    assert [item["key"] for item in deletions] == [metric_report]
    assert cleanup._is_protected(
        "eval_silhouette_mask_object/silhouette_mask_vis/"
        "tiled_silhouette_mask.mp4"
    )


def test_cleanup_refuses_active_processing(monkeypatch):
    client = MemoryClient({})
    export = _install_lineage(monkeypatch, client)
    monkeypatch.setattr(cleanup.db, "list_stage_requests", lambda **_k: [{
        "id": 999, "sequence_name": "sequence", "stage": "revalidation",
    }])
    with pytest.raises(ValueError, match="active newer processing"):
        cleanup.cleanup_export(export, db_path="db", apply=False)


def test_backfill_audit_supersedes_stale_legacy_pending_marker(monkeypatch):
    root = "data_output/sequence"
    metric_payload = b'{"status":"PASS"}'
    client = MemoryClient({
        f"{root}/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes\n",
        f"{root}/mv_preprocess/labeled_bboxes/left.json": b"{}",
        f"{root}/mv_preprocess/images/camera.h5": b"heavy",
        (
            "data_export_2/sequence/metrics/reconstruction/200/"
            "accuracy/check_accuracy.json"
        ): metric_payload,
    })
    export = _install_lineage(monkeypatch, client)
    expected = cleanup.cleanup_export(export, db_path="db", apply=False)
    client.objects[f"{root}/{cleanup.PENDING_NAME}"] = json.dumps({
        "schema": cleanup.CLEANUP_SCHEMA, "status": "PENDING",
        "export_run_id": export["stage_run_id"],
        "export_manifest_sha256": "stale-pre-repair-commit",
    }).encode()

    result = cleanup.cleanup_export(
        export, db_path="db", apply=True, expected_report=expected,
    )

    assert result["status"] == "COMPLETE"
    assert f"{root}/mv_preprocess/images/camera.h5" not in client.objects


def test_backfill_audit_accepts_exact_partial_deletion(monkeypatch):
    root = "data_output/sequence"
    metric_payload = b'{"status":"PASS"}'
    client = MemoryClient({
        f"{root}/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes\n",
        f"{root}/mv_preprocess/labeled_bboxes/left.json": b"{}",
        f"{root}/mv_preprocess/images/camera.h5": b"heavy",
        f"{root}/face_detector/camera.h5": b"also-heavy",
        (
            "data_export_2/sequence/metrics/reconstruction/200/"
            "accuracy/check_accuracy.json"
        ): metric_payload,
    })
    export = _install_lineage(monkeypatch, client)
    expected = cleanup.cleanup_export(export, db_path="db", apply=False)
    del client.objects[f"{root}/face_detector/camera.h5"]

    result = cleanup.cleanup_export(
        export, db_path="db", apply=True, expected_report=expected,
    )

    assert result["status"] == "COMPLETE"
    assert f"{root}/mv_preprocess/images/camera.h5" not in client.objects


def test_backfill_audit_refuses_changed_deletion_object(monkeypatch):
    root = "data_output/sequence"
    metric_payload = b'{"status":"PASS"}'
    client = MemoryClient({
        f"{root}/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes\n",
        f"{root}/mv_preprocess/labeled_bboxes/left.json": b"{}",
        f"{root}/mv_preprocess/images/camera.h5": b"heavy",
        (
            "data_export_2/sequence/metrics/reconstruction/200/"
            "accuracy/check_accuracy.json"
        ): metric_payload,
    })
    export = _install_lineage(monkeypatch, client)
    expected = cleanup.cleanup_export(export, db_path="db", apply=False)
    client.objects[f"{root}/mv_preprocess/images/camera.h5"] = b"changed"

    with pytest.raises(ValueError, match="inputs drifted"):
        cleanup.cleanup_export(
            export, db_path="db", apply=True, expected_report=expected,
        )


def test_cleanup_retries_objects_still_visible_after_bulk_delete(monkeypatch):
    root = "data_output/sequence"
    metric_payload = b'{"status":"PASS"}'
    client = EventuallyConsistentDeleteClient({
        f"{root}/mv_preprocess/object_bbox_source.txt": b"manual_labeled_bboxes\n",
        f"{root}/mv_preprocess/labeled_bboxes/left.json": b"{}",
        f"{root}/mv_preprocess/images/camera.h5": b"heavy",
        (
            "data_export_2/sequence/metrics/reconstruction/200/"
            "accuracy/check_accuracy.json"
        ): metric_payload,
    })
    export = _install_lineage(monkeypatch, client)

    result = cleanup.cleanup_export(export, db_path="db", apply=True)

    assert result["status"] == "COMPLETE"
    assert client.delete_calls >= 2


def test_cleanup_requires_validated_ffv1_pairs():
    export, *_ = _lineage_rows()
    commit = _commit()
    commit["export_validation"]["camera_counts"]["depth"] = 3
    with pytest.raises(ValueError, match="required FFV1"):
        cleanup._validate_export_commit(commit, sequence="sequence", export=export)


def test_queued_selection_applies_limit_before_any_css_work(monkeypatch):
    jobs = [
        {"id": value, "dataset": "dataset", "export_run_id": value,
         "sequence_name": f"sequence-{value}"}
        for value in range(1, 5)
    ]
    observed = []
    monkeypatch.setattr(
        cleanup.db, "list_intermediate_cleanup_jobs", lambda **_kwargs: jobs,
    )
    monkeypatch.setattr(
        cleanup.db, "get_stage_run",
        lambda value, **_kwargs: {"stage_run_id": value, "sequence_name": f"sequence-{value}"},
    )
    monkeypatch.setattr(
        cleanup, "cleanup_export",
        lambda export, **_kwargs: observed.append(export["stage_run_id"]) or {
            "export_run_id": export["stage_run_id"], "status": "DRY_RUN",
        },
    )
    results = cleanup.run_cleanup_jobs(
        "dataset", campaign="campaign", db_path="db", apply=False,
        limit=2, workers=2,
    )

    assert observed == [1, 2]
    assert [item["export_run_id"] for item in results] == [1, 2]


def test_postgresql_connection_timeouts_are_retryable():
    postgres_timeout = type(
        "ConnectionTimeout", (Exception,), {"__module__": "psycopg.errors"},
    )("connection timeout expired")
    wrapped = RuntimeError("database read failed")
    wrapped.__cause__ = postgres_timeout

    assert cleanup._is_retryable_cleanup_failure(postgres_timeout)
    assert cleanup._is_retryable_cleanup_failure(wrapped)
    assert not cleanup._is_retryable_cleanup_failure(
        RuntimeError("connection timeout expired")
    )


@pytest.mark.parametrize("code", [
    "TooManyRequests", "SlowDown", "Throttling",
    "RequestLimitExceeded", "429",
])
def test_swift_throttling_cleanup_failures_are_retryable(code):
    error = ClientError(
        {
            "Error": {"Code": code, "Message": "temporary"},
            "ResponseMetadata": {"HTTPStatusCode": 429},
        },
        "ListObjectsV2",
    )
    assert cleanup._is_retryable_cleanup_failure(error)


def test_broken_cleanup_response_streams_are_retryable_when_nested():
    wrapped = RuntimeError("CSS read failed")
    wrapped.__cause__ = IncompleteRead(b"partial", 100)
    assert cleanup._is_retryable_cleanup_failure(wrapped)
    response_stream_error = type(
        "ResponseStreamingError", (Exception,), {"__module__": "botocore.exceptions"},
    )("response stream ended prematurely")
    assert cleanup._is_retryable_cleanup_failure(response_stream_error)
    assert not cleanup._is_retryable_cleanup_failure(
        RuntimeError("Remote committed export key set differs from destination")
    )


def test_cleanup_queue_is_idempotent_reservable_and_recovers_missing_jobs(tmp_path):
    path = str(tmp_path / "queue.db")
    cleanup.db.init_db(path)
    cleanup.db.ensure_version_cached("1.0.0", db_path=path)
    campaign = cleanup.db.create_campaign(
        name="campaign", campaign_type="LEGACY_REVALIDATION",
        dataset="dataset", pipeline_version="1.0.0",
        output_uri="swift://bucket/data_export_2", created_by="operator",
        db_path=path,
    )
    campaign = cleanup.db.freeze_campaign(
        campaign["id"], inventory_uri="swift://bucket/inventory.json",
        inventory_sha256="a" * 64, configuration_uri="swift://bucket/config.json",
        configuration_sha256="b" * 64, db_path=path,
    )
    sequence_id = cleanup.db.upsert_sequence("dataset", "sequence", db_path=path)
    requests = [
        cleanup.db.create_stage_request(
            sequence_name="sequence", dataset="dataset", stage="revalidation",
            pipeline_version="1.0.0", campaign=campaign["id"], cohort="CANARY",
            db_path=path,
        )
    ]
    requests.append(cleanup.db.create_stage_request(
        sequence_name="sequence-2", dataset="dataset", stage="revalidation",
        pipeline_version="1.0.0", campaign=campaign["id"], cohort="CANARY",
        db_path=path,
    ))
    sequence_2 = cleanup.db.upsert_sequence("dataset", "sequence-2", db_path=path)
    execution = cleanup.db.create_workflow_execution(
        workflow_name="cleanup-export", pipeline_type=cleanup.db.EXPORT_STAGE,
        pipeline_version="1.0.0", status="SUCCEEDED", db_path=path,
    )
    now = "2026-08-02T00:00:00+00:00"
    conn = cleanup.db.get_connection(path)
    try:
        conn.execute(
            "UPDATE stage_requests SET status='SUCCEEDED', completed_at=?",
            (now,),
        )
        for run_id, request, seq_id in (
            (100, requests[0], sequence_id), (101, requests[1], sequence_2),
        ):
            conn.execute(
                """INSERT INTO export_runs
                   (id, sequence_id, workflow_execution_id, pipeline_version, status,
                    trigger, output_uri, is_current, created_at, completed_at,
                    updated_at, request_id, authorization_type)
                   VALUES (?, ?, ?, '1.0.0', 'SUCCEEDED', 'AUTO', ?, 0,
                           ?, ?, ?, ?, 'REVALIDATION')""",
                (
                    run_id, seq_id, execution,
                    f"swift://bucket/data_export_2/{request['sequence_name']}",
                    now, now, now, request["id"],
                ),
            )
        conn.commit()
    finally:
        conn.close()

    first = cleanup.db.enqueue_intermediate_cleanup(
        100, source="AUTOMATIC", db_path=path,
    )
    assert cleanup.db.enqueue_intermediate_cleanup(
        100, source="AUTOMATIC", db_path=path,
    )["id"] == first["id"]
    reserved = cleanup.db.reserve_intermediate_cleanup_jobs(
        reserved_by="test", limit=1, campaign=campaign["id"], db_path=path,
    )
    assert [item["export_run_id"] for item in reserved] == [100]
    completed = cleanup.db.finish_intermediate_cleanup_job(
        reserved[0]["id"], status="SUCCEEDED", manifest_uri="swift://manifest",
        manifest_sha256="c" * 64, deleted_object_count=2,
        reclaimed_bytes=32_237_408_713,
        db_path=path,
    )
    assert completed["status"] == "SUCCEEDED"
    assert completed["reclaimed_bytes"] == 32_237_408_713
    assert cleanup.db.reserve_intermediate_cleanup_jobs(
        reserved_by="empty-test", limit=1, campaign=campaign["id"], db_path=path,
    ) == []
    recovered = cleanup.db.enqueue_missing_intermediate_cleanups(
        campaign["id"], completed_after="2026-08-01T00:00:00+00:00",
        db_path=path,
    )
    assert [item["export_run_id"] for item in recovered] == [101]
