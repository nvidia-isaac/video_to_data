import hashlib
import io
import json
from pathlib import Path
import sys
import threading
import time

from botocore.exceptions import ClientError, ResponseStreamingError
import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import campaign_controller as controller
from orchestration import db


class _HeadClient:
    def __init__(self, objects):
        self.objects = objects

    def head_object(self, *, Bucket, Key):
        del Bucket
        value = self.objects[Key]
        return {"ContentLength": value["size"], "ETag": f'"{value["etag"]}"'}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        objects = self.objects

        class _Paginator:
            def paginate(self, *, Bucket, Prefix):
                del Bucket
                yield {"Contents": [
                    {
                        "Key": key,
                        "Size": value["size"],
                        "ETag": f'"{value["etag"]}"',
                    }
                    for key, value in objects.items()
                    if key.startswith(Prefix)
                ]}

        return _Paginator()


class _RangedGetFallbackClient:
    def get_paginator(self, operation):
        assert operation == "list_objects_v2"

        class _Paginator:
            def paginate(self, *, Bucket, Prefix):
                del Bucket
                relative = "source.bin" if "/data_output/" in Prefix else "export.bin"
                yield {"Contents": [{
                    "Key": Prefix.rstrip("/") + "/" + relative,
                    "Size": 10,
                    "ETag": '"stale"',
                }]}

        return _Paginator()

    def head_object(self, *, Bucket, Key):
        del Bucket, Key
        raise ClientError(
            {
                "Error": {"Code": "BadRequest", "Message": "Bad Request"},
                "ResponseMetadata": {"HTTPStatusCode": 400},
            },
            "HeadObject",
        )

    def get_object(self, *, Bucket, Key, Range):
        del Bucket, Key
        assert Range == "bytes=0-0"
        return {
            "ContentRange": "bytes 0-0/10",
            "ContentLength": 1,
            "ETag": '"pose"',
            "Body": io.BytesIO(b"x"),
        }


def test_revalidation_publication_detects_active_repair_marker(monkeypatch):
    class Client:
        def get_object(self, *, Bucket, Key):
            assert Bucket == "bucket"
            assert Key.endswith("/request_7/pending.json")
            return {"Body": io.BytesIO(b"{}")}

    monkeypatch.setattr(
        controller, "_client", lambda _url: (Client(), "bucket", _url),
    )
    assert controller._revalidation_repair_active(
        {"id": 7}, {"name": "campaign"},
        {"swift_base": "swift://host/account/bucket/dataset"},
    ) is True


def test_reconcile_can_select_only_recoverable_publication_failures(monkeypatch):
    request = {
        "id": 7,
        "workflow_execution_id": 11,
        "details": "export_run_commit_failed: transient NoSuchKey",
    }
    statuses = []
    monkeypatch.setattr(
        controller.db, "get_campaign",
        lambda *_args, **_kwargs: {"dataset": "dataset"},
    )
    monkeypatch.setattr(
        controller, "load_config",
        lambda *_args, **_kwargs: {"datasets": {"dataset": {}}},
    )

    def list_requests(*_args, status, **_kwargs):
        statuses.append(status)
        return [request] if status == "FAILED" else []

    monkeypatch.setattr(controller.db, "list_stage_requests", list_requests)
    monkeypatch.setattr(
        controller.db, "get_workflow_execution",
        lambda *_args, **_kwargs: {
            "id": 11, "workflow_name": "workflow", "osmo_workflow_id": "workflow-1",
        },
    )
    monkeypatch.setattr(
        controller, "osmo_query", lambda *_args, **_kwargs: {
            "status": "UNKNOWN", "error": "capacity query timed out",
            "not_found": False,
        },
    )
    observed = []
    monkeypatch.setattr(
        controller.db, "apply_execution_observation",
        lambda *args, **kwargs: observed.append((args, kwargs)),
    )

    result = controller.reconcile_revalidation(
        "campaign", db_path="db", recover_publication_failures=True,
    )

    assert statuses == [("SUBMITTED", "RUNNING", "UNKNOWN"), "FAILED"]
    assert result == [{"request_id": 7, "status": "UNKNOWN"}]
    assert observed[0][1]["request_outcomes"] == [
        {"request_id": 7, "status": "UNKNOWN"},
    ]


def test_pending_revalidation_publication_caps_aggregate_copy_workers(monkeypatch):
    requests = [{
        "id": request_id,
        "workflow_execution_id": 100 + request_id,
        "details": controller.REVALIDATION_RESULT_PENDING,
    } for request_id in (1, 2)]
    monkeypatch.setattr(
        controller.db, "get_campaign",
        lambda *_a, **_k: {"dataset": "dataset"},
    )
    monkeypatch.setattr(
        controller, "load_config",
        lambda *_a, **_k: {"datasets": {"dataset": {}}},
    )
    monkeypatch.setattr(
        controller.db, "list_stage_requests",
        lambda *_a, **_k: list(requests),
    )
    monkeypatch.setattr(
        controller.db, "get_workflow_execution",
        lambda execution_id, **_k: {
            "id": execution_id,
            "last_query_payload_json": json.dumps({"status": "COMPLETED"}),
        },
    )
    copy_workers = []

    def handle(request, *_args, publication_copy_workers, **_kwargs):
        copy_workers.append(publication_copy_workers)
        return {"request_id": request["id"], "status": "SUCCEEDED"}

    monkeypatch.setattr(controller, "_handle_revalidation_observation", handle)

    result = controller.reconcile_revalidation(
        "campaign", db_path="db", pending_publication_only=True,
        publish_workers=128,
    )

    assert [item["request_id"] for item in result] == [1, 2]
    assert copy_workers == [1, 1]


def test_bulk_revalidation_observation_is_parallel_and_defers_publication(
    monkeypatch,
):
    requests = [{
        "id": request_id,
        "workflow_execution_id": 100 + request_id,
        "cohort": "BULK",
        "details": "workflow_running",
        "campaign_name": "campaign",
        "campaign_type": "LEGACY_REVALIDATION",
        "dataset": "dataset",
        "sequence_name": f"sequence-{request_id}",
    } for request_id in range(1, 5)]
    monkeypatch.setattr(
        controller.db, "get_campaign",
        lambda *_a, **_k: {"dataset": "dataset", "phase": "BULK"},
    )
    monkeypatch.setattr(
        controller, "load_config",
        lambda *_a, **_k: {"datasets": {"dataset": {}}},
    )
    monkeypatch.setattr(
        controller.db, "list_stage_requests",
        lambda *_a, **_k: list(requests),
    )
    monkeypatch.setattr(
        controller.db, "get_workflow_execution",
        lambda execution_id, **_k: {
            "id": execution_id, "status": "RUNNING",
            "workflow_name": f"workflow-{execution_id}",
        },
    )
    monkeypatch.setattr(
        controller, "_revalidation_repair_active", lambda *_a, **_k: False,
    )
    lock = threading.Lock()
    active = 0
    maximum = 0

    def query(_workflow):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return {"status": "COMPLETED", "task_details": {}}

    monkeypatch.setattr(controller, "osmo_query", query)
    observations = []
    monkeypatch.setattr(
        controller.db, "apply_execution_observation",
        lambda *args, **kwargs: observations.append((args, kwargs)),
    )
    monkeypatch.setattr(
        controller, "validate_frozen_objects",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("deferred observation must not validate payloads")
        ),
    )

    results = controller.reconcile_revalidation(
        "campaign", db_path="db", query_workers=4,
        defer_bulk_publication=True,
    )

    assert maximum > 1
    assert [item["status"] for item in results] == ["WAITING_EXPORT"] * 4
    assert len(observations) == 4
    for _args, kwargs in observations:
        assert kwargs["execution_status"] == "SUCCEEDED"
        assert kwargs["request_outcomes"][0]["status"] == "RUNNING"
        assert kwargs["request_outcomes"][0]["details"].startswith(
            controller.REVALIDATION_RESULT_PENDING
        )


def test_read_commit_retries_transient_missing_commit(monkeypatch):
    commit = {
        "files": [],
        "complete": True,
    }

    class Client:
        calls = 0

        def get_object(self, *, Bucket, Key):
            del Bucket, Key
            self.calls += 1
            if self.calls == 1:
                raise ClientError(
                    {"Error": {"Code": "NoSuchKey", "Message": "missing"}},
                    "GetObject",
                )
            return {"Body": io.BytesIO(json.dumps(commit).encode())}

        def get_paginator(self, operation):
            assert operation == "list_objects_v2"

            class Paginator:
                def paginate(self, **_kwargs):
                    yield {"Contents": [{"Key": "root/commit.json", "Size": 1}]}

            return Paginator()

    client = Client()
    monkeypatch.setattr(
        controller, "_client", lambda *_args: (client, "bucket", "root"),
    )
    monkeypatch.setattr(controller.time, "sleep", lambda *_args: None)

    observed, _digest = controller._read_commit("swift://result")

    assert observed == commit
    assert client.calls == 2


class _SequencedHeadClient(_HeadClient):
    def __init__(self, objects, responses):
        super().__init__(objects)
        self.responses = list(responses)
        self.calls = 0

    def head_object(self, *, Bucket, Key):
        if self.responses:
            value = self.responses.pop(0)
            self.calls += 1
            return {"ContentLength": value["size"], "ETag": f'"{value["etag"]}"'}
        self.calls += 1
        return super().head_object(Bucket=Bucket, Key=Key)

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        objects = self.objects

        class _Paginator:
            def paginate(self, *, Bucket, Prefix):
                del Bucket
                contents = []
                for key, value in objects.items():
                    if not key.startswith(Prefix):
                        continue
                    etag = "stale" if key.endswith("source.bin") else value["etag"]
                    contents.append({
                        "Key": key, "Size": value["size"], "ETag": f'"{etag}"',
                    })
                yield {"Contents": contents}

        return _Paginator()


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_commit_retries_until_css_listing_converges(monkeypatch):
    commit = {
        "files": [{"path": "payload.bin", "size": 3, "sha256": "unused"}],
    }
    payload = json.dumps(commit).encode()

    class Client:
        def get_object(self, *, Bucket, Key):
            assert Bucket == "bucket"
            assert Key == "prefix/commit.json"
            return {"Body": io.BytesIO(payload)}

    listings = [
        {"commit.json": {"size": len(payload)}},
        {
            "commit.json": {"size": len(payload)},
            "payload.bin": {"size": 3},
        },
    ]
    sleeps = []
    monkeypatch.setattr(
        controller, "_client", lambda _url: (Client(), "bucket", "prefix"),
    )
    monkeypatch.setattr(
        controller, "_object_map", lambda *_args: listings.pop(0),
    )
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)

    observed, digest = controller._read_commit("swift://host/account/bucket/prefix")

    assert observed == commit
    assert digest == hashlib.sha256(payload).hexdigest()
    assert sleeps == [controller.COMMIT_VISIBILITY_BASE_DELAY_SECONDS]


def test_read_commit_retries_truncated_payload_hash_from_byte_zero(monkeypatch):
    content = b"payload"
    commit = {
        "files": [{
            "path": "payload.bin",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }],
    }
    commit_payload = json.dumps(commit).encode()

    class TruncatedBody(io.BytesIO):
        def read(self, *_args, **_kwargs):
            raise ResponseStreamingError(error=RuntimeError("truncated"))

    class Client:
        payload_reads = 0
        ranges = []

        def get_object(self, *, Bucket, Key, Range=None):
            assert Bucket == "bucket"
            if Key == "prefix/commit.json":
                assert Range is None
                return {"Body": io.BytesIO(commit_payload)}
            assert Key == "prefix/payload.bin"
            self.ranges.append(Range)
            first, last = map(int, Range.removeprefix("bytes=").split("-"))
            self.payload_reads += 1
            selected = content[first:last + 1]
            body = TruncatedBody(selected) if self.payload_reads == 1 else io.BytesIO(selected)
            return {"Body": body}

    client = Client()
    sleeps = []
    monkeypatch.setattr(
        controller, "_client", lambda _url: (client, "bucket", "prefix"),
    )
    monkeypatch.setattr(
        controller,
        "_object_map",
        lambda *_args: {
            "commit.json": {"size": len(commit_payload)},
            "payload.bin": {"size": len(content)},
        },
    )
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)
    monkeypatch.setattr(controller, "COMMIT_HASH_RANGE_BYTES", 4)

    observed, _digest = controller._read_commit(
        "swift://host/account/bucket/prefix", verify_hashes=True,
    )

    assert observed == commit
    assert client.payload_reads == 3
    assert client.ranges == ["bytes=0-3", "bytes=0-3", "bytes=4-6"]
    assert sleeps == [controller.COMMIT_VISIBILITY_BASE_DELAY_SECONDS]


def _campaign(tmp_path: Path, campaign_type: str, inventory: dict):
    path = tmp_path / "inventory.json"
    inventory_hash = _write_json(path, inventory)
    config_path = tmp_path / "configuration.json"
    config_hash = _write_json(config_path, {"pose_comparison_tolerances": {}})
    campaign = db.create_campaign(
        name="campaign", campaign_type=campaign_type, dataset="dataset",
        pipeline_version="1.0.0", output_uri="swift://host/account/bucket/data_export_2",
        created_by="operator", db_path=str(tmp_path / "db.sqlite"),
    )
    db.freeze_campaign(
        campaign["id"], inventory_uri="swift://host/account/bucket/inventory.json",
        inventory_sha256=inventory_hash,
        configuration_uri="swift://host/account/bucket/configuration.json",
        configuration_sha256=config_hash, db_path=str(tmp_path / "db.sqlite"),
    )
    return path, str(tmp_path / "db.sqlite")


def test_enqueue_revalidation_canary_and_blocked_bulk(tmp_path):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "legacy_revalidation", "sequence_count": 2,
        "sequences": [
            {"sequence": "canary", "cohort": "CANARY", "route": "revalidation",
             "data_output_objects": [], "data_export_objects": []},
            {"sequence": "bulk", "cohort": "BULK", "route": "revalidation",
             "data_output_objects": [], "data_export_objects": []},
        ],
    }
    path, db_path = _campaign(tmp_path, "LEGACY_REVALIDATION", inventory)
    result = controller.enqueue_campaign("campaign", path, db_path=db_path)
    assert result == {"inventory": 2, "created": 2, "existing": 0, "blocked": 0}
    assert db.get_campaign("campaign", db_path=db_path)["inventory_sequence_count"] == 2
    requests = {row["sequence_name"]: row for row in db.list_stage_requests(
        campaign="campaign", db_path=db_path,
    )}
    assert requests["canary"]["status"] == "PENDING"
    assert requests["bulk"]["status"] == "BLOCKED"
    assert requests["bulk"]["blocked_reason"] == "WAITING_CANARY_APPROVAL"
    assert requests["canary"]["source_manifest_sha256"] == hashlib.sha256(
        controller._canonical(inventory["sequences"][0]).encode()
    ).hexdigest()
    canary_parameters = json.loads(requests["canary"]["parameters_json"])
    assert canary_parameters["inventory_uri"].endswith(
        "/revalidation_records/canary.json"
    )
    assert controller.enqueue_campaign("campaign", path, db_path=db_path)["existing"] == 2


def test_enqueue_backlog_requires_every_sequence_to_start_at_preprocess(tmp_path):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 2,
        "sequences": [
            {"sequence": "needs_preprocess", "recommended_stage": "preprocess",
             "raw_input_objects": [{"relative_path": "recording.mcap"}]},
            {"sequence": "also_preprocess", "recommended_stage": "preprocess",
             "raw_input_objects": [{"relative_path": "recording.mcap"}]},
        ],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    result = controller.enqueue_campaign("campaign", path, db_path=db_path)
    assert result["blocked"] == 0
    requests = {row["sequence_name"]: row for row in db.list_stage_requests(
        campaign="campaign", db_path=db_path,
    )}
    assert requests["needs_preprocess"]["stage"] == "preprocess"
    assert requests["also_preprocess"]["stage"] == "preprocess"


def test_enqueue_backlog_blocks_missing_raw_input(tmp_path):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{
            "sequence": "missing_raw", "recommended_stage": "preprocess",
            "raw_input_objects": [],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    result = controller.enqueue_campaign("campaign", path, db_path=db_path)
    assert result["blocked"] == 1
    request = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    assert request["status"] == "BLOCKED"
    assert request["blocked_reason"] == "MISSING_RAW_INPUT"


def test_enqueue_backlog_rejects_legacy_shortcut_route(tmp_path):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{"sequence": "sequence", "recommended_stage": "reconstruction"}],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    import pytest

    with pytest.raises(ValueError, match="must start at preprocessing"):
        controller.enqueue_campaign("campaign", path, db_path=db_path)


def test_enqueue_backlog_reuses_exact_validated_preprocess(tmp_path):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    db.ensure_version_cached("1.0.0", db_path=db_path)
    execution_id = db.create_workflow_execution(
        workflow_name="preprocess-workflow", pipeline_type=db.PREPROCESS_STAGE,
        pipeline_version="1.0.0", status="SUCCEEDED", db_path=db_path,
    )
    preprocess = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.0.0", status="SUCCEEDED",
        output_uri="swift://example/data_output/sequence",
        execution_id=execution_id, trigger="MIGRATION",
        db_path=db_path,
    )
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "reconstruction",
            "validated_preprocess_run_id": preprocess["stage_run_id"],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)

    result = controller.enqueue_campaign("campaign", path, db_path=db_path)

    assert result == {"inventory": 1, "created": 1, "existing": 0, "blocked": 0}
    request = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    assert request["stage"] == "reconstruction"
    parameters = json.loads(request["parameters_json"])
    assert parameters["preprocess_run_id"] == preprocess["stage_run_id"]
    assert parameters["preprocess_reused"] is True


def test_revalidation_workflow_defines_commit_gate():
    workflow = (
        Path(__file__).parents[1] / "osmo" / "mv_hoi_revalidation.yaml"
    ).read_text()
    assert "--backend nvidia_tensorrt" in workflow
    assert "compare_object_poses" in workflow
    assert "finalize_revalidation_export" in workflow
    assert '--sequence-name "{{sequence_name}}"' in workflow
    assert 'get("interaction_trim", {})' in workflow
    assert 'sha256sum "$CONFIG"' in workflow
    assert "--interaction-pre-contact-padding-seconds \"$TRIM_PRE_PADDING\"" in workflow
    assert "--interaction-post-contact-padding-seconds \"$TRIM_POST_PADDING\"" in workflow
    assert "{{legacy_export_url}}/failure_segments.json" in workflow
    assert '--failure-segments-path "$FAILURE_SEGMENTS"' in workflow
    assert "--rebuild-incomplete" in workflow
    assert "cpu_export:\n      cpu: 16" in workflow
    assert (
        "- name: export_revalidated\n"
        "    image: nvcr.io/nvstaging/isaac-amr/mv_hoi_mv_postprocess:"
        "{{image_tag}}\n"
        "    resource: cpu_export"
    ) in workflow
    assert "upload_hitl" not in workflow

    retry_workflow = (
        Path(__file__).parents[1]
        / "osmo"
        / "mv_hoi_revalidation_export_retry.yaml"
    ).read_text()
    assert "cpu_export:\n      cpu: 16" in retry_workflow
    assert "{{legacy_export_url}}/failure_segments.json" in retry_workflow
    assert '--failure-segments-path "$FAILURE_SEGMENTS"' in retry_workflow
    assert '--sequence-name "{{sequence_name}}"' in retry_workflow
    assert "--interaction-post-contact-padding-seconds \"$TRIM_POST_PADDING\"" in retry_workflow
    assert (
        "- name: export_revalidated\n"
        "    image: nvcr.io/nvstaging/isaac-amr/mv_hoi_mv_postprocess:"
        "{{image_tag}}\n"
        "    resource: cpu_export"
    ) in retry_workflow


def test_all_revalidation_destinations_are_staged_until_controller_publication():
    request = {
        "cohort": "CANARY", "sequence_name": "sequence", "pipeline_version": "1.0.0",
        "id": 7, "source_manifest_sha256": "a" * 64,
        "source_manifest_json": json.dumps({
            "reconstruction_pipeline_version": "1.4.0",
        }),
            "parameters_json": json.dumps({
                "inventory_uri": "swift://host/account/bucket/revalidation_records/sequence.json",
                "work_output_layout": "request_scoped_v1",
            }),
    }
    campaign = {
        "name": "campaign", "output_uri": "swift://host/account/bucket/data_export_2",
        "inventory_uri": "swift://host/account/bucket/inventory.json",
        "configuration_uri": "swift://host/account/bucket/configuration.json",
        "configuration_sha256": "b" * 64,
    }
    dataset = {
        "swift_base": "swift://host/account/bucket/root",
        "weights_base_url": "swift://host/account/bucket/weights",
        "pipelines": {
            "mv_hoi_revalidation": {
                "input_path": "data_output", "legacy_export_path": "data_export",
                "work_output_path": "_revalidation_work",
                "workflows": {"revalidation": {"qc_thresholds": {}}},
            },
        },
    }
    values = controller._revalidation_values(
        request, campaign, dataset, db_path="unused.sqlite",
    )
    assert values["destination_url"].endswith(
        "/_revalidation_work/campaign/sequence/request_7/candidate_export"
    )
    assert values["work_output_url"].endswith(
        "/_revalidation_work/campaign/sequence/request_7"
    )
    assert values["inventory_url"].endswith(
        "/revalidation_records/sequence.json"
    )
    assert values["legacy_export_url"].endswith("/data_export/sequence")
    assert values["min_object_silhouette_bbox_containment"] == "0.8"
    assert values["min_object_segment_pixels"] == "10"
    assert values["min_object_bbox_component_pixels"] == "3"
    assert values["min_object_bbox_component_fraction_of_largest"] == "0.001"
    assert values["object_silhouette_render_bbox_padding_pixels"] == "8"
    assert values["max_object_silhouette_bad_frame_fraction"] == "0.05"
    assert values["max_chamfer_object"] == "40.0"
    assert values["max_chamfer_human"] == "40.0"
    assert values["max_chamfer_segment_object"] == "50.0"
    assert values["max_chamfer_segment_human"] == "50.0"
    assert values["max_silhouette_failure_coverage"] == "0.5"
    assert "max_accuracy_failure_segments" not in values
    assert values["legacy_pipeline_version"] == "1.4.0"
    assert values["legacy_pose_frame"] == "aligned"
    assert "max_object_unexplained_sam2_ratio" not in values
    assert "interaction_pre_contact_padding_seconds" not in values
    assert "interaction_post_contact_padding_seconds" not in values
    request["cohort"] = "BULK"
    values = controller._revalidation_values(
        request, campaign, dataset, db_path="unused.sqlite",
    )
    assert values["destination_url"].endswith(
        "/_revalidation_work/campaign/sequence/request_7/candidate_export"
    )

    # Requests submitted before request-scoped staging remain reconcilable.
    request["parameters_json"] = json.dumps({
        "inventory_uri": (
            "swift://host/account/bucket/revalidation_records/sequence.json"
        ),
    })
    legacy_values = controller._revalidation_values(
        request, campaign, dataset, db_path="unused.sqlite",
    )
    assert legacy_values["destination_url"].endswith(
        "/_revalidation_work/campaign/sequence/candidate_export"
    )


def test_frozen_source_validation_checks_selected_objects_and_ignores_unrelated():
    manifest = {
        "data_output_objects": [
            {"relative_path": "foundation_pose/poses.npy", "size": 10, "etag": "pose"},
        ],
        "data_export_objects": [
            {"relative_path": "poses.npy", "size": 10, "etag": "export"},
        ],
    }
    request = {
        "sequence_name": "sequence",
        "source_manifest_json": json.dumps(manifest),
    }
    dataset = {
        "swift_base": "swift://host/account/bucket/root",
        "pipelines": {
            "mv_hoi_revalidation": {
                "input_path": "data_output", "legacy_export_path": "data_export",
            },
        },
    }
    client = _HeadClient({
        "root/data_output/sequence/foundation_pose/poses.npy": {
            "size": 10, "etag": "pose",
        },
        "root/data_export/sequence/poses.npy": {"size": 10, "etag": "export"},
        "root/data_output/sequence/debug/untracked.bin": {"size": 999, "etag": "extra"},
    })
    controller.validate_frozen_objects(request, dataset, client=client)

    client.objects["root/data_output/sequence/foundation_pose/poses.npy"]["etag"] = "changed"
    with pytest.raises(ValueError, match="foundation_pose/poses.npy"):
        controller.validate_frozen_objects(request, dataset, client=client)


def test_frozen_source_validation_falls_back_to_ranged_get_after_swift_head_400():
    request = {
        "sequence_name": "sequence",
        "source_manifest_json": json.dumps({
            "data_output_objects": [
                {"relative_path": "source.bin", "size": 10, "etag": "pose"},
            ],
            "data_export_objects": [
                {"relative_path": "export.bin", "size": 10, "etag": "pose"},
            ],
        }),
    }
    dataset = {
        "swift_base": "swift://host/account/bucket/root",
        "pipelines": {
            "mv_hoi_revalidation": {
                "input_path": "data_output", "legacy_export_path": "data_export",
            },
        },
    }

    controller.validate_frozen_objects(
        request, dataset, client=_RangedGetFallbackClient(),
    )


def test_frozen_source_validation_requires_two_matches_after_transient_mismatch():
    manifest = {
        "data_output_objects": [
            {"relative_path": "source.bin", "size": 10, "etag": "expected"},
        ],
        "data_export_objects": [
            {"relative_path": "export.bin", "size": 10, "etag": "expected"},
        ],
    }
    request = {
        "sequence_name": "sequence", "source_manifest_json": json.dumps(manifest),
    }
    dataset = {
        "swift_base": "swift://host/account/bucket/root",
        "pipelines": {"mv_hoi_revalidation": {
            "input_path": "data_output", "legacy_export_path": "data_export",
        }},
    }
    objects = {
        "root/data_output/sequence/source.bin": {"size": 10, "etag": "expected"},
        "root/data_export/sequence/export.bin": {"size": 10, "etag": "expected"},
    }
    client = _SequencedHeadClient(objects, [
        {"size": 10, "etag": "expected"},
        {"size": 10, "etag": "expected"},
    ])
    controller.validate_frozen_objects(request, dataset, client=client)
    assert client.calls == 2

    client = _SequencedHeadClient(objects, [
        {"size": 10, "etag": "expected"},
        {"size": 10, "etag": "stale"},
    ])
    with pytest.raises(ValueError, match="source.bin"):
        controller.validate_frozen_objects(request, dataset, client=client)


def test_revalidation_dry_run_preflight_failure_does_not_mutate_request_or_blacklist(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "legacy_revalidation", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "cohort": "CANARY", "route": "revalidation",
            "reconstruction_pipeline_version": "1.4.0",
            "data_output_objects": [{"relative_path": "source", "size": 1}],
            "data_export_objects": [{"relative_path": "export", "size": 1}],
        }],
    }
    path, db_path = _campaign(tmp_path, "LEGACY_REVALIDATION", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    request = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    monkeypatch.setattr(
        controller, "validate_frozen_objects",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("source changed")),
    )
    monkeypatch.setattr(controller, "prepare_destination", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        controller, "get_workflow_cfg",
        lambda *_args, **_kwargs: {"workflow_yaml": "workflow.yaml"},
    )
    config = {"datasets": {"dataset": {
        "swift_base": "swift://host/account/bucket/root",
        "weights_base_url": "swift://host/account/bucket/weights",
        "osmo_pool": "pool",
        "pipelines": {"mv_hoi_revalidation": {
            "input_path": "data_output", "legacy_export_path": "data_export",
            "work_output_path": "_work", "max_concurrent": 40,
        }},
    }}}

    result = controller.dispatch_revalidation(
        "campaign", db_path=db_path, config=config, dry_run=True,
    )

    assert result[0]["status"] == "DRY_RUN_FAILED"
    assert db.get_stage_request(request["id"], db_path=db_path)["status"] == "PENDING"
    assert db.get_blacklisted_sequence("dataset", "sequence", db_path=db_path) is None


def test_revalidation_export_retry_uses_prior_work_outputs(
    tmp_path, monkeypatch,
):
    db.init_db(str(tmp_path / "db.sqlite"))
    inventory = {
        "dataset": "dataset", "kind": "legacy_revalidation", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "cohort": "CANARY", "route": "revalidation",
            "reconstruction_pipeline_version": "1.4.0",
            "data_output_objects": [{"relative_path": "source", "size": 1}],
            "data_export_objects": [{"relative_path": "export", "size": 1}],
        }],
    }
    path, db_path = _campaign(tmp_path, "LEGACY_REVALIDATION", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    monkeypatch.setattr(
        controller, "validate_frozen_objects", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        controller, "prepare_destination", lambda *_args, **_kwargs: None,
    )
    workflow_keys = []

    def workflow_config(_dataset, _pipeline, workflow_key):
        workflow_keys.append(workflow_key)
        return {"workflow_yaml": f"{workflow_key}.yaml"}

    monkeypatch.setattr(controller, "get_workflow_cfg", workflow_config)
    config = {"datasets": {"dataset": {
        "swift_base": "swift://host/account/bucket/root",
        "weights_base_url": "swift://host/account/bucket/weights",
        "osmo_pool": "pool",
        "pipelines": {"mv_hoi_revalidation": {
            "input_path": "data_output", "legacy_export_path": "data_export",
            "work_output_path": "_work", "max_concurrent": 40,
        }},
    }}}
    prior = "swift://host/account/bucket/root/_work/prior/sequence"

    result = controller.dispatch_revalidation(
        "campaign",
        db_path=db_path,
        config=config,
        dry_run=True,
        reuse_work_output_url=prior,
    )

    assert controller.REVALIDATION_EXPORT_RETRY_WORKFLOW in workflow_keys
    assert result[0]["status"] == "DRY_RUN"
    assert result[0]["values"]["retry_work_output_url"] == prior


def test_revalidation_request_can_select_export_retry_from_parameters(
    tmp_path, monkeypatch,
):
    db.init_db(str(tmp_path / "db.sqlite"))
    inventory = {
        "dataset": "dataset", "kind": "legacy_revalidation", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "cohort": "CANARY", "route": "revalidation",
            "reconstruction_pipeline_version": "1.4.0",
            "data_output_objects": [{"relative_path": "source", "size": 1}],
            "data_export_objects": [{"relative_path": "export", "size": 1}],
        }],
    }
    path, db_path = _campaign(tmp_path, "LEGACY_REVALIDATION", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    request = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    prior = "swift://host/account/bucket/root/_work/prior/sequence"
    db.update_stage_request(
        request["id"], parameters={"retry_work_output_url": prior},
        db_path=db_path,
    )
    monkeypatch.setattr(
        controller, "validate_frozen_objects", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        controller, "prepare_destination", lambda *_args, **_kwargs: None,
    )
    workflow_keys = []
    monkeypatch.setattr(
        controller,
        "get_workflow_cfg",
        lambda _dataset, _pipeline, workflow_key: (
            workflow_keys.append(workflow_key)
            or {"workflow_yaml": f"{workflow_key}.yaml"}
        ),
    )
    config = {"datasets": {"dataset": {
        "swift_base": "swift://host/account/bucket/root",
        "weights_base_url": "swift://host/account/bucket/weights",
        "osmo_pool": "pool",
        "pipelines": {"mv_hoi_revalidation": {
            "input_path": "data_output", "legacy_export_path": "data_export",
            "work_output_path": "_work", "max_concurrent": 40,
        }},
    }}}

    result = controller.dispatch_revalidation(
        "campaign", db_path=db_path, config=config, dry_run=True,
    )

    assert controller.REVALIDATION_EXPORT_RETRY_WORKFLOW in workflow_keys
    assert result[0]["values"]["retry_work_output_url"] == prior


def test_revalidation_submission_starts_before_later_preflights_finish(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "legacy_revalidation",
        "sequence_count": 2,
        "sequences": [{
            "sequence": sequence, "cohort": "CANARY", "route": "revalidation",
            "reconstruction_pipeline_version": "1.4.0",
            "data_output_objects": [{"relative_path": "source", "size": 1}],
            "data_export_objects": [{"relative_path": "export", "size": 1}],
        } for sequence in ("sequence-a", "sequence-b")],
    }
    path, db_path = _campaign(tmp_path, "LEGACY_REVALIDATION", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    submit_started = threading.Event()

    def validate(request, *_args, **_kwargs):
        if request["sequence_name"] == "sequence-b":
            assert submit_started.wait(1), (
                "the first reserved workflow was not submitted while the next "
                "candidate was being prepared"
            )

    def osmo_submit(_template, _pool, values):
        submit_started.set()
        time.sleep(0.02)
        return values["workflow_name"] + "-1"

    class Decision:
        pool = "pool"

        @staticmethod
        def detail(event):
            return event

    class Selector:
        @staticmethod
        def choose(*_args, **_kwargs):
            return Decision()

    monkeypatch.setattr(controller, "validate_frozen_objects", validate)
    monkeypatch.setattr(
        controller, "prepare_destination", lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        controller, "get_workflow_cfg",
        lambda *_args, **_kwargs: {"workflow_yaml": "workflow.yaml"},
    )
    monkeypatch.setattr(
        controller, "_revalidation_values",
        lambda request, *_args, **_kwargs: {
            "workflow_name": f"workflow-{request['id']}",
            "destination_url": f"swift://candidate/{request['id']}",
        },
    )
    monkeypatch.setattr(controller.submit, "osmo_submit", osmo_submit)
    config = {"datasets": {"dataset": {
        "swift_base": "swift://host/account/bucket/root",
        "pipelines": {"mv_hoi_revalidation": {
            "work_output_path": "_work", "max_concurrent": 40,
        }},
    }}}

    result = controller.dispatch_revalidation(
        "campaign", db_path=db_path, config=config,
        pool_selector=Selector(), submit_workers=2,
    )

    assert [item["status"] for item in result] == ["RUNNING", "RUNNING"]
    assert submit_started.is_set()


def test_backlog_dry_run_source_failure_does_not_mutate_request(tmp_path, monkeypatch):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap", "size": 1}],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    request = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    monkeypatch.setattr(
        controller, "validate_frozen_raw_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("source changed")),
    )
    config = {"datasets": {"dataset": {
        "pipelines": {
            "mv_preprocess": {"max_concurrent": 40},
            "mv_hoi_reconstruction": {"max_concurrent": 40},
        },
    }}}

    result = controller.dispatch_backlog(
        "campaign", db_path=db_path, config=config,
        preprocess_limit=1, dry_run=True,
    )

    assert result[0]["status"] == "DRY_RUN_BLOCKED"
    assert db.get_stage_request(request["id"], db_path=db_path)["status"] == "PENDING"
    assert db.get_blacklisted_sequence("dataset", "sequence", db_path=db_path) is None


def test_backlog_dispatch_submits_candidates_concurrently(monkeypatch):
    requests = [{
        "id": request_id,
        "stage": "preprocess",
        "sequence_name": f"sequence-{request_id}",
    } for request_id in range(1, 5)]
    monkeypatch.setattr(
        controller.db, "get_campaign",
        lambda *_a, **_k: {
            "campaign_type": "BACKLOG_REPROCESSING",
            "dataset": "dataset", "pipeline_version": "1.6.0",
        },
    )
    monkeypatch.setattr(
        controller.db, "list_stage_requests",
        lambda *_a, status=None, **_k: list(requests) if status == "PENDING" else [],
    )
    monkeypatch.setattr(
        controller.db, "count_active_stage_executions", lambda *_a, **_k: 0,
    )
    monkeypatch.setattr(
        controller, "validate_frozen_raw_input", lambda *_a, **_k: None,
    )
    active = 0
    maximum = 0
    lock = threading.Lock()

    def submit_sequence(sequence, *_args, **_kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return type("Outcome", (), {"workflow_name": f"wf-{sequence}"})()

    monkeypatch.setattr(controller.submit, "submit_sequence", submit_sequence)
    config = {"datasets": {"dataset": {"pipelines": {
        controller.PREPROCESS_PIPELINE: {"max_concurrent": 4},
        controller.RECON_PIPELINE: {"max_concurrent": 4},
    }}}}

    result = controller.dispatch_backlog(
        "campaign", db_path="db", config=config, submit_workers=4,
        pool_selector=object(),
    )

    assert maximum > 1
    assert len(result) == 4


def test_backlog_canary_prerequisite_skip_is_terminal_expected_data(monkeypatch):
    request = {
        "id": 17,
        "stage": "reconstruction",
        "sequence_name": "source-data-canary",
        "cohort": "CANARY",
        "queue_priority": 0,
        "created_at": "2026-08-03T00:00:00Z",
    }
    monkeypatch.setattr(
        controller.db, "get_campaign",
        lambda *_a, **_k: {
            "campaign_type": "BACKLOG_REPROCESSING",
            "dataset": "dataset", "pipeline_version": "1.6.34",
        },
    )
    monkeypatch.setattr(
        controller.db, "list_stage_requests",
        lambda *_a, status=None, **_k: [request] if status == "PENDING" else [],
    )
    monkeypatch.setattr(
        controller.db, "count_active_stage_executions", lambda *_a, **_k: 0,
    )
    monkeypatch.setattr(
        controller.db, "get_stage_request",
        lambda *_a, **_k: {
            **request, "status": "BLOCKED",
            "blocked_reason": (
                "backlog reconstruction requires "
                "object_bbox_source.txt=manual_labeled_bboxes"
            ),
            "details": "missing manual bbox provenance",
        },
    )
    updates = []
    monkeypatch.setattr(
        controller.db, "update_stage_request",
        lambda request_id, **kwargs: updates.append((request_id, kwargs)),
    )
    monkeypatch.setattr(
        controller.submit, "submit_sequence",
        lambda *_a, **_k: controller.submit.SubmitResult(
            "unused", prereq_skipped=True,
        ),
    )
    config = {"datasets": {"dataset": {"pipelines": {
        controller.PREPROCESS_PIPELINE: {"max_concurrent": 4},
        controller.RECON_PIPELINE: {"max_concurrent": 4},
    }}}}

    result = controller.dispatch_backlog(
        "campaign", db_path="db", config=config, pool_selector=object(),
    )

    assert result == [{
        "request_id": 17,
        "workflow_name": None,
        "status": "EXPECTED_QC_DATA",
        "details": (
            "backlog reconstruction requires "
            "object_bbox_source.txt=manual_labeled_bboxes"
        ),
    }]
    assert updates[-1][0] == 17
    assert updates[-1][1]["status"] == "FAILED"
    assert updates[-1][1]["result_summary"]["canary_outcome"] == "EXPECTED_QC_DATA"


def test_backlog_canary_lineage_skip_is_not_classified_as_source_data(monkeypatch):
    request = {
        "id": 17,
        "stage": "reconstruction",
        "sequence_name": "lineage-canary",
        "cohort": "CANARY",
        "queue_priority": 100,
        "created_at": "2026-08-03T00:00:00Z",
    }
    monkeypatch.setattr(
        controller.db,
        "get_campaign",
        lambda *_a, **_k: {
            "campaign_type": "BACKLOG_REPROCESSING",
            "dataset": "dataset",
            "pipeline_version": "1.6.44",
        },
    )
    monkeypatch.setattr(
        controller.db,
        "list_stage_requests",
        lambda *_a, status=None, **_k: [request] if status == "PENDING" else [],
    )
    monkeypatch.setattr(
        controller.db, "count_active_stage_executions", lambda *_a, **_k: 0,
    )
    observed = {
        **request,
        "status": "BLOCKED",
        "blocked_reason": "current preprocessing run does not match campaign lineage",
    }
    monkeypatch.setattr(
        controller.db, "get_stage_request", lambda *_a, **_k: observed,
    )
    updates = []
    monkeypatch.setattr(
        controller.db,
        "update_stage_request",
        lambda request_id, **kwargs: updates.append((request_id, kwargs)),
    )
    monkeypatch.setattr(
        controller.submit,
        "submit_sequence",
        lambda *_a, **_k: controller.submit.SubmitResult(
            "unused", prereq_skipped=True,
        ),
    )
    config = {"datasets": {"dataset": {"pipelines": {
        controller.PREPROCESS_PIPELINE: {"max_concurrent": 4},
        controller.RECON_PIPELINE: {"max_concurrent": 4},
    }}}}

    result = controller.dispatch_backlog(
        "campaign", db_path="db", config=config, pool_selector=object(),
    )

    assert result == [{
        "request_id": 17,
        "workflow_name": None,
        "status": "IMPLEMENTATION_BLOCKED",
        "details": "current preprocessing run does not match campaign lineage",
    }]
    assert updates == []


def test_revalidation_canary_qc_failure_requires_validated_evidence(monkeypatch):
    request = {
        "id": 17, "cohort": "CANARY", "dataset": "dataset",
        "sequence_name": "sequence", "campaign_name": "campaign",
        "campaign_type": "LEGACY_REVALIDATION", "stage": "revalidation",
    }
    execution = {"id": 8, "workflow_name": "workflow"}
    campaign = {"name": "campaign", "dataset": "dataset"}
    summary = {
        "canary_outcome": "EXPECTED_QC_DATA",
        "failure_category": "accuracy_limit", "failed_task": "check_accuracy",
        "evidence_valid": True,
    }
    monkeypatch.setattr(
        controller, "handle_campaign_infrastructure_failure",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        controller, "_expected_revalidation_canary_failure",
        lambda *args, **kwargs: summary,
    )
    observations = []
    monkeypatch.setattr(
        controller.db, "apply_execution_observation",
        lambda *args, **kwargs: observations.append((args, kwargs)),
    )
    blacklists = []
    monkeypatch.setattr(
        controller, "_blacklist_campaign_failure",
        lambda *args, **kwargs: blacklists.append((args, kwargs)),
    )

    result = controller._handle_revalidation_observation(
        request, execution,
        {"status": "FAILED", "tasks": {"check_accuracy": "FAILED"}},
        campaign=campaign, dataset_cfg={}, db_path="db",
        verify_payload_hashes=True, retry_infrastructure=True,
        recover_publication_failures=False, defer_bulk_publication=False,
        publication_copy_workers=1,
    )

    assert result["status"] == "EXPECTED_QC_DATA"
    assert observations[0][1]["request_outcomes"][0]["result_summary"] == summary
    assert blacklists


def test_frozen_backlog_membership_bypasses_blacklist_without_removing_it(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap", "size": 1}],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    db.upsert_blacklisted_sequence(
        "dataset", "sequence", reason="legacy failure", db_path=db_path,
    )
    monkeypatch.setattr(controller, "validate_frozen_raw_input", lambda *_args: None)
    captured = {}

    def _submit(*_args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(controller.submit, "submit_sequence", _submit)
    config = {"datasets": {"dataset": {
        "pipelines": {
            "mv_preprocess": {"max_concurrent": 40},
            "mv_hoi_reconstruction": {"max_concurrent": 40},
        },
    }}}

    controller.dispatch_backlog(
        "campaign", db_path=db_path, config=config,
        preprocess_limit=1, dry_run=True,
    )

    assert captured["force"] is True
    assert db.get_blacklisted_sequence("dataset", "sequence", db_path=db_path) is not None


def test_backlog_concurrency_is_independent_across_stages(tmp_path, monkeypatch):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    sequences = [
        {
            "sequence": f"sequence-{index:02d}",
            "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap", "size": 1}],
        }
        for index in range(61)
    ]
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing",
        "sequence_count": len(sequences), "sequences": sequences,
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    requests = db.list_stage_requests(campaign="campaign", db_path=db_path)
    conn = db.get_connection(db_path)
    try:
        conn.executemany(
            "UPDATE stage_requests SET status='RUNNING' WHERE id=?",
            [(row["id"],) for row in requests[:30]],
        )
        conn.executemany(
            "UPDATE stage_requests SET status='SUCCEEDED' WHERE id=?",
            [(row["id"],) for row in requests[30:60]],
        )
        conn.commit()
    finally:
        conn.close()
    campaign = db.get_campaign("campaign", db_path=db_path)
    for row in requests[30:60]:
        db.create_stage_request(
            sequence_name=row["sequence_name"], dataset="dataset",
            stage="reconstruction", pipeline_version="1.0.0",
            campaign=campaign["id"], cohort="BULK",
            source_manifest={"labeled_bboxes": []}, db_path=db_path,
        )
    monkeypatch.setattr(controller, "validate_frozen_raw_input", lambda *_args: None)
    submitted = []
    monkeypatch.setattr(
        controller.submit, "submit_sequence",
        lambda sequence, *_args, **_kwargs: submitted.append(sequence),
    )
    config = {"datasets": {"dataset": {
        "pipelines": {
            "mv_preprocess": {
                "max_concurrent": 60, "campaign_max_concurrent": 30,
            },
            "mv_hoi_reconstruction": {
                "max_concurrent": 60, "campaign_max_concurrent": 30,
            },
        },
    }}}

    controller.dispatch_backlog(
        "campaign", db_path=db_path, config=config, dry_run=True,
    )

    assert len(submitted) == 30
    assert "sequence-60" not in submitted


def test_backlog_stage_capacity_counts_campaignless_active_work(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{
            "sequence": "campaign-sequence", "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap", "size": 1}],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    campaign = db.get_campaign("campaign", db_path=db_path)
    db.create_stage_request(
        sequence_name="campaign-sequence", dataset="dataset",
        stage="reconstruction", pipeline_version="1.0.0",
        campaign=campaign["id"], cohort="BULK",
        source_manifest={"labeled_bboxes": []}, db_path=db_path,
    )
    mainline_ids = []
    for index in range(30):
        request = db.create_stage_request(
            sequence_name=f"mainline-{index:02d}", dataset="dataset",
            stage="preprocess", pipeline_version="1.0.0", db_path=db_path,
        )
        mainline_ids.append(request["id"])
    conn = db.get_connection(db_path)
    try:
        conn.executemany(
            "UPDATE stage_requests SET status='RUNNING' WHERE id=?",
            [(request_id,) for request_id in mainline_ids],
        )
        conn.commit()
    finally:
        conn.close()
    submitted = []
    monkeypatch.setattr(
        controller.submit, "submit_sequence",
        lambda _sequence, _dataset, _cfg, pipeline, **_kwargs:
            submitted.append(pipeline),
    )
    config = {"datasets": {"dataset": {"pipelines": {
        "mv_preprocess": {"max_concurrent": 30},
        "mv_hoi_reconstruction": {"max_concurrent": 30},
    }}}}

    controller.dispatch_backlog(
        "campaign", db_path=db_path, config=config, dry_run=True,
    )

    assert submitted == ["mv_hoi_reconstruction"]


def test_backlog_stage_filter_dispatches_only_reconstruction(tmp_path, monkeypatch):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 2,
        "sequences": [
            {
                "sequence": "preprocess", "recommended_stage": "preprocess",
                "raw_input_objects": [{"relative_path": "recording.mcap", "size": 1}],
            },
            {
                "sequence": "reconstruction", "recommended_stage": "preprocess",
                "raw_input_objects": [{"relative_path": "recording.mcap", "size": 1}],
            },
        ],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    campaign = db.get_campaign("campaign", db_path=db_path)
    db.create_stage_request(
        sequence_name="reconstruction", dataset="dataset", stage="reconstruction",
        pipeline_version="1.0.0", campaign=campaign["id"], cohort="BULK",
        source_manifest={"labeled_bboxes": []}, db_path=db_path,
    )
    called = []
    monkeypatch.setattr(
        controller.submit, "submit_sequence",
        lambda _sequence, _dataset, _cfg, pipeline, **_kwargs: called.append(pipeline),
    )
    config = {"datasets": {"dataset": {
        "pipelines": {
            "mv_preprocess": {"max_concurrent": 40},
            "mv_hoi_reconstruction": {"max_concurrent": 40},
        },
    }}}

    controller.dispatch_backlog(
        "campaign", db_path=db_path, config=config,
        reconstruction_limit=40, dry_run=True,
        requested_stage="reconstruction",
    )

    assert called == ["mv_hoi_reconstruction"]


def test_backlog_sequence_filter_scopes_canary_dispatch(tmp_path, monkeypatch):
    db.init_db(str(tmp_path / "db.sqlite"))
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 3,
        "sequences": [{
            "sequence": name, "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap", "size": 1}],
        } for name in ("marked", "unmarked", "new-label")],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    monkeypatch.setattr(controller, "validate_frozen_raw_input", lambda *_args: None)
    submitted = []
    monkeypatch.setattr(
        controller.submit, "submit_sequence",
        lambda sequence, *_args, **_kwargs: submitted.append(sequence),
    )
    config = {"datasets": {"dataset": {"pipelines": {
        "mv_preprocess": {"max_concurrent": 30},
        "mv_hoi_reconstruction": {"max_concurrent": 30},
    }}}}

    controller.dispatch_backlog(
        "campaign", db_path=db_path, config=config, dry_run=True,
        sequences=("marked", "new-label"),
    )

    assert submitted == ["marked", "new-label"]


def test_failed_pool_builds_reproducible_remediation_inventory(tmp_path):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    source_record = {
        "sequence": "failed", "cohort": "CANARY", "route": "revalidation",
        "recommended_stage": "revalidation",
        "data_output_objects": [{"relative_path": "source", "size": 1, "etag": "a"}],
        "data_export_objects": [], "raw_input_objects": [],
    }
    inventory = {
        "dataset": "dataset", "kind": "legacy_revalidation", "sequence_count": 1,
        "sequences": [source_record],
    }
    path, db_path = _campaign(tmp_path, "LEGACY_REVALIDATION", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    failed = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    db.update_stage_request(
        failed["id"], status="FAILED", details="pose gate failed", db_path=db_path,
    )
    db.upsert_blacklisted_sequence(
        "dataset", "failed", reason="failed campaign timeout box", db_path=db_path,
    )

    remediation = controller.build_failed_remediation_inventory(
        "campaign", db_path=db_path,
    )

    assert remediation["kind"] == "remediation"
    assert remediation["sequence_count"] == 1
    record = remediation["sequences"][0]
    assert record["recommended_stage"] == "preprocess"
    assert record["failed_request_id"] == failed["id"]
    assert record["failed_details"] == "pose gate failed"


def test_advance_backlog_preserves_existing_bbox_identity(tmp_path, monkeypatch):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    bbox_objects = [
        {
            "relative_path": f"mv_preprocess/labeled_bboxes/{camera}.json",
            "key": f"root/sequence/mv_preprocess/labeled_bboxes/{camera}.json",
            "size": 12,
            "etag": camera,
        }
        for camera in (
            "back_stereo_camera_left", "front_stereo_camera_left",
            "left_stereo_camera_left", "right_stereo_camera_left",
        )
    ]
    current_objects = bbox_objects + [{
        "relative_path": "mv_preprocess/object_bbox_source.txt",
        "size": len("manual_labeled_bboxes"),
        "etag": "marker",
        "content": "manual_labeled_bboxes",
    }]
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "preprocess",
            "data_output_objects": bbox_objects,
            "raw_input_objects": [{"relative_path": "recording.mcap"}],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    preprocess = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    db.update_stage_request(preprocess["id"], status="SUCCEEDED", db_path=db_path)
    monkeypatch.setattr(
        controller, "_current_sequence_objects", lambda *_: current_objects,
    )
    result = controller.advance_backlog(
        "campaign", db_path=db_path,
        config={"datasets": {"dataset": {
            "swift_base": "swift://host/account/bucket/root",
                "pipelines": {
                    "mv_preprocess": {
                        "output_path": "data_output",
                        "campaign_output_path": "data_output",
                    },
                    "mv_hoi_reconstruction": {
                        "output_path": "data_output",
                        "campaign_output_path": "data_output",
                },
            },
        }}},
    )
    assert result[0]["status"] == "PENDING"
    reconstruction = db.get_active_stage_request(
        "dataset", "sequence", "reconstruction", db_path=db_path,
    )
    assert json.loads(reconstruction["source_manifest_json"])["bbox_preview_required"] is True


def test_advance_backlog_refreshes_blocked_reused_preprocess_lineage(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    db.ensure_version_cached("1.0.0", db_path=db_path)
    source_request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="preprocess",
        pipeline_version="1.0.0", trigger="MIGRATION", requested_by="test",
        db_path=db_path,
    )
    db.update_stage_request(
        source_request["id"], status="SUCCEEDED", db_path=db_path,
    )
    execution_id = db.create_workflow_execution(
        workflow_name="preprocess-workflow", pipeline_type=db.PREPROCESS_STAGE,
        pipeline_version="1.0.0", status="SUCCEEDED", db_path=db_path,
    )
    preprocess = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.PREPROCESS_STAGE,
        pipeline_version="1.0.0", status="SUCCEEDED",
        output_uri="swift://host/account/bucket/root/data_output/sequence",
        request_id=source_request["id"], execution_id=execution_id,
        trigger="MIGRATION", db_path=db_path,
    )
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing",
        "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "reconstruction",
            "validated_preprocess_run_id": preprocess["stage_run_id"],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    reconstruction = db.list_stage_requests(
        campaign="campaign", stage="reconstruction", db_path=db_path,
    )[0]
    db.update_stage_request(
        reconstruction["id"], status="BLOCKED",
        blocked_reason="WAITING_LABELS", details="manual labels required",
        db_path=db_path,
    )
    cameras = (
        "back_stereo_camera_left", "front_stereo_camera_left",
        "left_stereo_camera_left", "right_stereo_camera_left",
    )
    current_objects = [{
        "relative_path": f"mv_preprocess/labeled_bboxes/{camera}.json",
        "size": 12, "etag": camera, "sha256": "a" * 64,
        "schema_valid": True,
    } for camera in cameras] + [{
        "relative_path": "mv_preprocess/object_bbox_source.txt",
        "content": "manual_labeled_bboxes",
    }]
    monkeypatch.setattr(
        controller, "_current_sequence_objects", lambda *_: current_objects,
    )

    result = controller.advance_backlog(
        "campaign", db_path=db_path,
        config={"datasets": {"dataset": {
            "swift_base": "swift://host/account/bucket/root",
            "pipelines": {"mv_preprocess": {
                "output_path": "data_output",
                "campaign_output_path": "data_output",
            }},
        }}},
    )

    assert result == [{
        "sequence": "sequence", "request_id": reconstruction["id"],
        "status": "PENDING", "blocked_reason": None,
    }]
    refreshed = db.get_stage_request(reconstruction["id"], db_path=db_path)
    assert refreshed["status"] == "PENDING"
    assert refreshed["blocked_reason"] is None
    assert json.loads(refreshed["parameters_json"])["preprocess_request_id"] == (
        source_request["id"]
    )
    assert json.loads(refreshed["source_manifest_json"])[
        "preprocess_request_id"
    ] == source_request["id"]


def test_advance_backlog_reads_label_state_concurrently_and_updates_in_order(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    sequences = [f"sequence-{index}" for index in range(4)]
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing",
        "sequence_count": len(sequences),
        "sequences": [{
            "sequence": sequence, "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap"}],
        } for sequence in sequences],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    for request in db.list_stage_requests(
        campaign="campaign", stage="preprocess", db_path=db_path,
    ):
        db.update_stage_request(request["id"], status="SUCCEEDED", db_path=db_path)

    lock = threading.Lock()
    active = 0
    maximum = 0

    def read_labels(*_args):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return []

    monkeypatch.setattr(controller, "_current_sequence_objects", read_labels)
    result = controller.advance_backlog(
        "campaign", db_path=db_path, query_workers=4,
        config={"datasets": {"dataset": {
            "swift_base": "swift://host/account/bucket/root",
            "pipelines": {"mv_preprocess": {
                "output_path": "data_output",
                "campaign_output_path": "data_output",
            }},
        }}},
    )

    assert maximum > 1
    assert [item["sequence"] for item in result] == sequences
    assert {item["blocked_reason"] for item in result} == {"WAITING_LABELS"}


def test_advance_backlog_creates_reconstruction_for_new_preprocess_lineage(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing",
        "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap"}],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    first_preprocess = db.list_stage_requests(
        campaign="campaign", stage="preprocess", db_path=db_path,
    )[0]
    db.update_stage_request(
        first_preprocess["id"], status="SUCCEEDED", db_path=db_path,
    )
    cameras = (
        "back_stereo_camera_left", "front_stereo_camera_left",
        "left_stereo_camera_left", "right_stereo_camera_left",
    )
    current_objects = [{
        "relative_path": f"mv_preprocess/labeled_bboxes/{camera}.json",
        "size": 12, "etag": camera, "sha256": "a" * 64,
    } for camera in cameras] + [{
        "relative_path": "mv_preprocess/object_bbox_source.txt",
        "content": "manual_labeled_bboxes",
    }]
    monkeypatch.setattr(
        controller, "_current_sequence_objects", lambda *_: current_objects,
    )
    config = {"datasets": {"dataset": {
        "swift_base": "swift://host/account/bucket/root",
        "pipelines": {"mv_preprocess": {
            "output_path": "data_output",
            "campaign_output_path": "data_output",
        }},
    }}}

    controller.advance_backlog(
        "campaign", db_path=db_path, config=config,
    )
    first_reconstruction = db.list_stage_requests(
        campaign="campaign", stage="reconstruction", db_path=db_path,
    )[0]
    db.update_stage_request(
        first_reconstruction["id"], status="CANCELED",
        details="superseded preprocessing lineage", db_path=db_path,
    )
    retry_preprocess = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="preprocess",
        pipeline_version="1.0.0", trigger="MIGRATION",
        requested_by="test", reason="missing_face_videos_retry",
        campaign="campaign", cohort="BULK",
        parameters=json.loads(first_preprocess["parameters_json"]),
        source_manifest=json.loads(first_preprocess["source_manifest_json"]),
        db_path=db_path,
    )
    db.update_stage_request(
        retry_preprocess["id"], status="SUCCEEDED", db_path=db_path,
    )

    result = controller.advance_backlog(
        "campaign", db_path=db_path, config=config,
    )

    assert result == [{
        "sequence": "sequence",
        "request_id": result[0]["request_id"],
        "status": "PENDING",
        "blocked_reason": None,
    }]
    reconstructions = db.list_stage_requests(
        campaign="campaign", stage="reconstruction", db_path=db_path,
    )
    assert len(reconstructions) == 2
    assert reconstructions[0]["status"] == "CANCELED"
    assert reconstructions[1]["status"] == "PENDING"
    assert json.loads(
        reconstructions[1]["parameters_json"]
    )["preprocess_request_id"] == retry_preprocess["id"]


def test_advance_backlog_accepts_post_run_sha_when_frozen_inventory_omitted_it(
    tmp_path, monkeypatch,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    frozen = [
        {
            "relative_path": f"mv_preprocess/labeled_bboxes/{camera}.json",
            "size": 12, "etag": camera,
        }
        for camera in (
            "back_stereo_camera_left", "front_stereo_camera_left",
            "left_stereo_camera_left", "right_stereo_camera_left",
        )
    ]
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "preprocess",
            "data_output_objects": frozen,
            "raw_input_objects": [{"relative_path": "recording.mcap"}],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    preprocess = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    db.update_stage_request(preprocess["id"], status="SUCCEEDED", db_path=db_path)
    current = [{**item, "sha256": "a" * 64} for item in frozen]
    current.append({
        "relative_path": "mv_preprocess/object_bbox_source.txt",
        "size": len("manual_labeled_bboxes"),
        "etag": "marker",
        "content": "manual_labeled_bboxes",
    })
    monkeypatch.setattr(controller, "_current_sequence_objects", lambda *_: current)

    result = controller.advance_backlog(
        "campaign", db_path=db_path,
        config={"datasets": {"dataset": {
            "swift_base": "swift://host/account/bucket/root",
                "pipelines": {
                    "mv_preprocess": {
                        "output_path": "data_output",
                        "campaign_output_path": "data_output",
                    },
                    "mv_hoi_reconstruction": {
                        "output_path": "data_output",
                        "campaign_output_path": "data_output",
                },
            },
        }}},
    )

    assert result[0]["status"] == "PENDING"
    reconstruction = db.get_active_stage_request(
        "dataset", "sequence", "reconstruction", db_path=db_path,
    )
    assert reconstruction["status"] == "PENDING"


def test_advance_backlog_blocks_unmarked_bbox_files(tmp_path, monkeypatch):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    old = [{
        "relative_path": "mv_preprocess/labeled_bboxes/back_stereo_camera_left.json",
        "size": 12, "etag": "old",
    }]
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing", "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "preprocess",
            "data_output_objects": old,
            "raw_input_objects": [{"relative_path": "recording.mcap"}],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    preprocess = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    db.update_stage_request(preprocess["id"], status="SUCCEEDED", db_path=db_path)
    monkeypatch.setattr(
        controller, "_current_sequence_objects",
        lambda *_: [{**old[0], "etag": "changed"}],
    )
    result = controller.advance_backlog(
        "campaign", db_path=db_path,
        config={"datasets": {"dataset": {
            "swift_base": "swift://host/account/bucket/root",
                "pipelines": {
                    "mv_preprocess": {
                        "output_path": "data_output",
                        "campaign_output_path": "data_output",
                    },
                    "mv_hoi_reconstruction": {
                        "output_path": "data_output",
                        "campaign_output_path": "data_output",
                },
            },
        }}},
    )
    assert result[0]["blocked_reason"] == "WAITING_LABELS"

    monkeypatch.setattr(
        controller.db, "update_stage_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged blocked label state must not be rewritten")
        ),
    )
    repeated = controller.advance_backlog(
        "campaign", db_path=db_path,
        config={"datasets": {"dataset": {
            "swift_base": "swift://host/account/bucket/root",
            "pipelines": {"mv_preprocess": {
                "output_path": "data_output",
                "campaign_output_path": "data_output",
            }},
        }}},
    )
    assert repeated[0]["blocked_reason"] == "WAITING_LABELS"


def test_transferred_frozen_bboxes_restore_only_missing_manual_marker(monkeypatch):
    cameras = (
        "back_stereo_camera_left", "front_stereo_camera_left",
        "left_stereo_camera_left", "right_stereo_camera_left",
    )
    frozen = [{
        "relative_path": f"mv_preprocess/labeled_bboxes/{camera}.json",
        "size": 12, "etag": f"etag-{camera}",
    } for camera in cameras]
    current = [{
        **item, "schema_valid": True, "sha256": "a" * 64,
    } for item in frozen]
    request = {
        "id": 9, "stage": "preprocess", "queue_priority": 100,
        "sequence_name": "sequence",
        "source_manifest_json": json.dumps({
            "route": "reprocessing_missing_revalidation_input",
            "membership_transfer": {"source_campaign": "legacy_revalidation"},
            "labeled_bboxes_ready": True,
            "historical_preprocess_lineage": {
                "preprocess_run_id": 77, "reuse_permitted": False,
            },
            "data_output_objects": frozen,
        }),
    }
    writes = []

    class Client:
        def put_object(self, **kwargs):
            writes.append(kwargs)
            return {"ETag": '"marker-etag"'}

    monkeypatch.setattr(
        controller, "_canonical_preprocess_output_url",
        lambda *_args: "swift://host/account/bucket/root/sequence",
    )
    monkeypatch.setattr(
        controller, "_client", lambda _uri: (Client(), "bucket", "root/sequence"),
    )
    monkeypatch.setattr(controller, "require_submit_authority", lambda _reason: None)

    objects, evidence = controller._restore_transferred_manual_bbox_marker(
        {}, request, current,
    )

    assert writes[0]["Body"] == b"manual_labeled_bboxes\n"
    assert objects[-1]["content"] == "manual_labeled_bboxes"
    assert evidence["historical_preprocess_run_id"] == 77
    assert evidence["reuse_permitted"] is False

    changed = [dict(item) for item in current]
    changed[0]["etag"] = "changed"
    writes.clear()
    unchanged, evidence = controller._restore_transferred_manual_bbox_marker(
        {}, request, changed,
    )
    assert unchanged == changed
    assert evidence is None
    assert writes == []


def test_advance_backlog_blocks_invalid_manual_bbox_schema(tmp_path, monkeypatch):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    cameras = (
        "back_stereo_camera_left", "front_stereo_camera_left",
        "left_stereo_camera_left", "right_stereo_camera_left",
    )
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing",
        "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap"}],
        }],
    }
    path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    controller.enqueue_campaign("campaign", path, db_path=db_path)
    preprocess = db.list_stage_requests(campaign="campaign", db_path=db_path)[0]
    db.update_stage_request(preprocess["id"], status="SUCCEEDED", db_path=db_path)
    bboxes = [{
        "relative_path": f"mv_preprocess/labeled_bboxes/{camera}.json",
        "size": 12, "etag": camera, "sha256": "a" * 64,
        "schema_valid": camera != "front_stereo_camera_left",
        "schema_error": "bbox coordinates must be ordered and nonnegative",
    } for camera in cameras]
    monkeypatch.setattr(
        controller, "_current_sequence_objects",
        lambda *_: bboxes + [{
            "relative_path": "mv_preprocess/object_bbox_source.txt",
            "content": "manual_labeled_bboxes",
        }],
    )

    result = controller.advance_backlog(
        "campaign", db_path=db_path,
        config={"datasets": {"dataset": {
            "swift_base": "swift://host/account/bucket/root",
            "pipelines": {
                "mv_preprocess": {
                    "output_path": "data_output",
                    "campaign_output_path": "data_output",
                },
            },
        }}},
    )

    assert result[0]["status"] == "BLOCKED"
    assert result[0]["blocked_reason"] == "WAITING_LABELS"
    request = db.get_active_stage_request(
        "dataset", "sequence", "reconstruction", db_path=db_path,
    )
    assert "invalid manual bbox schema" in request["details"]


@pytest.mark.parametrize("execution_status", ["RUNNING", "FAILED"])
def test_backlog_candidate_is_published_without_terminalizing_shared_execution(
    tmp_path, monkeypatch, execution_status,
):
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    inventory = {
        "dataset": "dataset", "kind": "backlog_reprocessing",
        "sequence_count": 1,
        "sequences": [{
            "sequence": "sequence", "recommended_stage": "preprocess",
            "raw_input_objects": [{"relative_path": "recording.mcap"}],
        }],
    }
    _path, db_path = _campaign(tmp_path, "BACKLOG_REPROCESSING", inventory)
    campaign = db.get_campaign("campaign", db_path=db_path)
    reconstruction = db.insert_workflow(
        sequence_name="sequence", dataset="dataset",
        pipeline_type=db.RECONSTRUCTION_STAGE, pipeline_version="1.0.0",
        workflow_name="legacy-reconstruction", status="PASS",
        trigger="migration", db_path=db_path,
    )
    review_id = db.record_qc_review(
        reconstruction["stage_run_id"], "PASS", source="test", db_path=db_path,
    )
    request = db.create_stage_request(
        sequence_name="sequence", dataset="dataset", stage="export",
        pipeline_version="1.0.0", campaign=campaign["id"], cohort="BULK",
        parameters={
            "candidate_uri": "swift://host/account/bucket/work/sequence",
            "output_uri": "swift://host/account/bucket/data_export_2/sequence",
        },
        db_path=db_path,
    )
    reservation = db.reserve_request_with_execution(
        request["id"], reserved_by="dispatcher", workflow_name="export-workflow",
        pipeline_version="1.0.0", workflow_spec_path="export.yaml", pool="pool",
        db_path=db_path,
    )
    db.apply_execution_observation(
        reservation["execution_id"], execution_status="RUNNING",
        request_outcomes=[{"request_id": request["id"], "status": "RUNNING"}],
        db_path=db_path,
    )
    run = db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0", status="RUNNING",
        execution_id=reservation["execution_id"], request_id=request["id"],
        reconstruction_run_id=reconstruction["stage_run_id"],
        qc_review_id=review_id, authorization_type="QC",
        output_uri="swift://host/account/bucket/data_export_2/sequence",
        set_current=False, db_path=db_path,
    )
    db.update_stage_run(
        run["stage_run_id"], stage=db.EXPORT_STAGE, status="RUNNING",
        details="candidate_export_verified", db_path=db_path,
    )
    db.upsert_blacklisted_sequence(
        "dataset", "sequence", reason="legacy failure", db_path=db_path,
    )
    commit = {
        "schema": "v2d.mv_hoi.export_commit.v1", "complete": True,
        "sequence_name": "sequence", "pipeline_version": "1.0.0",
        "reconstruction_run_id": reconstruction["stage_run_id"],
        "file_count": 1, "total_bytes": 3, "files": [],
    }
    monkeypatch.setattr(controller, "_read_commit", lambda *args, **kwargs: (commit, "c" * 64))
    monkeypatch.setattr(controller, "_client", lambda *_: (object(), "bucket", "prefix"))
    monkeypatch.setattr(controller, "_list", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(controller, "require_submit_authority", lambda *_: None)
    monkeypatch.setattr(
        controller, "_publish_committed_prefix",
        lambda *args, **kwargs: (commit, "c" * 64),
    )
    if execution_status == "FAILED":
        db.apply_execution_observation(
            reservation["execution_id"], execution_status="FAILED",
            db_path=db_path,
        )

    result = controller.publish_backlog_exports(
        "campaign", db_path=db_path, verify_payload_hashes=True,
    )

    assert result == [{
        "request_id": request["id"], "status": "SUCCEEDED",
        "export_run_id": run["stage_run_id"],
        "blacklist_cleared": True,
    }]
    assert db.get_stage_request(request["id"], db_path=db_path)["status"] == "SUCCEEDED"
    assert (
        db.get_workflow_execution(reservation["execution_id"], db_path=db_path)["status"]
        == execution_status
    )
    assert db.get_blacklisted_sequence("dataset", "sequence", db_path=db_path) is None


def test_committed_prefix_publish_resumes_valid_partial_destination(monkeypatch):
    commit = {
        "files": [
            {"path": "already.bin", "size": 3, "sha256": "a" * 64},
            {"path": "missing.bin", "size": 4, "sha256": "b" * 64},
        ],
    }
    copied = []

    class Client:
        def copy_object(self, **kwargs):
            copied.append((kwargs["Key"], kwargs["CopySource"]["Key"]))

    client = Client()
    monkeypatch.setattr(
        controller, "_client",
        lambda url: (
            client, "bucket", "source" if url == "source" else "destination",
        ),
    )
    monkeypatch.setattr(
        controller, "_list",
        lambda *_: [{
            "key": "destination/already.bin", "size": 3, "etag": "etag",
        }],
    )
    monkeypatch.setattr(
        controller, "_read_commit",
        lambda url, **_: (commit, "c" * 64),
    )
    monkeypatch.setattr(
        controller, "cleanup_promoted_export_candidate", lambda *_args, **_kwargs: {},
    )

    published, digest = controller._publish_committed_prefix(
        "source", "destination",
    )

    assert published == commit
    assert digest == "c" * 64
    assert copied == [
        ("destination/missing.bin", "source/missing.bin"),
        ("destination/commit.json", "source/commit.json"),
    ]


def test_committed_prefix_uses_multipart_copy_for_large_payload(monkeypatch):
    large_size = controller.SINGLE_COPY_MAX_BYTES + 1
    commit = {
        "files": [{"path": "large.bin", "size": large_size, "sha256": "a" * 64}],
    }
    calls = []

    class Client:
        def create_multipart_upload(self, **kwargs):
            calls.append(("create", kwargs))
            return {"UploadId": "upload"}

        def upload_part_copy(self, **kwargs):
            calls.append(("part", kwargs))
            return {"CopyPartResult": {"ETag": f"etag-{kwargs['PartNumber']}"}}

        def complete_multipart_upload(self, **kwargs):
            calls.append(("complete", kwargs))

        def abort_multipart_upload(self, **kwargs):
            calls.append(("abort", kwargs))

        def copy_object(self, **kwargs):
            calls.append(("copy", kwargs))

    client = Client()
    monkeypatch.setattr(
        controller, "_client", lambda url: (
            client, "bucket", "source" if url == "source" else "destination",
        ),
    )
    monkeypatch.setattr(controller, "_list", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(controller, "_read_commit", lambda *_args, **_kwargs: (commit, "c" * 64))
    monkeypatch.setattr(
        controller, "cleanup_promoted_export_candidate", lambda *_args, **_kwargs: {},
    )

    controller._publish_committed_prefix("source", "destination")

    part_calls = [kwargs for kind, kwargs in calls if kind == "part"]
    assert len(part_calls) == 11
    assert part_calls[0]["CopySourceRange"] == "bytes=0-536870911"
    assert part_calls[-1]["CopySourceRange"] == (
        f"bytes={10 * controller.MULTIPART_COPY_PART_BYTES}-{large_size - 1}"
    )
    assert any(kind == "complete" for kind, _kwargs in calls)
    assert not any(kind == "abort" for kind, _kwargs in calls)
    assert [kwargs["Key"] for kind, kwargs in calls if kind == "copy"] == [
        "destination/commit.json",
    ]


def test_committed_prefix_publish_rejects_unexpected_partial_key(monkeypatch):
    commit = {"files": []}

    class Client:
        def copy_object(self, **_kwargs):
            raise AssertionError("unexpected destination must not be modified")

    monkeypatch.setattr(
        controller, "_client", lambda url: (
            Client(), "bucket", "source" if url == "source" else "destination",
        ),
    )
    monkeypatch.setattr(
        controller, "_list",
        lambda *_: [{"key": "destination/stray.bin", "size": 1, "etag": "etag"}],
    )
    monkeypatch.setattr(
        controller, "_read_commit", lambda *_args, **_kwargs: (commit, "c" * 64),
    )

    with pytest.raises(ValueError, match="unexpected keys"):
        controller._publish_committed_prefix("source", "destination")


def test_matching_destination_without_commit_is_resumable(monkeypatch):
    monkeypatch.setattr(
        controller, "_client", lambda *_args: (object(), "bucket", "destination"),
    )
    monkeypatch.setattr(
        controller, "_object_map",
        lambda *_args, **_kwargs: {"partial.bin": {"size": 3, "etag": "etag"}},
    )
    monkeypatch.setattr(
        controller, "_read_commit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("partial publication has no commit to read")
        ),
    )

    assert controller.matching_destination_commit(
        {}, {}, "destination",
    ) is None


def test_campaign_cycle_owns_refresh_dispatch_and_filtered_export(monkeypatch):
    calls = []
    pool_snapshots = []
    monkeypatch.setattr(
        controller.PoolSelector, "collect",
        lambda *args, **kwargs: pool_snapshots.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(
        controller, "recover_terminal_campaign_infrastructure_failures",
        lambda name, **kwargs: (
            calls.append(("infrastructure_retry", name))
                or [{
                    "request_id": 11 if name == "legacy" else 22,
                    "status": "RETRY_CREATED",
                }]
        ),
    )
    monkeypatch.setattr(
        controller, "reconcile_revalidation",
        lambda name, **kwargs: calls.append(
            (
                "reconcile", name, kwargs.get("skip_request_ids"),
                kwargs["recover_publication_failures"],
                kwargs.get("defer_bulk_publication", False),
                kwargs.get("pending_publication_only", False),
            )
        ) or [],
    )
    monkeypatch.setattr(
        controller, "publish_canary_results",
        lambda name, **kwargs: calls.append(("canary_publish", name)) or [],
    )
    monkeypatch.setattr(
        controller, "dispatch_revalidation",
        lambda name, **kwargs: calls.append(("revalidation_dispatch", name)) or [],
    )
    monkeypatch.setattr(
        controller.db, "get_campaign",
        lambda name, **kwargs: {"name": name, "dataset": "dataset"},
    )
    monkeypatch.setattr(
        controller, "refresh_workflow_states",
        lambda dataset, **kwargs: calls.append(
            (
                "refresh", dataset, kwargs["pipeline_type"],
                kwargs["skip_request_ids"],
            )
        ),
    )
    monkeypatch.setattr(
        controller, "publish_backlog_exports",
        lambda name, **kwargs: calls.append(("backlog_publish", name)) or [],
    )
    monkeypatch.setattr(
        controller, "advance_backlog",
        lambda name, **kwargs: calls.append(("backlog_advance", name)) or [],
    )
    monkeypatch.setattr(
        controller, "dispatch_backlog",
        lambda name, **kwargs: calls.append(("backlog_dispatch", name)) or [],
    )
    monkeypatch.setattr(
        controller.export_controller, "run_export",
        lambda dataset, cfg, **kwargs: calls.append(
            ("filtered_export", dataset, kwargs["campaign"], kwargs["refresh"])
        ),
    )

    controller.run_campaign_cycle(
        revalidation_campaign="legacy", backlog_campaign="backlog",
        db_path="db.sqlite", config={"datasets": {"dataset": {}}},
    )

    assert calls[:3] == [
        ("infrastructure_retry", "legacy"),
        ("reconcile", "legacy", {11}, True, True, False),
        ("revalidation_dispatch", "legacy"),
    ]
    assert ("infrastructure_retry", "backlog") in calls
    assert ("backlog_publish", "backlog") in calls
    assert ("backlog_advance", "backlog") in calls
    assert ("backlog_dispatch", "backlog") in calls
    assert ("refresh", "dataset", None, {22}) in calls
    assert calls.index(("revalidation_dispatch", "legacy")) < calls.index(
        ("backlog_publish", "backlog")
    )
    assert calls.index(("backlog_dispatch", "backlog")) < calls.index(
        ("backlog_publish", "backlog")
    )
    assert ("filtered_export", "dataset", "backlog", False) in calls
    assert calls[-2:] == [
        ("reconcile", "legacy", None, True, False, True),
        ("canary_publish", "legacy"),
    ]
    assert len(pool_snapshots) == 3


def test_campaign_cycle_cleanup_is_explicitly_config_gated(monkeypatch):
    calls = []
    enqueued = []
    monkeypatch.setattr(
        controller, "recover_terminal_campaign_infrastructure_failures",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(controller, "reconcile_revalidation", lambda *_a, **_k: [])
    monkeypatch.setattr(
        controller, "publish_canary_results",
        lambda *_a, **_k: [{"status": "PUBLISHED", "export_run_id": 77}],
    )
    monkeypatch.setattr(controller, "dispatch_revalidation", lambda *_a, **_k: [])
    monkeypatch.setattr(
        controller.db, "get_campaign",
        lambda *_a, **_k: {"name": "campaign", "dataset": "dataset"},
    )
    monkeypatch.setattr(
        controller.cleanup_intermediates, "run_cleanup_jobs",
        lambda *args, **kwargs: calls.append((args, kwargs)) or [],
    )
    monkeypatch.setattr(
        controller.db, "enqueue_missing_intermediate_cleanups",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        controller.db, "enqueue_intermediate_cleanup",
        lambda export_run_id, **_k: enqueued.append(export_run_id) or {},
    )
    config = {"datasets": {"dataset": {"pipelines": {
        "mv_hoi_export": {"cleanup_intermediates_after_export": False},
    }}}}
    controller.run_campaign_cycle(
        revalidation_campaign="campaign", backlog_campaign=None,
        db_path="db", config=config,
    )
    assert calls == []
    assert enqueued == []

    config["datasets"]["dataset"]["pipelines"]["mv_hoi_export"][
        "cleanup_intermediates_after_export"
    ] = {
        "enabled": True,
        "campaign_allowlist": ["campaign"],
        "automatic_export_cutoff": "2026-08-01T00:00:00Z",
        "mode": "inline",
        "max_per_cycle": 2,
        "workers": 2,
    }
    controller.run_campaign_cycle(
        revalidation_campaign="campaign", backlog_campaign=None,
        db_path="db", config=config,
    )
    assert calls == [(('dataset',), {
        "campaign": "campaign", "db_path": "db", "apply": True,
        "limit": 2, "workers": 2,
    })]
    assert enqueued == [77]

    calls.clear()
    enqueued.clear()
    config["datasets"]["dataset"]["pipelines"]["mv_hoi_export"][
        "cleanup_intermediates_after_export"
    ].update({"mode": "asynchronous", "max_per_cycle": 40, "workers": 8})
    result = controller.run_campaign_cycle(
        revalidation_campaign="campaign", backlog_campaign=None,
        db_path="db", config=config,
    )
    assert calls == []
    assert enqueued == [77]
    assert result["revalidation_cleanup"] == "deferred_to_async_worker"
