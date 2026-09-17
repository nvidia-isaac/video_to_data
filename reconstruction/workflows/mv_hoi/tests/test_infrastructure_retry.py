from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys

import pytest


WORKFLOW_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_DIR))

from orchestration import campaign_controller, db, query


def _failure_info(*, retry_id: int = 0, exit_code: int = 2031) -> dict:
    detail = {
        "name": "face_detector",
        "retry_id": retry_id,
        "status": "FAILED",
        "failure_message": (
            "Failure reason: Exit code 2031 due to OSMO Control failure."
            if exit_code == 2031
            else "application failed"
        ),
        "exit_code": exit_code,
    }
    return {
        "status": "FAILED",
        "tasks": {"face_detector": "FAILED"},
        "task_details": {"face_detector": detail},
        "task_attempts": {"face_detector": [detail]},
    }


def _temporary_handoff_failure_info(*, upstream_completed: bool = True) -> dict:
    detail = {
        "name": "face_detector",
        "retry_id": 0,
        "status": "FAILED",
        "failure_message": "application failed",
        "exit_code": 75,
    }
    tasks = {"face_detector": "FAILED"}
    if upstream_completed:
        tasks["mv_preprocess"] = "COMPLETED"
    return {
        "status": "FAILED",
        "tasks": tasks,
        "task_details": {"face_detector": detail},
        "task_attempts": {"face_detector": [detail]},
    }


def _start_timeout_info(*, retry_id: int = 0) -> dict:
    detail = {
        "name": "validate_source",
        "retry_id": retry_id,
        "status": "FAILED_START_TIMEOUT",
        "failure_message": None,
        "exit_code": 3005,
    }
    return {
        "status": "FAILED",
        "tasks": {"validate_source": "FAILED_START_TIMEOUT"},
        "task_details": {"validate_source": detail},
        "task_attempts": {"validate_source": [detail]},
    }


def test_start_timeout_is_retryable_infrastructure_failure() -> None:
    info = _start_timeout_info(retry_id=1)

    assert query.is_retryable_infrastructure_failure(info)
    assert query.infrastructure_restart_number(info) == 2


def test_preprocess_output_incomplete_is_retryable_infrastructure_failure() -> None:
    info = {
        "status": "FAILED",
        "tasks": {"mv_preprocess": "COMPLETED"},
        "preprocess_output_incomplete": True,
    }

    assert query.is_retryable_infrastructure_failure(info)
    assert query.infrastructure_restart_number(info) == 1


def test_remote_preprocess_output_requires_every_nonempty_runtime_asset(
    monkeypatch,
) -> None:
    prefix = "data_output/sequence/mv_preprocess"
    objects = [
        {"key": f"{prefix}/{path}", "size": 1}
        for path in query.PREPROCESS_REQUIRED_OUTPUTS
    ]
    monkeypatch.setattr(
        query, "_client", lambda *_args, **_kwargs: (object(), "recordings", prefix),
    )
    monkeypatch.setattr(query, "_list", lambda *_args, **_kwargs: objects)

    result = query.validate_remote_preprocess_output(
        "swift://host/account/recordings/data_output/sequence/mv_preprocess"
    )

    assert result["required_file_count"] == len(query.PREPROCESS_REQUIRED_OUTPUTS)

    objects.pop()
    with pytest.raises(ValueError, match="remote preprocess output is incomplete"):
        query.validate_remote_preprocess_output(
            "swift://host/account/recordings/data_output/sequence/mv_preprocess"
        )


def test_image_pull_is_retryable_infrastructure_failure() -> None:
    detail = {
        "name": "foundation_pose",
        "retry_id": 0,
        "status": "FAILED_IMAGE_PULL",
        "failure_message": "Task failed with ErrImagePull",
        "exit_code": 302,
    }
    upstream = {
        "name": "export_revalidated",
        "retry_id": 0,
        "status": "FAILED_UPSTREAM",
        "failure_message": "Upstream task failed.",
        "exit_code": 3000,
    }
    info = {
        "status": "FAILED",
        "tasks": {
            "foundation_pose": "FAILED_IMAGE_PULL",
            "export_revalidated": "FAILED_UPSTREAM",
        },
        "task_details": {
            "foundation_pose": detail,
            "export_revalidated": upstream,
        },
        "task_attempts": {
            "foundation_pose": [detail],
            "export_revalidated": [upstream],
        },
    }

    assert query.is_retryable_infrastructure_failure(info)
    assert query.infrastructure_restart_number(info) == 1


def test_node_resource_rejection_is_retryable_infrastructure_failure() -> None:
    detail = {
        "name": "eval_silhouette_mask_object",
        "retry_id": 0,
        "status": "FAILED",
        "failure_message": (
            "Pod 4a6cfecdca074759-96969c8200df42da error message: "
            "Pod was rejected: Node didn't have enough resource: cpu, "
            "requested: 9000, used: 85975, capacity: 94940\n"
        ),
        "exit_code": 4000,
    }
    info = {
        "status": "FAILED",
        "tasks": {"eval_silhouette_mask_object": "FAILED"},
        "task_details": {"eval_silhouette_mask_object": detail},
        "task_attempts": {"eval_silhouette_mask_object": [detail]},
    }

    assert query.is_retryable_infrastructure_failure(info)
    assert query.infrastructure_restart_number(info) == 1


@pytest.mark.parametrize("resource", ["nvidia.com/mlnxnics", "nvidia.com/gpu"])
def test_unhealthy_device_backend_rejection_is_retryable(
    resource: str,
) -> None:
    detail = {
        "name": "eval_silhouette_mask_object",
        "retry_id": 0,
        "status": "FAILED_BACKEND_ERROR",
        "failure_message": (
            "Pod 21bf61e5e55241cc-cc7dbe5c79a14758 error message: "
            "Pod was rejected: Allocate failed due to no healthy devices present; "
            f"cannot allocate unhealthy devices {resource}, which is unexpected\n"
        ),
        "exit_code": 3001,
    }
    upstream = {
        "name": "check_accuracy",
        "retry_id": 0,
        "status": "FAILED_UPSTREAM",
        "failure_message": "Upstream task failed.",
        "exit_code": 3000,
    }
    info = {
        "status": "FAILED",
        "tasks": {
            "eval_silhouette_mask_object": "FAILED_BACKEND_ERROR",
            "check_accuracy": "FAILED_UPSTREAM",
        },
        "task_details": {
            "eval_silhouette_mask_object": detail,
            "check_accuracy": upstream,
        },
        "task_attempts": {
            "eval_silhouette_mask_object": [detail],
            "check_accuracy": [upstream],
        },
    }

    assert query.is_retryable_infrastructure_failure(info)
    assert query.infrastructure_restart_number(info) == 1


def test_generic_backend_error_is_not_retryable() -> None:
    detail = {
        "name": "eval_silhouette_mask_object",
        "retry_id": 0,
        "status": "FAILED_BACKEND_ERROR",
        "failure_message": "Backend rejected the task for an unknown reason",
        "exit_code": 3001,
    }
    info = {
        "status": "FAILED",
        "tasks": {"eval_silhouette_mask_object": "FAILED_BACKEND_ERROR"},
        "task_details": {"eval_silhouette_mask_object": detail},
        "task_attempts": {"eval_silhouette_mask_object": [detail]},
    }

    assert not query.is_retryable_infrastructure_failure(info)


def test_failed_upstream_is_not_a_retryable_root_failure() -> None:
    info = _start_timeout_info()
    detail = info["task_details"]["validate_source"]
    detail["status"] = "FAILED_UPSTREAM"
    detail["exit_code"] = 3000

    assert not query.is_retryable_infrastructure_failure(info)


def test_start_timeout_mixed_with_application_failure_is_not_retryable() -> None:
    info = _start_timeout_info()
    application = {
        "name": "check_accuracy",
        "retry_id": 0,
        "status": "FAILED",
        "failure_message": "quality gate failed",
        "exit_code": 1,
    }
    info["tasks"]["check_accuracy"] = "FAILED"
    info["task_details"]["check_accuracy"] = application
    info["task_attempts"]["check_accuracy"] = [application]

    assert not query.is_retryable_infrastructure_failure(info)


def test_incomplete_preprocess_handoff_is_narrowly_retryable() -> None:
    assert query.is_retryable_infrastructure_failure(
        _temporary_handoff_failure_info()
    )
    assert not query.is_retryable_infrastructure_failure(
        _temporary_handoff_failure_info(upstream_completed=False)
    )
    info = _temporary_handoff_failure_info()
    info["task_details"]["face_detector"]["name"] = "some_other_task"
    assert not query.is_retryable_infrastructure_failure(info)


@pytest.mark.parametrize("signature", [
    "FileNotFoundError: missing input",
    "filenotfounderror: missing input",
    "OSError: [Errno 2] No such file or directory: '/input/edex'",
    "bash: /input/edex: No such file or directory",
])
def test_file_not_found_task_log_is_retryable(
    monkeypatch, signature: str,
) -> None:
    info = _failure_info(exit_code=1)
    monkeypatch.setattr(
        query, "_osmo_task_log_tail",
        lambda *_args, **_kwargs: signature,
    )

    query._classify_file_not_found_failures("workflow-1", info)

    detail = info["task_details"]["face_detector"]
    assert detail["retryable_failure_category"] == "file_not_found"
    assert detail["retryable_failure_evidence"]["excerpt"] == signature
    assert len(detail["retryable_failure_evidence"]["sha256"]) == 64
    assert query.is_retryable_infrastructure_failure(info)


@pytest.mark.parametrize("signature", [
    "EntityTooLarge: Upload exceeds quota for this account",
    "HTTP 429 Too Many Requests while uploading to CSS object storage",
    "RetriesExceededError: Max Retries Exceeded",
])
def test_css_transport_task_log_is_retryable(monkeypatch, signature: str) -> None:
    info = _failure_info(exit_code=1)
    monkeypatch.setattr(
        query, "_osmo_task_log_tail", lambda *_args, **_kwargs: signature,
    )

    query._classify_file_not_found_failures("workflow-1", info)

    detail = info["task_details"]["face_detector"]
    assert detail["retryable_failure_category"] == "css_transport"
    assert query.is_retryable_infrastructure_failure(info)


def test_export_task_failure_info_ignores_failed_siblings() -> None:
    infra = {
        "name": "export_infra", "retry_id": 0,
        "status": "FAILED_IMAGE_PULL", "failure_message": "image pull",
        "exit_code": 302,
    }
    qc = {
        "name": "export_qc", "retry_id": 0,
        "status": "FAILED", "failure_message": "human QC gate",
        "exit_code": 1,
    }
    info = {
        "status": "FAILED",
        "tasks": {
            "export_ok": "COMPLETED",
            "export_infra": "FAILED_IMAGE_PULL",
            "export_qc": "FAILED",
        },
        "task_details": {"export_infra": infra, "export_qc": qc},
    }

    scoped = query.export_task_failure_info(info, ["export_infra"])

    assert scoped["tasks"] == {"export_infra": "FAILED_IMAGE_PULL"}
    assert query.is_retryable_infrastructure_failure(scoped)
    assert not query.is_retryable_infrastructure_failure(
        query.export_task_failure_info(info, ["export_qc"])
    )


def test_osmo_query_enriches_file_not_found_from_error_log(monkeypatch) -> None:
    payload = {
        "status": "FAILED",
        "groups": [{
            "tasks": [{
                "name": "mv_preprocess",
                "retry_id": 0,
                "status": "FAILED",
                "failure_message": "Exit code 1",
                "exit_code": 1,
            }],
        }],
    }
    commands = []

    def run(command, **_kwargs):
        commands.append(command)
        if command[2] == "query":
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(
            command, 0,
            "FileNotFoundError: [Errno 2] No such file or directory: '/input/edex'",
            "",
        )

    monkeypatch.setattr(query.subprocess, "run", run)

    info = query.osmo_query("workflow-1")

    assert [command[2] for command in commands] == ["query", "logs"]
    assert info["task_details"]["mv_preprocess"][
        "retryable_failure_category"
    ] == "file_not_found"
    assert query.is_retryable_infrastructure_failure(info)


def test_file_not_found_classification_falls_back_to_regular_log(
    monkeypatch,
) -> None:
    calls = []

    def log_tail(_workflow, _detail, *, error):
        calls.append(error)
        return None if error else "FileNotFoundError: missing handoff"

    info = _failure_info(exit_code=1)
    monkeypatch.setattr(query, "_osmo_task_log_tail", log_tail)

    query._classify_file_not_found_failures("workflow-1", info)

    assert calls == [True, False]
    evidence = info["task_details"]["face_detector"][
        "retryable_failure_evidence"
    ]
    assert evidence["source"] == "osmo_task_log"


def test_existing_infrastructure_failure_does_not_fetch_logs(monkeypatch) -> None:
    monkeypatch.setattr(
        query, "_osmo_task_log_tail",
        lambda *_args, **_kwargs: pytest.fail("unexpected log fetch"),
    )

    query._classify_file_not_found_failures(
        "workflow-1", _start_timeout_info(),
    )


def test_file_not_found_requires_every_root_failure_to_be_retryable(
    monkeypatch,
) -> None:
    info = _start_timeout_info()
    application = {
        "name": "check_accuracy",
        "retry_id": 0,
        "status": "FAILED",
        "failure_message": "application failed",
        "exit_code": 1,
    }
    info["tasks"]["check_accuracy"] = "FAILED"
    info["task_details"]["check_accuracy"] = application
    info["task_attempts"]["check_accuracy"] = [application]
    monkeypatch.setattr(
        query, "_osmo_task_log_tail",
        lambda *_args, **_kwargs: "ValueError: deterministic failure",
    )

    query._classify_file_not_found_failures("workflow-1", info)

    assert not query.is_retryable_infrastructure_failure(info)


def test_task_log_timeout_is_not_classified(monkeypatch) -> None:
    monkeypatch.setattr(
        query.subprocess, "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("osmo", 30)
        ),
    )

    assert query._osmo_task_log_tail(
        "workflow-1", _failure_info(exit_code=1)["task_details"]["face_detector"],
        error=True,
    ) is None


def _running_campaign_request(
    tmp_path: Path, *, campaign_type: str = "LEGACY_REVALIDATION",
    request_stage: str = "revalidation",
) -> tuple[str, dict, int]:
    db_path = str(tmp_path / "db.sqlite")
    db.init_db(db_path)
    campaign = db.create_campaign(
        name="campaign",
        campaign_type=campaign_type,
        dataset="dataset",
        pipeline_version="1.0.0",
        output_uri="swift://host/account/bucket/data_export_2",
        created_by="pytest",
        db_path=db_path,
    )
    campaign = db.freeze_campaign(
        campaign["id"],
        inventory_uri="swift://host/account/bucket/inventory.json",
        inventory_sha256="a" * 64,
        configuration_uri="swift://host/account/bucket/configuration.json",
        configuration_sha256="b" * 64,
        inventory_sequence_count=1,
        db_path=db_path,
    )
    request = db.create_stage_request(
        sequence_name="sequence",
        dataset="dataset",
        stage=request_stage,
        pipeline_version="1.0.0",
        campaign=campaign["id"],
        cohort="CANARY",
        queue_priority=100,
        source_manifest={"sequence": "sequence"},
        db_path=db_path,
    )
    reservation = db.reserve_request_with_execution(
        request["id"],
        reserved_by="pytest",
        workflow_name="workflow",
        pipeline_version="1.0.0",
        workflow_spec_path="workflow.yaml",
        pool="pool",
        db_path=db_path,
    )
    db.apply_execution_observation(
        reservation["execution_id"],
        execution_status="RUNNING",
        osmo_workflow_id="workflow-1",
        request_outcomes=[{
            "request_id": request["id"],
            "status": "RUNNING",
        }],
        db_path=db_path,
    )
    return db_path, request, reservation["execution_id"]


def test_completed_preprocess_with_missing_remote_output_is_retried(
    tmp_path, monkeypatch,
) -> None:
    db_path, request, execution_id = _running_campaign_request(
        tmp_path,
        campaign_type="BACKLOG_REPROCESSING",
        request_stage="preprocess",
    )
    run = db.create_stage_run(
        sequence_name="sequence",
        dataset="dataset",
        stage=db.PREPROCESS_STAGE,
        pipeline_version="1.0.0",
        status="RUNNING",
        details="workflow_running",
        trigger="MIGRATION",
        execution_id=execution_id,
        request_id=request["id"],
        output_uri=(
            "swift://host/account/bucket/data_output/sequence/mv_preprocess"
        ),
        db_path=db_path,
    )
    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _id: {
            "status": "COMPLETED",
            "tasks": {"mv_preprocess": "COMPLETED"},
        },
    )
    monkeypatch.setattr(
        query,
        "validate_remote_preprocess_output",
        lambda _uri: (_ for _ in ()).throw(ValueError("missing images")),
    )

    query.refresh_waiting(
        "dataset",
        pipeline_type=db.PREPROCESS_STAGE,
        db_path=db_path,
        retry_infrastructure=True,
    )

    assert (
        db.get_workflow_execution(execution_id, db_path=db_path)["status"]
        == "FAILED"
    )
    assert db.get_stage_run(
        run["stage_run_id"], stage=db.PREPROCESS_STAGE, db_path=db_path,
    )["run_status"] == "FAILED"
    assert db.get_stage_request(request["id"], db_path=db_path)["status"] == "FAILED"
    replacements = db.list_stage_requests(
        campaign="campaign", stage="preprocess", status="PENDING", db_path=db_path,
    )
    assert len(replacements) == 1
    assert replacements[0]["reason"] == (
        f"infrastructure_retry_of_request_{request['id']}"
    )


def test_osmo_query_preserves_latest_task_retry_details(monkeypatch) -> None:
    payload = {
        "status": "RUNNING",
        "groups": [{
            "tasks": [
                {
                    "name": "face_detector",
                    "retry_id": 0,
                    "status": "FAILED",
                    "failure_message": "OSMO Control failure",
                    "exit_code": 2031,
                },
                {
                    "name": "face_detector",
                    "retry_id": 1,
                    "status": "RUNNING",
                    "failure_message": None,
                    "exit_code": None,
                },
            ],
        }],
    }
    captured = {}

    def run(command, **_kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload))

    monkeypatch.setattr(query.subprocess, "run", run)

    result = query.osmo_query("workflow-1")

    assert "--verbose" in captured["command"]
    assert result["tasks"]["face_detector"] == "RUNNING"
    assert result["task_details"]["face_detector"]["retry_id"] == 1
    assert len(result["task_attempts"]["face_detector"]) == 2


def test_retryable_failure_waits_without_authority_or_blacklist(
    tmp_path,
) -> None:
    db_path, request, execution_id = _running_campaign_request(tmp_path)

    result = query.handle_campaign_infrastructure_failure(
        workflow_id="workflow-1",
        execution_id=execution_id,
        dataset="dataset",
        sequence_name="sequence",
        campaign_name="campaign",
        campaign_type="LEGACY_REVALIDATION",
        info=_failure_info(),
        retry_infrastructure=False,
        db_path=db_path,
    )

    assert result == "RETRY_PENDING"
    assert db.get_workflow_execution(execution_id, db_path=db_path)["status"] == "UNKNOWN"
    assert db.get_stage_request(request["id"], db_path=db_path)["status"] == "UNKNOWN"
    assert db.get_blacklisted_sequence("dataset", "sequence", db_path=db_path) is None


def test_authorized_retry_creates_fresh_request_and_clears_owned_blacklist(
    tmp_path,
) -> None:
    db_path, request, execution_id = _running_campaign_request(tmp_path)
    db.update_campaign_pipeline_version(
        "campaign", "2.0.0", db_path=db_path,
    )
    info = _failure_info()
    db.apply_execution_observation(
        execution_id,
        execution_status="FAILED",
        details="task_failed: face_detector",
        query_payload=info,
        request_outcomes=[{
            "request_id": request["id"],
            "status": "FAILED",
        }],
        db_path=db_path,
    )
    db.upsert_blacklisted_sequence(
        "dataset",
        "sequence",
        "revalidation_failed: task_failed: face_detector",
        created_by="campaign:campaign",
        db_path=db_path,
    )
    result = query.handle_campaign_infrastructure_failure(
        workflow_id="workflow-1",
        execution_id=execution_id,
        dataset="dataset",
        sequence_name="sequence",
        campaign_name="campaign",
        campaign_type="LEGACY_REVALIDATION",
        info=info,
        retry_infrastructure=True,
        db_path=db_path,
    )

    assert result == "RETRY_CREATED"
    execution = db.get_workflow_execution(execution_id, db_path=db_path)
    assert execution["status"] == "FAILED"
    assert execution["osmo_workflow_id"] == "workflow-1"
    assert db.get_stage_request(request["id"], db_path=db_path)["status"] == "FAILED"
    replacement = db.list_stage_requests(
        campaign="campaign", status="PENDING", db_path=db_path,
    )
    assert len(replacement) == 1
    assert replacement[0]["id"] != request["id"]
    assert replacement[0]["pipeline_version"] == "2.0.0"
    assert replacement[0]["queue_priority"] == 100
    replacement_parameters = json.loads(replacement[0]["parameters_json"])
    assert replacement_parameters["retry_not_before"]
    assert replacement_parameters == {
        "infrastructure_retry_attempt": 1,
        "prior_execution_id": execution_id,
        "prior_request_id": request["id"],
        "retry_backoff_seconds": 600,
        "retry_not_before": replacement_parameters["retry_not_before"],
        "work_output_layout": "request_scoped_v1",
    }
    assert db.get_blacklisted_sequence("dataset", "sequence", db_path=db_path) is None


def test_export_retry_replaces_only_one_request_from_shared_execution(tmp_path) -> None:
    db_path, first, execution_id = _running_campaign_request(
        tmp_path, campaign_type="BACKLOG_REPROCESSING", request_stage="export",
    )
    db.create_stage_run(
        sequence_name="sequence", dataset="dataset", stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0", status="RUNNING", trigger="MIGRATION",
        execution_id=execution_id, request_id=first["id"],
        authorization_type="LEGACY", workflow_task_name="export_sequence",
        db_path=db_path,
    )
    second = db.create_stage_request(
        sequence_name="sequence_2", dataset="dataset", stage="export",
        pipeline_version="1.0.0", campaign="campaign", cohort="CANARY",
        queue_priority=50, source_manifest={"sequence": "sequence_2"},
        db_path=db_path,
    )
    assert db.reserve_stage_request(
        second["id"], reserved_by="pytest", db_path=db_path,
    )
    db.attach_request_execution(second["id"], execution_id, db_path=db_path)
    second_run = db.create_stage_run(
        sequence_name="sequence_2", dataset="dataset", stage=db.EXPORT_STAGE,
        pipeline_version="1.0.0", status="RUNNING", trigger="MIGRATION",
        execution_id=execution_id, request_id=second["id"],
        authorization_type="LEGACY", workflow_task_name="export_sequence_2",
        db_path=db_path,
    )

    replacement = db.replace_campaign_request_infrastructure_attempt(
        first["id"], execution_id, details="image pull",
        query_payload={"status": "FAILED"},
        failure_evidence={
            "classification": "failed_image_pull",
            "task": "export_sequence",
        }, max_attempts=2,
        backoff_seconds=(600, 1800), db_path=db_path,
    )

    assert replacement["stage"] == "export"
    assert replacement["status"] == "PENDING"
    assert db.get_stage_request(first["id"], db_path=db_path)["status"] == "FAILED"
    assert db.get_stage_request(second["id"], db_path=db_path)["status"] == "SUBMITTED"
    assert db.get_stage_run(
        second_run["stage_run_id"], stage=db.EXPORT_STAGE, db_path=db_path,
    )["run_status"] == "RUNNING"
    assert json.loads(replacement["parameters_json"])[
        "infrastructure_failure_evidence"
    ] == {
        "classification": "failed_image_pull",
        "task": "export_sequence",
    }
    assert db.replace_campaign_request_infrastructure_attempt(
        first["id"], execution_id, details="image pull",
        query_payload={"status": "FAILED"}, max_attempts=2,
        backoff_seconds=(600, 1800), db_path=db_path,
    )["id"] == replacement["id"]


def test_campaign_recovery_finds_preexisting_terminal_infrastructure_failure(
    tmp_path, monkeypatch,
) -> None:
    db_path, request, execution_id = _running_campaign_request(tmp_path)
    db.apply_execution_observation(
        execution_id,
        execution_status="FAILED",
        details="task_failed: face_detector",
        # This intentionally resembles an observation made before task details
        # were retained; recovery must query OSMO once.
        query_payload={"status": "FAILED", "tasks": {"face_detector": "FAILED"}},
        request_outcomes=[{
            "request_id": request["id"],
            "status": "FAILED",
        }],
        db_path=db_path,
    )
    monkeypatch.setattr(
        campaign_controller,
        "osmo_query",
        lambda _workflow_id, **_kwargs: _failure_info(),
    )
    result = (
        campaign_controller.recover_terminal_campaign_infrastructure_failures(
            "campaign", db_path=db_path, request_ids={request["id"]},
        )
    )

    assert result == [{"request_id": request["id"], "status": "RETRY_CREATED"}]
    assert db.get_workflow_execution(execution_id, db_path=db_path)["status"] == "FAILED"
    assert db.get_stage_request(request["id"], db_path=db_path)["status"] == "FAILED"
    assert len(db.list_stage_requests(
        campaign="campaign", status="PENDING", db_path=db_path,
    )) == 1


def test_campaign_recovery_skips_terminal_campaign(
    tmp_path, monkeypatch,
) -> None:
    db_path, request, execution_id = _running_campaign_request(tmp_path)
    failure = _failure_info()
    db.apply_execution_observation(
        execution_id,
        execution_status="FAILED",
        details="task_failed: face_detector",
        query_payload=failure,
        request_outcomes=[{"request_id": request["id"], "status": "FAILED"}],
        db_path=db_path,
    )
    db.finish_campaign("campaign", db_path=db_path)
    monkeypatch.setattr(
        campaign_controller,
        "osmo_query",
        lambda _workflow_id, **_kwargs: pytest.fail(
            "terminal campaign queried OSMO"
        ),
    )

    assert campaign_controller.recover_terminal_campaign_infrastructure_failures(
        "campaign", db_path=db_path,
    ) == []
    assert db.list_stage_requests(
        campaign="campaign", status="PENDING", db_path=db_path,
    ) == []


def test_campaign_recovery_skips_failure_with_active_successor(
    tmp_path, monkeypatch,
) -> None:
    db_path, request, execution_id = _running_campaign_request(tmp_path)
    failure = _failure_info()
    db.apply_execution_observation(
        execution_id,
        execution_status="FAILED",
        details="task_failed: face_detector",
        query_payload=failure,
        request_outcomes=[{"request_id": request["id"], "status": "FAILED"}],
        db_path=db_path,
    )
    successor = db.create_stage_request(
        sequence_name="sequence",
        dataset="dataset",
        stage="revalidation",
        pipeline_version="1.0.0",
        campaign="campaign",
        cohort="CANARY",
        reason=f"manual_retry_reset_of_request_{request['id']}",
        source_manifest={"sequence": "sequence"},
        db_path=db_path,
    )
    monkeypatch.setattr(
        campaign_controller,
        "osmo_query",
        lambda _workflow_id, **_kwargs: pytest.fail(
            "superseded failure queried OSMO"
        ),
    )

    assert campaign_controller.recover_terminal_campaign_infrastructure_failures(
        "campaign", db_path=db_path, request_ids={request["id"]},
    ) == []
    assert db.get_stage_request(successor["id"], db_path=db_path)["status"] == "PENDING"


def test_targeted_recovery_requeries_previously_unclassified_failure(
    tmp_path, monkeypatch,
) -> None:
    db_path, request, execution_id = _running_campaign_request(
        tmp_path, campaign_type="BACKLOG_REPROCESSING", request_stage="preprocess",
    )
    unclassified = _failure_info(exit_code=1)
    db.apply_execution_observation(
        execution_id,
        execution_status="FAILED",
        details="task_failed: mv_preprocess",
        query_payload=unclassified,
        request_outcomes=[{"request_id": request["id"], "status": "FAILED"}],
        db_path=db_path,
    )
    classified = _failure_info(exit_code=1)
    classified["task_details"]["face_detector"].update({
        "retryable_failure_category": "file_not_found",
        "retryable_failure_evidence": {
            "task": "face_detector",
            "source": "osmo_error_log",
            "excerpt": "FileNotFoundError: missing input",
            "sha256": "a" * 64,
        },
    })
    monkeypatch.setattr(
        campaign_controller,
        "osmo_query",
        lambda _workflow_id, **_kwargs: classified,
    )

    result = campaign_controller.recover_terminal_campaign_infrastructure_failures(
        "campaign", db_path=db_path, request_ids={request["id"]},
    )

    assert result == [{"request_id": request["id"], "status": "RETRY_CREATED"}]
    successors = db.list_stage_requests(
        campaign="campaign", stage="preprocess", status="PENDING", db_path=db_path,
    )
    assert len(successors) == 1
    parameters = json.loads(successors[0]["parameters_json"])
    assert parameters["retry_backoff_seconds"] == 600
    assert parameters["infrastructure_retry_attempt"] == 1


def test_campaign_recovery_rejects_request_from_another_campaign(
    tmp_path,
) -> None:
    db_path, _request, _execution_id = _running_campaign_request(tmp_path)

    with pytest.raises(ValueError, match="do not belong"):
        campaign_controller.recover_terminal_campaign_infrastructure_failures(
            "campaign", db_path=db_path, request_ids={999999},
        )


def test_infrastructure_retry_is_bounded_and_code_failure_is_not_retried(
    tmp_path,
) -> None:
    assert query.MAX_CAMPAIGN_INFRASTRUCTURE_RESTARTS == 2
    assert query.CAMPAIGN_INFRASTRUCTURE_BACKOFF_SECONDS == (600, 1800)
    db_path, _request, execution_id = _running_campaign_request(tmp_path)
    common = {
        "workflow_id": "workflow-1",
        "execution_id": execution_id,
        "dataset": "dataset",
        "sequence_name": "sequence",
        "campaign_name": "campaign",
        "campaign_type": "LEGACY_REVALIDATION",
        "retry_infrastructure": True,
        "db_path": db_path,
    }

    assert query.handle_campaign_infrastructure_failure(
        info=_failure_info(retry_id=2), **common,
    ) == "EXHAUSTED"
    assert query.handle_campaign_infrastructure_failure(
        info=_failure_info(exit_code=1), **common,
    ) is None


def test_osmo_server_error_retries_are_bounded_without_task_metadata(
    tmp_path,
) -> None:
    info = {
        "status": "FAILED_SERVER_ERROR",
        "tasks": {},
        "task_details": {},
    }
    assert query.is_retryable_infrastructure_failure(info)
    db_path, _request, execution_id = _running_campaign_request(tmp_path)
    common = {
        "dataset": "dataset",
        "sequence_name": "sequence",
        "campaign_name": "campaign",
        "campaign_type": "LEGACY_REVALIDATION",
        "info": info,
        "retry_infrastructure": True,
        "db_path": db_path,
    }

    assert query.handle_campaign_infrastructure_failure(
        workflow_id="workflow-1", execution_id=execution_id, **common,
    ) == "RETRY_CREATED"
    first = db.list_stage_requests(
        campaign="campaign", status="PENDING", db_path=db_path,
    )[0]
    first_execution = db.reserve_request_with_execution(
        first["id"], reserved_by="pytest", workflow_name="workflow-retry-1",
        pipeline_version="1.0.0", workflow_spec_path="workflow.yaml",
        pool="pool", db_path=db_path,
    )["execution_id"]
    db.apply_execution_observation(
        first_execution, execution_status="RUNNING",
        osmo_workflow_id="workflow-retry-1-1",
        request_outcomes=[{"request_id": first["id"], "status": "RUNNING"}],
        db_path=db_path,
    )
    assert query.handle_campaign_infrastructure_failure(
        workflow_id="workflow-retry-1-1", execution_id=first_execution, **common,
    ) == "RETRY_CREATED"
    second = db.list_stage_requests(
        campaign="campaign", status="PENDING", db_path=db_path,
    )[0]
    assert json.loads(second["parameters_json"])["retry_backoff_seconds"] == 1800
    second_execution = db.reserve_request_with_execution(
        second["id"], reserved_by="pytest", workflow_name="workflow-retry-2",
        pipeline_version="1.0.0", workflow_spec_path="workflow.yaml",
        pool="pool", db_path=db_path,
    )["execution_id"]
    db.apply_execution_observation(
        second_execution, execution_status="RUNNING",
        osmo_workflow_id="workflow-retry-2-1",
        request_outcomes=[{"request_id": second["id"], "status": "RUNNING"}],
        db_path=db_path,
    )
    assert query.handle_campaign_infrastructure_failure(
        workflow_id="workflow-retry-2-1", execution_id=second_execution, **common,
    ) == "EXHAUSTED"


def test_ambiguous_submission_grace_uses_persisted_timestamp_boundaries() -> None:
    assert query.AMBIGUOUS_SUBMIT_NOT_FOUND_GRACE_SECONDS == 600
    submitted = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
    workflow = {"execution_submitted_at": submitted.isoformat()}

    assert not query._ambiguous_submit_not_found_grace_expired(
        workflow, now=submitted + timedelta(seconds=599),
    )
    assert query._ambiguous_submit_not_found_grace_expired(
        workflow, now=submitted + timedelta(seconds=600),
    )


def test_expired_ambiguous_submission_is_failed_and_retried(
    tmp_path, monkeypatch,
) -> None:
    db_path, request, execution_id = _running_campaign_request(
        tmp_path, campaign_type="BACKLOG_REPROCESSING",
        request_stage="reconstruction",
    )
    run = db.create_stage_run(
        sequence_name="sequence", dataset="dataset",
        stage=db.RECONSTRUCTION_STAGE, pipeline_version="1.0.0",
        status="RUNNING", details="submit_ambiguous: timeout",
        trigger="MIGRATION", execution_id=execution_id,
        request_id=request["id"], db_path=db_path,
    )
    monkeypatch.setattr(
        query,
        "osmo_query",
        lambda _id: {"status": "UNKNOWN", "tasks": {}, "not_found": True},
    )
    monkeypatch.setattr(
        query, "_ambiguous_submit_not_found_grace_expired", lambda _wf: True,
    )

    query.refresh_waiting(
        "dataset", pipeline_type=db.RECONSTRUCTION_STAGE, db_path=db_path,
        retry_infrastructure=True,
    )

    execution = db.get_workflow_execution(execution_id, db_path=db_path)
    failed_request = db.get_stage_request(request["id"], db_path=db_path)
    assert execution["status"] == "FAILED"
    assert failed_request["status"] == "FAILED"
    assert "osmo_submission_not_found_after_grace" in execution["details"]
    persisted_query = json.loads(execution["last_query_payload_json"])
    assert persisted_query == {
        "not_found": True,
        "status": "UNKNOWN",
        "tasks": {},
    }
    assert "FAILED_SERVER_ERROR" not in execution["last_query_payload_json"]
    replacements = db.list_stage_requests(
        campaign="campaign", status="PENDING", db_path=db_path,
    )
    assert len(replacements) == 1
    assert replacements[0]["id"] != request["id"]

    assert query.handle_campaign_infrastructure_failure(
        workflow_id="workflow-1", execution_id=execution_id,
        dataset="dataset", sequence_name="sequence",
        campaign_name="campaign", campaign_type="BACKLOG_REPROCESSING",
        info={"status": "UNKNOWN", "not_found": True, "missing_submission": True},
        retry_infrastructure=True, db_path=db_path,
        failure_reason="osmo_submission_not_found_after_grace",
        query_payload={"status": "UNKNOWN", "not_found": True, "tasks": {}},
    ) == "RETRY_CREATED"
    assert len(db.list_stage_requests(
        campaign="campaign", status="PENDING", db_path=db_path,
    )) == 1


def test_expired_revalidation_submission_uses_same_retry_path(
    tmp_path, monkeypatch,
) -> None:
    db_path, original_request, execution_id = _running_campaign_request(tmp_path)
    db.update_workflow_execution(
        execution_id, status="UNKNOWN", details="submit_ambiguous: timeout",
        db_path=db_path,
    )
    request = db.get_stage_request(original_request["id"], db_path=db_path)
    execution = db.get_workflow_execution(execution_id, db_path=db_path)
    campaign = db.get_campaign("campaign", db_path=db_path)
    monkeypatch.setattr(
        campaign_controller,
        "_ambiguous_submit_not_found_grace_expired",
        lambda _wf: True,
    )

    result = campaign_controller._handle_revalidation_observation(
        request, execution,
        {"status": "UNKNOWN", "tasks": {}, "not_found": True},
        campaign=campaign, dataset_cfg={}, db_path=db_path,
        verify_payload_hashes=False, retry_infrastructure=True,
        recover_publication_failures=False, defer_bulk_publication=False,
        publication_copy_workers=1,
    )

    assert result == {
        "request_id": original_request["id"], "status": "RETRY_CREATED",
    }
    failed_execution = db.get_workflow_execution(execution_id, db_path=db_path)
    assert failed_execution["status"] == "FAILED"
    assert "osmo_submission_not_found_after_grace" in failed_execution["details"]
    assert json.loads(failed_execution["last_query_payload_json"]) == {
        "not_found": True, "status": "UNKNOWN", "tasks": {},
    }
    replacements = db.list_stage_requests(
        campaign="campaign", status="PENDING", db_path=db_path,
    )
    assert len(replacements) == 1
    assert replacements[0]["stage"] == "revalidation"
