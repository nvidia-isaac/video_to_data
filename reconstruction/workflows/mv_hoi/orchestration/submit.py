# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submit OSMO workflows for MV calibration and HOI reconstruction.

Auto mode — scan Swift for new sequences and submit up to max_concurrent:
    python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction

Manual mode — submit a single named sequence:
    python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --sequence <name>
"""

from __future__ import annotations

try:
    from .storage import parse_storage_url, s3_client_kwargs
except ImportError:  # Direct script execution.
    from storage import parse_storage_url, s3_client_kwargs

import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import math
import os
import subprocess
import sys
import shlex
from datetime import datetime

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MV_HOI_DIR = os.path.dirname(SCRIPT_DIR)
FACE_DETECTOR_CAMERAS = (
    "back_stereo_camera_left",
    "back_stereo_camera_right",
    "front_stereo_camera_left",
    "front_stereo_camera_right",
    "left_stereo_camera_left",
    "left_stereo_camera_right",
    "right_stereo_camera_left",
    "right_stereo_camera_right",
)


def missing_face_detector_videos(s3_client, bucket: str, face_detector_prefix: str) -> list[str]:
    """Return cameras whose anonymized face-detector video is absent."""
    return [
        camera
        for camera in FACE_DETECTOR_CAMERAS
        if not object_exists(
            s3_client,
            bucket,
            f"{face_detector_prefix.rstrip('/')}/videos/{camera}.mp4",
        )
    ]


try:
    from .db import (
        DB_PATH,
        EXECUTION_STAGE_NAMES,
        TEST_DB_PATH,
        PIPELINES_TABLE,
        PIPELINES_TEST_TABLE,
        apply_execution_observation,
        ensure_version_cached,
        attach_request_execution,
        create_stage_request,
        get_active_stage_request,
        get_blacklisted_sequence,
        get_blacklisted_sequences,
        get_latest_workflow,
        get_latest_successful_stage_run,
        get_stage_request,
        init_db,
        insert_workflow,
        list_current_stage_runs,
        list_workflow_executions,
        maybe_blacklist_repeated_failure,
        reserve_stage_request,
        upsert_sequence,
        update_stage_request,
        update_workflow,
    )
    from .registry_versions import RegistryVersionError, image_registry, resolve_submission_version
    from .config_utils import (
        CALIBRATION_PIPELINE,
        CALIBRATION_WORKFLOW,
        EXPORT_CONFIG_PIPELINE,
        PREPROCESS_PIPELINE,
        PREPROCESS_WORKFLOW,
        RECON_PIPELINE,
        RECONSTRUCTION_WORKFLOW,
        apply_test_mode,
        get_pipeline_input_path,
        get_pipeline_max_concurrent,
        get_pipeline_output_path,
        get_workflow_cfg,
        load_config as _load_config,
    )
    from .pool_selection import (
        PoolSelector,
        load_active_counts,
        osmo_submission_priority,
    )
    from .query import DEFAULT_REFRESH_WORKERS, osmo_cancel, refresh_workflow_states
    from .runtime import require_submit_authority
except ImportError:  # Direct script execution.
    from db import (
        DB_PATH,
        EXECUTION_STAGE_NAMES,
        TEST_DB_PATH,
        PIPELINES_TABLE,
        PIPELINES_TEST_TABLE,
        apply_execution_observation,
        ensure_version_cached,
        attach_request_execution,
        create_stage_request,
        get_active_stage_request,
        get_blacklisted_sequence,
        get_blacklisted_sequences,
        get_latest_workflow,
        get_latest_successful_stage_run,
        get_stage_request,
        init_db,
        insert_workflow,
        list_current_stage_runs,
        list_workflow_executions,
        maybe_blacklist_repeated_failure,
        reserve_stage_request,
        upsert_sequence,
        update_stage_request,
        update_workflow,
    )
    from registry_versions import RegistryVersionError, image_registry, resolve_submission_version
    from config_utils import (
        CALIBRATION_PIPELINE,
        CALIBRATION_WORKFLOW,
        EXPORT_CONFIG_PIPELINE,
        PREPROCESS_PIPELINE,
        PREPROCESS_WORKFLOW,
        RECON_PIPELINE,
        RECONSTRUCTION_WORKFLOW,
        apply_test_mode,
        get_pipeline_input_path,
        get_pipeline_max_concurrent,
        get_pipeline_output_path,
        get_workflow_cfg,
        load_config as _load_config,
    )
    from pool_selection import PoolSelector, load_active_counts, osmo_submission_priority
    from query import DEFAULT_REFRESH_WORKERS, osmo_cancel, refresh_workflow_states
    from runtime import require_submit_authority

TABLE = PIPELINES_TABLE
OBJECT_BBOX_SOURCE_MARKER = "object_bbox_source.txt"
OBJECT_BBOX_SOURCE_MANUAL = "manual_labeled_bboxes"


@dataclass(frozen=True)
class SubmitResult:
    workflow_name: str
    ambiguous: bool = False
    prereq_skipped: bool = False


class AmbiguousSubmitError(RuntimeError):
    """Raised when OSMO may have accepted the submit but did not return an ID."""

    def __init__(self, error: subprocess.CalledProcessError):
        self.error = error
        super().__init__(_short_submit_error(error))


_AMBIGUOUS_SUBMIT_MARKERS = (
    "read timed out",
    "cannot connect to osmo service",
    "httpsconnectionpool",
    "connectionerror",
    "connection aborted",
    "max retries exceeded",
    "timed out",
)


def _apply_test_mode(dataset_cfg: dict) -> None:
    apply_test_mode(dataset_cfg)


def load_config() -> dict:
    return _load_config(MV_HOI_DIR)


def campaign_processing_output_path(
    dataset_cfg: dict, pipeline: str, scheduled_request: dict | None,
) -> str:
    """Resolve the output root for mainline or campaign processing."""
    configured = get_pipeline_output_path(dataset_cfg, pipeline)
    campaign_rerun = bool(
        scheduled_request
        and scheduled_request.get("campaign_type")
        in ("BACKLOG_REPROCESSING", "REMEDIATION")
    )
    if not campaign_rerun:
        return configured
    allowed = dataset_cfg["pipelines"][pipeline].get("campaign_output_path")
    if not allowed:
        raise ValueError(
            f"{pipeline} must define campaign_output_path for campaign work"
        )
    if str(allowed).strip("/") != str(configured).strip("/"):
        raise ValueError(
            f"{pipeline} campaign_output_path must match output_path for "
            "canonical processing"
        )
    return str(allowed)


def processing_sequence_output_url(
    swift_base: str,
    output_path: str,
    sequence_name: str,
    pipeline_type: str,
    scheduled_request: dict | None,
    *,
    dry_run_id: str | None = None,
) -> str:
    """Return the canonical preprocess or isolated reconstruction prefix."""
    base = f"{swift_base.rstrip('/')}/{output_path.strip('/')}/{sequence_name}"
    if pipeline_type == PREPROCESS_PIPELINE:
        return base
    if pipeline_type != RECON_PIPELINE:
        raise ValueError(f"Unsupported processing pipeline: {pipeline_type}")
    unique_id = (
        str(int(scheduled_request["id"]))
        if scheduled_request is not None
        else dry_run_id
    )
    if not unique_id:
        raise ValueError("Reconstruction output requires a durable request ID")
    return f"{base}/reconstruction_{unique_id}"


def _matches_adopted_preprocess_lineage(
    scheduled_request: dict,
    expected_request_id: int,
    preprocess_run: dict,
) -> bool:
    """Accept a prior-patch run only through its immutable adoption chain.

    Campaign replacements can adopt an already-adopted preprocessing result.
    Follow those virtual request records to the request that owns the physical
    run, while validating every hop.  The bounded walk rejects cycles and
    malformed or partially rewritten provenance.
    """
    target = get_stage_request(int(expected_request_id), db_path=DB_PATH)
    if not target:
        return False
    try:
        observed_run_id = int(
            preprocess_run.get("stage_run_id", preprocess_run.get("id"))
        )
        observed_request_id = int(preprocess_run.get("request_id"))
    except (TypeError, ValueError):
        return False

    if not bool(
        target.get("id") == int(expected_request_id)
        and target.get("status") == "SUCCEEDED"
        and target.get("stage") == "preprocess"
        and target.get("workflow_execution_id") is None
        and int(target.get("queue_priority") or 0) == 100
        and target.get("campaign_id") == scheduled_request.get("campaign_id")
        and target.get("sequence_name") == scheduled_request.get("sequence_name")
        and target.get("pipeline_version") == scheduled_request.get("pipeline_version")
        # Joined stage-run rows expose the canonical run state as ``run_status``;
        # plain stage-run rows use ``status`` directly.
        and (preprocess_run.get("run_status") or preprocess_run.get("status"))
        == "SUCCEEDED"
        and bool(preprocess_run.get("is_current"))
    ):
        return False

    current = target
    visited: set[int] = set()
    for _ in range(16):
        try:
            current_id = int(current["id"])
            if current_id in visited:
                return False
            visited.add(current_id)
            summary = json.loads(current.get("result_summary_json") or "{}")
            evidence = summary["preprocess_adoption"]
            source_request_id = int(evidence["source_request_id"])
            run_id = int(evidence["preprocess_run_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

        source = get_stage_request(source_request_id, db_path=DB_PATH)
        if not bool(
            evidence.get("schema") == "v2d.mv_hoi.preprocess_patch_adoption.v1"
            and evidence.get("adopted_by") == "accuracy-segment-rollout"
            and evidence.get("reason")
            == "implementation_patch_does_not_affect_preprocessing"
            and current.get("status") == "SUCCEEDED"
            and current.get("stage") == "preprocess"
            and current.get("source_manifest_sha256")
            == evidence.get("source_manifest_sha256")
            and evidence.get("target_pipeline_version")
            == current.get("pipeline_version")
            and evidence.get("preprocess_pipeline_version")
            == preprocess_run.get("pipeline_version")
            and observed_run_id == run_id
            and str(preprocess_run.get("output_uri") or "").rstrip("/")
            == str(evidence.get("preprocess_output_uri") or "").rstrip("/")
            and source
            and source.get("id") == source_request_id
            and source.get("status") == "SUCCEEDED"
            and source.get("stage") == "preprocess"
            and source.get("sequence_name")
            == scheduled_request.get("sequence_name")
            and source.get("source_manifest_sha256")
            == current.get("source_manifest_sha256")
            and source.get("campaign_id") != current.get("campaign_id")
            and evidence.get("source_campaign_id") == source.get("campaign_id")
        ):
            return False

        if source_request_id == observed_request_id:
            return True
        current = source
    return False


# Swift / S3 helpers

def _parse_swift_url(url: str) -> tuple[str | None, str, str]:
    """Compatibility alias accepting S3, Swift and bare bucket paths."""
    return parse_storage_url(url)


def get_s3_client(swift_url: str):
    import boto3

    endpoint, bucket, prefix = _parse_swift_url(swift_url)
    client = boto3.client("s3", **s3_client_kwargs(endpoint))
    return client, bucket, prefix


def list_sequences(client, bucket: str, prefix: str) -> list[str]:
    """List immediate subdirectory names under *prefix*."""
    if not prefix.endswith("/"):
        prefix += "/"
    paginator = client.get_paginator("list_objects_v2")
    sequences: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            name = cp["Prefix"].rstrip("/").rsplit("/", 1)[-1]
            sequences.add(name)
    return sorted(sequences)


def path_exists(client, bucket: str, prefix: str) -> bool:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return resp.get("KeyCount", 0) > 0


def object_exists(client, bucket: str, key: str) -> bool:
    resp = client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return any(obj["Key"] == key for obj in resp.get("Contents", []))


def prefix_has_json_files(client, bucket: str, prefix: str) -> bool:
    if not prefix.endswith("/"):
        prefix += "/"
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
    return any(obj["Key"].endswith(".json") for obj in resp.get("Contents", []))


def labeled_bbox_manifest(client, bucket: str, prefix: str) -> tuple[str, str]:
    """Return a deterministic JSON manifest and logical content SHA-256."""
    normalized_prefix = prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    entries: list[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
            entries.append({
                "name": key[len(normalized_prefix):],
                "size": len(body),
                "etag": str(obj.get("ETag", "")).strip('"') or None,
                "sha256": hashlib.sha256(body).hexdigest(),
            })
    entries.sort(key=lambda item: item["name"])
    if not entries:
        raise ValueError(f"No labeled bbox JSON files found at s3://{bucket}/{prefix}")
    manifest_json = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return manifest_json, hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()


def get_s3_text(client, bucket: str, key: str) -> str | None:
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    data = resp["Body"].read()
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return data


def edex_has_camera_transforms(edex_text: str | None) -> bool:
    if not edex_text:
        return False
    try:
        edex = json.loads(edex_text)
    except (TypeError, json.JSONDecodeError):
        return False
    header = edex[0] if isinstance(edex, list) and edex else edex
    if not isinstance(header, dict):
        return False
    cameras = header.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        return False
    for camera in cameras:
        if not isinstance(camera, dict):
            return False
        transform = camera.get("transform")
        if not isinstance(transform, list) or len(transform) != 3:
            return False
        for row in transform:
            if not isinstance(row, list) or len(row) != 4:
                return False
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return False
                if not math.isfinite(float(value)):
                    return False
    return True


def get_hoi_metadata(client, bucket: str, seq_prefix: str) -> dict | None:
    key = f"{seq_prefix}/hoi_metadata.yaml".lstrip("/")
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    return yaml.safe_load(resp["Body"].read())


def resolve_mesh_url(
    client, bucket: str, mesh_prefix: str, object_id: str, mesh_base: str,
) -> str | None:
    """Prefer einstar/, fall back to bundlesdf/. Return swift:// URL or None."""
    base = f"{mesh_prefix}/{object_id}".lstrip("/")
    for method in ("einstar", "bundlesdf"):
        method_pfx = f"{base}/{method}/"
        aligned_key = f"{method_pfx}output_aligned.glb"
        resp = client.list_objects_v2(Bucket=bucket, Prefix=aligned_key, MaxKeys=1)
        if any(obj["Key"] == aligned_key for obj in resp.get("Contents", [])):
            return f"{mesh_base.rstrip('/')}/{object_id}/{method}/"
    return None


# OSMO helpers

def _submit_error_text(error: subprocess.CalledProcessError) -> str:
    return "\n".join(
        part for part in (error.output, error.stderr) if part
    )


def _is_ambiguous_submit_error(error: subprocess.CalledProcessError) -> bool:
    """Return True if OSMO may have accepted the submit before the CLI failed."""
    if error.returncode == 10:
        return True
    text = _submit_error_text(error).lower()
    return any(marker in text for marker in _AMBIGUOUS_SUBMIT_MARKERS)


def _short_submit_error(error: subprocess.CalledProcessError) -> str:
    for line in _submit_error_text(error).splitlines():
        line = line.strip()
        if not line or line == "Error message:" or line.startswith("Error code:"):
            continue
        return f"exit {error.returncode}: {line[:160]}"
    return f"exit {error.returncode}"


def osmo_submit(
    workflow_yaml: str, pool: str, set_vars: dict[str, str], *, dry_run: bool = False,
) -> str:
    """Submit workflow and return the OSMO-assigned Workflow ID."""
    yaml_path = os.path.join(MV_HOI_DIR, workflow_yaml)
    cmd = ["osmo", "workflow", "submit", yaml_path]
    if set_vars:
        cmd.append("--set")
        cmd.extend(f"{key}={value}" for key, value in set_vars.items())
    cmd.extend(["--pool", pool])
    priority = osmo_submission_priority(pool)
    if priority:
        cmd.extend(["--priority", priority])
    print(f"  CMD: {shlex.join(cmd)}")
    if dry_run:
        print("  [dry-run] skipping osmo submit")
        return set_vars.get("workflow_name", "dry-run")
    require_submit_authority("submit an OSMO workflow")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error = subprocess.CalledProcessError(
            result.returncode, cmd,
            output=result.stdout, stderr=result.stderr,
        )
        if _is_ambiguous_submit_error(error):
            raise AmbiguousSubmitError(error)
        raise error
    stdout = result.stdout.strip()
    print(f"  OSMO: {stdout}")
    for line in stdout.splitlines():
        if line.strip().startswith("Workflow ID"):
            return line.split("-", 1)[1].strip()
    return set_vars.get("workflow_name", stdout)


# Core logic

def _generate_workflow_name(pipeline_type: str, version: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ver = version.replace(".", "-")
    return f"v2d_{pipeline_type}_{ver}_{ts}"


def _record_prereq_skip(
    sequence_name: str,
    dataset_name: str,
    pipeline_type: str,
    version: str,
    workflow_name: str,
    reason: str,
    latest: dict | None,
    *,
    dry_run: bool = False,
    log: bool = True,
) -> None:
    # Runtime prerequisites are represented by the derived sequence_status
    # view (WAITING_CALIBRATION, WAITING_PREPROCESS, WAITING_LABELS, and so
    # on). They are not execution attempts and therefore do not get stage-run
    # or workflow-execution rows. SKIPPED rows are retained only for migration
    # evidence from the legacy database.
    del version, workflow_name, latest, log
    if not dry_run:
        request = get_active_stage_request(
            dataset_name, sequence_name, EXECUTION_STAGE_NAMES[pipeline_type],
            db_path=DB_PATH,
        )
        if request:
            update_stage_request(
                request["id"], status="BLOCKED", blocked_reason=reason,
                details=reason, db_path=DB_PATH,
            )


def _handle_prereq_skip(
    sequence_name: str,
    dataset_name: str,
    pipeline_type: str,
    version: str,
    workflow_name: str,
    reason: str,
    latest: dict | None,
    *,
    dry_run: bool = False,
    log: bool = True,
) -> SubmitResult:
    if log:
        print(f"  {sequence_name}: {reason}, skipping")
    _record_prereq_skip(
        sequence_name, dataset_name, pipeline_type, version,
        workflow_name, reason, latest, dry_run=dry_run, log=log,
    )
    return SubmitResult(workflow_name, prereq_skipped=True)


def _blacklist_skip_message(sequence_name: str, dataset_name: str, reason: str | None) -> str:
    suffix = f": {reason}" if reason else ""
    return (
        f"  {sequence_name}: skipping (blacklisted for {dataset_name}{suffix}). "
        "Use --force to submit anyway."
    )


def submit_sequence(
    sequence_name: str,
    dataset_name: str,
    dataset_cfg: dict,
    pipeline_type: str,
    *,
    force: bool = False,
    dry_run: bool = False,
    log_prereq_skips: bool = True,
    pipeline_version: str | None = None,
    trigger: str = "manual",
    request_id: int | None = None,
    pool: str | None = None,
    pool_selector: PoolSelector | None = None,
) -> SubmitResult | None:
    """Build --set vars, submit OSMO workflow, record in DB. Return workflow name."""
    scheduled_request = (
        get_stage_request(request_id, db_path=DB_PATH)
        if request_id is not None else None
    )
    campaign_rerun = bool(
        scheduled_request
        and scheduled_request.get("campaign_type")
        in ("BACKLOG_REPROCESSING", "REMEDIATION")
    )
    blacklist_entry = get_blacklisted_sequence(
        dataset_name, sequence_name, db_path=DB_PATH,
    )
    if blacklist_entry and not force and not campaign_rerun:
        print(
            _blacklist_skip_message(
                sequence_name, dataset_name, blacklist_entry.get("reason"),
            )
        )
        return None

    latest = get_latest_workflow(
        sequence_name, dataset_name, pipeline_type, db_path=DB_PATH, table=TABLE,
    )

    if latest and latest["status"] in (
        "WAITING_WF", "UNKNOWN", "WAITING_QC", "WAITING_EXPORT", "PASS",
    ) and not force and not campaign_rerun:
        print(f"  {sequence_name}: skipping (status={latest['status']}). Use --force to resubmit.")
        return None

    if latest and latest["status"] in ("WAITING_WF", "UNKNOWN") and force:
        osmo_id = latest.get("osmo_workflow_id") or latest["workflow_name"]
        if not osmo_cancel(osmo_id):
            print(f"  {sequence_name}: cancel failed, aborting")
            return None
        if latest.get("execution_id"):
            apply_execution_observation(
                latest["execution_id"], execution_status="CANCELED",
                details="cancelled_for_resubmit", run_outcomes=[{
                    "run_id": latest["stage_run_id"], "stage": pipeline_type,
                    "status": "CANCELED",
                }], db_path=DB_PATH,
            )
        else:
            update_workflow(
                latest["workflow_name"], status="FAIL",
                details="cancelled_for_resubmit", db_path=DB_PATH, table=TABLE,
            )
        print(f"  {sequence_name}: cancelled previous run, resubmitting")

    version = pipeline_version or resolve_submission_version(
        registry=dataset_cfg.get("image_registry"),
    )
    if not dry_run:
        ensure_version_cached(version, db_path=DB_PATH)
        request_stage = EXECUTION_STAGE_NAMES[pipeline_type]
        if request_id is None:
            active_request = get_active_stage_request(
                dataset_name, sequence_name, request_stage, db_path=DB_PATH,
            )
            if active_request and active_request["status"] in (
                "SUBMITTED", "RUNNING", "UNKNOWN",
            ):
                print(
                    f"  {sequence_name}: active {request_stage} request "
                    f"{active_request['id']} is {active_request['status']}"
                )
                return None
            if active_request:
                request_id = active_request["id"]
                if active_request["status"] == "BLOCKED":
                    update_stage_request(
                        request_id, status="PENDING", blocked_reason=None,
                        details="dependency_recheck", db_path=DB_PATH,
                    )
            else:
                request = create_stage_request(
                    sequence_name=sequence_name,
                    dataset=dataset_name,
                    stage=request_stage,
                    pipeline_version=version,
                    trigger=trigger,
                    requested_by=os.environ.get("USER", "automation"),
                    reason="forced_manual_run" if force else None,
                    parameters={"force_blacklist_bypass": bool(force)},
                    db_path=DB_PATH,
                )
                request_id = request["id"]
        scheduled_request = get_stage_request(request_id, db_path=DB_PATH)
    swift_base = dataset_cfg["swift_base"]
    s3, bucket, base_pfx = get_s3_client(swift_base)
    workflow_name = _generate_workflow_name(pipeline_type, version)

    if pipeline_type == CALIBRATION_PIPELINE:
        workflow_key = CALIBRATION_WORKFLOW
    elif pipeline_type == PREPROCESS_PIPELINE:
        workflow_key = PREPROCESS_WORKFLOW
    else:
        workflow_key = RECONSTRUCTION_WORKFLOW
    workflow_cfg = get_workflow_cfg(dataset_cfg, pipeline_type, workflow_key)
    workflow_yaml = workflow_cfg["workflow_yaml"]

    set_vars: dict[str, str] = {
        "workflow_name": workflow_name,
        "image_tag": version,
        "image_registry": image_registry(dataset_cfg.get("image_registry")),
        "continuous_symmetry_step_deg": str(
            workflow_cfg.get("continuous_symmetry_step_deg", 10.0)
        ),
    }
    if pipeline_type in (PREPROCESS_PIPELINE, RECON_PIPELINE):
        weights_base_url = str(dataset_cfg.get("weights_base_url", "")).rstrip("/")
        if not weights_base_url:
            raise ValueError(
                f"Dataset {dataset_name!r} must define weights_base_url for "
                f"{pipeline_type}"
            )
        set_vars["weights_base_url"] = weights_base_url
    stage_fields: dict = {"requested_by": os.environ.get("USER", "automation")}

    if pipeline_type == PREPROCESS_PIPELINE:
        latest_recon = get_latest_workflow(
            sequence_name, dataset_name, RECON_PIPELINE, db_path=DB_PATH, table=TABLE,
        )
        if (
            latest_recon and latest_recon["status"] == "PASS"
            and not force and not campaign_rerun
        ):
            if log_prereq_skips:
                print(
                    f"  {sequence_name}: reconstruction already PASS; "
                    "leaving preprocess stage history unchanged"
                )
            if request_id is not None:
                update_stage_request(
                    request_id, status="CANCELED",
                    details="reconstruction_already_succeeded", db_path=DB_PATH,
                )
            return None

        input_path = get_pipeline_input_path(dataset_cfg, pipeline_type)
        output_path = campaign_processing_output_path(
            dataset_cfg, pipeline_type, scheduled_request,
        )
        set_vars["rosbag_url"] = (
            f"{swift_base}/{input_path}/{sequence_name}/"
        )

        # Metadata
        seq_data_pfx = f"{base_pfx}/{input_path}/{sequence_name}"
        meta = get_hoi_metadata(s3, bucket, seq_data_pfx)
        if meta is None:
            reason = "no hoi_metadata.yaml"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )

        # Calibration parameters via calib_seq_name
        calib_seq = meta.get("calib_seq_name")
        if not calib_seq:
            reason = "no calib_seq_name in hoi_metadata"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )
        object_id = (
            meta.get("object", {}).get("id")
            or meta.get("object_id")
            or meta.get("object_name")
        )
        metadata_uri = f"{swift_base}/{input_path}/{sequence_name}/hoi_metadata.yaml"
        if not dry_run:
            upsert_sequence(
                dataset_name,
                sequence_name,
                source_uri=set_vars["rosbag_url"],
                hoi_metadata_uri=metadata_uri,
                calibration_sequence_name=calib_seq,
                object_id=str(object_id) if object_id is not None else None,
                db_path=DB_PATH,
            )
        calib_pfx = (
            f"{base_pfx}/{get_pipeline_output_path(dataset_cfg, CALIBRATION_PIPELINE)}"
            f"/{calib_seq}/calibrate_extrinsics"
        )
        if not path_exists(s3, bucket, calib_pfx):
            reason = f"calibration not found for {calib_seq}"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )
        set_vars["calibration_url"] = (
            f"{swift_base}/{get_pipeline_output_path(dataset_cfg, CALIBRATION_PIPELINE)}"
            f"/{calib_seq}/calibrate_extrinsics"
        )
        calibration_run = get_latest_successful_stage_run(
            calib_seq, dataset_name, CALIBRATION_PIPELINE,
            db_path=DB_PATH, table=TABLE,
        )
        if not calibration_run:
            reason = f"no current successful calibration run for {calib_seq}"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )

        # Object ID + mesh
        if not object_id:
            reason = "no object_id in hoi_metadata"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )

        _, mesh_bucket, mesh_pfx = _parse_swift_url(
            dataset_cfg["mesh_base"]
        )
        mesh_url = resolve_mesh_url(
            s3, mesh_bucket, mesh_pfx, object_id, dataset_cfg["mesh_base"],
        )
        if not mesh_url:
            reason = f"no mesh for object {object_id}"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )
        set_vars["mesh_url"] = mesh_url

        # Output base
        set_vars["swift_output_base"] = processing_sequence_output_url(
            swift_base, output_path, sequence_name,
            PREPROCESS_PIPELINE,
            scheduled_request,
            dry_run_id=workflow_name if dry_run else None,
        )
        object_id = str(object_id)
        stage_fields.update({
            "calibration_run_id": calibration_run["stage_run_id"],
            "source_uri": set_vars["rosbag_url"],
            "mesh_uri": mesh_url,
            "metadata_uri": metadata_uri,
            "output_uri": set_vars["swift_output_base"],
        })

    elif pipeline_type == RECON_PIPELINE:
        output_path = campaign_processing_output_path(
            dataset_cfg, pipeline_type, scheduled_request,
        )
        preprocess_run = get_latest_successful_stage_run(
            sequence_name, dataset_name, PREPROCESS_PIPELINE,
            db_path=DB_PATH, table=TABLE,
        )
        if not preprocess_run:
            reason = "no current successful preprocessing run"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )
        preprocess_output_uri = str(preprocess_run.get("output_uri") or "").rstrip("/")
        if not preprocess_output_uri:
            if campaign_rerun:
                reason = "current preprocessing run has no output URI"
                return _handle_prereq_skip(
                    sequence_name, dataset_name, pipeline_type, version, workflow_name,
                    reason, latest, dry_run=dry_run, log=log_prereq_skips,
                )
            preprocess_output_uri = (
                f"{swift_base}/"
                f"{campaign_processing_output_path(dataset_cfg, PREPROCESS_PIPELINE, None)}"
                f"/{sequence_name}"
            )
        if campaign_rerun:
            parameters = json.loads(scheduled_request.get("parameters_json") or "{}")
            expected_request_id = parameters.get("preprocess_request_id")
            direct_lineage = (
                expected_request_id is not None
                and int(expected_request_id)
                == int(preprocess_run.get("request_id") or -1)
            )
            adopted_lineage = (
                expected_request_id is not None
                and _matches_adopted_preprocess_lineage(
                    scheduled_request, int(expected_request_id), preprocess_run,
                )
            )
            if not (direct_lineage or adopted_lineage):
                reason = "current preprocessing run does not match campaign lineage"
                return _handle_prereq_skip(
                    sequence_name, dataset_name, pipeline_type, version, workflow_name,
                    reason, latest, dry_run=dry_run, log=log_prereq_skips,
                )
            expected_preprocess_output = processing_sequence_output_url(
                swift_base,
                campaign_processing_output_path(
                    dataset_cfg, PREPROCESS_PIPELINE, {
                        **scheduled_request,
                        "id": int(expected_request_id),
                    },
                ),
                sequence_name,
                PREPROCESS_PIPELINE,
                {
                    **scheduled_request,
                    "id": int(expected_request_id),
                },
            )
            if preprocess_output_uri != expected_preprocess_output.rstrip("/"):
                reason = (
                    "current preprocessing run is outside the canonical "
                    "campaign prefix"
                )
                return _handle_prereq_skip(
                    sequence_name, dataset_name, pipeline_type, version, workflow_name,
                    reason, latest, dry_run=dry_run, log=log_prereq_skips,
                )
        _, preprocess_bucket, preprocess_pfx = _parse_swift_url(
            f"{preprocess_output_uri}/mv_preprocess"
        )
        if preprocess_bucket != bucket:
            reason = "current preprocessing output uses an unexpected bucket"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )

        required_paths = [
            ("hoi_metadata.yaml", f"{preprocess_pfx}/hoi_metadata.yaml"),
            ("edex", f"{preprocess_pfx}/edex"),
            ("images", f"{preprocess_pfx}/images/"),
            ("videos", f"{preprocess_pfx}/videos/"),
        ]
        for label, prefix in required_paths:
            if not path_exists(s3, bucket, prefix):
                reason = f"mv_preprocess missing {label}"
                return _handle_prereq_skip(
                    sequence_name, dataset_name, pipeline_type, version, workflow_name,
                    reason, latest, dry_run=dry_run, log=log_prereq_skips,
                )

        if campaign_rerun:
            _, face_bucket, face_pfx = _parse_swift_url(
                f"{preprocess_output_uri}/face_detector"
            )
            missing_face_videos = (
                list(FACE_DETECTOR_CAMERAS)
                if face_bucket != bucket
                else missing_face_detector_videos(s3, bucket, face_pfx)
            )
            if missing_face_videos:
                reason = (
                    "face_detector missing anonymized videos: "
                    + ", ".join(missing_face_videos)
                )
                return _handle_prereq_skip(
                    sequence_name, dataset_name, pipeline_type, version,
                    workflow_name, reason, latest, dry_run=dry_run,
                    log=log_prereq_skips,
                )

        edex_text = get_s3_text(s3, bucket, f"{preprocess_pfx}/edex")
        if not edex_has_camera_transforms(edex_text):
            reason = "mv_preprocess edex missing calibrated camera transforms"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )

        mesh_key = f"{preprocess_pfx}/object_mesh/output_aligned.glb"
        if not object_exists(s3, bucket, mesh_key):
            reason = "mv_preprocess missing object_mesh/output_aligned.glb"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )

        bbox_prefix = f"{preprocess_pfx}/labeled_bboxes"
        if not prefix_has_json_files(s3, bucket, bbox_prefix):
            reason = "mv_preprocess missing labeled_bboxes/*.json"
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )

        marker = get_s3_text(
            s3, bucket, f"{preprocess_pfx}/{OBJECT_BBOX_SOURCE_MARKER}",
        )
        if marker is not None:
            marker = marker.strip()
            if marker != OBJECT_BBOX_SOURCE_MANUAL:
                reason = f"unknown object bbox source marker: {marker}"
                return _handle_prereq_skip(
                    sequence_name, dataset_name, pipeline_type, version, workflow_name,
                    reason, latest, dry_run=dry_run, log=log_prereq_skips,
                )
        elif campaign_rerun:
            reason = (
                "backlog reconstruction requires "
                "object_bbox_source.txt=manual_labeled_bboxes"
            )
            return _handle_prereq_skip(
                sequence_name, dataset_name, pipeline_type, version, workflow_name,
                reason, latest, dry_run=dry_run, log=log_prereq_skips,
            )
        else:
            prompt = get_s3_text(s3, bucket, f"{preprocess_pfx}/prompt.txt")
            if prompt is None or not prompt.strip():
                reason = "mv_preprocess missing manual marker or nonempty prompt.txt"
                return _handle_prereq_skip(
                    sequence_name, dataset_name, pipeline_type, version, workflow_name,
                    reason, latest, dry_run=dry_run, log=log_prereq_skips,
                )

        set_vars["mv_preprocess_url"] = f"{preprocess_output_uri}/mv_preprocess/"
        set_vars["face_detector_url"] = f"{preprocess_output_uri}/face_detector/"
        bbox_manifest_json, bbox_sha256 = labeled_bbox_manifest(
            s3, bucket, bbox_prefix,
        )
        if campaign_rerun:
            required_bbox_names = {
                "back_stereo_camera_left.json",
                "front_stereo_camera_left.json",
                "left_stereo_camera_left.json",
                "right_stereo_camera_left.json",
            }
            observed_bbox_names = {
                item["name"] for item in json.loads(bbox_manifest_json)
            }
            if not required_bbox_names.issubset(observed_bbox_names):
                reason = "backlog reconstruction requires four manual bbox views"
                return _handle_prereq_skip(
                    sequence_name, dataset_name, pipeline_type, version,
                    workflow_name, reason, latest, dry_run=dry_run,
                    log=log_prereq_skips,
                )
        if request_id is not None:
            update_stage_request(
                request_id,
                source_manifest=json.loads(bbox_manifest_json),
                source_manifest_sha256=bbox_sha256,
                db_path=DB_PATH,
            )
        set_vars["expected_labeled_bboxes_sha256"] = bbox_sha256
        set_vars["expected_labeled_bboxes_manifest_b64"] = base64.b64encode(
            bbox_manifest_json.encode("utf-8")
        ).decode("ascii")
        set_vars["swift_output_base"] = processing_sequence_output_url(
            swift_base, output_path, sequence_name,
            RECON_PIPELINE,
            scheduled_request,
            dry_run_id=workflow_name if dry_run else None,
        )
        stage_fields.update({
            "preprocess_run_id": preprocess_run["stage_run_id"],
            "labeled_bboxes_uri": f"{set_vars['mv_preprocess_url']}labeled_bboxes/",
            "labeled_bboxes_manifest_json": bbox_manifest_json,
            "labeled_bboxes_sha256": bbox_sha256,
            "bbox_source": marker or "grounding_dino",
            "hitl_item_id": f"{workflow_name}.json",
            "output_uri": set_vars["swift_output_base"],
        })

        # QC thresholds
        thresholds = workflow_cfg.get("qc_thresholds", {})
        set_vars["max_chamfer_object"] = str(
            thresholds.get("max_chamfer_object", 40.0)
        )
        set_vars["max_chamfer_human"] = str(
            thresholds.get("max_chamfer_human", 40.0)
        )
        set_vars["max_chamfer_segment_object"] = str(
            thresholds.get("max_chamfer_segment_object", 50.0)
        )
        set_vars["max_chamfer_segment_human"] = str(
            thresholds.get("max_chamfer_segment_human", 50.0)
        )
        set_vars["min_mask_containment"] = str(thresholds.get("min_mask_containment", 0.8))
        set_vars["mask_bbox_padding"] = str(thresholds.get("mask_bbox_padding", 0.1))
        set_vars["min_object_silhouette_bbox_containment"] = str(
            thresholds.get("min_object_silhouette_bbox_containment", 0.8)
        )
        set_vars["min_object_segment_pixels"] = str(
            thresholds.get("min_object_segment_pixels", 10)
        )
        set_vars["min_object_bbox_component_pixels"] = str(
            thresholds.get("min_object_bbox_component_pixels", 3)
        )
        set_vars["min_object_bbox_component_fraction_of_largest"] = str(
            thresholds.get(
                "min_object_bbox_component_fraction_of_largest", 0.001
            )
        )
        set_vars["object_silhouette_render_bbox_padding_pixels"] = str(
            thresholds.get(
                "silhouette_render_bbox_padding_pixels",
                thresholds.get("object_silhouette_render_bbox_padding_pixels", 8),
            )
        )
        set_vars["max_object_silhouette_bad_frame_fraction"] = str(
            thresholds.get("max_object_silhouette_bad_frame_fraction", 0.05)
        )
        for key, default in (
            ("min_silhouette_bbox_containment", 0.8),
            ("min_silhouette_mask_pixels", 10),
            ("min_silhouette_bbox_component_pixels", 3),
            ("min_silhouette_bbox_component_fraction_of_largest", 0.001),
            ("silhouette_render_bbox_padding_pixels", 8),
            ("silhouette_render_bbox_padding_fraction", 0.1),
            ("min_accuracy_failure_run_frames", 5),
            ("max_silhouette_failure_coverage", 0.5),
        ):
            set_vars[key] = str(thresholds.get(key, default))

        # HITL upload metadata (object_id / action_desc are read inside the
        # task from the preprocess-module hoi_metadata.yaml)
        set_vars["hitl_s3_base"] = workflow_cfg["hitl_s3_base"]
        batch_template = workflow_cfg.get("hitl_batch_name_template", "batch_{date}")
        set_vars["hitl_batch_name"] = batch_template.format(
            date=datetime.now().strftime("%Y%m%d"),
        )
        set_vars["s3_region"] = workflow_cfg.get("s3_region", "us-west-2")

    elif pipeline_type == CALIBRATION_PIPELINE:
        input_path = get_pipeline_input_path(dataset_cfg, pipeline_type)
        output_path = get_pipeline_output_path(dataset_cfg, pipeline_type)
        if calibration_setup := workflow_cfg.get("calibration_setup"):
            set_vars["calibration_setup"] = calibration_setup
        set_vars["rosbag_url"] = (
            f"{swift_base}/{input_path}/{sequence_name}/"
        )
        set_vars["swift_output_base"] = (
            f"{swift_base}/{output_path}/{sequence_name}"
        )
        stage_fields.update({
            "calibration_setup": workflow_cfg.get("calibration_setup"),
            "source_uri": set_vars["rosbag_url"],
            "output_uri": set_vars["swift_output_base"],
        })

    selector = pool_selector or PoolSelector.collect(
        dataset_cfg,
        active_counts=load_active_counts(
            dataset_cfg, lambda: list_workflow_executions(
                status=("SUBMITTING", "RUNNING", "UNKNOWN"), db_path=DB_PATH,
            ),
        ),
    )
    pool_decision = selector.choose(
        pipeline_type,
        workflow_path=os.path.join(MV_HOI_DIR, workflow_yaml),
        override=pool,
    )
    selected_pool = pool_decision.pool

    print(f"  Submitting {pipeline_type} for {sequence_name} ({version})...")
    if dry_run:
        osmo_submit(workflow_yaml, selected_pool, set_vars, dry_run=True)
        print(f"  [dry-run] would reserve and insert workflow {workflow_name}")
        if force and blacklist_entry:
            print(
                f"  [dry-run] would bypass blacklist once for {sequence_name}; "
                "the entry would remain active"
            )
        return SubmitResult(workflow_name)

    if request_id is None:
        raise RuntimeError("Non-dry-run submission requires a durable stage request")
    request = reserve_stage_request(
        request_id,
        reserved_by=os.environ.get("HOSTNAME", os.environ.get("USER", "submit-host")),
        force_blacklist=force,
        db_path=DB_PATH,
    )
    if request is None:
        print(f"  {sequence_name}: request {request_id} is not reservable")
        return None

    # Reserve the per-sequence stage attempt before talking to OSMO. The
    # partial active-run index makes concurrent autosubmitters race safely:
    # only one can reserve this sequence/stage. It becomes current only after
    # OSMO accepted the request or returned an ambiguous response.
    try:
        reserved = insert_workflow(
            sequence_name=sequence_name,
            dataset=dataset_name,
            pipeline_type=pipeline_type,
            pipeline_version=version,
            workflow_name=workflow_name,
            workflow_spec_path=workflow_yaml,
            pool=selected_pool,
            status="SUBMITTING",
            details=pool_decision.detail("submit_reserved"),
            db_path=DB_PATH,
            table=TABLE,
            trigger=trigger,
            request_id=request_id,
            set_current=False,
            **stage_fields,
        )
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            print(f"  {sequence_name}: another active submission already reserved")
            return None
        raise

    attach_request_execution(
        request_id, reserved["execution_id"], status="SUBMITTED", db_path=DB_PATH,
    )

    try:
        osmo_workflow_id = osmo_submit(
            workflow_yaml, selected_pool, set_vars, dry_run=False,
        )
        apply_execution_observation(
            reserved["execution_id"], execution_status="RUNNING",
            osmo_workflow_id=osmo_workflow_id,
            details=pool_decision.detail("workflow_running"),
            run_outcomes=[{
                "run_id": reserved["stage_run_id"], "stage": pipeline_type,
                "status": "RUNNING", "set_current": True,
            }], db_path=DB_PATH,
        )
        if force and blacklist_entry:
            print(f"  {sequence_name}: blacklist bypassed once; entry remains active")
        print(f"  Workflow ID: {osmo_workflow_id}")
        return SubmitResult(workflow_name)
    except AmbiguousSubmitError as e:
        error = e.error
        if error.stdout and error.stdout.strip():
            print(f"  STDOUT: {error.stdout.strip()}")
        if error.stderr and error.stderr.strip():
            print(f"  STDERR: {error.stderr.strip()}")
        print(f"  ERROR (exit {error.returncode})")
        details = pool_decision.detail(
            f"submit_ambiguous: {_short_submit_error(error)}"
        )
        apply_execution_observation(
            reserved["execution_id"], execution_status="UNKNOWN",
            osmo_workflow_id=f"{workflow_name}-1", details=details,
            run_outcomes=[{
                "run_id": reserved["stage_run_id"], "stage": pipeline_type,
                "status": "UNKNOWN", "set_current": True,
            }], db_path=DB_PATH,
        )
        print(
            f"  Ambiguous OSMO submit recorded as {workflow_name}-1; "
            "assuming it may be queued."
        )
        return SubmitResult(workflow_name, ambiguous=True)
    except subprocess.CalledProcessError as e:
        if e.stdout and e.stdout.strip():
            print(f"  STDOUT: {e.stdout.strip()}")
        if e.stderr and e.stderr.strip():
            print(f"  STDERR: {e.stderr.strip()}")
        print(f"  ERROR (exit {e.returncode})")
        details = pool_decision.detail(
            f"submit_failed: {_short_submit_error(e)}"
        )
        apply_execution_observation(
            reserved["execution_id"], execution_status="FAILED", details=details,
            run_outcomes=[{
                "run_id": reserved["stage_run_id"], "stage": pipeline_type,
                "status": "FAILED",
            }], db_path=DB_PATH,
        )
        maybe_blacklist_repeated_failure(
            reserved["stage_run_id"], stage=pipeline_type, db_path=DB_PATH,
        )
        return None


def _normalize_time_arg(s: str) -> str:
    """Normalize a user time arg to the 19-char `YYYY-MM-DD_HH-MM-SS` form.

    `YYYY-MM-DD` is expanded to midnight (`_00-00-00`). Combined with an
    inclusive start and exclusive end, passing a bare date to `--end_time`
    excludes the entire day.
    """
    if len(s) == 10:
        return s + "_00-00-00"
    if len(s) == 19:
        return s
    raise ValueError(
        f"Time must be YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS: {s!r}"
    )


def _filter_sequences_by_time(
    sequences: list[str],
    start_time: str | None,
    end_time: str | None,
) -> list[str]:
    """Keep sequences whose `YYYY-MM-DD_HH-MM-SS` prefix is in [start, end).

    `start_time` is inclusive; `end_time` is exclusive.
    """
    if not start_time and not end_time:
        return sequences
    lo = _normalize_time_arg(start_time) if start_time else None
    hi = _normalize_time_arg(end_time) if end_time else None
    kept: list[str] = []
    for seq in sequences:
        prefix = seq[:19]
        if len(prefix) < 19 or prefix[4] != "-" or prefix[10] != "_":
            continue
        if lo and prefix < lo:
            continue
        if hi and prefix >= hi:
            continue
        kept.append(seq)
    return kept


def auto_submit(
    dataset_name: str, dataset_cfg: dict, pipeline_type: str,
    *, force: bool = False, dry_run: bool = False, retry_failed: bool = False,
    start_time: str | None = None, end_time: str | None = None,
    pipeline_version: str | None = None,
    pool: str | None = None,
) -> None:
    """Discover sequences from Swift and submit workflows up to concurrency limit."""
    version = pipeline_version or resolve_submission_version(
        registry=dataset_cfg.get("image_registry"),
    )
    if not dry_run:
        ensure_version_cached(version, db_path=DB_PATH)

    max_concurrent = get_pipeline_max_concurrent(dataset_cfg, pipeline_type)

    in_progress = list_current_stage_runs(
        dataset_name, stage=pipeline_type,
        status=["WAITING_WF", "UNKNOWN"], db_path=DB_PATH, table=TABLE,
    )
    available = max_concurrent - len(in_progress)
    print(f"In progress: {len(in_progress)}, available slots: {available}")
    if available <= 0:
        print(f"Max concurrent ({max_concurrent}) reached.")
        return

    swift_base = dataset_cfg["swift_base"]
    s3, bucket, base_pfx = get_s3_client(swift_base)

    scan_pfx = f"{base_pfx}/{get_pipeline_input_path(dataset_cfg, pipeline_type)}/"
    sequences = list_sequences(s3, bucket, scan_pfx)
    print(f"Found {len(sequences)} sequences in {dataset_name}")

    if not dry_run:
        sequence_kind = (
            "calibration" if pipeline_type == CALIBRATION_PIPELINE else "hoi"
        )
        input_path = get_pipeline_input_path(dataset_cfg, pipeline_type)
        for sequence in sequences:
            upsert_sequence(
                dataset_name,
                sequence,
                sequence_kind=sequence_kind,
                source_uri=f"{swift_base}/{input_path}/{sequence}/",
                db_path=DB_PATH,
            )

    if start_time or end_time:
        sequences = _filter_sequences_by_time(sequences, start_time, end_time)
        bounds = f"[{start_time or '-inf'}, {end_time or '+inf'}]"
        print(f"Filtered to {len(sequences)} sequences in time range {bounds}")

    skipped_blacklisted = 0
    if not force:
        blacklist = {
            row["sequence_name"]: row
            for row in get_blacklisted_sequences(dataset_name, db_path=DB_PATH)
        }
        if blacklist:
            before = len(sequences)
            sequences = [seq for seq in sequences if seq not in blacklist]
            skipped_blacklisted = before - len(sequences)
            if skipped_blacklisted:
                print(
                    f"Skipped {skipped_blacklisted} blacklisted sequence(s). "
                    "Use --force to submit anyway."
                )

    skip_statuses = {
        "PASS", "WAITING_WF", "UNKNOWN", "WAITING_QC", "WAITING_EXPORT",
    }
    if not retry_failed:
        skip_statuses.add("FAIL")

    skipped = 0
    skipped_prereq = 0
    submitted = 0
    pool_selector = PoolSelector.collect(
        dataset_cfg,
        active_counts=load_active_counts(
            dataset_cfg, lambda: list_workflow_executions(
                status=("SUBMITTING", "RUNNING", "UNKNOWN"), db_path=DB_PATH,
            ),
        ),
    )
    for seq in sequences:
        if submitted >= available:
            print(f"Reached max concurrent limit ({max_concurrent})")
            break
        latest = get_latest_workflow(
            seq, dataset_name, pipeline_type, db_path=DB_PATH, table=TABLE,
        )
        if latest and latest["status"] in skip_statuses and not force:
            skipped += 1
            continue
        wf = submit_sequence(
            seq, dataset_name, dataset_cfg, pipeline_type,
            force=force, dry_run=dry_run, log_prereq_skips=False,
            pipeline_version=version,
            trigger="automatic",
            pool=pool,
            pool_selector=pool_selector,
        )
        if wf:
            if isinstance(wf, SubmitResult) and wf.prereq_skipped:
                skipped_prereq += 1
                continue
            submitted += 1
            if isinstance(wf, SubmitResult) and wf.ambiguous:
                print(
                    "Ambiguous OSMO submit recorded; stopping auto-submit "
                    "to avoid duplicate/capacity runaway."
                )
                break

    if skipped:
        print(f"Skipped {skipped} sequence(s) with status in {sorted(skip_statuses)}. "
              "Use --force to resubmit.")
    if skipped_prereq:
        print(
            f"Skipped {skipped_prereq} sequence(s) due to unmet prerequisites "
            "(recorded as SKIPPED)."
        )

    print(f"\nSubmitted {submitted} new workflow(s)")


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit OSMO workflows")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--pipeline", required=True,
                        help="Pipeline type (e.g. mv_calibration, mv_hoi_reconstruction)")
    parser.add_argument("--sequence", help="Single sequence (manual mode)")
    parser.add_argument(
        "--version",
        help="Immutable pipeline image semver (default: latest complete registry release)",
    )
    parser.add_argument("--image-registry", help="Container registry/namespace (or V2D_IMAGE_REGISTRY)")
    parser.add_argument("--force", action="store_true",
                        help="Force resubmit even if blacklisted, WAITING_WF, WAITING_QC, "
                             "WAITING_EXPORT, or PASS; blacklist entries remain active")
    parser.add_argument("--retry_failed", action="store_true",
                        help="In auto mode, also retry sequences whose latest run failed")
    parser.add_argument("--start_time",
                        help="Auto mode: only include sequences with timestamp >= this "
                             "(YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS, inclusive)")
    parser.add_argument("--end_time",
                        help="Auto mode: only include sequences with timestamp < this "
                             "(YYYY-MM-DD or YYYY-MM-DD_HH-MM-SS, exclusive; "
                             "a bare date excludes that entire day)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build and print the osmo submit command without running it")
    parser.add_argument("--test", action="store_true",
                        help="Use pipelines_test table and append _test to output paths")
    parser.add_argument("--refresh-workers", type=int, default=DEFAULT_REFRESH_WORKERS,
                        help="Concurrent OSMO queries for WAITING_WF refresh "
                             f"(default: {DEFAULT_REFRESH_WORKERS}; env: "
                             "MV_HOI_REFRESH_WORKERS)")
    parser.add_argument(
        "--pool",
        help="Explicit configured OSMO pool override (otherwise select by capacity)",
    )
    args = parser.parse_args()

    global DB_PATH, TABLE
    if args.test:
        TABLE = PIPELINES_TEST_TABLE
        DB_PATH = TEST_DB_PATH

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(config['datasets'].keys())}")
        sys.exit(1)

    dataset_cfg = config["datasets"][args.dataset]
    submit_pipelines = {
        CALIBRATION_PIPELINE,
        PREPROCESS_PIPELINE,
        RECON_PIPELINE,
    }
    if args.pipeline == EXPORT_CONFIG_PIPELINE:
        print(
            f"{EXPORT_CONFIG_PIPELINE} is submitted through orchestration/export.py, "
            "not orchestration/submit.py"
        )
        sys.exit(1)
    if (
        args.pipeline not in submit_pipelines
        or args.pipeline not in dataset_cfg.get("pipelines", {})
    ):
        print(f"Unknown pipeline: {args.pipeline}")
        print(f"Available: {sorted(submit_pipelines)}")
        sys.exit(1)

    try:
        from .config_utils import validate_deployment_config
    except ImportError:
        from config_utils import validate_deployment_config
    if args.image_registry:
        dataset_cfg["image_registry"] = args.image_registry
    try:
        validate_deployment_config(dataset_cfg, args.pipeline)
    except ValueError as exc:
        parser.error(str(exc))

    if args.test:
        _apply_test_mode(dataset_cfg)

    init_db(DB_PATH)

    try:
        version = resolve_submission_version(
            args.version, registry=dataset_cfg.get("image_registry"),
        )
    except RegistryVersionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    if not args.dry_run:
        ensure_version_cached(version, db_path=DB_PATH)
    print(f"Pipeline image version: {version}")

    refresh_workflow_states(args.dataset, pipeline_type=args.pipeline, db_path=DB_PATH,
                            table=TABLE, max_workers=args.refresh_workers)

    if args.sequence:
        submit_sequence(
            args.sequence, args.dataset, dataset_cfg, args.pipeline,
            force=args.force, dry_run=args.dry_run,
            pipeline_version=version,
            pool=args.pool,
        )
    else:
        auto_submit(
            args.dataset, dataset_cfg, args.pipeline,
            force=args.force, dry_run=args.dry_run,
            retry_failed=args.retry_failed,
            start_time=args.start_time, end_time=args.end_time,
            pipeline_version=version,
            pool=args.pool,
        )


if __name__ == "__main__":
    main()
