"""Submit OSMO workflows for MV calibration and HOI reconstruction.

Auto mode — scan Swift for new sequences and submit up to max_concurrent:
    python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction

Manual mode — submit a single named sequence:
    python submit.py --dataset sc_office_4exo_1 --pipeline mv_hoi_reconstruction --sequence <name>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

import boto3
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from db import (
    get_latest_version,
    get_latest_workflow,
    get_workflows_by_dataset,
    init_db,
    insert_workflow,
    update_workflow,
)
from query import osmo_cancel, refresh_waiting

DB_PATH = os.path.join(SCRIPT_DIR, "processing.db")
TABLE = "workflows"


def _apply_test_mode(dataset_cfg: dict) -> None:
    """In-place: append `_test` to output paths used by the workflow."""
    dataset_cfg["calibration_output_path"] += "_test"
    dataset_cfg["data_output_path"] += "_test"
    dataset_cfg["mesh_base"] = dataset_cfg["mesh_base"].rstrip("/") + "_test"


def load_config() -> dict:
    with open(os.path.join(SCRIPT_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


# Swift / S3 helpers

def _parse_swift_url(url: str) -> tuple[str, str, str]:
    """Return (endpoint, bucket, prefix) from a swift:// URL.

    Swift URLs: swift://host/account/container/prefix...
    The account (AUTH_*) is handled by credentials. The S3 bucket is the
    Swift container, and everything after it is the key prefix.
    """
    stripped = url.rstrip("/").replace("swift://", "")
    parts = stripped.split("/", 3)
    endpoint = f"https://{parts[0]}"
    # parts[1] is the account (e.g. AUTH_team-isaac) — skip it
    bucket = parts[2] if len(parts) > 2 else ""
    prefix = parts[3] if len(parts) > 3 else ""
    return endpoint, bucket, prefix


def get_s3_client(swift_url: str):
    endpoint, bucket, prefix = _parse_swift_url(swift_url)
    access_key = os.environ.get("CSS_ACCESS_KEY", "")
    secret_key = os.environ.get("CSS_SECRET_KEY", "")
    if not access_key or not secret_key:
        print(
            "Error: Set CSS_ACCESS_KEY and CSS_SECRET_KEY environment variables.\n"
            "  source reconstruction/scripts/setup_css_env.sh",
            file=sys.stderr,
        )
        sys.exit(1)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
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


def get_hoi_metadata(client, bucket: str, seq_prefix: str) -> dict | None:
    key = f"{seq_prefix}/hoi_metadata.yaml".lstrip("/")
    try:
        resp = client.get_object(Bucket=bucket, Key=key)
        return yaml.safe_load(resp["Body"].read())
    except Exception:
        return None


def resolve_mesh_url(
    client, bucket: str, mesh_prefix: str, object_id: str, mesh_base: str,
) -> str | None:
    """Prefer einstar/, fall back to bsdf/. Return swift:// URL or None."""
    base = f"{mesh_prefix}/{object_id}".lstrip("/")
    for method in ("einstar", "bsdf"):
        method_pfx = f"{base}/{method}/"
        resp = client.list_objects_v2(Bucket=bucket, Prefix=method_pfx, MaxKeys=20)
        for obj in resp.get("Contents", []):
            if any(obj["Key"].endswith(ext) for ext in (".obj", ".glb", ".ply", ".stl")):
                return f"{mesh_base.rstrip('/')}/{object_id}/{method}/"
    return None


# OSMO helpers

def osmo_submit(
    workflow_yaml: str, pool: str, set_vars: dict[str, str], *, dry_run: bool = False,
) -> str:
    """Submit workflow and return the OSMO-assigned Workflow ID."""
    yaml_path = os.path.join(SCRIPT_DIR, workflow_yaml)
    set_str = " ".join(f'{k}="{v}"' for k, v in set_vars.items())
    cmd = f'osmo workflow submit {yaml_path} --set {set_str} --pool {pool}'
    print(f"  CMD: {cmd}")
    if dry_run:
        print("  [dry-run] skipping osmo submit")
        return set_vars.get("workflow_name", "dry-run")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd,
            output=result.stdout, stderr=result.stderr,
        )
    stdout = result.stdout.strip()
    print(f"  OSMO: {stdout}")
    for line in stdout.splitlines():
        if line.strip().startswith("Workflow ID"):
            return line.split("-", 1)[1].strip()
    return set_vars.get("workflow_name", stdout)


# Core logic

def _generate_workflow_name(pipeline_type: str, version: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ver = version.replace(".", "-")
    return f"v2d_{pipeline_type}_{ver}_{ts}"


def submit_sequence(
    sequence_name: str,
    dataset_name: str,
    dataset_cfg: dict,
    pipeline_type: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> str | None:
    """Build --set vars, submit OSMO workflow, record in DB. Return workflow name."""
    latest = get_latest_workflow(
        sequence_name, dataset_name, pipeline_type, db_path=DB_PATH, table=TABLE,
    )

    if latest and not force:
        if latest["status"] == "WAITING_WF":
            if not _confirm(
                f"Sequence {sequence_name} is running ({latest['workflow_name']}). "
                "Cancel and resubmit?"
            ):
                return None
            osmo_id = latest.get("osmo_workflow_id") or latest["workflow_name"]
            if not osmo_cancel(osmo_id):
                print(f"  {sequence_name}: cancel failed, aborting")
                return None
            update_workflow(latest["workflow_name"], status="FAIL",
                           details="cancelled_for_resubmit", db_path=DB_PATH,
                           table=TABLE)
            print(f"  {sequence_name}: cancelled previous run, resubmitting")
        elif latest["status"] == "WAITING_QC":
            if not _confirm(
                f"Sequence {sequence_name} is awaiting QC. Resubmit?"
            ):
                return None
        elif latest["status"] == "PASS":
            if not _confirm(f"Sequence {sequence_name} already PASS. Resubmit?"):
                return None

    swift_base = dataset_cfg["swift_base"]
    s3, bucket, base_pfx = get_s3_client(swift_base)
    version = get_latest_version(db_path=DB_PATH)
    if version is None:
        print("  ERROR: no pipeline version found. Run push_images.sh first.")
        return None
    workflow_name = _generate_workflow_name(pipeline_type, version)

    pipeline_cfg = dataset_cfg["pipelines"][pipeline_type]
    workflow_yaml = pipeline_cfg["workflow_yaml"]

    set_vars: dict[str, str] = {
        "workflow_name": workflow_name,
    }

    if pipeline_type == "mv_hoi_reconstruction":
        set_vars["rosbag_url"] = (
            f"{swift_base}/{dataset_cfg['data_path']}/{sequence_name}/"
        )

        # Metadata
        seq_data_pfx = f"{base_pfx}/{dataset_cfg['data_path']}/{sequence_name}"
        meta = get_hoi_metadata(s3, bucket, seq_data_pfx)
        if meta is None:
            print(f"  {sequence_name}: no hoi_metadata.yaml, skipping")
            return None

        # Extrinsics via calib_seq_name
        calib_seq = meta.get("calib_seq_name")
        if not calib_seq:
            print(f"  {sequence_name}: no calib_seq_name in hoi_metadata, skipping")
            return None
        calib_pfx = (
            f"{base_pfx}/{dataset_cfg['calibration_output_path']}"
            f"/{calib_seq}/calibrate_extrinsics"
        )
        if not path_exists(s3, bucket, calib_pfx):
            print(f"  {sequence_name}: calibration not found for {calib_seq}, skipping")
            return None
        set_vars["extrinsics_url"] = (
            f"{swift_base}/{dataset_cfg['calibration_output_path']}"
            f"/{calib_seq}/calibrate_extrinsics"
        )

        # Object ID + mesh
        object_id = (
            meta.get("object", {}).get("id")
            or meta.get("object_id")
            or meta.get("object_name")
        )
        if not object_id:
            print(f"  {sequence_name}: no object_id in hoi_metadata, skipping")
            return None

        _, mesh_bucket, mesh_pfx = _parse_swift_url(
            dataset_cfg["mesh_base"]
        )
        mesh_url = resolve_mesh_url(
            s3, mesh_bucket, mesh_pfx, object_id, dataset_cfg["mesh_base"],
        )
        if not mesh_url:
            print(f"  {sequence_name}: no mesh for object {object_id}, skipping")
            return None
        set_vars["mesh_url"] = mesh_url

        # Output base
        set_vars["swift_output_base"] = (
            f"{swift_base}/{dataset_cfg['data_output_path']}/{sequence_name}"
        )

        # QC thresholds
        thresholds = dataset_cfg.get("qc_thresholds", {})
        set_vars["max_chamfer_object"] = str(thresholds.get("max_chamfer_object", 30.0))
        set_vars["max_chamfer_human"] = str(thresholds.get("max_chamfer_human", 30.0))
        set_vars["min_mask_containment"] = str(thresholds.get("min_mask_containment", 0.8))
        set_vars["mask_bbox_padding"] = str(thresholds.get("mask_bbox_padding", 0.1))

        # HITL upload metadata (object_id / action_desc are read inside the
        # task from the preprocess-module hoi_metadata.yaml)
        set_vars["hitl_s3_base"] = dataset_cfg["hitl_s3_base"]
        batch_template = dataset_cfg.get("hitl_batch_name_template", "batch_{date}")
        set_vars["hitl_batch_name"] = batch_template.format(
            date=datetime.now().strftime("%Y%m%d"),
        )
        set_vars["s3_region"] = dataset_cfg.get("s3_region", "us-west-2")

    elif pipeline_type == "mv_calibration":
        set_vars["rosbag_url"] = (
            f"{swift_base}/{dataset_cfg['calibration_path']}/{sequence_name}/"
        )
        set_vars["swift_output_base"] = (
            f"{swift_base}/{dataset_cfg['calibration_output_path']}/{sequence_name}"
        )

    pool = dataset_cfg["osmo_pool"]

    print(f"  Submitting {pipeline_type} for {sequence_name} ({version})...")
    try:
        osmo_workflow_id = osmo_submit(workflow_yaml, pool, set_vars, dry_run=dry_run)
        if dry_run:
            print(f"  [dry-run] would insert workflow {workflow_name}")
            return workflow_name
        insert_workflow(
            sequence_name=sequence_name,
            dataset=dataset_name,
            pipeline_type=pipeline_type,
            pipeline_version=version,
            workflow_name=workflow_name,
            osmo_workflow_id=osmo_workflow_id,
            status="WAITING_WF",
            details="workflow_running",
            db_path=DB_PATH,
            table=TABLE,
        )
        print(f"  Workflow ID: {osmo_workflow_id}")
        return workflow_name
    except subprocess.CalledProcessError as e:
        if e.stdout and e.stdout.strip():
            print(f"  STDOUT: {e.stdout.strip()}")
        if e.stderr and e.stderr.strip():
            print(f"  STDERR: {e.stderr.strip()}")
        print(f"  ERROR (exit {e.returncode})")
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
    *, dry_run: bool = False, retry_failed: bool = False,
    start_time: str | None = None, end_time: str | None = None,
) -> None:
    """Discover sequences from Swift and submit workflows up to concurrency limit."""
    max_concurrent = dataset_cfg.get("max_concurrent", 10)

    in_progress = get_workflows_by_dataset(
        dataset_name, pipeline_type=pipeline_type,
        status="WAITING_WF", db_path=DB_PATH, table=TABLE,
    )
    available = max_concurrent - len(in_progress)
    print(f"In progress: {len(in_progress)}, available slots: {available}")
    if available <= 0:
        print(f"Max concurrent ({max_concurrent}) reached.")
        return

    swift_base = dataset_cfg["swift_base"]
    s3, bucket, base_pfx = get_s3_client(swift_base)

    if pipeline_type == "mv_calibration":
        scan_pfx = f"{base_pfx}/{dataset_cfg['calibration_path']}/"
    else:
        scan_pfx = f"{base_pfx}/{dataset_cfg['data_path']}/"
    sequences = list_sequences(s3, bucket, scan_pfx)
    print(f"Found {len(sequences)} sequences in {dataset_name}")

    if start_time or end_time:
        sequences = _filter_sequences_by_time(sequences, start_time, end_time)
        bounds = f"[{start_time or '-inf'}, {end_time or '+inf'}]"
        print(f"Filtered to {len(sequences)} sequences in time range {bounds}")

    skip_statuses = {"PASS", "WAITING_WF", "WAITING_QC"}
    if not retry_failed:
        skip_statuses.add("FAIL")

    submitted = 0
    for seq in sequences:
        if submitted >= available:
            print(f"Reached max concurrent limit ({max_concurrent})")
            break
        latest = get_latest_workflow(
            seq, dataset_name, pipeline_type, db_path=DB_PATH, table=TABLE,
        )
        if latest and latest["status"] in skip_statuses:
            continue
        wf = submit_sequence(
            seq, dataset_name, dataset_cfg, pipeline_type, dry_run=dry_run,
        )
        if wf:
            submitted += 1

    print(f"\nSubmitted {submitted} new workflow(s)")


def _confirm(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit OSMO workflows")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--pipeline", required=True,
                        help="Pipeline type (e.g. mv_calibration, mv_hoi_reconstruction)")
    parser.add_argument("--sequence", help="Single sequence (manual mode)")
    parser.add_argument("--force", action="store_true",
                        help="Force resubmit even if already PASS")
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
                        help="Use workflows_test table and append _test to output paths")
    args = parser.parse_args()

    global TABLE
    if args.test:
        TABLE = "workflows_test"

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(config['datasets'].keys())}")
        sys.exit(1)

    dataset_cfg = config["datasets"][args.dataset]
    if args.pipeline not in dataset_cfg.get("pipelines", {}):
        print(f"Unknown pipeline: {args.pipeline}")
        print(f"Available: {list(dataset_cfg['pipelines'].keys())}")
        sys.exit(1)

    if args.test:
        _apply_test_mode(dataset_cfg)

    init_db(DB_PATH)

    print("Refreshing waiting workflow statuses...")
    refresh_waiting(args.dataset, pipeline_type=args.pipeline, db_path=DB_PATH,
                    table=TABLE)

    if args.sequence:
        submit_sequence(
            args.sequence, args.dataset, dataset_cfg, args.pipeline,
            force=args.force, dry_run=args.dry_run,
        )
    else:
        auto_submit(
            args.dataset, dataset_cfg, args.pipeline,
            dry_run=args.dry_run, retry_failed=args.retry_failed,
            start_time=args.start_time, end_time=args.end_time,
        )


if __name__ == "__main__":
    main()
