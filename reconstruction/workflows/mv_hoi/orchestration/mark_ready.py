# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a ready_for_processing marker in the HITL S3 batch folder.

Usage:
    python mark_ready.py --dataset sc_office_4exo_1 --batch batch_20260419
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MV_HOI_DIR = os.path.dirname(SCRIPT_DIR)
try:
    from .config_utils import (
        RECON_PIPELINE, RECONSTRUCTION_WORKFLOW, get_workflow_cfg,
        load_config as _load_config,
    )
    from .runtime import require_submit_authority
    from . import db
except ImportError:  # Direct script execution.
    from config_utils import (
        RECON_PIPELINE, RECONSTRUCTION_WORKFLOW, get_workflow_cfg,
        load_config as _load_config,
    )
    from runtime import require_submit_authority
    import db


def load_config() -> dict:
    return _load_config(MV_HOI_DIR)


def expected_batch_items(dataset: str, batch: str, *, db_path: str) -> tuple[set[str], list[str]]:
    date_token = batch.removeprefix("batch_")
    if len(date_token) != 8 or not date_token.isdigit():
        raise ValueError(f"Unsupported HITL batch name: {batch}")
    connection = db.get_connection(db_path)
    try:
        rows = connection.execute(
            """SELECT r.hitl_item_id, r.status, e.workflow_name
               FROM reconstruction_runs r
               JOIN sequences s ON s.id=r.sequence_id
               JOIN workflow_executions e ON e.id=r.workflow_execution_id
               LEFT JOIN stage_requests sr ON sr.id=r.request_id
               LEFT JOIN processing_campaigns c ON c.id=sr.campaign_id
               WHERE s.dataset=? AND e.workflow_name LIKE ?
                 AND r.hitl_item_id IS NOT NULL
                 AND (
                     r.request_id IS NULL
                     OR (
                         sr.status NOT IN ('FAILED','CANCELED')
                         AND (
                             sr.campaign_id IS NULL
                             OR c.status IN ('FROZEN','RUNNING')
                         )
                     )
                 )""",
            (dataset, f"%_{date_token}_%"),
        ).fetchall()
    finally:
        connection.close()
    active = [
        row["workflow_name"] for row in rows
        if row["status"] in ("SUBMITTING", "RUNNING", "UNKNOWN")
    ]
    expected = {
        Path(row["hitl_item_id"]).stem
        for row in rows if row["status"] == "SUCCEEDED"
    }
    return expected, active


def uploaded_batch_items(listing: str, batch: str) -> tuple[set[str], set[str]]:
    videos: set[str] = set()
    jsons: set[str] = set()
    for line in listing.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) != 4:
            continue
        try:
            size = int(parts[2])
        except ValueError:
            continue
        key = parts[3]
        if size <= 0:
            raise ValueError(f"HITL object is empty: {key}")
        marker = f"/{batch}/"
        relative = key.split(marker, 1)[-1] if marker in key else key
        if relative.startswith("dataset/") and relative.endswith(".mp4"):
            videos.add(Path(relative).stem)
        elif relative.startswith("jsons/") and relative.endswith(".json"):
            jsons.add(Path(relative).stem)
    return videos, jsons


def _conditional_marker_command(marker_path: str, payload_path: str) -> list[str]:
    parsed = urlparse(marker_path)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise ValueError(f"Unsupported HITL marker path: {marker_path}")
    return [
        "aws", "s3api", "put-object",
        "--bucket", parsed.netloc,
        "--key", parsed.path.lstrip("/"),
        "--body", payload_path,
        "--content-type", "application/json",
        "--if-none-match", "*",
    ]


def put_ready_marker(marker_path: str, payload: str) -> bool:
    """Create the marker once; return False if another run already created it."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json",
    ) as payload_file:
        payload_file.write(payload)
        payload_file.flush()
        command = _conditional_marker_command(marker_path, payload_file.name)
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    error = "\n".join((result.stdout, result.stderr))
    if "PreconditionFailed" in error or "412" in error:
        return False
    raise RuntimeError(error.strip() or "conditional marker upload failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ready_for_processing marker for a HITL batch",
    )
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--batch", required=True, help="Batch name (e.g. batch_20260419)")
    parser.add_argument("--db", default=db.DB_PATH)
    parser.add_argument("--dry_run", action="store_true", help="Print command without running")
    args = parser.parse_args()

    config = load_config()
    if args.dataset not in config["datasets"]:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {list(config['datasets'].keys())}")
        sys.exit(1)

    dataset_cfg = config["datasets"][args.dataset]
    workflow_cfg = get_workflow_cfg(
        dataset_cfg,
        RECON_PIPELINE,
        RECONSTRUCTION_WORKFLOW,
    )
    base_path = workflow_cfg["hitl_s3_base"]
    batch_path = os.path.join(base_path, args.batch) + "/"
    marker_path = os.path.join(base_path, args.batch, "markers", "ready_for_processing")

    expected, active = expected_batch_items(
        args.dataset, args.batch, db_path=args.db,
    )
    if active:
        print(
            "  Reconstruction workflows for this batch are still active; "
            f"refusing readiness: {active[:10]}", file=sys.stderr,
        )
        sys.exit(1)

    # List and reconcile every uploaded item in the batch folder.
    ls_result = subprocess.run(
        ["aws", "s3", "ls", batch_path, "--recursive"],
        capture_output=True, text=True,
    )
    if ls_result.returncode != 0:
        print(f"  ERROR listing batch: {ls_result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    videos, jsons = uploaded_batch_items(ls_result.stdout, args.batch)

    print(f"Batch: {args.batch}")
    print(f"  Expected items: {len(expected)}")
    print(f"  Uploaded videos: {len(videos)}")
    print(f"  Uploaded JSONs: {len(jsons)}")
    print()

    if not expected:
        print("  No successful database runs belong to this batch, aborting.")
        sys.exit(1)
    if videos != expected or jsons != expected:
        print(
            "  HITL batch does not reconcile with successful reconstruction runs: "
            f"missing_videos={sorted(expected - videos)[:10]}, "
            f"missing_jsons={sorted(expected - jsons)[:10]}, "
            f"unexpected_videos={sorted(videos - expected)[:10]}, "
            f"unexpected_jsons={sorted(jsons - expected)[:10]}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create the batch marker exactly once. The conditional write prevents an
    # hourly retry from changing its timestamp or emitting another object event.
    payload = json.dumps({"created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    preview_command = _conditional_marker_command(marker_path, "<payload.json>")
    print(f"  {shlex.join(preview_command)}")

    if args.dry_run:
        print("  [dry-run] skipping")
        return

    require_submit_authority("publish a HITL ready marker")

    try:
        created = put_ready_marker(marker_path, payload)
    except RuntimeError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    if created:
        print(f"  Marker created: {marker_path}")
    else:
        print(f"  Marker already exists; unchanged: {marker_path}")


if __name__ == "__main__":
    main()
