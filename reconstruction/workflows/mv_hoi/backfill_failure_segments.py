#!/usr/bin/env python3
"""Backfill exported failure segment frame indices after frontend subsampling.

The affected exported files already use the repository convention of
0-indexed, half-open frame ranges, but the frame numbers are half-scale. This
script maps each segment [x, y) to [2x, 2y), clipping the rare odd-length
sequence end from frame_count + 1 back to frame_count.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config_utils import RECON_PIPELINE, get_dataset_cfg, get_pipeline_export_path
from config_utils import load_config as _load_config


DEFAULT_DATASET = "sc_office_4exo_1"
FAILURE_SEGMENTS_NAME = "failure_segments.json"
BACKUP_SEGMENTS_NAME = "failure_segments.pre_subsampling_backfill.json"
MANIFEST_DIR_NAME = "_backfills"


@dataclass
class BackfillSummary:
    total_files: int = 0
    empty_files: int = 0
    candidate_files: int = 0
    would_update_files: int = 0
    updated_files: int = 0
    skipped_already_backfilled: int = 0
    rejected_files: int = 0
    error_files: int = 0
    total_segments: int = 0
    would_update_segments: int = 0
    updated_segments: int = 0
    clipped_files: int = 0
    clipped_segments: int = 0


def load_config() -> dict:
    return _load_config(SCRIPT_DIR)


def _parse_swift_url(url: str) -> tuple[str, str, str]:
    stripped = url.rstrip("/").replace("swift://", "")
    parts = stripped.split("/", 3)
    if len(parts) < 3:
        raise ValueError(
            "Swift URL must have form swift://host/account/container/prefix"
        )
    endpoint = f"https://{parts[0]}"
    bucket = parts[2]
    prefix = parts[3] if len(parts) > 3 else ""
    return endpoint, bucket, prefix.strip("/")


def get_s3_client(swift_url: str):
    import boto3

    endpoint, bucket, prefix = _parse_swift_url(swift_url)
    access_key = os.environ.get("CSS_ACCESS_KEY", "")
    secret_key = os.environ.get("CSS_SECRET_KEY", "")
    if not access_key or not secret_key:
        print(
            "Error: Set CSS_ACCESS_KEY and CSS_SECRET_KEY environment variables.\n"
            "  source ~/secrets/setup_css_env.sh",
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


def dataset_export_url(dataset_cfg: dict) -> str:
    return (
        f"{dataset_cfg['swift_base'].rstrip('/')}/"
        f"{get_pipeline_export_path(dataset_cfg, RECON_PIPELINE).strip('/')}"
    )


def frame_count_from_edex_text(text: str) -> int | None:
    try:
        edex = json.loads(text)
        header = edex[0] if isinstance(edex, list) and edex else edex
        frame_start = int(header["frame_start"])
        frame_end = int(header["frame_end"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    frame_count = frame_end - frame_start
    return frame_count if frame_count > 0 else None


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a frame number")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    raise ValueError(f"not an integer: {value!r}")


def backfill_segment(segment: dict, frame_count: int) -> tuple[dict, bool]:
    try:
        start_frame = _coerce_int(segment["start_frame"])
        end_frame = _coerce_int(segment["end_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid segment frame field: {exc}") from exc

    new_start = start_frame * 2
    new_end = end_frame * 2
    clipped = False
    if new_end == frame_count + 1:
        new_end = frame_count
        clipped = True

    if new_start < 0 or new_end <= new_start or new_end > frame_count:
        raise ValueError(
            "invalid doubled segment: "
            f"start_frame={new_start}, end_frame={new_end}, "
            f"frame_count={frame_count}"
        )

    updated = dict(segment)
    updated["start_frame"] = new_start
    updated["end_frame"] = new_end
    return updated, clipped


def backup_key_for(failure_key: str) -> str:
    if not failure_key.endswith(f"/{FAILURE_SEGMENTS_NAME}"):
        raise ValueError(f"not a failure segment key: {failure_key}")
    return failure_key[: -len(FAILURE_SEGMENTS_NAME)] + BACKUP_SEGMENTS_NAME


def sequence_name_from_key(export_prefix: str, key: str) -> str:
    prefix = export_prefix.strip("/")
    rel = key[len(prefix) :].lstrip("/") if key.startswith(prefix) else key
    return rel.split("/", 1)[0]


def list_failure_segment_keys(client, bucket: str, export_prefix: str) -> list[str]:
    prefix = export_prefix.strip("/")
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith(f"/{FAILURE_SEGMENTS_NAME}"):
                keys.append(key)
    return sorted(keys)


def read_text(client, bucket: str, key: str) -> str:
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read()
    if isinstance(body, bytes):
        return body.decode("utf-8")
    return str(body)


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    return True


def put_text(client, bucket: str, key: str, text: str) -> None:
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="application/json",
    )


def build_manifest_key(export_prefix: str, timestamp: str) -> str:
    return (
        f"{export_prefix.strip('/')}/{MANIFEST_DIR_NAME}/"
        f"failure_segments_subsampling_{timestamp}.json"
    )


def build_local_manifest_path(timestamp: str) -> Path:
    return SCRIPT_DIR / "logs" / f"backfill_failure_segments_manifest_{timestamp}.json"


def process_failure_file(
    client,
    bucket: str,
    export_prefix: str,
    failure_key: str,
    *,
    apply: bool,
    force: bool,
) -> dict:
    sequence_name = sequence_name_from_key(export_prefix, failure_key)
    backup_key = backup_key_for(failure_key)
    record: dict[str, Any] = {
        "sequence_name": sequence_name,
        "key": failure_key,
        "backup_key": backup_key,
    }

    try:
        failure_text = read_text(client, bucket, failure_key)
        segments = json.loads(failure_text)
    except Exception as exc:
        record.update(status="error", reason=f"could not read segments: {exc}")
        return record

    if not isinstance(segments, list):
        record.update(status="rejected", reason="failure_segments is not a list")
        return record

    record["segment_count"] = len(segments)
    if not segments:
        record.update(status="empty")
        return record

    if object_exists(client, bucket, backup_key) and not force:
        record.update(status="already_backfilled")
        return record

    edex_key = f"{export_prefix.strip('/')}/{sequence_name}/edex"
    try:
        frame_count = frame_count_from_edex_text(read_text(client, bucket, edex_key))
    except Exception as exc:
        record.update(status="rejected", reason=f"could not read edex: {exc}")
        return record

    if frame_count is None:
        record.update(status="rejected", reason="could not resolve frame_count")
        return record

    updated_segments: list[dict] = []
    clipped_indices: list[int] = []
    try:
        for index, segment in enumerate(segments):
            updated_segment, clipped = backfill_segment(segment, frame_count)
            updated_segments.append(updated_segment)
            if clipped:
                clipped_indices.append(index)
    except ValueError as exc:
        record.update(
            status="rejected",
            reason=str(exc),
            frame_count=frame_count,
            old_segments=segments,
        )
        return record

    record.update(
        status="updated" if apply else "would_update",
        frame_count=frame_count,
        old_segments=segments,
        new_segments=updated_segments,
        clipped_indices=clipped_indices,
    )

    if apply:
        if not object_exists(client, bucket, backup_key):
            put_text(client, bucket, backup_key, failure_text)
        put_text(
            client,
            bucket,
            failure_key,
            json.dumps(updated_segments, indent=2) + "\n",
        )

    return record


def summarize_records(records: list[dict], *, apply: bool) -> BackfillSummary:
    summary = BackfillSummary(total_files=len(records))
    for record in records:
        status = record.get("status")
        segment_count = int(record.get("segment_count") or 0)
        clipped_count = len(record.get("clipped_indices") or [])
        if status == "empty":
            summary.empty_files += 1
        elif status == "already_backfilled":
            summary.candidate_files += 1
            summary.total_segments += segment_count
            summary.skipped_already_backfilled += 1
        elif status == "would_update":
            summary.candidate_files += 1
            summary.total_segments += segment_count
            summary.would_update_files += 1
            summary.would_update_segments += segment_count
        elif status == "updated":
            summary.candidate_files += 1
            summary.total_segments += segment_count
            summary.updated_files += 1
            summary.updated_segments += segment_count
        elif status == "rejected":
            summary.candidate_files += 1
            summary.total_segments += segment_count
            summary.rejected_files += 1
        elif status == "error":
            summary.error_files += 1

        if clipped_count:
            summary.clipped_files += 1
            summary.clipped_segments += clipped_count

    if not apply:
        summary.updated_files = 0
        summary.updated_segments = 0
    return summary


def run_backfill(
    client,
    bucket: str,
    export_prefix: str,
    *,
    dataset: str,
    export_url: str,
    apply: bool,
    force: bool = False,
    sequence_names: set[str] | None = None,
    limit: int | None = None,
    timestamp: str | None = None,
    write_manifest: bool = True,
) -> dict:
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    failure_keys = list_failure_segment_keys(client, bucket, export_prefix)
    if sequence_names:
        failure_keys = [
            key
            for key in failure_keys
            if sequence_name_from_key(export_prefix, key) in sequence_names
        ]
    if limit is not None:
        failure_keys = failure_keys[:limit]

    records = [
        process_failure_file(
            client,
            bucket,
            export_prefix,
            key,
            apply=apply,
            force=force,
        )
        for key in failure_keys
    ]
    summary = summarize_records(records, apply=apply)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "export_url": export_url,
        "export_prefix": export_prefix,
        "apply": apply,
        "force": force,
        "summary": asdict(summary),
        "files": records,
    }

    if apply and write_manifest:
        manifest_text = json.dumps(manifest, indent=2) + "\n"
        put_text(
            client,
            bucket,
            build_manifest_key(export_prefix, timestamp),
            manifest_text,
        )
        local_path = build_local_manifest_path(timestamp)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(manifest_text)
        manifest["local_manifest_path"] = str(local_path)
        manifest["remote_manifest_key"] = build_manifest_key(export_prefix, timestamp)

    return manifest


def print_summary(manifest: dict) -> None:
    summary = manifest["summary"]
    mode = "APPLY" if manifest["apply"] else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Dataset: {manifest['dataset']}")
    print(f"Export: {manifest['export_url']}")
    print(f"Total failure segment files: {summary['total_files']}")
    print(f"Empty files: {summary['empty_files']}")
    print(f"Candidate files: {summary['candidate_files']}")
    if manifest["apply"]:
        print(f"Updated files: {summary['updated_files']}")
        print(f"Updated segments: {summary['updated_segments']}")
    else:
        print(f"Would update files: {summary['would_update_files']}")
        print(f"Would update segments: {summary['would_update_segments']}")
    print(f"Skipped already backfilled: {summary['skipped_already_backfilled']}")
    print(f"Clipped files: {summary['clipped_files']}")
    print(f"Clipped segments: {summary['clipped_segments']}")
    print(f"Rejected files: {summary['rejected_files']}")
    print(f"Error files: {summary['error_files']}")

    rejected = [
        record
        for record in manifest["files"]
        if record.get("status") in {"rejected", "error"}
    ]
    if rejected:
        print("Rejected/error samples:")
        for record in rejected[:10]:
            print(
                f"  {record['sequence_name']}: "
                f"{record.get('reason', record.get('status'))}"
            )

    if manifest.get("local_manifest_path"):
        print(f"Local manifest: {manifest['local_manifest_path']}")
    if manifest.get("remote_manifest_key"):
        print(f"Remote manifest: {manifest['remote_manifest_key']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill exported failure segment frame indices."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--export-prefix",
        help=(
            "Swift URL for data_export. Defaults to the dataset swift_base plus "
            "the reconstruction export_path from config.yaml."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write backups, rewritten failure_segments.json files, and manifests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Process files even when the backup sidecar already exists.",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Only process this sequence name. May be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, help="Only process the first N files.")
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="With --apply, do not write local or remote manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    dataset_cfg = get_dataset_cfg(config, args.dataset)
    export_url = args.export_prefix or dataset_export_url(dataset_cfg)
    client, bucket, export_prefix = get_s3_client(export_url)

    manifest = run_backfill(
        client,
        bucket,
        export_prefix,
        dataset=args.dataset,
        export_url=export_url,
        apply=args.apply,
        force=args.force,
        sequence_names=set(args.sequence) if args.sequence else None,
        limit=args.limit,
        write_manifest=not args.no_manifest,
    )
    print_summary(manifest)

    summary = manifest["summary"]
    if summary["rejected_files"] or summary["error_files"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
