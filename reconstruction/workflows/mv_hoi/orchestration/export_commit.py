"""Read-only validation of a committed export in CSS."""

from __future__ import annotations

try:
    from .storage import parse_storage_url, s3_client_kwargs
except ImportError:  # Direct script execution.
    from storage import parse_storage_url, s3_client_kwargs

import json
import hashlib

import boto3
from botocore.config import Config


SUPPORTED_SCHEMAS = {
    "v2d.mv_hoi.export_commit.v1",
    "v2d.mv_hoi.export_commit.v2",
    "v2d.mv_hoi.export_commit.v3",
    "v2d.mv_hoi.revalidation_export_commit.v1",
    "v2d.mv_hoi.revalidation_export_commit.v2",
    "v2d.mv_hoi.revalidation_export_commit.v3",
}
REJECTION_SCHEMA = "v2d.mv_hoi.export_rejection.v1"


class CandidateCleanupError(RuntimeError):
    """A final export committed, but its request-scoped staging cleanup did not."""


def _parse(url: str) -> tuple[object, str, str]:
    endpoint, bucket, prefix = parse_storage_url(url)
    client = boto3.client(
        "s3", **s3_client_kwargs(endpoint),
        config=Config(
            signature_version="s3v4",
            retries={"mode": "adaptive", "max_attempts": 6},
        ),
    )
    return client, bucket, prefix


def verify_remote_export_commit(url: str) -> dict:
    client, bucket, prefix = _parse(url)
    prefix = prefix.rstrip("/")
    raw = client.get_object(Bucket=bucket, Key=f"{prefix}/commit.json")["Body"].read()
    commit = json.loads(raw)
    if commit.get("schema") not in SUPPORTED_SCHEMAS or commit.get("complete") is not True:
        raise ValueError("Unsupported or incomplete remote export commit")
    expected = {item["path"]: item for item in commit.get("files", [])}
    actual: dict[str, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
        for item in page.get("Contents", []):
            relative = item["Key"][len(prefix) + 1:]
            if relative != "commit.json":
                actual[relative] = int(item["Size"])
    if set(expected) != set(actual):
        raise ValueError("Remote committed export key set differs from destination")
    for path, size in actual.items():
        if int(expected[path]["size"]) != size:
            raise ValueError(f"Remote committed export size mismatch: {path}")
    if int(commit.get("file_count", -1)) != len(expected):
        raise ValueError("Remote committed export file count is inconsistent")
    if int(commit.get("total_bytes", -1)) != sum(actual.values()):
        raise ValueError("Remote committed export byte count is inconsistent")
    return commit


def validate_export_rejection(report: dict) -> dict:
    """Validate a compact early-QC rejection without importing image code."""
    if report.get("schema") != REJECTION_SCHEMA or report.get("status") != "REJECTED":
        raise ValueError("Unsupported export rejection report")
    expected = dict(report)
    observed_hash = expected.pop("report_sha256", None)
    payload = json.dumps(
        expected, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    if observed_hash != hashlib.sha256(payload).hexdigest():
        raise ValueError("Export rejection report hash is inconsistent")
    gates = report.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("Export rejection report has no gates")
    failed = 0
    for gate in gates:
        if gate.get("comparison") != ">":
            raise ValueError("Export rejection gate comparison is invalid")
        expected_status = (
            "FAIL"
            if float(gate["observed"]) > float(gate["threshold"])
            else "PASS"
        )
        if gate.get("status") != expected_status:
            raise ValueError("Export rejection gate result is inconsistent")
        failed += expected_status == "FAIL"
    if not failed:
        raise ValueError("Export rejection report contains no failed gate")
    return report


def read_remote_export_rejection(url: str) -> dict:
    client, bucket, prefix = _parse(url)
    raw = client.get_object(
        Bucket=bucket, Key=prefix.rstrip("/") + "/export_rejection.json",
    )["Body"].read()
    report = json.loads(raw)
    return validate_export_rejection(report)


def cleanup_rejected_export_candidate(
    candidate_url: str, *, expected_report: dict,
) -> dict:
    """Delete one exact request-scoped rejected prefix, evidence file last."""
    validate_export_rejection(expected_report)
    client, bucket, prefix = _parse(candidate_url)
    root = prefix.rstrip("/")
    leaf = root.rsplit("/", 1)[-1]
    if not (leaf.startswith("request_") or leaf == "candidate_export"):
        raise ValueError("Refusing to delete a non-request-scoped rejected export")
    objects: list[dict] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=root + "/"):
        objects.extend(page.get("Contents", []))
    report_key = root + "/export_rejection.json"
    matching = [item for item in objects if item["Key"] == report_key]
    if len(matching) != 1:
        raise ValueError("Rejected candidate lacks unique rejection evidence")
    observed = json.loads(
        client.get_object(Bucket=bucket, Key=report_key)["Body"].read()
    )
    if observed != expected_report:
        raise ValueError("Rejected candidate evidence changed before cleanup")
    payloads = [item for item in objects if item["Key"] != report_key]
    for offset in range(0, len(payloads), 1000):
        response = client.delete_objects(
            Bucket=bucket,
            Delete={
                "Objects": [
                    {"Key": item["Key"]}
                    for item in payloads[offset:offset + 1000]
                ],
                "Quiet": True,
            },
        ) or {}
        if response.get("Errors"):
            raise CandidateCleanupError(str(response["Errors"][:3]))
    client.delete_object(Bucket=bucket, Key=report_key)
    remaining = []
    for page in paginator.paginate(Bucket=bucket, Prefix=root + "/"):
        remaining.extend(page.get("Contents", []))
    if remaining:
        raise CandidateCleanupError(
            f"Rejected candidate prefix retains {len(remaining)} object(s)"
        )
    return {
        "deleted_object_count": len(objects),
        "reclaimed_bytes": sum(int(item.get("Size") or 0) for item in objects),
    }


def cleanup_promoted_export_candidate(
    candidate_url: str, destination_url: str, *, expected_commit: dict | None = None,
) -> dict:
    """Delete one verified request-scoped candidate, with commit.json last."""

    if candidate_url.rstrip("/") == destination_url.rstrip("/"):
        raise ValueError("Candidate and destination export prefixes must differ")
    candidate_client, candidate_bucket, candidate_prefix = _parse(candidate_url)
    destination_client, destination_bucket, _ = _parse(destination_url)
    if candidate_client.meta.endpoint_url != destination_client.meta.endpoint_url:
        raise ValueError("Candidate and destination must use the same CSS endpoint")
    if candidate_bucket != destination_bucket:
        raise ValueError("Candidate and destination must use the same CSS bucket")
    leaf = candidate_prefix.rstrip("/").rsplit("/", 1)[-1]
    if not (leaf.startswith("request_") or leaf == "candidate_export"):
        raise ValueError("Refusing to delete a non-request-scoped export candidate")

    published = verify_remote_export_commit(destination_url)
    if expected_commit is not None and published != expected_commit:
        raise ValueError("Published export differs from the expected candidate commit")

    root = candidate_prefix.rstrip("/")
    objects: list[dict] = []
    try:
        paginator = candidate_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=candidate_bucket, Prefix=root + "/"):
            objects.extend(page.get("Contents", []))
    except Exception as exc:
        raise CandidateCleanupError(str(exc)) from exc
    if not objects:
        return {"deleted_object_count": 0, "reclaimed_bytes": 0}

    commit_key = f"{root}/commit.json"
    commit_objects = [item for item in objects if item["Key"] == commit_key]
    if commit_objects:
        try:
            raw = candidate_client.get_object(
                Bucket=candidate_bucket, Key=commit_key,
            )["Body"].read()
        except Exception as exc:
            raise CandidateCleanupError(str(exc)) from exc
        staged_commit = json.loads(raw)
        if staged_commit != published:
            raise ValueError("Staged and published export commits differ")

    def list_objects() -> list[dict]:
        found: list[dict] = []
        paginator = candidate_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=candidate_bucket, Prefix=root + "/",
        ):
            found.extend(page.get("Contents", []))
        return found

    try:
        remaining = objects
        for _attempt in range(3):
            payloads = [item for item in remaining if item["Key"] != commit_key]
            delete_errors: list[dict] = []
            for offset in range(0, len(payloads), 1000):
                response = candidate_client.delete_objects(
                    Bucket=candidate_bucket,
                    Delete={
                        "Objects": [
                            {"Key": item["Key"]}
                            for item in payloads[offset:offset + 1000]
                        ],
                        "Quiet": True,
                    },
                ) or {}
                delete_errors.extend(response.get("Errors", []))
            remaining = list_objects()
            if delete_errors or any(
                item["Key"] != commit_key for item in remaining
            ):
                continue
            if any(item["Key"] == commit_key for item in remaining):
                candidate_client.delete_object(
                    Bucket=candidate_bucket, Key=commit_key,
                )
            remaining = list_objects()
            if not remaining:
                break
        if remaining:
            raise RuntimeError(
                "Candidate prefix is not empty after deletion: "
                f"{len(remaining)} object(s) remain"
            )
    except Exception as exc:
        raise CandidateCleanupError(str(exc)) from exc
    return {
        "deleted_object_count": len(objects),
        "reclaimed_bytes": sum(int(item.get("Size") or 0) for item in objects),
    }


def publish_remote_export_commit(candidate_url: str, destination_url: str) -> dict:
    """Replace a canonical export from an isolated candidate, commit last."""
    if candidate_url.rstrip("/") == destination_url.rstrip("/"):
        raise ValueError("Candidate and destination export prefixes must differ")
    candidate_client, candidate_bucket, candidate_prefix = _parse(candidate_url)
    destination_client, destination_bucket, destination_prefix = _parse(
        destination_url
    )
    if candidate_client.meta.endpoint_url != destination_client.meta.endpoint_url:
        raise ValueError("Candidate and destination must use the same CSS endpoint")
    if candidate_bucket != destination_bucket:
        raise ValueError("Candidate and destination must use the same CSS bucket")
    commit = verify_remote_export_commit(candidate_url)
    destination_prefix = destination_prefix.rstrip("/")
    candidate_prefix = candidate_prefix.rstrip("/")

    existing_keys: list[str] = []
    paginator = destination_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=destination_bucket, Prefix=destination_prefix + "/",
    ):
        existing_keys.extend(item["Key"] for item in page.get("Contents", []))
    for offset in range(0, len(existing_keys), 1000):
        destination_client.delete_objects(
            Bucket=destination_bucket,
            Delete={
                "Objects": [
                    {"Key": key} for key in existing_keys[offset:offset + 1000]
                ],
                "Quiet": True,
            },
        )

    for item in commit["files"]:
        relative = item["path"]
        destination_client.copy_object(
            Bucket=destination_bucket,
            Key=f"{destination_prefix}/{relative}",
            CopySource={
                "Bucket": candidate_bucket,
                "Key": f"{candidate_prefix}/{relative}",
            },
        )
    destination_client.copy_object(
        Bucket=destination_bucket,
        Key=f"{destination_prefix}/commit.json",
        CopySource={
            "Bucket": candidate_bucket,
            "Key": f"{candidate_prefix}/commit.json",
        },
    )
    published = verify_remote_export_commit(destination_url)
    if published != commit:
        raise ValueError("Published export commit differs from candidate")
    return published
