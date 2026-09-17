"""Verify mounted revalidation inputs against one frozen inventory entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _inventory_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(path.glob("*.json")) if path.is_dir() else []
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one inventory JSON at {path}")
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_revalidation_source(
    *, source_dir: str | Path, legacy_export_dir: str | Path,
    inventory_path: str | Path, sequence: str,
    expected_manifest_sha256: str, output_path: str | Path,
) -> dict:
    inventory = json.loads(_inventory_path(inventory_path).read_text())
    records = [item for item in inventory.get("sequences", []) if item["sequence"] == sequence]
    if len(records) != 1:
        raise ValueError(f"Inventory must contain exactly one entry for {sequence}")
    record = records[0]
    actual_hash = hashlib.sha256(_canonical(record).encode()).hexdigest()
    if actual_hash != expected_manifest_sha256:
        raise ValueError(
            f"Frozen manifest mismatch: expected {expected_manifest_sha256}, got {actual_hash}"
        )
    checked = 0
    for field, root in (
        ("data_output_objects", Path(source_dir)),
        ("data_export_objects", Path(legacy_export_dir)),
    ):
        for item in record.get(field, []):
            path = root / item["relative_path"]
            if not path.is_file():
                raise ValueError(f"Frozen source object is missing: {path}")
            if path.stat().st_size != int(item["size"]):
                raise ValueError(f"Frozen source object size changed: {path}")
            if item.get("sha256") and _sha256(path) != item["sha256"]:
                raise ValueError(f"Frozen source object SHA-256 changed: {path}")
            checked += 1
    result = {
        "schema": "v2d.mv_hoi.revalidation_source_verification.v1",
        "status": "PASS",
        "sequence": sequence,
        "source_manifest_sha256": actual_hash,
        "checked_objects": checked,
        "controller_validates_etags": True,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def verify_revalidation_manifest(
    *, inventory_path: str | Path, sequence: str,
    expected_manifest_sha256: str, output_path: str | Path,
) -> dict:
    """Bind a workflow to a frozen record without mounting source payloads.

    The campaign controller validates every frozen CSS object immediately
    before submission, after workflow completion, and before publication.
    This in-workflow check verifies the immutable record identity without
    downloading the complete legacy source solely to repeat those checks.
    """
    inventory = json.loads(_inventory_path(inventory_path).read_text())
    records = [item for item in inventory.get("sequences", []) if item["sequence"] == sequence]
    if len(records) != 1:
        raise ValueError(f"Inventory must contain exactly one entry for {sequence}")
    actual_hash = hashlib.sha256(_canonical(records[0]).encode()).hexdigest()
    if actual_hash != expected_manifest_sha256:
        raise ValueError(
            f"Frozen manifest mismatch: expected {expected_manifest_sha256}, got {actual_hash}"
        )
    result = {
        "schema": "v2d.mv_hoi.revalidation_manifest_verification.v1",
        "status": "PASS",
        "sequence": sequence,
        "source_manifest_sha256": actual_hash,
        "checked_objects": 0,
        "object_validation_authority": "campaign_controller",
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir")
    parser.add_argument("--legacy-export-dir")
    parser.add_argument("--inventory-path", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    values = vars(args)
    if bool(args.source_dir) != bool(args.legacy_export_dir):
        parser.error("--source-dir and --legacy-export-dir must be provided together")
    if args.source_dir:
        result = verify_revalidation_source(**values)
    else:
        values.pop("source_dir")
        values.pop("legacy_export_dir")
        result = verify_revalidation_manifest(**values)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
