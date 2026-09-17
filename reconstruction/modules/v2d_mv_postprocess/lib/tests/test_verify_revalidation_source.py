import hashlib
import json
from pathlib import Path
import sys

import pytest

LIB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB_DIR))

from verify_revalidation_source import (
    verify_revalidation_manifest,
    verify_revalidation_source,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_frozen_source_detects_same_size_content_change(tmp_path: Path):
    source = tmp_path / "source"
    legacy = tmp_path / "legacy"
    source.mkdir()
    legacy.mkdir()
    source_payload = source / "metadata.json"
    legacy_payload = legacy / "poses.npy"
    source_payload.write_bytes(b"source")
    legacy_payload.write_bytes(b"legacy")
    record = {
        "sequence": "sequence",
        "data_output_objects": [{
            "relative_path": "metadata.json", "size": 6,
            "sha256": hashlib.sha256(b"source").hexdigest(),
        }],
        "data_export_objects": [{
            "relative_path": "poses.npy", "size": 6,
            "sha256": hashlib.sha256(b"legacy").hexdigest(),
        }],
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"sequences": [record]}))
    manifest_hash = hashlib.sha256(_canonical(record).encode()).hexdigest()
    output = tmp_path / "verification.json"

    result = verify_revalidation_source(
        source_dir=source, legacy_export_dir=legacy,
        inventory_path=inventory, sequence="sequence",
        expected_manifest_sha256=manifest_hash, output_path=output,
    )
    assert result["status"] == "PASS"
    assert result["checked_objects"] == 2

    source_payload.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="SHA-256 changed"):
        verify_revalidation_source(
            source_dir=source, legacy_export_dir=legacy,
            inventory_path=inventory, sequence="sequence",
            expected_manifest_sha256=manifest_hash, output_path=output,
        )


def test_manifest_only_verification_binds_workflow_to_frozen_record(tmp_path: Path):
    record = {"sequence": "sequence", "data_output_objects": [{"size": 6}]}
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"sequences": [record]}))
    manifest_hash = hashlib.sha256(_canonical(record).encode()).hexdigest()
    output = tmp_path / "verification.json"

    result = verify_revalidation_manifest(
        inventory_path=inventory,
        sequence="sequence",
        expected_manifest_sha256=manifest_hash,
        output_path=output,
    )

    assert result["status"] == "PASS"
    assert result["checked_objects"] == 0
    assert result["object_validation_authority"] == "campaign_controller"
    with pytest.raises(ValueError, match="Frozen manifest mismatch"):
        verify_revalidation_manifest(
            inventory_path=inventory,
            sequence="sequence",
            expected_manifest_sha256="0" * 64,
            output_path=output,
        )
