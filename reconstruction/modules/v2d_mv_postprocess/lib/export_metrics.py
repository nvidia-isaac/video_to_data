"""Copy small, immutable processing evidence into a committed export."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping


METRICS_MANIFEST_SCHEMA = "v2d.mv_hoi.metrics_manifest.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_metrics_bundle(
    output_root: str | Path,
    *,
    stage: str,
    request_id: int,
    evidence: Mapping[str, str | Path | None],
    metadata: Mapping[str, object | None],
    required: frozenset[str] = frozenset(),
) -> dict:
    """Copy named evidence and write a hash manifest beside it.

    Evidence names are paths relative to the attempt metrics directory.  The
    manifest deliberately records unavailable object-store attributes as null:
    export containers see mounted files, not the source objects' ETags.  The
    cleanup reconciler enriches the retained copy from storage HEAD responses.
    """
    destination = Path(output_root) / "metrics" / stage / str(int(request_id))
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for name, source in sorted(evidence.items()):
        if source is None:
            if name in required:
                raise ValueError(f"Missing required evidence source: {name}")
            continue
        source_path = Path(source)
        if not source_path.is_file():
            if name in required:
                raise ValueError(
                    f"Missing required {stage} evidence {name}: {source_path}"
                )
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe evidence name: {name}")
        copied = destination / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, copied)
        records.append({
            "path": relative.as_posix(),
            "original_key": str(source_path),
            "size": copied.stat().st_size,
            "etag": None,
            "sha256": sha256_file(copied),
        })
    manifest = {
        "schema": METRICS_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "request_id": int(request_id),
        "metadata": {key: value for key, value in metadata.items() if value is not None},
        "files": records,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return manifest


def reconstruction_evidence(processing_root: str | Path) -> dict[str, Path]:
    """Return the known small reports emitted by a reconstruction lineage."""
    root = Path(processing_root)
    candidates = {
        "accuracy/check_accuracy.json": root / "check_accuracy" / "check_accuracy.json",
        "chamfer/object.json": root / "eval_chamfer_object" / "chamfer_metrics.json",
        "chamfer/human.json": root / "eval_chamfer_human" / "chamfer_metrics.json",
        "silhouette/object.json": (
            root / "eval_silhouette_mask_object" / "silhouette_mask_metrics.json"
        ),
        "silhouette/human.json": (
            root / "eval_silhouette_mask_human" / "silhouette_mask_metrics.json"
        ),
        "object_mask/check_object_mask.json": (
            root / "check_object_mask" / "check_object_mask.json"
        ),
        "task_manifests/face_detector_manifest.json": (
            root / "face_detector" / "face_detector_manifest.json"
        ),
    }
    bbox_validation = root / "validate_labeled_bboxes"
    bbox_sha = bbox_validation / "sha256"
    if bbox_sha.is_file():
        candidates["label_validation/sha256"] = bbox_sha
    preview_root = bbox_validation / "prompts"
    if preview_root.is_dir():
        for preview in sorted(preview_root.rglob("*")):
            if preview.is_file():
                candidates[
                    "label_validation/previews/"
                    + preview.relative_to(preview_root).as_posix()
                ] = preview
    for manifest in sorted(root.glob("*/manifest.json")):
        candidates[f"task_manifests/{manifest.parent.name}.json"] = manifest
    return {name: path for name, path in candidates.items() if path.is_file()}
