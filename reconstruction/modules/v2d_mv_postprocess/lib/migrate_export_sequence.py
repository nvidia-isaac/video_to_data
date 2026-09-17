"""Stage one sequence using production JPEG/gzip or opt-in FFV1 pairs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from v2d.common.hdf5_transcode import (
    has_hdf5_filters,
    is_jpeg_hdf5,
    transcode_h5_lossless,
    transcode_rgb_h5_to_jpeg_h5,
)
from v2d.common.ffv1_sidecar import transcode_h5_to_ffv1_sidecar


RGB_DIRS = {"images", "images_anonymized"}
REQUIRED_TARGET_DIRS = {"images", "depth"}
TARGET_DIRS = ("images", "images_anonymized", "depth")


def _atomic_transform(source: Path, destination: Path, kind: str) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        if kind in RGB_DIRS:
            if is_jpeg_hdf5(source):
                import shutil

                shutil.copy2(source, temp_path)
                stats = {
                    "kind": "rgb_jpeg_copy",
                    "frames_already_encoded": True,
                    "source_bytes": source.stat().st_size,
                    "output_bytes": temp_path.stat().st_size,
                }
            else:
                stats = transcode_rgb_h5_to_jpeg_h5(source, temp_path)
        elif kind == "depth":
            if has_hdf5_filters(
                source,
                compression="gzip",
                compression_opts=6,
                shuffle=True,
            ):
                import shutil

                shutil.copy2(source, temp_path)
                stats = {
                    "kind": "depth_copy",
                    "filters_already_compliant": True,
                    "source_bytes": source.stat().st_size,
                    "output_bytes": temp_path.stat().st_size,
                }
            else:
                stats = transcode_h5_lossless(source, temp_path)
        else:
            raise ValueError(f"Unsupported migration kind: {kind}")
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return stats


def migrate_export_sequence(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    storage_format: str = "jpeg_depth",
) -> dict:
    if storage_format not in {"jpeg_depth", "ffv1"}:
        raise ValueError(f"Unsupported storage format: {storage_format}")
    started = time.perf_counter()
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    records: list[dict] = []
    errors: list[dict] = []

    for kind in TARGET_DIRS:
        source_kind_dir = source_dir / kind
        if not source_kind_dir.is_dir():
            if kind in REQUIRED_TARGET_DIRS:
                errors.append(
                    {"kind": kind, "reason": f"missing directory: {source_kind_dir}"}
                )
            continue
        files = sorted(source_kind_dir.glob("*.h5"))
        if not files:
            if kind in REQUIRED_TARGET_DIRS:
                errors.append({"kind": kind, "reason": "no HDF5 files found"})
            continue
        for source in files:
            destination = output_dir / kind / source.name
            try:
                if storage_format == "ffv1":
                    stats = transcode_h5_to_ffv1_sidecar(
                        source,
                        destination,
                        kind="rgb" if kind in RGB_DIRS else "depth",
                    )
                else:
                    stats = _atomic_transform(source, destination, kind)
                stats.update(
                    kind_dir=kind,
                    filename=source.name,
                    source_bytes=source.stat().st_size,
                )
                if storage_format == "ffv1":
                    stats["metadata_filename"] = destination.name
                else:
                    stats["output_bytes"] = destination.stat().st_size
                stats["size_ratio"] = stats["output_bytes"] / max(1, stats["source_bytes"])
                records.append(stats)
            except Exception as exc:
                errors.append({"kind": kind, "filename": source.name, "reason": str(exc)})

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "storage_format": storage_format,
        "status": "complete" if not errors else "failed",
        "files": records,
        "errors": errors,
        "input_bytes": sum(int(item["source_bytes"]) for item in records),
        "artifact_output_bytes": sum(int(item["output_bytes"]) for item in records),
        "transform_seconds": time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "migration_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    if errors:
        raise RuntimeError("Sequence migration failed: " + json.dumps(errors))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--storage_format",
        choices=("jpeg_depth", "ffv1"),
        default="jpeg_depth",
    )
    args = parser.parse_args()
    manifest = migrate_export_sequence(
        args.source_dir,
        args.output_dir,
        storage_format=args.storage_format,
    )
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
