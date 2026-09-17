"""Create one fully validated MV-HOI export and write its commit last."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

try:
    from .export_metrics import copy_metrics_bundle, reconstruction_evidence
    from .export_qc import (
        ExportQCRejected,
        build_rejection_report,
        evaluate_post_trim_qc,
        write_rejection_report,
    )
    from .export_sequence import export_sequence
    from .interaction_trim import (
        InteractionTrimConfig,
        detect_interaction_trim,
        merged_interval_coverage,
        prepare_trimmed_source,
        write_trim_manifest,
    )
    from .validate_export_payload import validate_export_payload
except ImportError:
    from export_metrics import copy_metrics_bundle, reconstruction_evidence
    from export_qc import (
        ExportQCRejected,
        build_rejection_report,
        evaluate_post_trim_qc,
        write_rejection_report,
    )
    from export_sequence import export_sequence
    from interaction_trim import (
        InteractionTrimConfig,
        detect_interaction_trim,
        merged_interval_coverage,
        prepare_trimmed_source,
        write_trim_manifest,
    )
    from validate_export_payload import validate_export_payload


COMMIT_SCHEMA = "v2d.mv_hoi.export_commit.v3"
SUPPORTED_COMMIT_SCHEMAS = {
    "v2d.mv_hoi.export_commit.v1",
    "v2d.mv_hoi.export_commit.v2",
    COMMIT_SCHEMA,
}
_FFV1_CONTENT_ADDRESS = re.compile(r"\.ffv1\.([0-9a-f]{64})\.mkv$")


def _available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _log_phase(timings: dict[str, float], name: str, started: float) -> None:
    elapsed = time.perf_counter() - started
    timings[name] = round(elapsed, 3)
    print(f"export_phase {name}: {elapsed:.3f}s", flush=True)


def _commit_summary(commit: dict) -> dict:
    return {
        key: commit.get(key)
        for key in (
            "schema",
            "complete",
            "sequence_name",
            "pipeline_version",
            "file_count",
            "total_bytes",
        )
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_payload_sha256(path: Path) -> str:
    """Reuse the digest that atomically content-addressed a new FFV1 sidecar."""
    match = _FFV1_CONTENT_ADDRESS.search(path.name)
    return match.group(1) if match is not None else _sha256(path)


def _payloads(
    root: Path,
    *,
    hash_payloads: bool = True,
    trust_new_content_addresses: bool = False,
) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "commit.json":
            continue
        record = {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
        }
        if hash_payloads:
            record["sha256"] = (
                _new_payload_sha256(path)
                if trust_new_content_addresses
                else _sha256(path)
            )
        records.append(record)
    return records


def verify_export_commit(root: str | Path, *, hash_payloads: bool = True) -> dict:
    root = Path(root)
    commit_path = root / "commit.json"
    if not commit_path.is_file():
        raise ValueError(f"Missing export commit: {commit_path}")
    commit = json.loads(commit_path.read_text())
    if (
        commit.get("schema") not in SUPPORTED_COMMIT_SCHEMAS
        or commit.get("complete") is not True
    ):
        raise ValueError("Unsupported or incomplete export commit")
    expected = {item["path"]: item for item in commit.get("files", [])}
    actual = {
        item["path"]: item
        for item in _payloads(root, hash_payloads=hash_payloads)
    }
    if set(expected) != set(actual):
        raise ValueError("Export commit key set differs from payload")
    for path, item in actual.items():
        if int(expected[path]["size"]) != int(item["size"]):
            raise ValueError(f"Export commit size mismatch: {path}")
        if hash_payloads and expected[path]["sha256"] != item["sha256"]:
            raise ValueError(f"Export commit SHA-256 mismatch: {path}")
    return commit


def _clear_directory_contents(directory: Path) -> None:
    """Clear an incomplete export without removing its mounted root."""
    for child in directory.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def _merged_processing_root(
    reconstruction_root: Path,
    preprocess_root: Path,
    merged_root: Path,
) -> Path:
    """Expose split request-scoped lineage through the legacy flat layout."""
    if not reconstruction_root.is_dir():
        raise ValueError(
            f"Reconstruction source is not a directory: {reconstruction_root}"
        )
    if not preprocess_root.is_dir():
        raise ValueError(f"Preprocess source is not a directory: {preprocess_root}")
    merged_root.mkdir(parents=True, exist_ok=True)
    for child in reconstruction_root.iterdir():
        (merged_root / child.name).symlink_to(child, target_is_directory=child.is_dir())
    for name in ("mv_preprocess", "face_detector"):
        child = preprocess_root / name
        if not child.is_dir():
            raise ValueError(f"Preprocess source is missing required directory: {child}")
        destination = merged_root / name
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                f"Reconstruction and preprocess sources overlap at {name}"
            )
        destination.symlink_to(child, target_is_directory=True)
    return merged_root


def finalize_export(
    *, source_dir: str | Path, failure_segments_path: str | Path,
    output_dir: str | Path, sequence_name: str, pipeline_version: str,
    authorization_type: str, reconstruction_run_id: int,
    request_id: int | None = None,
    campaign_name: str | None = None,
    qc_review_id: int | None = None,
    export_run_id: int | None = None,
    checkpoint_manifest_path: str | Path | None = None,
    label_sha256: str | None = None,
    configuration_sha256: str | None = None,
    metric_failure_segments_path: str | Path | None = None,
    preprocess_source_dir: str | Path | None = None,
    rebuild_incomplete: bool = False,
    interaction_trim_enabled: bool = True,
    interaction_distance_threshold_m: float = 0.10,
    interaction_pre_contact_padding_seconds: float = 3.0,
    interaction_post_contact_padding_seconds: float = 3.0,
    interaction_window_frames: int = 7,
    interaction_required_under_threshold_frames: int = 5,
    max_failure_annotations: int = 10,
    max_failure_coverage: float = 0.5,
    max_silhouette_failure_coverage: float = 0.5,
) -> dict:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    output = Path(output_dir)
    if output.exists():
        if not output.is_dir() or output.is_symlink():
            raise ValueError(f"Export destination is not a directory: {output}")
        if any(output.iterdir()):
            try:
                return verify_export_commit(output)
            except (ValueError, json.JSONDecodeError):
                if not rebuild_incomplete:
                    raise ValueError(
                        "Destination exists without a valid commit; use --rebuild-incomplete"
                    )
                _clear_directory_contents(output)
    # OSMO provides output_dir as a mounted directory. Preserve that mount root
    # and build within it; only create the directory for local callers.
    output.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="v2d-export-build-") as temporary:
            temporary_root = Path(temporary)
            if preprocess_source_dir is None:
                processing_root = Path(source_dir)
            else:
                processing_root = _merged_processing_root(
                    Path(source_dir), Path(preprocess_source_dir),
                    temporary_root / "merged",
                )
            phase_started = time.perf_counter()
            trim_manifest = detect_interaction_trim(
                processing_root,
                config=InteractionTrimConfig(
                    enabled=interaction_trim_enabled,
                    distance_threshold_m=interaction_distance_threshold_m,
                    pre_contact_padding_seconds=(
                        interaction_pre_contact_padding_seconds
                    ),
                    post_contact_padding_seconds=(
                        interaction_post_contact_padding_seconds
                    ),
                    window_frames=interaction_window_frames,
                    required_under_threshold_frames=(
                        interaction_required_under_threshold_frames
                    ),
                ),
            )
            _log_phase(timings, "detect_interaction_trim", phase_started)
            human_failure_segments = json.loads(
                Path(failure_segments_path).read_text()
            )
            if not isinstance(human_failure_segments, list):
                raise ValueError("failure_segments_path must contain a JSON list")
            metric_source = Path(metric_failure_segments_path) if (
                metric_failure_segments_path is not None
            ) else Path(source_dir) / "check_accuracy" / "failure_segments.json"
            metric_failure_segments = json.loads(metric_source.read_text())
            if not isinstance(metric_failure_segments, list):
                raise ValueError(
                    "metric_failure_segments_path must contain a JSON list"
                )
            post_trim_qc = evaluate_post_trim_qc(
                trim_manifest=trim_manifest,
                human_segments=human_failure_segments,
                metric_segments=metric_failure_segments,
                max_failure_annotations=max_failure_annotations,
                max_failure_coverage=max_failure_coverage,
                max_silhouette_failure_coverage=(
                    max_silhouette_failure_coverage
                ),
            )
            if post_trim_qc["status"] == "REJECTED":
                rejection = build_rejection_report(
                    decision=post_trim_qc,
                    trim_manifest=trim_manifest,
                    human_segments=human_failure_segments,
                    metric_segments=metric_failure_segments,
                    provenance={
                        "sequence_name": sequence_name,
                        "pipeline_version": pipeline_version,
                        "authorization_type": authorization_type,
                        "request_id": int(request_id or reconstruction_run_id),
                        "reconstruction_run_id": int(reconstruction_run_id),
                        "qc_review_id": qc_review_id,
                        "export_run_id": export_run_id,
                        "campaign_name": campaign_name,
                        "configuration_sha256": configuration_sha256,
                        "label_sha256": label_sha256,
                    },
                )
                write_rejection_report(output / "export_rejection.json", rejection)
                raise ExportQCRejected(rejection)
            available_cpus = _available_cpus()
            video_workers = min(4, available_cpus)
            phase_started = time.perf_counter()
            trimmed_source = prepare_trimmed_source(
                processing_root,
                temporary_root / "trimmed",
                trim_manifest,
                defer_frame_archives=True,
                max_video_workers=video_workers,
                video_codec_threads=max(1, available_cpus // video_workers),
            )
            _log_phase(timings, "prepare_trimmed_source", phase_started)
            phase_started = time.perf_counter()
            export_sequence(
                source_dir=str(trimmed_source), output_dir=str(output),
                include_anonymized_rgb=True,
                rgb_storage="ffv1_sidecar", depth_storage="ffv1_sidecar",
                max_camera_workers=min(4, available_cpus),
                source_start_frame=int(
                    trim_manifest["export_source_start_frame"]
                ),
                source_end_frame=int(trim_manifest["export_source_end_frame"]),
            )
            _log_phase(timings, "export_sequence", phase_started)
        combined_source_segments = human_failure_segments + metric_failure_segments
        trimmed_human_segments = post_trim_qc["trimmed_human_segments"]
        trimmed_metric_segments = post_trim_qc["trimmed_metric_segments"]
        trimmed_failure_segments = trimmed_human_segments + trimmed_metric_segments
        failure_count = len(trimmed_human_segments)
        failure_coverage_frames = merged_interval_coverage(
            trimmed_human_segments
        )
        export_frame_count = int(trim_manifest["export_frame_count"])
        failure_coverage = failure_coverage_frames / export_frame_count
        (output / "failure_segments.json").write_text(
            json.dumps(trimmed_failure_segments, indent=2, sort_keys=True) + "\n"
        )
        combined_source_path = output / ".combined_source_failure_segments.json"
        combined_source_path.write_text(
            json.dumps(combined_source_segments, indent=2, sort_keys=True) + "\n"
        )
        write_trim_manifest(output / "interaction_trim.json", trim_manifest)
        metrics_request_id = int(request_id or reconstruction_run_id)
        evidence = reconstruction_evidence(source_dir)
        preprocess_evidence = reconstruction_evidence(
            preprocess_source_dir or source_dir
        )
        if "task_manifests/face_detector_manifest.json" in preprocess_evidence:
            evidence["task_manifests/face_detector_manifest.json"] = (
                preprocess_evidence["task_manifests/face_detector_manifest.json"]
            )
        evidence.update({
            "failure_segments/human_source_failure_segments.json": failure_segments_path,
            "failure_segments/metric_source_failure_segments.json": metric_source,
            "failure_segments/source_failure_segments.json": combined_source_path,
            "failure_segments/export_failure_segments.json": output / "failure_segments.json",
            "interaction_trim/interaction_trim.json": output / "interaction_trim.json",
        })
        if checkpoint_manifest_path is not None:
            evidence["checkpoint/manifest.json"] = Path(checkpoint_manifest_path)
        copy_metrics_bundle(
            output,
            stage="reconstruction",
            request_id=metrics_request_id,
            evidence=evidence,
            required=frozenset({
                "failure_segments/human_source_failure_segments.json",
                "failure_segments/metric_source_failure_segments.json",
                "failure_segments/source_failure_segments.json",
                "failure_segments/export_failure_segments.json",
                "interaction_trim/interaction_trim.json",
            }),
            metadata={
                "campaign_name": campaign_name,
                "reconstruction_request_id": metrics_request_id,
                "reconstruction_run_id": int(reconstruction_run_id),
                "qc_review_id": qc_review_id,
                "export_run_id": export_run_id,
                "pipeline_version": pipeline_version,
                "authorization_type": authorization_type,
                "label_sha256": label_sha256,
                "configuration_sha256": configuration_sha256,
                "reconstruction_source_uri": str(source_dir),
                "preprocess_source_uri": (
                    str(preprocess_source_dir) if preprocess_source_dir else str(source_dir)
                ),
            },
        )
        combined_source_path.unlink()
        phase_started = time.perf_counter()
        payloads = _payloads(output, trust_new_content_addresses=True)
        _log_phase(timings, "hash_payloads", phase_started)
        known_payload_sha256 = {
            item["path"]: item["sha256"] for item in payloads
        }
        phase_started = time.perf_counter()
        validation = validate_export_payload(
            output, known_payload_sha256=known_payload_sha256,
        )
        _log_phase(timings, "validate_export_payload", phase_started)
        commit = {
            "schema": COMMIT_SCHEMA,
            "complete": True,
            "committed_at": datetime.now(timezone.utc).isoformat(),
            "sequence_name": sequence_name,
            "pipeline_version": pipeline_version,
            "authorization_type": authorization_type,
            "request_id": metrics_request_id,
            "campaign_name": campaign_name,
            "reconstruction_run_id": int(reconstruction_run_id),
            "qc_review_id": qc_review_id,
            "export_run_id": export_run_id,
            "label_sha256": label_sha256,
            "configuration_sha256": configuration_sha256,
            "interaction_trim": {
                "schema": trim_manifest["schema"],
                "decision_sha256": trim_manifest["decision_sha256"],
                "contact_frame_source": trim_manifest["contact_frame_source"],
                "first_contact_frame_source": trim_manifest.get(
                    "first_contact_frame_source",
                    trim_manifest["contact_frame_source"],
                ),
                "last_contact_frame_source": trim_manifest.get(
                    "last_contact_frame_source",
                    trim_manifest["contact_frame_source"],
                ),
                "source_start_frame": trim_manifest[
                    "export_source_start_frame"
                ],
                "source_end_frame": trim_manifest["export_source_end_frame"],
                "export_frame_count": export_frame_count,
                "trimmed_prefix_frames": trim_manifest.get(
                    "trimmed_prefix_frames", 0,
                ),
                "trimmed_suffix_frames": trim_manifest.get(
                    "trimmed_suffix_frames", 0,
                ),
            },
            "trimmed_qc": {
                "gate": "human_qc_and_subject_containment",
                "failure_annotation_count": failure_count,
                "failure_coverage_frames": failure_coverage_frames,
                "failure_coverage": failure_coverage,
                "max_failure_annotations": int(max_failure_annotations),
                "max_failure_coverage": float(max_failure_coverage),
                "max_silhouette_failure_coverage": float(
                    max_silhouette_failure_coverage
                ),
                "gates": post_trim_qc["gates"],
            },
            "accuracy_segments": {
                "source_segment_count": len(metric_failure_segments),
                "export_segment_count": len(trimmed_metric_segments),
            },
            "combined_failure_segments": {
                "source_segment_count": len(combined_source_segments),
                "export_segment_count": len(trimmed_failure_segments),
                "export_coverage_frames": merged_interval_coverage(
                    trimmed_failure_segments
                ),
            },
            "export_validation": validation,
            "file_count": len(payloads),
            "total_bytes": sum(int(item["size"]) for item in payloads),
            "files": payloads,
        }
        (output / "commit.json").write_text(
            json.dumps(commit, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        # The payload bytes were hashed once immediately above. Re-enumerate
        # paths and sizes after writing the commit, but leave full rehashing to
        # independent integrity verification.
        verified = verify_export_commit(output, hash_payloads=False)
        timings["total"] = round(time.perf_counter() - total_started, 3)
        print(
            "export_phase_timings: "
            + json.dumps(timings, sort_keys=True),
            flush=True,
        )
        return verified
    except Exception:
        (output / "commit.json").unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--preprocess-source-dir")
    parser.add_argument("--failure-segments-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--pipeline-version", required=True)
    parser.add_argument("--authorization-type", required=True)
    parser.add_argument("--reconstruction-run-id", type=int, required=True)
    parser.add_argument("--request-id", type=int)
    parser.add_argument("--campaign-name")
    parser.add_argument("--qc-review-id", type=int)
    parser.add_argument("--export-run-id", type=int)
    parser.add_argument("--checkpoint-manifest-path")
    parser.add_argument("--label-sha256")
    parser.add_argument("--configuration-sha256")
    parser.add_argument("--metric-failure-segments-path")
    parser.add_argument("--rebuild-incomplete", action="store_true")
    parser.add_argument(
        "--disable-interaction-trim",
        action="store_false",
        dest="interaction_trim_enabled",
    )
    parser.set_defaults(interaction_trim_enabled=True)
    parser.add_argument(
        "--interaction-distance-threshold-m", type=float, default=0.10,
    )
    parser.add_argument(
        "--interaction-pre-contact-padding-seconds", type=float, default=3.0,
    )
    parser.add_argument(
        "--interaction-post-contact-padding-seconds", type=float, default=3.0,
    )
    parser.add_argument("--interaction-window-frames", type=int, default=7)
    parser.add_argument(
        "--interaction-required-under-threshold-frames",
        type=int,
        default=5,
    )
    parser.add_argument("--max-failure-annotations", type=int, default=10)
    parser.add_argument("--max-failure-coverage", type=float, default=0.5)
    parser.add_argument(
        "--max-silhouette-failure-coverage", type=float, default=0.5,
    )
    args = parser.parse_args()
    commit = finalize_export(**vars(args))
    print(json.dumps(_commit_summary(commit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
