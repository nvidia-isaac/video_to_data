"""Build and commit one verified commercial-FoundationPose export.

``commit.json`` is generated after every payload file and covers every byte in
the export.  Consumers must treat a destination without a valid commit as
incomplete.  Object-store publication can therefore upload payload objects
first and this commit object last.
"""

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

import numpy as np

try:
    from .export_metrics import copy_metrics_bundle
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
        prepare_trimmed_source,
        validate_trim_manifest,
        write_trim_manifest,
    )
    from .validate_export_payload import validate_export_payload
except ImportError:  # Direct test/script execution from the lib directory.
    from export_metrics import copy_metrics_bundle
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
        prepare_trimmed_source,
        validate_trim_manifest,
        write_trim_manifest,
    )
    from validate_export_payload import validate_export_payload


COMMIT_SCHEMA = "v2d.mv_hoi.revalidation_export_commit.v3"
SUPPORTED_COMMIT_SCHEMAS = {
    "v2d.mv_hoi.revalidation_export_commit.v1",
    "v2d.mv_hoi.revalidation_export_commit.v2",
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
            "request_id",
            "campaign_name",
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


def _files(
    root: Path,
    *,
    hash_payloads: bool = True,
    trust_new_content_addresses: bool = False,
) -> list[dict]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "commit.json":
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
        raise ValueError("Unsupported or incomplete revalidation export commit")
    expected = {item["path"]: item for item in commit.get("files", [])}
    actual = {
        item["path"]: item
        for item in _files(root, hash_payloads=hash_payloads)
    }
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(f"Export payload mismatch: missing={missing}, unexpected={unexpected}")
    for relative, record in actual.items():
        if int(expected[relative]["size"]) != int(record["size"]):
            raise ValueError(f"Export size mismatch: {relative}")
        if hash_payloads and expected[relative]["sha256"] != record["sha256"]:
            raise ValueError(f"Export SHA-256 mismatch: {relative}")
    return commit


def _symlink_contents(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"Missing source directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        os.symlink(item.resolve(), destination / item.name, target_is_directory=item.is_dir())


def _replace_link(root: Path, name: str, source: str | Path) -> None:
    destination = root / name
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)
    os.symlink(Path(source).resolve(), destination, target_is_directory=True)


def _copy_evidence(output: Path, evidence: dict[str, str | Path | None]) -> None:
    evidence_dir = output / "revalidation"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, source in evidence.items():
        if source is None:
            continue
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"Missing revalidation evidence {name}: {path}")
        shutil.copy2(path, evidence_dir / name)


def _clear_directory_contents(directory: Path) -> None:
    """Clear an incomplete export without removing its mounted root."""
    for child in directory.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


def _verify_object_pose_artifact_pair(
    output: Path,
    comparison: dict,
    commercial_pose_path: Path,
    trim_manifest: dict,
    known_payload_sha256: dict[str, str] | None = None,
) -> dict:
    """Bind commercial poses to the exact aligned mesh and symmetry metadata."""
    if comparison.get("comparison_pose_frame") != "aligned":
        raise ValueError("Pose comparison did not use the aligned object frame")
    start, end, output_count = validate_trim_manifest(trim_manifest)
    expected_full_pose_sha256 = comparison.get("commercial_pose_sha256")
    if (
        not isinstance(expected_full_pose_sha256, str)
        or len(expected_full_pose_sha256) != 64
    ):
        raise ValueError("Pose comparison is missing commercial_pose_sha256")
    actual_full_pose_sha256 = _sha256(commercial_pose_path)
    if actual_full_pose_sha256 != expected_full_pose_sha256:
        raise ValueError(
            "Full commercial pose artifact no longer matches pose comparison"
        )

    paths = {
        "mesh_sha256": output / "object_mesh" / "output_aligned.glb",
        "symmetry_sha256": output / "object_mesh" / "output_symmetry.json",
    }
    verified = {
        "pose_frame": "aligned",
        "commercial_source_pose_sha256": actual_full_pose_sha256,
        "source_start_frame": start,
        "source_end_frame": end,
        "export_frame_count": output_count,
    }
    for field, path in paths.items():
        if not path.is_file():
            raise ValueError(f"Missing aligned object-pose artifact: {path}")
        expected = comparison.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Pose comparison is missing {field}")
        relative = path.relative_to(output).as_posix()
        actual = (
            known_payload_sha256[relative]
            if known_payload_sha256 is not None
            else _sha256(path)
        )
        if actual != expected:
            raise ValueError(
                f"Exported object-pose artifact hash mismatch for {path.name}: "
                f"comparison={expected}, export={actual}"
            )
        verified[field] = actual

    source_poses = np.load(commercial_pose_path, allow_pickle=False)
    exported_pose_path = output / "poses.npy"
    exported_poses = np.load(exported_pose_path, allow_pickle=False)
    if len(source_poses) != int(trim_manifest["source_frame_count"]):
        raise ValueError("Full commercial pose frame count differs from trim source")
    expected_exported_poses = source_poses[start:end]
    if (
        exported_poses.shape != expected_exported_poses.shape
        or not np.array_equal(exported_poses, expected_exported_poses)
    ):
        raise ValueError("Exported object poses are not the exact retained source slice")
    relative = exported_pose_path.relative_to(output).as_posix()
    exported_pose_sha256 = (
        known_payload_sha256[relative]
        if known_payload_sha256 is not None
        else _sha256(exported_pose_path)
    )
    verified["exported_trimmed_pose_sha256"] = exported_pose_sha256
    return verified


def finalize_revalidation_export(
    *,
    source_dir: str | Path,
    commercial_foundation_pose_dir: str | Path,
    anonymized_rgb_dir: str | Path,
    overlay_dir: str | Path,
    pose_comparison_path: str | Path,
    check_accuracy_path: str | Path,
    object_silhouette_path: str | Path,
    failure_segments_path: str | Path,
    output_dir: str | Path,
    request_id: int,
    campaign_name: str,
    sequence_name: str,
    pipeline_version: str,
    source_manifest_sha256: str,
    configuration_sha256: str,
    reconstruction_run_id: int | None = None,
    export_run_id: int | None = None,
    checkpoint_manifest_path: str | Path | None = None,
    object_chamfer_path: str | Path | None = None,
    human_silhouette_path: str | Path | None = None,
    human_chamfer_path: str | Path | None = None,
    metric_failure_segments_path: str | Path | None = None,
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
                existing_commit = verify_export_commit(output)
                existing_paths = {
                    item["path"] for item in existing_commit.get("files", [])
                }
                required_qc_paths = {
                    "failure_segments.json",
                    (
                        f"metrics/revalidation/{request_id}/failure_segments/"
                        "source_failure_segments.json"
                    ),
                    (
                        f"metrics/revalidation/{request_id}/failure_segments/"
                        "export_failure_segments.json"
                    ),
                }
                missing_qc_paths = sorted(required_qc_paths - existing_paths)
                if missing_qc_paths:
                    raise ValueError(
                        "Existing revalidation export predates human-QC "
                        f"preservation: missing={missing_qc_paths}"
                    )
                return existing_commit
            except (ValueError, json.JSONDecodeError):
                if not rebuild_incomplete:
                    raise ValueError(
                        "Destination exists without a valid commit; use --rebuild-incomplete"
                    )
                _clear_directory_contents(output)
    comparison = json.loads(Path(pose_comparison_path).read_text())
    accuracy = json.loads(Path(check_accuracy_path).read_text())
    silhouette = json.loads(Path(object_silhouette_path).read_text())
    failure_segments_source = Path(failure_segments_path)
    human_failure_segments = json.loads(failure_segments_source.read_text())
    if not isinstance(human_failure_segments, list):
        raise ValueError("failure_segments_path must contain a JSON list")
    metric_failure_segments_source = (
        Path(metric_failure_segments_path)
        if metric_failure_segments_path is not None
        else Path(check_accuracy_path).parent / "failure_segments.json"
    )
    metric_failure_segments = json.loads(
        metric_failure_segments_source.read_text()
    )
    if not isinstance(metric_failure_segments, list):
        raise ValueError("metric_failure_segments_path must contain a JSON list")
    combined_source_segments = human_failure_segments + metric_failure_segments
    if comparison.get("status") != "PASS":
        raise ValueError("Commercial-versus-legacy pose comparison did not pass")
    if accuracy.get("status") != "PASS":
        raise ValueError("Current accuracy gate did not pass")
    silhouette_status = silhouette.get("quality_gate", {}).get("status")

    # OSMO provides output_dir as a mounted directory. Preserve that mount root
    # and build within it; only create the directory for local callers.
    output.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="v2d-revalidation-") as temporary:
            temporary_root = Path(temporary)
            merged = temporary_root / "source"
            _symlink_contents(Path(source_dir), merged)
            _replace_link(merged, "foundation_pose", commercial_foundation_pose_dir)
            _replace_link(merged, "face_detector", anonymized_rgb_dir)
            _replace_link(merged, "render_hoi_overlay", overlay_dir)
            phase_started = time.perf_counter()
            trim_manifest = detect_interaction_trim(
                merged,
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
                        "authorization_type": "REVALIDATION",
                        "request_id": int(request_id),
                        "reconstruction_run_id": reconstruction_run_id,
                        "export_run_id": export_run_id,
                        "campaign_name": campaign_name,
                        "configuration_sha256": configuration_sha256,
                        "source_manifest_sha256": source_manifest_sha256,
                    },
                )
                write_rejection_report(output / "export_rejection.json", rejection)
                raise ExportQCRejected(rejection)
            available_cpus = _available_cpus()
            video_workers = min(4, available_cpus)
            phase_started = time.perf_counter()
            trimmed_source = prepare_trimmed_source(
                merged,
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
        trimmed_human_segments = post_trim_qc["trimmed_human_segments"]
        trimmed_metric_segments = post_trim_qc["trimmed_metric_segments"]
        trimmed_failure_segments = trimmed_human_segments + trimmed_metric_segments
        (output / "failure_segments.json").write_text(
            json.dumps(trimmed_failure_segments, indent=2, sort_keys=True) + "\n"
        )
        write_trim_manifest(output / "interaction_trim.json", trim_manifest)
        combined_source_path = output / ".combined_source_failure_segments.json"
        combined_source_path.write_text(
            json.dumps(combined_source_segments, indent=2, sort_keys=True) + "\n"
        )
        _copy_evidence(output, {
            "foundation_pose_comparison.json": pose_comparison_path,
            "check_accuracy.json": check_accuracy_path,
            "object_silhouette.json": object_silhouette_path,
            "object_chamfer.json": object_chamfer_path,
            "checkpoint_manifest.json": checkpoint_manifest_path,
        })
        copy_metrics_bundle(
            output,
            stage="revalidation",
            request_id=request_id,
            evidence={
                "foundation_pose/foundation_pose_comparison.json": pose_comparison_path,
                "accuracy/check_accuracy.json": check_accuracy_path,
                "silhouette/object.json": object_silhouette_path,
                "silhouette/human.json": human_silhouette_path,
                "chamfer/object.json": object_chamfer_path,
                "chamfer/human.json": human_chamfer_path,
                "checkpoint/manifest.json": checkpoint_manifest_path,
                "interaction_trim/interaction_trim.json": output / "interaction_trim.json",
                "failure_segments/human_source_failure_segments.json": failure_segments_source,
                "failure_segments/metric_source_failure_segments.json": (
                    metric_failure_segments_source
                ),
                "failure_segments/source_failure_segments.json": combined_source_path,
                "failure_segments/export_failure_segments.json": (
                    output / "failure_segments.json"
                ),
            },
            required=frozenset({
                "foundation_pose/foundation_pose_comparison.json",
                "accuracy/check_accuracy.json",
                "silhouette/object.json",
                "interaction_trim/interaction_trim.json",
                "failure_segments/source_failure_segments.json",
                "failure_segments/human_source_failure_segments.json",
                "failure_segments/metric_source_failure_segments.json",
                "failure_segments/export_failure_segments.json",
            }),
            metadata={
                "campaign_name": campaign_name,
                "revalidation_request_id": int(request_id),
                "reconstruction_run_id": reconstruction_run_id,
                "export_run_id": export_run_id,
                "pipeline_version": pipeline_version,
                "source_uri": str(source_dir),
                "source_manifest_sha256": source_manifest_sha256,
                "configuration_sha256": configuration_sha256,
                "authorization_type": "REVALIDATION",
            },
        )
        combined_source_path.unlink()
        phase_started = time.perf_counter()
        payloads = _files(output, trust_new_content_addresses=True)
        _log_phase(timings, "hash_payloads", phase_started)
        known_payload_sha256 = {
            item["path"]: item["sha256"] for item in payloads
        }
        phase_started = time.perf_counter()
        object_pose_artifacts = _verify_object_pose_artifact_pair(
            output,
            comparison,
            Path(commercial_foundation_pose_dir) / "poses.npy",
            trim_manifest,
            known_payload_sha256,
        )
        validation = validate_export_payload(
            output, known_payload_sha256=known_payload_sha256,
        )
        _log_phase(timings, "validate_export_payload", phase_started)
        commit = {
            "schema": COMMIT_SCHEMA,
            "complete": True,
            "committed_at": datetime.now(timezone.utc).isoformat(),
            "request_id": int(request_id),
            "campaign_name": campaign_name,
            "sequence_name": sequence_name,
            "pipeline_version": pipeline_version,
            "authorization_type": "REVALIDATION",
            "reconstruction_run_id": reconstruction_run_id,
            "export_run_id": export_run_id,
            "source_manifest_sha256": source_manifest_sha256,
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
                "export_frame_count": trim_manifest["export_frame_count"],
                "trimmed_prefix_frames": trim_manifest.get(
                    "trimmed_prefix_frames", 0,
                ),
                "trimmed_suffix_frames": trim_manifest.get(
                    "trimmed_suffix_frames", 0,
                ),
            },
            "revalidation_evidence_index_space": "source_frames",
            "human_qc": {
                "source_index_space": "source_frames",
                "export_index_space": "export_frames",
                "source_failure_segment_count": len(human_failure_segments),
                "export_failure_segment_count": len(trimmed_human_segments),
                "post_trim_gates": post_trim_qc["gates"],
                "max_failure_annotations": int(max_failure_annotations),
                "max_failure_coverage": float(max_failure_coverage),
                "max_silhouette_failure_coverage": float(
                    max_silhouette_failure_coverage
                ),
            },
            "accuracy_segments": {
                "source_segment_count": len(metric_failure_segments),
                "export_segment_count": len(trimmed_metric_segments),
            },
            "combined_failure_segments": {
                "source_segment_count": len(combined_source_segments),
                "export_segment_count": len(trimmed_failure_segments),
            },
            "pose_comparison_status": comparison["status"],
            "pose_comparison_summary": {
                key: comparison.get(key) for key in (
                    "valid_coverage", "newly_invalid_fraction", "translation_m",
                    "rotation_deg", "normalized_adds", "divergent_frames",
                    "divergent_frame_fraction", "divergence_segments",
                )
            },
            "object_pose_artifacts": object_pose_artifacts,
            "accuracy_status": accuracy["status"],
            "silhouette_status": silhouette_status,
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
        # Never leave a local result looking committed after a failed build.
        (output / "commit.json").unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--commercial-foundation-pose-dir", required=True)
    parser.add_argument("--anonymized-rgb-dir", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--pose-comparison-path", required=True)
    parser.add_argument("--check-accuracy-path", required=True)
    parser.add_argument("--object-silhouette-path", required=True)
    parser.add_argument("--failure-segments-path", required=True)
    parser.add_argument("--object-chamfer-path")
    parser.add_argument("--human-silhouette-path")
    parser.add_argument("--human-chamfer-path")
    parser.add_argument("--metric-failure-segments-path")
    parser.add_argument("--checkpoint-manifest-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--request-id", type=int, required=True)
    parser.add_argument("--campaign-name", required=True)
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--pipeline-version", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--reconstruction-run-id", type=int)
    parser.add_argument("--export-run-id", type=int)
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
    commit = finalize_revalidation_export(**vars(args))
    print(json.dumps(_commit_summary(commit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
