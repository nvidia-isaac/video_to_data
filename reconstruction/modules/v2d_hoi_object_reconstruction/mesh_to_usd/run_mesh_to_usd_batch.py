#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare and drop-test every catalog mesh with matching HOI support data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR / "runtime"))

from foundation_pose_support import (  # noqa: E402
    DEFAULT_FOUNDATION_POSE_CONFIG,
    DEFAULT_FOUNDATION_POSE_WEIGHTS_DIR,
    foundation_pose_support_inputs,
    resolve_foundation_pose_weights_dir,
)
from docker_runtime import IMAGE_NAME, isaac_sim_eula_accepted  # noqa: E402
from recorded_support import (  # noqa: E402
    DEFAULT_INITIAL_SEARCH_FRAMES,
    DEFAULT_STABLE_WINDOW_FRAMES,
    EXACT_MESH_SUPPORT_CALIBRATION,
    FOUNDATION_POSE_SUPPORT_CALIBRATION,
    RecordedSupportPose,
    file_sha256,
    recording_to_support_pose,
)
from run_drop_test import run_drop_test  # noqa: E402
from run_mesh_to_usd_workflow import run_mesh_to_usd_workflow  # noqa: E402
from validation_docker_runtime import VALIDATOR_IMAGE_NAME  # noqa: E402
from workflow_io import utc_now, write_json  # noqa: E402


SUPPORTED_METHODS = ("einstar", "bundlesdf", "sam3d")
DEFAULT_MESH_FILENAME = "output_aligned.glb"
BATCH_REPORT_NAME = "mesh_to_usd_batch_report.json"
JOB_REPORT_NAME = "batch_job_report.json"
BATCH_RESUME_SCHEMA_VERSION = 2
BATCH_WORKFLOW_VERSION = "mesh-to-usd-batch-v3"
_ARTIFACT_PATH_FIELDS = (
    "output_usd",
    "visual_asset",
    "generation_report",
    "validation_report",
    "recorded_support_pose",
    "foundation_pose_support_report",
    "foundation_pose_support_poses",
    "foundation_pose_tracking_metadata",
    "drop_test_report",
)
_ARTIFACT_TREE_PATH_FIELDS = ("mesh_to_usd_dir",)


@dataclass(frozen=True)
class BatchOptions:
    """Options shared by support selection, generation, and drop testing."""

    stable_window_frames: int
    initial_search_frames: int
    max_up_deviation_degrees: float
    max_rotation_deviation_degrees: float
    max_translation_deviation_m: float
    foundation_pose_weights_dir: str | None
    foundation_pose_config_path: str | None
    foundation_pose_debug: int
    validate: bool
    run_drop_tests: bool
    video: bool
    fail_if_not_standing: bool
    image: str
    validator_image: str
    accept_eula: bool
    cache_dir: str | None
    dev: bool
    gpu_device: str | None


def _path_content_sha256(
    path: str | Path,
    cache: dict[Path, str],
) -> str:
    """Hash one file or a directory tree, including relative file names."""

    source = Path(path).expanduser().resolve()
    cached = cache.get(source)
    if cached is not None:
        return cached
    if source.is_file():
        value = file_sha256(source)
        cache[source] = value
        return value
    if not source.is_dir():
        raise ValueError(f"cannot fingerprint missing input: {source}")

    digest = hashlib.sha256()
    for child in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = child.relative_to(source).as_posix().encode("utf-8")
        if child.is_dir():
            digest.update(b"D\0")
            digest.update(relative)
            digest.update(b"\0")
        elif child.is_file():
            digest.update(b"F\0")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(bytes.fromhex(_path_content_sha256(child, cache)))
        else:
            raise ValueError(f"unsupported input while fingerprinting: {child}")
    value = digest.hexdigest()
    cache[source] = value
    return value


def _resume_options(options: BatchOptions) -> dict:
    """Return only options that can affect generated or validated results."""

    return {
        "stable_window_frames": options.stable_window_frames,
        "initial_search_frames": options.initial_search_frames,
        "max_up_deviation_degrees": options.max_up_deviation_degrees,
        "max_rotation_deviation_degrees": (
            options.max_rotation_deviation_degrees
        ),
        "max_translation_deviation_m": options.max_translation_deviation_m,
        "foundation_pose_debug": options.foundation_pose_debug,
        "validate": options.validate,
        "run_drop_tests": options.run_drop_tests,
        "video": options.video,
        "fail_if_not_standing": options.fail_if_not_standing,
        "image": options.image,
        "validator_image": (
            options.validator_image if options.validate else None
        ),
        "dev": options.dev,
        "gpu_device": options.gpu_device,
    }


def _resume_signature(
    *,
    mesh: Path,
    sequence: Path | None,
    support_pose_calibration: str | None,
    options: BatchOptions,
    digest_cache: dict[Path, str],
) -> dict:
    """Build a content- and configuration-aware signature for one batch job."""

    target_symmetry = mesh.with_name("output_symmetry.json")
    inputs = {
        "mesh": {
            "path": str(mesh),
            "sha256": _path_content_sha256(mesh, digest_cache),
        },
        "target_symmetry": {
            "path": str(target_symmetry),
            "sha256": _path_content_sha256(target_symmetry, digest_cache),
        },
        "sequence": str(sequence) if sequence is not None else None,
        "support_pose_calibration": support_pose_calibration,
        "support_inputs": {},
    }
    support_inputs = inputs["support_inputs"]
    if sequence is not None:
        if support_pose_calibration == EXACT_MESH_SUPPORT_CALIBRATION:
            relative_paths = (
                "object_mesh/output_aligned.glb",
                "poses.npy",
                "ground_plane.json",
            )
        elif support_pose_calibration == FOUNDATION_POSE_SUPPORT_CALIBRATION:
            relative_paths = (
                "edex",
                "images",
                "depth",
                "object_masks",
                "ground_plane.json",
            )
        else:
            raise ValueError(
                "selected support sequence lacks a supported calibration mode"
            )
        for relative in relative_paths:
            source = sequence / relative
            support_inputs[relative] = {
                "path": str(source),
                "sha256": _path_content_sha256(source, digest_cache),
            }

    foundation_pose = None
    if support_pose_calibration == FOUNDATION_POSE_SUPPORT_CALIBRATION:
        weights = resolve_foundation_pose_weights_dir(
            options.foundation_pose_weights_dir
        )
        config = Path(
            options.foundation_pose_config_path or DEFAULT_FOUNDATION_POSE_CONFIG
        ).expanduser().resolve()
        foundation_pose = {
            "weights": {
                "path": str(weights),
                "sha256": _path_content_sha256(weights, digest_cache),
            },
            "config": {
                "path": str(config),
                "sha256": _path_content_sha256(config, digest_cache),
            },
        }

    payload = {
        "schema_version": BATCH_RESUME_SCHEMA_VERSION,
        "workflow_version": BATCH_WORKFLOW_VERSION,
        "inputs": inputs,
        "foundation_pose": foundation_pose,
        "options": _resume_options(options),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _capture_artifact_fingerprints(job: dict) -> dict:
    """Record hashes for reported artifacts and complete output packages."""

    fingerprints = {}
    for field in _ARTIFACT_PATH_FIELDS:
        value = job.get(field)
        if value is None:
            continue
        path = Path(value)
        if not path.is_file():
            raise RuntimeError(f"batch output is missing {field}: {path}")
        fingerprints[field] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
    for field in _ARTIFACT_TREE_PATH_FIELDS:
        value = job.get(field)
        if value is None:
            continue
        path = Path(value)
        if not path.is_dir():
            raise RuntimeError(f"batch output is missing {field}: {path}")
        fingerprints[field] = {
            "path": str(path),
            "sha256": _path_content_sha256(path, {}),
        }
    videos = []
    for value in job.get("video_files", []):
        path = Path(value)
        if not path.is_file():
            raise RuntimeError(f"batch output is missing video: {path}")
        videos.append({"path": str(path), "sha256": file_sha256(path)})
    fingerprints["video_files"] = videos
    return fingerprints


def _artifacts_match(result: dict) -> bool:
    fingerprints = result.get("artifact_fingerprints")
    if not isinstance(fingerprints, dict):
        return False
    for field in _ARTIFACT_PATH_FIELDS:
        value = result.get(field)
        if value is None:
            continue
        expected = fingerprints.get(field)
        if not isinstance(expected, dict) or expected.get("path") != value:
            return False
        path = Path(value)
        if not path.is_file() or file_sha256(path) != expected.get("sha256"):
            return False

    for field in _ARTIFACT_TREE_PATH_FIELDS:
        value = result.get(field)
        if value is None:
            continue
        expected = fingerprints.get(field)
        if not isinstance(expected, dict) or expected.get("path") != value:
            return False
        path = Path(value)
        if (
            not path.is_dir()
            or _path_content_sha256(path, {}) != expected.get("sha256")
        ):
            return False

    expected_videos = fingerprints.get("video_files")
    videos = result.get("video_files", [])
    if not isinstance(expected_videos, list) or len(expected_videos) != len(videos):
        return False
    for path_value, expected in zip(videos, expected_videos, strict=True):
        if not isinstance(expected, dict) or expected.get("path") != path_value:
            return False
        path = Path(path_value)
        if not path.is_file() or file_sha256(path) != expected.get("sha256"):
            return False
    return True


def _sequence_object_id(sequence: Path) -> str:
    metadata_path = sequence / "hoi_metadata.yaml"
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid HOI metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"HOI metadata must be an object: {metadata_path}")
    object_data = metadata.get("object")
    object_id = object_data.get("id") if isinstance(object_data, dict) else None
    if not isinstance(object_id, str) or not object_id.strip():
        raise ValueError(f"HOI metadata lacks object.id: {metadata_path}")
    return object_id.strip()


def discover_sequences(sequence_root: str | Path) -> tuple[dict[str, list[Path]], list[dict]]:
    """Return candidate sequences grouped by exact metadata object ID."""

    root = Path(sequence_root).expanduser().resolve()
    grouped: dict[str, list[Path]] = defaultdict(list)
    rejected: list[dict] = []
    for metadata_path in sorted(root.glob("*/hoi_metadata.yaml")):
        sequence = metadata_path.parent
        try:
            object_id = _sequence_object_id(sequence)
        except ValueError as error:
            rejected.append(
                {
                    "sequence": str(sequence),
                    "reason": "invalid-metadata",
                    "error": str(error),
                }
            )
            continue
        grouped[object_id].append(sequence)
    for sequences in grouped.values():
        sequences.sort(reverse=True)
    return dict(grouped), rejected


def discover_meshes(
    mesh_root: str | Path,
    methods: tuple[str, ...] = SUPPORTED_METHODS,
    mesh_filename: str = DEFAULT_MESH_FILENAME,
) -> tuple[dict[str, dict[str, Path]], list[dict]]:
    """Return selected catalog meshes grouped by object ID and method."""

    mesh_name = Path(mesh_filename)
    if mesh_name.name != mesh_filename or mesh_name.suffix.lower() != ".glb":
        raise ValueError("mesh_filename must be a GLB basename")

    root = Path(mesh_root).expanduser().resolve()
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    rejected: list[dict] = []
    for object_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for method in methods:
            method_dir = object_dir / method
            mesh = method_dir / mesh_filename
            symmetry = method_dir / "output_symmetry.json"
            if not mesh.is_file():
                continue
            if not symmetry.is_file():
                rejected.append(
                    {
                        "object_id": object_dir.name,
                        "method": method,
                        "mesh": str(mesh),
                        "reason": "missing-output_symmetry.json",
                    }
                )
                continue
            grouped[object_dir.name][method] = mesh
    return dict(grouped), rejected


def select_support_sequence_for_mesh(
    sequences: list[Path],
    mesh: Path,
    *,
    options: BatchOptions,
) -> tuple[Path | None, RecordedSupportPose | None, str | None, list[dict]]:
    """Select support inputs per target mesh without cross-mesh pose transfer."""

    rejected: list[dict] = []
    target_hash = file_sha256(mesh)
    for sequence in sequences:
        recording_mesh = sequence / "object_mesh" / "output_aligned.glb"
        exact_mesh = (
            recording_mesh.is_file()
            and file_sha256(recording_mesh) == target_hash
        )
        if exact_mesh and (sequence / "poses.npy").is_file():
            try:
                pose = recording_to_support_pose(
                    sequence,
                    stable_window_frames=options.stable_window_frames,
                    initial_search_frames=options.initial_search_frames,
                    max_up_deviation_degrees=(
                        options.max_up_deviation_degrees
                    ),
                    max_rotation_deviation_degrees=(
                        options.max_rotation_deviation_degrees
                    ),
                    max_translation_deviation_m=(
                        options.max_translation_deviation_m
                    ),
                )
            except (OSError, ValueError) as error:
                rejected.append(
                    {
                        "sequence": str(sequence),
                        "support_pose_calibration": (
                            EXACT_MESH_SUPPORT_CALIBRATION
                        ),
                        "error": str(error),
                    }
                )
                continue
            return (
                sequence,
                pose,
                EXACT_MESH_SUPPORT_CALIBRATION,
                rejected,
            )

        try:
            foundation_pose_support_inputs(
                sequence,
                mesh,
                options.foundation_pose_weights_dir,
                config_path=options.foundation_pose_config_path,
            )
        except (OSError, ValueError) as error:
            rejected.append(
                {
                    "sequence": str(sequence),
                    "support_pose_calibration": (
                        FOUNDATION_POSE_SUPPORT_CALIBRATION
                    ),
                    "error": str(error),
                }
            )
            continue
        return (
            sequence,
            None,
            FOUNDATION_POSE_SUPPORT_CALIBRATION,
            rejected,
        )
    return None, None, None, rejected


def _resume_result(
    job_report_path: Path,
    *,
    expected_mesh: Path,
    expected_sequence: Path | None,
    expected_resume_signature: dict,
    require_validation: bool,
    require_drop_test: bool,
    require_video: bool,
    expected_support_pose_calibration: str | None,
) -> dict | None:
    if not job_report_path.is_file():
        return None
    try:
        result = json.loads(job_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if result.get("status") not in {"passed", "generated-without-drop-test"}:
        return None
    if result.get("mesh") != str(expected_mesh):
        return None
    expected_sequence_path = (
        str(expected_sequence) if expected_sequence is not None else None
    )
    if result.get("sequence") != expected_sequence_path:
        return None
    signature = result.get("resume_signature")
    if (
        not isinstance(signature, dict)
        or signature.get("schema_version") != BATCH_RESUME_SCHEMA_VERSION
        or signature.get("sha256") != expected_resume_signature.get("sha256")
    ):
        return None
    if (
        result.get("support_pose_calibration")
        != expected_support_pose_calibration
    ):
        return None
    if expected_sequence is not None:
        support_provenance = result.get("support_provenance")
        if (
            not isinstance(support_provenance, dict)
            or support_provenance.get("sequence")
            != expected_sequence_path
            or not isinstance(
                support_provenance.get("camera_provenance"),
                dict,
            )
        ):
            return None
    if (
        not result.get("output_usd")
        or not result.get("visual_asset")
        or not result.get("generation_report")
        or not result.get("mesh_to_usd_dir")
    ):
        return None
    if require_validation and not result.get("validation_report"):
        return None
    if require_drop_test and result.get("status") != "passed":
        return None
    if require_drop_test and (
        not result.get("recorded_support_pose")
        or not result.get("drop_test_report")
    ):
        return None
    videos = [Path(path) for path in result.get("video_files", [])]
    if (
        require_video
        and result.get("status") == "passed"
        and (not videos or not all(path.is_file() for path in videos))
    ):
        return None
    if not _artifacts_match(result):
        return None
    result["resumed"] = True
    return result


def _planned_job_status(
    *,
    dry_run: bool,
    run_drop_tests: bool,
    drop_test_available: bool,
) -> str:
    if not dry_run:
        return "running"
    if not run_drop_tests or drop_test_available:
        return "planned"
    return "planned-generation-only"


def _support_selection_report(
    *,
    mesh: Path,
    sequence: Path | None,
    support_pose: RecordedSupportPose | None,
    calibration: str | None,
    rejections: list[dict],
) -> dict:
    return {
        "target_mesh": str(mesh),
        "support_pose_calibration": calibration,
        "selected_sequence": (
            str(sequence) if sequence is not None else None
        ),
        "selected_frame_start": (
            support_pose.frame_start if support_pose is not None else None
        ),
        "selected_frame_end_exclusive": (
            support_pose.frame_end_exclusive
            if support_pose is not None
            else None
        ),
        "target_frame_selection_deferred": (
            calibration == FOUNDATION_POSE_SUPPORT_CALIBRATION
        ),
        "rejected_candidates": rejections,
    }


def _execute_job(
    job: dict,
    *,
    mesh: Path,
    sequence: Path | None,
    drop_test_available: bool,
    options: BatchOptions,
) -> None:
    """Generate and, when support is available, drop-test one mesh."""

    preparation = run_mesh_to_usd_workflow(
        job["mesh_to_usd_dir"],
        asset_path=str(mesh),
        support_sequence_dir=(
            str(sequence) if drop_test_available else None
        ),
        stable_window_frames=options.stable_window_frames,
        initial_search_frames=options.initial_search_frames,
        max_up_deviation_degrees=options.max_up_deviation_degrees,
        max_rotation_deviation_degrees=(
            options.max_rotation_deviation_degrees
        ),
        max_translation_deviation_m=options.max_translation_deviation_m,
        foundation_pose_weights_dir=(
            options.foundation_pose_weights_dir
            if drop_test_available
            else None
        ),
        foundation_pose_config_path=(
            options.foundation_pose_config_path
            if drop_test_available
            else None
        ),
        foundation_pose_debug=(
            options.foundation_pose_debug if drop_test_available else 0
        ),
        image=options.image,
        validate=options.validate,
        validator_image=options.validator_image,
        cache_dir=options.cache_dir,
        accept_eula=options.accept_eula,
        dev=options.dev,
        gpu_device=options.gpu_device,
    )
    job.update(
        {
            "output_usd": preparation["output_usd"],
            "visual_asset": preparation["visual_asset"],
            "recorded_support_pose": preparation.get(
                "recorded_support_pose"
            ),
            "generation_report": preparation["generation_report"],
            "validation_report": preparation.get("validation_report"),
            "support_pose_calibration": preparation.get(
                "support_pose_calibration"
            ),
            "support_provenance": preparation.get("support_provenance"),
            "foundation_pose_support_report": preparation.get(
                "foundation_pose_support_report"
            ),
            "foundation_pose_support_poses": preparation.get(
                "foundation_pose_support_poses"
            ),
            "foundation_pose_tracking_metadata": preparation.get(
                "foundation_pose_tracking_metadata"
            ),
        }
    )

    if options.run_drop_tests and drop_test_available:
        drop_result = run_drop_test(
            preparation["output_usd"],
            job["drop_test_dir"],
            initial_pose="recorded-support",
            recorded_support_path=preparation["recorded_support_pose"],
            video=options.video,
            fail_if_not_standing=options.fail_if_not_standing,
            image=options.image,
            cache_dir=options.cache_dir,
            accept_eula=options.accept_eula,
            dev=options.dev,
            gpu_device=options.gpu_device,
        )
        job.update(
            {
                "drop_test_report": drop_result["report_file"],
                "video_files": drop_result["video_files"],
                "standing": drop_result.get("standing", False),
            }
        )
    else:
        job["video_files"] = []

    job["status"] = (
        "generated-without-drop-test"
        if options.run_drop_tests and not drop_test_available
        else "passed"
    )
    job["artifact_fingerprints"] = _capture_artifact_fingerprints(job)


def _batch_status(counts: Counter, *, dry_run: bool) -> str:
    if dry_run:
        return "planned"
    if counts.get("failed", 0) or counts.get("skipped", 0):
        return "completed-with-failures"
    if counts.get("generated-without-drop-test", 0):
        return "completed-with-incomplete-drop-tests"
    return "passed"


def run_batch(
    mesh_root: str,
    sequence_root: str,
    output_root: str,
    *,
    methods: tuple[str, ...] = SUPPORTED_METHODS,
    mesh_filename: str = DEFAULT_MESH_FILENAME,
    object_ids: tuple[str, ...] = (),
    limit: int | None = None,
    stable_window_frames: int = DEFAULT_STABLE_WINDOW_FRAMES,
    initial_search_frames: int = DEFAULT_INITIAL_SEARCH_FRAMES,
    max_up_deviation_degrees: float = 2.0,
    max_rotation_deviation_degrees: float = 3.0,
    max_translation_deviation_m: float = 0.02,
    foundation_pose_weights_dir: str | None = None,
    foundation_pose_config_path: str | None = None,
    foundation_pose_debug: int = 0,
    validate: bool = True,
    run_drop_tests: bool = True,
    video: bool = True,
    fail_if_not_standing: bool = True,
    image: str = IMAGE_NAME,
    validator_image: str = VALIDATOR_IMAGE_NAME,
    accept_eula: bool = False,
    cache_dir: str | None = None,
    dev: bool = False,
    gpu_device: str | None = "0",
    resume: bool = True,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> dict:
    """Run a resumable, failure-isolated batch over matched object meshes."""

    if not dry_run and not isaac_sim_eula_accepted(accept_eula):
        raise ValueError(
            "mesh-to-USD batch requires Isaac Sim EULA acceptance; "
            "set ACCEPT_EULA=Y or pass --accept-eula"
        )
    if not methods or any(method not in SUPPORTED_METHODS for method in methods):
        raise ValueError(f"methods must be chosen from {SUPPORTED_METHODS}")
    options = BatchOptions(
        stable_window_frames=stable_window_frames,
        initial_search_frames=initial_search_frames,
        max_up_deviation_degrees=max_up_deviation_degrees,
        max_rotation_deviation_degrees=max_rotation_deviation_degrees,
        max_translation_deviation_m=max_translation_deviation_m,
        foundation_pose_weights_dir=foundation_pose_weights_dir,
        foundation_pose_config_path=foundation_pose_config_path,
        foundation_pose_debug=foundation_pose_debug,
        validate=validate,
        run_drop_tests=run_drop_tests,
        video=video,
        fail_if_not_standing=fail_if_not_standing,
        image=image,
        validator_image=validator_image,
        accept_eula=accept_eula,
        cache_dir=cache_dir,
        dev=dev,
        gpu_device=gpu_device,
    )
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / BATCH_REPORT_NAME

    sequences_by_object, rejected_sequences = discover_sequences(sequence_root)
    meshes_by_object, rejected_meshes = discover_meshes(
        mesh_root,
        methods,
        mesh_filename,
    )
    matched_objects = sorted(set(sequences_by_object) & set(meshes_by_object))
    if object_ids:
        requested = set(object_ids)
        unknown = sorted(requested - set(matched_objects))
        if unknown:
            raise ValueError(
                "requested object IDs are not matched: " + ", ".join(unknown)
            )
        matched_objects = [name for name in matched_objects if name in requested]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        matched_objects = matched_objects[:limit]

    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "mesh_root": str(Path(mesh_root).expanduser().resolve()),
        "sequence_root": str(Path(sequence_root).expanduser().resolve()),
        "output_root": str(output),
        "methods": list(methods),
        "mesh_filename": mesh_filename,
        "gpu_device": gpu_device,
        "foundation_pose_weights_dir": str(
            resolve_foundation_pose_weights_dir(
                foundation_pose_weights_dir
            )
        ),
        "foundation_pose_config_path": (
            str(Path(foundation_pose_config_path).expanduser().resolve())
            if foundation_pose_config_path is not None
            else None
        ),
        "foundation_pose_debug": foundation_pose_debug,
        "run_drop_tests": run_drop_tests,
        "video_required": bool(run_drop_tests and video),
        "validation_required": validate,
        "standing_required": bool(run_drop_tests and fail_if_not_standing),
        "matched_object_count": len(matched_objects),
        "planned_job_count": sum(
            len(meshes_by_object[object_id]) for object_id in matched_objects
        ),
        "rejected_sequences": rejected_sequences,
        "rejected_meshes": rejected_meshes,
        "support_selection": {},
        "jobs": [],
    }
    write_json(report_path, report)
    if report["planned_job_count"] == 0:
        report["status"] = "no-matched-jobs"
        report["status_counts"] = {}
        report["completed_at"] = utc_now()
        write_json(report_path, report)
        raise RuntimeError(
            "batch has no matched mesh/sequence jobs; "
            f"review {report_path}"
        )

    digest_cache: dict[Path, str] = {}
    for object_id in matched_objects:
        report["support_selection"][object_id] = {}
        for method, mesh in sorted(meshes_by_object[object_id].items()):
            (
                sequence,
                support_pose,
                support_pose_calibration,
                selection_rejections,
            ) = select_support_sequence_for_mesh(
                sequences_by_object[object_id],
                mesh,
                options=options,
            )
            report["support_selection"][object_id][method] = (
                _support_selection_report(
                    mesh=mesh,
                    sequence=sequence,
                    support_pose=support_pose,
                    calibration=support_pose_calibration,
                    rejections=selection_rejections,
                )
            )
            job_output = output / object_id / method
            preparation_output = job_output / "mesh_to_usd"
            drop_output = job_output / "drop_test"
            job_report_path = job_output / JOB_REPORT_NAME
            support_error = None
            if sequence is None or support_pose_calibration is None:
                support_error = "no usable support sequence for target mesh"
            drop_test_available = support_error is None
            resume_signature = _resume_signature(
                mesh=mesh,
                sequence=sequence,
                support_pose_calibration=support_pose_calibration,
                options=options,
                digest_cache=digest_cache,
            )

            resumed = None
            if resume and not dry_run:
                resumed = _resume_result(
                    job_report_path,
                    expected_mesh=mesh,
                    expected_sequence=sequence,
                    expected_resume_signature=resume_signature,
                    require_validation=validate,
                    require_drop_test=bool(
                        run_drop_tests and drop_test_available
                    ),
                    require_video=bool(
                        run_drop_tests and video and drop_test_available
                    ),
                    expected_support_pose_calibration=(
                        support_pose_calibration
                    ),
                )
            if resumed is not None:
                report["jobs"].append(resumed)
                write_json(report_path, report)
                continue

            job = {
                "object_id": object_id,
                "method": method,
                "status": _planned_job_status(
                    dry_run=dry_run,
                    run_drop_tests=run_drop_tests,
                    drop_test_available=drop_test_available,
                ),
                "started_at": utc_now(),
                "mesh": str(mesh),
                "sequence": str(sequence) if sequence is not None else None,
                "support_error": support_error,
                "support_pose_calibration": support_pose_calibration,
                "resume_signature": resume_signature,
                "mesh_to_usd_dir": str(preparation_output),
                "drop_test_dir": (
                    str(drop_output)
                    if run_drop_tests and drop_test_available
                    else None
                ),
                "job_report": str(job_report_path),
            }
            if dry_run:
                write_json(job_report_path, job)
                report["jobs"].append(job)
                write_json(report_path, report)
                continue

            write_json(job_report_path, job)
            try:
                _execute_job(
                    job,
                    mesh=mesh,
                    sequence=sequence,
                    drop_test_available=drop_test_available,
                    options=options,
                )
            except Exception as error:
                job["status"] = "failed"
                job["error_type"] = type(error).__name__
                job["error"] = str(error)
                existing_videos = sorted(drop_output.glob("drop_test*.mp4"))
                job["video_files"] = [str(path) for path in existing_videos]
            finally:
                job["completed_at"] = utc_now()
                write_json(job_report_path, job)
                report["jobs"].append(job)
                write_json(report_path, report)
            if fail_fast and job["status"] == "failed":
                report["status"] = "failed"
                report["completed_at"] = utc_now()
                write_json(report_path, report)
                raise RuntimeError(
                    f"batch failed at {object_id}/{method}: {job['error']}"
                )

    counts = Counter(job["status"] for job in report["jobs"])
    report["status_counts"] = dict(sorted(counts.items()))
    report["status"] = _batch_status(counts, dry_run=dry_run)
    report["completed_at"] = utc_now()
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", required=True)
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=SUPPORTED_METHODS,
    )
    parser.add_argument(
        "--mesh-filename",
        default=DEFAULT_MESH_FILENAME,
        help=(
            "GLB basename selected inside each <object>/<method> directory "
            f"(default: {DEFAULT_MESH_FILENAME})"
        ),
    )
    parser.add_argument("--object-id", dest="object_ids", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--stable-window-frames",
        type=int,
        default=DEFAULT_STABLE_WINDOW_FRAMES,
    )
    parser.add_argument(
        "--initial-search-frames",
        type=int,
        default=DEFAULT_INITIAL_SEARCH_FRAMES,
    )
    parser.add_argument("--max-up-deviation-degrees", type=float, default=2.0)
    parser.add_argument("--max-rotation-deviation-degrees", type=float, default=3.0)
    parser.add_argument("--max-translation-deviation-m", type=float, default=0.02)
    parser.add_argument(
        "--foundation-pose-weights-dir",
        help=(
            "FoundationPose weights override "
            f"(default: {DEFAULT_FOUNDATION_POSE_WEIGHTS_DIR})"
        ),
    )
    parser.add_argument("--foundation-pose-config-path")
    parser.add_argument(
        "--foundation-pose-debug",
        type=int,
        default=0,
        choices=(0, 1, 2),
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-drop-tests",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-if-not-standing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--image", default=IMAGE_NAME)
    parser.add_argument("--validator-image", default=VALIDATOR_IMAGE_NAME)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--gpu-device",
        default="0",
        help="Expose exactly one GPU to Isaac Sim (default: 0)",
    )
    parser.add_argument("--dev", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = vars(parser.parse_args())
    args["methods"] = tuple(args["methods"])
    args["object_ids"] = tuple(args["object_ids"])
    result = run_batch(**args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "matched_object_count": result["matched_object_count"],
                "planned_job_count": result["planned_job_count"],
                "status_counts": result.get("status_counts", {}),
                "report": str(
                    Path(args["output_root"]).expanduser().resolve()
                    / BATCH_REPORT_NAME
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
