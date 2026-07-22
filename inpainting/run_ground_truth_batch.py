"""Plan or execute the calibrated TACO ground-truth overlay batch.

The default is a read-only JSON plan.  Full robot rendering is possible only
when both ``--execute`` and ``--gpu`` are explicit.  Completed artifacts are
resumed from strict metadata contracts; existing incomplete outputs are never
overwritten without ``--overwrite``.  The safe default stage order is robot
render, TACO object depth, depth-aware composite, and comparison grid.  The
``--allow-hard-composite`` escape hatch is an explicitly degraded/debug mode
for cases where no valid object-depth bundle exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

from .composite_robot import (
    validate_robot_render_bundle,
    validate_taco_object_render_bundle,
)
from .contracts import (
    RESOLVED_EXPERIMENT_SCHEMA,
    VideoGeometry,
    validate_depth_file,
    validate_mask_file,
    validate_robot_trajectory_file,
)
from .robot_renderer.container_runner import (
    resolve_local_image_id,
    validate_gpu_selector,
)
from .robot_renderer.provenance import verify_file_record
from .taco_object_depth import (
    OBJECT_RENDER_PROVENANCE_SCHEMA,
    SOURCE_INPUT_NAMES,
    object_render_source_records,
)
from .taco_camera import load_taco_camera
from .video_io import probe_video


PLAN_SCHEMA = "v2d.inpainting.gt-batch-plan/v1"
GRID_SCHEMA = "v2d.inpainting.gt-comparison-grid/v1"
STAGES = ("render", "object_depth", "composite", "grid")
GPU_STAGES = frozenset(("render", "object_depth"))
PENDING_STATES = ("pending", "pending_overwrite")


class BatchPlanError(ValueError):
    """Raised when a batch cannot be planned or executed safely."""


@dataclass(frozen=True)
class SequencePaths:
    sequence_id: str
    source_video: Path
    motion_parquet: Path
    trajectory: Path
    intrinsic: Path
    world_to_camera: Path
    inpaint_masks: Path
    inpaint_video: Path
    inpaint_metadata: Path
    robot_dir: Path
    robot_rgb: Path
    robot_mask: Path
    robot_depth: Path
    robot_metadata: Path
    object_dir: Path
    object_mask: Path
    object_depth: Path
    object_metadata: Path
    composite_video: Path
    composite_metadata: Path
    grid_video: Path
    grid_metadata: Path


@dataclass(frozen=True)
class Action:
    sequence_id: str
    stage: str
    status: str
    reason: str
    command: tuple[str, ...] | None
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "stage": self.stage,
            "status": self.status,
            "reason": self.reason,
            "command": list(self.command) if self.command is not None else None,
            "command_shell": shlex.join(self.command)
            if self.command is not None
            else None,
            "inputs": [str(path) for path in self.inputs],
            "outputs": [str(path) for path in self.outputs],
        }


@dataclass(frozen=True)
class PlanOptions:
    manifest_path: Path
    run_root: Path | None = None
    sequence_ids: tuple[str, ...] = ()
    stages: tuple[str, ...] = STAGES
    scene_utils_root: Path | None = None
    repository_root: Path | None = None
    python_executable: Path | None = None
    gpu: str | None = None
    renderer_image: str = "robotic-grounding:photo-render-v6"
    object_mesh_root: Path | None = None
    overwrite: bool = False
    depth_guard_m: float = 0.003
    allow_hard_composite: bool = False
    grid_tile_width: int = 640
    grid_columns: int = 3
    grid_max_frames: int | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchPlanError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchPlanError(f"expected a JSON object at {path}")
    return value


def _manifest_path(value: Any, label: str, *, kind: str) -> Path:
    if kind not in {"file", "dir", "path"}:
        raise AssertionError(f"unsupported path kind {kind!r}")
    if not isinstance(value, str) or not value:
        raise BatchPlanError(f"manifest {label} must be a non-empty absolute path")
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise BatchPlanError(f"manifest {label} must be absolute, got {value!r}")
    path = raw.resolve()
    if kind == "file" and not path.is_file():
        raise BatchPlanError(f"manifest {label} file does not exist: {path}")
    if kind == "dir" and not path.is_dir():
        raise BatchPlanError(f"manifest {label} directory does not exist: {path}")
    return path


def _geometry_matches(metadata: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(metadata, dict):
        return False
    try:
        return (
            int(metadata["frame_count"]) == int(expected["frame_count"])
            and int(metadata["width"]) == int(expected["width"])
            and int(metadata["height"]) == int(expected["height"])
            and abs(float(metadata["fps"]) - float(expected["fps"])) <= 1e-3
        )
    except (KeyError, TypeError, ValueError):
        return False


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _verify_exact_file_record(
    path: Path,
    record: Any,
    *,
    label: str,
    expected_keys: set[str],
) -> None:
    """Verify one exact fingerprint record and reject concurrent replacement."""

    if not isinstance(record, dict) or set(record) != expected_keys:
        raise BatchPlanError(
            f"{label} fingerprint must contain exactly {sorted(expected_keys)}"
        )
    if "name" in expected_keys and record.get("name") != path.name:
        raise BatchPlanError(
            f"{label} fingerprint name {record.get('name')!r} != {path.name!r}"
        )
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise BatchPlanError(f"{label} fingerprint bytes must be a positive integer")
    before = _stat_signature(path)
    verify_file_record(path, record, label=label)
    after = _stat_signature(path)
    if before != after:
        raise BatchPlanError(f"{label} changed while its fingerprint was verified")


def _validate_unresolved_file_record(record: Any, *, label: str) -> None:
    """Validate a portable record whose current host path is not in the manifest."""

    expected_keys = {"name", "bytes", "sha256"}
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise BatchPlanError(f"{label} must contain exactly {sorted(expected_keys)}")
    name = record.get("name")
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
    ):
        raise BatchPlanError(f"{label} name must be one non-empty basename")
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise BatchPlanError(f"{label} bytes must be a positive integer")
    digest = record.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise BatchPlanError(f"{label} sha256 must be 64 lowercase hex digits")


def _video_geometry(value: dict[str, Any], *, label: str) -> VideoGeometry:
    try:
        frame_count = value["frame_count"]
        width = value["width"]
        height = value["height"]
        fps = value["fps"]
    except KeyError as exc:
        raise BatchPlanError(f"{label} is missing {exc.args[0]}") from exc
    for key, item in (
        ("frame_count", frame_count),
        ("width", width),
        ("height", height),
    ):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise BatchPlanError(f"{label} {key} must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise BatchPlanError(f"{label} fps must be numeric")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise BatchPlanError(f"{label} fps must be positive and finite")
    return VideoGeometry(
        frame_count=frame_count,
        width=width,
        height=height,
        fps=fps,
    )


def _expected_artifact_files(paths: SequencePaths, stage: str) -> tuple[Path, ...]:
    if stage == "render":
        return (
            paths.robot_rgb,
            paths.robot_mask,
            paths.robot_depth,
            paths.robot_metadata,
        )
    if stage == "object_depth":
        return (paths.object_mask, paths.object_depth, paths.object_metadata)
    if stage == "composite":
        return (paths.composite_video, paths.composite_metadata)
    if stage == "grid":
        return (paths.grid_video, paths.grid_metadata)
    raise AssertionError(stage)


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def _any_stage_output_exists(paths: SequencePaths, stage: str) -> bool:
    expected = _expected_artifact_files(paths, stage)
    candidates = (*expected, *(_partial_path(path) for path in expected))
    return any(path.exists() for path in candidates)


def _render_complete(
    paths: SequencePaths, geometry: dict[str, Any]
) -> tuple[bool, str]:
    files = _expected_artifact_files(paths, "render")
    missing = [
        path.name for path in files if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        return False, f"missing/empty render artifacts: {missing}"
    video_geometry = _video_geometry(geometry, label="manifest video geometry")
    try:
        metadata, artifacts = validate_robot_render_bundle(
            paths.robot_metadata, video_geometry
        )
    except (OSError, ValueError) as exc:
        return False, f"invalid robot render bundle: {exc}"
    if artifacts != {
        "rgb": paths.robot_rgb,
        "mask": paths.robot_mask,
        "depth": paths.robot_depth,
    }:
        return False, "robot metadata resolves to artifacts outside this run"
    verification = (metadata.get("render_statistics") or {}).get("video_verification")
    if not isinstance(verification, dict):
        return False, "robot metadata lacks decoded video verification"
    verification_geometry = {
        "frame_count": verification.get("decoded_frame_count"),
        "width": verification.get("width"),
        "height": verification.get("height"),
        "fps": verification.get("fps"),
    }
    if not _geometry_matches(verification_geometry, geometry):
        return False, "robot video verification does not match the manifest"
    expected_shape = (
        int(geometry["frame_count"]),
        int(geometry["height"]),
        int(geometry["width"]),
    )
    try:
        mask = np.load(paths.robot_mask, mmap_mode="r", allow_pickle=False)
        depth = np.load(paths.robot_depth, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        return False, f"could not inspect render arrays: {exc}"
    if mask.shape != expected_shape or mask.dtype != np.bool_:
        return (
            False,
            f"robot mask is {mask.shape}/{mask.dtype}, expected {expected_shape}/bool",
        )
    if depth.shape != expected_shape or depth.dtype != np.float32:
        return False, (
            f"robot depth is {depth.shape}/{depth.dtype}, expected "
            f"{expected_shape}/float32"
        )
    if paths.robot_metadata.stat().st_mtime_ns < max(
        paths.robot_rgb.stat().st_mtime_ns,
        paths.robot_mask.stat().st_mtime_ns,
        paths.robot_depth.stat().st_mtime_ns,
    ):
        return False, "robot metadata predates one or more render artifacts"
    return True, "complete render metadata and artifacts validated"


def _object_depth_complete(
    paths: SequencePaths,
    geometry: dict[str, Any],
    *,
    renderer_image: str | None = None,
    repository_root: Path | None = None,
) -> tuple[bool, str]:
    files = _expected_artifact_files(paths, "object_depth")
    missing = [
        path.name for path in files if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        return False, f"missing/empty object-depth artifacts: {missing}"
    video_geometry = _video_geometry(geometry, label="manifest video geometry")
    try:
        metadata, artifacts = validate_taco_object_render_bundle(
            paths.object_metadata, video_geometry
        )
        if artifacts != {"mask": paths.object_mask, "depth": paths.object_depth}:
            return False, "object metadata resolves to artifacts outside this run"
        mask = validate_mask_file(paths.object_mask, video_geometry)
        validate_depth_file(
            paths.object_depth,
            mask,
            video_geometry,
            name="TACO object depth",
        )
        if renderer_image is not None:
            if repository_root is None:
                raise BatchPlanError(
                    "repository root is required for object-depth provenance validation"
                )
            _verify_current_object_depth_generation(
                metadata=metadata,
                paths=paths,
                renderer_image=renderer_image,
                repository_root=repository_root,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"invalid object-depth render bundle: {exc}"
    if paths.object_metadata.stat().st_mtime_ns < max(
        paths.object_mask.stat().st_mtime_ns,
        paths.object_depth.stat().st_mtime_ns,
    ):
        return False, "object metadata predates one or more object-depth artifacts"
    reason = "complete object-depth metadata and artifacts validated"
    if renderer_image is not None:
        reason += "; current image and source-input provenance validated"
    return True, reason


def _verify_current_object_depth_generation(
    *,
    metadata: dict[str, Any],
    paths: SequencePaths,
    renderer_image: str,
    repository_root: Path,
) -> None:
    """Reject object-depth resume when its image or source files have changed."""

    if metadata.get("container_image") != renderer_image:
        raise BatchPlanError(
            "object-depth requested image reference differs from the selected image"
        )
    recorded_image_id = metadata.get("container_image_id")
    if not isinstance(recorded_image_id, str):
        raise BatchPlanError(
            "object-depth metadata lacks an immutable container image ID"
        )
    current_image_id = resolve_local_image_id(renderer_image)
    if recorded_image_id != current_image_id:
        raise BatchPlanError(
            "object-depth immutable container image ID differs from the selected image"
        )

    provenance = metadata.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != OBJECT_RENDER_PROVENANCE_SCHEMA
    ):
        raise BatchPlanError("object-depth metadata lacks committed source provenance")
    if provenance.get("hash_algorithm") != "sha256":
        raise BatchPlanError("object-depth provenance hash algorithm is not sha256")
    records = provenance.get("inputs")
    if not isinstance(records, dict) or set(records) != set(SOURCE_INPUT_NAMES):
        raise BatchPlanError(
            "object-depth provenance must record the exact source input set"
        )
    recorded_sources = provenance.get("implementation_sources")
    current_sources = object_render_source_records(repository_root)
    if recorded_sources != current_sources:
        raise BatchPlanError(
            "object-depth implementation source provenance differs from the current "
            "repository"
        )

    sources = {
        "source_parquet": paths.motion_parquet,
        "source_video": paths.source_video,
        "intrinsic": paths.intrinsic,
        "world_to_camera": paths.world_to_camera,
    }
    container_paths = {
        "source_parquet": "/inputs/motion.parquet",
        "source_video": "/inputs/source.mp4",
        "intrinsic": f"/inputs/intrinsic{paths.intrinsic.suffix}",
        "world_to_camera": (f"/inputs/world_to_camera{paths.world_to_camera.suffix}"),
    }
    for name, source in sources.items():
        record = records[name]
        if not isinstance(record, dict):
            raise BatchPlanError(
                f"object-depth input {name} provenance is not an object"
            )
        if record.get("host_path") != str(source.resolve()):
            raise BatchPlanError(
                f"object-depth input {name} host path differs from this run"
            )
        if record.get("container_path") != container_paths[name]:
            raise BatchPlanError(
                f"object-depth input {name} container path is not canonical"
            )
        verify_file_record(source, record, label=f"object-depth input {name}")


def _composite_complete(
    paths: SequencePaths,
    geometry: dict[str, Any],
    options: PlanOptions,
    *,
    use_object_depth: bool,
) -> tuple[bool, str]:
    required_inputs = [
        paths.inpaint_video,
        paths.robot_rgb,
        paths.robot_mask,
        paths.robot_depth,
        paths.robot_metadata,
    ]
    if use_object_depth:
        required_inputs.extend(
            (paths.object_mask, paths.object_depth, paths.object_metadata)
        )
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        return False, f"missing composite inputs: {missing_inputs}"
    files = _expected_artifact_files(paths, "composite")
    missing = [
        path.name for path in files if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        return False, f"missing/empty composite artifacts: {missing}"
    try:
        metadata = _load_json(paths.composite_metadata)
    except (BatchPlanError, FileNotFoundError) as exc:
        return False, str(exc)
    if metadata.get("schema_version") != "v2d.inpainting.composite/v1":
        return False, "composite metadata schema is not v2d.inpainting.composite/v1"
    if metadata.get("state") != "complete":
        return False, f"composite metadata state is {metadata.get('state')!r}"
    if not _geometry_matches(metadata.get("geometry"), geometry):
        return False, "composite geometry does not match the manifest"
    if metadata.get("frames_written") != int(geometry["frame_count"]):
        return False, "composite frame count does not match the manifest"
    expected_mode = "taco_object_depth" if use_object_depth else "hard_robot_mask"
    if metadata.get("compositing") != expected_mode:
        return False, f"composite mode is not the selected {expected_mode!r} mode"
    if use_object_depth:
        try:
            guard_matches = (
                abs(float(metadata.get("depth_guard_m")) - options.depth_guard_m)
                <= 1e-12
            )
        except (TypeError, ValueError):
            guard_matches = False
        if not guard_matches:
            return False, "composite depth guard differs from the selected batch option"
    expected_paths = {
        "base_video": paths.inpaint_video,
        "robot_video": paths.robot_rgb,
        "robot_mask": paths.robot_mask,
        "robot_metadata": paths.robot_metadata,
        "output_video": paths.composite_video,
    }
    if use_object_depth:
        expected_paths.update(
            {
                "object_mask": paths.object_mask,
                "object_depth": paths.object_depth,
                "object_metadata": paths.object_metadata,
            }
        )
    elif any(
        key in metadata for key in ("object_mask", "object_depth", "object_metadata")
    ):
        return False, "hard-mask composite unexpectedly declares object-depth inputs"
    for key, expected in expected_paths.items():
        value = metadata.get(key)
        if not isinstance(value, str) or Path(value).resolve() != expected:
            return False, f"composite metadata {key} does not match this run"
    expected_fingerprint_paths = {
        "base_video": paths.inpaint_video,
        "robot_metadata": paths.robot_metadata,
        "robot_rgb": paths.robot_rgb,
        "robot_mask": paths.robot_mask,
        "robot_depth": paths.robot_depth,
    }
    if use_object_depth:
        expected_fingerprint_paths.update(
            {
                "object_mask": paths.object_mask,
                "object_depth": paths.object_depth,
                "object_metadata": paths.object_metadata,
            }
        )
    fingerprints = metadata.get("input_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != set(
        expected_fingerprint_paths
    ):
        return False, "composite input fingerprints do not declare the exact selected inputs"
    try:
        for key, path in expected_fingerprint_paths.items():
            _verify_exact_file_record(
                path,
                fingerprints[key],
                label=f"composite input {key}",
                expected_keys={"bytes", "sha256"},
            )
        _verify_exact_file_record(
            paths.composite_video,
            metadata.get("output_fingerprint"),
            label="composite output",
            expected_keys={"bytes", "sha256"},
        )
    except (OSError, ValueError) as exc:
        return False, f"composite fingerprint validation failed: {exc}"
    input_mtimes = [
        paths.inpaint_video.stat().st_mtime_ns,
        paths.robot_rgb.stat().st_mtime_ns,
        paths.robot_mask.stat().st_mtime_ns,
        paths.robot_depth.stat().st_mtime_ns,
        paths.robot_metadata.stat().st_mtime_ns,
    ]
    if use_object_depth:
        input_mtimes.extend(
            (
                paths.object_mask.stat().st_mtime_ns,
                paths.object_depth.stat().st_mtime_ns,
                paths.object_metadata.stat().st_mtime_ns,
            )
        )
    if paths.composite_video.stat().st_mtime_ns < max(input_mtimes):
        return False, "composite video predates one or more inputs"
    if (
        paths.composite_metadata.stat().st_mtime_ns
        < paths.composite_video.stat().st_mtime_ns
    ):
        return False, "composite metadata predates the output video"
    return True, f"complete {expected_mode} composite metadata and artifact validated"


def _grid_spec(
    paths: SequencePaths,
    options: PlanOptions,
    *,
    comparison_label: str,
) -> dict[str, Any]:
    return {
        "videos": [
            str(paths.source_video),
            str(paths.inpaint_video),
            str(paths.composite_video),
        ],
        "labels": ["Source", "E2FGVI", comparison_label],
        "tile_width": options.grid_tile_width,
        "columns": options.grid_columns,
        "max_frames": options.grid_max_frames,
    }


def _grid_complete(
    paths: SequencePaths,
    geometry: dict[str, Any],
    options: PlanOptions,
    *,
    comparison_label: str,
) -> tuple[bool, str]:
    required_inputs = (paths.source_video, paths.inpaint_video, paths.composite_video)
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        return False, f"missing grid inputs: {missing_inputs}"
    files = _expected_artifact_files(paths, "grid")
    missing = [
        path.name for path in files if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        return False, f"missing/empty grid artifacts: {missing}"
    try:
        metadata = _load_json(paths.grid_metadata)
    except (BatchPlanError, FileNotFoundError) as exc:
        return False, str(exc)
    if (
        metadata.get("schema_version") != GRID_SCHEMA
        or metadata.get("state") != "complete"
    ):
        return False, "grid sidecar is not a complete supported grid run"
    if metadata.get("specification") != _grid_spec(
        paths, options, comparison_label=comparison_label
    ):
        return False, "grid specification differs from the selected inputs/options"
    output_geometry = metadata.get("geometry")
    if not isinstance(output_geometry, dict):
        return False, "grid sidecar lacks output geometry"
    expected_frames = int(geometry["frame_count"])
    if options.grid_max_frames is not None:
        expected_frames = min(expected_frames, options.grid_max_frames)
    try:
        matches = (
            int(output_geometry["frame_count"]) == expected_frames
            and abs(float(output_geometry["fps"]) - float(geometry["fps"])) <= 1e-3
        )
    except (KeyError, TypeError, ValueError):
        matches = False
    if not matches:
        return False, "grid frame count/FPS differs from the selected inputs"
    if paths.grid_video.stat().st_mtime_ns < max(
        paths.source_video.stat().st_mtime_ns,
        paths.inpaint_video.stat().st_mtime_ns,
        paths.composite_video.stat().st_mtime_ns,
    ):
        return False, "grid video predates one or more inputs"
    if paths.grid_metadata.stat().st_mtime_ns < paths.grid_video.stat().st_mtime_ns:
        return False, "grid metadata predates the grid video"
    return True, "complete comparison grid sidecar and artifact validated"


def _validate_inpaint(paths: SequencePaths, geometry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    video_exists = paths.inpaint_video.is_file() and paths.inpaint_video.stat().st_size > 0
    if not video_exists:
        errors.append(f"missing shared E2FGVI video: {paths.inpaint_video}")
    mask_exists = paths.inpaint_masks.is_file() and paths.inpaint_masks.stat().st_size > 0
    if not mask_exists:
        errors.append(f"missing shared E2FGVI masks: {paths.inpaint_masks}")
    if not paths.inpaint_metadata.is_file():
        errors.append(f"missing shared E2FGVI metadata: {paths.inpaint_metadata}")
        return errors
    try:
        metadata = _load_json(paths.inpaint_metadata)
    except BatchPlanError as exc:
        errors.append(str(exc))
        return errors
    if metadata.get("schema") != "v2d.e2fgvi.inpainting.v1":
        errors.append("shared E2FGVI metadata has an unsupported schema")
    if metadata.get("status") != "completed":
        errors.append(f"shared E2FGVI status is {metadata.get('status')!r}")
    output = metadata.get("output")
    if not _geometry_matches(output, geometry):
        errors.append("shared E2FGVI geometry does not match the manifest")
    implementation = metadata.get("implementation")
    if not isinstance(implementation, dict):
        errors.append("shared E2FGVI implementation provenance is missing")
    else:
        image_id = implementation.get("container_image_id")
        provenance = implementation.get("container_image_provenance")
        if image_id is not None:
            if (
                not isinstance(image_id, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
                or provenance != "recorded_immutable_id"
            ):
                errors.append("shared E2FGVI immutable container provenance is invalid")
        elif provenance != "legacy_unrecorded":
            errors.append(
                "shared E2FGVI metadata lacks immutable or explicitly legacy container provenance"
            )

    inputs = metadata.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"video", "masks", "checkpoint"}:
        errors.append("shared E2FGVI metadata does not declare the exact input set")
    else:
        try:
            if video_exists:
                _verify_exact_file_record(
                    paths.source_video,
                    inputs["video"],
                    label="shared E2FGVI source video",
                    expected_keys={"name", "bytes", "sha256"},
                )
            if mask_exists:
                _verify_exact_file_record(
                    paths.inpaint_masks,
                    inputs["masks"],
                    label="shared E2FGVI masks",
                    expected_keys={"name", "bytes", "sha256"},
                )
            _validate_unresolved_file_record(
                inputs["checkpoint"], label="shared E2FGVI checkpoint record"
            )
        except (OSError, ValueError) as exc:
            errors.append(f"shared E2FGVI input fingerprint validation failed: {exc}")
    if video_exists:
        try:
            _verify_exact_file_record(
                paths.inpaint_video,
                output,
                label="shared E2FGVI output",
                expected_keys={
                    "name",
                    "bytes",
                    "sha256",
                    "width",
                    "height",
                    "frame_count",
                    "fps",
                },
            )
        except (OSError, ValueError) as exc:
            errors.append(f"shared E2FGVI output fingerprint validation failed: {exc}")
        if paths.inpaint_metadata.stat().st_mtime_ns < paths.inpaint_video.stat().st_mtime_ns:
            errors.append("shared E2FGVI metadata predates its output video")
    return errors


def _classify(
    *,
    complete: bool,
    completion_reason: str,
    outputs_exist: bool,
    prerequisite_errors: Iterable[str],
    overwrite: bool,
) -> tuple[str, str]:
    errors = list(prerequisite_errors)
    if errors:
        return "blocked", "; ".join(errors)
    if overwrite:
        return "pending_overwrite", (
            "explicit overwrite requested"
            if outputs_exist
            else "no current output; overwrite permission recorded"
        )
    if complete:
        return "skipped_complete", completion_reason
    if outputs_exist:
        return "blocked", (
            f"existing output is incomplete ({completion_reason}); use --overwrite only "
            "after inspecting it"
        )
    return "pending", "required output is absent"


def _sequence_paths(
    run_root: Path,
    sequence_id: str,
    source_video: Path,
    motion_parquet: Path,
    intrinsic: Path,
    world_to_camera: Path,
) -> SequencePaths:
    sequence_root = (run_root / sequence_id).resolve()
    try:
        sequence_root.relative_to(run_root)
    except ValueError as exc:
        raise BatchPlanError(
            f"resolved sequence output escapes run root: {sequence_root}"
        ) from exc
    condition = sequence_root / "ground_truth"
    tracking = condition / "tracking"
    shared_arm_mask = sequence_root / "shared_arm_mask"
    shared_inpaint = sequence_root / "shared_inpaint"
    robot = condition / "robot_render"
    objects = condition / "object_render"
    return SequencePaths(
        sequence_id=sequence_id,
        source_video=source_video,
        motion_parquet=motion_parquet,
        trajectory=(tracking / "robot_trajectory.npz").resolve(),
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
        inpaint_masks=(shared_arm_mask / "arm_mask.npy").resolve(),
        inpaint_video=(shared_inpaint / "e2fgvi_960.mp4").resolve(),
        inpaint_metadata=(shared_inpaint / "e2fgvi_960.json").resolve(),
        robot_dir=robot.resolve(),
        robot_rgb=(robot / "robot_rgb.mp4").resolve(),
        robot_mask=(robot / "robot_mask.npy").resolve(),
        robot_depth=(robot / "robot_depth.npy").resolve(),
        robot_metadata=(robot / "render_metadata.json").resolve(),
        object_dir=objects.resolve(),
        object_mask=(objects / "object_mask.npy").resolve(),
        object_depth=(objects / "object_depth.npy").resolve(),
        object_metadata=(objects / "object_render_metadata.json").resolve(),
        composite_video=(condition / "final_overlay.mp4").resolve(),
        composite_metadata=(condition / "final_overlay.json").resolve(),
        grid_video=(condition / "final_comparison_grid.mp4").resolve(),
        grid_metadata=(condition / "final_comparison_grid.json").resolve(),
    )


def _render_command(
    paths: SequencePaths,
    geometry: dict[str, Any],
    *,
    asset_root: Path,
    scene_utils_root: Path,
    repository_root: Path,
    python_executable: Path,
    options: PlanOptions,
) -> tuple[str, ...]:
    command = [
        str(python_executable),
        "-m",
        "inpainting.robot_renderer.container_runner",
        "--trajectory",
        str(paths.trajectory),
        "--intrinsics",
        str(paths.intrinsic),
        "--world-to-camera",
        str(paths.world_to_camera),
        "--asset-root",
        str(asset_root),
        "--scene-utils-root",
        str(scene_utils_root),
        "--output-dir",
        str(paths.robot_dir),
        "--repository-root",
        str(repository_root),
        "--image",
        options.renderer_image,
        "--width",
        str(int(geometry["width"])),
        "--height",
        str(int(geometry["height"])),
        "--fps",
        f"{float(geometry['fps']):.12g}",
        "--execute",
    ]
    if options.gpu is not None:
        command.extend(("--gpu", options.gpu))
    if options.overwrite:
        command.append("--overwrite")
    return tuple(command)


def _object_depth_command(
    paths: SequencePaths,
    *,
    object_mesh_root: Path,
    repository_root: Path,
    python_executable: Path,
    options: PlanOptions,
) -> tuple[str, ...]:
    command = [
        str(python_executable),
        "-m",
        "inpainting.taco_object_depth_container",
        "--sequence-id",
        paths.sequence_id,
        "--parquet",
        str(paths.motion_parquet),
        "--source-video",
        str(paths.source_video),
        "--intrinsics",
        str(paths.intrinsic),
        "--world-to-camera",
        str(paths.world_to_camera),
        "--mesh-root",
        str(object_mesh_root),
        "--output-dir",
        str(paths.object_dir),
        "--repository-root",
        str(repository_root),
        "--image",
        options.renderer_image,
        "--execute",
    ]
    if options.gpu is not None:
        command.extend(("--gpu", options.gpu))
    if options.overwrite:
        command.append("--overwrite")
    return tuple(command)


def _composite_command(
    paths: SequencePaths,
    *,
    python_executable: Path,
    options: PlanOptions,
    use_object_depth: bool,
) -> tuple[str, ...]:
    command = [
        str(python_executable),
        "-m",
        "inpainting.composite_robot",
        "--base-video",
        str(paths.inpaint_video),
        "--robot-video",
        str(paths.robot_rgb),
        "--robot-mask",
        str(paths.robot_mask),
        "--robot-metadata",
        str(paths.robot_metadata),
        "--output-video",
        str(paths.composite_video),
        "--metadata",
        str(paths.composite_metadata),
    ]
    if use_object_depth:
        command.extend(
            (
                "--object-mask",
                str(paths.object_mask),
                "--object-depth",
                str(paths.object_depth),
                "--object-metadata",
                str(paths.object_metadata),
                "--depth-guard-m",
                f"{options.depth_guard_m:.12g}",
            )
        )
    if options.overwrite:
        command.append("--overwrite")
    return tuple(command)


def _grid_command(
    paths: SequencePaths,
    *,
    python_executable: Path,
    options: PlanOptions,
    comparison_label: str,
) -> tuple[str, ...]:
    command = [str(python_executable), "-m", "inpainting.make_video_grid"]
    for video, label in zip(
        (paths.source_video, paths.inpaint_video, paths.composite_video),
        ("Source", "E2FGVI", comparison_label),
        strict=True,
    ):
        command.extend(("--video", str(video), "--label", label))
    command.extend(
        (
            "--output",
            str(paths.grid_video),
            "--tile-width",
            str(options.grid_tile_width),
            "--columns",
            str(options.grid_columns),
        )
    )
    if options.grid_max_frames is not None:
        command.extend(("--max-frames", str(options.grid_max_frames)))
    return tuple(command)


def build_plan(options: PlanOptions) -> dict[str, Any]:
    """Build a read-only, fully resolved action plan."""

    manifest_path = options.manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != RESOLVED_EXPERIMENT_SCHEMA:
        raise BatchPlanError(
            f"manifest schema must be {RESOLVED_EXPERIMENT_SCHEMA!r}, got "
            f"{manifest.get('schema_version')!r}"
        )
    if (
        manifest.get("robot") != "dexmate_vega"
        or manifest.get("gripper") != "sharpa_wave"
    ):
        raise BatchPlanError("GT batch supports only dexmate_vega + sharpa_wave")
    run_root = (options.run_root or manifest_path.parent).expanduser().resolve()
    repository_root = (
        (options.repository_root or Path(__file__).resolve().parents[1])
        .expanduser()
        .resolve()
    )
    if not repository_root.is_dir():
        raise BatchPlanError(f"repository root does not exist: {repository_root}")
    python_executable = (
        (options.python_executable or Path(sys.executable)).expanduser().resolve()
    )
    if not python_executable.is_file():
        raise BatchPlanError(f"Python executable does not exist: {python_executable}")
    stages = tuple(dict.fromkeys(options.stages))
    if not stages or any(stage not in STAGES for stage in stages):
        raise BatchPlanError(
            f"stages must be a non-empty subset of {STAGES}, got {stages}"
        )
    stages = tuple(stage for stage in STAGES if stage in stages)
    if options.grid_tile_width <= 0 or options.grid_columns <= 0:
        raise BatchPlanError("grid tile width and columns must be positive")
    if options.grid_max_frames is not None and options.grid_max_frames <= 0:
        raise BatchPlanError("grid max frames must be positive")
    if (
        isinstance(options.depth_guard_m, bool)
        or not np.isfinite(options.depth_guard_m)
        or options.depth_guard_m < 0.0
    ):
        raise BatchPlanError("depth guard must be a finite non-negative number")
    try:
        validate_gpu_selector(options.gpu)
    except ValueError as exc:
        raise BatchPlanError(str(exc)) from exc
    if not options.renderer_image.strip():
        raise BatchPlanError("renderer image must not be empty")

    roots = manifest.get("roots")
    if not isinstance(roots, dict):
        raise BatchPlanError("manifest lacks roots")
    asset_root = _manifest_path(
        roots.get("robot_assets"), "roots.robot_assets", kind="dir"
    )
    camera_root = _manifest_path(roots.get("camera"), "roots.camera", kind="dir")
    derived_scene_utils = asset_root.parent / "tasks" / "scene_utils"
    scene_utils_root = (
        (options.scene_utils_root or derived_scene_utils).expanduser().resolve()
    )
    derived_object_mesh_root = (
        asset_root
        / "human_motion_data"
        / "taco_v2.0"
        / "object_assets"
        / "meshes"
        / "taco"
    )
    object_mesh_root = (
        (options.object_mesh_root or derived_object_mesh_root).expanduser().resolve()
    )
    global_render_errors: list[str] = []
    if not scene_utils_root.is_dir():
        global_render_errors.append(
            f"scene-utils directory does not exist: {scene_utils_root}"
        )
    else:
        for name in ("arm_ik.py", "arm_mount_opt.py"):
            if not (scene_utils_root / name).is_file():
                global_render_errors.append(
                    f"missing external IK source: {scene_utils_root / name}"
                )
    global_object_errors: list[str] = []
    if not object_mesh_root.is_dir():
        global_object_errors.append(
            f"TACO object mesh directory does not exist: {object_mesh_root}"
        )

    sequence_entries = manifest.get("sequences")
    if not isinstance(sequence_entries, list) or not sequence_entries:
        raise BatchPlanError("manifest sequences must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in sequence_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("sequence_id"), str):
            raise BatchPlanError(
                "each manifest sequence must have a string sequence_id"
            )
        sequence_id = entry["sequence_id"]
        if (
            not sequence_id
            or sequence_id in {".", ".."}
            or Path(sequence_id).name != sequence_id
        ):
            raise BatchPlanError(
                f"manifest sequence_id must be one safe path segment, got {sequence_id!r}"
            )
        if sequence_id in by_id:
            raise BatchPlanError(f"duplicate manifest sequence {sequence_id!r}")
        by_id[sequence_id] = entry
    selected_ids = options.sequence_ids or tuple(by_id)
    if len(set(selected_ids)) != len(selected_ids):
        raise BatchPlanError("sequence selection contains duplicates")
    unknown = [sequence_id for sequence_id in selected_ids if sequence_id not in by_id]
    if unknown:
        raise BatchPlanError(f"unknown sequence selection: {unknown}")

    actions: list[Action] = []
    for sequence_id in selected_ids:
        entry = by_id[sequence_id]
        video = entry.get("video")
        motion = entry.get("motion")
        camera = entry.get("camera")
        if (
            not isinstance(video, dict)
            or not isinstance(motion, dict)
            or not isinstance(camera, dict)
        ):
            raise BatchPlanError(
                f"manifest sequence {sequence_id} lacks video/motion/camera entries"
            )
        source_video = _manifest_path(
            video.get("path"), f"{sequence_id}.video.path", kind="file"
        )
        motion_parquet = _manifest_path(
            motion.get("path"), f"{sequence_id}.motion.path", kind="path"
        )
        geometry = {
            key: video.get(key) for key in ("frame_count", "width", "height", "fps")
        }
        _video_geometry(
            geometry, label=f"manifest sequence {sequence_id} video geometry"
        )
        intrinsic = _manifest_path(
            camera.get("intrinsic"), f"{sequence_id}.camera.intrinsic", kind="path"
        )
        world_to_camera = _manifest_path(
            camera.get("extrinsic"), f"{sequence_id}.camera.extrinsic", kind="path"
        )
        for label, camera_path in (
            ("intrinsic", intrinsic),
            ("extrinsic", world_to_camera),
        ):
            try:
                camera_path.relative_to(camera_root)
            except ValueError as exc:
                raise BatchPlanError(
                    f"{sequence_id} camera {label} escapes official camera root {camera_root}"
                ) from exc
        paths = _sequence_paths(
            run_root,
            sequence_id,
            source_video,
            motion_parquet,
            intrinsic,
            world_to_camera,
        )

        shared_gt_errors: list[str] = []
        ground_truth = (entry.get("conditions") or {}).get("ground_truth")
        if not isinstance(ground_truth, dict):
            shared_gt_errors.append("manifest lacks the ground_truth condition")
        else:
            blockers = ground_truth.get("blockers")
            if blockers:
                shared_gt_errors.append(f"manifest ground_truth blockers: {blockers}")
        if camera.get("available") is not True:
            shared_gt_errors.append("official camera entry is not marked available")
        try:
            load_taco_camera(
                paths.intrinsic,
                paths.world_to_camera,
                expected_frames=int(geometry["frame_count"]),
                width=int(geometry["width"]),
                height=int(geometry["height"]),
            )
        except (OSError, ValueError) as exc:
            shared_gt_errors.append(f"invalid official TACO camera: {exc}")

        render_errors = [*global_render_errors, *shared_gt_errors]
        if not paths.trajectory.is_file():
            render_errors.append(f"missing GT robot trajectory: {paths.trajectory}")
        else:
            try:
                validate_robot_trajectory_file(
                    paths.trajectory,
                    expected_frames=int(geometry["frame_count"]),
                )
            except (OSError, ValueError) as exc:
                render_errors.append(f"invalid GT robot trajectory: {exc}")
        render_complete, render_reason = _render_complete(paths, geometry)
        render_status, render_status_reason = _classify(
            complete=render_complete,
            completion_reason=render_reason,
            outputs_exist=_any_stage_output_exists(paths, "render"),
            prerequisite_errors=render_errors,
            overwrite=options.overwrite if "render" in stages else False,
        )
        render_future = render_complete or (
            "render" in stages and render_status in PENDING_STATES
        )
        if "render" in stages:
            actions.append(
                Action(
                    sequence_id=sequence_id,
                    stage="render",
                    status=render_status,
                    reason=(
                        render_status_reason
                        + (
                            "; full command withheld until an explicit --gpu is supplied"
                            if render_status in PENDING_STATES and options.gpu is None
                            else ""
                        )
                    ),
                    command=(
                        _render_command(
                            paths,
                            geometry,
                            asset_root=asset_root,
                            scene_utils_root=scene_utils_root,
                            repository_root=repository_root,
                            python_executable=python_executable,
                            options=options,
                        )
                        if render_status in PENDING_STATES and options.gpu is not None
                        else None
                    ),
                    inputs=(
                        paths.trajectory,
                        paths.intrinsic,
                        paths.world_to_camera,
                        asset_root,
                        scene_utils_root,
                    ),
                    outputs=_expected_artifact_files(paths, "render"),
                )
            )

        object_errors = [*global_object_errors, *shared_gt_errors]
        if not paths.motion_parquet.is_file():
            object_errors.append(f"missing TACO motion parquet: {paths.motion_parquet}")
        object_complete, object_reason = _object_depth_complete(
            paths,
            geometry,
            renderer_image=options.renderer_image,
            repository_root=repository_root,
        )
        object_status, object_status_reason = _classify(
            complete=object_complete,
            completion_reason=object_reason,
            outputs_exist=_any_stage_output_exists(paths, "object_depth"),
            prerequisite_errors=object_errors,
            overwrite=options.overwrite if "object_depth" in stages else False,
        )
        object_future = object_complete or (
            "object_depth" in stages and object_status in PENDING_STATES
        )
        if "object_depth" in stages:
            actions.append(
                Action(
                    sequence_id=sequence_id,
                    stage="object_depth",
                    status=object_status,
                    reason=(
                        object_status_reason
                        + (
                            "; full command withheld until an explicit --gpu is supplied"
                            if object_status in PENDING_STATES and options.gpu is None
                            else ""
                        )
                    ),
                    command=(
                        _object_depth_command(
                            paths,
                            object_mesh_root=object_mesh_root,
                            repository_root=repository_root,
                            python_executable=python_executable,
                            options=options,
                        )
                        if object_status in PENDING_STATES and options.gpu is not None
                        else None
                    ),
                    inputs=(
                        paths.motion_parquet,
                        paths.source_video,
                        paths.intrinsic,
                        paths.world_to_camera,
                        object_mesh_root,
                    ),
                    outputs=_expected_artifact_files(paths, "object_depth"),
                )
            )

        composite_errors = _validate_inpaint(paths, geometry)
        if not render_future:
            composite_errors.append(
                "robot render is neither complete nor scheduled successfully in this plan"
            )
        hard_fallback = not object_future and options.allow_hard_composite
        use_object_depth = not hard_fallback
        if not object_future and not options.allow_hard_composite:
            composite_errors.append(
                "object-depth render is neither complete nor scheduled successfully in this plan"
            )
        composite_complete, composite_reason = _composite_complete(
            paths,
            geometry,
            options,
            use_object_depth=use_object_depth,
        )
        composite_status, composite_status_reason = _classify(
            complete=composite_complete,
            completion_reason=composite_reason,
            outputs_exist=_any_stage_output_exists(paths, "composite"),
            prerequisite_errors=composite_errors,
            overwrite=options.overwrite if "composite" in stages else False,
        )
        composite_future = composite_complete or (
            "composite" in stages and composite_status in PENDING_STATES
        )
        if "composite" in stages:
            actions.append(
                Action(
                    sequence_id=sequence_id,
                    stage="composite",
                    status=composite_status,
                    reason=(
                        composite_status_reason
                        + (
                            "; HARD-MASK FALLBACK explicitly allowed: object contacts may "
                            "be physically incorrect"
                            if hard_fallback
                            else ""
                        )
                    ),
                    command=(
                        _composite_command(
                            paths,
                            python_executable=python_executable,
                            options=options,
                            use_object_depth=use_object_depth,
                        )
                        if composite_status in PENDING_STATES
                        else None
                    ),
                    inputs=(
                        paths.inpaint_video,
                        paths.robot_rgb,
                        paths.robot_mask,
                        paths.robot_depth,
                        paths.robot_metadata,
                        *(
                            (
                                paths.object_mask,
                                paths.object_depth,
                                paths.object_metadata,
                            )
                            if use_object_depth
                            else ()
                        ),
                    ),
                    outputs=_expected_artifact_files(paths, "composite"),
                )
            )

        grid_errors = _validate_inpaint(paths, geometry)
        if not source_video.is_file():
            grid_errors.append(f"missing source video: {source_video}")
        if not composite_future:
            grid_errors.append(
                "composite is neither complete nor scheduled successfully in this plan"
            )
        comparison_label = (
            "GT Vega + Sharpa (HARD-MASK FALLBACK)"
            if hard_fallback
            else "GT Vega + Sharpa"
        )
        grid_complete, grid_reason = _grid_complete(
            paths,
            geometry,
            options,
            comparison_label=comparison_label,
        )
        grid_status, grid_status_reason = _classify(
            complete=grid_complete,
            completion_reason=grid_reason,
            outputs_exist=_any_stage_output_exists(paths, "grid"),
            prerequisite_errors=grid_errors,
            overwrite=options.overwrite if "grid" in stages else False,
        )
        if "grid" in stages:
            actions.append(
                Action(
                    sequence_id=sequence_id,
                    stage="grid",
                    status=grid_status,
                    reason=grid_status_reason,
                    command=(
                        _grid_command(
                            paths,
                            python_executable=python_executable,
                            options=options,
                            comparison_label=comparison_label,
                        )
                        if grid_status in PENDING_STATES
                        else None
                    ),
                    inputs=(source_video, paths.inpaint_video, paths.composite_video),
                    outputs=_expected_artifact_files(paths, "grid"),
                )
            )

    counts: dict[str, int] = {}
    for action in actions:
        counts[action.status] = counts.get(action.status, 0) + 1
    return {
        "schema_version": PLAN_SCHEMA,
        "created_at": _utc_now(),
        "mode": "plan",
        "manifest": str(manifest_path),
        "experiment_id": manifest.get("experiment_id"),
        "run_root": str(run_root),
        "repository_root": str(repository_root),
        "python_executable": str(python_executable),
        "asset_root": str(asset_root),
        "scene_utils_root": str(scene_utils_root),
        "object_mesh_root": str(object_mesh_root),
        "selected_sequences": list(selected_ids),
        "selected_stages": list(stages),
        "gpu": options.gpu,
        "renderer_image": options.renderer_image,
        "overwrite": options.overwrite,
        "depth_guard_m": options.depth_guard_m,
        "allow_hard_composite": options.allow_hard_composite,
        "summary": counts,
        "actions": [action.as_dict() for action in actions],
    }


def _write_grid_metadata(action: dict[str, Any], options: PlanOptions) -> None:
    output_video, metadata_path = (Path(value) for value in action["outputs"])
    geometry = probe_video(output_video)
    paths = [Path(value) for value in action["inputs"]]
    command = action.get("command")
    if not isinstance(command, list):
        raise BatchPlanError("grid action lacks its executed command")
    labels = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--label"
    ]
    if len(labels) != len(paths):
        raise BatchPlanError("grid command label count does not match its inputs")
    specification = {
        "videos": [str(path) for path in paths],
        "labels": labels,
        "tile_width": options.grid_tile_width,
        "columns": options.grid_columns,
        "max_frames": options.grid_max_frames,
    }
    payload = {
        "schema_version": GRID_SCHEMA,
        "state": "complete",
        "completed_at": _utc_now(),
        "specification": specification,
        "output_video": str(output_video),
        "geometry": geometry.as_dict(),
    }
    temporary = metadata_path.with_name(
        f"{metadata_path.stem}.partial{metadata_path.suffix}"
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(metadata_path)


def execute_plan(
    plan: dict[str, Any],
    options: PlanOptions,
    *,
    run_command=subprocess.run,
) -> dict[str, Any]:
    """Execute pending plan actions sequentially and stop on the first failure."""

    blocked = [action for action in plan["actions"] if action["status"] == "blocked"]
    if blocked:
        labels = [
            f"{item['sequence_id']}/{item['stage']}: {item['reason']}"
            for item in blocked
        ]
        raise BatchPlanError(
            "refusing execution with blocked actions: " + " | ".join(labels)
        )
    pending = [
        action for action in plan["actions"] if action["status"] in PENDING_STATES
    ]
    pending_gpu = [action for action in pending if action["stage"] in GPU_STAGES]
    if pending_gpu and options.gpu is None:
        raise BatchPlanError(
            "full GPU renderer actions require an explicit --gpu together with --execute"
        )
    if (
        any(action["status"] == "pending_overwrite" for action in pending)
        and not options.overwrite
    ):
        raise BatchPlanError(
            "plan requests overwrite actions but execution options do not"
        )

    executed: list[dict[str, Any]] = []
    repository_root = Path(plan["repository_root"])
    for action in pending:
        if not options.overwrite:
            outputs = [Path(value) for value in action["outputs"]]
            collision = [
                path
                for path in (*outputs, *(_partial_path(path) for path in outputs))
                if path.exists()
            ]
            if collision:
                raise BatchPlanError(
                    f"refusing to overwrite outputs that appeared after planning for "
                    f"{action['sequence_id']}/{action['stage']}: "
                    + ", ".join(str(path) for path in collision)
                )
        command = action.get("command")
        if not isinstance(command, list) or not command:
            raise BatchPlanError(
                f"pending action {action['sequence_id']}/{action['stage']} has no command"
            )
        print(f"EXEC {shlex.join(command)}", flush=True)
        completed = run_command(command, cwd=repository_root, check=False)
        return_code = int(completed.returncode)
        if return_code != 0:
            raise RuntimeError(
                f"{action['sequence_id']}/{action['stage']} failed with exit {return_code}"
            )
        if action["stage"] == "grid":
            _write_grid_metadata(action, options)
        executed.append(
            {
                "sequence_id": action["sequence_id"],
                "stage": action["stage"],
                "return_code": return_code,
            }
        )

    # Rebuild from disk so success means every selected output now satisfies
    # the same strict resume contract used by a future invocation.
    verified = build_plan(replace(options, overwrite=False))
    not_complete = [
        action
        for action in verified["actions"]
        if action["status"] not in ("skipped_complete",)
    ]
    if not_complete:
        labels = [
            f"{item['sequence_id']}/{item['stage']}={item['status']}: {item['reason']}"
            for item in not_complete
        ]
        raise RuntimeError(
            "post-execution artifact verification failed: " + " | ".join(labels)
        )
    return {
        "schema_version": "v2d.inpainting.gt-batch-execution/v1",
        "state": "complete",
        "completed_at": _utc_now(),
        "manifest": plan["manifest"],
        "executed": executed,
        "resumed": [
            {
                "sequence_id": action["sequence_id"],
                "stage": action["stage"],
            }
            for action in plan["actions"]
            if action["status"] == "skipped_complete"
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--sequence", action="append", dest="sequences")
    parser.add_argument("--stage", action="append", choices=STAGES, dest="stages")
    parser.add_argument("--scene-utils-root", type=Path)
    parser.add_argument("--object-mesh-root", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--renderer-image", default="robotic-grounding:photo-render-v6")
    parser.add_argument("--gpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--depth-guard-m", type=float, default=0.003)
    parser.add_argument(
        "--allow-hard-composite",
        action="store_true",
        help=(
            "Allow an explicitly labelled hard-mask fallback when no valid/scheduled "
            "object-depth bundle exists."
        ),
    )
    parser.add_argument("--grid-tile-width", type=int, default=640)
    parser.add_argument("--grid-columns", type=int, default=3)
    parser.add_argument("--grid-max-frames", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute pending stages; default prints a read-only plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = PlanOptions(
        manifest_path=args.manifest,
        run_root=args.run_root,
        sequence_ids=tuple(args.sequences or ()),
        stages=tuple(args.stages or STAGES),
        scene_utils_root=args.scene_utils_root,
        object_mesh_root=args.object_mesh_root,
        repository_root=args.repository_root,
        python_executable=args.python_executable,
        gpu=args.gpu,
        renderer_image=args.renderer_image,
        overwrite=args.overwrite,
        depth_guard_m=args.depth_guard_m,
        allow_hard_composite=args.allow_hard_composite,
        grid_tile_width=args.grid_tile_width,
        grid_columns=args.grid_columns,
        grid_max_frames=args.grid_max_frames,
    )
    try:
        plan = build_plan(options)
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        result = execute_plan(plan, options)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (BatchPlanError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
