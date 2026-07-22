"""Plan or execute final overlays for learned hand-tracking conditions.

The default is a read-only JSON plan.  ``--execute`` is required to launch
subprocesses, and an exact Docker GPU device selector is additionally required
only when a robot-render action is pending.  Each learned condition consumes
its own ``tracking/robot_trajectory.npz``, the sequence's shared E2FGVI video,
and the already validated ground-truth TACO object-depth bundle.

Completed bundles are resumed through the same strict validators used by the
ground-truth batch.  Existing incomplete or crash-partial outputs block unless
``--overwrite`` is explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

from .contracts import RESOLVED_EXPERIMENT_SCHEMA, validate_robot_trajectory_file
from .robot_renderer.container_runner import (
    resolve_local_image_id,
    validate_gpu_selector,
)
from .robot_renderer.enrich_metadata import enrich_render_metadata
from .robot_renderer.inputs import load_render_inputs
from .robot_renderer.provenance import PROVENANCE_SCHEMA, sha256_file
from .run_ground_truth_batch import (
    BatchPlanError,
    PENDING_STATES,
    _classify,
    _composite_complete,
    _load_json,
    _manifest_path,
    _object_depth_complete,
    _render_complete,
    _utc_now,
    _validate_inpaint,
    _video_geometry,
)
from .taco_camera import load_taco_camera
from .video_io import probe_video


PLAN_SCHEMA = "v2d.inpainting.learned-condition-batch-plan/v1"
EXECUTION_SCHEMA = "v2d.inpainting.learned-condition-batch-execution/v1"
GRID_SCHEMA = "v2d.inpainting.learned-comparison-grid/v1"
CONDITIONS = ("v2d", "phantom")
STAGES = ("render", "composite", "grid")
GPU_STAGES = frozenset(("render",))


@dataclass(frozen=True)
class SequencePaths:
    sequence_id: str
    condition: str
    source_video: Path
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
    condition: str
    stage: str
    status: str
    reason: str
    command: tuple[str, ...] | None
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "condition": self.condition,
            "stage": self.stage,
            "status": self.status,
            "reason": self.reason,
            "command": list(self.command) if self.command is not None else None,
            "command_shell": (
                shlex.join(self.command) if self.command is not None else None
            ),
            "inputs": [str(path) for path in self.inputs],
            "outputs": [str(path) for path in self.outputs],
        }


@dataclass(frozen=True)
class PlanOptions:
    manifest_path: Path
    run_root: Path | None = None
    sequence_ids: tuple[str, ...] = ()
    conditions: tuple[str, ...] = CONDITIONS
    stages: tuple[str, ...] = STAGES
    scene_utils_root: Path | None = None
    repository_root: Path | None = None
    python_executable: Path | None = None
    gpu: str | None = None
    renderer_image: str = "robotic-grounding:photo-render-v6"
    max_ik_residual_m: float = 0.01
    max_joint_step_rad: float = 0.4
    overwrite: bool = False
    depth_guard_m: float = 0.003
    grid_tile_width: int = 640
    grid_columns: int = 3
    grid_max_frames: int | None = None


def _expected_artifact_files(paths: SequencePaths, stage: str) -> tuple[Path, ...]:
    if stage == "render":
        return (
            paths.robot_rgb,
            paths.robot_mask,
            paths.robot_depth,
            paths.robot_metadata,
        )
    if stage == "composite":
        return (paths.composite_video, paths.composite_metadata)
    if stage == "grid":
        return (paths.grid_video, paths.grid_metadata)
    raise AssertionError(stage)


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def _partial_paths_for_outputs(outputs: tuple[Path, ...]) -> tuple[Path, ...]:
    found: set[Path] = set()
    for path in outputs:
        conventional = _partial_path(path)
        if conventional.exists() or conventional.is_symlink():
            found.add(conventional)
        if path.parent.is_dir():
            # Composite JSON uses .<name>.<uuid>.partial; composite video uses
            # .<stem>.<run-id>.partial<suffix>.  Match both without sweeping
            # unrelated partials from another stage.
            patterns = (
                f".{path.name}.*.partial",
                f".{path.stem}.*.partial{path.suffix}",
            )
            for pattern in patterns:
                found.update(candidate for candidate in path.parent.glob(pattern))
    return tuple(sorted(found, key=str))


def _stage_partial_paths(paths: SequencePaths, stage: str) -> tuple[Path, ...]:
    return _partial_paths_for_outputs(_expected_artifact_files(paths, stage))


def _stage_collisions(paths: SequencePaths, stage: str) -> tuple[Path, ...]:
    """Return final and recognized crash-partial paths that currently exist."""

    found = {
        path
        for path in _expected_artifact_files(paths, stage)
        if path.exists() or path.is_symlink()
    }
    found.update(_stage_partial_paths(paths, stage))
    return tuple(sorted(found, key=str))


def _classify_stage(
    *,
    paths: SequencePaths,
    stage: str,
    complete: bool,
    completion_reason: str,
    prerequisite_errors: list[str],
    overwrite: bool,
) -> tuple[str, str]:
    partials = _stage_partial_paths(paths, stage)
    if partials and not overwrite:
        errors = [*prerequisite_errors]
        errors.append(
            "stale partial outputs exist; inspect or pass --overwrite: "
            + ", ".join(str(path) for path in partials)
        )
        return "blocked", "; ".join(errors)
    return _classify(
        complete=complete,
        completion_reason=completion_reason,
        outputs_exist=bool(_stage_collisions(paths, stage)),
        prerequisite_errors=prerequisite_errors,
        overwrite=overwrite,
    )


def _sequence_paths(
    run_root: Path,
    sequence_id: str,
    condition: str,
    source_video: Path,
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
    condition_root = sequence_root / condition
    tracking = condition_root / "tracking"
    shared_arm_mask = sequence_root / "shared_arm_mask"
    shared_inpaint = sequence_root / "shared_inpaint"
    robot = condition_root / "robot_render"
    # Object geometry is tracker-independent and is intentionally reused from
    # the validated GT condition instead of being rendered once per tracker.
    objects = sequence_root / "ground_truth" / "object_render"
    return SequencePaths(
        sequence_id=sequence_id,
        condition=condition,
        source_video=source_video,
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
        composite_video=(condition_root / "final_overlay.mp4").resolve(),
        composite_metadata=(condition_root / "final_overlay.json").resolve(),
        grid_video=(condition_root / "final_comparison_grid.mp4").resolve(),
        grid_metadata=(condition_root / "final_comparison_grid.json").resolve(),
    )


def _condition_label(condition: str) -> str:
    return f"{'Video2Data' if condition == 'v2d' else 'Phantom'} Vega + Sharpa"


def _effective_renderer_float(value: float) -> float:
    """Return the exact float the nested renderer CLI receives."""

    return float(f"{value:.12g}")


def _render_policy(options: PlanOptions) -> dict[str, float]:
    return {
        "max_position_residual_m": _effective_renderer_float(
            options.max_ik_residual_m
        ),
        "max_joint_step_rad": _effective_renderer_float(options.max_joint_step_rad),
    }


def _learned_render_complete(
    paths: SequencePaths,
    geometry: dict[str, Any],
    options: PlanOptions,
    *,
    asset_root: Path,
    scene_utils_root: Path,
    repository_root: Path,
) -> tuple[bool, str]:
    complete, reason = _render_complete(paths, geometry)
    if not complete:
        return complete, reason
    try:
        metadata = _load_json(paths.robot_metadata)
    except (BatchPlanError, FileNotFoundError) as exc:
        return False, str(exc)
    selected = _render_policy(options)
    if metadata.get("kinematics_policy") != selected:
        return False, (
            "robot render kinematics policy differs from the selected policy: "
            f"metadata={metadata.get('kinematics_policy')!r}, selected={selected!r}"
        )
    try:
        _verify_current_render_generation(
            paths=paths,
            options=options,
            asset_root=asset_root,
            scene_utils_root=scene_utils_root,
            repository_root=repository_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"robot render provenance is stale or invalid: {exc}"
    return (
        True,
        reason
        + "; selected kinematics policy and current input/image/source provenance "
        "validated",
    )


def _verify_current_render_generation(
    *,
    paths: SequencePaths,
    options: PlanOptions,
    asset_root: Path,
    scene_utils_root: Path,
    repository_root: Path,
) -> None:
    """Bind resume to the exact current inputs, image, implementation, and assets."""

    image_id = resolve_local_image_id(options.renderer_image)
    on_disk = _load_json(paths.robot_metadata)
    if on_disk.get("container_image_id") != image_id:
        raise BatchPlanError(
            "render sidecar does not contain the currently resolved immutable image ID"
        )
    provenance = on_disk.get("provenance")
    if not isinstance(provenance, dict) or provenance.get(
        "schema_version"
    ) != PROVENANCE_SCHEMA:
        raise BatchPlanError("render sidecar lacks committed renderer provenance")
    provenance_inputs = provenance.get("inputs")
    if not isinstance(provenance_inputs, dict) or set(provenance_inputs) != {
        "trajectory",
        "intrinsic",
        "world_to_camera",
    }:
        raise BatchPlanError("render sidecar lacks exact input provenance records")
    source_records = provenance.get("renderer_source_files")
    if not isinstance(source_records, list) or not source_records:
        raise BatchPlanError("render sidecar lacks renderer source provenance records")
    artifact_hashes = on_disk.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != {
        "rgb",
        "mask",
        "depth",
    }:
        raise BatchPlanError("render sidecar lacks exact artifact SHA-256 records")
    assets = on_disk.get("assets")
    if not isinstance(assets, dict):
        raise BatchPlanError("render sidecar lacks robot asset provenance")
    for part in ("arms", "left_hand", "right_hand"):
        record = assets.get(part)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("urdf_file"), dict)
            or not isinstance(record.get("referenced_asset_files"), list)
            or not record["referenced_asset_files"]
        ):
            raise BatchPlanError(
                f"render sidecar lacks committed {part} URDF/mesh provenance"
            )
    metadata = enrich_render_metadata(
        metadata_path=paths.robot_metadata,
        trajectory=paths.trajectory,
        intrinsic=paths.intrinsic,
        world_to_camera=paths.world_to_camera,
        asset_root=asset_root,
        repository_root=repository_root,
        image=options.renderer_image,
        image_id=image_id,
        write=False,
    )
    kinematics = metadata.get("kinematics")
    external_sources = (
        kinematics.get("external_sources") if isinstance(kinematics, dict) else None
    )
    if not isinstance(external_sources, dict):
        raise BatchPlanError("render metadata lacks external IK source provenance")
    for filename, metadata_key in (
        ("arm_ik.py", "arm_ik_sha256"),
        ("arm_mount_opt.py", "arm_mount_opt_sha256"),
    ):
        source = scene_utils_root / filename
        if not source.is_file():
            raise BatchPlanError(f"current external IK source is missing: {source}")
        actual = sha256_file(source)
        recorded = external_sources.get(metadata_key)
        if recorded != actual:
            raise BatchPlanError(
                f"external IK source {filename} SHA-256 differs from the render: "
                f"recorded={recorded!r}, current={actual!r}"
            )


def _grid_spec(paths: SequencePaths, options: PlanOptions) -> dict[str, Any]:
    return {
        "videos": [
            str(paths.source_video),
            str(paths.inpaint_video),
            str(paths.composite_video),
        ],
        "labels": ["Source", "E2FGVI", _condition_label(paths.condition)],
        "tile_width": options.grid_tile_width,
        "columns": options.grid_columns,
        "max_frames": options.grid_max_frames,
    }


def _grid_complete(
    paths: SequencePaths, geometry: dict[str, Any], options: PlanOptions
) -> tuple[bool, str]:
    required_inputs = (paths.source_video, paths.inpaint_video, paths.composite_video)
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        return False, f"missing grid inputs: {missing_inputs}"
    missing = [
        path.name
        for path in _expected_artifact_files(paths, "grid")
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        return False, f"missing/empty grid artifacts: {missing}"
    try:
        metadata = _load_json(paths.grid_metadata)
    except (BatchPlanError, FileNotFoundError) as exc:
        return False, str(exc)
    if metadata.get("schema_version") != GRID_SCHEMA or metadata.get("state") != "complete":
        return False, "grid sidecar is not a complete learned-comparison grid run"
    if metadata.get("condition") != paths.condition:
        return False, "grid sidecar condition differs from this run"
    if metadata.get("specification") != _grid_spec(paths, options):
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
        path.stat().st_mtime_ns for path in required_inputs
    ):
        return False, "grid video predates one or more inputs"
    if paths.grid_metadata.stat().st_mtime_ns < paths.grid_video.stat().st_mtime_ns:
        return False, "grid metadata predates the grid video"
    return True, "complete learned comparison grid sidecar and artifact validated"


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
        "--max-ik-residual-m",
        f"{options.max_ik_residual_m:.12g}",
        "--max-joint-step-rad",
        f"{options.max_joint_step_rad:.12g}",
        "--execute",
        "--gpu",
        options.gpu or "",
    ]
    if options.overwrite:
        command.append("--overwrite")
    return tuple(command)


def _composite_command(
    paths: SequencePaths,
    *,
    python_executable: Path,
    options: PlanOptions,
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
        "--object-mask",
        str(paths.object_mask),
        "--object-depth",
        str(paths.object_depth),
        "--object-metadata",
        str(paths.object_metadata),
        "--depth-guard-m",
        f"{options.depth_guard_m:.12g}",
        "--output-video",
        str(paths.composite_video),
        "--metadata",
        str(paths.composite_metadata),
    ]
    if options.overwrite:
        command.append("--overwrite")
    return tuple(command)


def _grid_command(
    paths: SequencePaths,
    *,
    python_executable: Path,
    options: PlanOptions,
) -> tuple[str, ...]:
    command = [str(python_executable), "-m", "inpainting.make_video_grid"]
    specification = _grid_spec(paths, options)
    for video, label in zip(
        specification["videos"], specification["labels"], strict=True
    ):
        command.extend(("--video", video, "--label", label))
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


def _validated_selection(
    values: tuple[str, ...], allowed: tuple[str, ...], label: str
) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(values))
    if not selected or any(value not in allowed for value in selected):
        raise BatchPlanError(
            f"{label} must be a non-empty subset of {allowed}, got {selected}"
        )
    return tuple(value for value in allowed if value in selected)


def build_plan(options: PlanOptions) -> dict[str, Any]:
    """Build a fully resolved plan without creating or modifying any files."""

    manifest_path = options.manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != RESOLVED_EXPERIMENT_SCHEMA:
        raise BatchPlanError(
            f"manifest schema must be {RESOLVED_EXPERIMENT_SCHEMA!r}, got "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("robot") != "dexmate_vega" or manifest.get("gripper") != "sharpa_wave":
        raise BatchPlanError(
            "learned-condition batch supports only dexmate_vega + sharpa_wave"
        )

    run_root = (options.run_root or manifest_path.parent).expanduser().resolve()
    repository_root = (
        options.repository_root or Path(__file__).resolve().parents[1]
    ).expanduser().resolve()
    if not repository_root.is_dir():
        raise BatchPlanError(f"repository root does not exist: {repository_root}")
    python_executable = (
        options.python_executable or Path(sys.executable)
    ).expanduser().resolve()
    if not python_executable.is_file():
        raise BatchPlanError(f"Python executable does not exist: {python_executable}")

    conditions = _validated_selection(options.conditions, CONDITIONS, "conditions")
    stages = _validated_selection(options.stages, STAGES, "stages")
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
    if (
        isinstance(options.max_ik_residual_m, bool)
        or not np.isfinite(options.max_ik_residual_m)
        or options.max_ik_residual_m <= 0.0
    ):
        raise BatchPlanError("max IK residual must be a finite positive number")
    if (
        isinstance(options.max_joint_step_rad, bool)
        or not np.isfinite(options.max_joint_step_rad)
        or options.max_joint_step_rad <= 0.0
    ):
        raise BatchPlanError("max joint step must be a finite positive number")

    roots = manifest.get("roots")
    if not isinstance(roots, dict):
        raise BatchPlanError("manifest lacks roots")
    asset_root = _manifest_path(
        roots.get("robot_assets"), "roots.robot_assets", kind="path"
    )
    camera_root = _manifest_path(roots.get("camera"), "roots.camera", kind="dir")
    scene_utils_root = (
        options.scene_utils_root or asset_root.parent / "tasks" / "scene_utils"
    ).expanduser().resolve()
    global_render_errors: list[str] = []
    if not asset_root.is_dir():
        global_render_errors.append(f"robot asset directory does not exist: {asset_root}")
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

    sequence_entries = manifest.get("sequences")
    if not isinstance(sequence_entries, list) or not sequence_entries:
        raise BatchPlanError("manifest sequences must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in sequence_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("sequence_id"), str):
            raise BatchPlanError("each manifest sequence must have a string sequence_id")
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
    object_inputs: dict[str, dict[str, Any]] = {}
    for sequence_id in selected_ids:
        entry = by_id[sequence_id]
        video = entry.get("video")
        camera = entry.get("camera")
        condition_entries = entry.get("conditions")
        if not isinstance(video, dict) or not isinstance(camera, dict):
            raise BatchPlanError(
                f"manifest sequence {sequence_id} lacks video/camera entries"
            )
        if not isinstance(condition_entries, dict):
            raise BatchPlanError(
                f"manifest sequence {sequence_id} lacks conditions"
            )
        source_video = _manifest_path(
            video.get("path"), f"{sequence_id}.video.path", kind="file"
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
        camera_errors: list[str] = []
        for label, camera_path in (
            ("intrinsic", intrinsic),
            ("extrinsic", world_to_camera),
        ):
            try:
                camera_path.relative_to(camera_root)
            except ValueError:
                camera_errors.append(
                    f"{sequence_id} camera {label} escapes official camera root "
                    f"{camera_root}"
                )
        if camera.get("available") is not True:
            camera_errors.append("official camera entry is not marked available")
        try:
            load_taco_camera(
                intrinsic,
                world_to_camera,
                expected_frames=int(geometry["frame_count"]),
                width=int(geometry["width"]),
                height=int(geometry["height"]),
            )
        except (OSError, ValueError) as exc:
            camera_errors.append(f"invalid official TACO camera: {exc}")

        # Validate the shared GT object bundle once per sequence; all selected
        # learned conditions consume these exact paths.
        representative = _sequence_paths(
            run_root,
            sequence_id,
            conditions[0],
            source_video,
            intrinsic,
            world_to_camera,
        )
        object_complete, object_reason = _object_depth_complete(
            representative, geometry
        )
        object_inputs[sequence_id] = {
            "state": "complete" if object_complete else "invalid",
            "reason": object_reason,
            "metadata": str(representative.object_metadata),
        }

        for condition in conditions:
            paths = _sequence_paths(
                run_root,
                sequence_id,
                condition,
                source_video,
                intrinsic,
                world_to_camera,
            )
            condition_errors = list(camera_errors)
            condition_entry = condition_entries.get(condition)
            if not isinstance(condition_entry, dict):
                condition_errors.append(
                    f"manifest lacks the {condition!r} condition"
                )
            else:
                if condition_entry.get("tracker") != condition:
                    condition_errors.append(
                        f"manifest {condition!r} tracker does not match its condition"
                    )
                blockers = condition_entry.get("blockers")
                if blockers:
                    condition_errors.append(
                        f"manifest {condition} blockers: {blockers}"
                    )

            render_errors = [*global_render_errors, *condition_errors]
            if not paths.trajectory.is_file():
                render_errors.append(
                    f"missing {condition} robot trajectory: {paths.trajectory}"
                )
            else:
                try:
                    validate_robot_trajectory_file(
                        paths.trajectory,
                        expected_frames=int(geometry["frame_count"]),
                    )
                    load_render_inputs(
                        trajectory_path=paths.trajectory,
                        intrinsic_path=paths.intrinsic,
                        world_to_camera_path=paths.world_to_camera,
                        width=int(geometry["width"]),
                        height=int(geometry["height"]),
                        fps=float(geometry["fps"]),
                    )
                except (OSError, ValueError) as exc:
                    render_errors.append(
                        f"trajectory cannot enter the strict renderer: {exc}"
                    )
            render_complete, render_reason = _learned_render_complete(
                paths,
                geometry,
                options,
                asset_root=asset_root,
                scene_utils_root=scene_utils_root,
                repository_root=repository_root,
            )
            render_status, render_status_reason = _classify_stage(
                paths=paths,
                stage="render",
                complete=render_complete,
                completion_reason=render_reason,
                prerequisite_errors=render_errors,
                overwrite=options.overwrite if "render" in stages else False,
            )
            render_future = render_status == "skipped_complete" or (
                "render" in stages and render_status in PENDING_STATES
            )
            if "render" in stages:
                actions.append(
                    Action(
                        sequence_id=sequence_id,
                        condition=condition,
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
                            if render_status in PENDING_STATES
                            and options.gpu is not None
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

            composite_errors = [*_validate_inpaint(paths, geometry)]
            if not render_future:
                composite_errors.append(
                    "robot render is neither complete nor scheduled successfully "
                    "in this plan"
                )
            if not object_complete:
                composite_errors.append(
                    "validated ground-truth object-depth bundle is unavailable: "
                    + object_reason
                )
            composite_complete, composite_reason = _composite_complete(
                paths, geometry, options, use_object_depth=True
            )
            composite_status, composite_status_reason = _classify_stage(
                paths=paths,
                stage="composite",
                complete=composite_complete,
                completion_reason=composite_reason,
                prerequisite_errors=composite_errors,
                overwrite=options.overwrite if "composite" in stages else False,
            )
            composite_future = composite_status == "skipped_complete" or (
                "composite" in stages and composite_status in PENDING_STATES
            )
            if "composite" in stages:
                actions.append(
                    Action(
                        sequence_id=sequence_id,
                        condition=condition,
                        stage="composite",
                        status=composite_status,
                        reason=composite_status_reason,
                        command=(
                            _composite_command(
                                paths,
                                python_executable=python_executable,
                                options=options,
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
                            paths.object_mask,
                            paths.object_depth,
                            paths.object_metadata,
                        ),
                        outputs=_expected_artifact_files(paths, "composite"),
                    )
                )

            grid_errors = _validate_inpaint(paths, geometry)
            if not composite_future:
                grid_errors.append(
                    "composite is neither complete nor scheduled successfully in this plan"
                )
            grid_complete, grid_reason = _grid_complete(paths, geometry, options)
            grid_status, grid_status_reason = _classify_stage(
                paths=paths,
                stage="grid",
                complete=grid_complete,
                completion_reason=grid_reason,
                prerequisite_errors=grid_errors,
                overwrite=options.overwrite if "grid" in stages else False,
            )
            if "grid" in stages:
                actions.append(
                    Action(
                        sequence_id=sequence_id,
                        condition=condition,
                        stage="grid",
                        status=grid_status,
                        reason=grid_status_reason,
                        command=(
                            _grid_command(
                                paths,
                                python_executable=python_executable,
                                options=options,
                            )
                            if grid_status in PENDING_STATES
                            else None
                        ),
                        inputs=(
                            paths.source_video,
                            paths.inpaint_video,
                            paths.composite_video,
                        ),
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
        "selected_sequences": list(selected_ids),
        "selected_conditions": list(conditions),
        "selected_stages": list(stages),
        "gpu": options.gpu,
        "renderer_image": options.renderer_image,
        "kinematics_policy": _render_policy(options),
        "overwrite": options.overwrite,
        "depth_guard_m": options.depth_guard_m,
        "object_depth_inputs": object_inputs,
        "summary": counts,
        "actions": [action.as_dict() for action in actions],
    }


def _write_grid_metadata(action: dict[str, Any], options: PlanOptions) -> None:
    output_video, metadata_path = (Path(value) for value in action["outputs"])
    geometry = probe_video(output_video)
    command = action.get("command")
    if not isinstance(command, list):
        raise BatchPlanError("grid action lacks its executed command")
    labels = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--label"
    ]
    inputs = [Path(value) for value in action["inputs"]]
    if len(labels) != len(inputs):
        raise BatchPlanError("grid command label count does not match its inputs")
    payload = {
        "schema_version": GRID_SCHEMA,
        "state": "complete",
        "completed_at": _utc_now(),
        "condition": action["condition"],
        "specification": {
            "videos": [str(path) for path in inputs],
            "labels": labels,
            "tile_width": options.grid_tile_width,
            "columns": options.grid_columns,
            "max_frames": options.grid_max_frames,
        },
        "output_video": str(output_video),
        "geometry": geometry.as_dict(),
    }
    temporary = _partial_path(metadata_path)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(metadata_path)


def execute_plan(
    plan: dict[str, Any],
    options: PlanOptions,
    *,
    run_command=subprocess.run,
) -> dict[str, Any]:
    """Execute pending actions sequentially and verify strict resume state."""

    blocked = [action for action in plan["actions"] if action["status"] == "blocked"]
    if blocked:
        labels = [
            f"{item['sequence_id']}/{item['condition']}/{item['stage']}: "
            f"{item['reason']}"
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
            "pending robot-render actions require an explicit --gpu together with --execute"
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
        output_paths = tuple(Path(value) for value in action["outputs"])
        if not options.overwrite:
            collisions = {
                path for path in output_paths if path.exists() or path.is_symlink()
            }
            collisions.update(_partial_paths_for_outputs(output_paths))
            if collisions:
                raise BatchPlanError(
                    "refusing to overwrite outputs that appeared after planning for "
                    f"{action['sequence_id']}/{action['condition']}/{action['stage']}: "
                    + ", ".join(str(path) for path in sorted(collisions, key=str))
                )
        else:
            # Explicit overwrite authorizes removal of only the narrowly
            # recognized crash temporaries for this action.  Final artifacts
            # remain the responsibility of the underlying contract-aware tool.
            for partial in _partial_paths_for_outputs(output_paths):
                if partial.is_dir() and not partial.is_symlink():
                    raise BatchPlanError(
                        f"refusing to remove partial output directory: {partial}"
                    )
                partial.unlink(missing_ok=True)
        command = action.get("command")
        if not isinstance(command, list) or not command:
            raise BatchPlanError(
                f"pending action {action['sequence_id']}/{action['condition']}/"
                f"{action['stage']} has no command"
            )
        print(f"EXEC {shlex.join(command)}", flush=True)
        completed = run_command(command, cwd=repository_root, check=False)
        return_code = int(completed.returncode)
        if return_code != 0:
            raise RuntimeError(
                f"{action['sequence_id']}/{action['condition']}/{action['stage']} "
                f"failed with exit {return_code}"
            )
        if action["stage"] == "grid":
            _write_grid_metadata(action, options)
        executed.append(
            {
                "sequence_id": action["sequence_id"],
                "condition": action["condition"],
                "stage": action["stage"],
                "return_code": return_code,
            }
        )

    verified = build_plan(replace(options, overwrite=False))
    not_complete = [
        action
        for action in verified["actions"]
        if action["status"] != "skipped_complete"
    ]
    if not_complete:
        labels = [
            f"{item['sequence_id']}/{item['condition']}/{item['stage']}="
            f"{item['status']}: {item['reason']}"
            for item in not_complete
        ]
        raise RuntimeError(
            "post-execution artifact verification failed: " + " | ".join(labels)
        )
    return {
        "schema_version": EXECUTION_SCHEMA,
        "state": "complete",
        "completed_at": _utc_now(),
        "manifest": plan["manifest"],
        "executed": executed,
        "resumed": [
            {
                "sequence_id": action["sequence_id"],
                "condition": action["condition"],
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
    parser.add_argument(
        "--condition", action="append", choices=CONDITIONS, dest="conditions"
    )
    parser.add_argument("--stage", action="append", choices=STAGES, dest="stages")
    parser.add_argument("--scene-utils-root", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument(
        "--renderer-image", default="robotic-grounding:photo-render-v6"
    )
    parser.add_argument("--max-ik-residual-m", type=float, default=0.01)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.4)
    parser.add_argument("--gpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--depth-guard-m", type=float, default=0.003)
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
        conditions=tuple(args.conditions or CONDITIONS),
        stages=tuple(args.stages or STAGES),
        scene_utils_root=args.scene_utils_root,
        repository_root=args.repository_root,
        python_executable=args.python_executable,
        gpu=args.gpu,
        renderer_image=args.renderer_image,
        max_ik_residual_m=args.max_ik_residual_m,
        max_joint_step_rad=args.max_joint_step_rad,
        overwrite=args.overwrite,
        depth_guard_m=args.depth_guard_m,
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
