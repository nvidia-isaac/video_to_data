"""Plan or execute one five-panel parallel-jaw embodiment comparison.

The default mode is read-only.  ``--execute`` is required to launch the three
robot renders, depth-aware composites, and final five-panel grid.  All three
tracking conditions reuse the selected clip's completed ground-truth Vega
``kinematics.arm_center_world`` as one shared ``T_world_hub``.

The pipeline intentionally consumes the already committed shared inputs:

* ``parallel_jaw/targets/{ground_truth,v2d,phantom}``
* ``shared_inpaint/e2fgvi_960.mp4``
* ``ground_truth/object_render`` (with a 3 mm depth guard by default)

Existing complete artifacts are resumed only after their hashes, geometry,
policy, and lineage are validated.  Existing incomplete outputs block unless
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
from typing import Any, Callable
import uuid

import numpy as np

from .contracts import RESOLVED_EXPERIMENT_SCHEMA
from .parallel_jaw_renderer.bundle import BUNDLE_SCHEMA
from .parallel_jaw_renderer.cli import load_world_hub_from_metadata
from .parallel_jaw_renderer.container_runner import (
    DEFAULT_IMAGE,
    resolve_local_image_id,
    validate_gpu_selector,
)
from .parallel_jaw_renderer.inputs import load_parallel_jaw_inputs
from .parallel_jaw_renderer.provenance import (
    PROVENANCE_SCHEMA,
    renderer_source_records,
    sha256_file,
)
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
    _verify_exact_file_record,
    _video_geometry,
)
from .video_io import probe_video


PLAN_SCHEMA = "v2d.inpainting.parallel-jaw-comparison-plan/v1"
EXECUTION_SCHEMA = "v2d.inpainting.parallel-jaw-comparison-execution/v1"
GRID_SCHEMA = "v2d.inpainting.parallel-jaw-5panel-grid/v1"
TARGET_RUN_SCHEMA = "v2d.inpainting.parallel-jaw-retarget-run/v1"
CONDITIONS = ("ground_truth", "v2d", "phantom")
GRID_TILE_WIDTH = 640
GRID_COLUMNS = 3
STAGES = ("render", "composite", "grid")
GPU_STAGES = frozenset(("render",))


@dataclass(frozen=True)
class PlanOptions:
    manifest_path: Path
    sequence_id: str
    bundle: Path
    robot_asset_root: Path
    run_root: Path | None = None
    scene_utils_root: Path | None = None
    repository_root: Path | None = None
    python_executable: Path | None = None
    gpu: str | None = None
    renderer_image: str = DEFAULT_IMAGE
    background_rgb: str = "0,0,0"
    max_ik_residual_m: float = 0.01
    ik_orientation_cost: float = 0.010
    max_orientation_residual_deg: float = 20.0
    max_joint_step_rad: float = 0.4
    depth_guard_m: float = 0.003
    overwrite: bool = False
    stages: tuple[str, ...] = STAGES


@dataclass(frozen=True)
class ConditionPaths:
    sequence_id: str
    condition: str
    target: Path
    target_metadata: Path
    source_video: Path
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


@dataclass(frozen=True)
class PipelinePaths:
    sequence_id: str
    robot_id: str
    sequence_root: Path
    robot_root: Path
    source_video: Path
    intrinsic: Path
    world_to_camera: Path
    inpaint_masks: Path
    inpaint_video: Path
    inpaint_metadata: Path
    world_hub_metadata: Path
    object_dir: Path
    object_mask: Path
    object_depth: Path
    object_metadata: Path
    conditions: dict[str, ConditionPaths]
    grid_video: Path
    grid_metadata: Path


@dataclass(frozen=True)
class Action:
    sequence_id: str
    robot_id: str
    condition: str | None
    stage: str
    status: str
    reason: str
    command: tuple[str, ...] | None
    inputs: tuple[Path, ...]
    outputs: tuple[Path, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "robot_id": self.robot_id,
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


def _safe_segment(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise BatchPlanError(f"{label} must be one safe non-empty path segment")
    return value


def _positive_finite(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
        raise BatchPlanError(f"{label} must be a finite positive number")
    return float(value)


def _nonnegative_finite(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or value < 0.0:
        raise BatchPlanError(f"{label} must be a finite non-negative number")
    return float(value)


def _effective_float(value: float) -> float:
    """Return the exact float received by nested CLIs formatted with ``.12g``."""

    return float(f"{value:.12g}")


def _kinematics_policy(options: PlanOptions) -> dict[str, Any]:
    return {
        "max_position_residual_m": _effective_float(options.max_ik_residual_m),
        "max_orientation_residual_deg": _effective_float(
            options.max_orientation_residual_deg
        ),
        "max_joint_step_rad": _effective_float(options.max_joint_step_rad),
        "elbow_out_gain": 0.0,
        "orientation_cost": _effective_float(options.ik_orientation_cost),
        "root_placement": "explicit_shared_T_world_hub",
    }


def _background_rgb(value: str) -> tuple[int, int, int]:
    try:
        result = tuple(int(component.strip()) for component in value.split(","))
    except (AttributeError, ValueError) as exc:
        raise BatchPlanError(
            "background RGB must be three comma-separated integers"
        ) from exc
    if len(result) != 3 or any(not 0 <= component <= 255 for component in result):
        raise BatchPlanError("background RGB values must be in [0,255]")
    return result


def _bundle_info(path: Path) -> dict[str, Any]:
    bundle = path.expanduser().resolve()
    metadata = _load_json(bundle)
    if metadata.get("schema_version") != BUNDLE_SCHEMA:
        raise BatchPlanError(
            f"robot bundle schema must be {BUNDLE_SCHEMA!r}, got "
            f"{metadata.get('schema_version')!r}"
        )
    robot_id = _safe_segment(metadata.get("robot_id"), label="bundle robot_id")
    resolved_files: dict[str, Path] = {}
    for key in ("render_urdf", "ik_urdf"):
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise BatchPlanError(f"bundle {key} must be a non-empty path string")
        candidate = Path(value)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (bundle.parent / candidate).resolve()
        )
        try:
            resolved.relative_to(bundle.parent)
        except ValueError as exc:
            raise BatchPlanError(f"bundle {key} escapes its directory: {resolved}") from exc
        if not resolved.is_file():
            raise BatchPlanError(f"bundle {key} does not exist: {resolved}")
        resolved_files[key] = resolved
    return {
        "path": bundle,
        "metadata": metadata,
        "robot_id": robot_id,
        **resolved_files,
    }


def _condition_paths(
    *,
    sequence_root: Path,
    robot_root: Path,
    sequence_id: str,
    condition: str,
    source_video: Path,
    intrinsic: Path,
    world_to_camera: Path,
) -> ConditionPaths:
    target_root = sequence_root / "parallel_jaw" / "targets" / condition
    output_root = robot_root / condition
    robot = output_root / "robot_render"
    objects = sequence_root / "ground_truth" / "object_render"
    shared_inpaint = sequence_root / "shared_inpaint"
    return ConditionPaths(
        sequence_id=sequence_id,
        condition=condition,
        target=(target_root / "parallel_jaw_trajectory.npz").resolve(),
        target_metadata=(target_root / "parallel_jaw_trajectory.json").resolve(),
        source_video=source_video,
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
        inpaint_masks=(sequence_root / "shared_arm_mask" / "arm_mask.npy").resolve(),
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
        composite_video=(output_root / "final_overlay.mp4").resolve(),
        composite_metadata=(output_root / "final_overlay.json").resolve(),
    )


def _pipeline_paths(
    *,
    run_root: Path,
    sequence_id: str,
    robot_id: str,
    source_video: Path,
    intrinsic: Path,
    world_to_camera: Path,
) -> PipelinePaths:
    sequence_root = (run_root / sequence_id).resolve()
    try:
        sequence_root.relative_to(run_root)
    except ValueError as exc:
        raise BatchPlanError(
            f"sequence output escapes the selected run root: {sequence_root}"
        ) from exc
    robot_root = (sequence_root / "parallel_jaw" / robot_id).resolve()
    conditions = {
        condition: _condition_paths(
            sequence_root=sequence_root,
            robot_root=robot_root,
            sequence_id=sequence_id,
            condition=condition,
            source_video=source_video,
            intrinsic=intrinsic,
            world_to_camera=world_to_camera,
        )
        for condition in CONDITIONS
    }
    representative = conditions["ground_truth"]
    return PipelinePaths(
        sequence_id=sequence_id,
        robot_id=robot_id,
        sequence_root=sequence_root,
        robot_root=robot_root,
        source_video=source_video,
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
        inpaint_masks=representative.inpaint_masks,
        inpaint_video=representative.inpaint_video,
        inpaint_metadata=representative.inpaint_metadata,
        world_hub_metadata=(
            sequence_root / "ground_truth" / "robot_render" / "render_metadata.json"
        ).resolve(),
        object_dir=representative.object_dir,
        object_mask=representative.object_mask,
        object_depth=representative.object_depth,
        object_metadata=representative.object_metadata,
        conditions=conditions,
        grid_video=(robot_root / "final_5panel_comparison.mp4").resolve(),
        grid_metadata=(robot_root / "final_5panel_comparison.json").resolve(),
    )


def _target_complete(
    paths: ConditionPaths,
    geometry: dict[str, Any],
) -> tuple[bool, str]:
    for path in (paths.target, paths.target_metadata):
        if not path.is_file() or path.stat().st_size == 0:
            return False, f"missing/empty shared target artifact: {path}"
    try:
        metadata = _load_json(paths.target_metadata)
        if (
            metadata.get("schema_version") != TARGET_RUN_SCHEMA
            or metadata.get("state") != "complete"
        ):
            return False, "parallel-jaw target sidecar is not a complete retarget run"
        if metadata.get("tracker") != paths.condition:
            return False, "parallel-jaw target sidecar tracker differs from condition"
        if metadata.get("frame_count") != int(geometry["frame_count"]):
            return False, "parallel-jaw target sidecar frame count differs from source"
        output = metadata.get("output")
        record = output.get("trajectory") if isinstance(output, dict) else None
        if not isinstance(record, dict) or set(record) != {
            "filename",
            "sha256",
            "size_bytes",
        }:
            return False, "parallel-jaw target sidecar lacks exact output provenance"
        if record.get("filename") != paths.target.name:
            return False, "parallel-jaw target sidecar output filename differs"
        if record.get("size_bytes") != paths.target.stat().st_size:
            return False, "parallel-jaw target sidecar output size differs"
        if record.get("sha256") != sha256_file(paths.target):
            return False, "parallel-jaw target sidecar output SHA-256 differs"
        inputs = load_parallel_jaw_inputs(
            target_path=paths.target,
            intrinsic_path=paths.intrinsic,
            world_to_camera_path=paths.world_to_camera,
            width=int(geometry["width"]),
            height=int(geometry["height"]),
            fps=float(geometry["fps"]),
        )
        if inputs.tracker != paths.condition:
            return False, "parallel-jaw target NPZ tracker differs from condition"
    except (OSError, ValueError, BatchPlanError) as exc:
        return False, f"invalid shared parallel-jaw target: {exc}"
    if paths.target_metadata.stat().st_mtime_ns < paths.target.stat().st_mtime_ns:
        return False, "parallel-jaw target sidecar predates its NPZ"
    return True, "shared parallel-jaw target contract and provenance validated"


def _map_recorded_asset_path(
    recorded: str,
    *,
    bundle_dir: Path,
    robot_asset_root: Path,
) -> Path:
    if recorded == "/robot_bundle":
        return bundle_dir
    if recorded.startswith("/robot_bundle/"):
        return (bundle_dir / recorded.removeprefix("/robot_bundle/")).resolve()
    if recorded == "/robot_assets":
        return robot_asset_root
    if recorded.startswith("/robot_assets/"):
        return (
            robot_asset_root / recorded.removeprefix("/robot_assets/")
        ).resolve()
    candidate = Path(recorded)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (bundle_dir / candidate).resolve()
    )


def _verify_parallel_render_generation(
    *,
    metadata: dict[str, Any],
    paths: ConditionPaths,
    geometry: dict[str, Any],
    options: PlanOptions,
    bundle_info: dict[str, Any],
    robot_asset_root: Path,
    scene_utils_root: Path,
    world_hub: np.ndarray,
    image_id: str,
) -> None:
    if metadata.get("renderer_kind") != "generic_parallel_jaw":
        raise BatchPlanError("render was not produced by the parallel-jaw renderer")
    if metadata.get("tracker") != paths.condition:
        raise BatchPlanError("render tracker differs from the selected condition")
    if metadata.get("preview") is not False:
        raise BatchPlanError("preview render cannot satisfy a full comparison")
    if metadata.get("rendered_source_frame_indices") != list(
        range(int(geometry["frame_count"]))
    ):
        raise BatchPlanError("rendered source frame indices are not the full clip")
    if metadata.get("trajectory_coordinate_frame") != "world":
        raise BatchPlanError("render trajectory coordinate frame is not world")
    robot_bundle = metadata.get("robot_bundle")
    if (
        not isinstance(robot_bundle, dict)
        or robot_bundle.get("robot_id") != bundle_info["robot_id"]
    ):
        raise BatchPlanError("render robot bundle differs from the selected embodiment")
    if metadata.get("kinematics_policy") != _kinematics_policy(options):
        raise BatchPlanError("render kinematics policy differs from selected thresholds")
    if metadata.get("background_rgb") != list(_background_rgb(options.background_rgb)):
        raise BatchPlanError("render background differs from the selected RGB")
    try:
        recorded_hub = np.asarray(metadata.get("T_world_hub"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BatchPlanError("render lacks a numeric T_world_hub") from exc
    if recorded_hub.shape != (4, 4) or not np.allclose(
        recorded_hub, world_hub, atol=1e-10, rtol=0.0
    ):
        raise BatchPlanError("render T_world_hub differs from shared GT Vega mount")
    if metadata.get("container_image") != options.renderer_image:
        raise BatchPlanError("render image reference differs from selected image")
    if metadata.get("container_image_id") != image_id:
        raise BatchPlanError("render immutable image ID differs from current local image")

    provenance = metadata.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != PROVENANCE_SCHEMA
        or provenance.get("hash_algorithm") != "sha256"
    ):
        raise BatchPlanError("render lacks committed SHA-256 provenance")
    records = provenance.get("inputs")
    expected_inputs = {
        "target": paths.target,
        "intrinsic": paths.intrinsic,
        "world_to_camera": paths.world_to_camera,
        "robot_bundle": bundle_info["path"],
        "world_hub": (
            paths.robot_dir.parents[3]
            / "ground_truth"
            / "robot_render"
            / "render_metadata.json"
        ).resolve(),
    }
    if not isinstance(records, dict) or set(records) != set(expected_inputs):
        raise BatchPlanError("render provenance does not declare the exact input set")
    for name, source in expected_inputs.items():
        _verify_exact_file_record(
            source,
            records[name],
            label=f"parallel-jaw render input {name}",
            expected_keys={"path", "bytes", "sha256"},
        )

    if provenance.get("renderer_source_files") != renderer_source_records():
        raise BatchPlanError("parallel-jaw renderer source provenance is stale")
    _verify_exact_file_record(
        scene_utils_root / "arm_ik.py",
        provenance.get("external_ik_source"),
        label="parallel-jaw external arm_ik.py",
        expected_keys={"path", "bytes", "sha256"},
    )
    assets = provenance.get("robot_assets")
    if not isinstance(assets, dict):
        raise BatchPlanError("render lacks robot asset provenance")
    for key, source in (
        ("render_urdf", bundle_info["render_urdf"]),
        ("ik_urdf", bundle_info["ik_urdf"]),
    ):
        _verify_exact_file_record(
            source,
            assets.get(key),
            label=f"parallel-jaw {key}",
            expected_keys={"path", "bytes", "sha256"},
        )
    visual_records = assets.get("visual_meshes")
    if not isinstance(visual_records, list) or not visual_records:
        raise BatchPlanError("render lacks visual-mesh provenance")
    for index, record in enumerate(visual_records):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise BatchPlanError(f"visual mesh record {index} is invalid")
        source = _map_recorded_asset_path(
            record["path"],
            bundle_dir=bundle_info["path"].parent,
            robot_asset_root=robot_asset_root,
        )
        _verify_exact_file_record(
            source,
            record,
            label=f"parallel-jaw visual mesh {index}",
            expected_keys={"path", "bytes", "sha256"},
        )


def _parallel_render_complete(
    paths: ConditionPaths,
    geometry: dict[str, Any],
    options: PlanOptions,
    *,
    bundle_info: dict[str, Any],
    robot_asset_root: Path,
    scene_utils_root: Path,
    world_hub: np.ndarray,
    resolve_image_id: Callable[[], str],
) -> tuple[bool, str]:
    complete, reason = _render_complete(paths, geometry)
    if not complete:
        return False, reason
    try:
        metadata = _load_json(paths.robot_metadata)
        _verify_parallel_render_generation(
            metadata=metadata,
            paths=paths,
            geometry=geometry,
            options=options,
            bundle_info=bundle_info,
            robot_asset_root=robot_asset_root,
            scene_utils_root=scene_utils_root,
            world_hub=world_hub,
            image_id=resolve_image_id(),
        )
    except (OSError, RuntimeError, ValueError, BatchPlanError) as exc:
        return False, f"parallel-jaw render provenance is stale or invalid: {exc}"
    return True, reason + "; current embodiment/input/policy provenance validated"


def _expected_outputs(paths: ConditionPaths, stage: str) -> tuple[Path, ...]:
    if stage == "render":
        return (
            paths.robot_rgb,
            paths.robot_mask,
            paths.robot_depth,
            paths.robot_metadata,
        )
    if stage == "composite":
        return (paths.composite_video, paths.composite_metadata)
    raise AssertionError(stage)


def _partial_paths(outputs: tuple[Path, ...]) -> tuple[Path, ...]:
    found: set[Path] = set()
    for path in outputs:
        conventional = path.with_name(f"{path.stem}.partial{path.suffix}")
        if conventional.exists() or conventional.is_symlink():
            found.add(conventional)
        if path.parent.is_dir():
            for pattern in (
                f".{path.name}.*.partial",
                f".{path.stem}.*.partial{path.suffix}",
            ):
                found.update(path.parent.glob(pattern))
    return tuple(sorted(found, key=str))


def _collisions(outputs: tuple[Path, ...]) -> tuple[Path, ...]:
    found = {
        path for path in outputs if path.exists() or path.is_symlink()
    }
    found.update(_partial_paths(outputs))
    return tuple(sorted(found, key=str))


def _classify_stage(
    *,
    outputs: tuple[Path, ...],
    complete: bool,
    completion_reason: str,
    prerequisite_errors: list[str],
    overwrite: bool,
) -> tuple[str, str]:
    partials = _partial_paths(outputs)
    if partials and not overwrite:
        prerequisite_errors = [
            *prerequisite_errors,
            "stale partial outputs exist; inspect or pass --overwrite: "
            + ", ".join(str(path) for path in partials),
        ]
    return _classify(
        complete=complete,
        completion_reason=completion_reason,
        outputs_exist=bool(_collisions(outputs)),
        prerequisite_errors=prerequisite_errors,
        overwrite=overwrite,
    )


def _render_command(
    paths: ConditionPaths,
    geometry: dict[str, Any],
    *,
    bundle: Path,
    robot_asset_root: Path,
    scene_utils_root: Path,
    world_hub_metadata: Path,
    repository_root: Path,
    python_executable: Path,
    options: PlanOptions,
) -> tuple[str, ...]:
    command = [
        str(python_executable),
        "-m",
        "inpainting.parallel_jaw_renderer.container_runner",
        "--target",
        str(paths.target),
        "--bundle",
        str(bundle),
        "--intrinsics",
        str(paths.intrinsic),
        "--world-to-camera",
        str(paths.world_to_camera),
        "--robot-asset-root",
        str(robot_asset_root),
        "--scene-utils-root",
        str(scene_utils_root),
        "--output-dir",
        str(paths.robot_dir),
        "--width",
        str(int(geometry["width"])),
        "--height",
        str(int(geometry["height"])),
        "--fps",
        f"{float(geometry['fps']):.12g}",
        "--repository-root",
        str(repository_root),
        "--image",
        options.renderer_image,
        "--T-world-hub-metadata",
        str(world_hub_metadata),
        "--background-rgb",
        options.background_rgb,
        "--max-ik-residual-m",
        f"{options.max_ik_residual_m:.12g}",
        "--ik-orientation-cost",
        f"{options.ik_orientation_cost:.12g}",
        "--max-orientation-residual-deg",
        f"{options.max_orientation_residual_deg:.12g}",
        "--max-joint-step-rad",
        f"{options.max_joint_step_rad:.12g}",
        "--execute",
    ]
    if options.gpu is not None:
        command.extend(("--gpu", options.gpu))
    if options.overwrite:
        command.append("--overwrite")
    return tuple(command)


def _composite_command(
    paths: ConditionPaths,
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


def _robot_display_name(robot_id: str) -> str:
    return {
        "galbot_one_golf": "Galbot Golf",
        "yam_bimanual": "YAM",
    }.get(robot_id, robot_id)


def _grid_spec(paths: PipelinePaths) -> dict[str, Any]:
    display = _robot_display_name(paths.robot_id)
    return {
        "videos": [
            str(paths.source_video),
            str(paths.inpaint_video),
            str(paths.conditions["ground_truth"].composite_video),
            str(paths.conditions["v2d"].composite_video),
            str(paths.conditions["phantom"].composite_video),
        ],
        "labels": [
            "Source",
            "E2FGVI (arms masked)",
            f"GT -> {display}",
            f"V2D -> {display}",
            f"Phantom -> {display}",
        ],
        "tile_width": GRID_TILE_WIDTH,
        "columns": GRID_COLUMNS,
        "max_frames": None,
    }


def _grid_command(
    paths: PipelinePaths,
    *,
    python_executable: Path,
) -> tuple[str, ...]:
    command = [str(python_executable), "-m", "inpainting.make_video_grid"]
    specification = _grid_spec(paths)
    for video, label in zip(
        specification["videos"], specification["labels"], strict=True
    ):
        command.extend(("--video", video, "--label", label))
    command.extend(
        (
            "--output",
            str(paths.grid_video),
            "--tile-width",
            str(GRID_TILE_WIDTH),
            "--columns",
            str(GRID_COLUMNS),
        )
    )
    return tuple(command)


def _fingerprint(path: Path) -> dict[str, int | str]:
    source = path.resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise BatchPlanError(f"cannot fingerprint missing/empty file: {source}")
    before = source.stat()
    result = {"bytes": before.st_size, "sha256": sha256_file(source)}
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise BatchPlanError(f"file changed while fingerprinting: {source}")
    return result


def _grid_input_paths(paths: PipelinePaths) -> dict[str, Path]:
    return {
        "source": paths.source_video,
        "e2fgvi": paths.inpaint_video,
        "ground_truth": paths.conditions["ground_truth"].composite_video,
        "v2d": paths.conditions["v2d"].composite_video,
        "phantom": paths.conditions["phantom"].composite_video,
    }


def _lineage_paths(
    paths: PipelinePaths,
    *,
    manifest: Path,
    bundle: Path,
) -> dict[str, Path]:
    result = {
        "manifest": manifest,
        "robot_bundle": bundle,
        "shared_e2fgvi_metadata": paths.inpaint_metadata,
        "shared_gt_vega_mount_metadata": paths.world_hub_metadata,
        "shared_gt_object_metadata": paths.object_metadata,
    }
    for condition in CONDITIONS:
        condition_paths = paths.conditions[condition]
        result.update(
            {
                f"{condition}_target": condition_paths.target,
                f"{condition}_target_metadata": condition_paths.target_metadata,
                f"{condition}_render_metadata": condition_paths.robot_metadata,
                f"{condition}_composite_metadata": condition_paths.composite_metadata,
            }
        )
    return result


def _verify_fingerprint(path: Path, record: Any, *, label: str) -> None:
    _verify_exact_file_record(
        path,
        record,
        label=label,
        expected_keys={"bytes", "sha256"},
    )


def _grid_complete(
    paths: PipelinePaths,
    geometry: dict[str, Any],
    options: PlanOptions,
    *,
    manifest: Path,
    bundle: Path,
) -> tuple[bool, str]:
    inputs = _grid_input_paths(paths)
    missing_inputs = [str(path) for path in inputs.values() if not path.is_file()]
    if missing_inputs:
        return False, f"missing grid inputs: {missing_inputs}"
    if (
        not paths.grid_video.is_file()
        or paths.grid_video.stat().st_size == 0
        or not paths.grid_metadata.is_file()
        or paths.grid_metadata.stat().st_size == 0
    ):
        return False, "missing/empty five-panel grid artifacts"
    try:
        metadata = _load_json(paths.grid_metadata)
        if (
            metadata.get("schema_version") != GRID_SCHEMA
            or metadata.get("state") != "complete"
        ):
            return False, "grid sidecar is not a complete five-panel run"
        if metadata.get("sequence_id") != paths.sequence_id:
            return False, "grid sidecar sequence differs from this run"
        if metadata.get("robot_id") != paths.robot_id:
            return False, "grid sidecar embodiment differs from this run"
        if metadata.get("specification") != _grid_spec(paths):
            return False, "grid specification differs from the fixed five-panel layout"
        expected_policy = {
            "kinematics": _kinematics_policy(options),
            "depth_guard_m": _effective_float(options.depth_guard_m),
            "shared_world_hub_policy": (
                "ground_truth Vega kinematics.arm_center_world reused for all trackers"
            ),
        }
        if metadata.get("pipeline_policy") != expected_policy:
            return False, "grid pipeline policy differs from selected thresholds"
        input_records = metadata.get("input_fingerprints")
        if not isinstance(input_records, dict) or set(input_records) != set(inputs):
            return False, "grid sidecar lacks the exact five input fingerprints"
        for name, source in inputs.items():
            _verify_fingerprint(
                source, input_records[name], label=f"grid input {name}"
            )
        lineage = _lineage_paths(paths, manifest=manifest, bundle=bundle)
        lineage_records = metadata.get("lineage_fingerprints")
        if not isinstance(lineage_records, dict) or set(lineage_records) != set(lineage):
            return False, "grid sidecar lacks exact pipeline lineage fingerprints"
        for name, source in lineage.items():
            _verify_fingerprint(
                source, lineage_records[name], label=f"grid lineage {name}"
            )
        _verify_fingerprint(
            paths.grid_video,
            metadata.get("output_fingerprint"),
            label="five-panel grid output",
        )
        actual_geometry = probe_video(paths.grid_video)
        recorded_geometry = metadata.get("geometry")
        if actual_geometry.as_dict() != recorded_geometry:
            return False, "grid sidecar geometry differs from decoded output"
        expected_height = 2 * round(
            int(geometry["height"]) * GRID_TILE_WIDTH / int(geometry["width"])
        )
        if (
            actual_geometry.frame_count != int(geometry["frame_count"])
            or actual_geometry.width != GRID_COLUMNS * GRID_TILE_WIDTH
            or actual_geometry.height != expected_height
            or abs(actual_geometry.fps - float(geometry["fps"])) > 1e-3
        ):
            return False, "decoded grid geometry differs from the fixed layout"
    except (OSError, ValueError, BatchPlanError) as exc:
        return False, f"five-panel grid validation failed: {exc}"
    if paths.grid_video.stat().st_mtime_ns < max(
        source.stat().st_mtime_ns for source in inputs.values()
    ):
        return False, "grid video predates one or more panel inputs"
    if paths.grid_metadata.stat().st_mtime_ns < paths.grid_video.stat().st_mtime_ns:
        return False, "grid sidecar predates its output video"
    return True, "complete five-panel grid, hashes, geometry, and lineage validated"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_grid_metadata(action: dict[str, Any], plan: dict[str, Any]) -> None:
    outputs = action.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise BatchPlanError("grid action must declare video and sidecar outputs")
    output_video, metadata_path = (Path(value).resolve() for value in outputs)
    specification = plan.get("grid_specification")
    if not isinstance(specification, dict):
        raise BatchPlanError("plan lacks the fixed grid specification")
    input_values = specification.get("videos")
    if not isinstance(input_values, list) or len(input_values) != 5:
        raise BatchPlanError("grid specification must contain exactly five videos")
    inputs = {
        name: Path(value).resolve()
        for name, value in zip(
            ("source", "e2fgvi", "ground_truth", "v2d", "phantom"),
            input_values,
            strict=True,
        )
    }
    lineage_values = plan.get("lineage_paths")
    if not isinstance(lineage_values, dict):
        raise BatchPlanError("plan lacks grid lineage paths")
    lineage = {name: Path(value).resolve() for name, value in lineage_values.items()}
    geometry = probe_video(output_video)
    payload = {
        "schema_version": GRID_SCHEMA,
        "state": "complete",
        "completed_at": _utc_now(),
        "sequence_id": plan["sequence_id"],
        "robot_id": plan["robot_id"],
        "specification": specification,
        "pipeline_policy": plan["pipeline_policy"],
        "input_fingerprints": {
            name: _fingerprint(path) for name, path in inputs.items()
        },
        "lineage_fingerprints": {
            name: _fingerprint(path) for name, path in lineage.items()
        },
        "output_video": str(output_video),
        "output_fingerprint": _fingerprint(output_video),
        "geometry": geometry.as_dict(),
    }
    _write_json_atomic(metadata_path, payload)


def build_plan(options: PlanOptions) -> dict[str, Any]:
    """Build one fully resolved action plan without creating or changing files."""

    manifest_path = options.manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != RESOLVED_EXPERIMENT_SCHEMA:
        raise BatchPlanError(
            f"manifest schema must be {RESOLVED_EXPERIMENT_SCHEMA!r}, got "
            f"{manifest.get('schema_version')!r}"
        )
    sequence_id = _safe_segment(options.sequence_id, label="sequence")
    run_root = (options.run_root or manifest_path.parent).expanduser().resolve()
    if not run_root.is_dir():
        raise BatchPlanError(f"run root does not exist: {run_root}")
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
    bundle_info = _bundle_info(options.bundle)
    robot_asset_root = options.robot_asset_root.expanduser().resolve()
    if not robot_asset_root.is_dir():
        raise BatchPlanError(f"robot asset root does not exist: {robot_asset_root}")
    try:
        validate_gpu_selector(options.gpu)
    except ValueError as exc:
        raise BatchPlanError(str(exc)) from exc
    if not options.renderer_image.strip():
        raise BatchPlanError("renderer image must not be empty")
    _background_rgb(options.background_rgb)
    _positive_finite(options.max_ik_residual_m, label="max IK residual")
    _positive_finite(options.ik_orientation_cost, label="IK orientation cost")
    _positive_finite(
        options.max_orientation_residual_deg,
        label="max orientation residual",
    )
    _positive_finite(options.max_joint_step_rad, label="max joint step")
    _nonnegative_finite(options.depth_guard_m, label="depth guard")
    selected_stages = tuple(dict.fromkeys(options.stages))
    if not selected_stages or any(stage not in STAGES for stage in selected_stages):
        raise BatchPlanError(
            f"stages must be a non-empty subset of {STAGES}, got {selected_stages}"
        )
    selected_stages = tuple(stage for stage in STAGES if stage in selected_stages)

    roots = manifest.get("roots")
    if not isinstance(roots, dict):
        raise BatchPlanError("manifest lacks roots")
    camera_root = _manifest_path(
        roots.get("camera"), "roots.camera", kind="dir"
    )
    old_robot_assets = _manifest_path(
        roots.get("robot_assets"), "roots.robot_assets", kind="path"
    )
    scene_utils_root = (
        options.scene_utils_root
        or old_robot_assets.parent / "tasks" / "scene_utils"
    ).expanduser().resolve()
    if not scene_utils_root.is_dir():
        raise BatchPlanError(f"scene-utils root does not exist: {scene_utils_root}")
    if not (scene_utils_root / "arm_ik.py").is_file():
        raise BatchPlanError(
            f"scene-utils root lacks arm_ik.py: {scene_utils_root / 'arm_ik.py'}"
        )

    entries = manifest.get("sequences")
    if not isinstance(entries, list) or not entries:
        raise BatchPlanError("manifest sequences must be a non-empty list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("sequence_id") == sequence_id
    ]
    if len(matches) != 1:
        raise BatchPlanError(
            f"manifest must contain exactly one sequence {sequence_id!r}; "
            f"found {len(matches)}"
        )
    entry = matches[0]
    video = entry.get("video")
    camera = entry.get("camera")
    if not isinstance(video, dict) or not isinstance(camera, dict):
        raise BatchPlanError(f"manifest sequence {sequence_id} lacks video/camera")
    source_video = _manifest_path(
        video.get("path"), f"{sequence_id}.video.path", kind="file"
    )
    geometry = {
        key: video.get(key) for key in ("frame_count", "width", "height", "fps")
    }
    _video_geometry(geometry, label=f"{sequence_id} source video")
    intrinsic = _manifest_path(
        camera.get("intrinsic"), f"{sequence_id}.camera.intrinsic", kind="file"
    )
    world_to_camera = _manifest_path(
        camera.get("extrinsic"), f"{sequence_id}.camera.extrinsic", kind="file"
    )
    for label, source in (
        ("intrinsic", intrinsic),
        ("extrinsic", world_to_camera),
    ):
        try:
            source.relative_to(camera_root)
        except ValueError as exc:
            raise BatchPlanError(
                f"{sequence_id} camera {label} escapes official camera root"
            ) from exc
    if camera.get("available") is not True:
        raise BatchPlanError("official camera entry is not marked available")
    condition_entries = entry.get("conditions")
    if not isinstance(condition_entries, dict):
        raise BatchPlanError(f"manifest sequence {sequence_id} lacks conditions")

    paths = _pipeline_paths(
        run_root=run_root,
        sequence_id=sequence_id,
        robot_id=bundle_info["robot_id"],
        source_video=source_video,
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
    )
    if not paths.world_hub_metadata.is_file():
        raise BatchPlanError(
            f"shared GT Vega mount metadata is missing: {paths.world_hub_metadata}"
        )
    try:
        world_hub = load_world_hub_from_metadata(paths.world_hub_metadata)
    except (OSError, ValueError) as exc:
        raise BatchPlanError(f"invalid shared GT Vega mount metadata: {exc}") from exc

    representative = paths.conditions["ground_truth"]
    inpaint_errors = _validate_inpaint(representative, geometry)
    object_complete, object_reason = _object_depth_complete(
        representative, geometry
    )

    image_id_cache: list[str] = []

    def current_image_id() -> str:
        if not image_id_cache:
            image_id_cache.append(resolve_local_image_id(options.renderer_image))
        return image_id_cache[0]

    actions: list[Action] = []
    render_future: dict[str, bool] = {}
    for condition in CONDITIONS:
        condition_paths = paths.conditions[condition]
        errors: list[str] = []
        condition_entry = condition_entries.get(condition)
        if not isinstance(condition_entry, dict):
            errors.append(f"manifest lacks condition {condition!r}")
        else:
            if condition_entry.get("tracker") != condition:
                errors.append(f"manifest condition {condition!r} tracker differs")
            blockers = condition_entry.get("blockers")
            if blockers:
                errors.append(f"manifest condition {condition!r} blockers: {blockers}")
        target_complete, target_reason = _target_complete(condition_paths, geometry)
        if not target_complete:
            errors.append(target_reason)
        complete, reason = _parallel_render_complete(
            condition_paths,
            geometry,
            options,
            bundle_info=bundle_info,
            robot_asset_root=robot_asset_root,
            scene_utils_root=scene_utils_root,
            world_hub=world_hub,
            resolve_image_id=current_image_id,
        )
        outputs = _expected_outputs(condition_paths, "render")
        status, status_reason = _classify_stage(
            outputs=outputs,
            complete=complete,
            completion_reason=reason,
            prerequisite_errors=errors,
            overwrite=options.overwrite if "render" in selected_stages else False,
        )
        render_future[condition] = status == "skipped_complete" or (
            "render" in selected_stages and status in PENDING_STATES
        )
        command = (
            _render_command(
                condition_paths,
                geometry,
                bundle=bundle_info["path"],
                robot_asset_root=robot_asset_root,
                scene_utils_root=scene_utils_root,
                world_hub_metadata=paths.world_hub_metadata,
                repository_root=repository_root,
                python_executable=python_executable,
                options=options,
            )
            if status in PENDING_STATES and options.gpu is not None
            else None
        )
        if status in PENDING_STATES and options.gpu is None:
            status_reason += (
                "; render command withheld until an explicit --gpu is supplied"
            )
        if "render" in selected_stages:
            actions.append(
                Action(
                    sequence_id=sequence_id,
                    robot_id=paths.robot_id,
                    condition=condition,
                    stage="render",
                    status=status,
                    reason=status_reason,
                    command=command,
                    inputs=(
                        condition_paths.target,
                        bundle_info["path"],
                        intrinsic,
                        world_to_camera,
                        paths.world_hub_metadata,
                        robot_asset_root,
                        scene_utils_root,
                    ),
                    outputs=outputs,
                )
            )

    composite_future: dict[str, bool] = {}
    for condition in CONDITIONS:
        condition_paths = paths.conditions[condition]
        errors = list(inpaint_errors)
        if not render_future[condition]:
            errors.append(
                "robot render is neither complete nor scheduled successfully"
            )
        if not object_complete:
            errors.append(
                "validated shared GT object-depth bundle is unavailable: "
                + object_reason
            )
        complete, reason = _composite_complete(
            condition_paths,
            geometry,
            options,
            use_object_depth=True,
        )
        outputs = _expected_outputs(condition_paths, "composite")
        status, status_reason = _classify_stage(
            outputs=outputs,
            complete=complete,
            completion_reason=reason,
            prerequisite_errors=errors,
            overwrite=(
                options.overwrite if "composite" in selected_stages else False
            ),
        )
        composite_future[condition] = (
            status == "skipped_complete"
            or ("composite" in selected_stages and status in PENDING_STATES)
        )
        if "composite" in selected_stages:
            actions.append(
                Action(
                    sequence_id=sequence_id,
                    robot_id=paths.robot_id,
                    condition=condition,
                    stage="composite",
                    status=status,
                    reason=status_reason,
                    command=(
                        _composite_command(
                            condition_paths,
                            python_executable=python_executable,
                            options=options,
                        )
                        if status in PENDING_STATES
                        else None
                    ),
                    inputs=(
                        paths.inpaint_video,
                        condition_paths.robot_rgb,
                        condition_paths.robot_mask,
                        condition_paths.robot_depth,
                        condition_paths.robot_metadata,
                        paths.object_mask,
                        paths.object_depth,
                        paths.object_metadata,
                    ),
                    outputs=outputs,
                )
            )

    grid_errors = list(inpaint_errors)
    incomplete_conditions = [
        condition for condition in CONDITIONS if not composite_future[condition]
    ]
    if incomplete_conditions:
        grid_errors.append(
            "composites are neither complete nor scheduled for: "
            + ", ".join(incomplete_conditions)
        )
    grid_complete, grid_reason = _grid_complete(
        paths,
        geometry,
        options,
        manifest=manifest_path,
        bundle=bundle_info["path"],
    )
    grid_outputs = (paths.grid_video, paths.grid_metadata)
    grid_status, grid_status_reason = _classify_stage(
        outputs=grid_outputs,
        complete=grid_complete,
        completion_reason=grid_reason,
        prerequisite_errors=grid_errors,
        overwrite=options.overwrite if "grid" in selected_stages else False,
    )
    if "grid" in selected_stages:
        actions.append(
            Action(
                sequence_id=sequence_id,
                robot_id=paths.robot_id,
                condition=None,
                stage="grid",
                status=grid_status,
                reason=grid_status_reason,
                command=(
                    _grid_command(paths, python_executable=python_executable)
                    if grid_status in PENDING_STATES
                    else None
                ),
                inputs=tuple(_grid_input_paths(paths).values()),
                outputs=grid_outputs,
            )
        )

    counts: dict[str, int] = {}
    for action in actions:
        counts[action.status] = counts.get(action.status, 0) + 1
    lineage = _lineage_paths(
        paths, manifest=manifest_path, bundle=bundle_info["path"]
    )
    return {
        "schema_version": PLAN_SCHEMA,
        "mode": "plan",
        "created_at": _utc_now(),
        "manifest": str(manifest_path),
        "experiment_id": manifest.get("experiment_id"),
        "sequence_id": sequence_id,
        "robot_id": paths.robot_id,
        "run_root": str(run_root),
        "output_root": str(paths.robot_root),
        "bundle": str(bundle_info["path"]),
        "robot_asset_root": str(robot_asset_root),
        "scene_utils_root": str(scene_utils_root),
        "repository_root": str(repository_root),
        "python_executable": str(python_executable),
        "gpu": options.gpu,
        "renderer_image": options.renderer_image,
        "overwrite": options.overwrite,
        "selected_stages": list(selected_stages),
        "shared_world_hub": {
            "source_metadata": str(paths.world_hub_metadata),
            "T_world_hub": world_hub.tolist(),
            "applies_to": list(CONDITIONS),
        },
        "pipeline_policy": {
            "kinematics": _kinematics_policy(options),
            "depth_guard_m": _effective_float(options.depth_guard_m),
            "shared_world_hub_policy": (
                "ground_truth Vega kinematics.arm_center_world reused for all trackers"
            ),
        },
        "grid_specification": _grid_spec(paths),
        "lineage_paths": {name: str(path) for name, path in lineage.items()},
        "shared_inputs": {
            "source": str(paths.source_video),
            "e2fgvi": str(paths.inpaint_video),
            "object_metadata": str(paths.object_metadata),
            "object_state": "complete" if object_complete else "invalid",
            "object_reason": object_reason,
        },
        "summary": counts,
        "actions": [action.as_dict() for action in actions],
    }


def execute_plan(
    plan: dict[str, Any],
    options: PlanOptions,
    *,
    run_command: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Execute pending actions sequentially and verify the complete final state."""

    blocked = [action for action in plan["actions"] if action["status"] == "blocked"]
    if blocked:
        labels = [
            f"{item['condition'] or 'shared'}/{item['stage']}: {item['reason']}"
            for item in blocked
        ]
        raise BatchPlanError(
            "refusing execution with blocked actions: " + " | ".join(labels)
        )
    pending = [
        action for action in plan["actions"] if action["status"] in PENDING_STATES
    ]
    if any(action["stage"] in GPU_STAGES for action in pending) and options.gpu is None:
        raise BatchPlanError(
            "pending parallel-jaw renders require an explicit --gpu with --execute"
        )
    if (
        any(action["status"] == "pending_overwrite" for action in pending)
        and not options.overwrite
    ):
        raise BatchPlanError("plan contains overwrite actions without permission")

    repository_root = Path(plan["repository_root"]).resolve()
    executed: list[dict[str, Any]] = []
    for action in pending:
        output_paths = tuple(Path(value).resolve() for value in action["outputs"])
        if not options.overwrite:
            collisions = _collisions(output_paths)
            if collisions:
                raise BatchPlanError(
                    "refusing outputs that appeared after planning for "
                    f"{action['condition'] or 'shared'}/{action['stage']}: "
                    + ", ".join(str(path) for path in collisions)
                )
        else:
            for partial in _partial_paths(output_paths):
                if partial.is_dir() and not partial.is_symlink():
                    raise BatchPlanError(
                        f"refusing to remove partial output directory: {partial}"
                    )
                partial.unlink(missing_ok=True)
        command = action.get("command")
        if not isinstance(command, list) or not command:
            raise BatchPlanError(
                f"pending {action['condition'] or 'shared'}/{action['stage']} "
                "action has no command"
            )
        print(f"EXEC {shlex.join(command)}", flush=True)
        completed = run_command(command, cwd=repository_root, check=False)
        return_code = int(completed.returncode)
        if return_code:
            raise RuntimeError(
                f"{action['condition'] or 'shared'}/{action['stage']} failed "
                f"with exit {return_code}"
            )
        if action["stage"] == "grid":
            _write_grid_metadata(action, plan)
        executed.append(
            {
                "condition": action["condition"],
                "stage": action["stage"],
                "return_code": return_code,
            }
        )

    verified = build_plan(replace(options, overwrite=False))
    incomplete = [
        action
        for action in verified["actions"]
        if action["status"] != "skipped_complete"
    ]
    if incomplete:
        labels = [
            f"{item['condition'] or 'shared'}/{item['stage']}={item['status']}: "
            f"{item['reason']}"
            for item in incomplete
        ]
        raise RuntimeError(
            "post-execution artifact verification failed: " + " | ".join(labels)
        )
    return {
        "schema_version": EXECUTION_SCHEMA,
        "state": "complete",
        "completed_at": _utc_now(),
        "manifest": plan["manifest"],
        "sequence_id": plan["sequence_id"],
        "robot_id": plan["robot_id"],
        "final_video": str(
            Path(verified["output_root"]) / "final_5panel_comparison.mp4"
        ),
        "executed": executed,
        "resumed": [
            {
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
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--robot-asset-root", required=True, type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--scene-utils-root", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--renderer-image", default=DEFAULT_IMAGE)
    parser.add_argument("--background-rgb", default="0,0,0")
    parser.add_argument("--max-ik-residual-m", type=float, default=0.01)
    parser.add_argument("--ik-orientation-cost", type=float, default=0.010)
    parser.add_argument("--max-orientation-residual-deg", type=float, default=20.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.4)
    parser.add_argument("--depth-guard-m", type=float, default=0.003)
    parser.add_argument("--stage", action="append", choices=STAGES, dest="stages")
    parser.add_argument("--gpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute pending stages; default prints a read-only JSON plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = PlanOptions(
        manifest_path=args.manifest,
        sequence_id=args.sequence,
        bundle=args.bundle,
        robot_asset_root=args.robot_asset_root,
        run_root=args.run_root,
        scene_utils_root=args.scene_utils_root,
        repository_root=args.repository_root,
        python_executable=args.python_executable,
        gpu=args.gpu,
        renderer_image=args.renderer_image,
        background_rgb=args.background_rgb,
        max_ik_residual_m=args.max_ik_residual_m,
        ik_orientation_cost=args.ik_orientation_cost,
        max_orientation_residual_deg=args.max_orientation_residual_deg,
        max_joint_step_rad=args.max_joint_step_rad,
        depth_guard_m=args.depth_guard_m,
        overwrite=args.overwrite,
        stages=tuple(args.stages or STAGES),
    )
    try:
        plan = build_plan(options)
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        result = execute_plan(plan, options)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (BatchPlanError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
