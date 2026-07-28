"""Plan, execute, and resume the MECKA-to-Panda inpainting pipeline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inpainting.adapters import mecka, mecka_lerobot, mecka_parallel_jaw
from inpainting.mecka_panda import arm_mask, lerobot_source, propainter
from inpainting.mecka_panda.composite import (
    COMPOSITE_SCHEMA,
)
from inpainting.mecka_panda.composite import (
    execute as composite,
)
from inpainting.mecka_panda.contracts import artifact, sha256, write_json_atomic
from inpainting.mecka_panda.video_io import Mp4Writer, probe_video
from inpainting.panda_renderer import render as panda_render

STAGES = (
    "tracking",
    "mask",
    "inpaint",
    "retarget",
    "render",
    "composite",
    "review",
)
PIPELINE_SCHEMA = "v2d.inpainting.mecka-panda-pipeline/v2"
REVIEW_SCHEMA = "v2d.inpainting.four-stage-review/v2"
DEFAULT_RECONSTRUCTION_DIR = Path(__file__).resolve().parents[1] / "reconstruction"


def _nested_value(metadata: dict[str, Any], dotted_key: str) -> Any:
    value: Any = metadata
    parts = dotted_key.split(".")
    index = 0
    while index < len(parts):
        if not isinstance(value, dict):
            return None
        for stop in range(len(parts), index, -1):
            candidate = ".".join(parts[index:stop])
            if candidate in value:
                value = value[candidate]
                index = stop
                break
        else:
            return None
    return value


def _artifact_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if {"path", "size_bytes", "sha256"} <= set(value):
            return [value]
        records: list[dict[str, Any]] = []
        for child in value.values():
            records.extend(_artifact_records(child))
        return records
    if isinstance(value, list):
        records = []
        for child in value:
            records.extend(_artifact_records(child))
        return records
    return []


def _metadata_complete(path: Path, expected: dict[str, Any] | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if metadata.get("state") != "complete":
        return False
    if expected and any(
        _nested_value(metadata, key) != value for key, value in expected.items()
    ):
        return False
    for record in _artifact_records(metadata):
        artifact_path = Path(record["path"])
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != record.get("size_bytes")
            or sha256(artifact_path) != record.get("sha256")
        ):
            return False
    return True


def _layout(output: Path) -> dict[str, Path]:
    return {
        "tracking": output / "tracking",
        "mask": output / "arm_mask",
        "inpaint": output / "inpaint",
        "retarget": output / "retarget",
        "render": output / "robot_render",
        "composite": output / "composite",
        "review_video": output / "four_stage_compare.mp4",
        "review_metadata": output / "four_stage_compare.json",
    }


def _published_outputs(paths: dict[str, Path]) -> dict[str, tuple[Path, ...]]:
    return {
        "tracking": (
            paths["tracking"] / mecka.METADATA_FILENAME,
            paths["tracking"] / mecka.TRACKING_FILENAME,
            paths["tracking"] / mecka.INTRINSIC_FILENAME,
            paths["tracking"] / mecka.CAMERA_ROTATION_FILENAME,
            paths["tracking"] / mecka_lerobot.VIDEO_FILENAME,
        ),
        "mask": (
            paths["mask"] / arm_mask.METADATA_FILENAME,
            paths["mask"] / arm_mask.MASK_FILENAME,
            paths["mask"] / arm_mask.PREVIEW_FILENAME,
        ),
        "inpaint": (
            paths["inpaint"] / propainter.METADATA_FILENAME,
            paths["inpaint"] / propainter.OUTPUT_FILENAME,
        ),
        "retarget": (
            paths["retarget"] / mecka_parallel_jaw.METADATA_FILENAME,
            paths["retarget"] / mecka_parallel_jaw.TRAJECTORY_FILENAME,
        ),
        "render": (
            paths["render"] / panda_render.METADATA_FILENAME,
            paths["render"] / panda_render.RGB_FILENAME,
            paths["render"] / panda_render.MASK_FILENAME,
            paths["render"] / panda_render.DEPTH_FILENAME,
        ),
        "composite": (
            paths["composite"] / "final_overlay.json",
            paths["composite"] / "final_overlay.mp4",
        ),
        "review": (paths["review_metadata"], paths["review_video"]),
    }


def _default_stages(args: argparse.Namespace) -> tuple[str, ...]:
    """Select only the built-in stages needed by the requested overrides."""
    selected = {"tracking", "retarget", "render", "composite", "review"}
    if args.background is None:
        selected.update(("mask", "inpaint"))
    if args.mask_preview is None:
        selected.add("mask")
    return tuple(stage for stage in STAGES if stage in selected)


def _dependencies(args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
    """Return the runtime DAG after pruning externally supplied artifacts."""
    composite = ("render",)
    if args.background is None:
        composite += ("inpaint",)
    review = ["composite"]
    if args.mask_preview is None:
        review.append("mask")
    if args.background is None:
        review.append("inpaint")
    return {
        "tracking": (),
        "mask": ("tracking",),
        "inpaint": ("mask",),
        "retarget": ("tracking",),
        "render": ("retarget",),
        "composite": composite,
        "review": tuple(review),
    }


def _chosen_background(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> tuple[Path, int, bool]:
    if args.background is None:
        return paths["inpaint"] / propainter.OUTPUT_FILENAME, 0, False
    return (
        args.background.expanduser().resolve(),
        args.background_start_frame,
        True,
    )


def _chosen_mask_preview(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> tuple[Path, int, bool]:
    if args.mask_preview is None:
        return paths["mask"] / arm_mask.PREVIEW_FILENAME, 0, False
    return (
        args.mask_preview.expanduser().resolve(),
        args.mask_start_frame,
        True,
    )


def _bind_artifact(
    expected: dict[str, Any],
    dotted_prefix: str,
    path: Path,
    *,
    include_identity: bool,
) -> None:
    """Bind a requested artifact path, and external content when available."""
    expected[f"{dotted_prefix}.path"] = str(path)
    if include_identity and path.is_file():
        record = artifact(path)
        expected[f"{dotted_prefix}.size_bytes"] = record["size_bytes"]
        expected[f"{dotted_prefix}.sha256"] = record["sha256"]


def _arm_mask_config(args: argparse.Namespace) -> arm_mask.ArmMaskConfig:
    config_path = args.arm_mask_config
    if config_path is None:
        return arm_mask.ArmMaskConfig()
    payload = json.loads(config_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("--arm-mask-config must contain one JSON object")
    return arm_mask.ArmMaskConfig(**payload)


def _jsonable_config(config: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), allow_nan=False))


def _sequence_id(plan: dict[str, Any]) -> str:
    return f"episode_{int(plan['episode_index']):06d}"


def _dependency_preflight(
    args: argparse.Namespace,
    stage_states: dict[str, str],
) -> list[str]:
    """Check heavyweight stage dependencies only when that stage will run."""
    blockers: list[str] = []
    if stage_states.get("mask") not in {None, "skipped_complete"}:
        try:
            arm_mask.preflight(args.reconstruction_dir)
        except (OSError, RuntimeError, ValueError) as error:
            blockers.append(f"mask dependency preflight failed: {error}")
    if stage_states.get("inpaint") not in {None, "skipped_complete"}:
        try:
            propainter.preflight(
                args.propainter_dir,
                args.propainter_python,
            )
        except (OSError, ValueError) as error:
            blockers.append(f"ProPainter dependency preflight failed: {error}")
    return blockers


@dataclass(frozen=True)
class SourceInfo:
    """Episode facts needed to plan, resolved without decoding any video."""

    kind: str
    episode_index: int
    task_id: str
    frame_count: int
    width: int
    height: int
    fps: float
    source_video: Path
    source_parquet: str
    dataset_uri: str
    blockers: list[str] = field(default_factory=list)


def _is_lerobot(dataset: str) -> bool:
    if dataset.startswith("s3://"):
        return True
    root = Path(dataset).expanduser()
    if (root / "meta" / "info.json").is_file():
        return True
    # A container holding one dataset root per shard, as Cosmos3 publishes them.
    return root.is_dir() and any(
        (child / "meta" / "info.json").is_file()
        for child in root.iterdir()
        if child.is_dir()
    )


def _resolve_source(args: argparse.Namespace, paths: dict[str, Path]) -> SourceInfo:
    """Describe the selected episode for either supported dataset layout."""
    dataset = str(args.dataset)
    if _is_lerobot(dataset):
        dataset_uri = lerobot_source.resolve_shard(
            dataset, shard=args.shard, credentials=args.credentials
        )
        source = lerobot_source.open_source(dataset_uri, credentials=args.credentials)
        info = lerobot_source.load_info(source)
        episode = None if args.episode is None else int(args.episode)
        record = lerobot_source.find_episode(source, info, episode)
        shape = info["features"][lerobot_source.VIDEO_KEY]["shape"]
        return SourceInfo(
            kind="lerobot",
            episode_index=record.episode_index,
            task_id=record.task_id,
            frame_count=record.length,
            width=int(shape[1]),
            height=int(shape[0]),
            fps=float(info["fps"]),
            # Produced by the tracking stage; it does not exist while planning.
            source_video=paths["tracking"] / mecka_lerobot.VIDEO_FILENAME,
            source_parquet=(
                f"{dataset_uri} rows [{record.row_start}, {record.row_stop})"
            ),
            dataset_uri=dataset_uri,
        )

    root = Path(dataset).expanduser().resolve()
    record = mecka.resolve_episode(mecka.load_manifest(root), args.episode)
    source_video = (root / record["clip"]).resolve()
    source_parquet = (root / record["data"]).resolve()
    geometry = probe_video(source_video)
    return SourceInfo(
        kind="mecka_manifest",
        episode_index=int(record["episode_index"]),
        task_id=str(record.get("task_id", "")),
        frame_count=int(geometry["frame_count"]),
        width=int(geometry["width"]),
        height=int(geometry["height"]),
        fps=float(geometry["fps"]),
        source_video=source_video,
        source_parquet=str(source_parquet),
        dataset_uri=str(root),
        blockers=(
            [] if source_parquet.is_file() else [f"missing parquet: {source_parquet}"]
        ),
    )


def _ancestors(
    stage: str,
    dependencies: dict[str, tuple[str, ...]],
) -> set[str]:
    result: set[str] = set()
    pending = list(dependencies[stage])
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(dependencies[dependency])
    return result


def _identity_required(
    stage: str,
    selected: tuple[str, ...],
    dependencies: dict[str, tuple[str, ...]],
) -> bool:
    """Return whether planning needs the current implementation identity."""
    return any(
        candidate == stage or stage in _ancestors(candidate, dependencies)
        for candidate in selected
    )


def _plan_stage_states(
    *,
    selected: tuple[str, ...],
    dependencies: dict[str, tuple[str, ...]],
    stage_metadata: dict[str, Path],
    published_outputs: dict[str, tuple[Path, ...]],
    expected: dict[str, dict[str, Any]],
    overwrite: bool,
    blockers: list[str],
) -> tuple[dict[str, str], dict[str, bool], dict[str, bool]]:
    """Classify each selected stage without serializing independent branches."""
    selected_set = set(selected)
    raw_current = {
        stage: _metadata_complete(stage_metadata[stage], expected[stage])
        for stage in STAGES
    }
    dependency_current: dict[str, bool] = {}
    effective_current: dict[str, bool] = {}
    states: dict[str, str] = {}
    for stage in STAGES:
        dependencies_are_current = all(
            effective_current[dependency] for dependency in dependencies[stage]
        )
        dependency_current[stage] = dependencies_are_current
        if stage not in selected_set:
            effective_current[stage] = raw_current[stage] and dependencies_are_current
            continue
        if overwrite:
            state = "replace"
        elif raw_current[stage] and dependencies_are_current:
            state = "skipped_complete"
        elif raw_current[stage]:
            state = "refresh_dependency"
        else:
            state = "pending"
        states[stage] = state
        # A stage that will run changes the artifact identity seen downstream.
        effective_current[stage] = state == "skipped_complete"
        if (
            state == "pending"
            and any(path.exists() for path in published_outputs[stage])
            and not overwrite
        ):
            blockers.append(f"stale {stage!r} outputs exist; rerun with --overwrite")

    for stage in selected:
        for dependency in sorted(_ancestors(stage, dependencies), key=STAGES.index):
            if dependency not in selected_set and not effective_current[dependency]:
                blocker = (
                    f"stale dependency {dependency!r}; include --stage {dependency}"
                )
                if blocker not in blockers:
                    blockers.append(blocker)
    return states, raw_current, dependency_current


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build a read-only execution plan."""
    output = args.output_dir.expanduser().resolve()
    paths = _layout(output)
    source = _resolve_source(args, paths)
    count = (
        source.frame_count - args.start_frame
        if args.max_frames is None
        else min(args.max_frames, source.frame_count - args.start_frame)
    )
    if count <= 0:
        raise ValueError("Selected frame window is empty")
    selected = (
        _default_stages(args)
        if args.stage is None
        else tuple(stage for stage in STAGES if stage in set(args.stage))
    )
    dependencies = _dependencies(args)
    blockers: list[str] = list(source.blockers)
    if (
        source.kind == "lerobot"
        and any(stage in selected for stage in ("mask", "inpaint", "review"))
        and "tracking" not in selected
        and not source.source_video.is_file()
    ):
        blockers.append(f"missing extracted clip: {source.source_video}")
    if (args.object_mask is None) != (args.object_depth is None):
        blockers.append("--object-mask and --object-depth must be given together")
    elif args.object_mask is not None and not args.emit_depth:
        # Catch this here rather than after a multi-minute render that would
        # then have no depth for compositing to consult.
        blockers.append("depth-aware compositing needs --emit-depth")
    elif args.object_mask is not None and args.object_depth is not None:
        for label, value in (
            ("object mask", args.object_mask),
            ("object depth", args.object_depth),
        ):
            if not value.expanduser().resolve().is_file():
                blockers.append(f"missing {label}: {value}")
    if (
        args.background is not None
        and any(stage in selected for stage in ("composite", "review"))
        and not args.background.expanduser().resolve().is_file()
    ):
        blockers.append(f"missing background: {args.background}")
    if (
        "review" in selected
        and args.mask_preview is not None
        and not args.mask_preview.expanduser().resolve().is_file()
    ):
        blockers.append(f"missing mask preview: {args.mask_preview}")
    if args.background is not None and args.background_start_frame < 0:
        blockers.append("--background-start-frame must be non-negative")
    if args.mask_preview is not None and args.mask_start_frame < 0:
        blockers.append("--mask-start-frame must be non-negative")

    background, background_start_frame, external_background = _chosen_background(
        args, paths
    )
    mask_preview, mask_start_frame, external_mask_preview = _chosen_mask_preview(
        args, paths
    )
    stage_metadata = {
        "tracking": paths["tracking"] / mecka.METADATA_FILENAME,
        "mask": paths["mask"] / arm_mask.METADATA_FILENAME,
        "inpaint": paths["inpaint"] / propainter.METADATA_FILENAME,
        "retarget": paths["retarget"] / mecka_parallel_jaw.METADATA_FILENAME,
        "render": paths["render"] / panda_render.METADATA_FILENAME,
        "composite": paths["composite"] / "final_overlay.json",
        "review": paths["review_metadata"],
    }
    tracking_expected: dict[str, Any] = {
        "schema_version": (
            mecka_lerobot.RUN_SCHEMA if source.kind == "lerobot" else mecka.RUN_SCHEMA
        ),
        "episode_index": source.episode_index,
        "task_id": source.task_id,
        "frame_window.start": args.start_frame,
        "frame_window.count": count,
    }
    if source.kind == "lerobot":
        tracking_expected["source.dataset_uri"] = source.dataset_uri
    else:
        tracking_expected["source.parquet.path"] = source.source_parquet
        tracking_expected["source.video.path"] = str(source.source_video)

    mask_config = _arm_mask_config(args)
    reconstruction = args.reconstruction_dir.expanduser().resolve()
    working_width = int(mask_config.working_width)
    working_height = round(working_width * source.height / source.width)
    working_height -= working_height % 2
    mask_container_images: dict[str, dict[str, str]] = {}
    if _identity_required("mask", selected, dependencies):
        try:
            mask_container_images = arm_mask.resolve_container_images()
        except (OSError, RuntimeError, ValueError):
            mask_container_images = {
                name: {
                    "image": image,
                    "image_id": "__unresolved__",
                }
                for name, image in arm_mask.CONTAINER_IMAGE_NAMES.items()
            }
    mask_expected: dict[str, Any] = {
        "schema_version": arm_mask.RUN_SCHEMA,
        "sequence_id": f"episode_{source.episode_index:06d}",
        "episode_index": source.episode_index,
        "frame_window.source_start": args.start_frame,
        "frame_window.count": count,
        "frame_window.mask_start": 0,
        "geometry.source.frame_count": count,
        "geometry.source.width": source.width,
        "geometry.source.height": source.height,
        "geometry.source.fps": source.fps,
        "geometry.working.frame_count": count,
        "geometry.working.width": working_width,
        "geometry.working.height": working_height,
        "geometry.working.fps": source.fps,
        "source.tracking.path": str(paths["tracking"] / mecka.TRACKING_FILENAME),
        "source.intrinsic.path": str(paths["tracking"] / mecka.INTRINSIC_FILENAME),
        "source.tracking_metadata.path": str(
            paths["tracking"] / mecka.METADATA_FILENAME
        ),
        "source.video.path": str(source.source_video),
        "source.reconstruction_dir": str(reconstruction),
        "source.runners.grounding_dino.path": str(arm_mask.GROUNDING_DINO_RUNNER),
        "source.runners.sam2.path": str(arm_mask.SAM2_RUNNER),
        "source.runners.container.path": str(
            reconstruction / arm_mask.CONTAINER_HELPER_RELATIVE_PATH
        ),
        "source.model_weights.grounding_dino.path": str(
            reconstruction / "data" / "weights" / "grounding_dino"
        ),
        "source.model_weights.sam2.path": str(
            reconstruction / "data" / "weights" / "sam2"
        ),
    }
    for name, identity in mask_container_images.items():
        prefix = f"source.container_images.{name}"
        mask_expected[f"{prefix}.image"] = identity["image"]
        mask_expected[f"{prefix}.image_id"] = identity["image_id"]
    for key, value in _jsonable_config(mask_config).items():
        mask_expected[f"config.{key}"] = value

    inpaint_expected: dict[str, Any] = {
        "schema_version": propainter.PROPAINTER_SCHEMA,
        "geometry.frame_count": count,
        "geometry.width": source.width,
        "geometry.height": source.height,
        "geometry.fps": source.fps,
        "source_window.start_frame": args.start_frame,
        "source_window.stop_frame_exclusive": args.start_frame + count,
        "configuration.backend": "propainter",
        "configuration.fp16": args.propainter_fp16,
        "configuration.neighbor_length": args.propainter_neighbor_length,
        "configuration.ref_stride": args.propainter_ref_stride,
        "configuration.resize_ratio": args.propainter_resize_ratio,
        "configuration.save_frames": True,
        "configuration.source_start_frame": args.start_frame,
        "configuration.subvideo_length": args.propainter_subvideo_length,
        "source.mask.path": str(paths["mask"] / arm_mask.MASK_FILENAME),
        "source.source_video.path": str(source.source_video),
    }
    propainter_root = args.propainter_dir.expanduser().resolve()
    propainter_python = args.propainter_python.expanduser().resolve()
    if _identity_required("inpaint", selected, dependencies):
        source_tree_root = str(propainter_root)
        source_tree_sha256 = "__unresolved__"
        try:
            source_tree = propainter.source_tree_identity(propainter_root)
        except (OSError, ValueError):
            pass
        else:
            source_tree_root = source_tree["root"]
            source_tree_sha256 = source_tree["tree_sha256"]
        inpaint_expected["source.implementation.source_tree.root"] = source_tree_root
        inpaint_expected["source.implementation.source_tree.tree_sha256"] = (
            source_tree_sha256
        )
    _bind_artifact(
        inpaint_expected,
        "source.implementation.inference_script",
        propainter_root / "inference_propainter.py",
        include_identity=True,
    )
    _bind_artifact(
        inpaint_expected,
        "source.implementation.python",
        propainter_python,
        include_identity=True,
    )
    for filename in propainter.PROPAINTER_WEIGHT_FILENAMES:
        _bind_artifact(
            inpaint_expected,
            f"source.implementation.weights.{filename}",
            propainter_root / "weights" / filename,
            include_identity=True,
        )

    composite_expected: dict[str, Any] = {
        "schema_version": COMPOSITE_SCHEMA,
        "depth_guard_m": args.depth_guard_m,
        "base_start_frame": background_start_frame,
    }
    _bind_artifact(
        composite_expected,
        "source.base_video",
        background,
        include_identity=external_background,
    )
    if args.object_mask is None or args.object_depth is None:
        composite_expected["source.object_mask"] = None
        composite_expected["source.object_depth"] = None
    else:
        _bind_artifact(
            composite_expected,
            "source.object_mask",
            args.object_mask.expanduser().resolve(),
            include_identity=True,
        )
        _bind_artifact(
            composite_expected,
            "source.object_depth",
            args.object_depth.expanduser().resolve(),
            include_identity=True,
        )

    review_expected: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA,
        "geometry.frame_count": count,
        "source_offsets.source_start_frame": args.start_frame,
        "source_offsets.mask_start_frame": mask_start_frame,
        "source_offsets.background_start_frame": background_start_frame,
        "source_offsets.overlay_start_frame": 0,
        "source.source_video.path": str(source.source_video),
        "source.overlay.path": str(paths["composite"] / "final_overlay.mp4"),
    }
    _bind_artifact(
        review_expected,
        "source.background",
        background,
        include_identity=external_background,
    )
    _bind_artifact(
        review_expected,
        "source.mask_preview",
        mask_preview,
        include_identity=external_mask_preview,
    )
    expected = {
        "tracking": tracking_expected,
        "mask": mask_expected,
        "inpaint": inpaint_expected,
        "retarget": {
            "schema_version": mecka_parallel_jaw.RUN_SCHEMA,
            "algorithm.version": "thumb-index-palm-default/v1",
            "algorithm.conditioning.jump_k": args.jump_k,
            "algorithm.conditioning.max_gap": args.max_gap,
            "algorithm.conditioning.smooth_window": args.smooth_window,
            "algorithm.conditioning.smooth_poly": args.smooth_poly,
            "algorithm.orientation_stability.palm_ratio_max": (args.palm_ratio_max),
            "algorithm.orientation_stability.tip_ratio_min": args.tip_ratio_min,
            "algorithm.orientation_stability.rotation_alpha": (args.rotation_alpha),
            "algorithm.orientation_stability.max_rotation_step_deg": (
                args.max_rotation_step_deg
            ),
        },
        "render": {
            "schema_version": panda_render.ROBOT_RENDER_SCHEMA,
            "depth_emitted": args.emit_depth,
            "ik.backend": args.ik,
            "ik.orientation_weight": args.orientation_weight,
            "ik.max_joint_step_limit_rad": args.max_joint_step_rad,
        },
        "composite": composite_expected,
        "review": review_expected,
    }
    stage_states, raw_current, dependency_current = _plan_stage_states(
        selected=selected,
        dependencies=dependencies,
        stage_metadata=stage_metadata,
        published_outputs=_published_outputs(paths),
        expected=expected,
        overwrite=args.overwrite,
        blockers=blockers,
    )
    blockers.extend(_dependency_preflight(args, stage_states))
    stages = [
        {
            "name": stage,
            "state": stage_states[stage],
            "metadata": str(stage_metadata[stage]),
            "dependencies": list(dependencies[stage]),
            "raw_current": raw_current[stage],
            "dependency_current": dependency_current[stage],
        }
        for stage in selected
    ]
    return {
        "schema_version": PIPELINE_SCHEMA,
        "mode": "execute" if args.execute else "plan",
        "source_kind": source.kind,
        "dataset_uri": source.dataset_uri,
        "episode_index": source.episode_index,
        "task_id": source.task_id,
        "source_video": str(source.source_video),
        "source_parquet": source.source_parquet,
        "frame_window": {"start": args.start_frame, "count": count},
        "geometry": {
            "width": source.width,
            "height": source.height,
            "fps": source.fps,
        },
        "ik_backend": args.ik,
        "background": {
            "path": str(background),
            "start_frame": background_start_frame,
            "external": external_background,
        },
        "mask_preview": {
            "path": str(mask_preview),
            "start_frame": mask_start_frame,
            "external": external_mask_preview,
        },
        "stages": stages,
        "blockers": blockers,
    }


def _open_at(path: Path, frame_index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot decode {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    return capture


def _review_video(
    *,
    source_video: Path,
    source_start_frame: int,
    mask_preview: Path | None,
    mask_start_frame: int,
    background: Path,
    background_start_frame: int,
    overlay: Path,
    output_video: Path,
    output_metadata: Path,
    frame_count: int,
    width: int,
    height: int,
    fps: float,
    overwrite: bool,
) -> dict[str, Any]:
    if not overwrite and (output_video.exists() or output_metadata.exists()):
        raise FileExistsError("Refusing to overwrite the review artifact")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_video.with_name(".four_stage_compare.partial.mp4")
    temporary.unlink(missing_ok=True)
    sources = [
        _open_at(source_video, source_start_frame),
        _open_at(
            mask_preview or source_video,
            mask_start_frame if mask_preview else source_start_frame,
        ),
        _open_at(background, background_start_frame),
        _open_at(overlay, 0),
    ]
    writer = Mp4Writer(temporary, fps, (width * 4, height))
    labels = ("SOURCE", "ARM MASK", "INPAINT", "PANDA")
    try:
        for frame_index in range(frame_count):
            panels: list[np.ndarray] = []
            for capture, label in zip(sources, labels, strict=True):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Review source ended at frame {frame_index}")
                if frame.shape[:2] != (height, width):
                    raise ValueError("Review source geometry differs from source video")
                cv2.putText(
                    frame,
                    label,
                    (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(frame)
            writer.write(np.hstack(panels))
    finally:
        writer.close()
        for capture in sources:
            capture.release()
    os.replace(temporary, output_video)
    metadata = {
        "schema_version": REVIEW_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mask_panel_fallback": mask_preview is None,
        "source_offsets": {
            "source_start_frame": source_start_frame,
            "mask_start_frame": (
                mask_start_frame if mask_preview is not None else source_start_frame
            ),
            "background_start_frame": background_start_frame,
            "overlay_start_frame": 0,
        },
        "geometry": {
            "frame_count": frame_count,
            "panel_width": width,
            "height": height,
            "fps": fps,
        },
        "source": {
            "source_video": artifact(source_video),
            "mask_preview": artifact(mask_preview) if mask_preview else None,
            "background": artifact(background),
            "overlay": artifact(overlay),
            "implementation": artifact(__file__),
        },
        "output": {"video": artifact(output_video)},
    }
    write_json_atomic(output_metadata, metadata)
    return metadata


def execute_plan(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    """Execute pending selected stages in dependency order."""
    if plan["blockers"]:
        raise RuntimeError("Preflight blocked: " + "; ".join(plan["blockers"]))
    output = args.output_dir.expanduser().resolve()
    paths = _layout(output)
    geometry = plan["geometry"]
    frame_count = int(plan["frame_window"]["count"])
    selected = {stage["name"]: stage["state"] for stage in plan["stages"]}
    results: dict[str, Any] = {}

    def stage_overwrite(stage: str) -> bool:
        return args.overwrite or selected.get(stage) == "refresh_dependency"

    if selected.get("tracking") not in {None, "skipped_complete"}:
        if plan["source_kind"] == "lerobot":
            results["tracking"] = mecka_lerobot.execute(
                dataset_uri=plan["dataset_uri"],
                episode=None if args.episode is None else int(args.episode),
                output_dir=paths["tracking"],
                start_frame=args.start_frame,
                max_frames=frame_count,
                credentials=args.credentials,
                overwrite=stage_overwrite("tracking"),
            )
        else:
            results["tracking"] = mecka.execute(
                dataset_dir=args.dataset,
                episode=args.episode,
                output_dir=paths["tracking"],
                start_frame=args.start_frame,
                max_frames=frame_count,
                overwrite=stage_overwrite("tracking"),
            )
    if selected.get("mask") not in {None, "skipped_complete"}:
        results["mask"] = arm_mask.execute(
            tracking_path=paths["tracking"] / mecka.TRACKING_FILENAME,
            tracking_metadata=paths["tracking"] / mecka.METADATA_FILENAME,
            intrinsic_path=paths["tracking"] / mecka.INTRINSIC_FILENAME,
            source_video=Path(plan["source_video"]),
            output_dir=paths["mask"],
            source_start_frame=args.start_frame,
            reconstruction_dir=args.reconstruction_dir,
            config=_arm_mask_config(args),
            sequence_id=_sequence_id(plan),
            episode_index=int(plan["episode_index"]),
            overwrite=stage_overwrite("mask"),
        )
    if selected.get("inpaint") not in {None, "skipped_complete"}:
        results["inpaint"] = propainter.execute(
            source_video=Path(plan["source_video"]),
            mask=paths["mask"] / arm_mask.MASK_FILENAME,
            output_dir=paths["inpaint"],
            source_start_frame=args.start_frame,
            propainter_dir=args.propainter_dir,
            propainter_python=args.propainter_python,
            resize_ratio=args.propainter_resize_ratio,
            subvideo_length=args.propainter_subvideo_length,
            neighbor_length=args.propainter_neighbor_length,
            ref_stride=args.propainter_ref_stride,
            fp16=args.propainter_fp16,
            overwrite=stage_overwrite("inpaint"),
        )
    if selected.get("retarget") not in {None, "skipped_complete"}:
        results["retarget"] = mecka_parallel_jaw.execute(
            tracking=paths["tracking"] / mecka.TRACKING_FILENAME,
            output_dir=paths["retarget"],
            overwrite=stage_overwrite("retarget"),
            jump_k=args.jump_k,
            max_gap=args.max_gap,
            smooth_window=args.smooth_window,
            smooth_poly=args.smooth_poly,
            palm_ratio_max=args.palm_ratio_max,
            tip_ratio_min=args.tip_ratio_min,
            rotation_alpha=args.rotation_alpha,
            max_rotation_step_deg=args.max_rotation_step_deg,
        )
    if selected.get("render") not in {None, "skipped_complete"}:
        results["render"] = panda_render.execute(
            trajectory=paths["retarget"] / mecka_parallel_jaw.TRAJECTORY_FILENAME,
            intrinsic=paths["tracking"] / mecka.INTRINSIC_FILENAME,
            camera_to_world_xyzw=paths["tracking"] / mecka.CAMERA_ROTATION_FILENAME,
            rig_config=args.rig_config,
            output_dir=paths["render"],
            width=int(geometry["width"]),
            height=int(geometry["height"]),
            fps=float(geometry["fps"]),
            panda_dir=args.panda_dir,
            ik_backend=args.ik,
            emit_depth=args.emit_depth,
            orientation_weight=args.orientation_weight,
            max_joint_step_rad=args.max_joint_step_rad,
            overwrite=stage_overwrite("render"),
        )
    if selected.get("composite") not in {None, "skipped_complete"}:
        background = Path(plan["background"]["path"])
        results["composite"] = composite(
            base_video=background,
            robot_video=paths["render"] / panda_render.RGB_FILENAME,
            robot_mask=paths["render"] / panda_render.MASK_FILENAME,
            robot_depth=(
                paths["render"] / panda_render.DEPTH_FILENAME
                if args.emit_depth
                else None
            ),
            robot_metadata=paths["render"] / panda_render.METADATA_FILENAME,
            output_dir=paths["composite"],
            base_start_frame=int(plan["background"]["start_frame"]),
            object_mask=args.object_mask,
            object_depth=args.object_depth,
            depth_guard_m=args.depth_guard_m,
            overwrite=stage_overwrite("composite"),
        )
    if selected.get("review") not in {None, "skipped_complete"}:
        background = Path(plan["background"]["path"])
        mask_preview = Path(plan["mask_preview"]["path"])
        results["review"] = _review_video(
            source_video=Path(plan["source_video"]),
            source_start_frame=args.start_frame,
            mask_preview=mask_preview,
            mask_start_frame=int(plan["mask_preview"]["start_frame"]),
            background=background,
            background_start_frame=int(plan["background"]["start_frame"]),
            overlay=paths["composite"] / "final_overlay.mp4",
            output_video=paths["review_video"],
            output_metadata=paths["review_metadata"],
            frame_count=frame_count,
            width=int(geometry["width"]),
            height=int(geometry["height"]),
            fps=float(geometry["fps"]),
            overwrite=stage_overwrite("review"),
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Local MECKA export, or a LeRobot v3 dataset as a path or s3:// URI. "
        "May point either at one shard root or at a prefix holding many "
        "(v0/shard_00, v0/shard_01, ...), in which case --shard picks one",
    )
    parser.add_argument(
        "--shard",
        help="Shard to read when --dataset is a prefix holding many: an index "
        f"({lerobot_source.SHARD_TEMPLATE.format(index=12)} for 12) or a "
        "directory name (default: the first shard)",
    )
    parser.add_argument(
        "--episode",
        help="Episode index within the selected shard; indices restart at zero "
        "in every shard",
    )
    parser.add_argument(
        "--credentials",
        help=f"JSON credentials for s3:// datasets "
        f"(default {lerobot_source.DEFAULT_CREDENTIALS})",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--background-start-frame", type=int, default=0)
    parser.add_argument("--mask-preview", type=Path)
    parser.add_argument("--mask-start-frame", type=int, default=0)
    parser.add_argument(
        "--reconstruction-dir",
        type=Path,
        default=DEFAULT_RECONSTRUCTION_DIR,
    )
    parser.add_argument(
        "--arm-mask-config",
        type=Path,
        help="Optional JSON overrides for ArmMaskConfig",
    )
    parser.add_argument(
        "--propainter-dir",
        type=Path,
        default=propainter.DEFAULT_PROPAINTER_DIR,
    )
    parser.add_argument(
        "--propainter-python",
        type=Path,
        default=propainter.DEFAULT_PROPAINTER_PYTHON,
    )
    parser.add_argument(
        "--propainter-resize-ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--propainter-subvideo-length",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--propainter-neighbor-length",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--propainter-ref-stride",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--propainter-fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--object-mask", type=Path)
    parser.add_argument("--object-depth", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stage", action="append", choices=STAGES)
    parser.add_argument("--ik", choices=("dls", "hybrid"), default="dls")
    parser.add_argument("--rig-config", required=True, type=Path)
    parser.add_argument(
        "--panda-dir", type=Path, default=panda_render.DEFAULT_PANDA_DIR
    )
    parser.add_argument("--orientation-weight", type=float, default=0.5)
    parser.add_argument(
        "--max-joint-step-rad",
        type=float,
        default=panda_render.DEFAULT_MAX_JOINT_STEP_RAD,
    )
    parser.add_argument("--jump-k", type=float, default=6.0)
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--smooth-poly", type=int, default=2)
    parser.add_argument(
        "--palm-ratio-max",
        type=float,
        default=mecka_parallel_jaw.PALM_RATIO_MAX,
    )
    parser.add_argument(
        "--tip-ratio-min",
        type=float,
        default=mecka_parallel_jaw.TIP_RATIO_MIN,
    )
    parser.add_argument(
        "--rotation-alpha",
        type=float,
        default=mecka_parallel_jaw.ROTATION_ALPHA,
    )
    parser.add_argument(
        "--max-rotation-step-deg",
        type=float,
        default=mecka_parallel_jaw.MAX_ROTATION_STEP_DEG,
    )
    parser.add_argument("--depth-guard-m", type=float, default=0.003)
    parser.add_argument(
        "--emit-depth",
        action="store_true",
        help="Write robot_depth.npy (~8 MB per frame, 19 GB for a 2443-frame "
        "episode); required only for depth-aware compositing",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = build_plan(args)
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return
    results = execute_plan(args, plan)
    print(json.dumps({"executed": sorted(results)}, indent=2))


if __name__ == "__main__":
    main()
