"""Plan, execute, and resume the MECKA-to-Panda inpainting pipeline."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inpainting.adapters import mecka, mecka_parallel_jaw
from inpainting.composite_robot import execute as composite
from inpainting.contracts import artifact, sha256, write_json_atomic
from inpainting.panda_renderer import render as panda_render
from inpainting.video_io import Mp4Writer, probe_video

STAGES = ("tracking", "retarget", "render", "composite", "review")
PIPELINE_SCHEMA = "v2d.inpainting.mecka-panda-pipeline/v1"


def _nested_value(metadata: dict[str, Any], dotted_key: str) -> Any:
    value: Any = metadata
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
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


def _metadata_complete(
    path: Path, expected: dict[str, Any] | None = None
) -> bool:
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
        "retarget": output / "retarget",
        "render": output / "robot_render",
        "composite": output / "composite",
        "review_video": output / "four_stage_compare.mp4",
        "review_metadata": output / "four_stage_compare.json",
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build a read-only execution plan."""
    dataset = args.dataset.expanduser().resolve()
    record = mecka.resolve_episode(mecka.load_manifest(dataset), args.episode)
    source_video = (dataset / record["clip"]).resolve()
    source_parquet = (dataset / record["data"]).resolve()
    geometry = probe_video(source_video)
    count = (
        int(geometry["frame_count"]) - args.start_frame
        if args.max_frames is None
        else min(args.max_frames, int(geometry["frame_count"]) - args.start_frame)
    )
    if count <= 0:
        raise ValueError("Selected frame window is empty")
    output = args.output_dir.expanduser().resolve()
    paths = _layout(output)
    selected = tuple(args.stage or STAGES)
    blockers: list[str] = []
    if not source_parquet.is_file():
        blockers.append(f"missing parquet: {source_parquet}")
    if any(stage in selected for stage in ("composite", "review")):
        if args.background is None:
            blockers.append("--background is required for composite/review")
        elif not args.background.expanduser().resolve().is_file():
            blockers.append(f"missing background: {args.background}")
    if (
        "review" in selected
        and args.mask_preview is not None
        and not args.mask_preview.expanduser().resolve().is_file()
    ):
        blockers.append(f"missing mask preview: {args.mask_preview}")
    stage_metadata = {
        "tracking": paths["tracking"] / mecka.METADATA_FILENAME,
        "retarget": paths["retarget"] / mecka_parallel_jaw.METADATA_FILENAME,
        "render": paths["render"] / panda_render.METADATA_FILENAME,
        "composite": paths["composite"] / "final_overlay.json",
        "review": paths["review_metadata"],
    }
    expected = {
        "tracking": {
            "frame_window.start": args.start_frame,
            "frame_window.count": count,
        },
        "retarget": {
            "algorithm.conditioning.jump_k": args.jump_k,
            "algorithm.conditioning.max_gap": args.max_gap,
            "algorithm.conditioning.smooth_window": args.smooth_window,
            "algorithm.conditioning.smooth_poly": args.smooth_poly,
        },
        "render": {
            "ik.backend": args.ik,
            "ik.orientation_weight": args.orientation_weight,
        },
        "composite": {
            "depth_guard_m": args.depth_guard_m,
            "base_start_frame": args.background_start_frame,
        },
        "review": {
            "geometry.frame_count": count,
        },
    }
    stages = [
        {
            "name": stage,
            "state": (
                "replace"
                if args.overwrite
                else "skipped_complete"
                if _metadata_complete(stage_metadata[stage], expected[stage])
                else "pending"
            ),
            "metadata": str(stage_metadata[stage]),
        }
        for stage in selected
    ]
    return {
        "schema_version": PIPELINE_SCHEMA,
        "mode": "execute" if args.execute else "plan",
        "episode_index": int(record["episode_index"]),
        "task_id": str(record.get("task_id", "")),
        "source_video": str(source_video),
        "source_parquet": str(source_parquet),
        "frame_window": {"start": args.start_frame, "count": count},
        "geometry": {
            "width": int(geometry["width"]),
            "height": int(geometry["height"]),
            "fps": float(geometry["fps"]),
        },
        "ik_backend": args.ik,
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
        _open_at(mask_preview or source_video, mask_start_frame if mask_preview else source_start_frame),
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
        "schema_version": "v2d.inpainting.four-stage-review/v1",
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mask_panel_fallback": mask_preview is None,
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

    if selected.get("tracking") not in {None, "skipped_complete"}:
        results["tracking"] = mecka.execute(
            dataset_dir=args.dataset,
            episode=args.episode,
            output_dir=paths["tracking"],
            start_frame=args.start_frame,
            max_frames=frame_count,
            overwrite=args.overwrite,
        )
    if selected.get("retarget") not in {None, "skipped_complete"}:
        results["retarget"] = mecka_parallel_jaw.execute(
            tracking=paths["tracking"] / mecka.TRACKING_FILENAME,
            output_dir=paths["retarget"],
            overwrite=args.overwrite,
            jump_k=args.jump_k,
            max_gap=args.max_gap,
            smooth_window=args.smooth_window,
            smooth_poly=args.smooth_poly,
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
            orientation_weight=args.orientation_weight,
            overwrite=args.overwrite,
        )
    if selected.get("composite") not in {None, "skipped_complete"}:
        assert args.background is not None
        results["composite"] = composite(
            base_video=args.background,
            robot_video=paths["render"] / panda_render.RGB_FILENAME,
            robot_mask=paths["render"] / panda_render.MASK_FILENAME,
            robot_depth=paths["render"] / panda_render.DEPTH_FILENAME,
            robot_metadata=paths["render"] / panda_render.METADATA_FILENAME,
            output_dir=paths["composite"],
            base_start_frame=args.background_start_frame,
            object_mask=args.object_mask,
            object_depth=args.object_depth,
            depth_guard_m=args.depth_guard_m,
            overwrite=args.overwrite,
        )
    if selected.get("review") not in {None, "skipped_complete"}:
        assert args.background is not None
        results["review"] = _review_video(
            source_video=Path(plan["source_video"]),
            source_start_frame=args.start_frame,
            mask_preview=args.mask_preview,
            mask_start_frame=args.mask_start_frame,
            background=args.background.expanduser().resolve(),
            background_start_frame=args.background_start_frame,
            overlay=paths["composite"] / "final_overlay.mp4",
            output_video=paths["review_video"],
            output_metadata=paths["review_metadata"],
            frame_count=frame_count,
            width=int(geometry["width"]),
            height=int(geometry["height"]),
            fps=float(geometry["fps"]),
            overwrite=args.overwrite,
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--background-start-frame", type=int, default=0)
    parser.add_argument("--mask-preview", type=Path)
    parser.add_argument("--mask-start-frame", type=int, default=0)
    parser.add_argument("--object-mask", type=Path)
    parser.add_argument("--object-depth", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stage", action="append", choices=STAGES)
    parser.add_argument("--ik", choices=("dls", "hybrid"), default="dls")
    parser.add_argument("--rig-config", required=True, type=Path)
    parser.add_argument("--panda-dir", type=Path, default=panda_render.DEFAULT_PANDA_DIR)
    parser.add_argument("--orientation-weight", type=float, default=0.5)
    parser.add_argument("--jump-k", type=float, default=6.0)
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--smooth-poly", type=int, default=2)
    parser.add_argument("--depth-guard-m", type=float, default=0.003)
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

