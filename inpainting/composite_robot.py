"""Composite a validated robot render over a hand-removed base video."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inpainting.contracts import (
    ROBOT_RENDER_SCHEMA,
    artifact,
    sha256,
    write_json_atomic,
)
from inpainting.video_io import Mp4Writer, probe_video

COMPOSITE_SCHEMA = "v2d.inpainting.robot-composite/v1"
OUTPUT_FILENAME = "final_overlay.mp4"
METADATA_FILENAME = "final_overlay.json"


def depth_visible_robot_mask(
    robot_mask: np.ndarray,
    robot_depth: np.ndarray,
    object_mask: np.ndarray,
    object_depth: np.ndarray,
    depth_guard_m: float = 0.003,
) -> np.ndarray:
    """Apply the common metric camera-z object visibility rule."""
    valid_object = object_mask & np.isfinite(object_depth) & (object_depth > 0)
    return robot_mask & (
        ~valid_object | (robot_depth <= object_depth + depth_guard_m)
    )


def _validate_render_bundle(
    metadata_path: Path,
    rgb_path: Path,
    mask_path: Path,
    depth_path: Path,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != ROBOT_RENDER_SCHEMA:
        raise ValueError("Unsupported robot render metadata schema")
    if metadata.get("state") != "complete":
        raise ValueError("Robot render bundle is not complete")
    for key, path in (("rgb", rgb_path), ("mask", mask_path), ("depth", depth_path)):
        expected = metadata["output"][key]
        if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Robot render {key} does not match its metadata")
    return metadata


def execute(
    *,
    base_video: str | Path,
    robot_video: str | Path,
    robot_mask: str | Path,
    robot_depth: str | Path,
    robot_metadata: str | Path,
    output_dir: str | Path,
    base_start_frame: int = 0,
    object_mask: str | Path | None = None,
    object_depth: str | Path | None = None,
    depth_guard_m: float = 0.003,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Composite one bundle and atomically publish video then metadata."""
    if (object_mask is None) != (object_depth is None):
        raise ValueError("object_mask and object_depth must be supplied together")
    if depth_guard_m < 0:
        raise ValueError("depth_guard_m must be non-negative")
    base_path = Path(base_video).expanduser().resolve()
    rgb_path = Path(robot_video).expanduser().resolve()
    mask_path = Path(robot_mask).expanduser().resolve()
    depth_path = Path(robot_depth).expanduser().resolve()
    render_metadata_path = Path(robot_metadata).expanduser().resolve()
    render_metadata = _validate_render_bundle(
        render_metadata_path, rgb_path, mask_path, depth_path
    )
    geometry = render_metadata["geometry"]
    frame_count = int(geometry["frame_count"])
    width = int(geometry["width"])
    height = int(geometry["height"])
    fps = float(geometry["fps"])
    base_geometry = probe_video(base_path)
    if base_start_frame < 0 or base_start_frame + frame_count > base_geometry["frame_count"]:
        raise ValueError("Base video does not cover the robot frame window")
    if (base_geometry["width"], base_geometry["height"]) != (width, height):
        raise ValueError("Base and robot video geometry differ")
    if not np.isclose(float(base_geometry["fps"]), fps, atol=1e-3):
        raise ValueError("Base and robot video FPS differ")

    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (frame_count, height, width)
    if mask.shape != expected_shape or mask.dtype != np.bool_:
        raise ValueError(f"robot_mask must be bool with shape {expected_shape}")
    if depth.shape != expected_shape or depth.dtype != np.float32:
        raise ValueError(f"robot_depth must be float32 with shape {expected_shape}")
    object_mask_array = object_depth_array = None
    if object_mask is not None and object_depth is not None:
        object_mask_array = np.load(object_mask, mmap_mode="r", allow_pickle=False)
        object_depth_array = np.load(object_depth, mmap_mode="r", allow_pickle=False)
        if object_mask_array.shape != expected_shape or object_mask_array.dtype != np.bool_:
            raise ValueError(f"object_mask must be bool with shape {expected_shape}")
        if object_depth_array.shape != expected_shape:
            raise ValueError(f"object_depth must have shape {expected_shape}")

    output = Path(output_dir).expanduser().resolve()
    output_video = output / OUTPUT_FILENAME
    output_metadata = output / METADATA_FILENAME
    if not overwrite and (output_video.exists() or output_metadata.exists()):
        raise FileExistsError("Refusing to overwrite the existing composite")
    output.mkdir(parents=True, exist_ok=True)
    temporary_video = output / ".final_overlay.partial.mp4"
    temporary_video.unlink(missing_ok=True)
    base_capture = cv2.VideoCapture(str(base_path))
    robot_capture = cv2.VideoCapture(str(rgb_path))
    base_capture.set(cv2.CAP_PROP_POS_FRAMES, base_start_frame)
    writer = Mp4Writer(temporary_video, fps, (width, height))
    visible_pixels = 0
    robot_pixels = 0
    try:
        for frame in range(frame_count):
            base_ok, base = base_capture.read()
            robot_ok, robot = robot_capture.read()
            if not base_ok or not robot_ok:
                raise RuntimeError(f"Video decode ended at frame {frame}")
            visible = np.asarray(mask[frame])
            if object_mask_array is not None and object_depth_array is not None:
                visible = depth_visible_robot_mask(
                    visible,
                    np.asarray(depth[frame]),
                    np.asarray(object_mask_array[frame]),
                    np.asarray(object_depth_array[frame]),
                    depth_guard_m,
                )
            result = base.copy()
            result[visible] = robot[visible]
            writer.write(result)
            visible_pixels += int(visible.sum())
            robot_pixels += int(mask[frame].sum())
    finally:
        writer.close()
        base_capture.release()
        robot_capture.release()
    os.replace(temporary_video, output_video)
    metadata = {
        "schema_version": COMPOSITE_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "mode": "depth_aware" if object_mask is not None else "hard_mask",
        "depth_guard_m": depth_guard_m,
        "base_start_frame": base_start_frame,
        "geometry": geometry,
        "statistics": {
            "robot_pixels": robot_pixels,
            "visible_robot_pixels": visible_pixels,
            "visible_fraction": visible_pixels / max(robot_pixels, 1),
        },
        "source": {
            "base_video": artifact(base_path),
            "robot_metadata": artifact(render_metadata_path),
            "object_mask": artifact(object_mask) if object_mask is not None else None,
            "object_depth": artifact(object_depth) if object_depth is not None else None,
            "implementation": artifact(__file__),
        },
        "output": {"video": artifact(output_video)},
    }
    write_json_atomic(output_metadata, metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-video", required=True, type=Path)
    parser.add_argument("--robot-video", required=True, type=Path)
    parser.add_argument("--robot-mask", required=True, type=Path)
    parser.add_argument("--robot-depth", required=True, type=Path)
    parser.add_argument("--robot-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-start-frame", type=int, default=0)
    parser.add_argument("--object-mask", type=Path)
    parser.add_argument("--object-depth", type=Path)
    parser.add_argument("--depth-guard-m", type=float, default=0.003)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    metadata = execute(
        base_video=args.base_video,
        robot_video=args.robot_video,
        robot_mask=args.robot_mask,
        robot_depth=args.robot_depth,
        robot_metadata=args.robot_metadata,
        output_dir=args.output_dir,
        base_start_frame=args.base_start_frame,
        object_mask=args.object_mask,
        object_depth=args.object_depth,
        depth_guard_m=args.depth_guard_m,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

