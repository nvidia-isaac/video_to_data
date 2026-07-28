"""Adapt one MECKA episode from a LeRobot v3 dataset to the tracking contract.

Emits the same artifacts as `inpainting.adapters.mecka`, so downstream stages do
not care whether the episode came from a local MECKA export or from a LeRobot
shard on object storage. The frame arrays are built by the shared
`build_tracking_arrays`, because both layouts expose identical column names.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inpainting.adapters.mecka import (
    CALIBRATION_HEIGHT,
    CALIBRATION_WIDTH,
    CAMERA_ROTATION_FILENAME,
    INTRINSIC_FILENAME,
    METADATA_FILENAME,
    TRACKING_FILENAME,
    build_tracking_arrays,
)
from inpainting.mecka_panda import lerobot_source
from inpainting.mecka_panda.contracts import (
    artifact,
    load_npz,
    validate_tracking_arrays,
    write_json_atomic,
    write_npy_atomic,
    write_npz_atomic,
)

VIDEO_FILENAME = "source_video.mp4"
RUN_SCHEMA = "v2d.inpainting.mecka-lerobot-tracking-run/v1"

REQUIRED_COLUMNS = [
    "frame_index",
    "observation.state.hand_left_cam",
    "observation.state.hand_right_cam",
    "observation.state.hand_left_cam_rotation",
    "observation.state.hand_right_cam_rotation",
    "observation.state.camera_rotation",
]


def _probe(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot decode {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return frames, width, height, fps


def expected_video_geometry(info: dict[str, Any]) -> tuple[int, int, float]:
    """Return ``(width, height, fps)`` from LeRobot ``info.json`` metadata."""
    try:
        feature = info["features"][lerobot_source.VIDEO_KEY]
        shape = feature["shape"]
        fps = float(info["fps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"LeRobot info.json is missing valid metadata for "
            f"{lerobot_source.VIDEO_KEY}"
        ) from exc

    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in shape
        )
    ):
        raise ValueError(
            f"{lerobot_source.VIDEO_KEY} shape must contain three positive integers"
        )
    dimensions = tuple(int(value) for value in shape)

    names = feature.get("names")
    if names is not None:
        if not isinstance(names, (list, tuple)) or len(names) != 3:
            raise ValueError(
                f"{lerobot_source.VIDEO_KEY} names must describe all three axes"
            )
        normalized_names = tuple(str(name).lower() for name in names)
        required = {"height", "width", "channel"}
        if set(normalized_names) != required:
            raise ValueError(
                f"{lerobot_source.VIDEO_KEY} names must contain "
                "height, width, and channel"
            )
        height = dimensions[normalized_names.index("height")]
        width = dimensions[normalized_names.index("width")]
    else:
        channel_axes = [
            axis for axis, dimension in enumerate(dimensions) if dimension in (1, 3, 4)
        ]
        if channel_axes == [0]:
            _, height, width = dimensions
        elif channel_axes == [2]:
            height, width, _ = dimensions
        else:
            raise ValueError(
                f"Cannot infer channel axis from {lerobot_source.VIDEO_KEY} "
                f"shape {list(dimensions)}"
            )

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("LeRobot info.json fps must be finite and positive")
    return width, height, fps


def validate_video_geometry(
    info: dict[str, Any], *, width: int, height: int, fps: float
) -> None:
    """Hard-fail when decoded video geometry disagrees with ``info.json``."""
    expected_width, expected_height, expected_fps = expected_video_geometry(info)
    fps_matches = np.isfinite(fps) and np.isclose(
        fps, expected_fps, atol=1e-3, rtol=0.0
    )
    if width != expected_width or height != expected_height or not fps_matches:
        raise ValueError(
            f"Decoded video geometry {width}x{height} @ {fps:g} fps does not match "
            f"LeRobot info.json {expected_width}x{expected_height} @ "
            f"{expected_fps:g} fps"
        )


def intrinsic_matrix(
    intrinsics: tuple[float, ...] | np.ndarray, *, width: int, height: int
) -> np.ndarray:
    """Scale the 1920x1080 calibration to the decoded video resolution."""
    values = np.asarray(intrinsics, dtype=np.float64)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("camera_intrinsics must contain fx,fy,cx,cy,k1,k2,p1,p2")
    scale_x = width / CALIBRATION_WIDTH
    scale_y = height / CALIBRATION_HEIGHT
    return np.asarray(
        [
            [values[0] * scale_x, 0.0, values[2] * scale_x],
            [0.0, values[1] * scale_y, values[3] * scale_y],
            [0.0, 0.0, 1.0],
        ]
    )


def camera_rotations(table: Any, *, start_frame: int, frame_count: int) -> np.ndarray:
    """Extract per-frame xyzw camera quaternions for the tracking window."""
    frame = table.to_pandas().sort_values("frame_index").reset_index(drop=True)
    rotations = np.stack(
        frame["observation.state.camera_rotation"].to_numpy()[
            start_frame : start_frame + frame_count
        ]
    ).astype(np.float64)
    if rotations.shape != (frame_count, 4):
        raise ValueError("camera_rotation must contain one xyzw quaternion per frame")
    norms = np.linalg.norm(rotations, axis=1)
    if not np.isfinite(rotations).all() or np.any(norms < 1e-8):
        raise ValueError("camera_rotation contains invalid quaternions")
    return rotations / norms[:, None]


def execute(
    *,
    dataset_uri: str | Path,
    episode: int | None,
    output_dir: str | Path,
    start_frame: int = 0,
    max_frames: int | None = None,
    credentials: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Adapt one LeRobot episode and atomically commit the NPZ then JSON marker."""
    output = Path(output_dir).expanduser().resolve()
    tracking_path = output / TRACKING_FILENAME
    intrinsic_path = output / INTRINSIC_FILENAME
    camera_rotation_path = output / CAMERA_ROTATION_FILENAME
    metadata_path = output / METADATA_FILENAME
    video_path = output / VIDEO_FILENAME
    existing = [
        path
        for path in (tracking_path, intrinsic_path, camera_rotation_path, metadata_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    source = lerobot_source.open_source(dataset_uri, credentials=credentials)
    info = lerobot_source.load_info(source)
    record = lerobot_source.find_episode(source, info, episode)
    table = lerobot_source.read_episode_table(
        source, info, record, columns=REQUIRED_COLUMNS
    )
    if table.num_rows != record.length:
        raise ValueError(
            f"Episode {record.episode_index} declares {record.length} frames but "
            f"the parquet slice holds {table.num_rows}"
        )

    output.mkdir(parents=True, exist_ok=True)
    lerobot_source.extract_episode_video(
        source, info, record, video_path, overwrite=overwrite
    )
    video_frames, width, height, fps = _probe(video_path)
    if video_frames != record.length:
        raise ValueError(
            f"Extracted video holds {video_frames} frames but episode "
            f"{record.episode_index} declares {record.length}"
        )

    validate_video_geometry(info, width=width, height=height, fps=fps)
    arrays = build_tracking_arrays(
        table.to_pandas(), start_frame=start_frame, max_frames=max_frames
    )
    frame_count = validate_tracking_arrays(arrays)
    if start_frame + frame_count > video_frames:
        raise ValueError("Tracking frame window exceeds the extracted video")

    matrix = intrinsic_matrix(record.camera_intrinsics, width=width, height=height)
    rotations = camera_rotations(
        table, start_frame=start_frame, frame_count=frame_count
    )

    write_npz_atomic(tracking_path, arrays)
    write_npy_atomic(intrinsic_path, matrix)
    write_npy_atomic(camera_rotation_path, rotations)
    validate_tracking_arrays(load_npz(tracking_path))
    metadata = {
        "schema_version": RUN_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "episode_index": record.episode_index,
        "task_id": record.task_id,
        "task_description": record.task_description,
        "frame_window": {"start": start_frame, "count": frame_count},
        "video_geometry": {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "camera_intrinsics_full_resolution": list(record.camera_intrinsics),
        "source": {
            "dataset_uri": str(dataset_uri),
            "codebase_version": info["codebase_version"],
            "data_rows": [record.row_start, record.row_stop],
            "video_time_range_s": [record.video_from_s, record.video_to_s],
            "video": artifact(video_path),
            "implementation": artifact(__file__),
        },
        "output": {
            "tracking": artifact(tracking_path),
            "intrinsic": artifact(intrinsic_path),
            "camera_to_world_xyzw": artifact(camera_rotation_path),
        },
        "valid_counts": {
            side: int(np.count_nonzero(arrays[f"{side}_valid"]))
            for side in ("left", "right")
        },
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="LeRobot v3 shard root, either a local path or s3://bucket/prefix",
    )
    parser.add_argument("--episode", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--credentials",
        help=f"JSON credentials file (default {lerobot_source.DEFAULT_CREDENTIALS})",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    metadata = execute(
        dataset_uri=args.dataset,
        episode=args.episode,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        credentials=args.credentials,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
