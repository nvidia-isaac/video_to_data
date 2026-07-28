"""Adapt one MECKA episode to the common hand-tracking contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from inpainting.mecka_panda.contracts import (
    TRACKING_SCHEMA,
    artifact,
    load_npz,
    validate_tracking_arrays,
    write_json_atomic,
    write_npy_atomic,
    write_npz_atomic,
)

TRACKING_FILENAME = "tracking.npz"
INTRINSIC_FILENAME = "intrinsic.npy"
CAMERA_ROTATION_FILENAME = "camera_to_world_xyzw.npy"
METADATA_FILENAME = "tracking.json"
RUN_SCHEMA = "v2d.inpainting.mecka-tracking-run/v1"
CALIBRATION_WIDTH = 1920
CALIBRATION_HEIGHT = 1080


def load_manifest(dataset_dir: str | Path) -> dict[int, dict[str, Any]]:
    """Load the MECKA JSONL manifest indexed by episode number."""
    dataset = Path(dataset_dir).expanduser().resolve()
    records: dict[int, dict[str, Any]] = {}
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[int(record["episode_index"])] = record
    if not records:
        raise ValueError(f"No episodes found under {dataset}")
    return records


def resolve_episode(
    records: dict[int, dict[str, Any]], episode: str | int | None
) -> dict[str, Any]:
    """Resolve an episode index or clip stem."""
    if episode is None:
        return records[min(records)]
    try:
        return records[int(episode)]
    except (KeyError, TypeError, ValueError):
        requested = Path(str(episode)).stem
        for record in records.values():
            if Path(record["clip"]).stem == requested:
                return record
    raise ValueError(f"Episode {episode!r} is not present in the manifest")


def _hand_present(points: np.ndarray) -> bool:
    return bool(np.isfinite(points).all() and np.abs(points).max(initial=0.0) > 1e-6)


def _normalize_xyzw_batch(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Convert MECKA xyzw quaternions to contract wxyz with NaN invalid rows."""
    values = np.asarray(values, dtype=np.float64).reshape(-1, 4)
    output = np.full_like(values, np.nan)
    norms = np.linalg.norm(values, axis=1)
    usable = valid & np.isfinite(values).all(axis=1) & (norms > 1e-8)
    normalized = values[usable] / norms[usable, None]
    output[usable] = normalized[:, [3, 0, 1, 2]]
    valid &= usable
    return output


def build_tracking_arrays(
    frame_table: pd.DataFrame,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Build validated camera-frame arrays from a MECKA parquet table."""
    table = frame_table.sort_values("frame_index").reset_index(drop=True)
    stop = len(table) if max_frames is None else min(len(table), start_frame + max_frames)
    if start_frame < 0 or stop <= start_frame:
        raise ValueError(f"Invalid frame window [{start_frame}, {stop})")
    table = table.iloc[start_frame:stop].reset_index(drop=True)
    frame_count = len(table)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray("mecka"),
        "coordinate_frame": np.asarray("camera"),
        "frame_indices": np.arange(frame_count, dtype=np.int64),
    }
    for side in ("left", "right"):
        points = np.stack(
            table[f"observation.state.hand_{side}_cam"].to_numpy()
        ).reshape(frame_count, 21, 3).astype(np.float64)
        valid = np.asarray([_hand_present(row) for row in points], dtype=np.bool_)
        rotations = np.stack(
            table[f"observation.state.hand_{side}_cam_rotation"].to_numpy()
        ).reshape(frame_count, 21, 4)
        wrist_wxyz = _normalize_xyzw_batch(rotations[:, 0], valid)
        points[~valid] = np.nan
        arrays[f"{side}_valid"] = valid
        arrays[f"{side}_wrist_position"] = points[:, 0].copy()
        arrays[f"{side}_wrist_wxyz"] = wrist_wxyz
        arrays[f"{side}_joints_3d"] = points
    validate_tracking_arrays(arrays)
    return arrays


def execute(
    *,
    dataset_dir: str | Path,
    episode: str | int | None,
    output_dir: str | Path,
    start_frame: int = 0,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Adapt one episode and atomically commit the NPZ then JSON marker."""
    dataset = Path(dataset_dir).expanduser().resolve()
    record = resolve_episode(load_manifest(dataset), episode)
    parquet = (dataset / record["data"]).resolve()
    video = (dataset / record["clip"]).resolve()
    output = Path(output_dir).expanduser().resolve()
    tracking_path = output / TRACKING_FILENAME
    intrinsic_path = output / INTRINSIC_FILENAME
    camera_rotation_path = output / CAMERA_ROTATION_FILENAME
    metadata_path = output / METADATA_FILENAME
    outputs = (
        tracking_path,
        intrinsic_path,
        camera_rotation_path,
        metadata_path,
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    table = pd.read_parquet(parquet)
    arrays = build_tracking_arrays(
        table, start_frame=start_frame, max_frames=max_frames
    )
    frame_count = validate_tracking_arrays(arrays)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot decode {video}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if start_frame + frame_count > video_frames:
        raise ValueError("Tracking frame window exceeds source video")

    intrinsics = np.asarray(record["camera_intrinsics"], dtype=np.float64)
    if intrinsics.shape != (8,) or not np.isfinite(intrinsics).all():
        raise ValueError("camera_intrinsics must contain fx,fy,cx,cy,k1,k2,p1,p2")
    scale_x = width / CALIBRATION_WIDTH
    scale_y = height / CALIBRATION_HEIGHT
    intrinsic_matrix = np.asarray(
        [
            [intrinsics[0] * scale_x, 0.0, intrinsics[2] * scale_x],
            [0.0, intrinsics[1] * scale_y, intrinsics[3] * scale_y],
            [0.0, 0.0, 1.0],
        ]
    )
    sorted_table = table.sort_values("frame_index").reset_index(drop=True)
    camera_rotations = np.stack(
        sorted_table["observation.state.camera_rotation"].to_numpy()[
            start_frame : start_frame + frame_count
        ]
    ).astype(np.float64)
    if camera_rotations.shape != (frame_count, 4):
        raise ValueError("camera_rotation must contain one xyzw quaternion per frame")
    norms = np.linalg.norm(camera_rotations, axis=1)
    if not np.isfinite(camera_rotations).all() or np.any(norms < 1e-8):
        raise ValueError("camera_rotation contains invalid quaternions")
    camera_rotations /= norms[:, None]

    output.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(tracking_path, arrays)
    write_npy_atomic(intrinsic_path, intrinsic_matrix)
    write_npy_atomic(camera_rotation_path, camera_rotations)
    validate_tracking_arrays(load_npz(tracking_path))
    metadata = {
        "schema_version": RUN_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "episode_index": int(record["episode_index"]),
        "task_id": str(record.get("task_id", "")),
        "frame_window": {"start": start_frame, "count": frame_count},
        "video_geometry": {
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "camera_intrinsics_full_resolution": list(record["camera_intrinsics"]),
        "source": {
            "parquet": artifact(parquet),
            "video": artifact(video),
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
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    metadata = execute(
        dataset_dir=args.dataset,
        episode=args.episode,
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

