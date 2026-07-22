"""Resolve and validate all immutable inputs for an inpainting experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import EXPERIMENT_SCHEMA, RESOLVED_EXPERIMENT_SCHEMA, TRACKERS
from .video_io import probe_video


MOTION_LENGTH_COLUMNS = (
    "mano_right_trans",
    "mano_left_trans",
    "robot_right_wrist_position",
    "robot_left_wrist_position",
    "robot_right_finger_joints",
    "robot_left_finger_joints",
)


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _single_motion_file(motion_root: Path, sequence_id: str, gripper: str) -> Path:
    directory = motion_root / f"sequence_id={sequence_id}" / f"robot_name={gripper}"
    matches = sorted(directory.glob("*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one motion parquet under {directory}, found {len(matches)}"
        )
    return matches[0].resolve()


def _motion_lengths(path: Path) -> dict[str, int]:
    schema_names = set(pq.read_schema(path).names)
    missing = sorted(set(MOTION_LENGTH_COLUMNS) - schema_names)
    if missing:
        raise ValueError(f"Motion parquet {path} is missing required columns: {missing}")
    columns = list(MOTION_LENGTH_COLUMNS)
    table = pq.read_table(path, columns=columns)
    if table.num_rows != 1:
        raise ValueError(f"Expected one sequence row in {path}, got {table.num_rows}")
    result: dict[str, int] = {}
    for name in columns:
        value = table[name][0].as_py()
        result[name] = len(value)
    return result


def _camera_paths(camera_root: Path, relative_video: Path) -> dict[str, Any]:
    # RGB mapping entries are <triplet>/<sequence>/color.mp4. Camera parameters
    # use the same triplet/sequence nesting in the official TACO release.
    relative_sequence = relative_video.parent
    directory = (camera_root / relative_sequence).resolve()
    intrinsic = directory / "egocentric_intrinsic.txt"
    extrinsic = directory / "egocentric_frame_extrinsic.npy"
    return {
        "directory": str(directory),
        "intrinsic": str(intrinsic),
        "extrinsic": str(extrinsic),
        "available": intrinsic.is_file() and extrinsic.is_file(),
    }


def resolve_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != EXPERIMENT_SCHEMA:
        raise ValueError(
            f"Expected config schema {EXPERIMENT_SCHEMA!r}, got {config.get('schema_version')!r}"
        )
    trackers = config.get("trackers")
    if trackers != list(TRACKERS):
        raise ValueError(f"Initial experiment trackers must be exactly {list(TRACKERS)}")

    rgb_root = _expand(config["rgb_root"])
    mapping_path = _expand(config["rgb_mapping"])
    motion_root = _expand(config["motion_root"])
    camera_root = _expand(config["camera_root"])
    robot_asset_root = _expand(config["robot_asset_root"])
    with mapping_path.open() as stream:
        rgb_mapping = json.load(stream)

    resolved_sequences = []
    for sequence in config["sequences"]:
        sequence_id = sequence["sequence_id"]
        if sequence_id not in rgb_mapping:
            raise KeyError(f"Sequence is absent from RGB mapping: {sequence_id}")
        relative_video = Path(rgb_mapping[sequence_id])
        video_path = (rgb_root / relative_video).resolve()
        geometry = probe_video(video_path)
        motion_path = _single_motion_file(motion_root, sequence_id, config["gripper"])
        motion_lengths = _motion_lengths(motion_path)
        mismatches = {
            name: length
            for name, length in motion_lengths.items()
            if length != geometry.frame_count
        }
        if mismatches:
            raise ValueError(
                f"Frame mismatch for {sequence_id}: video={geometry.frame_count}, motion={mismatches}"
            )
        camera = _camera_paths(camera_root, relative_video)
        conditions = {}
        for tracker in trackers:
            blockers: list[str] = []
            if tracker == "ground_truth" and not camera["available"]:
                blockers.append("missing_taco_egocentric_camera_parameters")
            conditions[tracker] = {
                "tracker": tracker,
                # This resolver proves only that immutable RGB/motion/camera
                # inputs exist. Tracker weights, licensed MANO assets, and
                # runnable images are validated by the tracker stages.
                "state": "blocked" if blockers else "source_inputs_resolved",
                "blockers": blockers,
            }
        resolved_sequences.append(
            {
                **sequence,
                "video": {
                    "path": str(video_path),
                    "relative_mapping_path": str(relative_video),
                    "size_bytes": video_path.stat().st_size,
                    **geometry.as_dict(),
                },
                "motion": {
                    "path": str(motion_path),
                    "size_bytes": motion_path.stat().st_size,
                    "frame_lengths": motion_lengths,
                },
                "camera": camera,
                "conditions": conditions,
            }
        )

    return {
        "schema_version": RESOLVED_EXPERIMENT_SCHEMA,
        "source_schema_version": EXPERIMENT_SCHEMA,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "description": config.get("description", ""),
        "trackers": trackers,
        "inpainting": config["inpainting"],
        "robot": config["robot"],
        "gripper": config["gripper"],
        "roots": {
            "rgb": str(rgb_root),
            "motion": str(motion_root),
            "camera": str(camera_root),
            "robot_assets": str(robot_asset_root),
        },
        "robot_assets_available": robot_asset_root.is_dir(),
        "sequences": resolved_sequences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open() as stream:
        config = json.load(stream)
    resolved = resolve_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(resolved, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Resolved {len(resolved['sequences'])} sequences -> {args.output.resolve()}")
    blocked = [
        (seq["sequence_id"], name, condition["blockers"])
        for seq in resolved["sequences"]
        for name, condition in seq["conditions"].items()
        if condition["blockers"]
    ]
    for sequence_id, tracker, blockers in blocked:
        print(f"BLOCKED {sequence_id} / {tracker}: {', '.join(blockers)}")


if __name__ == "__main__":
    main()
