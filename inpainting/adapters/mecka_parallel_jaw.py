"""Retarget MECKA hand tracks to robot-neutral parallel-jaw targets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from inpainting.mecka_panda.contracts import (
    PARALLEL_JAW_SCHEMA,
    artifact,
    load_npz,
    scalar_text,
    validate_parallel_jaw_arrays,
    validate_tracking_arrays,
    write_json_atomic,
    write_npz_atomic,
)

TRAJECTORY_FILENAME = "parallel_jaw_trajectory.npz"
METADATA_FILENAME = "parallel_jaw_trajectory.json"
RUN_SCHEMA = "v2d.inpainting.mecka-parallel-jaw-run/v1"


def _present_runs(present: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(present, dtype=np.int8), (1, 1))
    transitions = np.flatnonzero(np.diff(padded))
    return list(zip(transitions[::2], transitions[1::2], strict=True))


def condition_hand(
    points: np.ndarray,
    present: np.ndarray,
    *,
    jump_k: float = 6.0,
    max_gap: int = 15,
    smooth_window: int = 11,
    smooth_poly: int = 2,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reject one-frame wrist spikes, fill short gaps, and Savgol smooth."""
    points = np.asarray(points, dtype=np.float64)
    present = np.asarray(present, dtype=np.bool_)
    valid = present.copy()
    wrist = points[:, 0]
    for start, stop in _present_runs(present):
        if stop - start < 3:
            continue
        displacement = np.linalg.norm(np.diff(wrist[start:stop], axis=0), axis=1)
        median = float(np.median(displacement))
        mad = float(np.median(np.abs(displacement - median))) * 1.4826 + 1e-9
        threshold = median + jump_k * mad
        for local in range(1, stop - start - 1):
            if (
                displacement[local - 1] > threshold
                and displacement[local] > threshold
            ):
                valid[start + local] = False
    jump_count = int(present.sum() - valid.sum())

    conditioned = points.copy()
    for start, stop in _present_runs(present):
        indices = np.arange(start, stop)
        good = indices[valid[start:stop]]
        for frame in indices[~valid[start:stop]]:
            before = good[good < frame]
            after = good[good > frame]
            if not before.size or not after.size:
                present[frame] = False
                continue
            left, right = int(before[-1]), int(after[0])
            if right - left > max_gap:
                present[frame] = False
                continue
            weight = (frame - left) / float(right - left)
            conditioned[frame] = (
                (1.0 - weight) * points[left] + weight * points[right]
            )

    for start, stop in _present_runs(present):
        length = stop - start
        window = min(smooth_window, length if length % 2 else length - 1)
        if window < 5:
            continue
        poly = min(smooth_poly, window - 1)
        flattened = conditioned[start:stop].reshape(length, -1)
        filtered = savgol_filter(flattened, window, poly, axis=0)
        conditioned[start:stop] = filtered.reshape(length, 21, 3)
    conditioned[~present] = np.nan
    return conditioned, present, jump_count


def semantic_pose(
    points: np.ndarray, *, is_right: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return pinch center, semantic rotation, and aperture."""
    thumb = points[4]
    virtual_finger = np.mean(points[[8, 12, 16, 20]], axis=0)
    finger_web = np.mean(points[[5, 9, 13, 17]], axis=0)
    hand_web = 0.5 * (points[1] + finger_web)
    center = 0.5 * (thumb + virtual_finger)
    aperture = float(np.linalg.norm(thumb - virtual_finger))
    jaw = (1.0 if is_right else -1.0) * (thumb - virtual_finger)
    jaw /= np.linalg.norm(jaw) + 1e-12
    approach = center - hand_web
    approach -= np.dot(approach, jaw) * jaw
    if np.linalg.norm(approach) < 1e-6:
        reference = (
            np.array([1.0, 0.0, 0.0])
            if abs(jaw[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        approach = reference - np.dot(reference, jaw) * jaw
    approach /= np.linalg.norm(approach) + 1e-12
    rotation = np.column_stack([approach, jaw, np.cross(approach, jaw)])
    return center, rotation, aperture


def grasp_pose(
    points: np.ndarray, *, is_right: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compatibility view of the semantic pose as approach/opening axes."""
    center, rotation, aperture = semantic_pose(points, is_right=is_right)
    return center, rotation[:, 0], rotation[:, 1], aperture


def retarget_tracking_arrays(
    tracking: dict[str, np.ndarray],
    *,
    jump_k: float = 6.0,
    max_gap: int = 15,
    smooth_window: int = 11,
    smooth_poly: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert validated tracking arrays to the parallel-jaw contract."""
    frame_count = validate_tracking_arrays(tracking)
    if scalar_text(tracking["coordinate_frame"], "coordinate_frame") != "camera":
        raise ValueError("The MECKA Panda rig requires camera-frame tracking")
    output: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PARALLEL_JAW_SCHEMA),
        "tracker": np.asarray(scalar_text(tracking["tracker"], "tracker")),
        "coordinate_frame": np.asarray("camera"),
        "frame_indices": np.arange(frame_count, dtype=np.int64),
    }
    diagnostics: dict[str, Any] = {}
    for side in ("left", "right"):
        conditioned, valid, jumps = condition_hand(
            tracking[f"{side}_joints_3d"],
            tracking[f"{side}_valid"].copy(),
            jump_k=jump_k,
            max_gap=max_gap,
            smooth_window=smooth_window,
            smooth_poly=smooth_poly,
        )
        positions = np.full((frame_count, 3), np.nan, dtype=np.float64)
        quaternions = np.full((frame_count, 4), np.nan, dtype=np.float64)
        apertures = np.full(frame_count, np.nan, dtype=np.float64)
        for frame in np.flatnonzero(valid):
            position, rotation, aperture = semantic_pose(
                conditioned[frame], is_right=side == "right"
            )
            positions[frame] = position
            quaternions[frame] = Rotation.from_matrix(rotation).as_quat(
                scalar_first=True
            )
            apertures[frame] = aperture
        output[f"{side}_valid"] = valid
        output[f"{side}_position"] = positions
        output[f"{side}_wxyz"] = quaternions
        output[f"{side}_aperture_m"] = apertures
        diagnostics[side] = {
            "valid_count": int(valid.sum()),
            "jumps_removed": jumps,
            "aperture_median": (
                float(np.median(apertures[valid])) if valid.any() else None
            ),
        }
    validate_parallel_jaw_arrays(output)
    return output, diagnostics


def execute(
    *,
    tracking: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    jump_k: float = 6.0,
    max_gap: int = 15,
    smooth_window: int = 11,
    smooth_poly: int = 2,
) -> dict[str, Any]:
    """Retarget and atomically commit an NPZ followed by its metadata."""
    tracking_path = Path(tracking).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    trajectory_path = output / TRAJECTORY_FILENAME
    metadata_path = output / METADATA_FILENAME
    existing = [path for path in (trajectory_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")
    arrays, diagnostics = retarget_tracking_arrays(
        load_npz(tracking_path),
        jump_k=jump_k,
        max_gap=max_gap,
        smooth_window=smooth_window,
        smooth_poly=smooth_poly,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_npz_atomic(trajectory_path, arrays)
    validate_parallel_jaw_arrays(load_npz(trajectory_path))
    metadata = {
        "schema_version": RUN_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "tracking": artifact(tracking_path),
            "implementation": artifact(__file__),
        },
        "algorithm": {
            "contact": "thumb_tip + mean(index,middle,ring,pinky tips)",
            "conditioning": {
                "jump_k": jump_k,
                "max_gap": max_gap,
                "smooth_window": smooth_window,
                "smooth_poly": smooth_poly,
            },
        },
        "diagnostics": diagnostics,
        "output": {"trajectory": artifact(trajectory_path)},
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jump-k", type=float, default=6.0)
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--smooth-poly", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    metadata = execute(
        tracking=args.tracking,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        jump_k=args.jump_k,
        max_gap=args.max_gap,
        smooth_window=args.smooth_window,
        smooth_poly=args.smooth_poly,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

