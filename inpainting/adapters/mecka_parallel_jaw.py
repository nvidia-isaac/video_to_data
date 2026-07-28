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
RUN_SCHEMA = "v2d.inpainting.mecka-parallel-jaw-run/v2"
PALM_RATIO_MAX = 0.50
TIP_RATIO_MIN = 0.75
ROTATION_ALPHA = 0.20
MAX_ROTATION_STEP_DEG = 6.0
_GEOMETRY_EPSILON_M = 1e-9
_PALM_WIDTH_EPSILON_M = 1e-6
_PARALLEL_JAW_FLIP = np.diag([1.0, -1.0, -1.0])


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
            if displacement[local - 1] > threshold and displacement[local] > threshold:
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
            conditioned[frame] = (1.0 - weight) * points[left] + weight * points[right]

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


def _unit(value: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if np.isfinite(norm) and norm > _GEOMETRY_EPSILON_M:
        return vector / norm
    fallback_vector = np.asarray(fallback, dtype=np.float64)
    fallback_norm = float(np.linalg.norm(fallback_vector))
    if not np.isfinite(fallback_norm) or fallback_norm <= _GEOMETRY_EPSILON_M:
        raise ValueError("Cannot normalize a degenerate vector and fallback")
    return fallback_vector / fallback_norm


def _orthogonal_unit(axis: np.ndarray) -> np.ndarray:
    normalized = _unit(axis, fallback=np.array([1.0, 0.0, 0.0]))
    reference = np.eye(3)[int(np.argmin(np.abs(normalized)))]
    return _unit(np.cross(normalized, reference), fallback=np.array([0.0, 1.0, 0.0]))


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.asarray(vector) - np.dot(vector, axis) * axis


def _palm_frame_with_diagnostics(
    points: np.ndarray, *, is_right: bool
) -> tuple[np.ndarray, int]:
    wrist = points[0]
    mcp_center = np.mean(points[[5, 9, 13, 17]], axis=0)
    forward_seed = mcp_center - wrist
    forward_degenerate = (
        not np.isfinite(forward_seed).all()
        or np.linalg.norm(forward_seed) <= _GEOMETRY_EPSILON_M
    )
    forward = _unit(forward_seed, fallback=np.array([0.0, 1.0, 0.0]))

    handedness = 1.0 if is_right else -1.0
    across_seed = handedness * (points[5] - points[17])
    across_projected = _project_perpendicular(across_seed, forward)
    across_degenerate = (
        not np.isfinite(across_projected).all()
        or np.linalg.norm(across_projected) <= _GEOMETRY_EPSILON_M
    )
    across = _unit(across_projected, fallback=_orthogonal_unit(forward))
    normal = _unit(np.cross(forward, across), fallback=_orthogonal_unit(forward))
    across = _unit(np.cross(normal, forward), fallback=across)
    rotation = np.column_stack([forward, across, normal])
    return rotation, int(forward_degenerate) + int(across_degenerate)


def palm_landmark_frame(points: np.ndarray, *, is_right: bool) -> np.ndarray:
    """Return a handedness-normalized wrist/MCP semantic gripper frame."""
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        raise ValueError("points must be finite with shape (21,3)")
    rotation, _ = _palm_frame_with_diagnostics(array, is_right=is_right)
    return rotation


def _thumb_index_pose_with_diagnostics(
    points: np.ndarray,
    *,
    is_right: bool,
    palm_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    wrist = points[0]
    thumb = points[4]
    index = points[8]
    position = 0.5 * (thumb + index)
    aperture = float(np.linalg.norm(thumb - index))
    handedness = 1.0 if is_right else -1.0
    opening_seed = handedness * (thumb - index)
    coincident_tips = int(aperture <= _GEOMETRY_EPSILON_M)
    jaw = _unit(opening_seed, fallback=palm_rotation[:, 1])

    approach = _unit(
        np.cross(jaw, palm_rotation[:, 2]),
        fallback=palm_rotation[:, 0],
    )
    if np.dot(approach, position - wrist) < 0.0:
        approach = -approach
    normal = _unit(np.cross(approach, jaw), fallback=palm_rotation[:, 2])
    jaw = _unit(np.cross(normal, approach), fallback=jaw)
    rotation = np.column_stack([approach, jaw, np.cross(approach, jaw)])

    palm_width = float(np.linalg.norm(points[5] - points[17]))
    palm_width_degenerate = int(palm_width <= _PALM_WIDTH_EPSILON_M)
    ratio = aperture / max(palm_width, _PALM_WIDTH_EPSILON_M)
    return (
        position,
        rotation,
        aperture,
        ratio,
        coincident_tips,
        palm_width_degenerate,
    )


def thumb_index_pose(
    points: np.ndarray, *, is_right: bool
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return p4/p8 midpoint, raw semantic frame, width, and normalized gap."""
    array = np.asarray(points, dtype=np.float64)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        raise ValueError("points must be finite with shape (21,3)")
    palm_rotation = palm_landmark_frame(array, is_right=is_right)
    position, rotation, aperture, ratio, _, _ = _thumb_index_pose_with_diagnostics(
        array,
        is_right=is_right,
        palm_rotation=palm_rotation,
    )
    return position, rotation, aperture, ratio


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            Rotation.from_matrix(np.asarray(first).T @ np.asarray(second)).as_rotvec()
        )
    )


def _align_parallel_jaw(
    target: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, bool]:
    alternative = np.asarray(target) @ _PARALLEL_JAW_FLIP
    if _rotation_distance(reference, alternative) < _rotation_distance(
        reference, target
    ):
        return alternative, True
    return np.asarray(target), False


def closest_parallel_jaw_equivalent(
    target: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Choose the equivalent parallel-jaw frame nearest to ``reference``."""
    return _align_parallel_jaw(target, reference)[0]


def blend_rotation(start: np.ndarray, end: np.ndarray, weight: float) -> np.ndarray:
    """Interpolate right-handed frames along the shortest SO(3) arc."""
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("rotation blend weight must be finite and in [0,1]")
    delta = Rotation.from_matrix(np.asarray(start).T @ np.asarray(end)).as_rotvec()
    return np.asarray(start) @ Rotation.from_rotvec(weight * delta).as_matrix()


def smooth_rotation(
    previous: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float = ROTATION_ALPHA,
    max_step_deg: float = MAX_ROTATION_STEP_DEG,
) -> np.ndarray:
    """Apply a geodesic low-pass and a hard per-frame angular limit."""
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("rotation alpha must be finite and in (0,1]")
    if not np.isfinite(max_step_deg) or max_step_deg <= 0.0:
        raise ValueError("max rotation step must be positive and finite")
    delta = Rotation.from_matrix(
        np.asarray(previous).T @ np.asarray(target)
    ).as_rotvec()
    angle = float(np.linalg.norm(delta))
    if angle <= _GEOMETRY_EPSILON_M:
        return np.asarray(previous).copy()
    fraction = min(alpha, np.deg2rad(max_step_deg) / angle)
    return np.asarray(previous) @ Rotation.from_rotvec(fraction * delta).as_matrix()


def _validate_stability_parameters(
    *,
    palm_ratio_max: float,
    tip_ratio_min: float,
    rotation_alpha: float,
    max_rotation_step_deg: float,
) -> None:
    values = (
        palm_ratio_max,
        tip_ratio_min,
        rotation_alpha,
        max_rotation_step_deg,
    )
    if not np.isfinite(values).all():
        raise ValueError("retarget stability parameters must be finite")
    if palm_ratio_max < 0.0 or tip_ratio_min <= palm_ratio_max:
        raise ValueError("ratio thresholds must satisfy 0 <= palm < tip")
    if not 0.0 < rotation_alpha <= 1.0:
        raise ValueError("rotation_alpha must be in (0,1]")
    if max_rotation_step_deg <= 0.0:
        raise ValueError("max_rotation_step_deg must be positive")


def _validate_rotation(rotation: np.ndarray, *, frame: int) -> None:
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError(f"Retarget frame {frame} is not a proper rotation")


def retarget_hand_sequence(
    points: np.ndarray,
    present: np.ndarray,
    *,
    is_right: bool,
    palm_ratio_max: float = PALM_RATIO_MAX,
    tip_ratio_min: float = TIP_RATIO_MIN,
    rotation_alpha: float = ROTATION_ALPHA,
    max_rotation_step_deg: float = MAX_ROTATION_STEP_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Build temporally stable p4/p8 parallel-jaw targets for one hand."""
    _validate_stability_parameters(
        palm_ratio_max=palm_ratio_max,
        tip_ratio_min=tip_ratio_min,
        rotation_alpha=rotation_alpha,
        max_rotation_step_deg=max_rotation_step_deg,
    )
    hand_points = np.asarray(points, dtype=np.float64)
    valid = np.asarray(present, dtype=np.bool_)
    if hand_points.ndim != 3 or hand_points.shape[1:] != (21, 3):
        raise ValueError("points must have shape (N,21,3)")
    if valid.shape != (hand_points.shape[0],):
        raise ValueError("present must have shape (N,)")
    if valid.any() and not np.isfinite(hand_points[valid]).all():
        raise ValueError("valid hand frames must contain finite points")

    frame_count = hand_points.shape[0]
    positions = np.full((frame_count, 3), np.nan, dtype=np.float64)
    quaternions = np.full((frame_count, 4), np.nan, dtype=np.float64)
    apertures = np.full(frame_count, np.nan, dtype=np.float64)
    ratios: list[float] = []
    final_steps_deg: list[float] = []
    mode_counts = {"palm": 0, "blend": 0, "pinch": 0}
    symmetry_flips = 0
    palm_axis_fallbacks = 0
    coincident_tip_fallbacks = 0
    palm_width_fallbacks = 0

    # Keep the last accepted orientation across missing observations.  Resetting
    # at a gap would let the first reappearing target bypass the angular limit.
    previous_rotation: np.ndarray | None = None
    previous_quaternion: np.ndarray | None = None
    for frame in np.flatnonzero(valid):
        palm_rotation, palm_fallbacks = _palm_frame_with_diagnostics(
            hand_points[frame], is_right=is_right
        )
        (
            position,
            raw_rotation,
            aperture,
            ratio,
            coincident_tips,
            palm_width_degenerate,
        ) = _thumb_index_pose_with_diagnostics(
            hand_points[frame],
            is_right=is_right,
            palm_rotation=palm_rotation,
        )
        palm_axis_fallbacks += palm_fallbacks
        coincident_tip_fallbacks += coincident_tips
        palm_width_fallbacks += palm_width_degenerate

        if previous_rotation is not None:
            palm_rotation, flipped = _align_parallel_jaw(
                palm_rotation, previous_rotation
            )
            symmetry_flips += int(flipped)
        raw_rotation, flipped = _align_parallel_jaw(raw_rotation, palm_rotation)
        symmetry_flips += int(flipped)

        linear_weight = float(
            np.clip(
                (ratio - palm_ratio_max) / (tip_ratio_min - palm_ratio_max),
                0.0,
                1.0,
            )
        )
        blend_weight = linear_weight**2 * (3.0 - 2.0 * linear_weight)
        desired_rotation = blend_rotation(palm_rotation, raw_rotation, blend_weight)
        if previous_rotation is None:
            stable_rotation = desired_rotation
        else:
            desired_rotation, flipped = _align_parallel_jaw(
                desired_rotation, previous_rotation
            )
            symmetry_flips += int(flipped)
            stable_rotation = smooth_rotation(
                previous_rotation,
                desired_rotation,
                alpha=rotation_alpha,
                max_step_deg=max_rotation_step_deg,
            )
            final_steps_deg.append(
                np.degrees(_rotation_distance(previous_rotation, stable_rotation))
            )
        _validate_rotation(stable_rotation, frame=frame)

        quaternion = Rotation.from_matrix(stable_rotation).as_quat(scalar_first=True)
        if (
            previous_quaternion is not None
            and np.dot(quaternion, previous_quaternion) < 0.0
        ):
            quaternion = -quaternion
        positions[frame] = position
        quaternions[frame] = quaternion
        apertures[frame] = aperture
        ratios.append(ratio)
        if ratio <= palm_ratio_max:
            mode_counts["palm"] += 1
        elif ratio >= tip_ratio_min:
            mode_counts["pinch"] += 1
        else:
            mode_counts["blend"] += 1
        previous_rotation = stable_rotation
        previous_quaternion = quaternion

    ratio_array = np.asarray(ratios, dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "valid_count": int(valid.sum()),
        "mode_counts": mode_counts,
        "symmetry_flip_count": symmetry_flips,
        "fallbacks": {
            "palm_axis": palm_axis_fallbacks,
            "coincident_thumb_index": coincident_tip_fallbacks,
            "degenerate_palm_width": palm_width_fallbacks,
        },
        "ratio": {
            "minimum": float(np.min(ratio_array)) if ratio_array.size else None,
            "median": float(np.median(ratio_array)) if ratio_array.size else None,
            "maximum": float(np.max(ratio_array)) if ratio_array.size else None,
        },
        "aperture_median": (
            float(np.median(apertures[valid])) if valid.any() else None
        ),
        "max_rotation_step_deg": max(final_steps_deg, default=0.0),
    }
    return positions, quaternions, apertures, diagnostics


def semantic_pose(
    points: np.ndarray, *, is_right: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the single-frame view of the stable p4/p8 target policy."""
    positions, quaternions, apertures, _ = retarget_hand_sequence(
        np.asarray(points, dtype=np.float64)[None, :, :],
        np.asarray([True]),
        is_right=is_right,
    )
    rotation = Rotation.from_quat(quaternions[0], scalar_first=True).as_matrix()
    return positions[0], rotation, float(apertures[0])


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
    palm_ratio_max: float = PALM_RATIO_MAX,
    tip_ratio_min: float = TIP_RATIO_MIN,
    rotation_alpha: float = ROTATION_ALPHA,
    max_rotation_step_deg: float = MAX_ROTATION_STEP_DEG,
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
        positions, quaternions, apertures, side_diagnostics = retarget_hand_sequence(
            conditioned,
            valid,
            is_right=side == "right",
            palm_ratio_max=palm_ratio_max,
            tip_ratio_min=tip_ratio_min,
            rotation_alpha=rotation_alpha,
            max_rotation_step_deg=max_rotation_step_deg,
        )
        output[f"{side}_valid"] = valid
        output[f"{side}_position"] = positions
        output[f"{side}_wxyz"] = quaternions
        output[f"{side}_aperture_m"] = apertures
        side_diagnostics["jumps_removed"] = jumps
        diagnostics[side] = side_diagnostics
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
    palm_ratio_max: float = PALM_RATIO_MAX,
    tip_ratio_min: float = TIP_RATIO_MIN,
    rotation_alpha: float = ROTATION_ALPHA,
    max_rotation_step_deg: float = MAX_ROTATION_STEP_DEG,
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
        palm_ratio_max=palm_ratio_max,
        tip_ratio_min=tip_ratio_min,
        rotation_alpha=rotation_alpha,
        max_rotation_step_deg=max_rotation_step_deg,
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
            "version": "thumb-index-palm-default/v1",
            "contact": {
                "thumb_tip": 4,
                "index_tip": 8,
                "position": "midpoint(p4,p8)",
                "aperture": "distance(p4,p8)",
            },
            "semantic_axes": {
                "x": "approach",
                "y": "parallel-jaw opening",
                "z": "cross(x,y)",
            },
            "conditioning": {
                "jump_k": jump_k,
                "max_gap": max_gap,
                "smooth_window": smooth_window,
                "smooth_poly": smooth_poly,
            },
            "orientation_stability": {
                "palm_ratio_max": palm_ratio_max,
                "tip_ratio_min": tip_ratio_min,
                "blend": "smoothstep SO(3)",
                "parallel_jaw_symmetry": "nearest of R and R@diag(1,-1,-1)",
                "rotation_alpha": rotation_alpha,
                "max_rotation_step_deg": max_rotation_step_deg,
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
    parser.add_argument("--palm-ratio-max", type=float, default=PALM_RATIO_MAX)
    parser.add_argument("--tip-ratio-min", type=float, default=TIP_RATIO_MIN)
    parser.add_argument("--rotation-alpha", type=float, default=ROTATION_ALPHA)
    parser.add_argument(
        "--max-rotation-step-deg",
        type=float,
        default=MAX_ROTATION_STEP_DEG,
    )
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
        palm_ratio_max=args.palm_ratio_max,
        tip_ratio_min=args.tip_ratio_min,
        rotation_alpha=args.rotation_alpha,
        max_rotation_step_deg=args.max_rotation_step_deg,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
