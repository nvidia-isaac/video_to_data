"""Render orthographic diagnostics for a common robot trajectory archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .contracts import validate_robot_trajectory_file


COLORS = {"left": (70, 210, 255), "right": (90, 90, 255)}
VIEWS = ((0, 1, "top: x/y"), (0, 2, "side: x/z"), (1, 2, "front: y/z"))


def _project(points: np.ndarray, axes: tuple[int, int], lo: np.ndarray, hi: np.ndarray, size: int) -> np.ndarray:
    margin = 44
    span = np.maximum(hi - lo, 1e-6)
    xy = (points[:, axes] - lo[list(axes)]) / span[list(axes)]
    result = np.empty_like(xy)
    result[:, 0] = margin + xy[:, 0] * (size - 2 * margin)
    result[:, 1] = size - margin - xy[:, 1] * (size - 2 * margin)
    return np.rint(result).astype(np.int32)


def render(trajectory_path: Path, output: Path, fps: float, panel_size: int = 420) -> int:
    frame_count = validate_robot_trajectory_file(trajectory_path)
    with np.load(trajectory_path, allow_pickle=False) as data:
        positions = {
            side: np.asarray(data[f"{side}_wrist_position"], dtype=np.float64)
            for side in ("left", "right")
        }
        validity = {
            side: np.asarray(data[f"{side}_valid"], dtype=np.bool_)
            for side in ("left", "right")
        }
        finger_activity = {
            side: np.asarray(data[f"{side}_finger_joints"], dtype=np.float64)
            for side in ("left", "right")
        }
    all_positions = np.concatenate(list(positions.values()), axis=0)
    lo = np.nanmin(all_positions, axis=0)
    hi = np.nanmax(all_positions, axis=0)
    padding = np.maximum((hi - lo) * 0.08, 0.015)
    lo, hi = lo - padding, hi + padding
    width, height = panel_size * 3, panel_size + 90
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {temporary}")
    try:
        for frame_index in range(frame_count):
            canvas = np.full((height, width, 3), 24, dtype=np.uint8)
            for view_index, (axis_a, axis_b, title) in enumerate(VIEWS):
                x0 = view_index * panel_size
                cv2.rectangle(canvas, (x0, 0), (x0 + panel_size - 1, panel_size - 1), (75, 75, 75), 1)
                cv2.putText(canvas, title, (x0 + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
                for side in ("left", "right"):
                    observed = validity[side][: frame_index + 1]
                    observed_indices = np.flatnonzero(observed)
                    if observed_indices.size:
                        # Draw contiguous runs separately so a missing
                        # observation never appears as an interpolated path.
                        split_at = np.flatnonzero(np.diff(observed_indices) > 1) + 1
                        for run in np.split(observed_indices, split_at):
                            path = _project(
                                positions[side][run],
                                (axis_a, axis_b),
                                lo,
                                hi,
                                panel_size,
                            )
                            path[:, 0] += x0
                            if len(path) > 1:
                                cv2.polylines(
                                    canvas, [path], False, COLORS[side], 2, cv2.LINE_AA
                                )
                        if observed[-1]:
                            cv2.circle(
                                canvas,
                                tuple(path[-1]),
                                7,
                                COLORS[side],
                                -1,
                                cv2.LINE_AA,
                            )
            cv2.putText(
                canvas,
                f"frame {frame_index:04d}/{frame_count - 1:04d}",
                (18, panel_size + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            for index, side in enumerate(("left", "right")):
                values = finger_activity[side][frame_index]
                text = (
                    f"{side} Sharpa joint RMS: {np.sqrt(np.mean(values * values)):.3f} rad"
                    if validity[side][frame_index]
                    else f"{side} observation invalid"
                )
                cv2.putText(
                    canvas,
                    text,
                    (300 + index * 440, panel_size + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    COLORS[side],
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                "TACO world frame (camera projection intentionally not inferred)",
                (18, panel_size + 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (185, 185, 185),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()
    temporary.replace(output)
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--panel-size", type=int, default=420)
    args = parser.parse_args()
    frames = render(args.trajectory, args.output, args.fps, args.panel_size)
    print(f"Wrote {frames} frames -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
