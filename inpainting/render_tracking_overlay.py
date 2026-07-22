"""Project common world-frame hand tracks into TACO RGB for calibration QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .contracts import ContractError, validate_tracking_file
from .taco_camera import load_taco_camera, project_world_points
from .video_io import probe_video


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
COLORS = {"left": (70, 210, 255), "right": (70, 90, 255)}


def render_overlay(
    video_path: Path,
    tracking_path: Path,
    intrinsic_path: Path,
    extrinsic_path: Path,
    output_path: Path,
    label: str | None = None,
) -> int:
    geometry = probe_video(video_path)
    frame_count = validate_tracking_file(tracking_path, expected_frames=geometry.frame_count)
    with np.load(tracking_path, allow_pickle=False) as tracking:
        if str(np.asarray(tracking["coordinate_frame"]).reshape(-1)[0]) != "world":
            raise ContractError("TACO camera projection requires a world-frame tracking archive")
        tracker = str(np.asarray(tracking["tracker"]).reshape(-1)[0])
        joints = {
            side: np.asarray(tracking[f"{side}_joints_3d"], dtype=np.float64)
            for side in ("left", "right")
        }
    overlay_label = label or f"{tracker} camera projection"
    camera = load_taco_camera(
        intrinsic_path,
        extrinsic_path,
        expected_frames=frame_count,
        width=geometry.width,
        height=geometry.height,
    )
    capture = cv2.VideoCapture(str(video_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        geometry.fps,
        (geometry.width, geometry.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {temporary}")
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ContractError(f"Video decode failed at frame {frame_index}")
            for side in ("left", "right"):
                pixels, _, valid = project_world_points(
                    joints[side][frame_index],
                    camera.intrinsic,
                    camera.world_to_camera[frame_index],
                )
                for first, second in HAND_EDGES:
                    if valid[first] and valid[second]:
                        cv2.line(
                            frame,
                            tuple(np.rint(pixels[first]).astype(int)),
                            tuple(np.rint(pixels[second]).astype(int)),
                            COLORS[side],
                            5,
                            cv2.LINE_AA,
                        )
                for point, is_valid in zip(pixels, valid, strict=True):
                    if is_valid:
                        cv2.circle(frame, tuple(np.rint(point).astype(int)), 5, COLORS[side], -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                overlay_label,
                (22, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    temporary.replace(output_path)
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument("--intrinsic", required=True, type=Path)
    parser.add_argument("--extrinsic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--label",
        help="Optional overlay caption (defaults to '<tracker> camera projection').",
    )
    args = parser.parse_args()
    frames = render_overlay(
        args.video,
        args.tracking,
        args.intrinsic,
        args.extrinsic,
        args.output,
        args.label,
    )
    print(f"Wrote {frames} frames -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
