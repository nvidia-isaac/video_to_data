"""Render labelled, frame-synchronized videos in a comparison grid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from .video_io import probe_video


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.65, result.shape[1] / 1100.0)
    thickness = max(1, int(round(scale * 2)))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 14, 14
    cv2.rectangle(
        result,
        (x - 6, y - 6),
        (x + text_width + 8, y + text_height + baseline + 8),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        result,
        text,
        (x, y + text_height),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return result


def make_grid(
    videos: list[Path],
    labels: list[str],
    output: Path,
    tile_width: int = 640,
    columns: int | None = None,
    max_frames: int | None = None,
) -> int:
    if not videos or len(videos) != len(labels):
        raise ValueError("Supply the same nonzero number of videos and labels")
    geometries = [probe_video(path) for path in videos]
    fps = geometries[0].fps
    if any(abs(item.fps - fps) > 1e-3 for item in geometries[1:]):
        raise ValueError(f"All videos must have the same FPS: {[item.fps for item in geometries]}")
    frame_count = min(item.frame_count for item in geometries)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    columns = columns or len(videos)
    rows = math.ceil(len(videos) / columns)
    tile_heights = [round(item.height * tile_width / item.width) for item in geometries]
    tile_height = max(tile_heights)
    canvas_size = (columns * tile_width, rows * tile_height)

    captures = [cv2.VideoCapture(str(path)) for path in videos]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.partial{output.suffix}")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, canvas_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temporary}")
    try:
        for _ in range(frame_count):
            canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            for index, (capture, label) in enumerate(zip(captures, labels, strict=True)):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode {videos[index]}")
                height = round(frame.shape[0] * tile_width / frame.shape[1])
                tile = cv2.resize(frame, (tile_width, height), interpolation=cv2.INTER_AREA)
                if height < tile_height:
                    top = (tile_height - height) // 2
                    tile = cv2.copyMakeBorder(
                        tile, top, tile_height - height - top, 0, 0, cv2.BORDER_CONSTANT
                    )
                tile = _label(tile, label)
                row, column = divmod(index, columns)
                y0, x0 = row * tile_height, column * tile_width
                canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
            writer.write(canvas)
    finally:
        for capture in captures:
            capture.release()
        writer.release()
    temporary.replace(output)
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, type=Path)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    count = make_grid(
        args.video,
        args.label,
        args.output,
        tile_width=args.tile_width,
        columns=args.columns,
        max_frames=args.max_frames,
    )
    print(f"Wrote {count} synchronized frames -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
