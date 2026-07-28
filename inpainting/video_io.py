"""Small OpenCV helpers with strict frame-geometry checks."""

from __future__ import annotations

from pathlib import Path

import cv2

from .contracts import ContractError, VideoGeometry


def probe_video(path: str | Path) -> VideoGeometry:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ContractError(f"Could not open video: {path}")
    try:
        geometry = VideoGeometry(
            frame_count=int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()
    if geometry.frame_count <= 0 or geometry.width <= 0 or geometry.height <= 0:
        raise ContractError(f"Video reports invalid geometry: {geometry}")
    if not (geometry.fps > 0):
        raise ContractError(f"Video reports invalid FPS: {geometry.fps}")
    return geometry
