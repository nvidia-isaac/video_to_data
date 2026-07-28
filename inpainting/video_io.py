"""Small strict OpenCV/FFmpeg video helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


class Mp4Writer:
    """Write BGR frames as browser-compatible H.264 when FFmpeg is available."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.process: subprocess.Popen[bytes] | None = None
        self.opencv: cv2.VideoWriter | None = None
        executable = shutil.which("ffmpeg")
        if executable is None:
            try:
                import imageio_ffmpeg

                executable = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                executable = None
        width, height = size
        if executable:
            self.process = subprocess.Popen(
                [
                    executable,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    f"{fps:.12g}",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
            self.backend = "ffmpeg/h264"
        else:
            self.opencv = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )
            if not self.opencv.isOpened():
                raise RuntimeError(f"Cannot create video {self.path}")
            self.backend = "opencv/mp4v"

    def write(self, frame: np.ndarray) -> None:
        """Write one uint8 BGR frame."""
        image = np.asarray(frame)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Video frames must be HxWx3 uint8 BGR")
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.write(image.tobytes())
        else:
            assert self.opencv is not None
            self.opencv.write(image)

    def close(self) -> None:
        """Flush the encoder and fail if FFmpeg did not finish cleanly."""
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            returncode = self.process.wait()
            if returncode:
                raise RuntimeError(f"FFmpeg exited with status {returncode}")
        elif self.opencv is not None:
            self.opencv.release()


def probe_video(path: str | Path) -> dict[str, int | float]:
    """Return frame count, geometry and FPS from one video."""
    source = Path(path).expanduser().resolve()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot decode {source}")
    result: dict[str, int | float] = {
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    capture.release()
    if (
        result["frame_count"] <= 0
        or result["width"] <= 0
        or result["height"] <= 0
        or result["fps"] <= 0
    ):
        raise ValueError(f"Invalid video geometry for {source}: {result}")
    return result

