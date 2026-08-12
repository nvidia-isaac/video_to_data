# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encode sequential JPEG frames into an MP4 with ffmpeg."""

import argparse
import subprocess
from pathlib import Path


def stitch_mp4(frames_dir: str | Path, output_mp4: str | Path, fps: int = 30) -> None:
    frames_dir = Path(frames_dir)
    output_mp4 = Path(output_mp4)
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    failures = []
    for codec in ("libx264", "h264_nvenc", "mpeg4"):
        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "%06d.jpg"),
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            str(output_mp4),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Wrote {output_mp4} with {codec}")
            return
        reason = result.stderr.strip().splitlines()
        detail = reason[-1][:200] if reason else f"exit {result.returncode}"
        failures.append(f"{codec}: {detail}")
        print(f"Codec {codec} failed ({detail}); trying next codec")

    raise RuntimeError(
        f"ffmpeg failed to stitch frames from {frames_dir}: " + "; ".join(failures)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output_mp4", required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    stitch_mp4(args.frames_dir, args.output_mp4, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
