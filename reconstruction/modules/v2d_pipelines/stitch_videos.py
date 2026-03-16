"""Stitch multiple videos side-by-side using ffmpeg."""
import os
import subprocess


def stitch_videos(video_paths: list[str], output_path: str) -> None:
    """Stitch videos horizontally side-by-side into a single video using ffmpeg.

    All input videos are scaled to the same height before stitching.
    Audio is dropped. Output is H.264 with yuv420p for broad compatibility.

    Args:
        video_paths: List of input video file paths (2 or more).
        output_path: Path for the output stitched video.
    """
    if len(video_paths) < 2:
        raise ValueError("Need at least 2 videos to stitch")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n = len(video_paths)
    inputs = []
    for p in video_paths:
        inputs += ["-i", p]

    # Scale all streams to the height of the first video, then hstack.
    # [i:v]scale=-2:H[vi] normalizes each stream to the reference height H.
    filter_parts = [f"[0:v]scale=iw:ih[v0]"]
    for i in range(1, n):
        filter_parts.append(f"[{i}:v]scale=-2:ih[v{i}]")
    stack_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{stack_inputs}hstack=inputs={n}[out]")
    filter_complex = ";".join(filter_parts)

    subprocess.run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ], check=True)
