"""Video utility functions: frame extraction, video encoding, and stitching."""
import glob
import os
import subprocess


def extract_images(video_path: str, output_folder: str) -> int:
    """Extract all frames from a video to a folder as numbered PNGs.

    Output files are named 000000.png, 000001.png, ...

    Returns:
        Number of frames extracted.
    """
    os.makedirs(output_folder, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        os.path.join(os.path.abspath(output_folder), "%06d.png"),
    ], check=True)
    return len(glob.glob(os.path.join(output_folder, "*.png")))


def frames_to_video(frames_dir: str, output_path: str, fps: float = 30.0) -> int:
    """Encode numbered PNG frames from a folder into a video file using ffmpeg.

    Expects files named 000000.png, 000001.png, ... (as written by extract_images).

    Returns:
        Number of frames encoded.
    """
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {frames_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(os.path.abspath(frames_dir), "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ], check=True)

    return len(paths)


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
