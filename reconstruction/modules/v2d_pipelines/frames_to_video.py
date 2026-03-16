"""Encode a folder of numbered PNG frames into a video using ffmpeg."""
import glob
import os
import subprocess


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
