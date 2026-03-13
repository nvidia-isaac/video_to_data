import subprocess
import os

IMAGE_NAME = "v2d_moge"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))

def run_video_to_depth(video_path: str, depth_folder: str, intrinsics_folder: str, weights_path: str, batch_size: int = 8, dev: bool = False) -> None:
    video_path = os.path.abspath(video_path)
    depth_folder = os.path.abspath(depth_folder)
    intrinsics_folder = os.path.abspath(intrinsics_folder)
    weights_path = os.path.abspath(weights_path)

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(intrinsics_folder, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{video_dir}:/data/video",
        "-v", f"{depth_folder}:/data/depth_out",
        "-v", f"{intrinsics_folder}:/data/intrinsics_out",
        "-v", f"{weights_path}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.moge.lib.video_to_depth",
        "--video_path", f"/data/video/{video_name}",
        "--depth_folder", "/data/depth_out",
        "--intrinsics_folder", "/data/intrinsics_out",
        "--weights_path", "/data/weights",
        "--batch_size", str(batch_size),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to depth")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Output folder for depth images")
    parser.add_argument("--intrinsics_folder", type=str, required=True, help="Output folder for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_video_to_depth(args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path, args.batch_size, dev=args.dev)
