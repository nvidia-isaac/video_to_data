import subprocess
import os

IMAGE_NAME = "v2d_unidepth"

def run_video_to_depth(video_path: str, depth_folder: str, intrinsics_folder: str, weights_path: str, batch_size: int = 8) -> None:
    video_path = os.path.abspath(video_path)
    depth_folder = os.path.abspath(depth_folder)
    intrinsics_folder = os.path.abspath(intrinsics_folder)
    weights_path = os.path.abspath(weights_path)

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    depth_dir = depth_folder
    intrinsics_dir = intrinsics_folder

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)
    subprocess.run([
        "docker", 
        "run", 
        "-it", 
        "--rm", 
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{video_dir}:/data/video", 
        "-v", f"{depth_dir}:/data/depth_out", 
        "-v", f"{intrinsics_dir}:/data/intrinsics_out", 
        "-v", f"{weights_path}:/data/weights", 
        IMAGE_NAME, 
        "python", "-m", "v2d.unidepth.lib.video_to_depth", 
        "--video_path", f"/data/video/{video_name}", 
        "--depth_folder", f"/data/depth_out", 
        "--intrinsics_folder", f"/data/intrinsics_out", 
        "--weights_path", "/data/weights", 
        "--batch_size", str(batch_size), 
    ], check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to depth")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Output folder for depth images")
    parser.add_argument("--intrinsics_folder", type=str, required=True, help="Output folder for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    args = parser.parse_args()
    run_video_to_depth(args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path, args.batch_size)