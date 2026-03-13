import subprocess
import os

IMAGE_NAME = "v2d_unidepth"

def run_image_to_depth(image_path: str, depth_path: str, intrinsics_path: str, weights_path: str) -> None:
    image_path = os.path.abspath(image_path)
    depth_path = os.path.abspath(depth_path)
    intrinsics_path = os.path.abspath(intrinsics_path)
    weights_path = os.path.abspath(weights_path)

    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    depth_dir = os.path.dirname(depth_path)
    depth_filename = os.path.basename(depth_path)
    intrinsics_dir = os.path.dirname(intrinsics_path)
    intrinsics_filename = os.path.basename(intrinsics_path)

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)

    subprocess.run([
        "docker", 
        "run", 
        "-it", 
        "--rm", 
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{image_dir}:/data/image", 
        "-v", f"{depth_dir}:/data/depth_out", 
        "-v", f"{intrinsics_dir}:/data/intrinsics_out", 
        "-v", f"{weights_path}:/data/weights", 
        IMAGE_NAME, 
        "python", "-m", "v2d.unidepth.lib.image_to_depth", 
        "--image_path", f"/data/image/{image_name}", 
        "--depth_path", f"/data/depth_out/{depth_filename}", 
        "--intrinsics_path", f"/data/intrinsics_out/{intrinsics_filename}", 
        "--weights_path", "/data/weights", 
    ], check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process image to depth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--depth_path", type=str, required=True, help="Output path for depth image")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    args = parser.parse_args()
    run_image_to_depth(args.image_path, args.depth_path, args.intrinsics_path, args.weights_path)