import subprocess
import os

IMAGE_NAME = "v2d_sam2"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))

def run_video_to_masks(video_path: str, prompts_path: str, masks_dir: str, weights_dir: str, dev: bool = False) -> None:
    video_path = os.path.abspath(video_path)
    prompts_path = os.path.abspath(prompts_path)
    masks_dir = os.path.abspath(masks_dir)
    weights_dir = os.path.abspath(weights_dir)

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    prompts_dir = os.path.dirname(prompts_path)
    prompts_name = os.path.basename(prompts_path)

    os.makedirs(masks_dir, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{video_dir}:/data/video",
        "-v", f"{prompts_dir}:/data/prompts",
        "-v", f"{masks_dir}:/data/masks_out",
        "-v", f"{weights_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.sam2.lib.video_to_masks",
        "--video_path", f"/data/video/{video_name}",
        "--prompts_path", f"/data/prompts/{prompts_name}",
        "--masks_dir", "/data/masks_out",
        "--weights_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--masks_dir", type=str, required=True, help="Output directory for masks")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to SAM2 weights directory")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_video_to_masks(args.video_path, args.prompts_path, args.masks_dir, args.weights_dir, dev=args.dev)
