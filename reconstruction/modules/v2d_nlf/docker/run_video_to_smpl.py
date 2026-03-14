import subprocess
import os

IMAGE_NAME = "v2d_nlf"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_video_to_smpl(
    video_path: str,
    masks_dir: str,
    intrinsics_path: str,
    gender: str,
    output_path: str,
    weights_dir: str,
    model_type: str = "smplh",
    chunk_size: int = 32,
    dev: bool = False,
) -> None:
    video_path = os.path.abspath(video_path)
    masks_dir = os.path.abspath(masks_dir)
    intrinsics_path = os.path.abspath(intrinsics_path)
    output_path = os.path.abspath(output_path)
    weights_dir = os.path.abspath(weights_dir)

    video_dir, video_name = os.path.dirname(video_path), os.path.basename(video_path)
    intrinsics_dir, intrinsics_name = os.path.dirname(intrinsics_path), os.path.basename(intrinsics_path)
    output_dir, output_name = os.path.dirname(output_path), os.path.basename(output_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{video_dir}:/data/video",
        "-v", f"{masks_dir}:/data/masks",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{output_dir}:/data/output",
        "-v", f"{weights_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.nlf.lib.video_to_smpl",
        "--video_path", f"/data/video/{video_name}",
        "--masks_dir", "/data/masks",
        "--intrinsics_path", f"/data/intrinsics/{intrinsics_name}",
        "--gender", gender,
        "--weights_dir", "/data/weights",
        "--model_type", model_type,
        "--output_path", f"/data/output/{output_name}",
        "--chunk_size", str(chunk_size),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NLF video to SMPL in Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--gender", required=True, choices=["male", "female", "neutral"])
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--model_type", default="smplh", choices=["smpl", "smplh"])
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_video_to_smpl(
        args.video_path, args.masks_dir, args.intrinsics_path, args.gender,
        args.output_path, args.weights_dir, model_type=args.model_type,
        chunk_size=args.chunk_size, dev=args.dev,
    )
