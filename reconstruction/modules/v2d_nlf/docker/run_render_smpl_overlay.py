import subprocess
import os

IMAGE_NAME = "v2d_nlf"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_render_smpl_overlay(
    video_path: str,
    smpl_params_path: str,
    intrinsics_path: str,
    output_dir: str,
    weights_dir: str,
    dev: bool = False,
) -> None:
    video_path = os.path.abspath(video_path)
    smpl_params_path = os.path.abspath(smpl_params_path)
    intrinsics_path = os.path.abspath(intrinsics_path)
    output_dir = os.path.abspath(output_dir)
    weights_dir = os.path.abspath(weights_dir)

    video_dir, video_name = os.path.dirname(video_path), os.path.basename(video_path)
    smpl_dir, smpl_name = os.path.dirname(smpl_params_path), os.path.basename(smpl_params_path)
    intrinsics_dir, intrinsics_name = os.path.dirname(intrinsics_path), os.path.basename(intrinsics_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{video_dir}:/data/video",
        "-v", f"{smpl_dir}:/data/smpl",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{output_dir}:/data/output",
        "-v", f"{weights_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.nlf.lib.render_smpl_overlay",
        "--video_path", f"/data/video/{video_name}",
        "--smpl_params_path", f"/data/smpl/{smpl_name}",
        "--intrinsics_path", f"/data/intrinsics/{intrinsics_name}",
        "--output_dir", "/data/output",
        "--weights_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL overlay rendering in Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--smpl_params_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_smpl_overlay(
        args.video_path, args.smpl_params_path, args.intrinsics_path,
        args.output_dir, args.weights_dir, dev=args.dev,
    )
