import subprocess
import os

IMAGE_NAME = "v2d_nlf"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_align_nlf_to_depth(
    smpl_results_path: str,
    depth_folder: str,
    masks_dir: str,
    intrinsics_path: str,
    output_path: str,
    weights_dir: str,
    dev: bool = False,
) -> None:
    smpl_results_path = os.path.abspath(smpl_results_path)
    depth_folder = os.path.abspath(depth_folder)
    masks_dir = os.path.abspath(masks_dir)
    intrinsics_path = os.path.abspath(intrinsics_path)
    output_path = os.path.abspath(output_path)
    weights_dir = os.path.abspath(weights_dir)

    smpl_dir, smpl_name = os.path.dirname(smpl_results_path), os.path.basename(smpl_results_path)
    intrinsics_dir, intrinsics_name = os.path.dirname(intrinsics_path), os.path.basename(intrinsics_path)
    output_dir, output_name = os.path.dirname(output_path), os.path.basename(output_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{smpl_dir}:/data/smpl",
        "-v", f"{depth_folder}:/data/depth",
        "-v", f"{masks_dir}:/data/masks",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{output_dir}:/data/output",
        "-v", f"{weights_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.nlf.lib.align_nlf_to_depth",
        "--smpl_results_path", f"/data/smpl/{smpl_name}",
        "--depth_folder", "/data/depth",
        "--masks_dir", "/data/masks",
        "--intrinsics_path", f"/data/intrinsics/{intrinsics_name}",
        "--output_path", f"/data/output/{output_name}",
        "--weights_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NLF-to-depth alignment in Docker")
    parser.add_argument("--smpl_results_path", required=True)
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_nlf_to_depth(
        args.smpl_results_path, args.depth_folder, args.masks_dir,
        args.intrinsics_path, args.output_path, args.weights_dir, dev=args.dev,
    )
