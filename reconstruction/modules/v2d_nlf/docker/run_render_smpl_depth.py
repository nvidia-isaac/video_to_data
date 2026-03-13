import subprocess
import os

IMAGE_NAME = "v2d_nlf"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_render_smpl_depth(
    smpl_params_path: str,
    intrinsics_path: str,
    output_depth_folder: str,
    output_mask_folder: str,
    weights_dir: str,
    dev: bool = False,
) -> None:
    smpl_params_path = os.path.abspath(smpl_params_path)
    intrinsics_path = os.path.abspath(intrinsics_path)
    output_depth_folder = os.path.abspath(output_depth_folder)
    output_mask_folder = os.path.abspath(output_mask_folder)
    weights_dir = os.path.abspath(weights_dir)

    smpl_dir, smpl_name = os.path.dirname(smpl_params_path), os.path.basename(smpl_params_path)
    intrinsics_dir, intrinsics_name = os.path.dirname(intrinsics_path), os.path.basename(intrinsics_path)

    os.makedirs(output_depth_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{smpl_dir}:/data/smpl",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{output_depth_folder}:/data/depth_output",
        "-v", f"{output_mask_folder}:/data/mask_output",
        "-v", f"{weights_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.nlf.lib.render_smpl_depth",
        "--smpl_params_path", f"/data/smpl/{smpl_name}",
        "--intrinsics_path", f"/data/intrinsics/{intrinsics_name}",
        "--output_depth_folder", "/data/depth_output",
        "--output_mask_folder", "/data/mask_output",
        "--weights_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL depth rendering in Docker")
    parser.add_argument("--smpl_params_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_depth_folder", required=True)
    parser.add_argument("--output_mask_folder", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_smpl_depth(
        args.smpl_params_path, args.intrinsics_path,
        args.output_depth_folder, args.output_mask_folder,
        args.weights_dir, dev=args.dev,
    )
