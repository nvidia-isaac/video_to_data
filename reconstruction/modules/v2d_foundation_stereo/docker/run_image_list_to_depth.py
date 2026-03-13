import subprocess
import os

IMAGE_NAME = "v2d_foundation_stereo"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_image_list_to_depth(
    left_dir: str,
    right_dir: str,
    depth_folder: str,
    intrinsics_folder: str,
    model_dir: str,
    calibration_file: str = None,
    fx: float = None, fy: float = None, cx: float = None, cy: float = None,
    baseline: float = None,
    dev: bool = False,
) -> None:
    left_dir = os.path.abspath(left_dir)
    right_dir = os.path.abspath(right_dir)
    depth_folder = os.path.abspath(depth_folder)
    intrinsics_folder = os.path.abspath(intrinsics_folder)
    model_dir = os.path.abspath(model_dir)

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(intrinsics_folder, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{left_dir}:/data/left",
        "-v", f"{right_dir}:/data/right",
        "-v", f"{depth_folder}:/data/depth_out",
        "-v", f"{intrinsics_folder}:/data/intrinsics_out",
        "-v", f"{model_dir}:/data/models",
    ]

    if calibration_file:
        calibration_file = os.path.abspath(calibration_file)
        cal_dir = os.path.dirname(calibration_file)
        cal_name = os.path.basename(calibration_file)
        cmd += ["-v", f"{cal_dir}:/data/calibration"]

    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]

    module_cmd = [
        IMAGE_NAME,
        "python", "-m", "v2d.foundation_stereo.lib.image_list_to_depth",
        "--left_dir", "/data/left",
        "--right_dir", "/data/right",
        "--depth_folder", "/data/depth_out",
        "--intrinsics_folder", "/data/intrinsics_out",
        "--model_dir", "/data/models",
    ]

    if calibration_file:
        module_cmd += ["--calibration_file", f"/data/calibration/{cal_name}"]
    elif fx is not None:
        module_cmd += [
            "--fx", str(fx), "--fy", str(fy),
            "--cx", str(cx), "--cy", str(cy),
            "--baseline", str(baseline),
        ]

    subprocess.run(cmd + module_cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Foundation Stereo: image list to depth maps")
    parser.add_argument('--left_dir', required=True)
    parser.add_argument('--right_dir', required=True)
    parser.add_argument('--depth_folder', required=True)
    parser.add_argument('--intrinsics_folder', required=True)
    parser.add_argument('--model_dir', required=True)

    cal_group = parser.add_mutually_exclusive_group(required=True)
    cal_group.add_argument('--calibration_file')
    cal_group.add_argument('--fx', type=float)

    parser.add_argument('--fy', type=float)
    parser.add_argument('--cx', type=float)
    parser.add_argument('--cy', type=float)
    parser.add_argument('--baseline', type=float)
    parser.add_argument('--dev', action='store_true', help='Mount local modules for development')

    args = parser.parse_args()
    run_image_list_to_depth(
        args.left_dir, args.right_dir,
        args.depth_folder, args.intrinsics_folder, args.model_dir,
        calibration_file=args.calibration_file,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        baseline=args.baseline, dev=args.dev,
    )
