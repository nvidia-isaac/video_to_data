import subprocess
import os

IMAGE_NAME = "v2d_foundation_stereo"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_image_to_depth(
    left_image_path: str,
    right_image_path: str,
    depth_path: str,
    intrinsics_path: str,
    model_dir: str,
    calibration_file: str = None,
    fx: float = None, fy: float = None, cx: float = None, cy: float = None,
    baseline: float = None,
    dev: bool = False,
) -> None:
    left_image_path = os.path.abspath(left_image_path)
    right_image_path = os.path.abspath(right_image_path)
    depth_path = os.path.abspath(depth_path)
    intrinsics_path = os.path.abspath(intrinsics_path)
    model_dir = os.path.abspath(model_dir)

    left_dir = os.path.dirname(left_image_path)
    left_name = os.path.basename(left_image_path)
    right_dir = os.path.dirname(right_image_path)
    right_name = os.path.basename(right_image_path)
    depth_dir = os.path.dirname(depth_path)
    depth_name = os.path.basename(depth_path)
    intrinsics_dir = os.path.dirname(intrinsics_path)
    intrinsics_name = os.path.basename(intrinsics_path)

    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{left_dir}:/data/left",
        "-v", f"{right_dir}:/data/right",
        "-v", f"{depth_dir}:/data/depth_out",
        "-v", f"{intrinsics_dir}:/data/intrinsics_out",
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
        "python", "-m", "v2d.foundation_stereo.lib.image_to_depth",
        "--left_image_path", f"/data/left/{left_name}",
        "--right_image_path", f"/data/right/{right_name}",
        "--depth_path", f"/data/depth_out/{depth_name}",
        "--intrinsics_path", f"/data/intrinsics_out/{intrinsics_name}",
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

    parser = argparse.ArgumentParser(description="Foundation Stereo: single stereo pair to depth")
    parser.add_argument('--left_image_path', required=True)
    parser.add_argument('--right_image_path', required=True)
    parser.add_argument('--depth_path', required=True)
    parser.add_argument('--intrinsics_path', required=True)
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
    run_image_to_depth(
        args.left_image_path, args.right_image_path,
        args.depth_path, args.intrinsics_path, args.model_dir,
        calibration_file=args.calibration_file,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        baseline=args.baseline, dev=args.dev,
    )
