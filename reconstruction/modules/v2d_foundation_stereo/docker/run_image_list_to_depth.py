import os
from v2d.docker.container import run_in_container

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
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_stereo.lib.image_list_to_depth",
        inputs={"left_dir": left_dir, "right_dir": right_dir, "model_dir": model_dir, "calibration_file": calibration_file},
        outputs={"depth_folder": depth_folder, "intrinsics_folder": intrinsics_folder},
        extra_args={"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline},
        dev=dev,
        modules_dir=_MODULES_DIR,
        gpus=True,
    )


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
