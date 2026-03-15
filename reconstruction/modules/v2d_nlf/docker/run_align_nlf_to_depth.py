import os
from v2d.docker.container import run_in_container

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
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.nlf.lib.align_nlf_to_depth",
        inputs={"smpl_results_path": smpl_results_path, "depth_folder": depth_folder, "masks_dir": masks_dir, "intrinsics_path": intrinsics_path, "weights_dir": weights_dir},
        outputs={"output_path": output_path},
        dev=dev,
        modules_dir=_MODULES_DIR,
        gpus=True,
    )


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
