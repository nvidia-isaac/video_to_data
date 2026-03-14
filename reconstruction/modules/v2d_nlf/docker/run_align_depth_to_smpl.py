import subprocess
import os

IMAGE_NAME = "v2d_nlf"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_align_depth_to_smpl(
    depth_folder: str,
    smpl_depth_folder: str,
    output_depth_folder: str,
    masks_folder: str,
    smpl_masks_folder: str = None,
    dev: bool = False,
) -> None:
    depth_folder = os.path.abspath(depth_folder)
    smpl_depth_folder = os.path.abspath(smpl_depth_folder)
    output_depth_folder = os.path.abspath(output_depth_folder)
    masks_folder = os.path.abspath(masks_folder)

    os.makedirs(output_depth_folder, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{depth_folder}:/data/depth",
        "-v", f"{smpl_depth_folder}:/data/smpl_depth",
        "-v", f"{output_depth_folder}:/data/output",
        "-v", f"{masks_folder}:/data/masks",
    ]
    if smpl_masks_folder:
        smpl_masks_folder = os.path.abspath(smpl_masks_folder)
        cmd += ["-v", f"{smpl_masks_folder}:/data/smpl_masks"]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]

    module_cmd = [
        IMAGE_NAME,
        "python", "-m", "v2d.nlf.lib.align_depth_to_smpl",
        "--depth_folder", "/data/depth",
        "--smpl_depth_folder", "/data/smpl_depth",
        "--output_depth_folder", "/data/output",
        "--masks_folder", "/data/masks",
    ]
    if smpl_masks_folder:
        module_cmd += ["--smpl_masks_folder", "/data/smpl_masks"]
    subprocess.run(cmd + module_cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run depth-to-SMPL alignment in Docker")
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--smpl_depth_folder", required=True)
    parser.add_argument("--output_depth_folder", required=True)
    parser.add_argument("--masks_folder", required=True)
    parser.add_argument("--smpl_masks_folder", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_depth_to_smpl(
        args.depth_folder, args.smpl_depth_folder, args.output_depth_folder,
        args.masks_folder, smpl_masks_folder=args.smpl_masks_folder, dev=args.dev,
    )
