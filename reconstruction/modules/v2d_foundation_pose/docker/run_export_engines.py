"""Build cached FoundationPose TensorRT engines on the current GPU."""

import os
import subprocess

from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_export_engines(weights_dir: str, force: bool = False, dev: bool = False) -> None:
    weights_dir = os.path.abspath(weights_dir)
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"FoundationPose weights directory does not exist: {weights_dir}. "
            "Download the ONNX models before exporting engines."
        )
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{weights_dir}:/data/weights",
    ]
    if dev:
        command += ["-v", f"{MODULES_DIR}:/workspace"]
    expression = (
        "from v2d.foundation_pose.lib.trt_engine import ensure_engines; "
        f"print(ensure_engines('/data/weights', force={force!r}))"
    )
    command += [IMAGE_NAME, "python", "-c", expression]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_export_engines(args.weights_dir, force=args.force, dev=args.dev)
