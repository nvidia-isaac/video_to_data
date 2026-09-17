import argparse
import os
import subprocess

from v2d.face_detector.docker._config import IMAGE_NAME, MODULES_DIR


def run_download(output_dir: str, dev: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    command = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{output_dir}:/data/weights",
    ]
    if dev:
        command += ["-v", f"{MODULES_DIR}:/workspace"]
    command += [
        IMAGE_NAME,
        "python",
        "-m",
        "v2d.face_detector.lib.download_weights",
        "--output_dir",
        "/data/weights",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_download(args.output_dir, args.dev)
