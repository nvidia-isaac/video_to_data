import os
import subprocess

from v2d.hamer.docker._config import IMAGE_NAME, MODULES_DIR


def run_shell(dev: bool = False) -> None:
    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [IMAGE_NAME, "bash"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run shell in v2d_hamer container")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_shell(dev=args.dev)
