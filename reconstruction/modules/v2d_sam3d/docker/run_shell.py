import subprocess
import os

IMAGE_NAME = "v2d_sam3d"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))

def run_shell(dev: bool = False) -> None:
    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [IMAGE_NAME, "bash"]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run shell in v2d_sam3d container")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_shell(dev=args.dev)
