import subprocess
import os

IMAGE_NAME = "v2d_mesh"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_tests(dev: bool = False) -> None:
    """Run the v2d_mesh pytest suite inside the container (requires OSMesa for render tests)."""
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-e", "PYOPENGL_PLATFORM=egl",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]

    cmd += [
        IMAGE_NAME,
        "python", "-m", "pytest", "/workspace/v2d_mesh/lib/tests/", "-v",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run v2d_mesh tests inside Docker")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for live code editing")
    args = parser.parse_args()

    run_tests(dev=args.dev)
