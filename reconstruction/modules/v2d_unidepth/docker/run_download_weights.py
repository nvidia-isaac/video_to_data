import subprocess
import os
from v2d.unidepth.docker._config import IMAGE_NAME, MODULES_DIR

def run_download(output_dir: str, dev: bool = False) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{output_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.unidepth.lib.download_weights",
        "--output_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download UniDepth v2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoint")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, dev=args.dev)
