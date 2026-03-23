import subprocess
import os
from v2d.depth_anything.docker._config import IMAGE_NAME, MODULES_DIR


def run_download(output_dir: str, model: str = "nested", dev: bool = False) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HF_HOME=/tmp/hf_cache",
        "-v", f"{output_dir}:/data/weights",
    ]
    if dev:
        cmd += ["-v", f"{MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.depth_anything.lib.download_weights",
        "--output_dir", "/data/weights",
        "--model", model,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Depth Anything V3 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoint")
    parser.add_argument("--model", type=str, default="nested", choices=["nested", "metric"],
                        help="Model variant to download (default: nested)")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, model=args.model, dev=args.dev)
