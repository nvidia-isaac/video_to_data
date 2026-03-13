import subprocess
import os

IMAGE_NAME = "v2d_moge"

def run_download(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)
    subprocess.run([
        "docker",
        "run",
        "-it",
        "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HF_HOME=/tmp/hf_cache",
        "-v", f"{output_dir}:/data/weights",
        IMAGE_NAME,
        "python", "-m", "v2d.moge.lib.download_weights",
        "--output_dir", "/data/weights",
    ], check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MoGE v2 checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoint")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir)
