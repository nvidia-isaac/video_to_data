import subprocess
import os

IMAGE_NAME = "v2d_sam3d"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))

def run_download(output_dir: str, dev: bool = False) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HF_HOME=/tmp/hf_cache",
        "-v", f"{output_dir}:/data/weights",
    ]
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.isfile(token_path):
            with open(token_path) as f:
                hf_token = f.read().strip()
    if hf_token:
        cmd += ["-e", f"HF_TOKEN={hf_token}"]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.sam3d.lib.download_weights",
        "--output_dir", "/data/weights",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download all SAM3D weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for all weights")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir, dev=args.dev)
