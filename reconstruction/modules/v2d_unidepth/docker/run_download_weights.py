import subprocess
import os


IMAGE_NAME = "v2d_unidepth"


def run_download(output_dir: str | None = None) -> None:
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
        "-v", 
        f"{output_dir}:/data/weights", 
        IMAGE_NAME, 
        "python", 
        "-m", 
        "v2d.unidepth.lib.download_weights", 
        "--output_dir", 
        "/data/weights"
    ], check=True)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download UniDepth v2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoint")
    args = parser.parse_args()
    run_download(output_dir=args.output_dir)