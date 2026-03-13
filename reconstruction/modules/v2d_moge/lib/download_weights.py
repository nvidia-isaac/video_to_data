import subprocess
import os

def download_moge(output_dir: str | None = None) -> None:
    if output_dir is None:
        output_dir = os.environ.get("CHECKPOINT_DIR")
        if output_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
    subprocess.run(["hf", "download", "Ruicheng/moge-2-vitl-normal", "--local-dir", output_dir], check=True)
    print("MoGE v2 checkpoint downloaded.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MoGE v2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoint")
    args = parser.parse_args()
    download_moge(output_dir=args.output_dir)
