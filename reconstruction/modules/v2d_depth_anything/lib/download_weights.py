import os
from huggingface_hub import snapshot_download

MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

def download_depth_anything(output_dir: str | None = None) -> None:
    if output_dir is None:
        output_dir = os.environ.get("CHECKPOINT_DIR")
        if output_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
    snapshot_download(repo_id=MODEL_REPO, local_dir=output_dir)
    print(f"Depth Anything 3 checkpoint downloaded to {output_dir}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Depth Anything 3 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoint")
    args = parser.parse_args()
    download_depth_anything(output_dir=args.output_dir)
