import os
from huggingface_hub import snapshot_download

MODEL_REPOS = {
    "nested": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "metric": "depth-anything/DA3METRIC-LARGE",
}


def download_depth_anything(output_dir: str | None = None, model: str = "nested") -> None:
    if model not in MODEL_REPOS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODEL_REPOS)}")
    if output_dir is None:
        output_dir = os.environ.get("CHECKPOINT_DIR")
        if output_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
    repo_id = MODEL_REPOS[model]
    snapshot_download(repo_id=repo_id, local_dir=output_dir)
    print(f"Depth Anything 3 '{model}' checkpoint downloaded to {output_dir}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Depth Anything 3 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoint")
    parser.add_argument("--model", type=str, default="nested", choices=list(MODEL_REPOS),
                        help="Model variant to download (default: nested)")
    args = parser.parse_args()
    download_depth_anything(output_dir=args.output_dir, model=args.model)
