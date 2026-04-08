import argparse
import getpass
import os
import subprocess
from pathlib import Path

DINOV3_REPO = "https://github.com/facebookresearch/dinov3.git"
DINOV3_CACHE_DIR = "torch_home/hub/facebookresearch_dinov3_main"


def _ensure_hf_token() -> None:
    """Check for HF_TOKEN and prompt the user if not found."""
    if os.environ.get("HF_TOKEN"):
        return

    result = subprocess.run(
        ["hf", "whoami"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return

    print("No HF_TOKEN found and you are not logged in to Hugging Face.")
    print("SAM 3D Body (facebook/sam-3d-body-dinov3) is a gated repo that requires authentication.")
    print("Request access at: https://huggingface.co/facebook/sam-3d-body-dinov3")
    print()
    token = getpass.getpass("Enter your Hugging Face token (or Ctrl+C to abort): ")
    os.environ["HF_TOKEN"] = token


def download_weights(output_dir: str) -> None:
    """Download SAM3D-Body, MoGe-2, and DINOv3 repo into a single weights directory.

    Layout:
      output_dir/
        sam-3d-body-dinov3/   - SAM3D-Body checkpoints (gated HuggingFace repo)
        moge-2-vitb-normal/   - MoGe-2 weights
        torch_home/hub/       - DINOv3 repo clone (for torch.hub.load)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_hf_token()

    sam3d_dir = output_dir / "sam-3d-body-dinov3"
    if not sam3d_dir.exists():
        print("Downloading SAM 3D Body checkpoints...")
        subprocess.run(
            ["hf", "download", "facebook/sam-3d-body-dinov3",
             "--local-dir", str(sam3d_dir)],
            check=True,
        )
        print("SAM 3D Body checkpoints downloaded.")
    else:
        print(f"SAM 3D Body checkpoints already exist at {sam3d_dir}")

    moge_dir = output_dir / "moge-2-vitb-normal"
    if not moge_dir.exists():
        print("Downloading MoGe-2 weights...")
        subprocess.run(
            ["hf", "download", "Ruicheng/moge-2-vitb-normal",
             "--local-dir", str(moge_dir)],
            check=True,
        )
        print("MoGe-2 weights downloaded.")
    else:
        print(f"MoGe-2 weights already exist at {moge_dir}")

    dinov3_dir = output_dir / DINOV3_CACHE_DIR
    if not dinov3_dir.exists():
        print("Cloning DINOv3 repo (for torch.hub cache)...")
        dinov3_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", DINOV3_REPO, str(dinov3_dir)],
            check=True,
        )
        print("DINOv3 repo cached.")
    else:
        print(f"DINOv3 repo already cached at {dinov3_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SAM3D-Body and MoGe-2 weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    args = parser.parse_args()
    download_weights(args.output_dir)
