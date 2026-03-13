"""Download GroundingDINO SwinT-OGC checkpoint."""
import argparse
import os
import urllib.request

CHECKPOINT_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
CHECKPOINT_FILENAME = "groundingdino_swint_ogc.pth"


def download_weights(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, CHECKPOINT_FILENAME)

    if os.path.exists(dest):
        print(f"Checkpoint already exists at {dest}, skipping download.")
        return

    print(f"Downloading GroundingDINO SwinT-OGC checkpoint to {dest}...")
    urllib.request.urlretrieve(CHECKPOINT_URL, dest)
    print(f"Download complete: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GroundingDINO checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoint")
    args = parser.parse_args()
    download_weights(args.output_dir)
