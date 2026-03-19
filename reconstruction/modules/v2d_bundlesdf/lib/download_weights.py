"""
Download BundleSDF weights (RoMA feature matcher).

RoMA is used for inter-frame feature matching during SDF training.
Two files are downloaded:
  roma_outdoor.pth          — RoMA outdoor model
  dinov2_vitl14_pretrain.pth — DINOv2 backbone used by RoMA
"""
import argparse
import os
import subprocess


ROMA_URL    = "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth"
DINOV2_URL  = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"


def download_weights(output_dir: str) -> None:
    roma_dir = os.path.join(output_dir, "roma")
    os.makedirs(roma_dir, exist_ok=True)

    for url in [ROMA_URL, DINOV2_URL]:
        filename = url.split("/")[-1]
        dest = os.path.join(roma_dir, filename)
        if os.path.isfile(dest):
            print(f"Already exists, skipping: {dest}")
        else:
            print(f"Downloading {filename} ...")
            subprocess.run(["wget", "-q", "--show-progress", "-O", dest, url], check=True)

    print(f"BundleSDF weights ready at {roma_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BundleSDF (RoMA) weights")
    parser.add_argument("--output_dir", required=True, help="Root weights directory (roma/ will be created inside)")
    args = parser.parse_args()
    download_weights(args.output_dir)
