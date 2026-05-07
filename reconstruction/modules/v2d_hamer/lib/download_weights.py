"""Download HaMeR demo data + checkpoint.

Downloads ``hamer_demo_data.tar.gz`` from the original HaMeR project page
and extracts it into ``<weights_dir>``. The expected layout afterwards:

    <weights_dir>/_DATA/hamer_ckpts/
        checkpoints/hamer.ckpt
        dataset_config.yaml
        model_config.yaml

Note: MANO_RIGHT.pkl is NOT downloaded (it requires registration at
https://mano.is.tue.mpg.de/). Place it manually under <weights_dir>/data/mano/
or wherever HaMeR's config expects it.
"""

import argparse
import os
import subprocess
import tarfile

_URL = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"


def run_download(weights_dir: str) -> None:
    os.makedirs(weights_dir, exist_ok=True)
    tar_path = os.path.join(weights_dir, "hamer_demo_data.tar.gz")
    ckpt = os.path.join(weights_dir, "_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt")
    if os.path.exists(ckpt):
        print(f"  HaMeR checkpoint already present at {ckpt}; skipping download.")
        return
    print(f"  Downloading HaMeR demo data → {tar_path}")
    subprocess.run(["wget", "-q", "--show-progress", _URL, "-O", tar_path], check=True)
    print(f"  Extracting → {weights_dir}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(weights_dir, filter="data")
    os.remove(tar_path)
    if not os.path.exists(ckpt):
        raise RuntimeError(
            f"Extraction succeeded but {ckpt} was not produced. "
            "The archive layout may have changed upstream."
        )
    print(f"  Done. HaMeR checkpoint at {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights_dir", required=True)
    args = parser.parse_args()
    run_download(args.weights_dir)
