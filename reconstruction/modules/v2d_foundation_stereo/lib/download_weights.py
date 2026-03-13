"""Download Foundation Stereo ONNX model from NVIDIA NGC.

The TensorRT engine is compiled at runtime (GPU-architecture-specific).
"""
import argparse
import os
import urllib.request

ONNX_FILENAME = "deployable_foundationstereo_small_576x960_v2.0.onnx"
ONNX_URL = (
    "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/"
    f"foundationstereo/deployable_v2.0/files?redirect=true&path={ONNX_FILENAME}"
)


def download_weights(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, ONNX_FILENAME)

    if os.path.exists(dest):
        print(f"ONNX already exists at {dest}, skipping download.")
        return

    print(f"Downloading Foundation Stereo ONNX to {dest} ...")
    urllib.request.urlretrieve(ONNX_URL, dest)
    print(f"Download complete: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Foundation Stereo ONNX model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model files")
    args = parser.parse_args()
    download_weights(args.output_dir)
