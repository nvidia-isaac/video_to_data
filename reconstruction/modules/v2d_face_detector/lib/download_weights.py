"""Download and verify the pinned OpenCV YuNet face detector."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/4.10.0/"
    f"models/face_detection_yunet/{MODEL_FILENAME}"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weights(output_dir: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / MODEL_FILENAME
    if destination.is_file() and sha256_file(destination) == MODEL_SHA256:
        print(f"Verified YuNet model already exists at {destination}")
        return destination

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{MODEL_FILENAME}.", suffix=".tmp", dir=output
    )
    os.close(fd)
    temporary = Path(temp_name)
    try:
        print(f"Downloading YuNet from {MODEL_URL} to {destination}")
        urllib.request.urlretrieve(MODEL_URL, temporary)
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != MODEL_SHA256:
            raise ValueError(
                "YuNet checksum mismatch: "
                f"expected {MODEL_SHA256}, got {actual_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Installed verified YuNet model at {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    download_weights(args.output_dir)
