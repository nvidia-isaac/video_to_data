# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
from pathlib import Path
import subprocess
import os


MOGE_REPOSITORY = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE_MODEL_BYTES = 1_323_815_904
MOGE_MODEL_SHA256 = (
    "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def download_moge(output_dir: str | None = None) -> None:
    if output_dir is None:
        output_dir = os.environ.get("CHECKPOINT_DIR")
        if output_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
    subprocess.run(
        [
            "hf",
            "download",
            MOGE_REPOSITORY,
            "--revision",
            MOGE_REVISION,
            "--local-dir",
            output_dir,
        ],
        check=True,
    )
    model_path = Path(output_dir) / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Downloaded MoGe checkpoint is missing {model_path}")
    if model_path.stat().st_size != MOGE_MODEL_BYTES:
        raise RuntimeError(
            f"MoGe checkpoint size mismatch: expected {MOGE_MODEL_BYTES}, "
            f"got {model_path.stat().st_size}"
        )
    actual_sha256 = _sha256(model_path)
    if actual_sha256 != MOGE_MODEL_SHA256:
        raise RuntimeError(
            f"MoGe checkpoint SHA-256 mismatch: expected {MOGE_MODEL_SHA256}, "
            f"got {actual_sha256}"
        )
    print("MoGE v2 checkpoint downloaded.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MoGE v2 checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for checkpoint")
    args = parser.parse_args()
    download_moge(output_dir=args.output_dir)
