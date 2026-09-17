# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision FoundationPose weights for the selected inference backend."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import tempfile
import urllib.request
from pathlib import Path

from v2d.foundation_pose.lib.backends import (
    DEFAULT_BACKEND,
    NVIDIA_TENSORRT,
    NVLABS_PYTORCH,
    SUPPORTED_BACKENDS,
)
from v2d.foundation_pose.lib.trt_engine import (
    MODEL_REGISTRY,
    MODEL_SPECS,
    MODEL_VERSION,
    commercial_model_dir,
    sha256_file,
    write_model_manifest,
)


NGC_BASE_URL = (
    f"https://api.ngc.nvidia.com/v2/models/{MODEL_REGISTRY}/"
    f"versions/{MODEL_VERSION}/files"
)

SCORER_FOLDER_ID = "12Te_3TELLes5cim1d7F7EBTwUSe7iRBj"
SCORER_RUN_NAME = "2024-01-11-20-02-45"
REFINER_FOLDER_ID = "1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC"
REFINER_RUN_NAME = "2023-10-28-18-33-37"

GDOWN_MAX_ATTEMPTS = 3
GDOWN_RETRY_DELAY_SECONDS = 10


def _gdown_folder(folder_id: str, output_dir: str) -> None:
    command = [
        "gdown", "--folder",
        f"https://drive.google.com/drive/folders/{folder_id}",
        "-O", output_dir,
    ]

    for attempt in range(1, GDOWN_MAX_ATTEMPTS + 1):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempt == GDOWN_MAX_ATTEMPTS:
                raise

            print(
                f"gdown failed with exit code {exc.returncode} "
                f"(attempt {attempt}/{GDOWN_MAX_ATTEMPTS}); retrying in "
                f"{GDOWN_RETRY_DELAY_SECONDS} seconds..."
            )
            time.sleep(GDOWN_RETRY_DELAY_SECONDS)


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        print(f"Model already verified: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        print(f"Downloading {url} -> {destination}")
        urllib.request.urlretrieve(url, temporary)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {destination.name}: "
                f"expected {expected_sha256}, got {actual}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download_nvidia_tensorrt(output_dir: str, *, accept_model_eula: bool) -> None:
    if not accept_model_eula:
        raise RuntimeError(
            "Downloading NVIDIA TAO FoundationPose requires accepting the NVIDIA Open "
            "Model License. Re-run with --accept_nvidia_model_eula after reviewing the "
            "NGC model terms."
        )
    model_dir = commercial_model_dir(output_dir)
    for spec in MODEL_SPECS.values():
        _download_verified(
            f"{NGC_BASE_URL}/{spec.onnx_filename}",
            model_dir / spec.onnx_filename,
            spec.sha256,
        )
    write_model_manifest(model_dir)
    print(f"Commercial FoundationPose ONNX models ready: {model_dir}")


def _download_nvlabs_pytorch(output_dir: str) -> None:
    root = Path(output_dir) / NVLABS_PYTORCH
    for folder_id, run_name in (
        (SCORER_FOLDER_ID, SCORER_RUN_NAME),
        (REFINER_FOLDER_ID, REFINER_RUN_NAME),
    ):
        run_dir = root / run_name
        checkpoint = run_dir / "model_best.pth"
        if checkpoint.is_file():
            print(f"NVLabs weights already exist: {run_dir}")
            continue
        _gdown_folder(folder_id, str(run_dir))
    print(f"NVLabs FoundationPose weights ready: {root}")


def download_weights(
    output_dir: str,
    backend: str = DEFAULT_BACKEND,
    *,
    accept_nvidia_model_eula: bool = False,
) -> None:
    if backend == NVIDIA_TENSORRT:
        _download_nvidia_tensorrt(
            output_dir, accept_model_eula=accept_nvidia_model_eula
        )
    elif backend == NVLABS_PYTORCH:
        _download_nvlabs_pytorch(output_dir)
    else:
        raise ValueError(f"Unsupported backend {backend!r}; expected {SUPPORTED_BACKENDS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FoundationPose weights")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=DEFAULT_BACKEND)
    parser.add_argument("--accept_nvidia_model_eula", action="store_true")
    args = parser.parse_args()
    download_weights(
        args.output_dir,
        backend=args.backend,
        accept_nvidia_model_eula=args.accept_nvidia_model_eula,
    )
