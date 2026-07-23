# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download and verify the immutable FoundationPose inference weights."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import NamedTuple
import uuid


SCORER_FOLDER_ID = "12Te_3TELLes5cim1d7F7EBTwUSe7iRBj"
SCORER_RUN_NAME = "2024-01-11-20-02-45"

REFINER_FOLDER_ID = "1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC"
REFINER_RUN_NAME = "2023-10-28-18-33-37"


class WeightFile(NamedTuple):
    size_bytes: int
    sha256: str


EXPECTED_WEIGHTS: dict[str, dict[str, WeightFile]] = {
    SCORER_RUN_NAME: {
        "config.yml": WeightFile(
            778,
            "a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f",
        ),
        "model_best.pth": WeightFile(
            190_229_389,
            "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
        ),
    },
    REFINER_RUN_NAME: {
        "config.yml": WeightFile(
            708,
            "28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686",
        ),
        "model_best.pth": WeightFile(
            68_220_109,
            "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
        ),
    },
}

FOLDER_IDS = {
    SCORER_RUN_NAME: SCORER_FOLDER_ID,
    REFINER_RUN_NAME: REFINER_FOLDER_ID,
}


class WeightIntegrityError(RuntimeError):
    """Raised when a FoundationPose artifact is missing or has unknown bytes."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_run_directory(directory: Path, run_name: str) -> None:
    for filename, expected in EXPECTED_WEIGHTS[run_name].items():
        path = directory / filename
        if not path.is_file():
            raise WeightIntegrityError(f"Missing FoundationPose weight: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected.size_bytes:
            raise WeightIntegrityError(
                f"FoundationPose weight size mismatch for {path}: "
                f"expected {expected.size_bytes}, got {actual_size}"
            )
        actual_hash = _sha256(path)
        if actual_hash != expected.sha256:
            raise WeightIntegrityError(
                f"FoundationPose weight SHA-256 mismatch for {path}: "
                f"expected {expected.sha256}, got {actual_hash}"
            )


def verify_weights(output_dir: str | Path) -> None:
    """Verify all four exact inference artifacts under ``output_dir``."""

    root = Path(output_dir).expanduser().resolve()
    for run_name in EXPECTED_WEIGHTS:
        _verify_run_directory(root / run_name, run_name)


def _publish_verified_run(staged: Path, destination: Path, run_name: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_WEIGHTS[run_name]:
        source = staged / filename
        target = destination / filename
        temporary = destination / f".{filename}.{uuid.uuid4().hex}.partial"
        try:
            with (
                source.open("rb") as source_stream,
                temporary.open("xb") as target_stream,
            ):
                shutil.copyfileobj(source_stream, target_stream, length=8 * 1024 * 1024)
                target_stream.flush()
                os.fsync(target_stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def _download_run(root: Path, run_name: str) -> None:
    folder_id = FOLDER_IDS[run_name]
    with tempfile.TemporaryDirectory(
        dir=root, prefix=f".{run_name}.download."
    ) as temporary_root:
        staged = Path(temporary_root) / run_name
        subprocess.run(
            [
                "gdown",
                "--folder",
                f"https://drive.google.com/drive/folders/{folder_id}",
                "-O",
                str(staged),
            ],
            check=True,
        )
        _verify_run_directory(staged, run_name)
        _publish_verified_run(staged, root / run_name, run_name)
        _verify_run_directory(root / run_name, run_name)


def download_weights(output_dir: str) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for label, run_name in (
        ("Scorer", SCORER_RUN_NAME),
        ("Refiner", REFINER_RUN_NAME),
    ):
        destination = root / run_name
        try:
            _verify_run_directory(destination, run_name)
        except WeightIntegrityError as exc:
            print(f"{label} weights are absent or invalid ({exc}); downloading.")
            _download_run(root, run_name)
        else:
            print(f"Verified {label.lower()} weights at {destination}; skipping.")

    verify_weights(root)
    print("All FoundationPose weights verified successfully.")
    print(f"  Scorer:  {root / SCORER_RUN_NAME}")
    print(f"  Refiner: {root / REFINER_RUN_NAME}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for weights"
    )
    args = parser.parse_args()
    download_weights(args.output_dir)
