# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch and verify the public WiLoR weights for later offline inference.

The separately licensed MANO model is never downloaded here.  Callers must
place ``MANO_RIGHT.pkl`` in ``<weights_dir>/pretrained_models`` first.
"""

import argparse
import hashlib
import os

from huggingface_hub import hf_hub_download


REPOSITORY_ID = "warmshao/WiLoR-mini"
REPOSITORY_REVISION = "b00adea9a6843bbb4c9042109c5eb29ab2a59dea"
PUBLIC_ARTIFACT_SHA256 = {
    "mano_mean_params.npz": "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988",
    "wilor_final.ckpt": "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2",
    "detector.pt": "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_download(weights_dir: str) -> None:
    os.makedirs(weights_dir, exist_ok=True)
    pretrained_dir = os.path.join(weights_dir, "pretrained_models")
    os.makedirs(pretrained_dir, exist_ok=True)
    mano_path = os.path.join(pretrained_dir, "MANO_RIGHT.pkl")
    if not os.path.isfile(mano_path) or os.path.getsize(mano_path) == 0:
        raise FileNotFoundError(
            "Licensed MANO_RIGHT.pkl must be supplied at " + mano_path
        )

    for filename, expected_sha256 in PUBLIC_ARTIFACT_SHA256.items():
        path = os.path.join(pretrained_dir, filename)
        if not os.path.isfile(path):
            print(f"  Downloading {filename} at {REPOSITORY_REVISION}")
            hf_hub_download(
                repo_id=REPOSITORY_ID,
                revision=REPOSITORY_REVISION,
                subfolder="pretrained_models",
                filename=filename,
                local_dir=weights_dir,
            )
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: {actual_sha256} != "
                f"{expected_sha256}"
            )
        print(f"  Verified {filename}: {actual_sha256}")
    print("  WiLoR public weights verified; licensed MANO model retained unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights_dir", required=True)
    args = parser.parse_args()
    run_download(args.weights_dir)
