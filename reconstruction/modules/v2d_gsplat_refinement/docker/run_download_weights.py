# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the perceptual checkpoint used by gsplat refinement.

The CUDA kernels ship in the image and MANO comes from the HaMeR weights, but
the default refinement profile also uses torchvision's VGG16 ImageNet weights.
Provision that file during setup so a reconstruction never needs network access
from inside the runtime container.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request

VGG16_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
VGG16_FILENAME = "vgg16-397923af.pth"
VGG16_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_download(weights_path: str | None = None) -> None:
    """Download and verify the VGG16 checkpoint in ``weights_path``."""
    output_dir = os.path.abspath(weights_path or "data/weights/gsplat_refinement")
    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, VGG16_FILENAME)

    if os.path.isfile(destination):
        digest = _sha256(destination)
        if digest == VGG16_SHA256:
            print(f"VGG16 checkpoint already present and verified: {destination}")
            return
        raise RuntimeError(
            f"Existing VGG16 checkpoint has an invalid SHA-256: {destination} ({digest})"
        )

    fd, temporary = tempfile.mkstemp(prefix=f".{VGG16_FILENAME}.", dir=output_dir)
    os.close(fd)
    try:
        print(f"Downloading VGG16 ImageNet weights to {destination}...")
        urllib.request.urlretrieve(VGG16_URL, temporary)
        digest = _sha256(temporary)
        if digest != VGG16_SHA256:
            raise RuntimeError(
                f"Downloaded VGG16 checkpoint has an invalid SHA-256: {digest}"
            )
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"VGG16 checkpoint downloaded and verified: {destination}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights_path", default=None)
    args = p.parse_args()
    run_download(args.weights_path)
