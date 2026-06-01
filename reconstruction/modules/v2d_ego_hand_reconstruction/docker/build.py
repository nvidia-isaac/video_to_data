# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build Docker images for ego hand reconstruction (ViPE + Dyn-HaMR).

Delegates to the vendored shell scripts so build flags stay in sync with
upstream IsaacTeleop.
"""

import subprocess
from pathlib import Path

from v2d_ego_hand_reconstruction.docker._config import MODULE_DIR, VENDOR_DIR


def _sync_vendor() -> None:
    """Fetch vendored sources from IsaacTeleop if not already present."""
    subprocess.run([str(Path(MODULE_DIR) / "sync.sh")], check=True)


def _run_vendor_script(script: str, *args: str) -> None:
    vendor = Path(VENDOR_DIR)
    if not vendor.is_dir():
        print(f"{vendor} not found. Running sync.sh to fetch vendored sources...")
        _sync_vendor()
    subprocess.run(
        [str(vendor / "docker" / script), *args],
        check=True,
    )


def build_vipe() -> None:
    """Build the ViPE camera estimation image."""
    _run_vendor_script("vipe.sh", "build")


def build_dynhamr() -> None:
    """Build the Dyn-HaMR hand reconstruction image."""
    _run_vendor_script("dynhamr.sh", "build")


def build_all() -> None:
    """Build both ViPE and Dyn-HaMR images."""
    build_vipe()
    build_dynhamr()


if __name__ == "__main__":
    build_all()
